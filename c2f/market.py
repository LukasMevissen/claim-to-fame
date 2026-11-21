"""Post-game market data pipeline: public transactions -> table, checks, opponent belief.

One canonical path (c2f/leaderboard.py does all fetching, into data/transactions/):

    refresh()  1. sync: download every completed game not on disk (gently, public
                  leaderboard endpoints only; a game not completed yet is simply not
                  listed and is picked up on the next call)
               2. results/ab_table.csv      (c2f/ab_table.py, with our_a / our_b)
               3. results/ab_checks.csv     (c2f/consistency.py, payoff-table checks)
               4. results/market/summary.json (per item: charges, t bounds, b bounds)
               5. results/opponent_belief.json refit (c2f/opponents.py), only when a new
                  game arrived and enough time is left before the deadline
    Every step logs and continues on failure; refresh() itself never raises, so the
    scheduler (`python -m c2f run`) can call it between games.

What one transaction reveals (payoff table in docs/CASE.md):
  accepted, amount x        -> issuer a = x (unless capped at c), reviewer b >= a
  rejected, amount x > 0    -> wrongful reject: a = x / 1.5 and a <= t; reviewer b < a
  rejected, amount 0        -> a > t (a itself is known from the issuer's other transactions)
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

from . import leaderboard
from .config import REPO_ROOT

CACHE = REPO_ROOT / "results" / "market"          # summary.json (+ legacy raw game_XXX.json)
TX_DIR = leaderboard.DEFAULT_DIR
TABLE = REPO_ROOT / "results" / "ab_table.csv"
CHECKS = REPO_ROOT / "results" / "ab_checks.csv"
BELIEF = REPO_ROOT / "results" / "opponent_belief.json"
QUOTES = REPO_ROOT / "runs" / "submissions.jsonl"
REFIT_SECONDS = 60      # start a refit only with this much time left before the deadline


def load_games(directory: str | Path = TX_DIR) -> dict[int, list[dict]]:
    """game id -> raw rows, from data/transactions/game<N>.json."""
    out = {}
    for f in sorted(Path(directory).glob("game*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        out[int(doc["game_id"])] = doc["items"]
    return dict(sorted(out.items()))


def summarize(txns: list[dict]) -> dict[int, dict]:
    """Per line item: each team's charge a (if known), fair/fraud flag, b bounds, t bounds."""
    by_item: dict[int, list[dict]] = defaultdict(list)
    for t in txns:
        by_item[t["line_item_index"]].append(t)
    out = {}
    for idx, ts in sorted(by_item.items()):
        a: dict[str, float] = {}
        fair: dict[str, bool] = {}
        for t in ts:
            i, x = t["issuer"], t["amount"]
            if not t["accepted"] and x > 0:
                a[i], fair[i] = x / 1.5, True
            elif not t["accepted"]:
                fair[i] = False
        for t in ts:                                  # accepted amounts: a unless capped
            if t["accepted"] and t["issuer"] not in a and t["amount"] > 0:
                a[t["issuer"]] = t["amount"]
        b_lo: dict[str, float] = defaultdict(float)
        b_hi: dict[str, float] = {}
        for t in ts:
            ai = a.get(t["issuer"])
            if ai is None:
                continue
            r = t["reviewer"]
            if t["accepted"]:
                b_lo[r] = max(b_lo[r], ai)
            else:
                b_hi[r] = min(b_hi.get(r, float("inf")), ai)
        t_lo = max([a[i] for i, f in fair.items() if f and i in a], default=0.0)
        t_hi = min([a[i] for i, f in fair.items() if not f and i in a], default=float("inf"))
        out[idx] = {"t_low": round(t_lo, 2), "t_high": None if t_hi == float("inf") else round(t_hi, 2),
                    "charges": {k: round(v, 2) for k, v in sorted(a.items(), key=lambda kv: kv[1])},
                    "fair": fair,
                    "b_bounds": {r: [round(b_lo.get(r, 0.0), 2),
                                     None if b_hi.get(r) is None else round(b_hi[r], 2)]
                                 for r in sorted(set(b_lo) | set(b_hi))}}
    return out


def write_summary(games: dict[int, list[dict]], path: str | Path = CACHE / "summary.json") -> dict:
    summary = {g: summarize(rows) for g, rows in games.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, indent=1))
    return summary


def _stale(target: Path, sources: list[Path]) -> bool:
    if not target.exists():
        return True
    t = target.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > t for s in sources)


def refresh(fetch: bool = True, refit: bool = True, deadline: float | None = None,
            force: bool = False, log=print, tx_dir: str | Path = TX_DIR,
            quotes: str | Path = QUOTES, table: str | Path = TABLE,
            checks: str | Path = CHECKS, belief: str | Path = BELIEF,
            summary: str | Path = CACHE / "summary.json", pause: float = 0.3,
            limit_cap: float | None = None) -> dict:
    """Sync, rebuild the table and checks, refit the belief. Never raises.

    The table and checks are rebuilt when a transaction file or the submission log is newer
    than the table (or force); the belief is refit only when a transaction file is newer.
    Returns a report: new games, row count, findings, what was skipped or failed."""
    from . import ab_table, consistency
    from .opponents import load_transactions

    rep: dict = {"new_games": [], "errors": []}
    if limit_cap is None:        # [pricing] acceptance_shrink_cap; no adaptive factor without it
        try:
            from .config import load_config
            pr = load_config().raw.get("pricing", {})
            limit_cap = float(pr.get("acceptance_shrink_cap", 1.0)) if pr.get("acceptance_shrink_from") else 1.0
        except Exception:  # noqa: BLE001
            limit_cap = 1.0
    tx_dir, table, belief = Path(tx_dir), Path(table), Path(belief)

    def fail(step: str, e: BaseException) -> None:
        rep["errors"].append(f"{step}: {e!r}"[:300])
        log(f"{step} failed: {e!r}"[:300])

    if fetch:
        try:
            rep["new_games"] = leaderboard.sync(tx_dir, pause=pause, log=log, deadline=deadline)
        except leaderboard.DeadlineReached:
            log("sync stopped at the deadline; the rest follows next time")
        except Exception as e:  # noqa: BLE001  network, API changes: retry next call
            fail("sync", e)
    txs_files = sorted(tx_dir.glob("game*.json"))
    if not txs_files:
        log("no transactions on disk yet")
        return rep

    if force or _stale(table, txs_files + [Path(quotes)]):
        try:
            rows = ab_table.rebuild(tx_dir, quotes, table)
            per_round = defaultdict(int)
            for r in rows:
                per_round[r.round] += 1
            rep["rows"] = dict(sorted(per_round.items()))
            log(f"table {table.name}: {len(rows)} rows, per round {rep['rows']}")
        except Exception as e:  # noqa: BLE001
            fail("table", e)
        try:
            found = consistency.check(load_transactions(tx_dir, dedup=False),
                                      consistency.our_quotes(quotes))
            consistency.write_csv(found, checks)
            rep["findings"] = found
            n = defaultdict(int)
            for f in found:
                n[f.severity] += 1
            log(f"checks {Path(checks).name}: {dict(n) or 'clean'}")
            for line in consistency.summary([f for f in found if f.severity != "info"]).splitlines():
                if line != "no findings":
                    log(f"  {line}")
        except Exception as e:  # noqa: BLE001
            fail("checks", e)
        try:
            write_summary(load_games(tx_dir), summary)
        except Exception as e:  # noqa: BLE001
            fail("summary", e)
        try:                         # adaptive factor on our acceptance limit
            from . import limit_factor
            if Path(summary) == limit_factor.SUMMARY and Path(quotes) == limit_factor.QUOTES:
                lf = limit_factor.update(cap=limit_cap) if limit_cap > 1.0 else None
                if lf:
                    rep["limit_factor"] = lf["factor"]
                    log(f"limit factor {lf['factor']} ({lf['model']}, {len(lf['games'])} games, "
                        f"saving vs 1.0: {lf['saving_vs_1']})")
        except Exception as e:  # noqa: BLE001
            fail("limit factor", e)

    if refit and _stale(belief, txs_files):
        if deadline is not None and deadline - time.time() < REFIT_SECONDS:
            log("refit postponed: next game too close")
        else:
            try:
                rep["belief"] = _refit(tx_dir, belief)
                log(f"belief {belief.name}: refit on games {rep['belief']}")
            except Exception as e:  # noqa: BLE001
                fail("refit", e)
    return rep


def _refit(tx_dir: Path, path: Path) -> list[int]:
    """Refit the opponent belief; only replace the file if every parameter is finite."""
    from .opponents import fit, load_transactions, save_belief

    b = fit(load_transactions(tx_dir))
    vals = [b.pi_uncovered] + [v for tb in b.teams.values() for v in vars(tb).values()]
    if not b.teams or not all(math.isfinite(v) for v in vals):
        raise ValueError("fit produced no teams or non-finite parameters; belief kept")
    save_belief(b, path)
    return b.games
