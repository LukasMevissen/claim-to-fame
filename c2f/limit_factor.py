"""Adaptive factor on the acceptance limit b, learned from finished games.

The limit is b = factor x Q(1/3) of our belief about t (c2f/pricing.py). Proposition 1 puts the
optimum where the share of fair charges crosses 2/3; measured on the opponents' real charges
(games 26-47) 94% of the charges 0-5% above Q(1/3) were fair, so the optimum sits a few per cent
higher. This module re-measures that after every game:

    for each candidate factor k in [1, cap]: what would we have paid as insurer in the recent
    games played by the current model, had the limit been k x Q(1/3) (capped at our estimate)?

and keeps the k with the lowest cost. The cost is the pessimistic reading: charges nobody
accepted are hidden, and each one is assumed to sit just above the highest visible charge, so a
limit above everything visible pays them all. Walk-forward on games 26-47: a cap of 1.06 chose
1.04-1.06 throughout (+7.8k to +10.3k, worst game -20); caps of 1.10 and more chased single
lines and had losing games, so the cap stays low.

Only games served by the same model as the latest one count (the factor fits the Opus states
and does nothing for the GPT states), newest `window` games. With fewer than `min_games` the
file is not written and the fixed [pricing] acceptance_shrink applies.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .belief import TBelief
from .config import REPO_ROOT
from .replay import US

PATH = REPO_ROOT / "results" / "market" / "limit_factor.json"
QUOTES = REPO_ROOT / "runs" / "submissions.jsonl"
SUMMARY = REPO_ROOT / "results" / "market" / "summary.json"


def insurer_cost(b: float, s: dict, n_teams: int, us: str = US) -> float:
    """What limit b would have paid on one line of a finished game (pessimistic about hidden
    charges). s: the line's entry in results/market/summary.json."""
    lo, hi = s["t_low"], s["t_high"]
    cost, known = 0.0, 0
    for k, ak in s["charges"].items():
        if k == us:
            continue
        known += 1
        if not ak:
            continue
        fair = s["fair"].get(k)
        if fair is None:
            fair = bool(lo) and ak <= lo
        if ak <= b:
            cost += ak
        elif fair:
            cost += 1.5 * ak
    top = max([v for v in s["charges"].values() if v] + [hi or 0.0, lo or 0.0])
    if b > top and b > 0:
        cost += max(n_teams - known, 0) * b
    return cost


def _records(quotes: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not quotes.exists():
        return out
    for line in quotes.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("ok") and not r.get("dry_run") and r.get("game_id"):
            out[int(r["game_id"])] = r
    return out


def _model(rec: dict) -> str:
    t = rec.get("timing") or {}
    return str(t.get("served_by") or t.get("model") or "")


def _base_limits(rec: dict, quantile: float) -> dict[int, tuple[float, float]]:
    """line -> (Q(quantile), ceiling). From the logged beliefs when the record has them
    (game 44 on); older records were submitted with factor 1, so the submitted limit is Q."""
    out: dict[int, tuple[float, float]] = {}
    est = rec.get("estimates") or []
    if est:
        for e in est:
            try:
                bel = TBelief(float(e["median"]), float(e["sigma"]), float(e["p_zero"]))
                q = bel.quantile(quantile)
                out[int(str(e["item_id"]).strip())] = (q, max(q, bel.median))
            except (KeyError, ValueError, TypeError):
                continue
        return out
    for p in rec.get("payload") or []:
        b = float(p.get("acceptance_limit") or 0.0)
        out[int(p["index"])] = (b, float("inf"))
    return out


def evaluate(records: dict[int, dict], summary: dict, cap: float = 1.06, step: float = 0.01,
             window: int = 20, quantile: float = 1.0 / 3.0, us: str = US) -> dict:
    """Cost of every candidate factor on the newest `window` finished games of the current model."""
    played = [g for g in sorted(records) if str(g) in summary and (records[g].get("payload"))
              and (records[g].get("timing") or {}).get("final_stage") != "fixed"]
    if not played:
        return {"games": [], "costs": {}}
    current = _model(records[max(records)])
    games = [g for g in played if _model(records[g]) == current][-window:]
    n = int(round((cap - 1.0) / step))
    grid = [round(1.0 + i * step, 4) for i in range(max(n, 0) + 1)]
    costs = {k: 0.0 for k in grid}
    for g in games:
        lines = summary[str(g)]
        teams = set()
        for s in lines.values():
            teams |= set(s["charges"]) | set(s["b_bounds"])
        n_teams = len(teams - {us})
        for i, (q, ceil) in _base_limits(records[g], quantile).items():
            s = lines.get(str(i))
            if s is None or q <= 0:
                continue
            for k in grid:
                costs[k] += insurer_cost(min(q * k, ceil), s, n_teams, us)
    return {"games": games, "model": current, "costs": costs}


def update(path: Path = PATH, quotes: Path = QUOTES, summary: Path = SUMMARY, cap: float = 1.06,
           min_games: int = 5, **kw) -> dict | None:
    """Re-measure and write the factor; None (file untouched) with too little evidence."""
    if not Path(summary).exists():
        return None
    ev = evaluate(_records(Path(quotes)), json.loads(Path(summary).read_text()), cap=cap, **kw)
    if len(ev["games"]) < min_games or not ev["costs"]:
        return None
    best = min(ev["costs"], key=lambda k: (round(ev["costs"][k], 2), k))   # ties: the smaller factor
    out = {"factor": best, "cap": cap, "model": ev["model"], "games": ev["games"],
           "saving_vs_1": {str(k): round(ev["costs"][1.0] - c, 2) for k, c in ev["costs"].items()},
           "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(path)
    return out


def load(path: str | Path, cap: float, default: float) -> float:
    """Factor for the next game: the learned one clamped to [1, cap], else `default`."""
    f = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    try:
        k = float(json.loads(f.read_text())["factor"])
    except (OSError, ValueError, KeyError, TypeError):
        return default
    if not k == k:      # NaN
        return default
    return min(max(k, 1.0), cap)
