"""Public game data -> what every team charged/accepted per line item, and bounds on t.

Source: the public leaderboard API (same data as the Transactions tab), only for completed
games. Each transaction is issuer (charges a) -> reviewer (limit b), accepted?, amount paid.

What one transaction reveals (payoff table in docs/CASE.md):
  accepted, amount x        -> issuer a = x (unless capped at c), reviewer b >= a
  rejected, amount x > 0    -> wrongful reject: a = x / 1.5 and a <= t; reviewer b < a
  rejected, amount 0        -> a > t (a itself is known from the issuer's other transactions)

Raw transactions are cached in results/market/game_XXX.json; completed games never change.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import requests

from .config import REPO_ROOT

PUBLIC = "https://c2f.public.quantco.cloud/leaderboard/api"
CACHE = REPO_ROOT / "results" / "market"


def _get(path: str, **params) -> dict:
    r = requests.get(f"{PUBLIC}/{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def completed_games() -> list[int]:
    return sorted(g["id"] for g in _get("games", completed_only="true", page_size=1000)["items"])


def teams() -> list[str]:
    return [t["team_name"] for t in _get("matrix", page=1, game_limit=1)["items"]]


def fetch_game(game_id: int, refresh: bool = False) -> list[dict]:
    """All transactions of a completed game (deduplicated across team queries)."""
    path = CACHE / f"game_{game_id:03d}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    seen, out = set(), []
    names = teams()
    for team in names[:-1]:      # every pair involves at least one of the first n-1 teams
        page = 1
        while True:
            d = _get("transactions", game_id=game_id, team=team, page=page, page_size=100)
            for t in d["items"]:
                k = (t["issuer"], t["reviewer"], t["line_item_index"])
                if k not in seen:
                    seen.add(k)
                    out.append(t)
            if page >= d.get("total_pages", 1):
                break
            page += 1
        time.sleep(0.2)          # be gentle with the public API
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


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


def sync(refresh: bool = False) -> dict[int, dict]:
    """Fetch every completed game and write results/market/summary.json."""
    games = completed_games()
    summary = {g: summarize(fetch_game(g, refresh)) for g in games}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
