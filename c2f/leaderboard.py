"""Download public transactions from the QuantCo leaderboard.

Uses only the endpoints the public leaderboard page itself calls (Transactions tab):
    GET /leaderboard/api/games?completed_only=true&page_size=1000
    GET /leaderboard/api/transactions?game_id=<N>&team=<name>&page=<p>
One game = one request per team (the endpoint filters by team), spaced out by `pause`.
Each game is stored once as data/transactions/game<N>.json and never re-fetched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

BASE = "https://c2f.public.quantco.cloud/leaderboard/api"
US = "GemischtesHack"


def _get(path: str, params: dict, timeout: float = 10) -> dict:
    import requests
    r = requests.get(f"{BASE}/{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _pages(path: str, params: dict) -> list[dict]:
    out, page = [], 1
    while True:
        d = _get(path, {**params, "page": page})
        out += d.get("items", [])
        if page >= d.get("total_pages", 1):
            return out
        page += 1


def completed_games() -> list[int]:
    items = _get("games", {"completed_only": "true", "page_size": 1000}).get("items", [])
    return sorted(int(g.get("id", g.get("game_id"))) for g in items)


def fetch_game(game_id: int, pause: float = 0.3, seed_team: str = US) -> dict:
    """Every transaction of one game. The seed team's rows reveal all team names."""
    key = lambda r: (r["issuer"], r["reviewer"], r["line_item_index"])  # noqa: E731
    rows = {key(r): r for r in _pages("transactions", {"game_id": game_id, "team": seed_team})}
    teams = {r["issuer"] for r in rows.values()} | {r["reviewer"] for r in rows.values()}
    for t in sorted(teams - {seed_team}):
        time.sleep(pause)
        for r in _pages("transactions", {"game_id": game_id, "team": t}):
            rows[key(r)] = r
    return {"game_id": game_id, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "items": list(rows.values())}


def sync(directory: str | Path, pause: float = 0.3, log=print) -> list[int]:
    """Fetch every completed game not on disk yet. Returns the newly fetched game ids."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    new = []
    for gid in completed_games():
        f = d / f"game{gid}.json"
        if f.exists():
            continue
        doc = fetch_game(gid, pause)
        f.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        log(f"game {gid}: {len(doc['items'])} transactions")
        new.append(gid)
    return new
