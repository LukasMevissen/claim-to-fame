"""Download public transactions from the QuantCo leaderboard (the one canonical fetcher).

Uses only the endpoints the public leaderboard page itself calls (Transactions tab):
    GET /leaderboard/api/games?completed_only=true&page_size=1000
    GET /leaderboard/api/transactions?game_id=<N>&team=<name>&page=<p>&page_size=100
One game = one request per team (the endpoint filters by team), spaced out by `pause`.
The last team is skipped: every ordered pair (issuer, reviewer) involves at least one of
the other teams, so its rows are already in. Games that are not completed yet are simply
not listed, so a caller retries them on its next call.

Each game is stored once as data/transactions/game<N>.json and never re-fetched, unless
it came back incomplete (not every ordered pair present on every item): then the file
is marked "complete": false and re-fetched on later syncs, at most MAX_ATTEMPTS times.
The earlier cache results/market/game_XXX.json (same rows, older fetcher) is imported
once, so nothing already downloaded is requested again.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

from .config import REPO_ROOT

BASE = "https://c2f.public.quantco.cloud/leaderboard/api"
US = "GemischtesHack"
DEFAULT_DIR = REPO_ROOT / "data" / "transactions"
LEGACY_DIR = REPO_ROOT / "results" / "market"
MAX_ATTEMPTS = 3


class DeadlineReached(RuntimeError):
    """The caller's time budget ran out between two requests (nothing half-written)."""


def _get(path: str, params: dict, timeout: float = 10) -> dict:
    import requests
    r = requests.get(f"{BASE}/{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _pages(path: str, params: dict, deadline: float | None = None) -> list[dict]:
    out, page = [], 1
    while True:
        if deadline is not None and time.time() > deadline:
            raise DeadlineReached(f"{path} {params}")
        d = _get(path, {**params, "page": page, "page_size": 100})
        out += d.get("items", [])
        if page >= d.get("total_pages", 1):
            return out
        page += 1


def completed_games() -> list[int]:
    items = _get("games", {"completed_only": "true", "page_size": 1000}).get("items", [])
    return sorted(int(g.get("id", g.get("game_id"))) for g in items)


def missing_pairs(items: list[dict]) -> dict[int, int]:
    """Line item -> number of ordered (issuer, reviewer) pairs absent from the rows.
    A finished game has n (n - 1) rows per item for n teams."""
    teams = {r["issuer"] for r in items} | {r["reviewer"] for r in items}
    n = len(teams)
    pairs: dict[int, set] = defaultdict(set)
    for r in items:
        pairs[int(r["line_item_index"])].add((r["issuer"], r["reviewer"]))
    return {j: n * (n - 1) - len(p) for j, p in sorted(pairs.items()) if len(p) != n * (n - 1)}


def fetch_game(game_id: int, pause: float = 0.3, seed_team: str = US,
               deadline: float | None = None) -> dict:
    """Every transaction of one game. The seed team's rows reveal all team names."""
    key = lambda r: (r["issuer"], r["reviewer"], r["line_item_index"])  # noqa: E731
    rows = {key(r): r for r in _pages("transactions", {"game_id": game_id, "team": seed_team},
                                      deadline)}
    teams = {r["issuer"] for r in rows.values()} | {r["reviewer"] for r in rows.values()}
    for t in sorted(teams - {seed_team})[:-1]:
        time.sleep(pause)
        for r in _pages("transactions", {"game_id": game_id, "team": t}, deadline):
            rows[key(r)] = r
    items = list(rows.values())
    return {"game_id": game_id, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "complete": bool(items) and not missing_pairs(items), "items": items}


def _write(path: Path, doc: dict) -> None:
    """Atomic write: a reader (or a crash) never sees half a file."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _needs_fetch(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return doc.get("complete", True) is False and doc.get("attempts", 1) < MAX_ATTEMPTS


def import_legacy(directory: str | Path = DEFAULT_DIR, legacy: str | Path | None = None,
                  log=print) -> list[int]:
    """Copy results/market/game_XXX.json (bare list of rows) into data/transactions/ when
    that game is not there yet. Returns the imported game ids."""
    d, src = Path(directory), Path(legacy or LEGACY_DIR)
    if not src.is_dir():
        return []
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(src.glob("game_*.json")):
        try:
            gid = int(f.stem.split("_")[1])
            rows = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, IndexError, OSError):
            continue
        if not isinstance(rows, list) or (d / f"game{gid}.json").exists():
            continue
        _write(d / f"game{gid}.json", {"game_id": gid, "fetched_at": f"imported from {f.name}",
                                       "complete": bool(rows) and not missing_pairs(rows),
                                       "items": rows})
        log(f"game {gid}: imported {len(rows)} transactions from {f}")
        out.append(gid)
    return out


def sync(directory: str | Path = DEFAULT_DIR, pause: float = 0.3, log=print,
         deadline: float | None = None) -> list[int]:
    """Fetch every completed game not on disk yet. Returns the newly stored game ids.

    deadline (epoch seconds): no request is started after it; DeadlineReached is raised
    and the game in progress is not written (it is fetched again next time)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    new = import_legacy(d, log=log)
    for gid in completed_games():
        f = d / f"game{gid}.json"
        if not _needs_fetch(f):
            continue
        attempts = 1
        if f.exists():
            attempts += json.loads(f.read_text(encoding="utf-8")).get("attempts", 1)
        time.sleep(pause)
        doc = fetch_game(gid, pause, deadline=deadline)
        doc["attempts"] = attempts
        _write(f, doc)
        gap = "" if doc["complete"] else f" INCOMPLETE, missing pairs per item {missing_pairs(doc['items'])}"
        log(f"game {gid}: {len(doc['items'])} transactions{gap}")
        new.append(gid)
    return new
