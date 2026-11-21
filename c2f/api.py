"""QuantCo API client (vendor/quantco/API_HANDBOOK.md).

Fair play: the key is only requested at or after a game's start_time, at most once per second.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone

import requests

from .config import Config


class ApiNotConfigured(RuntimeError):
    pass


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


class QuantCoClient:
    def __init__(self, cfg: Config):
        api = cfg.section("api")
        for k in ("base_url", "games_endpoint", "key_endpoint", "submit_endpoint"):
            if not api.get(k):
                raise ApiNotConfigured(f"[api].{k} missing in config")
        self.base = api["base_url"].rstrip("/")
        self.api = api
        key = os.environ.get("TEAM_API_KEY", "").strip()
        if not key:
            raise ApiNotConfigured("TEAM_API_KEY not set")
        self.s = requests.Session()
        self.s.headers["X-API-Key"] = key

    def _url(self, ep: str, game_id: int | str) -> str:
        return self.base + self.api[ep].format(game_id=game_id)

    def list_games(self) -> list[dict]:
        r = self.s.get(self.base + self.api["games_endpoint"], timeout=10)
        r.raise_for_status()
        return r.json()

    def fetch_key(self, game_id: int | str, start_time: float | None = None,
                  poll_seconds: float | None = None) -> str:
        """Wait for start_time (never ask early), then poll once per second on 403."""
        poll = self.api.get("key_poll_seconds", 20) if poll_seconds is None else poll_seconds
        if start_time is not None:
            wait = start_time - time.time()
            if wait > 0:
                time.sleep(wait + 0.2)
        give_up = time.monotonic() + poll
        while True:
            try:
                r = self.s.get(self._url("key_endpoint", game_id), timeout=5)
            except requests.RequestException as e:   # slow server at game start: retry
                if time.monotonic() > give_up:
                    raise RuntimeError(f"key for game {game_id}: {e!r}") from e
                time.sleep(1.0)
                continue
            if r.status_code == 200:
                return r.json()["decryption_key"]
            # 403 = not started yet. 5xx / 429 = the server is overloaded at the game start
            # (game 48: one 503 and the game was lost; the server answered again 35 s later).
            retry = r.status_code == 403 or r.status_code == 429 or r.status_code >= 500
            if not retry or time.monotonic() > give_up:
                raise RuntimeError(f"key for game {game_id}: {r.status_code} {r.text[:200]}")
            time.sleep(1.0)

    def submit(self, game_id: int | str, payload: list[dict]) -> list[dict]:
        for p in payload:
            for k in ("charge_price", "acceptance_limit"):
                v = p[k]
                if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                    raise ValueError(f"invalid {k}={v!r} for index {p['index']}")
        r = self.s.put(self._url("submit_endpoint", game_id), json=payload, timeout=8)
        if r.status_code != 200:
            raise RuntimeError(f"submit game {game_id}: {r.status_code} {r.text[:300]}")
        return r.json()
