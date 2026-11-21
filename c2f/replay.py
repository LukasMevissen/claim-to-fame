"""Counterfactual replay: what would quotes (a, b) have earned in a finished game?

Uses results/market/summary.json (c2f/market.py): per line the opponents' charges, whether
each charge was proven fair, bounds on every opponent's limit, and bounds on t.

For our charge a against opponent k with limit b_k in [lo_k, hi_k):
    P(accept) = 1 if a <= lo_k, 0 if a >= hi_k, linear in between
                (hi_k unknown: 1/2 up to 2 lo_k, then 0)
    revenue   = P(accept) a + (1 - P(accept)) 1.5 a 1[a <= t]
For opponent k's charge a_k against our limit b:
    cost      = a_k if a_k <= b, else 1.5 a_k if a_k <= t, else 0
t is a point inside the revealed bounds: the midpoint if both bounds are known,
`up` x the lower bound if only that is known (reported for up = 1.15 and 1.4),
0.85 x the upper bound if only that is known, 0 if nobody was ever paid.
Not modelled: the payment cap c and hidden charges (rejected by everyone) that a high b
of ours would have accepted.
"""

from __future__ import annotations

import json

from .config import REPO_ROOT

SUMMARY = REPO_ROOT / "results" / "market" / "summary.json"
US = "GemischtesHack"


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}


def t_point(s: dict, up: float = 1.15) -> float:
    if not any(v > 0 for v in s["charges"].values()):
        return 0.0
    lo, hi = s["t_low"], s["t_high"]
    if lo > 0 and hi:
        return (lo + hi) / 2
    if lo > 0:
        return lo * up
    return hi * 0.85 if hi else 0.0


def item_net(a: float, b: float, s: dict, t: float, us: str = US) -> float:
    rec = paid = 0.0
    for k in (set(s["fair"]) | set(s["b_bounds"]) | set(s["charges"])) - {us}:
        lo, hi = s["b_bounds"].get(k) or [0.0, None]
        if t == 0:
            pacc = 0.0
        elif a <= lo:
            pacc = 1.0
        elif hi is None:
            pacc = 0.5 if a <= 2 * lo else 0.0
        elif a >= hi:
            pacc = 0.0
        else:
            pacc = (hi - a) / (hi - lo)
        rec += pacc * a + (1 - pacc) * (1.5 * a if a <= t else 0.0)
        ak = s["charges"].get(k)
        if ak is not None:
            paid += ak if ak <= b else (1.5 * ak if ak <= t else 0.0)
    return rec - paid


def game_net(quotes: dict[int, tuple[float, float]], game: int, summary: dict | None = None,
             up: float = 1.15) -> float | None:
    """Net payoff of quotes {line: (a, b)} in a finished game; None if not published yet."""
    summary = summary if summary is not None else load_summary()
    g = summary.get(str(game))
    if g is None:
        return None
    return sum(item_net(*quotes.get(int(i), (0.0, 0.0)), s, t_point(s, up)) for i, s in g.items())
