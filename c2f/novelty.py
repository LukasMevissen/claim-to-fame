"""Novelty detector: has the market already priced this kind of invoice line?

Every finished game reveals bounds on the secret threshold t of each line (c2f/market.py)
and we keep the line texts (results/casebook). A line of a new case is

    anchored      a similar earlier line had a charge proven fair (t >= t_low > 0); the
                  model has seen that bound in the price book and prices close to t
    seen_unpaid   similar earlier lines exist, but none was ever proven fair
    new           no similar earlier line: the model prices from general knowledge, which
                  ran well below t in games 10-22, while most teams reject (and pay 1.5x)

Runs in milliseconds, no network. Not wired into pricing; scripts/novelty_backtest.py
replays charge rules that act on the flag.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .pricebook import CASEBOOK, SUMMARY

_STOP = {"the", "and", "for", "from", "with", "incl", "pcs", "flat", "rate", "hrs", "per", "new",
         "one", "of", "to", "in", "on", "a", "an", "under", "into", "its", "at", "was", "kwh"}
MIN_SIM = 0.6


def tokens(text: str) -> frozenset[str]:
    text = re.sub(r"^.*POS\. DESCRIPTION AMOUNT UNIT TOTAL \| \d+ ", "", text)
    return frozenset(w for w in re.findall(r"[a-zäöüß]{2,}", text.lower()) if w not in _STOP)


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard; a short line fully contained in a longer one (>= 2 words) counts as 0.9."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    jac = inter / len(a | b)
    small = min(len(a), len(b))
    return max(jac, 0.9 * inter / small) if small >= 2 else jac


@dataclass(frozen=True)
class Novelty:
    status: str                  # "new" | "anchored" | "seen_unpaid"
    sim: float = 0.0             # similarity of the precedent used
    ref: tuple[int, int] | None = None   # (game, line) of that precedent
    t_low: float = 0.0
    t_high: float | None = None


def precedents(before_game: int | None = None, summary: dict | None = None) -> list[tuple]:
    """(tokens, game, line, t_low, t_high) for every earlier line with published transactions."""
    if summary is None:
        summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    out = []
    for g, items in summary.items():
        if before_game is not None and int(g) >= before_game:
            continue
        f = CASEBOOK / f"game{g}.json"
        if not f.exists():
            continue
        lines = json.loads(f.read_text(encoding="utf-8"))["lines"]
        for idx, it in items.items():
            if str(idx) in lines:
                out.append((tokens(lines[str(idx)]), int(g), int(idx), it["t_low"], it["t_high"]))
    return out


def classify(lines: dict[int, str], before_game: int | None = None,
             summary: dict | None = None, min_sim: float = MIN_SIM) -> dict[int, Novelty]:
    """Line number -> Novelty, using only games before `before_game`."""
    refs = precedents(before_game, summary)
    out = {}
    for pos, text in lines.items():
        tok = tokens(text)
        hits = [(similarity(tok, rt), g, i, lo, hi) for rt, g, i, lo, hi in refs]
        hits = [h for h in hits if h[0] >= min_sim]
        fair = [h for h in hits if h[3] > 0]
        if fair:
            s, g, i, lo, hi = max(fair, key=lambda h: (h[0], h[1]))
            out[int(pos)] = Novelty("anchored", round(s, 2), (g, i), lo, hi)
        elif hits:
            s, g, i, lo, hi = max(hits, key=lambda h: (h[0], h[1]))
            out[int(pos)] = Novelty("seen_unpaid", round(s, 2), (g, i), lo, hi)
        else:
            out[int(pos)] = Novelty("new")
    return out
