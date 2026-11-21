"""Last-resort prices without any model call.

If every model call fails or is too slow (network, API outage, key out of credit), each
invoice line is priced from the most similar line of an earlier game whose threshold the
public transactions revealed (c2f/pricebook.py). Lines without a similar earlier line get
the fixed [fallback] prices. Runs in milliseconds and needs no network.
"""

from __future__ import annotations

import json
import re

from .extract import ItemEstimate
from .pricebook import CASEBOOK, SUMMARY, invoice_lines

_STOP = {"the", "and", "for", "from", "with", "incl", "pcs", "flat", "rate", "hrs", "per", "new",
         "one", "of", "to", "in", "on", "a", "an", "under", "into", "its"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zäöüß]{3,}", text.lower()) if w not in _STOP}


def _t_estimate(it: dict) -> float | None:
    """Point value for t from the revealed bounds; None if unknown, 0.0 if nobody was paid."""
    lo, hi = it["t_low"], it["t_high"]
    if not any(v > 0 for v in it["charges"].values()):
        return 0.0
    if lo > 0 and hi:
        return (lo + hi) / 2
    if lo > 0:
        return lo                      # proven fair; the truth may be higher
    return None                        # only an upper bound: too weak to price from


def references(before_game: int | None = None) -> list[tuple[set[str], float]]:
    if not SUMMARY.exists() or not CASEBOOK.exists():
        return []
    summary = json.loads(SUMMARY.read_text())
    out = []
    for g, items in summary.items():
        if before_game is not None and int(g) >= before_game:
            continue
        f = CASEBOOK / f"game{g}.json"
        if not f.exists():
            continue
        lines = json.loads(f.read_text())["lines"]
        for idx, it in items.items():
            t = _t_estimate(it)
            if t is not None and str(idx) in lines:
                out.append((_tokens(lines[str(idx)]), t))
    return out


def estimate_lines(invoice_text: str, before_game: int | None = None, min_sim: float = 0.5,
                   sigma: float = 0.35) -> list[ItemEstimate]:
    """One ItemEstimate per invoice line that has a similar earlier line (Jaccard similarity
    of the description words >= min_sim). Lines without a match are left out."""
    refs = references(before_game)
    out = []
    for pos, text in invoice_lines(invoice_text).items():
        tok = _tokens(text)
        best, best_t = 0.0, None
        for rt, t in refs:
            if tok and rt:
                sim = len(tok & rt) / len(tok | rt)
                if sim > best:
                    best, best_t = sim, t
        if best_t is None or best < min_sim:
            continue
        covered = 1.0 if best_t > 0 else 0.0
        out.append(ItemEstimate(str(pos), text[:80], 1.0, "", covered, 1.0,
                                best_t if best_t > 0 else 0.0, sigma,
                                f"local price book match (similarity {best:.2f})"))
    return out
