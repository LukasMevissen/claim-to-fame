"""Price book: what earlier games revealed about the secret thresholds t.

After a game completes, the public transactions bound t for every line (c2f/market.py):
a wrongly rejected charge proves t >= a, a rightly rejected one proves t < a, and a line
on which nobody was ever paid has t = 0. Cases repeat themes (same policy editions, same
kinds of repair), so these bounds are the best price evidence we have. We store a short
digest of every case we played (results/casebook/gameN.json) and show the model the
earlier cases with their revealed bounds as reference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import REPO_ROOT

CASEBOOK = REPO_ROOT / "results" / "casebook"
SUMMARY = REPO_ROOT / "results" / "market" / "summary.json"


def invoice_lines(text: str) -> dict[int, str]:
    """POS number -> 'description | quantity unit' from the extracted invoice text."""
    body = text.split("TOTAL", 1)[-1]
    out: dict[int, str] = {}
    cur, expected = None, 1
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("INVOICE", "Created on", "Invoice")):
            if line.startswith(("INVOICE", "Invoice")):
                cur = None
            continue
        m = re.match(rf"^{expected}\s+(\S.*)$", line)
        if m and len(m.group(1)) >= 6:
            cur = expected
            out[cur] = m.group(1)
            expected += 1
        elif cur is not None:
            out[cur] += " | " + line if re.match(r"^[\d.,]+\s+\S+", line) else " " + line
    return out


def policy_title(policy: str) -> str:
    m = re.search(r'Policy wording[^\n]*', policy)
    kind = re.search(r'^[A-Z][A-Z ]+INSURANCE', policy, re.MULTILINE)
    return " / ".join(x.group(0).strip() for x in (kind, m) if x)


def save_digest(game_id: int, case) -> None:
    CASEBOOK.mkdir(parents=True, exist_ok=True)
    (CASEBOOK / f"game{game_id}.json").write_text(json.dumps({
        "game": game_id, "policy": policy_title(case.policy),
        "description": " ".join(case.description.split()),
        "lines": invoice_lines(case.invoice_text)}, indent=1, ensure_ascii=False))


def _bounds(it: dict) -> str:
    lo, hi = it["t_low"], it["t_high"]
    paid = any(v > 0 for v in it["charges"].values())
    if not paid:
        return "t = 0 (not indemnified: nobody was paid anything)"
    if lo > 0 and hi:
        return f"t between {lo:.0f} and {hi:.0f}"
    if lo > 0:
        return f"t >= {lo:.0f} (highest charge proven fair; true t may be higher)"
    if hi:
        return f"t below {hi:.0f}, possibly 0 (lowest charge proven too high)"
    return "t unknown"


def reference(before_game: int | None = None, max_chars: int = 24000) -> str:
    """Earlier cases with revealed bounds, newest first, as prompt text ('' if none)."""
    if not SUMMARY.exists() or not CASEBOOK.exists():
        return ""
    summary = json.loads(SUMMARY.read_text())
    blocks = []
    for g in sorted((int(k) for k in summary), reverse=True):
        if before_game is not None and g >= before_game:
            continue
        f = CASEBOOK / f"game{g}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        rows = [f"  - {d['lines'].get(str(i), d['lines'].get(i, '?'))} => {_bounds(it)}"
                for i, it in sorted(((int(k), v) for k, v in summary[str(g)].items()))]
        blocks.append(f"Case {g} [{d['policy']}]\n  Claim: {d['description'][:600]}\n" + "\n".join(rows))
    out = ""
    for b in blocks:
        if len(out) + len(b) > max_chars:
            break
        out += b + "\n\n"
    return out.strip()
