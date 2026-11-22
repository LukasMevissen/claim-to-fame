"""Analyst prompts of the overhaul (approach A: learning in context, docs/research/overhaul.md).

The live prompt (x3) shows the model the newest earlier cases and a PRICE PRIORS block. Two
things it does not do, both measured on the thresholds revealed so far:

1. Which cases it shows. With 50+ games only the newest 14 fit into 24,000 characters. Game 47
   was the fourth draw of the same case as games 16, 25 and 32; none of them was shown.
   `rel` sorts the reference cases by similarity to the current case (c2f/pricebook.py had
   the ranking, the agent never passed the case).
2. How a threshold is built. Read against the revealed thresholds, t is a unit rate times the
   INVOICED quantity times the policy's arithmetic. The model prices "the work actually
   needed" and trims: 14 skilled-worker hours estimated at 599 (t >= 1,260), 6.4 m copper pipe
   at 15 (t >= 211), a dryer for 7 days at 105 (t >= 287). One line in four that turned out
   covered was estimated below 0.75 x the highest charge proven fair. `x5` states the
   construction and adds, per recurring line type, the highest rate revealed so far
   (FULL-COVER LEVELS): a lower amount in another case is that case's policy arithmetic.

    system(prompt, claude)                       the system text of an arm
    reference(case, prompt, before_game, ...)    the reference text of an arm
"""

from __future__ import annotations

import math

from . import item_prior as ip, pricebook
from .extract import SYSTEM, SYSTEM_UPPER

PROMPTS = ("rel", "x5")

SYSTEM_X5 = """You are a senior claims adjuster. You receive an insurance policy, a damage description,
an invoice WITHOUT prices and sometimes a photo. For every invoice line a secret threshold t
exists: the highest gross total (incl. VAT, whole line) that a claims expert applying THIS
policy to THIS claim would indemnify. Estimate t for every line.

How t is built (read off the thresholds revealed in the earlier cases):
    t = unit rate x INVOICED quantity x policy arithmetic, or 0 if the line is not indemnified.
- The unit rate is the upper end of what a contractor customarily bills for that line, incl.
  VAT: installed trade rates, not material cost and not DIY-store prices. A line type keeps
  its rate from case to case.
- The quantity is the one printed on the invoice. Hours, metres, days and pieces are NOT
  trimmed because the description makes the job sound smaller than billed.
- A line is reduced only by something explicit: a policy clause (cap, sub-limit incl. VAT,
  rateable share of an aggregate limit, percentage for a breached duty, betterment, purchase
  price, deductible) or a figure in the claim that such a clause refers to.
- When the same line type was paid less per unit in an earlier case, that was that case's
  policy arithmetic, not a lower rate.

Work policy-first; the policy wording decides far more than market prices do:
1. Find the insured peril and the cover sections actually in force. Optional covers or
   endorsements count only if the description or schedule says they were taken.
2. Per line: is it excluded, unrelated to the damage, pre-existing, maintenance, an improvement,
   a duplicate of another line, or an ancillary/site charge the policy does not carry? Then
   t = 0. Check definitions and exclusions literally (volume limits, "standing on its own",
   external surfaces, combined lines that mix insured and uninsured work, who is paid, etc.).
3. If indemnified, take the unit rate from PRICE PRIORS / FULL-COVER LEVELS when the line
   type is listed there; otherwise from the most similar reference line; otherwise the upper
   end of the customary price range. Multiply by the invoiced quantity. Then apply this
   policy's arithmetic and nothing else.
   Figures in the description (purchase price, age, sizes, volumes) are there to be used.
4. REFERENCE CASES are earlier cases with what was later revealed about t, the most similar
   first. A reference case with the same invoice lines is the same case drawn again: a line
   indemnified there and here has the same threshold, so take its revealed range as it
   stands; what differs between two draws is which lines are covered and which clause bites.
5. A photo shows what was damaged: read off brand, model class, size and number of units
   and price that, not a generic item.

Return JSON {"items": [...]} with one object per invoice line, in invoice order:
- item_id: position number exactly as printed
- description: short
- reasoning: one or two sentences naming the decisive clause(s) and the arithmetic
- indemnified_prob: probability (0..1) that t > 0
- line_total_gross_eur: your best estimate of t IF it is indemnified, gross total for the
  whole line after all policy arithmetic. Always give a positive number, even if you
  believe the line is excluded.
- price_log_sd: uncertainty of that number as a log standard deviation (0.05 = fixed by the
  policy or a near-identical reference, 0.3 = normal, 0.7 = rough guess)
Do not skip lines. Do not invent lines."""


def system(prompt: str, claude: bool = False) -> str:
    if prompt == "x5":
        return SYSTEM_X5
    return SYSTEM_UPPER if claude else SYSTEM


def _rng(lo: float, hi: float) -> str:
    return f"{ip._eur(lo)}-{ip._eur(hi)}" if hi < math.inf else f">={ip._eur(lo)}"


def full_cover_levels(lines: dict, obs: list[ip.Obs], spread: float = 1.2) -> str:
    """Per recurring line type whose revealed rates differ between cases: the highest one."""
    by_key: dict[tuple[str, str], list[ip.Obs]] = {}
    for o in obs:
        if o.kind in ("bracket", "lower"):
            by_key.setdefault((o.key, o.unit), []).append(o)
    rows = []
    for pos, text in sorted((int(k), v) for k, v in lines.items()):
        desc, qty, unit = ip.parse_line(text)
        key, _ = ip.line_key(desc, unit)
        seen = by_key.get((key, unit), [])
        if len(seen) < 2:
            continue
        top = max(seen, key=lambda o: o.lo)
        low = min(seen, key=lambda o: o.lo)
        if top.lo < spread * low.lo:
            continue                                  # one level: the PRICE PRIORS row says it
        per = unit or "pc"
        others = ", ".join(f"{_rng(o.lo, o.hi)} (case {o.game})"
                           for o in sorted(seen, key=lambda o: -o.lo) if o.lo * spread < top.lo)
        total = f"; this line ({qty:g} {per}) at the full rate: {_rng(top.lo * qty, top.hi * qty)}" if qty != 1 else ""
        rows.append(f'- line {pos} "{desc}": full rate per {per} {_rng(top.lo, top.hi)} (case {top.game}); '
                    f"paid less in: {others}{total}")
    if not rows:
        return ""
    return ("FULL-COVER LEVELS (line types whose revealed rate differed between earlier cases: the\n"
            "highest revealed rate is the full rate, the lower ones were cut by their case's policy\n"
            "arithmetic. Start from the full rate and apply only what THIS policy and claim say):\n"
            + "\n".join(rows))


def reference(case, prompt: str = "x5", before_game: int | None = None, summary: dict | None = None,
              max_chars: int = 24000, with_prior: bool = True) -> str:
    """Reference text of an arm: the most similar earlier cases first, the PRICE PRIORS block,
    and for x5 the FULL-COVER LEVELS. Live call: no before_game / summary, everything revealed
    so far (the fitted prior is shared by the parallel stages). Backtest: games before
    `before_game` only, fitted on `summary`."""
    ref = pricebook.reference(before_game=before_game, max_chars=max_chars, case=case)
    lines = pricebook.invoice_lines(case.invoice_text)
    if before_game is None and summary is None:
        prior, block = ip.shared(), ip.context_for_case(case)
    else:
        prior = ip.ItemPrior(ip.observations(summary, pricebook.CASEBOOK, before_game=before_game))
        block = ip.context(lines, prior, policy=pricebook.policy_title(case.policy))
    parts = [ref, block if with_prior else ""]
    if prompt == "x5":
        parts.append(full_cover_levels(lines, [o for v in prior.by_key.values() for o in v]))
    return "\n\n".join(p for p in parts if p)
