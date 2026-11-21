"""Recover every team's (a, b) per round and line item from the public transactions.

Rules (docs/CASE.md payoff table), for issuer i and reviewer r on one item:
    accepted, amount m      -> a_i = m  (exact unless a_i > t and the cap c >= 4t bit),
                               b_r >= a_i
    rejected, amount m > 0  -> a_i = m / 1.5 (wrongful rejection, so a_i <= t), b_r < a_i
    rejected, amount 0      -> a_i > t (fraudulent), b_r < a_i, a_i itself not shown
    a team charging 0 on every item of a round did not submit: a = b = 0

    no positive payment on the whole item -> almost surely t = 0 (uncovered): accepted
                               charges are capped to 0 too, so a is not identified
So a is exact whenever any reviewer accepted it or wrongly rejected it; otherwise only a
lower bound is known (a_i > b_r >= b_low_r for every reviewer that rejected it).
b is never shown, only bracketed: b_low = largest known charge the team accepted,
b_high = smallest known charge it rejected, b_low <= b < b_high. The table reports
b = b_low, the smallest b consistent with the data (infimum of the feasible set).

t_low / t_high / cap: item-level facts from c2f.consistency.item_facts (same for every
team of a round and item): t_low <= t <= t_high, and c when a cap hit revealed it. A
charge accepted at exactly c by a fraudulent issuer is reported as a_source "capped":
a >= c, a_low = c, a itself hidden.

our_a / our_b: what our agent quoted for the same round and item (runs/submissions.jsonl,
last successful non-dry-run record per game, since later submissions overwrite earlier
ones). Empty where we have no live quote for that round.

rebuild() is the one entry point (CLI `python -m c2f market`, the scheduler after every
game, scripts/extract_ab.py): transactions on disk -> results/ab_table.csv.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .consistency import item_facts, our_quotes  # noqa: F401  (our_quotes re-exported)
from .opponents import Tx, _absent, load_transactions


@dataclass
class ABRow:
    team: str
    round: int
    line_item: int
    a: float | None          # exact charge, None if hidden
    a_low: float | None      # lower bound when a is hidden
    a_source: str            # accepted | wrongful_reject | capped | hidden | paid_zero_t0 | absent
    a_vs_t: str              # fair (a <= t) | fraud (a > t) | unknown
    b: float                 # infimum of feasible b (= b_low)
    b_high: float            # b < b_high (inf = never rejected a known charge)
    n_accepted: int          # how many charges this team accepted as reviewer
    n_rejected: int
    absent: bool
    t_low: float = 0.0           # item: t >= t_low (largest wrongly rejected charge)
    t_high: float = math.inf     # item: t <= t_high (accepted fraud charge, cap / 4, 0)
    cap: float | None = None     # item: payment cap c, when a cap hit revealed it
    our_a: float | None = None   # our agent's charge for this round and item
    our_b: float | None = None   # our agent's acceptance limit


def ab_table(txs: list[Tx]) -> list[ABRow]:
    absent = _absent(txs)
    by_item: dict[tuple[int, int], list[Tx]] = defaultdict(list)
    for x in txs:
        by_item[(x.game, x.item)].append(x)

    out = []
    for (g, j), rows in sorted(by_item.items()):
        a, src = {}, {}
        for x in rows:
            if x.amount > 0 and x.accepted:
                a[x.issuer], src[x.issuer] = x.amount, "accepted"
        for x in rows:
            if x.amount > 0 and not x.accepted and x.issuer not in a:
                a[x.issuer], src[x.issuer] = x.amount / 1.5, "wrongful_reject"
        verdict: dict[str, str] = {}
        for x in rows:
            if not x.accepted:
                verdict[x.issuer] = "fair" if x.amount > 0 else "fraud"

        it = item_facts(g, j, rows)
        lo: dict[str, float] = defaultdict(float)
        hi: dict[str, float] = defaultdict(lambda: math.inf)
        n_acc: dict[str, int] = defaultdict(int)
        n_rej: dict[str, int] = defaultdict(int)
        for x in rows:
            if x.accepted:
                n_acc[x.reviewer] += 1
            else:
                n_rej[x.reviewer] += 1
            ai = a.get(x.issuer)
            if ai is None:
                continue
            if x.accepted:
                lo[x.reviewer] = max(lo[x.reviewer], ai)
            elif x.issuer not in it.capped:             # a capped charge is only a >= c
                hi[x.reviewer] = min(hi[x.reviewer], ai)

        item_paid = any(x.amount > 0 for x in rows)   # any positive payment proves t > 0
        tb = dict(t_low=round(it.t_low, 2),
                  t_high=round(it.t_high, 2) if it.t_high < math.inf else math.inf,
                  cap=None if it.cap is None else round(it.cap, 2))
        teams = sorted({x.issuer for x in rows} | {x.reviewer for x in rows})
        for k in teams:
            if (g, k) in absent:
                out.append(ABRow(k, g, j, 0.0, None, "absent", "fair", 0.0, 0.0,
                                 n_acc[k], n_rej[k], True, **tb))
                continue
            if k in it.capped and src.get(k) == "accepted":   # paid c, its a is above
                ak, alow, s = None, round(it.capped[k], 2), "capped"
            elif k in a:
                ak, alow, s = round(a[k], 2), None, src[k]
            elif any(x.issuer == k and x.accepted and x.amount == 0 for x in rows):
                if item_paid:                            # t > 0 here, so paid 0 means a = 0
                    ak, alow, s = 0.0, None, "accepted"
                else:                                    # likely t = 0: cap pays 0, a unknown
                    ak, alow, s = None, None, "paid_zero_t0"
            else:
                rejecters = [x.reviewer for x in rows if x.issuer == k and not x.accepted]
                alow = max((lo[r] for r in rejecters), default=0.0)
                ak, alow, s = None, round(alow, 2), "hidden"
            bl, bh = lo[k], hi[k]
            out.append(ABRow(k, g, j, ak, alow, s, verdict.get(k, "unknown"),
                             round(bl, 2), round(bh, 2) if bh < math.inf else math.inf,
                             n_acc[k], n_rej[k], False, **tb))
    return out


def attach_our_quotes(rows: list[ABRow], quotes: dict[tuple[int, int], tuple[float, float]]) -> int:
    """Fill our_a / our_b in place. Returns how many (round, item) pairs matched."""
    hit = set()
    for r in rows:
        q = quotes.get((r.round, r.line_item))
        if q:
            r.our_a, r.our_b = q
            hit.add((r.round, r.line_item))
    return len(hit)


def write_csv(rows: list[ABRow], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ABRow.__dataclass_fields__))
        w.writeheader()
        for r in rows:
            d = asdict(r)
            w.writerow({k: ("" if v is None else v) for k, v in d.items()})


def rebuild(tx_dir: str | Path, quotes_path: str | Path, out: str | Path) -> list[ABRow]:
    """Transactions on disk + our submission log -> the table at `out`. Returns the rows."""
    txs = load_transactions(tx_dir)
    rows = ab_table(txs)
    attach_our_quotes(rows, our_quotes(quotes_path))
    write_csv(rows, out)
    return rows
