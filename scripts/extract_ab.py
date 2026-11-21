"""Team | Round | Line item | a | b table from the public leaderboard transactions.

    python scripts/extract_ab.py              # fetch new completed games, write the table
    python scripts/extract_ab.py --no-fetch   # rebuild from data/transactions/

Writes results/ab_table.csv (one row per team, round, line item). Column meaning in
c2f/ab_table.py: a is exact when any reviewer accepted it or wrongly rejected it,
b is the infimum of the feasible bracket
(b <= true b < b_high); our_a / our_b are our own quotes for the same item.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c2f.ab_table import ab_table, attach_our_quotes, our_quotes, write_csv  # noqa: E402
from c2f.leaderboard import sync  # noqa: E402
from c2f.opponents import load_transactions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--dir", default=str(ROOT / "data" / "transactions"))
    ap.add_argument("--out", default=str(ROOT / "results" / "ab_table.csv"))
    ap.add_argument("--pause", type=float, default=0.3)
    ap.add_argument("--quotes", default=str(ROOT / "runs" / "submissions.jsonl"),
                    help="our submission log, joined in as our_a / our_b")
    args = ap.parse_args()
    if not args.no_fetch:
        sync(args.dir, args.pause)
    txs = load_transactions(args.dir)
    if not txs:
        sys.exit("no transactions on disk")
    rows = ab_table(txs)
    matched = attach_our_quotes(rows, our_quotes(args.quotes))
    write_csv(rows, args.out)
    live = [r for r in rows if not r.absent]
    src = Counter(r.a_source for r in live)
    finite = sum(r.b > 0 and r.b_high != float("inf") for r in live)
    print(f"{len(rows)} rows ({len(live)} from submitting teams) -> {args.out}")
    print(f"a exact: {src['accepted'] + src['wrongful_reject']}, hidden: {src['hidden']}; "
          f"b bracketed on both sides: {finite}/{len(live)}; "
          f"our quotes joined for {matched} round/item pairs")


if __name__ == "__main__":
    main()
