"""The belief (median, sigma, p_zero) behind every quote we submitted, for every game.

    python scripts/backfill_live_estimates.py            # writes results/live_estimates/
    python scripts/backfill_live_estimates.py --check    # also re-derives the logged games

Games with a logged belief are copied, games whose submitted stage has a stored extraction
are read from it, and the rest (backup stage before game 44) are reconstructed from the
submitted quotes. Method and status values: c2f/live_estimates.py. Needs the git history
(old config versions and opponent belief files are read from it).
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c2f import live_estimates as le  # noqa: E402


def check() -> None:
    """Reconstruct the games whose belief was logged and compare with the log."""
    res: collections.Counter = collections.Counter()
    for _, rec in sorted(le.load_run_log().items()):
        if not rec.get("estimates"):
            continue
        rule = le.rule_for_sha(rec["stamp"]["config_sha"])
        allowed = le.allowed_for(rec["stamp"]["git_sha"], rec["stamp"]["config_sha"])
        pay = {int(p["index"]): p for p in rec["payload"]}
        for k, e in enumerate(rec["estimates"], 1):
            p = pay.get(le._index(e["item_id"], k))
            if not p or rule is None:
                continue
            r = le.reconstruct(p["charge_price"], p["acceptance_limit"], rule, allowed)
            m_ok = r["median"] is not None and abs(r["median"] - e["median"]) < 0.01
            full = (m_ok and r["sigma"] == e["sigma"] and isinstance(r["p_zero"], float)
                    and abs(r["p_zero"] - e["p_zero"]) < 0.0051)
            res[(r["status"], "median right" if m_ok else "median WRONG",
                 "sigma and p_zero right" if full else "-")] += 1
    print(f"check on {sum(res.values())} logged lines:")
    for key, n in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {' | '.join(key)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "live_estimates"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    games = le.all_games()
    jp, cp = le.write(games, Path(args.out))
    by_source = collections.Counter(g["source"] for g in games)
    status = collections.Counter(ln["status"] for g in games for ln in g["lines"])
    print(f"{len(games)} games, {sum(status.values())} lines -> {jp} and {cp}")
    print("games by source:", dict(by_source))
    print("lines by status:", dict(status))
    if args.check:
        check()


if __name__ == "__main__":
    main()
