"""Round-by-round update of the opponent belief.

    python scripts/update_opponents.py            # fetch new completed games, refit, save
    python scripts/update_opponents.py --no-fetch # refit on what is on disk

1. Downloads the public transactions of every completed game that is not on disk yet
   (leaderboard API, the same endpoint the leaderboard's Transactions tab uses; one
   request per team and game, spaced out) into data/transactions/game<N>.json.
2. Refits the latent model (c2f/opponents.py) on all games, newer games weighted more.
3. Writes results/opponent_belief.json (per team: charge and acceptance rule, miss rate,
   absent rate; per item: posterior of t), which pricing reads as the opponent population.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c2f.leaderboard import sync  # noqa: E402
from c2f.opponents import fit, load_transactions, save_belief  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--dir", default=str(ROOT / "data" / "transactions"))
    ap.add_argument("--out", default=str(ROOT / "results" / "opponent_belief.json"))
    ap.add_argument("--decay", type=float, default=0.85)
    ap.add_argument("--pause", type=float, default=0.3)
    args = ap.parse_args()
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)

    if not args.no_fetch:
        sync(d, args.pause)

    txs = load_transactions(d)
    if not txs:
        sys.exit("no transactions on disk")
    belief = fit(txs, decay=args.decay)
    save_belief(belief, args.out)
    print(f"fitted {len(belief.teams)} teams on games {belief.games}, "
          f"P(uncovered)={belief.pi_uncovered:.2f}")
    print(f"{'team':28s} {'absent':>6s} {'alpha':>6s} {'sigma':>6s} {'beta':>6s} {'tau':>6s} {'miss':>5s}")
    for k, tb in sorted(belief.teams.items(), key=lambda kv: kv[1].beta):
        print(f"{k:28s} {tb.absent_rate:6.2f} {tb.alpha:6.2f} {tb.sigma:6.2f} "
              f"{tb.beta:6.2f} {tb.tau:6.2f} {tb.miss:5.2f}")


if __name__ == "__main__":
    main()
