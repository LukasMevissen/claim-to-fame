# Observer tools

Used by the supervising agent between games (docs/REPORT.md, section 6). Run from the repo root.

- `watch.sh GAME HHMMSS`: wait until the UTC time, print the bot log for the game, sync the
  public transactions, print the per-line review and the standings.
- `review.py GAME...`: per line our a, b, revenue, cost, net, the revealed t interval and the best team.
- `decomp.py`: how much of the gap to a perfect player comes from the model's estimates
  (coverage, price level) and how much from the pricing rule.
- `ens.py`, `bench.py`, `grid.py`: one-off experiments (ensembles of models, model benchmark
  on decrypted past cases, charge and limit multipliers). They expect decrypted past cases and
  earlier run outputs under /tmp/c2f_work and are kept for reference.
