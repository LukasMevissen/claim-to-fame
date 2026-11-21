# Team ToDo

One section per contributor. Tick with `- [x]`, add with `- [ ] text #tag`.
Autosync shares this file within 2 minutes; the dashboard (`python scripts/dashboard.py serve`)
shows it next to the tasks it derives from the pipeline (unreviewed games, stale opponent
belief, unmerged branches, write-up gaps), so those do not need to be listed here.

## LukasMevissen
- [ ] Pick a charge rule from `scripts/novelty_backtest.py` and run the replay test (REPORT 6.2) before it goes live #strategy
- [ ] Autosync running on all three machines #infra
- [x] Opponent belief from public transactions #market
- [x] Autosync: commit, pull and push main #infra

## Diegocentrico2001
- [ ] Keep the scheduler (`python -m c2f run`) live for the remaining games #ops
- [ ] Extend docs/REPORT.md 7.5 live results past game 14 #report
- [x] Price book, gpt-6.2-sol extractor, charge floor 0.87 #strategy
- [x] Claude shadow models and shadow report #shadow

## milanova-n
- [ ] Read the write-up as someone who did not build the agent and mark unclear parts #writeup

## Team
- [ ] Final presentation: score chart and program diagram from the dashboard #presentation
