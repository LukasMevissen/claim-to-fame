# Claim to Fame: bot

> Standalone prototype, **not wired into `c2f/`** (not imported, not packaged, not run by CI).
> Run it from inside this folder: `cd bot && python bot.py --game 0`.

## The math

For each line you submit **b** (insurer role) and **a** (handyman role). The two are separable.

**b, acceptance limit.** For a charge a, accepting costs a if it's fair and min(a, c) if it's
inflated. Rejecting costs 1.5·a if it's fair and 0 if it's inflated. Ignoring the cap, you
accept when P(t ≥ a) > 2/3:

> **b\* = the 1/3 quantile of your belief about t.** It doesn't depend on the other teams.

The cap c ≥ 4t, with an unknown absolute floor (GAME_DESCRIPTION). For excluded lines (t = 0),
wrongly accepting costs at most that floor. So for large charges it can be better to accept even
when exclusion is fairly likely. The code uses c = max(4t, `C2F_CAP_FLOOR`), default 100 € (an assumption).

**a, charge price.** Depends on the other teams' b:
R(a) = P(t≥a)·a·[1 + 0.5·P(b_other < a)] + P(t<a)·min(a,c)·P(b_other ≥ a).
It is solved by Monte Carlo with joint samples of (t, other teams' b).

**Belief about t.** The LLM gives a net unit price with a 10–90 % range and the probabilities
that the line is covered and related. The gross line total (qty × unit × 1.19) is computed in
code. The belief is recalibrated from resolved rounds:
- the bias and spread of t against our estimate;
- a logistic fit of P(t=0) on the LLM's probabilities;
- the other teams' b, resampled together with our error on that same line.

## Submissions per round (resubmitting is free, the last one counts)

1. **Emergency**, as soon as the line indices are parsed: a = 0, and b from the history-wide
   rule. This avoids the default b = 0, which rejects everything and pays 1.5·a on every fair line.
2. **Fast LLM** (Haiku, text only): submits at ~10–15 s.
3. **Full LLM** (Sonnet, with the photo), blended with the fast one: submits before second 50.

## Setup

```
pixi install                  # Python + 7-Zip (from the organizers' pixi.toml)
pixi add anthropic            # optional; llm.py uses plain HTTP
export TEAM_API_KEY=... ANTHROPIC_API_KEY=...
# download case_000.zip ... case_100.zip from Drive into ./cases/
pixi run python bot.py --game 0          # test game, stores nothing
nohup pixi run python bot.py > bot.log 2>&1 &
```

Check on game 0 that the line count matches the invoice and that totals are **gross**.

## Still open

- **Results feed.** The API has no results endpoint. The public leaderboard has Games /
  Matchup / Transactions tabs that fill in once games complete. Look at what they show, then fill
  in `api.get_results`. Ask the organizers whether reading that page is fine: the rules forbid
  extracting other teams' data by probing the API.
  - If t is not published, transactions still **bracket t**. A 1.5·a payment for our charge means
    a ≤ t; zero from a rejecting insurer means a > t. That gives interval-censored data to calibrate on.
- **Game → case mapping.** The code assumes game n uses `case_{n:03d}.zip`. Check this on the first real round.

## Scoring (EVALUATION_CRITERIA)

- **70 %: rank by cumulative net payoff**, scaled linearly from first to last.
- **30 %: write-up (max. 1 page) and pitch.** The criteria are method and uncertainty, evidence,
  resourcefulness ("no token maxing"), reliability within one minute, and clarity and honesty.
  - This design maps onto them: an explicit decision rule (b = q₁/₃), Monte Carlo for a, a
    simulator and walk-forward backtest as evidence, two cheap LLM calls per round, and staged
    submissions for reliability.

## Files

`bot.py` loop and submissions · `case.py` 7z and parsing · `llm.py` per-line judgement ·
`belief.py` calibrated belief · `decision.py` a\* and b\* · `learn.py` history, money, backtest,
autotune · `sim.py` all-vs-all game simulator.

## Simulation (made-up opponents)

Across seeds, the bot finished in the top 2 of 17 teams. The same LLM signal with a naive rule
(a = estimate, b = 1.1 × estimate) finished last or near last.
