# Charge agent: the charge a per line, decided from all earlier rounds

Status: 2026-11-22. Code `c2f/charge_agent.py`, switch `[charge_agent]` in
`configs/default.toml` (**shadow run** of rule / risk neutral since 2026-11-22, section 8), backtest `scripts/charge_agent_backtest.py`,
tests `tests/test_charge_agent.py`. Data: transactions and case digests of games 1 to 48,
stored `gpt-6.2-sol` assessments of games 10 to 47 (snapshot of `origin/main`).

## 1. Short answer

1. **The fixed rule is replaced by a decision per line.** Today a = max(optimiser, 0.87 × estimate)
   on every line; the 0.87 comes from a replay of games 1 to 12. The agent sets a from three
   things it measures on every earlier round: how wrong our estimates really are given the
   uncertainty the model states, which earlier lines predict the current threshold, and how
   the other teams accept charges.
2. **Walk-forward on games 13 to 47 (35 games, 367 lines), handyman revenue against the fixed
   rule including the confident-line floor of `origin/main`:**

   | Mode / setting | Revenue | Difference | 90 % interval | Games better / worse |
   |---|---|---|---|---|
   | fixed rule (0.87, confident lines 0.95) | 420.1k | | | |
   | rule / risk neutral | 443.6k | **+23.5k** | −5.9k to +53.3k | 17 / 14 |
   | rule / risk loving | 435.1k | +15.0k | −34.6k to +62.6k | 18 / 14 |
   | llm (judge gpt-4.1) / risk neutral | 437.6k | +17.5k | −7.4k to +42.6k | 19 / 12 |
   | llm (judge gpt-4.1) / risk loving | 426.8k | +6.7k | −48.2k to +60.0k | 17 / 15 |
   | llm (judge gpt-6.2-sol@low) / risk neutral | 449.6k | +29.5k | +0.2k to +61.5k | 17 / 14 |
   | llm (judge gpt-6.2-sol@low) / risk loving | 424.6k | +4.5k | −54.0k to +57.8k | 17 / 15 |

   Risk neutral is ahead in both modes by about +6 % of revenue; the intervals still touch
   zero. Risk loving does not earn more on average and doubles the spread.
3. **Time.** Rule mode: 10 to 250 ms per invoice. LLM mode: one judge call in parallel to the
   assessment (median 3.4 s, maximum 9.8 s with gpt-4.1), cut off after 15 s; pricing waits
   for it at most 20 s after the game start and never in the last 6 s. With the agent on,
   the slow search of the fixed rule is skipped: on the dry run of game 0 the final answer
   came after 12 s instead of 24 s.
4. **Recommendation:** switch on `mode = "rule"`, `risk = "neutral"`. LLM mode is in the same
   range and costs one extra call per game. Do not use risk loving unless we need variance
   to change rank (section 6).

## 2. Design

Per line the assessment gives an estimate m and a stated log-sd. The agent builds a
distribution of t and picks the a with the highest expected utility.

**Belief of the assessment.** log(t / m) is normal with a spread per class of stated log-sd,
plus a 10 % wide component for estimates that miss by policy arithmetic. The spread is fitted
on all earlier rounds by censored maximum likelihood (a bracket counts with the probability of
its interval, a lower bound with an exponential gap above it). Fitted by game 47:

| Stated log-sd | below 0.12 | 0.12 to 0.2 | 0.2 to 0.4 | 0.4 and above |
|---|---|---|---|---|
| Actual spread | 0.05 | 0.2 | 0.5 | 0.9 |

No bias is fitted: it adds spread without a gain (section 5).

**Which earlier rounds are informative.** Every earlier line with a proven bracket or lower
bound is a precedent. Its relevance w is the measured share of earlier pairs of the same match
class whose thresholds per unit were within 15 % of each other:

| Earlier line is | Relevance by game 47 |
|---|---|
| same line type, same policy wording | 0.58 to 0.76 |
| same line type, other wording, similar claim | 0.61 |
| same line type, other wording, other claim | 0.17 |
| similar text only | 0.24 to 0.48 |

A precedent enters the belief as (1 − w) × "tells nothing" + w × "t per unit within 8 % of
what was proven". So a tight bracket from the same wording moves the belief strongly, a lower
bound from another wording hardly at all, and a precedent far from the estimate (another
indemnity basis) is ignored instead of averaged in. Lines in another unit and other labour
grades are never compared.

**Decision.** a maximises E[revenue(a, t)^γ]. Revenue per opponent is a × (1.5 − 0.5 × P(accept))
while a ≤ t and min(a, 4t) × P(accept) above t. P(accept) as a function of a / t is measured
on the limit bounds of all opponents in earlier games (33 % accept a charge at t, 21 % at 1.5 t).
γ = 1 is risk neutral, γ = 2 risk loving. Guard rails: 0.5 m ≤ a ≤ 1.6 m.

**Two modes for the relevance.**

- `rule`: the table above. No model call.
- `llm`: a judge model reads the current claim and invoice next to all earlier cases with their
  revealed thresholds per unit (17k tokens at game 47) and names, per line, the informative
  earlier lines with a weight. The weight is not used as a probability: lines named with
  w ≥ 0.9 proved the same in about 70 % of cases, with 0.7 to 0.9 in 40 to 60 %, below that in
  5 to 20 %, and named lines that are no text match in under 10 %. The relevance is therefore
  the measured share per (rule class, judge class), learned from the stored judgements of
  earlier games (`results/charge_agent/judge/`). Without an answer in time: rule weights.

The limit b is not touched.

## 3. Results in detail

Reproduce: `python scripts/charge_agent_backtest.py --llm gpt-4.1` (judgements are cached, so
the rerun makes no model calls). For game g the agent sees only games before g: history,
relevance, calibration, acceptance curve and judge calibration are all refitted per game.
Scoring as in `c2f/replay.py` with the cap at 4t; where t is bracketed, the mean over the
bracket; where only a lower bound is known, t = 1.15 × the bound.

Where the difference comes from (rule / neutral):

| Lines | Count | Difference |
|---|---|---|
| stated log-sd 0.4 and above | 115 | +24.7k |
| stated log-sd 0.2 to 0.4 | 186 | +8.6k |
| stated log-sd below 0.2 | 66 | −9.9k |
| with at least one precedent | 132 | +16.9k |
| without | 235 | +6.6k |
| t proven only from below | 98 | +33.4k |
| t bracketed | 105 | −8.2k |

The agent charges confident estimates at about 0.9 m and uncertain ones at m or above. The
gain sits on lines the whole market undercharged (large repair items, m / t about 0.65). The
loss on confident lines is a few policy-fixed amounts where t was at m and 0.95 m beat 0.9 m.

Robustness of rule / neutral (difference to the fixed rule):

| Variant | Difference |
|---|---|
| games 13 to 29 / 30 to 47 | +8.6k / +14.9k |
| newest game not yet published at decision time | +18.4k |
| t = the lower bound itself / 1.4 × the bound | +0.7k / +25.3k |
| between-case spread 4 % / 15 % instead of 8 % | +20.2k / +19.0k |
| precedent evidence at half strength | +27.0k |
| upper guard rail 1.1 m / 1.3 m / 2.0 m instead of 1.6 m | −0.3k / +10.6k / +24.1k |

## 4. Time budget

| Step | Time | Limit |
|---|---|---|
| History, relevance, calibration (once per game, in the background) | 0.6 s | not on the pricing path |
| Rule mode, 22-line invoice | 10 to 250 ms | |
| Judge call gpt-4.1, 35 games | median 3.4 s, max 9.8 s | cut off at `judge_timeout_seconds` = 15 |
| Judge call gpt-6.2-sol@low, 35 games | median 4.8 s, max 12.7 s; 15.2 s once on 22 lines | same cut-off, then rule weights |
| Waiting for the judge | 0 s when the assessment is slower (the usual case) | `judge_wait_seconds` = 20 after the start; never in the last 6 s |

The judge starts when pricing starts and runs next to the assessment (9 to 45 s in the live
log). The quick safety-net stage never waits. Any error in the agent leaves the fixed-rule
charges in place. Dry run of game 0 (4 lines, nothing submitted): final answer after 24.4 s
without the agent's shortcut, 12.8 s (llm) and 11.8 s (rule) with it, because the fixed rule
needs about 3.6 s per line against the 22 learned opponent types and the agent skips it.

## 5. What did not work

- **Fitting a bias of the estimates per class** (within ±0.1): the fit says t is 5 to 10 %
  above m, and confident lines are then charged above t more often. Risk neutral unchanged
  (+23.3k, interval −17.7k to +61.8k), risk loving −9.8k instead of +15.0k.
- **Hand-set similarity weights** (text similarity × 0.5 for another wording): one loose
  precedent halved a 1,490 charge whose threshold was at least 1,400 (game 27).
- **The judge's weights taken at face value**: +4.7k to +8.5k instead of +17.5k to +29.5k.

## 6. Risk loving

γ = 2 charges higher on every line (never lower, tested). Even when t lies well above the
proven bound (1.4 × reading) it only draws level with risk neutral, and it loses 20k to 27k
against it when t is at the bound. γ = 1.5 gives +7.3k, γ = 3 gives
−18.6k against the fixed rule. Use it only if a higher chance of a large swing is worth a lower
expected score, for example in the last games when only a rank change counts.

## 7. Limits

1. **The intervals include zero.** 35 games; a few large lines decide a game (game 38: one
   air-conditioning line, −8k).
2. **No gain if t equals the highest proven-fair charge** on lines where no charge was proven
   too high. When all teams charge fairly, t is probably well above the highest charge, but
   that is not proven.
3. **The gain depends on charging uncertain lines above the estimate.** With the upper guard
   rail at 1.1 m the agent only matches the fixed rule.
4. **Calibrated on `gpt-6.2-sol` assessments.** The live answer usually comes from
   `claude-opus-5-5`, whose stated log-sd is not stored. If it means something else, the
   classes are off. Store its assessments (`results/extractions`) and refit.
5. **Replay limits** as in `c2f/replay.py`: hidden charges and the true cap are not modelled.
6. This checkout was 24 commits behind `origin/main` when tested; the baseline replicates the
   confident-line floor of `origin/main`, not its other changes.

## 8. Shadow run and takeover

    [charge_agent]
    mode = "rule"        # or "llm"
    risk = "neutral"     # or "loving"
    shadow = true        # false: submit the agent's charges at once
    takeover = false     # true: submit the agent's charges after takeover_rounds wins in a row
    takeover_rounds = 3

- **In shadow** the agent prices every live game, but the fixed rule's charges are submitted.
  The agent's charges are recorded per line in `runs/submissions.jsonl` under
  `timing.charge_agent.shadow_charges`. A shadow run never waits for the judge and adds
  about 0.2 s to pricing.
- **Scoring.** When a round's transactions are public, the scheduler's market refresh scores
  both sets of charges against the real opponents of that round (handyman revenue, rules of
  section 3) and writes `results/charge_agent/shadow.json`: per round the fixed rule's revenue,
  the agent's and the difference, the current streak and whether the agent is live. The same
  line appears in the scheduler log.
- **Takeover.** A round is a win when the agent's charges would have earned more. A round
  without a difference (nothing covered) neither counts nor breaks the streak. After
  `takeover_rounds` wins in a row the agent's charges are submitted from the next game on.
  The decision is recomputed from the run log and the market summary, so both machines reach
  the same answer. It is not taken back automatically: set `shadow = false` / `mode = "off"`
  by hand, or `takeover = false` to return to shadow.
- **After the takeover** the fixed rule's search for a is skipped (faster), so there is no
  further comparison.

**Takeover is off since 2026-11-22.** Replay of games 56 to 63 through the live workflow (dry
runs, each game priced from earlier games only, the same assessment for both; where only a
lower bound of t is known, t = 1.15 x the bound):

| Game | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | Sum |
|---|---|---|---|---|---|---|---|---|---|
| Fixed rule | 38.7k | 37.9k | 29.3k | 30.2k | 55.8k | 35.2k | 33.6k | 47.5k | 308.1k |
| Agent | 40.3k | 36.1k | 24.9k | 33.6k | 48.8k | 39.1k | 25.7k | 43.2k | 291.7k |
| Difference | +1.6k | -1.7k | -4.4k | +3.4k | -7.0k | +3.9k | -7.8k | -4.3k | -16.3k |

With t at the bound itself the difference is -7.3k. The loss sits on a few lines: uncertain
lines charged far above t (game 62 line 3: 1,040 against t between 621 and 732) and large lines
charged below the fixed rule (game 60 line 1: 1,246 against 1,610, t at least 1,981). Seven of
the eight games were answered by `claude-opus-5-5` with the ranked-reference analyst, while the
agent is calibrated on `gpt-6.2-sol` assessments (limit 4 in section 7); that this is the cause
is not tested. One replay per game; the opponent belief was not refitted after game 57.

**How strong is the criterion?** Weak. In the backtest the agent earned more in 19 of 35
rounds with a difference: its gain is in the size of the wins, not in how often it wins.
Three wins in a row would have first happened at the ninth scored round (game 21), and for a
coin flip they happen within 14 rounds on average. The rule will therefore switch the agent
live sooner or later almost regardless of its quality. A criterion on the summed difference
(for example: ahead in total over the last 10 scored rounds) would test what the backtest
measures.
