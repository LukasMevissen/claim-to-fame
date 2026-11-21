# Pricing strategy review: exploitation vs exploration, robustness of (a, b)

Snapshot: games 1 to 8 from `results/market/game_00*.json` (on disk 2026-11-21 15:35 CET), 12 teams,
27 line items. Our quotes from `runs/submissions.jsonl` (live games 6, 7, 8), our beliefs from
`results/extractions/game{5,6,7,8}__*.json`. Scratch scripts: `/tmp/c2f_review/*.py` on Diego's
machine (not committed). No game API, scheduler or LLM call was used.
Companion doc: `docs/research/bid_model_options.md` (the opponent b model being built in parallel).

**Intent.** Decide whether to move the charge price a to "pure exploitation" of a learned
opponent b model, add deliberate exploration, or do something else. Goal: the most cumulative
net payoff (and so rank) over the ~92 remaining games.
**Constraints.** No deliberate exploration (reasons in section 3). No submission may ever contain
a = 0 on a line again. No strategy change goes live during a game window; changes go through a
PR and tag (`docs/WORKFLOW.md`). No extra paid LLM calls beyond the 4 one-off re-extractions
proposed in C1.
**Check-in.** After game 20 (about 18:15 CET today, 2026-11-21): refit the calibration (C1) on
all items that have our estimate, rerun the replay of the sigma-floor options (section 6), and
check the rank.

## 1. Verdict

- **Exploration: no.** It costs real money, about 1.2% / 2.3% / 6% of handyman receipts if 10% /
  20% / 50% of items are explored. It buys information that is (i) mostly redundant, because the
  other ~10 active teams' charges already bracket each reviewer's b, (ii) worth at most about 1 to
  2% of receipts even if it were perfect, and (iii) public. The other 11 teams learn exactly what
  we pay to learn, which is bad in a rank game.
- **Pure exploitation of a per-team b model: fine to build, but expect almost no money from it.**
  The best-response a hardly depends on the opponents' b. Wrong guesses of every team's b level
  (±0.3 log), of its spread (×0.5 or ×2), or using the old hand-set mix all cost 0.0 to 1.6% of
  expected receipts. Only gross errors matter ("everyone accepts everything": 55 to 63%).
- **The proposal tunes the wrong variable. Our belief about t is where the money is.** It is
  overconfident (the LLM says sigma 0.3 to 0.4, the public data says about 0.8). Both a and b are
  quantiles of that belief, so the overconfidence feeds straight into both. On the three live games
  we played, the hindsight gap is about 12.9k EUR. Almost all of it is t-estimation error, not the
  pricing rule or the opponent model.

## 2. Where we stand (public data, games 1 to 8)

| | net | rank |
|---|---|---|
| 404 (leader) | +34,394 | 1 |
| Lumenauts | +2,062 | 8 |
| CHOOMS | -11,886 | 9 |
| vietgang | -25,992 | 10 |
| **GemischtesHack** | **-27,013** | **11** |
| Asgardians | -30,409 | 12 |

- We were absent in games 1 to 5 (a = b = 0). That cost -26,225, which is 97% of our deficit.
  Games 6, 7, 8 as a live team: -2,866 / +1,593 / +486 (per-game rank 10th, 6th, 7th).
- Per-game team net sd is 2.9k (games 6 to 8) to 4.8k (all games). Over 92 games that is a
  cumulative sd of about 28k to 46k per team. The gap to 8th place (29k) is under one sd, and to 1st
  (61k) about 1.5 to 2 sd. Rank is still open. **Maximise expected payoff now.** Deliberate
  variance-seeking only pays near the end (revisit around game 85 to 90 if a specific rank gap is
  larger than our expected edge but within 1 to 2 sd of the remaining games).
- Game 6 in one line: our median was 350 and the public data proves t ≥ 744. We rejected 9 fair
  charges (paying 1.5× = 6.5k) and charged 310 where 729 was provably safe. The rule was not the
  problem. The estimate was.

## 3. Is exploration informative? Structure and value of information

What one game reveals, whatever we charge (all public):

- **Each reviewer's b**: every known charge from every issuer is a probe. Out of 159 active
  reviewer-item pairs, 57% have b bracketed between two observed charges (median width 0.20 log,
  about ±10%; p90 0.69). 21% have b above every charge and 22% below every charge. Our own charge
  is one more probe among about 8 known charges per item. It only adds information when it lands
  outside the cluster.
- **t**: a wrongful rejection (amount 1.5a) proves t ≥ a, and a 0-amount rejection proves t < a.
  Strict teams (TRINNERS, 404) reject most charges, so these signals arrive without our help.
- **What we cannot learn from others**: how far above the cluster the lenient teams' b goes (the
  21%). Finding out takes charges above t, which strict reviewers turn into 0 instead of 1.5a.
  Overcharging only pays when most reviewers are lenient. If t is known, a = K·t beats a = t only
  when K × (#lenient) > 1.5 × (#rejecting) + (#accepting); with 2 to 3 lenient teams out of 11, K
  must exceed about 5, and the cap c ≥ 4t blocks that. So this information has little value.

The value of information is bounded by the value of a perfect opponent model. With the fitted
population (`results/opponent_belief.json`, `c2f.pricing.charge_price`), the loss from
misspecifying the opponents is:

| model used to pick a (truth = fitted) | regret, sigma 0.3 | regret, sigma 0.6 |
|---|---|---|
| all beta +0.3 / -0.3 | 1.55% / 0.01% | 0.73% / 0.34% |
| all tau ×2 / ×0.5 | 0.00% / 0.01% | 0.40% / 0.16% |
| no absent teams | 0.00% | 0.14% |
| hand-set default mix (`configs/default.toml`) | 0.00% | 0.00% |
| everyone rejects (absent) | 2.05% | 1.82% |
| beta +0.6 | 10.1% | 10.9% |
| everyone accepts everything | 63% | 55% |
| a fixed quantile instead of best response: q0.3 / q0.5 | 0.50% / 8.4% | 5.3% / 0.11% |

Reason: in the fair region a pays between a and 1.5a whatever b is, and above t most reviewers
reject. So a is driven by P(t ≥ a), our own uncertainty, which exploration cannot reduce. The true
t of an item is revealed only after the game, and calibration information arrives publicly anyway.

Cost of exploration, simulated (section 7, worlds A to D): put 10% / 20% / 50% of items at a random
a × LogNormal(0, 0.5). That changes net by -8 / -15 / -38 points of our estimate m per item, i.e.
1.2% / 2.3% / 6% of receipts in worlds A and C, and up to 9% in B and D. At about 1,000 EUR of m
per game, 10% exploration costs roughly 70 to 100 EUR per game, so 6k to 9k over the tournament,
for an upside of at most ~1%.

"Cheap" exploration, item by item:
- *Charging inside the fair region* is not cheap. Going from 0.9t to 0.5t loses ~45% on that
  item. It is not informative either, since low chargers (FDN, TRINNERS) already probe there.
- *Low-stakes items* really are cheaper: cost scales with item value, and game 8 items were 30 to
  100 EUR against 700 to 1,200 in games 5 and 6. The information is still redundant, though.
- *Dithering ±10% around the optimum* costs < 1% (the revenue curve is flat near its peak). It is
  harmless, but adds nothing over the market's own spread.

**Exploitability.** We face 11 counterparties with one a and one b per item. Another team cannot
tailor its single a to us; we carry ~1/11 of its objective. Predictable behaviour is therefore
almost unexploitable, and erratic behaviour gains nothing.

## 4. Weaknesses of the current strategy, ranked by expected money impact

1. **t-estimation error (median and coverage). Not a pricing-rule issue, but it dominates.** The
   hindsight best on games 6 to 8 with public t bounds: receipts ≥ 17.2k vs 9.3k actual, payments
   ≤ 5.2k vs 10.1k actual (11 lines, replay in section 7). In simulation, cutting our own
   idiosyncratic log error from 0.6 to 0.25 is worth about +190 points of m per item. Every pricing
   tweak below is worth ±30 to 80 points. This matches `bid_model_options.md`: "the anchor V is the
   bottleneck".
2. **Overconfident sigma.** The LLM states 0.3 to 0.4 (`[belief] min_sigma = 0.15`). Against the
   public t bounds, 4 of our 7 priced items fall entirely in the outer 6% tails of our belief (PIT
   6-1 [0.97, 1], 8-1 [0, 0.056], 8-2 [0, 0.002], 8-4 [0, 0.003]). Under calibration we would expect
   0.8 items there. A censored MLE of log(t/m) gives sigma = 1.07 (95% CI 0.52 to 1.99), or 0.83
   (0.42 to 1.99) without 8-1 and 8-4, which may be uncovered items. Bias is about 0 but noisy (game 6
   +0.75, game 8 -0.9). As a lower bound: even the median of ~10 teams' charges misses t with sd 0.54
   (0.31 to 1.2) on 13 covered items. In simulation, using sigma 0.75 when the truth is 0.75 is worth
   +30 to +50 points of m per item. Overstating (truth 0.51) costs -23. So **calibrate sigma;
   don't guess it.** On the real games 6 to 8, sigma 0.6 / 0.8 would have cut payments from 10.06k
   to 9.35k / 8.86k. The receipt side is not identified there (t is unbounded above on the big
   items).
3. **a = 0 on lines we flag as uncovered, or where the LLM returns price 0.** This happened on game 7
   item 2 and game 8 items 5 to 7. As handyman our payoff is never negative, so a = 0 is weakly
   dominated by any a > 0. It cost nothing in games 7 and 8, because every reviewer also spotted those
   exclusions (`bid_model_options.md`: miss rate ≈ 0). The real risk is a **false "uncovered" flag
   on a covered item**: a = 0 then forfeits about 11 × 1.5a. Charging the covered price is a free
   option. Also, `covered_prob` from the LLM is always exactly 0 or 1, so P(t = 0) is a hard switch.
4. **The opponent EM contradicts hard evidence** (`c2f/opponents.py`, `_EPS = 0.02`). Fair and fraud
   rejections follow deterministically from the payoff table, yet they are treated as soft and get
   outvoted by the charge likelihood. In the current `results/opponent_belief.json`: item 4-3 has t
   q90 = 103 while t ≥ 129 is proven, item 6-1 has q90 = 587 while t ≥ 744 is proven, and items 7-1
   and 8-3 have p_zero ≈ 1 although a 1.5a rejection proves t > 0. The direct cost through a is
   small (≤ 1 to 2%, see the table above). But it biases beta/tau, and any calibration built on these
   posteriors.
5. **a sits slightly low.** The best response under sigma 0.35 gives a ≈ 0.82 m. Across the 4
   simulated worlds the best a is 0.9 to 1.1 m. 0.82 m gives up 2 to 9%, while a ≈ 0.9 to 1.0 m stays
   within 3 to 4% of the best in every world.
6. **b = 1/3 quantile.** The math is right for a calibrated belief, and the cap c is immaterial:
   data show c ≥ 385 on an item with t < 5.58, so c is not 4t and is large. b is **flat** as a lever.
   Over 18 items with charges, any b between 0.5 m and 1.1 m costs 29.8k to 30.3k (proxy replay).
   That compares with 34.0k for reject-all, 41.7k+ for accept-all, and 22.7k for oracle b = t. The
   7k gap to the oracle is item-level discrimination (t again), not the rule. Items fall into two
   regimes: 7 of 13 covered items had every charge fair, while 1-1, 7-1 and game 8 had most charges
   fraudulent. One caveat to "b needs no opponent model": that holds only if P(t ≥ x) is calibrated
   *given that someone charges x*. Charges correlate with t, so the right object is P(fair | x/m),
   measured directly from public labels (C6).

## 5. Proposed changes (no implementation here; pointers only)

| # | change | where | expected effect |
|---|---|---|---|
| C1 | **Calibrate the t belief from public bounds.** Fit a censored normal of log(t/m) (bias, sigma) from `results/market/summary.json` (t_low/t_high) joined with `results/extractions` (our m). Refit every 5 games. Get 11 more items now by re-extracting cases 1 to 4 offline (4 gpt-4.1-mini calls; zips and keys are public after the game). Interim until the fit: `[belief] min_sigma = 0.6`. | new `scripts/calibrate_belief.py`; `c2f/agent.py::to_belief` (add `median *= exp(bias)`, sigma floor from config) | +30 to 50 pts of m per item in sim if true sigma ≈ 0.75; games 6 to 8 payments -7% to -12% |
| C2 | **Never charge 0.** If p_zero ≥ 1 or median = 0, price a from the covered-price belief `TBelief(median, sigma, 0)` and keep b = 0. The prompt must always return the price as if covered (game 8 items 5 to 7 came back as 0). The last guard is the `[fallback] charge_price`. | `c2f/pricing.py::charge_price` (the `return 0.0` branch); `c2f/extract.py` prompt; `c2f/agent.py::submission_payload` | free option; protects against false "uncovered" flags |
| C3 | **Robustify a without new code.** Narrow the search window to `charge_floor_quantile = 0.35`, `charge_ceiling_quantile = 0.6` (quantiles of t given t > 0, calibrated belief). The best response can then move a within the flat region, but a misfit opponent model cannot drag it out. | `configs/default.toml [pricing]` | caps model risk; in-model cost ≤ 1% |
| C4 | **Hard truth signals in the EM.** `_EPS` → 1e-4 for fair and fraud signals. A 1.5a rejection forbids t = 0. Add a test: no posterior t outside the public [t_low, t_high]. Pass this on to whoever builds the B0 model. | `c2f/opponents.py::_item_posterior`, `_soft_ge`; `tests/` | correctness; small direct money effect |
| C5 | **b stays q1/3 of the calibrated belief**, not driven by the opponent model. Keep `acceptance_quantile` and `acceptance_shrink = 1.0`. | none | avoids adding model risk to a flat lever |
| C6 | **Later (≥ 30 covered items with our m):** replace the b rule with an empirical acceptance rule. Fit a logistic P(charge is fair \| log(x/m)) on public labels (every known charge with a fair/fraud signal is a label, ~30 per game), clustered by item, and accept iff P ≥ 2/3. Today the 70 labels come from only 7 items and are non-monotone (item mix), so it is too early. | new fit in `scripts/update_opponents.py`; `c2f/pricing.py::acceptance_limit` | refinement; measure first |
| C7 | **Evaluate on public replay, not hand-set opponents.** The paying side is exactly identified and the receiving side bounded (intervals of each reviewer's b, public t bounds). The replay reproduces our actual receipts and payments on all 11 lines of games 6 to 8 to the cent. | `c2f/evaluate.py` (replace `EVAL_OPPONENTS`); logic in `/tmp/c2f_review/replay.py` | trustworthy A/B of rules |
| C8 | **Cut estimation error (outside pricing, highest value).** Take the median of 2 to 3 extraction samples, or two models, inside the existing parallel stages. Ask for coverage probabilities that are not 0/1. Clamp p_zero to [0.03, 0.97] in `to_belief`. | `c2f/extract.py`, `c2f/__main__.py` stages, `c2f/agent.py::to_belief` | sim: up to +190 pts of m per item if the idiosyncratic error halves |

Order: C2 and C3 are config or few-line changes; ship them first. C1 interim (min_sigma 0.6) next,
then the fitted version. Do C4 before any calibration that relies on EM posteriors.

## 6. Alternatives, pros and cons

| option | pros | cons | verdict |
|---|---|---|---|
| Team proposal A: pure exploitation of a per-team b model | principled; uses data we have | ≤ 1 to 2% upside; inherits estimation error of t; model risk | build B0 (companion doc), clip its influence (C3), expect little |
| Team proposal B: exploration, random or alternating by item | none measurable | costs 1.2 to 9% of receipts; information redundant and public | reject |
| Targeted probes (lowest-value items, upward probes above the expected market max, ±10% dithering) | cheap (< 1%) | answers a low-value question (how lenient the lenient are) | allowed, not worth the complexity |
| Robust rule: a = median of the calibrated belief (q0.5), b = q1/3, no opponent model | simple, explainable to the jury; within 3 to 4% of best in all simulated worlds; in-model regret 0.1% at sigma 0.6 | leaves 1 to 4% if the model were right | good fallback and baseline for A/B |
| Minimax-regret / distributionally robust a over sigma ∈ [0.4, 1.0] and opponent level | formal robustness | ends up in the same flat 0.9 to 1.1 m region as C3 | C3 is the cheap version of this |
| Market-anchored debiasing (learn our bias vs t per item category) | attacks the dominant error | few items; bias flips between regimes (+0.75 game 6, -0.9 game 8) | do via C1 once ≥ 30 items |
| Rank-aware variance (aggressive a/b when behind late) | right objective near the end | no edge now; rank open | revisit around game 85 |

## 7. Method and limits

- **Replay engine** (`/tmp/c2f_review/market.py`, `replay.py`). Per item it recovers each issuer's a
  (accepted amount, or 1.5a rejection / 1.5), hidden charges, absent teams, each reviewer's b
  interval [largest accepted, smallest rejected), and t bounds from fair and fraud rejections. Our
  payment for a hypothetical b is exact unless a charge's fairness is unknown. Our receipts for a
  hypothetical a are exact when a falls outside the reviewer's b interval and the fairness is known;
  otherwise we report lower and upper bounds. Validation: it reproduces the actual per-line receipts
  and payments for games 6 to 8 exactly. Limit: b is censored and t is often only bounded below
  (market underpricing), so the receipt side for a above the market cluster is unidentified.
- **Proxy replay** (`proxy.py`). We have our own estimate only for games 5 to 8, so we use each
  other team's known charge as a stand-in "estimate m" and evaluate b = k·m against the remaining
  teams, over 18 items. Teams shade their charges, so this measures the shape (flatness), not the
  absolute optimum.
- **Calibration** (`calib.py`). Interval-censored normal MLE on a grid, with a profile-likelihood
  CI. Our sample is 7 items, so the CIs are wide by construction.
- **Simulation** (`sim.py`, `run_sim.py`), 20,000 items per world.
  - t ~ broad lognormal; 25% of items uncovered.
  - Common market log error e ~ N(0, 0.45), shared by all teams including us, plus our own
    idiosyncratic error.
  - Opponents drawn from the fitted team parameters (alpha, sigma ≤ 0.6, beta, tau, miss), absent
    with probability 5%.
  - Our coverage detection: 80% hit, 5% false positive. Cap c = 4t, and ∞ for t = 0.
  - Results are in points of our estimate m per item, which keeps the per-item decision unchanged
    when rescaled.
  - Worlds: A (total error 0.75, opponents' b 0.25 below the fitted level), B (total error 0.51),
    C (fitted opponents as is), D (no common error).
  - Model risk: miss rates in the sim are near the EM prior (0.5 to 0.66), but ≈ 0 in the data. So
    the sim overstates the gain from C2 on truly uncovered items. C2's real value is the
    false-flag protection.
- **Rough money conversion**: ~3.3 items per game × ~300 EUR, so 1 point of m per item ≈ 10 EUR per
  game ≈ 900 EUR over the remaining tournament. Treat it as an order of magnitude only.

## 8. What to measure to decide (after every game, automated)

1. **Replay decomposition** per item: our net vs hindsight-best a and b, split into the receipt gap
   and the payment gap. If the gaps stay dominated by items where our median is off by > 0.4 log,
   effort goes to estimation (C8), not pricing.
2. **Calibration**: PIT intervals of our belief against public [t_low, t_high], and a rolling
   censored MLE of bias and sigma. Target: ≤ 15% of items in the outer 6% tails (now 4 of 7).
3. **Coverage confusion** against public evidence. Uncovered: all positive charges rejected at 0.
   Covered: any 1.5a rejection. Watch the false-"uncovered" rate.
4. **Candidate rules on replay** (sigma floor 0.4 / 0.6 / 0.8; a window from C3; b = q1/3 vs the
   empirical rule from C6). Decide on the identified payment side plus the lower bound of receipts.
5. **Opponent model sanity**: the share of EM posteriors outside public t bounds (target 0), plus the
   forward-chaining Brier check from the companion doc.
6. **Exploration gate**: reconsider exploration only if (4) shows the regret attributable to opponent
   uncertainty exceeding about 2% of receipts. The current in-model bound is ≤ 1.6%.
