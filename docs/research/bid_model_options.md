# Predicting the other teams' a and b: options, evidence, recommendation

Snapshot: games 1 to 8 from `results/market/game_00*.json` (as on disk 2026-11-21 15:35 CET),
12 teams, 26 line items, 17 of them covered and priced, 9 uncovered (t = 0).
Scratch scripts: `/tmp/bidmodel/{data,evidence,t0,evalab,anchor}.py` on Diego's machine (not committed).

**Intent.** Give the pricing step, for every opponent and line item, a predictive distribution of
that team's acceptance limit b, so we can best-respond with our charge a. Its precision should be
measured, not assumed.
**Constraints.** Nothing new may run inside the 60 s game window except a table lookup and a Monte
Carlo over a grid (< 0.5 s). Fitting happens offline after each game. No extra paid LLM calls
until option E is tested offline. Do not drop the current EM until the replacement beats it on
the forward-chaining check below.
**Check-in.** After game 28 (about 20:00 CET today, 2026-11-21): rerun the forward-chaining
Brier/log-loss table (section 3.3) on all games. Check whether per-team curves still beat the
pooled curve, and whether the anchor offset (section 3.4) is stable.

## 1. Summary

- **Predicting other teams' a beyond the crowd value is not possible yet.** Each team's charge sits
  about 0.4 log units (±50%) around the cross-team median of the same item. Only ~10% of that
  spread is a stable team effect (ICC 0.08 to 0.11). In leave-one-game-out tests, per-team bias
  terms do not beat "a = crowd median" (MAE 0.345 vs 0.348 to 0.383).
- **b is predictable per team.** Teams set b ≈ their own a times a team factor: pooled b/a factor
  1.00, log-sd 0.36. The factor differs by team (vietgang ~0.5, Kiebler/alpha ~1.5 to 1.9). A
  per-team acceptance curve with shrinkage, P(b_k ≥ a) = Φ((μ_k − log(a/V)) / s_k), beats both the
  pooled curve and the current EM on held-out games (Brier 0.197 vs 0.209 and 0.208; log-loss
  0.606 vs 0.606 and 0.652).
- **The anchor V is the bottleneck, not the team model.** With the true crowd median as anchor the
  curve reaches Brier 0.182. With our LLM estimate as anchor (what we actually have at decision
  time) it worsens to 0.217 on the same observations. That loss is about 3 times the gain from
  per-team modelling. On 7 items our estimate sits +0.20 log above the crowd median (sd 0.35).
- **Uncovered items:** every reviewer rejected every positive charge on all 9 t = 0 items (miss
  rate ≈ 0). The current EM keeps `miss` at its prior of 0.5 to 0.66.
- **Recommendation:** build option B0 now (option (b) without covariates: per-team shrunk censored acceptance curve, about 100
  lines of numpy, fitted offline in under 1 s). Feed it a shared, uncertain anchor. After about 20
  more games, add item covariates (B1) and a learned anchor correction (E).
  Gradient boosting (C) is not viable at this data size.

## 2. Ranked alternatives

Precision is measured on accept/reject outcomes of held-out games (section 3.3). Lower Brier and
log-loss are better. "Data needed" counts covered line items, the unit that limits anything
item-level.

| Rank | Option | Viability now | Precision (evidence) | Effort | Runtime | Data needed |
|---|---|---|---|---|---|---|
| 1 | **B0. Option (b) without covariates: per-team censored acceptance curve (probit/Tobit on log(a/V)), team μ_k, s_k shrunk to a pooled curve, no covariates** | high | best measured: Brier 0.197 LOGO / 0.204 forward, acc 0.71 / 0.68 | ~100 lines numpy, grid MLE | fit < 1 s offline, lookup at game time | works now (17 items); ~3 games per new team |
| 2 | **E. Option (e), better anchor V: calibrate our LLM estimate toward the "typical team" price (offset, later a regression; optionally a second cheap LLM sample)** | medium (offset now; extra LLM call needs a latency test) | largest headroom: Brier 0.217 → 0.182 if V matched the crowd median | offset: 10 lines; LLM variant: prompt + timing work | 0 s (offset) / +3 to 8 s (extra call) | offset: works now, SE ~0.13 on 7 items; regression: ~40 items |
| 3 | **B1. Option (b) in full: hierarchical censored regression with item covariates on μ and s (trade, unit type, log quantity, our price_log_sd, covered_prob)** | low now, medium after ~20 games | untestable now: features exist for only 7 items (games 5 to 8, extraction cache) | B0 + covariate design, ~150 lines | fit ~1 s | ≥ 40 covered items with logged features |
| 4 | **A. Option (a), current EM (`c2f/opponents.py`): log a = log v + α_k + σε, log b = log v + β_k + τη** | high (already built) | worse calibrated: Brier 0.208 / 0.228, log-loss 0.652 / 0.692; α_k carries no signal; `miss` stuck at prior | 0 (exists); fix = drop α, set miss → 0, widen τ | 6 s per fit offline | n/a |
| 5 | **D. Option (d), simple empirical ratios per team ("team X charges 1.3× crowd")** | high | worse than doing nothing on a: MAE 0.374 to 0.383 vs 0.345. On b it can only be an accept rate: Brier 0.241 | 20 lines | 0 | fine, but overfits |
| 6 | **C. Option (c), gradient boosting / quantile forests on features** | not viable | not testable; 17 covered items, team ICC 0.1, so a tree would fit noise | medium | fine | ≥ 150 to 200 items; maybe only for the anchor correction late in the tournament |

The ranking puts B0 first because it is buildable now and measurably better. E has the most upside
but needs an offline test before it costs latency.

## 3. Evidence

### 3.1 How tightly do teams agree on a? (crowd value vs team effects)

Per covered item, the cross-team spread of log a (robust sd, MAD × 1.48):

- Median over items 0.25. Games 5 to 8 average 0.37 (more teams active, more spread). Two
  items have huge raw spreads from single outliers: game 1 item 1 (max/min 22×) and game 5
  item 1 (47×, FDN charged 24 on a ~940 item). About 1% of residuals have |r| > 1.5, so use
  heavy tails.
- Between items, the sd of log crowd median is 1.00. **The item value explains most of the
  variance in log a. The team explains very little.**

Per team, r = log(a / leave-one-team-out item median), n = 140 exact charges:

| team | n | median r | sd |
|---|---|---|---|
| chineseModels | 15 | +0.17 | 0.24 (most consistent team, stable every game) |
| alpha | 15 | 0.00 | 0.47 (early −0.03, late +0.39: drifted up) |
| 404 | 17 | −0.01 | 0.36 |
| Kiebler | 16 | −0.07 | 0.51 |
| Humble Legends | 13 | −0.01 | 0.65 |
| CHOOMS | 9 | 0.00 | 0.24 |
| TRINNERS4WINNERS | 16 | −0.20 | 0.44 |
| FDN | 16 | −0.26 | 0.93 (one −3.7 outlier) |
| Lumenauts | 7 | −0.36 | 0.35 |
| vietgang | 5 | −0.42 | 0.52 |
| us (GemischtesHack) | 6 | −0.22 | 0.36 |

Random-effects decomposition: within-team sd 0.49 (0.39 without outliers), between-team sd
0.14 (0.13). **ICC 0.08 (0.11).** Stability: the correlation of team median r between games 1 to
4 and games 5 to 8 is 0.48 over 8 teams. Direction is somewhat stable, but the effect is small
next to the noise.

### 3.2 Leave-one-game-out prediction of a (relative to the crowd median)

| predictor | LOGO MAE (log) | forward MAE |
|---|---|---|
| P0: a = crowd median (no team effect) | **0.345** | **0.321** |
| P1: + team mean | 0.383 | 0.432 |
| P2: + team median | 0.374 | 0.422 |
| P3: + team mean shrunk (k = 9 pseudo-obs) | 0.349 | 0.323 |
| P3b: Huber + shrunk | 0.348 | 0.325 |

80% intervals with a pooled sd of 0.39 cover 0.79 (LOGO) and 0.82 (forward), so they are
calibrated. **Conclusion: model other teams' a as crowd value × LogNormal(0, ~0.4) with a
heavy tail. Revisit team terms only once a team has 30+ charges.**

### 3.3 Acceptance prediction: P(b_k ≥ a) on held-out games

1,200 accept/reject outcomes on covered items with known charge (base accept rate 0.53).
x = log(a / leave-reviewer-out crowd median). The effective n is smaller than 1,200, because each
(team, item) has a single b seen through ~10 charges, so there are about 150 independent b values.

| model | LOGO log-loss | Brier | acc | forward log-loss | Brier | acc |
|---|---|---|---|---|---|---|
| team accept rate (ignores a) | 0.694 | 0.241 | 0.633 | 0.712 | 0.252 | 0.623 |
| pooled probit (μ, s) | 0.606 | 0.209 | 0.666 | 0.616 | 0.213 | 0.651 |
| **B0: per-team probit, shrunk to pooled** | **0.606** | **0.197** | **0.714** | **0.602** | **0.204** | **0.677** |
| A: current EM (β_k, τ_k) | 0.652 | 0.208 | 0.708 | 0.692 | 0.228 | 0.656 |
| oracle: reviewer's own a known | 0.313 | 0.095 | 0.862 | | | |

- The pooled curve is P(accept) = Φ((0.05 − x) / 0.64). At 0.8×, 1.0× and 1.2× the crowd median,
  about 66%, 53% and 42% of teams accept.
- Fitted per-team median b as a multiple of the crowd median: vietgang 0.45, 404 0.64,
  TRINNERS 0.70, Lumenauts 0.82, Asgardians 0.90, FDN 0.95, us 0.95, Humble 1.11, chineseModels
  1.28, CHOOMS 1.35, alpha 1.82, Kiebler 1.92. The log-sd runs from 0.23 to 0.85.
- **The EM is overconfident:** its τ is too small, so log-loss suffers even where accuracy is
  similar.
- **The oracle row is the key structural fact.** Knowing the reviewer's own a on the item halves
  the Brier score. b ≈ own a (pooled factor 1.00, sd 0.36). Most of b's item-level noise is the
  team's own valuation noise, and we cannot predict that before the game (section 3.2).

Consistency of each team's decisions with b = c·(own a), best c:

| team | best c | consistent |
|---|---|---|
| Humble Legends | 1.00 | 96% |
| chineseModels | 1.00 | 95% |
| us | 1.00 | 94% |
| FDN | 1.10 | 97% |
| Lumenauts | 1.10 | 99% |
| alpha | 1.10 | 93% |
| TRINNERS | 0.90 | 92% |
| CHOOMS | 1.25 | 95% |
| vietgang | 0.50 | 92% |
| 404 | 0.67 | 81% |
| Kiebler | 1.50 | 82% |

### 3.4 Anchor: our LLM estimate vs the crowd median

The 7 covered items with a cached extraction give log(our estimate / crowd median) = +0.39,
−0.27, +0.04, +0.09, +0.80, −0.04, +0.40. That is a mean of +0.20, sd 0.35 and MAE 0.29.

Same 801 accept outcomes (games 5 to 8), curve B0 fitted on other games:

| anchor | log-loss | Brier | acc |
|---|---|---|---|
| crowd median (not available at decision time) | 0.574 | 0.182 | 0.738 |
| our LLM estimate, raw | 0.718 | 0.225 | 0.685 |
| our LLM estimate, s inflated by 0.4 | 0.648 | 0.217 | 0.685 |
| our LLM estimate, half-offset correction + inflation | 0.657 | 0.220 | 0.687 |

- The anchor error is shared by all opponents on an item: when our estimate is too high, every
  team's b looks low at the same time. The pricing step must sample one anchor shift per item,
  not independent noise per team.
- The offset correction does not help yet, with n = 7 and SE ≈ 0.13. Re-estimate it every game.

### 3.5 Uncovered items and absence

- **Uncovered items.** On all 9 t = 0 items the pattern is issuer-driven. FDN and alpha (and,
  from game 7, Asgardians and us) are accepted by everyone, which means they charged 0. Every other
  issuer is rejected by every reviewer, including FDN and alpha, which accept almost anything on
  covered items. So no reviewer missed an exclusion: miss ≈ 0. The EM's miss = 0.5 to 0.66 is
  prior only, because hidden charges give it no interval. For pricing this barely matters, since a
  positive charge on a t = 0 item earns 0 either way.
- **Absence changes over time.** Asgardians, Lumenauts, vietgang and CHOOMS were absent in games
  1 to 4 and are active from games 5 to 7. Estimate the absent probability from the last 2 to 3
  games only, not with the 0.85 decay over all games.
- **Crowd median vs t (side note).** log(t / crowd median) bounds range from −0.83 to +0.52. Teams
  undercharged in games 2 to 6 and overcharged in games 7 to 8. The crowd is not an unbiased
  estimate of t, so keep the t-belief separate from the anchor V.

## 4. What to build now vs after ~20 more games

**Now (option B0, replaces `opponent_types()` as the input to `charge_price`):**

1. Offline after each game, from all games with recency weight (decay 0.85, as today):
   - Pooled probit (μ0, s0) on x = log(a / crowd median, reviewer excluded).
   - Per team (μ_k, s_k) by grid MAP with priors μ_k ~ N(μ0, 0.4²) and log s_k ~ N(log s0, 0.5²).
   - Absent probability from the last 3 games.
   - The anchor offset δ = mean log(crowd median / our estimate) over items where both exist,
     shrunk toward 0 with k = 5.
2. Exclude ourselves. Give a team with < 30 accept outcomes the pooled curve.
3. Keep the EM only for its per-item t posterior (useful for labels and for evaluating our t
   estimates). Stop using its β, τ, miss for pricing once B0 beats it in the forward check.
4. Start logging, per game and item, our extraction features next to the outcome table: trade,
   unit, quantity, covered_prob, price_log_sd, our median. Option B1 needs them, and games 1 to 4
   have none.

**After ~20 more games (≈ 45 to 60 covered items):**

- **B1.** Let μ and log s depend linearly on a few item covariates (unit = flat rate vs
  hours/m²/pcs, log quantity, our price_log_sd, covered_prob < 1), with team random intercepts.
  Accept them only if forward-chaining log-loss improves by more than its game-to-game noise.
- **E.** Regress log(crowd median / our estimate) on the same covariates; this is item-level,
  ~50 rows. Test offline whether a second cheap LLM sample (a different prompt asking for a
  "typical German tradesman invoice price", median of the two) shrinks the 0.35 anchor sd. Only
  then pay for it inside the 60 s budget.
- **Team terms on a.** Reconsider once teams have 30+ charges each (ICC may rise as teams settle).
- **Drift.** Watch for strategy changes (alpha moved +0.4 between halves). Compare the decay 0.85
  fit with a last-5-games fit in the forward check.

**How to measure precision going forward:**

1. Forward-chaining (train on games < g, test on g) log-loss, Brier and a reliability plot of
   P(b_k ≥ a) on observed accept/reject outcomes.
2. Anchor error sd of log(V / crowd median).
3. A decision-level check: for each past item, the revenue of our submitted a against the observed
   accept/reject pattern, next to the revenue the model predicted. Report realized/predicted per
   game.

## 5. Interface for the pricing step

Per game, per line item j, the pricing step should receive:

```python
@dataclass
class BidModel:                      # results/bid_model.json, refitted offline after each game
    anchor_offset: float             # δ: log(crowd / our estimate), shrunk
    anchor_sd: float                 # sd of that log error (now ~0.35), shared across teams
    teams: dict[str, TeamCurve]      # opponents only

@dataclass
class TeamCurve:
    p_absent: float                  # absent next game (b = 0, a = 0)
    mu: float                        # median log(b / V)
    s: float                         # log-sd of b around V for this team (item-level noise)
    n_obs: int                       # accept outcomes behind the fit (for fallback)

def sample_b(model, our_estimate_j, n=2000, rng) -> np.ndarray:   # shape (n, K)
    # one shared anchor draw per sample (correlated across teams), then per-team noise
    logV = log(our_estimate_j) + model.anchor_offset + model.anchor_sd * rng.standard_normal((n, 1))
    logb = logV + mu[None, :] + s[None, :] * rng.standard_normal((n, K))
    b = exp(logb); b[rng.random((n, K)) < p_absent] = 0.0
    return b
```

The pricing step evaluates expected revenue on its existing a-grid as the mean over samples and
teams of `payoff(a, b, t)`. Draw t from our t-belief in the same sample; correlating the t draw
with the anchor draw is optional. On a truly uncovered item every reviewer rejects a positive charge (section 3.5), so any a earns 0
there. The model only matters through P(t > 0), which comes from our t-belief. Runtime is under 50 ms for 2,000 samples × 11 teams × a
100-point a-grid.
