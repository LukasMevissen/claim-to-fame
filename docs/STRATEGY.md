# Strategy: from a belief about t to (a, b)

Notation per line item: t = secret fair threshold (0 if not covered or unrelated),
c = cap (≥ 4t), a = our charge, b = our acceptance limit. All gross line-item totals.

## 1. Belief about t

One LLM call per case returns, per item, `covered_prob`, `related_prob`, a fair gross unit
price and a log-sd for its uncertainty (`c2f/extract.py`). We turn that into

- P(t = 0) = 1 − covered_prob · related_prob
- t | t > 0 ~ LogNormal(median = quantity · unit price, sigma = clamp(log-sd))

(`c2f/belief.py`, `c2f/agent.py::to_belief`).

## 2. Acceptance limit b (insurer role): closed form

Another team bills us a. Accepting costs a when a ≤ t and min(a, c) ≤ a otherwise.
Rejecting costs 1.5a when a ≤ t and nothing otherwise. Ignoring the cap (conservative):

    accept  ⇔  a ≤ 1.5 a · P(t ≥ a)  ⇔  P(t ≥ a) ≥ 2/3

P(t ≥ a) falls with a, so the optimal limit is **b = 1/3 quantile of our belief about t**.
Consequence: if P(t = 0) > 1/3 we set b = 0 and reject the item outright.
This is exact for any belief, needs no opponent model, and is unit tested
(`tests/test_pricing.py::test_acceptance_rule_is_optimal_without_cap`).

## 3. Charge price a (handyman role): opponent dependent

Charging a ≤ t always pays at least a (1.5a if they wrongly reject), so within the fair region
higher is better up to t. Above t we are only paid if the opponent's b ≥ a (capped at c).
We pick a on a grid between the 5% and 95% quantiles of t | t > 0, maximising expected revenue
against a configurable opponent mix (`configs/default.toml`, `[pricing.opponents]`):

| type | behaviour | why it matters |
|---|---|---|
| absent | b = 0 | every fair charge is paid 1.5×, push a up to t |
| accept_all | b = ∞ | overcharging pays up to c |
| noisy | b = t · exp(mu + s Z), misses exclusions with prob miss_zero | the realistic middle |

**Calibrate this mix** from the public leaderboard (Matchup and Transactions tabs) after the
first games. That is allowed (public data) and is the main lever after the price estimates.

## 3b. Learning the opponents round by round

After every game, `python scripts/update_opponents.py` pulls the public transactions
(leaderboard Transactions tab, `c2f/leaderboard.py`) and refits a belief about every team.

**What the transactions reveal** (`c2f/ab_table.py`, table `results/ab_table.csv` via
`python scripts/extract_ab.py`, columns team | round | line_item | a | b | ... | our_a | our_b):

| row | a of the issuer | b of the reviewer | t |
|---|---|---|---|
| accepted, amount m | m (capped at c if a > t) | b >= a | – |
| rejected, amount 1.5a | m / 1.5 | b < a | t >= a |
| rejected, amount 0 | hidden (a > t) | b < a | t < a |
| no positive payment on the whole item | not identified | – | t = 0 almost surely |

b is never shown. The table reports b = largest known charge the team accepted, the
infimum of the feasible set, plus b_high (smallest known charge it rejected).
A team that charged 0 everywhere and accepted nothing positive did not submit (absent).
our_a / our_b come from `runs/submissions.jsonl` (last record per game).

**Latent model** (`c2f/opponents.py`): per item an apparent value v and covered/uncovered;
per team log a = log v + alpha + sigma eps, log b = log v + beta + tau eta, a miss rate on
exclusions and an absent rate. EM on a log grid per item; b enters interval censored,
hidden charges left censored, fair/fraud rejections pin t absolutely. Older games are
weighted 0.85 per game, so the belief follows teams that change strategy.

**Best response:** `results/opponent_belief.json` replaces the hand-set opponent mix
(`[pricing] opponents_from`), so a is optimised against the learned b-rules of the actual
teams. b stays the 1/3 quantile of our belief about t, which is optimal for any opponent.

**Limits.** For lenient teams most charges they see are below their b, so b is only bounded
below; beta can be biased upwards (synthetic test: +0.1 to +0.2). Uncovered items reveal
almost nothing (all payments 0), so the miss rate stays near its prior.

## 4. Where the money is

1. Price estimates (median and honest uncertainty) drive both a and b.
2. Coverage detection: P(t = 0) > 1/3 flips b to 0, which saves 100% of a fraudulent charge.
3. Opponent mix only shifts a.

## 5. Evidence for the write-up

`python -m c2f eval` reports, against labelled case 0, our simulated net payoff next to an
oracle (a = b = t) and a naive baseline (a = b = best guess), plus estimation error and
uncovered-item detection. On synthetic items with honest uncertainty, our quotes beat the
naive baseline by roughly 30% in expectation (see `tests/test_pricing.py::test_beats_naive_in_expectation_over_belief`).
