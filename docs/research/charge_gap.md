# Why we charge below t instead of a = t

Status: 2026-11-21, games 10 to 24 (live model gpt-6.2-sol). Game 21 is excluded where noted:
no model answer arrived in time and every line went out at the fixed fallback a = 20.

## Short answer

1. **The "Undercutter" label is mostly one failed game.** Over all games the dashboard puts our
   median a/t at 0.63. Without game 21 (19 lines at a = 20, a/t ≈ 0.2) and the games before the
   current model, our median a/t on games 13 to 24 is **0.90, third highest of 12 teams**
   (TRINNERS4WINNERS 1.00, 404 0.95).
2. **Where we do shade, it is the right response to our estimation error, not a pricing bug.**
   a = t only pays when t is known. Our estimate m misses t by a factor of about e^0.47 ≈ 1.6
   (log-sd), and a charge above t earns nothing when it is rejected. Charging the estimate itself
   would have earned **2.6k to 7.1k less** than what we submitted.
3. **The gap to a = t is about +81k to +95k over 14 games, and all of it is information.** Better
   estimates of t are worth that much. A different pricing rule on the same estimates is worth at
   most about +1k.

## Is a = t the equilibrium?

From the payoff table (docs/CASE.md), for a charge a on a line with threshold t:

| | a ≤ t | a > t |
|---|---|---|
| accepted (a ≤ b) | a | min(a, c), c ≥ 4t |
| rejected (a > b) | 1.5a | 0 |

- For a ≤ t both outcomes rise with a. Given t, no a < t beats a = t, whatever the insurer does.
- Above t, an accepted charge still pays min(a, c), so against a lenient insurer (b > t) the
  best response is a = b, not a = t. a = t is the equilibrium only when every insurer sets b = t.
  It is also the best choice against strict insurers, because a rejected fair charge pays 1.5t.
- All of this assumes t is known. We only have a belief: t ~ lognormal(m, σ). Expected revenue
  is roughly (1.25 a) × P(t ≥ a) and falls to 0 past t. Maximising it puts the optimum below m,
  and lower the larger σ is. For σ = 0.4 the optimum is about 0.75 m.

## What our pricing actually does

`c2f/pricing.py` `charge_price`: search a between the 5% and 95% quantiles of the belief, take
the a with the highest expected revenue against the opponent belief, then apply the floor
a ≥ 0.87 m (config `charge_min_ratio`, live since game 13).

| Live lines, games 10–24 without 21 | Value |
|---|---|
| Floor 0.87 m binding (a = 0.87 m) | 76 of 117 lines |
| σ used (model `price_log_sd`, floored at 0.2) | median 0.30–0.35 |
| Actual error of m, lines where t is pinned to an interval | log(m/t) mean +0.16, sd **0.47** |
| m inside the revealed interval [t_low, t_high) | **5 of 23** (13 above, 5 below) |
| Our charge a on those lines | 12 proven fair, 11 above t |

The optimiser alone would shade to about 0.8 m. The floor lifts that to 0.87 m. The model's σ
of 0.3 is overconfident: the real error is 0.47. With the true error, the optimum would be
lower still, not higher.

Where the estimate is low, the shade adds to the error. On large lines where every team was
accepted (only a lower bound on t is known), m/t is about 0.65 and a/t at most about 0.45. Those
lines are where the "undercutting" money is lost. It comes from underestimating big repair
items, not from the pricing rule.

## Counterfactual replay

`python scripts/observer/charge_counterfactual.py`: the same games, our submitted b kept, only
a changed, scored against the real opponents. Two readings of t where only a lower bound is
known (1.15× and 1.4× that bound).

| Charge rule | Net, t = 1.15 × bound | Net, t = 1.4 × bound |
|---|---|---|
| actual (live) | 24.1k | 24.1k |
| a = 0.87 m on every line | 25.3k (+1.2k) | 25.3k (+1.2k) |
| a = m (no shade) | 17.0k (−7.1k) | 21.5k (−2.6k) |
| a = 1.15 m | 8.4k (−15.7k) | 15.6k (−8.5k) |
| a = 0.95 t (oracle) | 95.4k (+71.3k) | 108.4k (+84.3k) |
| a = t (oracle) | 105.2k (+81.1k) | 119.0k (+94.9k) |

Raising a towards or above m loses money every time. The oracle rows show what knowing t is
worth.

## Implications

- **Keep the pricing rule.** Do not raise the floor towards 1.0: the cliff at t is real
  (REPORT 7.4 found the same at 0.93).
- **Work on t, not on a.** Each step that narrows the estimation error moves the optimum
  towards a = t by itself. Candidates already in the repo:
  - price-book coverage of large repair items, where m/t ≈ 0.65;
  - the web price check (`c2f/webcheck.py`);
  - the novelty flag (`c2f/novelty.py`): shade less on lines the market has already priced,
    more on new ones.
- **Calibrate σ.** The model reports σ ≈ 0.3 against a real error of 0.47. A σ fitted on revealed
  intervals per line type would put the shade where the uncertainty actually is.
- **Prevent another game 21.** Fallback prices on every line cost −15.8k in one game. That is
  more than any pricing change above is worth.
