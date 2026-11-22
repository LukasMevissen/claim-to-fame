# Changelog

Agent versions follow `MAJOR.MINOR.PATCH`:
MINOR for strategy or prompt changes (anything that changes quotes), PATCH for fixes
that leave quotes unchanged, MAJOR if the submission format changes.
Each release is a git tag `vX.Y.Z`, created with `scripts/release.py`.

<!-- new entries below -->

## unreleased: item prior (2026-11-22, changes quotes: `[llm] use_item_prior = true`)

- `c2f/item_prior.py`: distribution of t per unit for every invoice line type that repeats
  across games (hourly rates by grade, vehicle costs, waste disposal, fittings, fees), fitted
  on the censored bounds the public transactions reveal: zero-inflated lognormal, lower bounds
  with a gap model, an outlier component for policy arithmetic, capped games detected and
  left out, level weighted by policy wording.
- The agent appends a PRICE PRIORS block for the recurring lines of the invoice to the
  reference cases (all stages). A/B on games 20-46 with `gpt-6.2-sol@low`, 3 runs per arm:
  replayed net 63.5k -> 90.9k. Not tested on `claude-opus-5-5`.
- `[belief] item_prior = "fuse" | "replace"`: the same prior combined with the model's
  estimate after the assessment. Off; smaller gain (+4k to +10k).
- `scripts/item_prior_backtest.py` (leave-future-out, no model calls),
  `scripts/item_prior_ab.py` (model A/B, nothing submitted), `docs/research/item_prior.md`.

## unreleased (2026-11-21, live from game 10)

- Live from game 26: the `claude-opus-5-5@low` answer is the one submitted (`[llm] prefer_backup`);
  the `gpt-6.2-sol` full stage only stands if Anthropic fails. Replay games 10-19: about 23-30k
  vs 13-17k.
- Live from game 28: Claude stages use `SYSTEM_UPPER` (t = upper end of the customary price
  range instead of an ordinary market price). Backtest games 10-19, 5 runs each: price error
  0.30 -> 0.23, replayed net 22.7k -> 29.3k; game 12 got worse (two large lines overshot).

- Extractor x3: policy-first prompt (optional covers, exclusions, sub-limits shared rateably,
  reductions, no betterment), one whole-line total per item instead of unit price x quantity.
- Price book (`c2f/pricebook.py`): every prompt shows the earlier cases with the t bounds the
  public transactions revealed. Cases repeat policy editions and repair types.
- Full stage on `gpt-6.2-sol` (low reasoning effort, 5-25 s), fast safety-net stage on `gpt-4.1`,
  both always run. Replay of games 1-8 against the real opponents' prices (t inside the revealed
  bounds): old gpt-4.1-mini about +2k, new about +26k to +41k, oracle 71k.
- Charge price is always positive, also on lines we think are uncovered: a rejected charge
  costs nothing, an accepted one pays up to the cap. b stays 0 on those lines.
- `min_sigma` 0.15 -> 0.2.
- From game 13: `charge_min_ratio = 0.87` (a at least 0.87 x our estimate). 12-game replay
  +16%; our charges had always been proven fair with room above.

## v0.1.0 (2026-11-21)

- Initial agent: one LLM extraction per case (coverage, relatedness, fair gross price with uncertainty),
  belief model for t, b = 1/3 quantile of t, a by expected revenue against an opponent mix.
- QC harness against labelled case 0, replayable without LLM calls.
- API client left unconfigured until API_HANDBOOK.md is in vendor/quantco/.
