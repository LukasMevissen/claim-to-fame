# Changelog

Agent versions follow `MAJOR.MINOR.PATCH`:
MINOR for strategy or prompt changes (anything that changes quotes), PATCH for fixes
that leave quotes unchanged, MAJOR if the submission format changes.
Each release is a git tag `vX.Y.Z`, created with `scripts/release.py`.

<!-- new entries below -->

## Overhaul of the case analysis (2026-11-22)

- **Live: `[llm] analyst = "rel"`.** The model now sees the earlier cases most similar to the
  current one first instead of the newest 14. Replay of games 30-55 on claude-opus-5-5@medium,
  2 runs: 69.7k -> 130.1k. Details and the rejected variants: `docs/research/overhaul.md`.
- Charge search cached per sigma (`c2f/pricing.py`): same quotes, pricing of a 20-line invoice
  drops from seconds per line to one search per distinct sigma.
- Off by default: analyst `x5`, decision table (`c2f/decision.py`), case memory (`c2f/memory.py`).
- The running scheduler loads code and config at start: restart it to pick this up.

## unreleased: belief behind every quote is recorded and backfilled (2026-11-22, quotes unchanged)

- Problem: from game 26 on the submitted answer usually came from the backup stage
  (claude-opus-5-5), whose estimates were stored nowhere; `results/extractions/` only holds the
  OpenAI stages and the run log only kept the quotes until game 44. So for games 26 to 43 the
  stored estimate was not the one we priced from.
- `c2f/__main__.py`: every stage that priced lines is written to
  `results/live/game<N>__<role>.json` (median, sigma, p_zero, a, b per line, model, whether it was
  submitted). One file per game and machine, so a second machine cannot overwrite or lose it.
  Each stage also carries the charge agent's answer (shadow charges), and a stage that answers
  after the deadline is added to the file as `late` instead of being dropped.
- `c2f/live_estimates.py`, `scripts/backfill_live_estimates.py`: one table for all games in
  `results/live_estimates/` (`lines.csv`, `games.json`). Games 26 to 43 are reconstructed from
  the submitted quotes with the config and opponent belief of that game; on the 100 lines whose
  belief was logged (games 44 to 55) the method returns the right median on 99 and flags the
  other one as ambiguous.

## unreleased: charge agent as shadow run with takeover (2026-11-22, changes quotes after the takeover)

- `[charge_agent] mode = "rule"`, `risk = "neutral"`, `shadow = true`: the agent prices every
  live game; the fixed rule's charges are still submitted and the agent's are recorded
  (`runs/submissions.jsonl`, `timing.charge_agent.shadow_charges`).
- After every published round the market refresh scores both against the real opponents and
  writes `results/charge_agent/shadow.json` (`c2f/charge_agent.py::takeover_status`).
- Takeover: after `takeover_rounds = 3` rounds in a row in which the agent's charges would have
  earned more, they are submitted from the next game on. Not taken back automatically;
  `takeover = false` keeps the shadow, `shadow = false` submits at once.
- Note: in the backtest the agent won 19 of 35 rounds, so three in a row is a weak test
  (`docs/research/charge_agent.md`, section 8).

## unreleased: charge agent (2026-11-22, off by default: quotes unchanged until `[charge_agent] mode` is set)

- `c2f/charge_agent.py`: decides the charge a per line instead of the fixed 0.87 x estimate.
  Belief of the assessment calibrated per class of stated uncertainty on all earlier rounds;
  earlier lines weighted by how often their match class predicted the later threshold;
  a maximises expected revenue (`risk = "neutral"`) or revenue squared (`risk = "loving"`)
  against the measured acceptance of the other teams.
- `mode = "rule"`: no model call, 10-250 ms per invoice. `mode = "llm"`: a judge model names
  the informative earlier lines, in parallel to the assessment; cut off after 15 s, pricing
  waits at most 20 s after the start, rule weights otherwise.
- With the agent on, `price_items` skips the fixed rule's search for a (about 3.6 s per line
  against 22 learned opponent types): dry run of game 0 final answer after 12 s instead of 24 s.
- Walk-forward games 13-47 against the fixed rule incl. the confident-line floor: rule/neutral
  +23.5k (90 % interval -5.9k to +53.3k), rule/loving +15.0k, llm/neutral +17.5k (gpt-4.1) to
  +29.5k (gpt-6.2-sol@low), llm/loving +4.5k to +6.7k.
- `scripts/charge_agent_backtest.py`, `tests/test_charge_agent.py`, `docs/research/charge_agent.md`.


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
