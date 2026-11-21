# Changelog

Agent versions follow `MAJOR.MINOR.PATCH`:
MINOR for strategy or prompt changes (anything that changes quotes), PATCH for fixes
that leave quotes unchanged, MAJOR if the submission format changes.
Each release is a git tag `vX.Y.Z`, created with `scripts/release.py`.

<!-- new entries below -->

## unreleased (2026-11-21, live from game 10)

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

## v0.1.0 (2026-11-21)

- Initial agent: one LLM extraction per case (coverage, relatedness, fair gross price with uncertainty),
  belief model for t, b = 1/3 quantile of t, a by expected revenue against an opponent mix.
- QC harness against labelled case 0, replayable without LLM calls.
- API client left unconfigured until API_HANDBOOK.md is in vendor/quantco/.
