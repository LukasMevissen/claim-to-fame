# Changelog

Agent versions follow `MAJOR.MINOR.PATCH`:
MINOR for strategy or prompt changes (anything that changes quotes), PATCH for fixes
that leave quotes unchanged, MAJOR if the submission format changes.
Each release is a git tag `vX.Y.Z`, created with `scripts/release.py`.

<!-- new entries below -->

## v0.1.0 (2026-11-21)

- Initial agent: one LLM extraction per case (coverage, relatedness, fair gross price with uncertainty),
  belief model for t, b = 1/3 quantile of t, a by expected revenue against an opponent mix.
- QC harness against labelled case 0, replayable without LLM calls.
- API client left unconfigured until API_HANDBOOK.md is in vendor/quantco/.
