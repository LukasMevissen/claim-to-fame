# Team workflow: versioning and quality control

## Branches and reviews

- `main` is what runs in the tournament. Only released (tagged) versions submit live:
  `python -m c2f live` refuses a dirty or untagged checkout unless `--allow-untagged` is passed.
- Work on a branch per change (`strategy/opponent-mix`, `prompt/vat-handling`, ...), open a PR,
  paste the case 0 eval output into the PR template, merge when CI is green.
- One person (the "runner") submits during games. Later submissions overwrite earlier ones.

## What is versioned

| Thing | Where | Rule |
|---|---|---|
| Agent version | `c2f/__init__.py` + git tag `vX.Y.Z` | bump via `scripts/release.py` |
| Prompt/schema | `EXTRACTOR_VERSION` in `c2f/extract.py` | bump on every prompt change (invalidates the extraction cache) |
| Strategy parameters | `configs/*.toml` | every run records the config's sha |
| Each run | stamp: version, extractor version, git sha, dirty flag, tag, config sha | written into every eval result and submission log |
| Submissions | `runs/submissions.jsonl` | append-only, committed: evidence for the write-up |
| Eval results | `results/eval/<case>/<timestamp>__<version>.json` | committed, so we can compare versions |
| LLM extractions | `results/extractions/<case>__<extractor>__<model>.json` | committed, so CI and teammates replay without tokens |

## Quality control against game 0

Step by step, with the meaning of every metric: README, "Checking our prices: labels and eval".

1. Label it: `data/labels/game0.json` with our best expert threshold t per line item, keyed by
   the invoice POS number (gross total, 0 if not covered or unrelated). Copy `data/labels/TEMPLATE.json`.
   If QuantCo publishes the real t for case 0, use those and note it in `source`.
   Labels are made by a teammate who did not tune the prompt, to keep the check honest.
2. Run `python -m c2f eval --game 0` (decrypts `data/cases/case_000.zip` with the game key;
   one LLM call unless the extraction is already cached, `--no-cache` forces a fresh one).
3. It checks the hard gates (all items priced, a and b valid, runtime within budget, ids match labels),
   then simulates the tournament and compares with the best previous result.
4. For pure strategy changes (no prompt change) use `--replay`: no LLM call, runs in CI.

A change is good when: gates pass, `net_payoff` is not worse than the best previous run by more
than 2% (`--fail-on-regression`), and runtime stays well below 50 s.

Caveat: one labelled case is a small sample. Add every case we label (`case1`, `case2`, ...)
and evaluate across all of them before big strategy changes.

## Release

    python scripts/release.py 0.2.0 "what changed"
    git push && git push --tags

## Live run (runner only)

    python -m c2f live --case-id <id> --dry-run   # rehearse; zip defaults to data/cases/case_<id:03d>.zip
    python -m c2f live --case-id <id>
    python -m c2f run                             # unattended over all upcoming games
