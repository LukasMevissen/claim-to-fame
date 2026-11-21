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

## Quality control against case 0

1. Put the decrypted test case in `data/cases/case0/` (policy.txt, description.txt, invoices.pdf, images.png).
2. Label it: `data/labels/case0.json` with our best expert threshold t per line item
   (gross total, 0 if not covered or unrelated). Copy `data/labels/TEMPLATE.json`.
   If QuantCo publishes the real t for case 0, use those and note it in `source`.
   Labels are made by a teammate who did not tune the prompt, to keep the check honest.
3. Run `python -m c2f eval --case-id case0 --case-dir data/cases/case0` (one LLM call, then cached).
4. It checks the hard gates (all items priced, a and b valid, runtime within budget, ids match labels),
   then simulates the tournament and compares with the best previous result.
5. For pure strategy changes (no prompt change) use `--replay`: no LLM call, runs in CI.

A change is good when: gates pass, `net_payoff` is not worse than the best previous run by more
than 2% (`--fail-on-regression`), and runtime stays well below 50 s.

Caveat: one labelled case is a small sample. Add every case we label (`case1`, `case2`, ...)
and evaluate across all of them before big strategy changes.

## Release

    python scripts/release.py 0.2.0 "what changed"
    git push && git push --tags

## Live run (runner only)

    python -m c2f live --case-id <id> --zip data/cases/<file>.zip --dry-run   # rehearse
    python -m c2f live --case-id <id> --zip data/cases/<file>.zip
