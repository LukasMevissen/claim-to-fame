# claim-to-fame

Team **GemischtesHack** at START Hack League Munich 2026, case **QuantCo: Claim to Fame**:
price every invoice line item as handyman (charge **a**) and as insurer (acceptance limit **b**)
within 60 seconds, against every other team at once.

| Read | What |
|---|---|
| [docs/CASE.md](docs/CASE.md) | full case brief, payoff table, rules, judging, links |
| [docs/STRATEGY.md](docs/STRATEGY.md) | how we turn a belief about t into a and b (with the math) |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | branches, versioning, case 0 quality control, releases, live runs |
| [writeup/WRITEUP.md](writeup/WRITEUP.md) | 1-page hand-in draft |
| [CHANGELOG.md](CHANGELOG.md) | agent versions |

Leaderboard and game schedule: https://c2f.public.quantco.cloud/leaderboard/

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    cp .env.example .env        # add TEAM_API_KEY and OPENAI_KEY, never commit .env
    pytest -q

Also needed: 7-Zip (`brew install p7zip` / `apt install p7zip-full`).
The organisers' `starter_script.py` and `API_HANDBOOK.md` go into `vendor/quantco/`;
case zips into `data/cases/`.

## Layout

    c2f/            agent package
      payoff.py       exact game payoffs
      belief.py       belief about t (point mass at 0 + lognormal)
      pricing.py      b = 1/3 quantile; a = argmax expected revenue vs opponent mix
      extract.py      one LLM call per case -> line items with coverage/price estimates (cached)
      agent.py        case -> quotes, deadline + fallback model
      evaluate.py     QC gates + tournament simulation vs labelled case
      api.py          QuantCo API client (fill in from API_HANDBOOK.md)
      versioning.py   version stamp on every run
    configs/        strategy parameters (TOML)
    data/           cases (git-ignored except labels), labels/
    results/        eval results and cached extractions (committed)
    runs/           submission log (committed)
    scripts/        release.py
    tests/          unit tests

## Status (v0.1.0)

- [x] payoff rules, belief model, optimal b, a vs opponent mix, tests
- [x] case 0 QC harness, versioning, CI
- [ ] get Drive folder: case zips, case 0, `starter_script.py`, `API_HANDBOOK.md`
- [ ] wire `c2f/api.py` + `[api]` config from the handbook, dry run on case 0
- [ ] label case 0 (`data/labels/case0.json`) and run the first eval
- [ ] calibrate the opponent mix from the first games' transactions
