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

Case zips are decrypted with `pyzipper` (installed with the package); 7-Zip is only a fallback.
The organisers' `starter_script.py` and `API_HANDBOOK.md` go into `vendor/quantco/`;
case zips into `data/cases/` as `case_000.zip` ... `case_100.zip`.

## Running

    python -m c2f live --case-id 0 --dry-run --allow-untagged   # game 0 = permanent test game, no submit
    python -m c2f live --case-id 0 --allow-untagged             # submit to the test game
    python -m c2f run  --allow-untagged                         # unattended: plays every upcoming game

Every live run appends to `runs/submissions.jsonl` (what was sent, timing, errors) and saves the
model's output to `results/extractions/`. If the model fails, every invoice line still gets the
`[fallback]` prices from `configs/default.toml`.

## Checking our prices: labels and eval

Submitting does not tell us how good our prices were: the API only confirms what it stored and
never returns the secret thresholds t. To measure quality we label cases ourselves and run `eval`.

1. **Price the case once.** A live run on game N, or `python -m c2f eval --game N`, saves the
   model's output as `results/extractions/gameN__<extractor>__<model>.json`.
   Only games that have started can be decrypted; game 0 always can.
2. **Label it.** Copy `data/labels/TEMPLATE.json` to `data/labels/gameN.json` and add one entry
   per invoice line, keyed by its POS number on the invoice (`"1"`, `"2"`, ...):
   - `t`: the fair gross total for the whole line incl. VAT that a claims expert would accept;
     `0` if the line is not covered or not related to the damage.
   - `price_if_covered` (optional): what the line would be worth if it were covered. Only used
     to simulate opponents who miss an exclusion.
   - `source`: who set the values and how. Labels should come from a teammate who did not tune
     the prompt. If QuantCo publishes real thresholds, use those and say so here.
3. **Run eval.**

       python -m c2f eval --game 0 --replay     # reuse the saved model output, no tokens
       python -m c2f eval --game 0 --no-cache   # fresh model call, e.g. after a prompt change

4. **Read the result.** The table (line, P(t = 0), median price, a, b) is printed and saved,
   with the metrics and the version stamp, to `results/eval/gameN/<timestamp>__<version>.json`.

| Metric | Meaning |
|---|---|
| `net_payoff` | our simulated net payoff on this case against the opponent mix in `c2f/evaluate.py` (deliberately different from the mix used for pricing) |
| `naive_baseline` | same simulation with a = b = the model's median price (0 if the line is probably not covered) |
| `oracle` | a = b = t, the best possible result if we knew t |
| `share_of_oracle` | `net_payoff / oracle` |
| `median_abs_log_error` | typical price error of the model; 0.1 is roughly 10% off |
| `zero_items` | lines with t = 0: `tp` caught, `fn` missed, `fp` lines we wrongly priced at 0 |

Problems such as unpriced lines, invalid a or b, a run over the time budget, or line numbers that
do not match the labels are printed as `GATE FAIL` and make the command exit non-zero. Each eval
is compared with the best earlier result for the same case; `--fail-on-regression` fails if the
net payoff drops by more than 2%. Without a label file, eval only checks the gates.
The scores are only as good as the labels.

## Layout

    c2f/            agent package
      payoff.py       exact game payoffs
      belief.py       belief about t (point mass at 0 + lognormal)
      pricing.py      b = 1/3 quantile; a = argmax expected revenue vs opponent mix
      extract.py      one LLM call per case -> line items with coverage/price estimates (cached)
      agent.py        case -> quotes, deadline + fallback model
      evaluate.py     QC gates + tournament simulation vs labelled case
      api.py          QuantCo API client (list games, fetch key, submit)
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
- [x] get Drive folder: case zips, case 0, `starter_script.py`, `API_HANDBOOK.md`
- [x] wire `c2f/api.py` + `[api]` config from the handbook, dry run and test submission on game 0
- [ ] label game 0 (`data/labels/game0.json`) and run the first eval
- [ ] calibrate the opponent mix from the first games' transactions
