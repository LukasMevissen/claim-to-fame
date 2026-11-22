# Item prior: the distribution of t for recurring invoice lines

Code: `c2f/item_prior.py`. Tests: `tests/test_item_prior.py`, `scripts/item_prior_backtest.py`
(no model calls), `scripts/item_prior_ab.py` (model calls, nothing submitted).
Data: public transactions and case digests of games 1 to 46 (`origin/main`, 2026-11-22).

## 1. Result in short

- Problem: invoice lines repeat across games and their threshold per unit is stable, but the
  assessment model sees the earlier cases only as raw text, cut at 24,000 characters, and
  prices recurring lines too low (estimate below the highest charge proven fair on 43 % of them).
- Approach: per line type, a distribution of t per unit fitted on the censored bounds the
  public transactions reveal (section 3), shown to the model as a PRICE PRIORS block for the
  lines of the current invoice.
- Result: fitted on earlier games only, the prior's median is closer to the revealed t than
  the model's stored estimate on recurring lines (log error 0.24 vs 0.27; 0.15 vs 0.31 without
  the one capped game). With the block in the prompt, `gpt-6.2-sol@low` replays to 90.9k net
  instead of 63.5k on games 20 to 46 (3 runs per arm, every run with the prior above every
  run without). Its price error on lines with a prior row falls from 0.27 to 0.23, on all
  priced lines from 0.35 to 0.29. Coverage decisions do not change.
- The wording of the block decides the sign. A first version said the amounts were "before
  policy arithmetic, apply reductions on top"; the model then took the prior's amount and cut
  it again (deductible allocated to each line, invoiced hours trimmed) and lost 29k on games
  30 to 46 in one run. Revealed thresholds are final amounts for the full invoiced quantity,
  and the block now says so.
- Second model, `gpt-4.1` (fast stage), 2 runs per arm: 22.2k vs 14.4k net, inside the
  run-to-run spread; no gain in price error and more coverage errors. Not conclusive.
- Not tested: `claude-opus-5-5`, the model whose answer is submitted when Anthropic answers.
  The Anthropic key had no credit on 2026-11-22. `[llm] use_item_prior` switches the block.

## 2. What repeats

46 finished games have 398 invoice lines. They fall into 225 line types; 58 types occur at
least twice and hold 231 of the 398 lines. A line type is the description without
qualifiers in brackets, without a "- suffix" and without "model NN", plus three rules:

- `<grade> hours`, or a grade name with unit hrs, is hourly labour of that grade
  (`labour: service technician`). All grades form one family.
- `Vehicle costs - return visit` is its own type, apart from `Vehicle costs`.
- administrative, processing and handling fees are one type.

Every amount is divided by the invoiced quantity, so the model works on t per hour, per
metre, per piece. "1 flat rate" counts as quantity 1.

What the public transactions reveal about one line (`c2f/market.py`):

| Outcome | Lines | What it proves |
|---|---|---|
| nobody was paid | 80 | t = 0 |
| paid, but no charge proven fair ("upper") | 112 | t below the lowest rejected charge; almost always t = 0 with one lenient reviewer |
| lower bound only | 101 | t >= highest charge proven fair |
| bracket | 105 | t_low <= t < t_high |

So t is never observed exactly, and for half of the indemnified lines only from below.

## 3. Model

For one line type, per unit:

    t = 0                      with probability p0
    t = exp(mu + sigma Z)      otherwise, Z standard normal

Likelihood of an earlier line of the type, with F the lognormal cdf:

| Outcome | Likelihood |
|---|---|
| nobody paid | p0 |
| upper (t < hi) | p0 + (1 - p0) kappa F(hi), kappa = 0.05 |
| bracket | (1 - p0) [F(hi) - F(lo)] |
| lower bound only | (1 - p0) E[lam exp(-lam (ln t - ln lo)); t >= lo], lam = 12 |

Four design decisions, each tested in section 5:

1. **Lower bounds are close to t.** A lower bound is the highest fair charge of about a dozen
   teams, several of which price near t. The gap ln t - ln t_low is modelled as exponential
   with mean 1/12 (about 8 %). Without this term a type seen only with lower bounds, which is
   the normal case for labour, would have no upper end.
2. **Policy arithmetic is an outlier, not a price.** Each positive observation comes from the
   main lognormal with probability 0.9 and from a wide component centred 50 % lower with
   probability 0.1. Game 21 paid the service technician 44 to 49 per hour, every later game 81
   or more; the level stays at the later value.
3. **A capped invoice is detected and left out.** If at least three recurring lines of a game
   sit jointly 26 % or more below the level of their types in other games, the amounts of
   that game are not used for the price level (they still count as "indemnified"). This
   finds game 41 (aggregate limit of EUR 500 shared across the bicycle invoice, median
   deviation -46 %) and nothing else.
4. **The level follows the policy wording.** `Vehicle costs` is 74 to 88 under the plain
   dwelling wording (pipe leaks), 37 to 44 under the frost edition, 13 to 16 in the glass
   case. If the type has history under the wording of the current case, lines from other
   wordings enter with weight 0.3.

Priors: flat on mu, except hourly labour, where mu of a grade is normal around the pooled
level of all grades (an unseen grade still gets a wide range); lognormal on sigma with median
0.12; Beta(0.5, 0.5) on p0. The posterior over (p0, mu, sigma) is computed exactly on a grid,
numpy only, about 0.1 s per line type. Reported per line type: the share of earlier cases in
which it was indemnified, and the 10 %, 50 % and 90 % quantiles of t per unit given that it is
indemnified.

## 4. What the model says (all 46 games)

Types seen at least twice with price evidence, t per unit when indemnified, median and 80 %
range in EUR gross:

| Line type | Seen | Indemnified | Unit | t per unit |
|---|---|---|---|---|
| labour: senior installer | 3 | 3 | hr | 99 (87 to 113) |
| labour: service technician | 7 | 6 | hr | 93 (83 to 104) |
| labour: helper | 3 | 3 | hr | 67 (58 to 80) |
| labour: painter | 3 | 1 | hr | 55 (43 to 80) |
| labour: skilled worker | 8 | 4 | hr | 52 (33 to 94) |
| labour: apprentice | 5 | 1 | hr | 38 (30 to 72) |
| vehicle costs (plain dwelling wording) | 33 | 17 | pc | 81 (74 to 88) |
| construction waste disposal | 5 | 5 | pc | 60 (55 to 64) |
| assembly costs | 5 | 3 | unit | 110 (94 to 135) |
| installation (air conditioning) | 7 | 2 | pc | 480 (412 to 560) |
| air conditioning unit | 3 | 2 | pc | 1413 (1209 to 1666) |
| tv set | 3 | 3 | pc | 561 (483 to 653) |
| speaker system | 4 | 3 | pc | 315 (232 to 456) |
| damage assessment and inspection | 2 | 2 | unit | 530 (444 to 649) |
| filling valve | 3 | 3 | pc | 52 (47 to 59) |
| connection hose | 3 | 3 | pc | 27 (24 to 31) |
| rock wool pipe insulation | 4 | 4 | m | 23 (18.9 to 29) |
| profipress elbow 90 copper 15mm | 3 | 3 | pc | 10.5 (9.3 to 12.1) |
| grub screw | 3 | 3 | pc | 1.20 (1.03 to 1.39) |
| two screw pipe clamp 10 mm | 2 | 2 | pc | 4.7 (4.0 to 5.5) |
| binding wire | 3 | 3 | kg | 13.9 (10.2 to 17.6) |
| small parts and consumables | 3 | 3 | pc | 45 (36 to 55) |
| freight shipping | 2 | 2 | pc | 43 (36 to 53) |
| shipping | 9 | 1 | pc | 26 (19.2 to 39) |

Never indemnified so far: administrative or processing fee (11 lines), vehicle costs return
visit (5), miscellaneous items (5), property and liability damage to electronics (4),
auxiliary and cleaning material (3), catering for work crew (2), material costs (2).

## 5. Tests

### 5.1 Leave-future-out backtest, no model calls

`python scripts/item_prior_backtest.py --sweep --replay`. For every game g from 5 to 46 the
prior is fitted on games before g and compared with what game g revealed. The comparison is
the assessment stored for game g at the time (`gpt-6.2-sol`, extractor x3), which already had
the reference cases in its prompt. Price error = log distance from the estimate to the
revealed set [t_low, t_high), 0 if inside; 0.1 is roughly 10 %.

| Lines | n | Prior median | Stored model estimate |
|---|---|---|---|
| priced in the outcome, type priced before, model answer stored | 71 | 0.240 | 0.270 |
| the same without game 41 (capped invoice) | 61 | 0.152 | 0.305 |
| estimate below t_low | 71 | 32 % | 46 % |
| log score per line, all outcomes incl. zeros | 132 | -1.62 | -1.86 |

- Without game 41 the prior halves the price error on recurring lines. In game 41 it is far
  off on all 11 lines, because a cap is a property of the case, not of the line type. The
  model handles that case well. This is the reason the prior is offered as context, and at
  belief level only where the model's own estimate is close to it.
- Whether a line is indemnified at all is the model's job: Brier score 0.21 for the prior
  (base rate 0.25), 0.09 for the stored model answers. The prior's history is only useful
  for types that were never paid.
- Calibration of the 80 % range: t_low is at or below the upper end in 92 % of the lines
  (target 90 %). t_high is at or above the lower end in 64 % of the bracketed lines, 78 %
  without game 41 (target 90 %). The range is too narrow on the low side. The misses are level
  changes between kinds of case (hole drilling 64+ in the glass case, 18 to 19 in the tile
  case), not noise around a level.

Sensitivity, all games (81 lines with the earlier default of borrowing from similar names):

| Variant | Price error | Log score |
|---|---|---|
| default | 0.260 | -1.656 |
| gap rate lam 5 / 20 / 40 | 0.258 / 0.262 / 0.266 | -1.727 / -1.641 / -1.634 |
| sigma prior median 0.06 / 0.2 / 0.3 | 0.260 / 0.261 / 0.264 | -1.666 / -1.668 / -1.694 |
| no outlier component (eps = 0) | 0.276 | -1.785 |
| all policy wordings pooled | 0.257 | -1.703 |
| no capped-game detection | 0.262 | -1.655 |
| no borrowing from similar names (now the default) | 0.244 (75 lines) | -1.620 |

The result does not hinge on a parameter. The outlier component and the policy weighting
improve the log score. Borrowing the history of a similarly named type made it worse
("freight shipping" took the level of "shipping", "moisture measurement of the utility room"
that of a combined measurement and drying line), so it is off.

### 5.2 Replay at belief level, no model calls

Same script. Quotes from the stored model estimates, priced with the live pricing rule and
replayed against the real opponents (`c2f/replay.py`), compared with the same estimates after
`item_prior.combine`: only lines the model believes indemnified, only where its estimate is
within a factor e of the prior.

| Quotes from | Net, 36 games | Net, games 30+ | Games better / worse |
|---|---|---|---|
| stored model estimate | 80.7k | 36.3k | |
| fused with the prior (precision-weighted) | 89.2k | 42.7k | 12 / 8 |
| replaced by the prior | 91.2k | 50.3k | 10 / 10 |

### 5.3 A/B with the model, prior as context

`python scripts/item_prior_ab.py --cases <dir> --games 20-46 --runs 3 --model gpt-6.2-sol
--effort low --arms "ref,ref+prior,ref>fuse,ref>replace"`. Every game is assessed with the
live prompt, photo and reference cases of earlier games, once without and once with the
PRICE PRIORS block (prior fitted on earlier games only). Both answers go through the live
pricing rule and are replayed against the real opponents. 3 runs per arm, 27 games, 287
lines per run. Answers are stored in `results/item_prior/ab/`, the table in
`results/item_prior/ab_gpt-6.2-sol_low.txt`.

| Arm | Net, 27 games | Per run | Games 20 to 29 | Games 30 to 46 |
|---|---|---|---|---|
| reference cases only | 63.5k | 68.5k, 62.6k, 59.3k | 14.2k | 49.3k |
| + PRICE PRIORS block | 90.9k | 99.4k, 97.3k, 76.0k | 20.8k | 70.1k |
| reference only, then fused with the prior | 67.3k | 72.4k, 66.5k, 62.9k | | |
| reference only, then replaced by the prior | 73.3k | 76.2k, 75.4k, 68.4k | | |

| Per run, mean | Reference only | + PRICE PRIORS block |
|---|---|---|
| price error, 78 priced lines with a prior row | 0.274 | 0.228 |
| estimate below t_low on these lines | 43 % | 33 % |
| price error, all 166 priced lines | 0.348 | 0.291 |
| covered-or-not wrong, of 287 lines | 35.0 | 36.7 |

- The block is worth about 27k on these 27 games, with a standard error of about 8k from
  the run-to-run spread (3 runs each). It is better in 13 games, worse in 10, equal in 4; the
  gain comes from a few large ones: game 40 (+10.9k: technician and helper lines at 566 and 394
  in all three runs instead of 328 to 410 and 147 to 184; revealed t >= 593 and >= 387), game 34 (+5.1k), game 46 (+5.3k), game 24 (+4.3k).
  Game 38 is 3.4k worse; none of its lines had a price row, so that is model noise.
- As context the prior is worth more than at belief level (+27k vs +4k to +10k), because
  the model also uses the per-unit rate where its own line total is far from the prior and
  the belief-level guard leaves the line alone.
- Games 30 to 46 were also used to find the wording problem of the first version (section 1),
  so they are not a clean test of the final wording. Games 20 to 29 were never looked at
  for that and show the same direction (+6.6k), on few prior rows.
- First wording of the block, one run, games 30 to 46: 22.6k vs 51.9k for the reference
  arm of the same run; price error on lines with a prior row 0.46 vs 0.31.

Second model, `gpt-4.1` (the fast safety-net stage), same games, 2 runs per arm
(`results/item_prior/ab_gpt-4.1.txt`):

| | Reference only | + PRICE PRIORS block |
|---|---|---|
| net, 27 games | 14.4k (23.7k, 5.0k) | 22.2k (26.1k, 18.4k) |
| price error, priced lines with a prior row | 0.242 | 0.255 |
| estimate below t_low on these lines | 45 % | 34 % |
| covered-or-not wrong, of 287 lines | 29.0 | 34.0 |

Not conclusive for this model. The net is higher with the block, but the two reference
runs alone differ by 19k. The price error does not improve, and the weaker model gets
coverage wrong more often with the block (on lines with a prior row 13.5 -> 19.0 per run): it
leans on the indemnified-or-not history, which the block tells it not to do. This stage is
only submitted first as a safety net and is overwritten by the full stage.

## 6. Limits

- A prior per line type knows nothing about the case. Caps shared across an invoice,
  duplicated lines, upgrades and unrelated work decide many lines, and they are visible only
  in the policy and the claim.
- 58 of 225 line types have repeated so far. A new kind of case gets few rows.
- Per-piece amounts assume the same thing is counted. "Removal and disposal of damaged
  insulation" was 68+ per piece for 2 pcs in game 21 and 44 to 48 for 1 pc in game 46.
- The 80 % range is too narrow on the low side (5.1).
- Levels can move: skilled worker hours were 35 to 58 until game 36 and 90+ in game 46.

## 7. Switches

    [llm]    use_item_prior = true | false          # PRICE PRIORS block in the prompt
    [belief] item_prior = "off" | "fuse" | "replace"   # combination after the assessment

Both read `results/market/summary.json` and `results/casebook/`, which `python -m c2f run`
refreshes between games, so the prior updates itself after every finished game. The block is
built once per case (under 2 s for a 22-line invoice) and shared by the parallel stages.
Use one of the two, not both: the same evidence would count twice.
