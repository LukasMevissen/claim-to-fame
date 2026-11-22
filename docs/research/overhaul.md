# Overhaul of the case analysis: two ways to learn from earlier rounds

Status: 2026-11-22, after game 58. Code: `c2f/analyst.py`, `c2f/decision.py`, `c2f/memory.py`,
harness `scripts/overhaul_backtest.py`, tests `tests/test_overhaul.py`.
Live since this change: `[llm] analyst = "rel"`. Everything else is off.

## 1. Result

Walk-forward replay of games 30 to 55 (22 games with a case digest), live model
claude-opus-5-5@medium, 2 runs per prompt. For game g only games before g are used. Scoring
as in `c2f/replay.py`, cap 4t, an open threshold read at 1.15 x its lower bound.

| Arm | Net | vs current | 90 % interval | Games better / worse |
|---|---|---|---|---|
| current (x3 prompt, newest cases first, PRICE PRIORS) | 69.7k | | | |
| **A1 `rel`: most similar cases first** | **130.1k** | **+60k** | +21k to +109k | 15 / 5 |
| A2 `x5`: rel + rewritten prompt + FULL-COVER LEVELS | 109k | +39k | +18k to +62k | 14 / 5 |
| B1 current + decision table for b | 77.2k | +8k | +1k to +16k | 6 / 2 |
| B2 current + case memory | 90.1k | +20k | +3k to +41k | 5 / 3 |
| A1 + B2 | 147.1k | +77k | +36k to +126k | 15 / 5 |
| A1 + B1 | 134.0k | +64k | +26k to +113k | 15 / 5 |

Per run: current 64.5k and 74.9k, `rel` 135.9k and 123.7k. The gap is several times the
run-to-run spread (about 8k).

Robustness of the reading of open thresholds (1.0 / 1.15 / 1.4 x the lower bound):
`rel` +54k / +60k / +67k (first run). Case memory on top of `rel`: -4k at 1.0 x, +17k at 1.15 x.
So `rel` goes live and the memory does not.

## 2. The two approaches

**A. Learning in context (the model reads the evidence).**
- `rel`: `c2f/pricebook.py::reference` could already rank earlier cases by similarity, but
  the agent never passed the case, so the model saw the newest 14 of 50+. Game 47 was the
  fourth draw of the case of games 16, 25 and 32 and saw none of them.
  Where the gain comes from: similar cases set the price level of the big lines. Game 38,
  air-conditioning unit, t 1,352 to 1,445: estimate 1,900 with the newest cases, 1,600 with
  similar ones. Game 33, laptop, t 1,220 to 1,472: 1,900 against 1,450.
- `x5`: a prompt that states t = unit rate x invoiced quantity x policy arithmetic, plus the
  highest revealed rate per recurring line type. It cuts under-pricing (lines estimated below
  0.75 x the proven-fair bound: 19 % to 16 %) but pushes large new lines above t. Worse than `rel`.

**B. Learning in code (statistics on top of the model's answer).**
- Decision table (`c2f/decision.py`): per class of stated uncertainty, the shares of the
  estimate to charge and to accept that earned most on all earlier games. For b: +8k, small
  but steady. For a: unstable, revenue falls to nothing just above t (one gpt run lost 17k).
- Case memory (`c2f/memory.py`): a line that returns in the same case or under the same
  policy wording takes the earlier revealed amount. Gain depends on how open thresholds are read.
- Nearest-line price model (text embeddings, same unit): mean log miss 0.54 against 0.26 for
  the model. Rejected, no code kept.
- Ensembles: three runs of one model +3k on 27 games; Claude + gpt-6.2-sol 7.7k worse than
  Claude alone on 8 games. Rejected.

## 3. What the diagnostics say

On the stored gpt-6.2-sol answers of games 20 to 46 (3 runs, net 119k):
- true price on every covered line, model's cover decisions: 416k. The money is in the
  price per line, not in cover.
- every estimate x 1.1: 63k. It is not a level problem; an estimate just above t earns nothing.
- charge floor 0.95 or 1.0 instead of 0.87: +9k / +15k on gpt, but on the answers Claude
  actually gave live (games 44 to 55) 1.0 loses 18k. The floor stays at 0.87.

## 4. Time

`rel` changes which cases are in the 24,000 characters, not how many: model call median 17 s
(maximum of the live log so far 43 s of a 54 s budget). The charge search is now cached per
sigma: same quotes, one search per distinct sigma instead of one per line.

## 5. Limits

- 22 games, 2 runs. Games 33 and 38 carry 50k of the 60k; the other 20 games are +10k, 13 better.
- On gpt-6.2-sol@low (3 runs) `rel` is +5k (interval -14k to +25k). That model only stands
  when Claude fails.
- The replay does not model hidden charges or the true cap.
- The running scheduler reads code and config at start. It must be restarted to use this.

## 6. Reproduce

    python scripts/overhaul_backtest.py --cases <decrypted cases> --games 30-55 --runs 2 \
        --model claude-opus-5-5 --effort medium --no-calls \
        --arms "base,rel,x5,base>cal:b,base>mem:case+wording,rel>mem:case+wording"

Answers are stored in `results/overhaul/answers`, so this makes no model calls.
