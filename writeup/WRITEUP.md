# Claim to Fame: GemischtesHack (max. 1 page)

**Problem.** For each line item, guess the hidden fair threshold t well enough to charge close to
it without exceeding it, and to accept bills that are fair while rejecting the rest, in 60 s.

**Method.**
1. One LLM call per case reads policy, damage description, invoice and photo and returns, per item,
   the probability it is covered and related, a fair gross price and an honest uncertainty.
2. This becomes a belief about t: point mass at 0 for uncovered items plus a lognormal price.
3. Acceptance limit: accepting is right iff P(t ≥ a) ≥ 2/3, so b = 1/3 quantile of the belief
   (b = 0 if the item is likely uncovered). Exact, opponent independent.
4. Charge price: maximise expected revenue against an opponent mix calibrated from public game data.

**Evidence.** _(fill in: case 0 eval vs oracle and naive baseline, tournament results by version)_

**Reliability.** _(fill in: runtimes, fallback model, submissions per game)_

**Strengths / weaknesses.** _(fill in honestly: e.g. price estimates for niche items, one labelled case)_
