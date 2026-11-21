# How we estimate t and turn it into prices

Team GemischtesHack, QuantCo Claim to Fame. Status: 2026-11-21, after game 14.
Code references are relative to the repository root.

## 1. Summary

- A reasoning model (`gpt-6.2-sol`, low effort) reads the policy, the claim and the invoice and
  returns, per invoice line, a probability that the line is indemnified and an estimate of the
  secret threshold t with an uncertainty.
- The prompt also contains a **price book**: every earlier case with the bounds on t that the
  public transactions revealed. This is the main source of price calibration.
- The acceptance limit b is the 1/3 quantile of our belief about t. This is provably optimal
  for the insurer role under the stated payoff rules (section 4).
- The charge a maximises expected revenue against the opponents' acceptance behaviour learned
  from public transactions, with a floor of 0.87 times our estimate (section 5).
- Every change was accepted only if it beat the previous version in a replay of finished games
  against the real opponents (section 7).
- Live, the new version ranked 1st, 1st, 5th, 1st and 3rd of 12 in games 10 to 14. Before it,
  we ranked between 6th and 10th.

## 2. The game in symbols

For one invoice line let

- t ≥ 0 be the secret fair-value threshold (t = 0 if the line is not indemnified),
- c the secret payment cap, with c ≥ 4t and never below an absolute floor,
- a our charge as handyman, b our acceptance limit as insurer.

When an issuer charging a meets a reviewer with limit b, the reviewer pays the issuer

$$
\pi(a, b, t) =
\begin{cases}
a & a \le b,\ a \le t \quad \text{(fair, accepted)}\\
\min(a, c) & a \le b,\ a > t \quad \text{(inflated, accepted)}\\
1.5\,a & a > b,\ a \le t \quad \text{(fair, wrongly rejected)}\\
0 & a > b,\ a > t \quad \text{(inflated, rejected)}
\end{cases}
$$

With K opponents, each with charge $a_k$ and limit $b_k$, our net payoff on the line is

$$
N(a, b) = \sum_{k=1}^{K} \pi(a, b_k, t) \;-\; \sum_{k=1}^{K} \pi(a_k, b, t).
$$

The two sums separate: a only enters the first (revenue), b only the second (cost). So a and
b can be optimised independently. Both depend on t, which we do not know.

## 3. Belief about t

### 3.1 Model

For every line the language model returns three numbers: $p$ (probability that the line is
indemnified), $m$ (best estimate of t if it is) and $s$ (uncertainty as a log standard
deviation). We use the mixture

$$
t = 0 \ \text{with probability } p_0 = 1 - p, \qquad
t \mid t > 0 \;\sim\; \mathrm{LogNormal}(\ln m,\ \sigma^2), \quad \sigma = \min(\max(s,\ 0.2),\ 1.2).
$$

Its survival function and quantile function are closed form (`c2f/belief.py`), with $\Phi$ the
standard normal distribution function:

$$
S(x) = P(t \ge x) = (1 - p_0)\left[1 - \Phi\!\left(\frac{\ln x - \ln m}{\sigma}\right)\right], \quad x > 0,
$$

$$
Q(q) =
\begin{cases}
0 & q \le p_0\\[2pt]
m \exp\!\left(\sigma\, \Phi^{-1}\!\left(\dfrac{q - p_0}{1 - p_0}\right)\right) & q > p_0 .
\end{cases}
$$

### 3.2 What the language model does

One call per case (`c2f/extract.py`, prompt version x3). Input: the full policy text (35 to
75 thousand characters), the damage description, the invoice lines without prices, the photo
if there is one, and the price book. The instructions are policy-first:

1. Identify the insured peril and the cover sections actually in force. Optional covers count
   only if the case says they were taken.
2. Per line, decide whether it is excluded, unrelated, pre-existing, maintenance, betterment or
   an ancillary charge the policy does not carry. If so, t = 0.
3. Otherwise start from an ordinary market price and apply the policy arithmetic: caps and
   sub-limits including VAT (an aggregate sub-limit is shared rateably across lines),
   percentage reductions for breached duties, no betterment, purchase price as ceiling.

Why policy-first: the thresholds turned out to be driven by the wording. In game 4 the policy
has a EUR 500 sub-limit shared rateably across the bicycle lines, and the revealed bounds on
the three lines (below 120, 228 to 253, at least 129) are consistent with thresholds that sum
to 500.

Two models run in parallel on the same prompt. `gpt-4.1` answers first and is submitted as a
safety net; the `gpt-6.2-sol` answer replaces it when it arrives (the API keeps the last
submission). In games 10 to 14 the final answer was submitted 10 to 18 seconds after the start.

### 3.3 The price book: learning t from public transactions

After a game, every transaction (issuer, reviewer, accepted or not, amount paid) is public.
From the payoff table, for an issuer with charge a:

| Observation | Implication | Why |
|---|---|---|
| rejected, amount $x > 0$ | $a = x / 1.5$ and $t \ge a$ | only a fair charge is paid when rejected |
| rejected, amount 0 | $t < a$ | a rejected charge that earns nothing was inflated |
| accepted, amount $x > 0$ | $a = x$ (unless capped at c) | accepted charges are paid in full up to c |
| nobody is paid anything on the line | t = 0 with near certainty | every positive charge was inflated |

Taking the tightest of these over all issuers gives an interval for each line,

$$
\underline{t} = \max\{a_k : a_k \text{ proven fair}\} \;\le\; t \;<\; \overline{t} = \min\{a_k : a_k \text{ proven inflated}\}.
$$

`c2f/market.py` computes these intervals, and `c2f/pricebook.py` writes them into the prompt
next to a short digest of each earlier case. Cases repeat policy editions and repair types
(the laptop damaged on the Berlin to Munich drive appeared in games 2, 5 and 6; the
storm-damaged shed in games 8 and 10), so the model can reuse a revealed price level instead
of guessing. In game 10 its estimate for the combined disposal line was 191, built from the
bounds of the four separate lines of game 8; the transactions later showed t between 157 and 218.

## 4. Acceptance limit b: the 1/3 quantile is optimal

Consider one incoming charge a and our decision to accept or reject it.

- Accepting costs a if $a \le t$ and $\min(a, c) \le a$ otherwise, so at most a.
- Rejecting costs 1.5a if $a \le t$ and 0 otherwise, so $1.5\,a\,S(a)$ in expectation.

Using the upper bound a for the cost of accepting (exact whenever $a \le c$), accepting is
better if and only if

$$
a \;\le\; 1.5\, a\, S(a) \quad\Longleftrightarrow\quad S(a) \ge \tfrac{2}{3} .
$$

S is non-increasing, so the set of charges worth accepting is an interval $[0, b^*]$, and a
single limit implements the optimal decision for every possible incoming charge:

$$
b^* = \sup\{a : P(t \ge a) \ge \tfrac{2}{3}\} = Q(\tfrac{1}{3}).
$$

Two consequences:

- $b^*$ does not depend on what the opponents charge. No opponent model is needed for b.
- If $p_0 > 1/3$ then $b^* = 0$: a line we think is probably uncovered is rejected outright.

For an indemnified line ($p_0 \approx 0$) this is $b^* = m\,e^{-0.43\sigma}$, for example
0.88 m at $\sigma = 0.3$. The result is exact given the belief; its quality in practice
depends on how well calibrated m and $\sigma$ are (section 8). The rule is unit tested in
`tests/test_pricing.py`.

## 5. Charge a: best response to the opponents

Expected revenue from charging a is

$$
R(a) = \mathbb{E}_{t}\!\left[\ \sum_{k=1}^{K} \Big( P(b_k \ge a \mid t)\,\big(a\,\mathbf{1}[a \le t] + \min(a, c)\,\mathbf{1}[a > t]\big)
\;+\; P(b_k < a \mid t)\;1.5\,a\,\mathbf{1}[a \le t] \Big) \right].
$$

The trade-off: below t, a higher charge always earns more (and 1.5 times from every opponent
who rejects it). Above t, we are paid only by opponents whose limit is at least a.

Opponent limits are not observed directly, but each transaction brackets them: an accepted
charge gives $b_k \ge a$, a rejected one $b_k < a$. `c2f/opponents.py` fits, per team, a rule
of the form $\ln b_k = \ln t + \beta_k + \tau_k Z$ (Z standard normal) with interval-censored
data, plus the rate at which the team is absent or misses an exclusion. `c2f/pricing.py` then
evaluates R on a grid of 60 candidates between the 5% and 95% quantiles of $t \mid t > 0$ and
takes the maximiser $a^{\circ}$. The submitted charge is

$$
a = \max\big(a^{\circ},\ 0.87\, m\big).
$$

Two deliberate choices:

- **The floor 0.87 m.** The maximiser alone came out near 0.8 m, and in the first live games
  on the new version our charges were always proven fair with room above. In the 12-game
  replay a floor of 0.87 raised the total by about 16%; 0.90 was marginally better and 0.93
  fell back to the level without a floor, because charges then start to exceed t. We chose
  0.87 to keep a margin from that cliff.
- **a is computed as if the line were indemnified, whatever $p_0$.** If t = 0, a positive
  charge costs nothing when rejected and is paid $\min(a, c)$ when accepted. Charging 0 gives
  that away and loses everything if our coverage call was wrong. This earned about 300 in
  each of games 11 and 13.

## 6. The observer: how runs are assessed and how changes are made

The scheduler plays every game unattended. On top of it runs an observer: an AI coding
agent (Claude, in a supervised session started by a team member) that reviews every finished
game and may change the strategy. It has no access to anything the rules do not allow: it
uses our own case files after the game's start time, our own submission log and the public
leaderboard transactions.

### 6.1 What is assessed after every game

About 80 seconds after a game starts its transactions are public. For each invoice line the
observer then looks at:

1. **Outcome.** Our revenue, our cost, our net, and the rank of our net among the 12 teams,
   computed directly from the transactions (no model involved).
2. **Coverage call.** Did we treat the line as indemnified, and does the revealed interval
   for t agree (section 3.3)?
3. **Price level.** Where our estimate m, our charge a and our limit b sit relative to the
   revealed interval $[\underline{t}, \overline{t})$, and whether our charge was proven fair or
   inflated.
4. **Reference.** What the best team on that line charged.
5. **Operations.** Seconds from game start to the final submission, and any model error.

The new bounds enter the price book automatically, so the next prompt already contains them.
That part is learning without any change to code or parameters.

### 6.2 When a change is made

One game is noise: a single line can decide a game, and the thresholds are only known as
intervals. The observer therefore does not react to a single result. A change needs:

1. **A hypothesis from the reviews**, stated as a mechanism. Example: "our charges were proven
   fair in every game and were twice the highest fair charge, so the charge rule shades too much".
2. **A replay test** (section 7.1). The candidate is re-run on all finished games, with the
   price book restricted to earlier games so it sees only what it would have seen live, and
   its quotes are scored against the real opponents next to the version currently live.
3. **A decision rule.** The candidate must beat the live version under both assumptions about
   t (section 7.1), and the result must not sit on a cliff: neighbouring parameter values have
   to point the same way. Where the best value is next to a drop, a value with a margin is
   taken instead (the charge floor is 0.87, not 0.90, for this reason).
4. **Checks before going live.** Unit tests pass and, for changes to the model call, a dry
   run on test game 0 completes inside the time window.
5. **Deployment between two games.** The scheduler is restarted; games already played are
   skipped. The change and its replay result go into `CHANGELOG.md` and the config comments.

Changes that failed the test were not deployed: medium reasoning effort (24.8k against 26.3k
at low), `gpt-6-luna` as live model (misjudged coverage in game 4), a wider acceptance limit
(+1% in replay, inside the noise).

### 6.3 Changes made so far

| Live from game | Change | Evidence before deployment |
|---|---|---|
| 10 | policy-first prompt, price book, `gpt-6.2-sol` with `gpt-4.1` safety net, always-positive charge | replay games 1 to 8: 1.7k / 17.0k before, 26.3k / 41.2k after |
| 13 | charge floor 0.87 m | replay games 1 to 12: 33.3k / 47.8k before, 39.1k / 54.9k after |
| 15 | Claude models run as shadows (not submitted) | none needed, does not affect submissions |

### 6.4 What limits this procedure

- The replay data is also the data the changes were tuned on. With 12 to 15 games there is
  no separate hold-out set, so a tuned parameter can be overfitted. The margin rule in step 3
  and the live results of the following games are the only protection.
- The observer shares the working directory with the running scheduler and with other
  sessions. Deployment is a restart, so a mistake in a change reaches the next game. The
  safety net is the staged submission: fixed prices first, then the quick model, then the
  full model.

## 7. Evidence

### 7.1 The replay estimator

For a finished game we know each opponent's charge $a_k$ where it was revealed, whether it was
proven fair, an interval $[\ell_k, h_k)$ for each opponent's limit, and the interval for t.
The replay (`c2f/replay.py`) scores candidate quotes (a, b) as

$$
\hat N(a, b) = \sum_k \Big[\hat p_k(a)\, a + \big(1 - \hat p_k(a)\big)\, 1.5\, a\, \mathbf{1}[a \le \hat t]\Big]
\;-\; \sum_k \Big[a_k\, \mathbf{1}[a_k \le b] + 1.5\, a_k\, \mathbf{1}[a_k > b]\, \mathbf{1}[a_k \le \hat t]\Big],
$$

with acceptance probability $\hat p_k(a) = 1$ for $a \le \ell_k$, 0 for $a \ge h_k$ and linear
in between, and a point value $\hat t$ inside the revealed interval (the midpoint when both
bounds are known).

Assumptions and their effect:

- Where only a lower bound $\underline{t}$ is known we report two values,
  $\hat t = 1.15\,\underline{t}$ and $\hat t = 1.4\,\underline{t}$. This is the largest source
  of uncertainty, and results are given as a pair.
- The cap c and charges that were rejected by everyone (and therefore hidden) are not modelled.
- Sanity check: replaying the quotes we actually submitted in games 12 and 13 gives 3,838,
  equal to the 2,700 + 1,138 on the leaderboard.

### 7.2 Model comparison (replay, games 1 to 8, same prompt and price book)

| Model | Net, $\hat t = 1.15\,\underline{t}$ | Net, $\hat t = 1.4\,\underline{t}$ |
|---|---|---|
| gpt-4.1-mini (live until game 9) | 1.7k | 17.0k |
| gpt-4.1 | 20.6k | 35.8k |
| gpt-6-luna, low effort | 28.9k | 28.9k |
| **gpt-6.2-sol, low effort (live since game 10)** | **26.3k** | **41.2k** |
| oracle, a = b = t | 71.3k | |

`gpt-6-luna` scored slightly higher under the first assumption, but it misjudged coverage in
game 4 (probability 0.08 on lines that were indemnified). `gpt-6.2-sol` applied the shared
sub-limit there and reused price-book values explicitly, so we chose it. Medium reasoning
effort was not better than low (24.8k against 26.3k) and takes longer.

### 7.3 Estimation quality of the live model (games 1 to 12, price book of earlier games only)

Define the log error of an estimate m against the revealed interval as 0 inside the interval
and $\ln(m / \text{nearest bound})$ outside it.

| Quantity | Value |
|---|---|
| lines proven indemnified ($\underline{t} > 0$) | 18 |
| of these, classified indemnified ($p \ge 0.5$) | 18 |
| lines on which nobody was paid | 10 |
| of these, classified not indemnified ($p < 0.5$) | 10 |
| estimates inside the revealed interval | 7 of 18 |
| median absolute log error | 0.08 |
| root mean square log error | 0.23 |
| mean log error (bias) | −0.04 |

So coverage was right on every line we can verify, and prices are typically within about 8%
of the revealed interval, with occasional misses of 30% or more. The clamp
$\sigma \ge 0.2$ is in line with the observed 0.23.

### 7.4 Charge floor (replay, games 1 to 12)

| Floor on a / m | Net, 1.15 | Net, 1.4 |
|---|---|---|
| none | 33.3k | 47.8k |
| 0.84 | 36.3k | 51.5k |
| **0.87** | **39.1k** | **54.9k** |
| 0.90 | 39.9k | 56.2k |
| 0.93 | 40.3k | 40.3k |

### 7.5 Live results

| Game | Version | Net | Rank of 12 |
|---|---|---|---|
| 1 to 5 | not played | −26.2k in total | |
| 6 | gpt-4.1-mini, old prompt | −2,866 | 10 |
| 7 | same | +1,593 | 6 |
| 8 | same | +486 | 7 |
| 9 | same | −1,232 | 7 |
| 10 | gpt-6.2-sol, policy-first, price book | +1,943 | 1 |
| 11 | same | +2,775 | 1 |
| 12 | same | +2,700 | 5 |
| 13 | same, floor 0.87 | +1,138 | 1 |
| 14 | same | +572 | 3 |

Cumulative position: 11th at −27.0k after game 8, 10th at −19.1k after game 14. The deficit
comes from the five games we did not play (a = b = 0 pays 1.5 times every fair charge).

## 8. Limitations

- **Small samples.** The replay covers 12 games and the live record of the new version 5.
  The floor of 0.87 is tuned on that sample, next to a cliff at 0.93.
- **Price level without a reference.** When the price book has no similar case, estimates can
  be 30% off. In game 12, line 2, we estimated about 510 where t was between 605 and 680; our
  limit of 429 then wrongly rejected fair charges at 1.5 times their value.
- **Small lines.** In game 14 (all lines under EUR 70) three of four charges were slightly
  above t. On one of them the 0.87 floor is the likely cause. The amounts were small, and the
  floor is re-checked on the replay as games accumulate.
- **Lines with only a lower bound on t.** The replay must assume where t sits, hence the pairs.
- **Belief shape.** A lognormal with one uncertainty number cannot express "either the
  sub-limit applies or it does not". Such cases show up as larger $\sigma$ only.

## 9. Shadow models (comparison only, nothing submitted)

`c2f/shadow.py` lets further models assess every case after the live submission, with the
same prompt and price book. Their estimates and the quotes they would have produced are stored
in `results/shadow/`, and `python -m c2f shadow-report` scores them with the replay of
section 7.1 next to what was actually submitted, on the same games.

Configured: `claude-fable-5-2`, `claude-opus-5-5` and `claude-sonnet-5-5` through the Anthropic
SDK (`c2f/anthropic_extract.py`), each at low effort, running after every game since game 15.
Past games are assessed with `python -m c2f shadow 1 2 3 ...`.

First comparison, games 1 to 14, one run per model and game, price book of earlier games only,
same pricing rules for all (replay net, the two assumptions about t):

| Model | Net, 1.15 | Net, 1.4 | Median / max seconds |
|---|---|---|---|
| claude-fable-5-2, low | 43.6k | 59.4k | 12 / 15 |
| claude-opus-5-5, low | 40.4k | 56.5k | 11 / 15 |
| gpt-6.2-sol, low (live model) | 38.3k | 54.3k | 14 / 20 |
| claude-sonnet-5-5, low | 32.4k | 48.2k | 11 / 14 |

All four answer well inside the window. `claude-fable-5-2` is ahead of the live model by about
5k (14%), but most of that comes from two games (5 and 9), and each model was run only once,
so run-to-run variation is not measured yet. That is not enough to switch. The shadows
keep running on every new game; a model is promoted only if it stays ahead on the same games
as the sample grows.
