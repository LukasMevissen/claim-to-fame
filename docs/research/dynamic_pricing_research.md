# Dynamic and competitive pricing: literature and practice review for Claim to Fame

Written 2026-11-21 for the team (Lukas, Diego, Natalia). Purpose: give the method section of
the write-up real sources, and check our approach (`docs/STRATEGY.md`) against what the
literature says. This page only reviews sources; it changes no code. Data-driven findings on
the actual opponents are in `docs/research/bid_model_options.md`, which this page refers to
where they agree or disagree.

**Intent.** Map our game onto established theory (dynamic pricing with learning, auctions,
learning in games, robust decisions, claims practice) so that (1) the write-up can justify
each design choice with a citable source and (2) we find the cheap improvements the
literature points to.
**Constraints.** No code changes from this page. Every citation below was checked to exist (DOI
or publisher/arXiv page resolved on 2026-11-21). Two toy simulations below are our own numbers,
not literature, and are marked as such.
**Check-in.** Before the write-up is frozen: check items K1 to K7 in the executive summary.
For each one, either we ran the check and cite it, or we drop the claim.

Notation (per line item, gross totals): t = secret fair value (0 if not covered), c = cap
(c ≥ 4t), a = our charge (handyman role), b = our acceptance limit (insurer role). Insurer pays:
a if a ≤ t and a ≤ b; min(a, c) if a > t and a ≤ b; 1.5a if a ≤ t and a > b; 0 if a > t and a > b.

---

## Executive summary: the most actionable takeaways

| # | Takeaway | Status vs our strategy | Main sources |
|---|---|---|---|
| K1 | **b is a critical-fractile (newsvendor) decision, and it is only as good as the calibration of our belief.** The 1/3-quantile rule is the textbook Bayes answer. But b = median · exp(−0.43 σ) moves directly with σ and with P(t = 0). LLM-stated uncertainty is known to be overconfident. So calibrate σ and covered_prob against the t-bounds revealed by public rejections, which are interval-censored labels (Turnbull/EM), before tuning anything else. | Keep rule, **add calibration** | [Qin 2011], [Xiong 2024], [Turnbull 1976] |
| K2 | **Our b rule ignores the information in the charge itself (a "cursed" decision).** "Accept iff P(t ≥ a) ≥ 2/3" is exact only if the charge a tells us nothing about t. In a common-value game the right probability is P(t ≥ a given that a team charged a). Opponents' charges are noisy estimates of t (or of the crowd value), so they carry information. In a toy model the optimal b moved from 0.81× to between 0.38× and 1.0× our median, depending on whether teams over- or undercharge. Cost saving: about 0.1% to 10%. | **Contradicts** "b is optimal for any opponent" (STRATEGY §3b). Gain is small unless the population over- or undercharges systematically. | [Milgrom & Weber 1982], [Eyster & Rabin 2005], [Kagel & Levin 1986] |
| K3 | **The insurer side has full-information feedback, so b-rules can be replayed offline.** Every past charge is public. Whether it was fair (a ≤ t) is revealed whenever any of the 11 reviewers rejected it. So the cost of *any* alternative b-rule on past games can be computed with no exploration and no propensity weights. Use this to tune the acceptance quantile per game, Hedge-style, and as evidence in the write-up. | **New**, cheap, strong evidence | [Freund & Schapire 1997], [Cesa-Bianchi & Lugosi 2006] |
| K4 | **Do not explore on the handyman side.** Each opponent's b is one threshold per item, and 11 issuers probe it every round. Their charges give us free price variation. With this much "covariate diversity", greedy or certainty-equivalent pricing on the posterior is near-optimal. The INFORMS pricing challenge found that too much exploration hurts. | **Supports** current greedy best response | [Bastani 2021], [Keskin & Zeevi 2014], [van de Geer 2019] |
| K5 | **Opponents' b is anchored on their own estimate, which is correlated with ours, not on the true t (affiliated signals).** Our pricing simulation draws opponent b around the sampled t. It should draw b around a shared item anchor, with one anchor error per item. In a toy model the t-anchored model chose a ≈ 0.90× our median, the anchored one ≈ 0.98×. The revenue difference was small (about 1.5%). | **Contradicts** `pricing._opp_limits` (b = t·exp(...)). Agrees with `bid_model_options.md` §3.4. | [Milgrom & Weber 1982], [Capen 1971] |
| K6 | **Best-respond to a fitted, forgetting opponent model, but cap the exploitation.** Our EM with 0.85 decay is in effect discounted fictitious play against a level-k population. The poker and robust-pricing literature says to restrict the best response: shrink toward a safe default in proportion to the evidence. Under model uncertainty, robust prices are lower than nominal ones. | **Mostly supports**. Add shrinkage and safety. | [Fudenberg & Levine 1998], [Camerer 2004], [Johanson 2007], [Bergemann & Schlag 2011] |
| K7 | **Price prediction quality decided past agent competitions, and reliability decides ours.** In TAC, accuracy of price prediction was the distinguishing factor. In the INFORMS challenge no algorithm won in every market, and ignoring competitors hurt more as the number of competitors grew. Our analog: the t-estimate and coverage call dominate (STRATEGY §4). The default a = b = 0 is catastrophic, so never miss a submission. | **Supports** current priorities | [Wellman 2004], [van de Geer 2019] |
| K8 | **The algorithmic-collusion literature does not apply to our strategy.** The game is constant-sum: everything one team receives, another team pays. Joint "high prices" cannot raise total payoff, and collusion is banned anyway. Cite it only to explain why we did not pursue it. | Neutral | [Calvano 2020], [Hansen 2021] |

What contradicts or changes our current strategy: K2 (the claim that b needs no opponent model),
K5 (b anchored on t in the pricing simulation) and K3 (b is not tuned on data at all today).
K1 is an addition, not a contradiction, but it is probably the largest lever of the eight.

---

## 0. The game in literature terms

| Our object | Closest literature object | Key difference in our game |
|---|---|---|
| Choosing a against 11 unknown thresholds b_j | Posted-price selling with binary (buy / no-buy) feedback; the buyer's "valuation" is b_j | Payoff jumps at t: below t a rejection pays 1.5a, above t it pays 0. The demand curve is learned from *other* sellers' prices too. |
| Choosing b | Newsvendor / critical fractile; reserve price in a procurement auction | Asymmetric loss: wrongful rejection costs 0.5a extra, a wrong acceptance costs up to a (capped at c). |
| t, unknown to all, estimated by all | Common-value auction; winner's curse | Everyone reads the same invoice, often with similar LLMs, so estimation errors are correlated across teams. |
| ~100 rounds, all transactions public | Repeated game with public monitoring; learning in games | b is never shown directly, only accept/reject (interval-censored). |
| Ranking by cumulative net payoff | Constant-sum tournament | Collusion cannot raise the total, only move it between teams. |

---

## 1. Dynamic pricing with demand learning (our a)

### 1.1 Posted prices with binary feedback

**Key idea.** A seller posts a price, and the buyer accepts if their private valuation is at
least the price. The seller only observes accept or reject (censored feedback) and must learn
the demand curve while selling. Kleinberg and Leighton give regret bounds for three valuation
models: fixed unknown valuation (O(log log T)), i.i.d. valuations (O(√T) for smooth demand,
O(T^{2/3}) in general) and adversarial valuations (O(T^{2/3})). Parametric models with maximum
likelihood estimation achieve √T regret [Broder & Rusmevichientong 2012]. Contextual versions
learn a valuation that is linear in item features from binary outcomes [Cohen et al. 2020;
Javanmard & Nazerzadeh 2019].

**For our game.** P(b_j ≥ a) is the "demand curve" we sell into. Three differences help us:
(i) we see everyone's accept/reject outcomes, not only our own; (ii) the item value (crowd
anchor) is a strong context variable, so the problem is contextual pricing on log(a / anchor),
which is exactly the probit curve in `bid_model_options.md` §3.3; (iii) a parametric
(probit/log-normal) model is justified by the √T vs T^{2/3} gap whenever the family fits, but
it is a misspecification risk [Besbes & Zeevi 2009], so keep heavy tails (§3.1 of the
bid-model page found 1% outliers).

Sources: [Kleinberg & Leighton 2003], [Broder & Rusmevichientong 2012], [Cohen et al. 2020].

### 1.2 Learning and earning: how much to explore

**Key idea.** The seller trades off "earning" (charging the currently best price) against
"learning" (charging informative prices). A purely myopic policy (estimate, then optimise) can
get stuck in *incomplete learning* when its own prices do not vary enough. Keskin and Zeevi show
that semi-myopic policies are asymptotically optimal if they keep enough price dispersion.
Bastani et al. show that greedy is rate-optimal when the contexts are diverse enough
("covariate diversity"), and that standard bandit algorithms often over-explore. den Boer's
survey is the standard map of this field.

**For our game.** The variation that drives learning comes for free: 11 other issuers charge
different a on every item, and each reviewer's b is a single threshold that all of them hit. Our
own charges add one probe out of eleven. So deliberate exploration (charging an a that we
believe is suboptimal just to learn) has almost no information value and a real cost. Keep the
greedy best response (STRATEGY §3b), but integrate over parameter uncertainty in the expected
payoff instead of plugging in point estimates. That costs nothing and guards against
overconfident fits (the bid-model page found the EM's τ too small).

Sources: [den Boer 2015a], [Besbes & Zeevi 2009], [Keskin & Zeevi 2014], [Bastani et al. 2021].

### 1.3 Bandits, Thompson sampling, adversarial bandits

**Key idea.** Thompson sampling (TS) draws a model from the posterior and acts optimally for
that draw. It is simple, Bayesian and close to optimal in many pricing settings, including
network revenue management [Ferreira et al. 2018], and it was deployed on a large e-commerce
site [Ganti et al. 2018]. If the environment may be adversarial, EXP3-type algorithms guarantee
low regret against the best fixed price without any stochastic assumptions [Auer et al. 2002].

**For our game.** We do not need TS for exploration (see 1.2). Its useful part for us is
*posterior averaging*: choose a to maximise expected revenue averaged over draws of the opponent
parameters and of t. We mostly do this already for t, but not for the opponent parameters. An
EXP3 / Hedge layer is useful one level up: run a few candidate pricing rules (current model,
anchored model, a safe "a = 0.9 × median" rule) and shift weight to the one that would have
earned most on past public data. For the a side this is counterfactually computable from the
published b-intervals.

Sources: [Russo et al. 2018], [Ferreira et al. 2018], [Auer et al. 2002].

### 1.4 Non-stationarity and censored estimation

**Key idea.** Markets drift. With a forgetting factor or a sliding window, a policy tracks a
changing demand curve, and the window length trades bias against variance
[den Boer 2015b]. For accept/reject data the natural nonparametric estimator of a distribution
from interval-censored observations is Turnbull's self-consistency (EM) algorithm.

**For our game.** Our 0.85 per-game decay is a forgetting factor of this kind; den Boer's
analysis supports tuning it by forward validation instead of fixing it (the bid-model page
already found absence needs a shorter window). Opponent b and the t-bounds from rejections are
interval-censored data, so EM / Turnbull is the right estimator. It is what `c2f/opponents.py`
does, and the write-up can cite it as such.

Sources: [den Boer 2015b], [Turnbull 1976].

---

## 2. The acceptance limit b as a newsvendor decision, and what it misses

**Key idea.** With asymmetric linear losses, the Bayes-optimal threshold is a quantile of the
predictive distribution: the critical fractile Cu / (Cu + Co) of the newsvendor problem.
Here a wrong acceptance costs a (the full overpayment) and a wrong rejection costs 0.5a (the
penalty). The fractile is 0.5 / 1.5 = 1/3, which is our b = F⁻¹(1/3) rule.

**What it implies.**
1. *Calibration is first-order (K1).* With a log-normal belief, b = median · exp(−0.43 σ). If
   the LLM's σ is too small, which is the usual direction for verbalised LLM confidence
   [Xiong et al. 2024], b is too high and we accept overcharges. Calibrate σ and covered_prob on
   the revealed bounds (a fair rejection shows t ≥ a, a fraud rejection shows t < a) with an
   interval-censored likelihood. Report a reliability plot in the write-up.
2. *Condition on the charge (K2).* The critical-fractile logic is right, but the probability must
   be P(t ≥ a | a was charged), not P(t ≥ a). In common-value auctions, ignoring what other
   players' actions reveal is exactly the winner's curse, formalised as "cursed equilibrium"
   [Eyster & Rabin 2005]. Since b is one threshold for all issuers, the practical fix is to
   choose b by minimising expected cost under a joint model of (t, charges of the population),
   or empirically by replay (K3).

**Toy check (ours, not literature).** Belief t ~ LogNormal(median 100, σ = 0.5); 1/3-quantile
b = 80.6. Each opponent charges a = t · exp(α + s·Z). Expected insurer cost per charge, best b
by grid search vs b = 80.6:

| opponents charge | best b | cost at best b | cost at b = 80.6 | reject all |
|---|---|---|---|---|
| α = 0, s = 0.3 (unbiased, sharp) | 64 | 66.5 | 67.1 | 68.0 |
| α = −0.15, s = 0.3 (undercharge) | 100 | 81.9 | 82.7 | 88.8 |
| α = +0.2, s = 0.3 (overcharge) | 38 | 36.2 | 40.4 | 36.3 |
| α = 0, s = 0.6 (unbiased, noisy) | 73 | 51.8 | 51.8 | 55.7 |
| α = +0.3, s = 0.6 (overcharge, noisy) | 61 | 36.0 | 36.9 | 37.3 |

Reading: the 1/3 rule is a robust default (never far from optimal here), which supports
keeping it. But when the population systematically overcharges, a lower b saves about 10%.
The bid-model page (§3.5) reports that teams undercharged relative to t in games 2 to 6 and
overcharged in 7 to 8, so the right shift changes over the tournament. That is a reason for
K3 (tune on replay) rather than a fixed shift.

Sources: [Qin et al. 2011], [Eyster & Rabin 2005], [Xiong et al. 2024].

---

## 3. Auction and bidding theory

### 3.1 Common value and the winner's curse

**Key idea.** When an object is worth the same to everyone but each bidder only has a noisy
estimate, the winner tends to be the one who overestimated most. Unless bids are shaded for
this selection effect, winning loses money. The effect was first documented for offshore oil
leases [Capen et al. 1971], formalised with affiliated signals [Milgrom & Weber 1982], and
shown experimentally to persist, especially with many bidders [Kagel & Levin 1986;
Thaler 1988].

**For our game.** Two selection effects matter.
- *Insurer side:* the charges we accept are disproportionately those where our b was too
  high, that is, where we overestimated t. Our belief should be calibrated *conditional on
  acceptance*: check whether the items we accepted had lower revealed t than our median
  predicted. This is K1 and K2 applied.
- *Handyman side:* an overcharge (a > t) is paid only by reviewers whose b exceeds a, which
  means reviewers who overestimated t like us. Because errors are correlated across teams
  (same invoice, similar models), "everyone overestimates together" happens more often than
  independent noise around t implies (K5). This raises the value of charging near the crowd
  anchor and lowers the penalty for our own estimation error.

**Toy check for K5 (ours).** Same belief as above; opponents' b = anchor · exp(0.3 Z), with
the anchor either the true t or the shared estimate (100). Best a under t-anchoring: 90.5;
under shared anchoring: 98. Choosing 90.5 when the world is anchor-based loses about 1.5% of
revenue, and the reverse mistake about 0.8%. So K5 is a correctness fix with modest money
attached, mainly useful as an honest modelling statement in the write-up.

Sources: [Capen et al. 1971], [Milgrom & Weber 1982], [Kagel & Levin 1986], [Thaler 1988].

### 3.2 Reserve prices

**Key idea.** The revenue-optimal reserve price in a single-item auction does not depend on
the number of bidders (for regular distributions). It is the price p where p − (1 − F(p)) / f(p)
equals the seller's value [Myerson 1981]. Online, reserves can be learned from censored
outcomes with √T regret [Cesa-Bianchi et al. 2015].

**For our game.** Our b is a reserve price in a procurement setting: a reverse auction where we
are the buyer and reject offers above b. Unlike Myerson's setting, our loss also depends on the
hidden t (the 1.5× penalty), which makes the critical-fractile rule (§2) the better template.
The online-reserve literature supports K3: learn the reserve from replayed censored outcomes.

Sources: [Myerson 1981], [Cesa-Bianchi et al. 2015].

### 3.3 Repeated auctions with learning bidders

**Key idea.** Bidders who do not know their value learn it from outcomes (bandit feedback), and
achieve low regret [Weed et al. 2016; Feng et al. 2018]. In the other direction, an observer
can infer bidders' values from their bid histories by assuming they are no-regret learners
instead of in equilibrium [Nekipelov et al. 2015]. A seller facing a mean-based no-regret
learner can extract far more than against a rational buyer [Braverman et al. 2018]. A
*strategic* buyer, however, can hide its value from a learning seller [Amin et al. 2013].

**For our game.** (i) Treating other teams as learners rather than equilibrium players is
standard and defensible (Nekipelov et al. is the citation for "we infer each team's rule from
its public history"). (ii) Most teams likely adapt b to what they were charged. A team that is
exploited by our high a may lower its b next game. The forgetting factor handles this
reactively; to model it explicitly, regress each team's b-factor on what it was charged
recently. (iii) Others can learn our b just as we learn theirs. Because our b depends only on
our belief about t, not on a team-specific rule, there is little for them to exploit beyond
our estimation errors.

Sources: [Weed et al. 2016], [Nekipelov et al. 2015], [Braverman et al. 2018], [Amin et al. 2013].

---

## 4. Competitive dynamic pricing and algorithmic collusion

**Key idea.** Independent pricing algorithms can end up at supra-competitive prices without
communicating. Q-learners learn punishment strategies [Calvano et al. 2020]. Even bandits that
ignore rivals can do it, through correlated price experiments that bias their demand estimates
[Hansen et al. 2021]. Field evidence comes from German gasoline stations [Assad et al. 2024].
Ignoring competitors in one's own demand model can lead to many different long-run prices
[Cooper et al. 2015].

**For our game.** Our payoffs are constant-sum: every euro one team receives as handyman is
paid by another team as insurer. Joint "high prices" cannot raise the teams' total, and the
fair-play rules forbid coordination anyway. What does transfer: (i) Cooper et al. and Hansen
et al. warn that a demand model which omits what drives competitors gives biased estimates.
For us that means opponent b must be modelled relative to the item anchor that all teams see
(K5), not as independent noise. (ii) The INFORMS challenge (§7) found that ignoring competition
hurts more as the number of competitors grows. With 11 opponents, explicit opponent modelling
is warranted.

Sources: [Calvano et al. 2020], [Hansen et al. 2021], [Assad et al. 2024], [Cooper et al. 2015].

---

## 5. Opponent modelling and robustness

### 5.1 Fictitious play, regret matching, level-k / cognitive hierarchy

**Key idea.** Fictitious play best-responds to the empirical frequency of opponents' past
actions [Brown 1951]. With discounting it tracks changing opponents [Fudenberg & Levine 1998].
Regret matching (play actions in proportion to the regret of not having played them) converges
to correlated equilibrium [Hart & Mas-Colell 2000]. Level-k and cognitive-hierarchy models
describe real players: level-0 plays naively, and level-k best-responds to lower levels. Across
many games the data fit about 1.5 reasoning steps on average [Nagel 1995; Camerer et al. 2004].

**For our game.** Our pipeline (fit each team's rule from public history with 0.85 decay, then
best-respond) is discounted fictitious play over parametric opponent types. That is a citable,
standard design. The level-k lens gives a usable prior. Level 0 charges and accepts at its own
estimate (b ≈ own a: the bid-model page finds this for most teams, consistency 92% to 99%).
Level 1 shades b down, or charges higher, against level 0. Level 2 anticipates that. Expect a
mix dominated by levels 0 and 1. Equilibrium play by everyone is unlikely and not needed.

Sources: [Brown 1951], [Fudenberg & Levine 1998], [Camerer et al. 2004], [Nagel 1995],
[Albrecht & Stone 2018] (survey of opponent-modelling methods).

### 5.2 Safe exploitation and robust pricing

**Key idea.** A best response to a fitted opponent model can be badly exploited if the model
is wrong. Restricted Nash responses mix "exploit the model" with "stay safe" and give most of
the gain at a fraction of the risk [Johanson et al. 2007]. In repeated zero-sum games one can
exploit only with gains already banked and keep a guaranteed value [Ganzfried & Sandholm 2015].
In robust monopoly pricing, uncertainty about demand makes the optimal price *lower* under both
maximin and minimax regret, and minimax regret prescribes a randomised price
[Bergemann & Schlag 2011].

**For our game.** (i) Shrink per-team fits toward the pooled curve in proportion to evidence.
The bid-model page already does this, and the citations justify it. (ii) Make the a-search
robust: optimise against a mixture of the fitted model and a pessimistic variant (for example
all b shifted down by one standard error) instead of the point fit. Our a only has upside below
t, because a rejection there pays 1.5a. Robustness therefore matters mainly for the part of
a above our t-median. (iii) A small randomisation of a around the optimum costs little near a
flat optimum and makes our behaviour harder to fit for teams that model us.

Sources: [Johanson et al. 2007], [Ganzfried & Sandholm 2015], [Bergemann & Schlag 2011].

---

## 6. Insurance claims practice (kept short)

**Key idea.** Insurers call overpayment against the policy "claims leakage": paying for
uncovered, unrelated or overpriced items, through errors or fraud [IRMI glossary]. US insurance
fraud alone is estimated at USD 308.6 billion a year across all lines [Coalition Against
Insurance Fraud 2022]. Two price controls dominate practice.
- *Reference price lists.* Property repair estimates in North America are written in Verisk's
  Xactimate. Its price lists are built from contractor and supplier surveys plus completed
  estimates; for each line item a cluster algorithm "selects a representative price from the
  largest cluster" [Verisk/Xactware pricing methodology]. Repairers price against the same list,
  so both sides anchor on one reference.
- *Percentile rules.* US health plans reimburse out-of-network bills up to "usual, customary and
  reasonable" amounts. New York defines this as the 80th percentile of charges for the same
  service in the same area (FAIR Health database) [NY DFS]. The insurer accepts anything up to
  a high percentile of the peer charge distribution, not up to the median.
- *Automated invoice review.* In German motor claims, specialised providers check repair
  invoices line by line against manufacturer labour times and price data, settle automatically
  when nothing is flagged, and query the repairer otherwise [ControlExpert product page].

**For our game.** (i) Coverage and relatedness checks are where real insurers find leakage.
That matches our finding that P(t = 0) > 1/3 → b = 0 is a large lever (STRATEGY §4.2), and the
bid-model page finds every team already rejects positive charges on uncovered items.
(ii) Xactimate's "largest cluster" rule is a published, defensible way to build a robust anchor
from many noisy prices. Use the mode or median of other teams' public charges per item type as
a crowd reference to calibrate our LLM estimates (the same idea as Galton's ox-weighing median
[Galton 1907]). The bid-model page notes the crowd itself is biased relative to t, so calibrate
against revealed t-bounds as well. (iii) The UCR practice is a real-world analog of our
asymmetric loss. Insurers set the limit high in the peer distribution because wrongful denial
is costly (appeals, regulation). Our 1.5× penalty plays that role. A sentence in the write-up
can connect the two.

Sources: [IRMI], [CAIF 2022], [Verisk 2015], [NY DFS], [ControlExpert].

---

## 7. Lessons from comparable competitions

- **INFORMS Dynamic Pricing Challenge 2017** (pricing algorithms competing in simulated duopoly
  and oligopoly markets). From the authors' conclusions: no algorithm beat the market
  consistently, and "each algorithm meets its Waterloo at some point". A greedy rule that
  followed the cheapest competitor was very hard to beat. Ignoring competition was "increasingly
  harmful" with more competitors. "Too much exploration can hurt performance significantly"
  [van de Geer et al. 2019]. Supports K4 and K6.
- **Trading Agent Competition (TAC Travel, 2000 onwards).** Agents bid in many simultaneous
  auctions. The post-mortems found that how agents predicted prices, especially using
  game-specific information, was a major distinguishing factor, and prediction accuracy was
  linked to final scores [Wellman et al. 2004]. The first competition's overview describes 12
  agents' strategies [Stone & Greenwald 2005]. Lesson: invest in the estimator (t and b), not
  in clever bidding logic on top of a weak one.
- **Axelrod's iterated prisoner's dilemma tournament.** A simple, reactive strategy (tit for tat)
  won, and "nice" strategies did best [Axelrod 1980]. Our game is constant-sum, so niceness
  does not transfer, but the lesson that simple, reactive rules beat over-engineered ones does.
- **Kaggle Rock, Paper, Scissors (2020/21).** The 1st-place solution used a multi-armed bandit
  over a pool of opponent-model agents, including agents designed to beat the strongest
  public strategies [1st-place write-up]. Lesson: keep several opponent models and let
  realised performance choose among them (the Hedge layer in §1.3), instead of committing to
  one fitted model.

Sources: [van de Geer et al. 2019], [Wellman et al. 2004], [Stone & Greenwald 2005],
[Axelrod 1980], [Kaggle RPS].

---

## 8. Suggested evidence for the write-up (cheap, offline)

1. **Calibration (K1):** reliability of P(t ≥ x) from our belief against the revealed fair/fraud
   bounds; coverage-probability calibration on items revealed as t = 0. Cite Turnbull and Xiong.
2. **b replay (K3):** cost of the 1/3 rule vs a few alternative quantiles and the
   conditioned-on-charge rule, replayed on all past charges with known fair/fraud status. Cite
   critical fractile and Hedge.
3. **a replay:** revenue of our submitted a vs the a our model would choose under t-anchored
   and anchor-based opponent b (K5), using the published accept/reject intervals.
4. **No-exploration argument (K4):** count how many distinct charges hit each (team, item) b;
   that number is the "free exploration" we rely on.

---

## References

All links were opened or resolved on 2026-11-21. DOIs resolve via doi.org. "Not fully verified"
means the source exists, but an automated check could not open its content.

1. **[Albrecht & Stone 2018]** S. V. Albrecht, P. Stone. Autonomous agents modelling other agents: A comprehensive survey and open problems. *Artificial Intelligence* 258:66–95. https://arxiv.org/abs/1709.08071
2. **[Amin et al. 2013]** K. Amin, A. Rostamizadeh, U. Syed. Learning prices for repeated auctions with strategic buyers. *NeurIPS 26*. https://proceedings.neurips.cc/paper_files/paper/2013/hash/996a7fa078cc36c46d02f9af3bef918b-Abstract.html
3. **[Assad et al. 2024]** S. Assad, R. Clark, D. Ershov, L. Xu. Algorithmic pricing and competition: Empirical evidence from the German retail gasoline market. *Journal of Political Economy* 132(3):723–771. https://doi.org/10.1086/726906
4. **[Auer et al. 2002]** P. Auer, N. Cesa-Bianchi, Y. Freund, R. E. Schapire. The nonstochastic multiarmed bandit problem. *SIAM J. Computing* 32(1):48–77. https://doi.org/10.1137/S0097539701398375
5. **[Axelrod 1980]** R. Axelrod. Effective choice in the prisoner's dilemma. *Journal of Conflict Resolution* 24(1):3–25. https://doi.org/10.1177/002200278002400101
6. **[Bastani et al. 2021]** H. Bastani, M. Bayati, K. Khosravi. Mostly exploration-free algorithms for contextual bandits. *Management Science* 67(3):1329–1349. https://arxiv.org/abs/1704.09011 (journal: https://doi.org/10.1287/mnsc.2020.3605)
7. **[Bergemann & Schlag 2011]** D. Bergemann, K. Schlag. Robust monopoly pricing. *Journal of Economic Theory* 146(6):2527–2543. https://doi.org/10.1016/j.jet.2011.10.018 (working paper: https://cowles.yale.edu/sites/default/files/2022-08/d1527-r2.pdf)
8. **[Besbes & Zeevi 2009]** O. Besbes, A. Zeevi. Dynamic pricing without knowing the demand function: Risk bounds and near-optimal algorithms. *Operations Research* 57(6):1407–1420. https://doi.org/10.1287/opre.1080.0640
9. **[Braverman et al. 2018]** M. Braverman, J. Mao, J. Schneider, S. M. Weinberg. Selling to a no-regret buyer. *ACM EC 2018*. https://arxiv.org/abs/1711.09176
10. **[Broder & Rusmevichientong 2012]** J. Broder, P. Rusmevichientong. Dynamic pricing under a general parametric choice model. *Operations Research* 60(4):965–980. https://doi.org/10.1287/opre.1120.1057
11. **[Brown 1951]** G. W. Brown. Iterative solution of games by fictitious play. In T. C. Koopmans (ed.), *Activity Analysis of Production and Allocation*, Wiley, pp. 374–376. (Book chapter, no stable URL; standard citation, see Fudenberg & Levine ch. 2.)
12. **[CAIF 2022]** Coalition Against Insurance Fraud. Fraud statistics (USD 308.6 bn/year, 2022 study). https://insurancefraud.org/fraud-stats/
13. **[Calvano et al. 2020]** E. Calvano, G. Calzolari, V. Denicolò, S. Pastorello. Artificial intelligence, algorithmic pricing, and collusion. *American Economic Review* 110(10):3267–97. https://doi.org/10.1257/aer.20190623
14. **[Camerer et al. 2004]** C. F. Camerer, T.-H. Ho, J.-K. Chong. A cognitive hierarchy model of games. *Quarterly Journal of Economics* 119(3):861–898. https://doi.org/10.1162/0033553041502225
15. **[Capen et al. 1971]** E. C. Capen, R. V. Clapp, W. M. Campbell. Competitive bidding in high-risk situations. *Journal of Petroleum Technology* 23(6):641–653. https://doi.org/10.2118/2993-PA
16. **[Cesa-Bianchi & Lugosi 2006]** N. Cesa-Bianchi, G. Lugosi. *Prediction, Learning, and Games*. Cambridge University Press. https://doi.org/10.1017/CBO9780511546921
17. **[Cesa-Bianchi et al. 2015]** N. Cesa-Bianchi, C. Gentile, Y. Mansour. Regret minimization for reserve prices in second-price auctions. *IEEE Trans. Information Theory* 61(1):549–564. https://doi.org/10.1109/TIT.2014.2365772
18. **[Cohen et al. 2020]** M. C. Cohen, I. Lobel, R. Paes Leme. Feature-based dynamic pricing. *Management Science* 66(11):4921–4943. https://doi.org/10.1287/mnsc.2019.3485
19. **[ControlExpert]** ControlExpert GmbH. Products (InvoiceCheck, ExpertiseCheck). Vendor page, not independent evidence. https://www.controlexpert.com/de-de/produkte
20. **[Cooper et al. 2015]** W. L. Cooper, T. Homem-de-Mello, A. J. Kleywegt. Learning and pricing with models that do not explicitly incorporate competition. *Operations Research* 63(1):86–103. https://doi.org/10.1287/opre.2014.1341
21. **[den Boer 2015a]** A. V. den Boer. Dynamic pricing and learning: Historical origins, current research, and new directions. *Surveys in OR and Management Science* 20(1):1–18. https://doi.org/10.1016/j.sorms.2015.03.001 (preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2334429)
22. **[den Boer 2015b]** A. V. den Boer. Tracking the market: Dynamic pricing and learning in a changing environment. *European Journal of Operational Research* 247(3):914–927. https://doi.org/10.1016/j.ejor.2015.06.059 (preprint: https://ris.utwente.nl/ws/files/5090706/memo2027.pdf)
23. **[Eyster & Rabin 2005]** E. Eyster, M. Rabin. Cursed equilibrium. *Econometrica* 73(5):1623–1672. https://doi.org/10.1111/j.1468-0262.2005.00631.x
24. **[Feng et al. 2018]** Z. Feng, C. Podimata, V. Syrgkanis. Learning to bid without knowing your value. *ACM EC 2018*, 505–522. https://doi.org/10.1145/3219166.3219208
25. **[Ferreira et al. 2018]** K. J. Ferreira, D. Simchi-Levi, H. Wang. Online network revenue management using Thompson sampling. *Operations Research* 66(6):1586–1602. https://doi.org/10.1287/opre.2018.1755
26. **[Freund & Schapire 1997]** Y. Freund, R. E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting (Hedge). *J. Computer and System Sciences* 55(1):119–139. https://doi.org/10.1006/jcss.1997.1504
27. **[Fudenberg & Levine 1998]** D. Fudenberg, D. K. Levine. *The Theory of Learning in Games*. MIT Press. https://books.google.com/books/about/The_Theory_of_Learning_in_Games.html?id=G6vTQFluxuEC (the MIT Press page https://mitpress.ublish.com/book/the-theory-of-learning-in-games timed out in our check)
28. **[Galton 1907]** F. Galton. Vox populi. *Nature* 75:450–451. Discussed and re-analysed in K. F. Wallis, Revisiting Francis Galton's forecasting competition, https://arxiv.org/abs/1410.3989
29. **[Ganti et al. 2018]** R. Ganti, M. Sustik, Q. Tran, B. Seaman. Thompson sampling for dynamic pricing. arXiv:1802.03050. https://arxiv.org/abs/1802.03050
30. **[Ganzfried & Sandholm 2015]** S. Ganzfried, T. Sandholm. Safe opponent exploitation. *ACM Trans. Economics and Computation* 3(2):8. https://doi.org/10.1145/2716322
31. **[Hansen et al. 2021]** K. T. Hansen, K. Misra, M. M. Pai. Frontiers: Algorithmic collusion: Supra-competitive prices via independent algorithms. *Marketing Science* 40(1):1–12. https://doi.org/10.1287/mksc.2020.1276
32. **[Hart & Mas-Colell 2000]** S. Hart, A. Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. *Econometrica* 68(5):1127–1150. https://doi.org/10.1111/1468-0262.00153
33. **[IRMI]** International Risk Management Institute. Glossary: claims leakage. https://www.irmi.com/glossary?page=28 (not fully verified: the page blocks automated access; definition taken from search snippets quoting it)
34. **[Javanmard & Nazerzadeh 2019]** A. Javanmard, H. Nazerzadeh. Dynamic pricing in high-dimensions. *JMLR* 20(9):1–49. https://jmlr.org/papers/volume20/17-357/17-357.pdf
35. **[Johanson et al. 2007]** M. Johanson, M. Zinkevich, M. Bowling. Computing robust counter-strategies. *NeurIPS 20*, 721–728. https://papers.nips.cc/paper/3347-computing-robust-counter-strategies
36. **[Kaggle RPS]** Kaggle Rock, Paper, Scissors competition (2020/21), https://www.kaggle.com/competitions/rock-paper-scissors ; 1st-place solution write-up, https://github.com/thisisbowen/RPS-Kaggle-1st-Place-Solution
37. **[Kagel & Levin 1986]** J. H. Kagel, D. Levin. The winner's curse and public information in common value auctions. *American Economic Review* 76(5):894–920. https://econpapers.repec.org/RePEc:aea:aecrev:v:76:y:1986:i:5:p:894-920
38. **[Keskin & Zeevi 2014]** N. B. Keskin, A. Zeevi. Dynamic pricing with an unknown demand model: Asymptotically optimal semi-myopic policies. *Operations Research* 62(5):1142–1167. https://doi.org/10.1287/opre.2014.1294
39. **[Kleinberg & Leighton 2003]** R. Kleinberg, T. Leighton. The value of knowing a demand curve: Bounds on regret for online posted-price auctions. *FOCS 2003*, 594–605. https://doi.org/10.1109/SFCS.2003.1238232
40. **[Milgrom & Weber 1982]** P. R. Milgrom, R. J. Weber. A theory of auctions and competitive bidding. *Econometrica* 50(5):1089–1122. https://doi.org/10.2307/1911865
41. **[Myerson 1981]** R. B. Myerson. Optimal auction design. *Mathematics of Operations Research* 6(1):58–73. https://doi.org/10.1287/moor.6.1.58
42. **[Nagel 1995]** R. Nagel. Unraveling in guessing games: An experimental study. *American Economic Review* 85(5):1313–1326. https://ideas.repec.org/a/aea/aecrev/v85y1995i5p1313-26.html
43. **[Nekipelov et al. 2015]** D. Nekipelov, V. Syrgkanis, É. Tardos. Econometrics for learning agents. *ACM EC 2015*. https://doi.org/10.1145/2764468.2764522
44. **[NY DFS]** New York State Department of Financial Services. Out-of-network law guidance (UCR = 80th percentile of charges, FAIR Health benchmark). https://www.dfs.ny.gov/apps_and_licensing/health_insurers/nyoon_law_guidance_questions_federal_ns_act ; see also FAIR Health, https://www.fairhealth.org/article/fair-health-data-support-state-and-federal-surprise-billing-laws
45. **[Qin et al. 2011]** Y. Qin, R. Wang, A. J. Vakharia, Y. Chen, M. M. H. Seref. The newsvendor problem: Review and directions for future research. *European Journal of Operational Research* 213(2):361–374. https://doi.org/10.1016/j.ejor.2010.11.024
46. **[Russo et al. 2018]** D. Russo, B. Van Roy, A. Kazerouni, I. Osband, Z. Wen. A tutorial on Thompson sampling. *Foundations and Trends in ML* 11(1):1–96. https://arxiv.org/abs/1707.02038
47. **[Stone & Greenwald 2005]** P. Stone, A. Greenwald. The first international trading agent competition: Autonomous bidding agents. *Electronic Commerce Research* 5(2):229–265. https://doi.org/10.1007/s10660-005-6158-z
48. **[Thaler 1988]** R. H. Thaler. Anomalies: The winner's curse. *Journal of Economic Perspectives* 2(1):191–202. https://doi.org/10.1257/jep.2.1.191
49. **[Turnbull 1976]** B. W. Turnbull. The empirical distribution function with arbitrarily grouped, censored and truncated data. *JRSS B* 38(3):290–295. https://doi.org/10.1111/j.2517-6161.1976.tb01597.x
50. **[van de Geer et al. 2019]** R. van de Geer, A. V. den Boer, et al. Dynamic pricing and learning with competition: Insights from the dynamic pricing challenge at the 2017 INFORMS RM & Pricing conference. *J. Revenue and Pricing Management* 18:185–203. https://doi.org/10.1057/s41272-018-00164-4 (preprint: https://arxiv.org/abs/1804.03219)
51. **[Verisk 2015]** Xactware (Verisk). Pricing Research Methodology (2015 edition, hosted by R&R Magazine). https://www.randrmagonline.com/ext/resources/Issues/2017/September/pricing/PricingResearchMethodology.pdf ; product page https://www.verisk.com/products/xactimate/
52. **[Weed et al. 2016]** J. Weed, V. Perchet, P. Rigollet. Online learning in repeated auctions. *COLT 2016*. https://arxiv.org/abs/1511.05720
53. **[Wellman et al. 2004]** M. P. Wellman, D. M. Reeves, K. M. Lochner, Y. Vorobeychik. Price prediction in a trading agent competition. *JAIR* 21:19–36. https://doi.org/10.1613/jair.1333 (open: https://arxiv.org/abs/1107.0034)
54. **[Xiong et al. 2024]** M. Xiong et al. Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. *ICLR 2024*. https://arxiv.org/abs/2306.13063

Toy simulations in §2 and §3.1: 400,000 Monte Carlo draws, t ~ LogNormal(log 100, 0.5),
c = 4t, b and a on a log grid; script kept outside the repo (`/tmp/c2fsim/sim.py` on Diego's
machine), easy to recreate from the description above.
