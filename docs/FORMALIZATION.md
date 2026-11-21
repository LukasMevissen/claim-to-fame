# Formalization: beliefs, objective, and learning between rounds

This document states the model behind the pricing agent in mathematical form. It covers three
things: the two belief systems (about the fair threshold and about the other teams), the
optimization problem that turns them into a charge price and an acceptance limit, and the update
that runs between rounds. Sections 1 to 6 describe what the code does today. Section 7 lists
where the code and the exact mathematics differ. `docs/STRATEGY.md` is the prose companion.

## 0. Notation

| Symbol | Meaning | Code |
|---|---|---|
| $g = 1, 2, \dots$ | round (game) | `game` |
| $j \in J_g$ | line item of round $g$ | `item` |
| $k, l \in K$ | teams; $0$ denotes us | `team_name` |
| $t_j \ge 0$ | secret fair threshold; $t_j = 0$ if not covered or unrelated | `t` |
| $c_j$ | secret payment cap, $c_j \ge \max(4 t_j, c_{\min})$ | `c` |
| $a_{kj}, b_{kj} \ge 0$ | charge price and acceptance limit of team $k$ | `charge_price`, `acceptance_limit` |
| $(p_0, m, s)$ | our belief about $t_j$: zero mass, median, log-sd | `TBelief(p_zero, median, sigma)` |
| $(\alpha_k, \sigma_k)$ | charge rule of team $k$: log bias, log noise | `TeamBelief.alpha, sigma` |
| $(\beta_k, \tau_k, q_k)$ | acceptance rule of team $k$: log bias, log noise, miss rate | `TeamBelief.beta, tau, miss` |
| $\rho_k$ | probability that team $k$ does not submit | `absent_rate` |
| $\pi_0$ | population share of uncovered items | `pi_uncovered` |
| $\delta$ | forgetting factor per round, $0.85$ | `decay` |
| $\Phi, \varphi, \bar\Phi$ | standard normal cdf, pdf, survival function | |

All amounts are gross line-item totals in EUR.

## 1. The game

Each team $k$ submits $(a_{kj}, b_{kj})$ for every item. For every ordered pair (issuer $i$,
reviewer $r$) the reviewer pays the issuer

$$
\psi(a, b;\, t, c) =
\begin{cases}
a & a \le t,\ a \le b \quad\text{(fair, accepted)}\\
1.5\,a & a \le t,\ a > b \quad\text{(fair, wrongly rejected)}\\
\min(a, c) & a > t,\ a \le b \quad\text{(inflated, accepted)}\\
0 & a > t,\ a > b \quad\text{(inflated, rejected)}
\end{cases}
$$

with $a = a_{ij}$ and $b = b_{rj}$ (`c2f/payoff.py::insurer_pays`). Our net payoff in round $g$ is

$$
U_g = \sum_{j \in J_g} \sum_{l \ne 0}
\Big[\underbrace{\psi(a_{0j}, b_{lj};\, t_j, c_j)}_{\text{received as issuer}}
 - \underbrace{\psi(a_{lj}, b_{0j};\, t_j, c_j)}_{\text{paid as reviewer}}\Big].
$$

Two properties of $U_g$ shape everything below.

1. **Separability.** $U_g$ is a sum over items, over opponents, and over the two roles. The first
   term depends on our decisions only through $a_{0j}$, the second only through $b_{0j}$. So
   maximizing $\mathbb{E}[U_g]$ splits into two independent one-dimensional problems per item,
   and only the marginal behaviour of each opponent matters, not the correlation between opponents.
2. **Zero sum.** Every payment is received by one team and paid by another, so
   $\sum_k U_g^{(k)} = 0$. The tournament ranks teams by $\sum_g U_g$. We maximize the expectation
   of our own sum, which is the risk-neutral proxy for rank.

## 2. Belief system I: the fair threshold

For one item, write $t$ for $t_j$. One LLM call per case returns a coverage probability
$p_{\text{cov}}$, a relatedness probability $p_{\text{rel}}$, a fair gross unit price $u$, a
quantity $n$ and a log-sd $s_{\text{LLM}}$ (`c2f/extract.py`). The belief is the mixture

$$
t =
\begin{cases}
0 & \text{with probability } p_0 = 1 - p_{\text{cov}}\, p_{\text{rel}}\\
m\, e^{sZ},\ Z \sim \mathcal N(0,1) & \text{with probability } 1 - p_0
\end{cases}
\qquad m = n\,u,\quad s = \operatorname{clip}(s_{\text{LLM}},\, 0.15,\, 1.2).
$$

The product $p_{\text{cov}} p_{\text{rel}}$ assumes coverage and relatedness are independent.
The survival function and the quantile function are closed form (`c2f/belief.py`):

$$
S(x) = \mathbb P(t \ge x) = (1 - p_0)\,\bar\Phi\!\Big(\frac{\ln x - \ln m}{s}\Big)\quad (x > 0),
\qquad
Q(q) =
\begin{cases}
0 & q \le p_0\\
m \exp\!\Big(s\,\Phi^{-1}\!\Big(\dfrac{q - p_0}{1 - p_0}\Big)\Big) & q > p_0 .
\end{cases}
$$

This belief is rebuilt from scratch for every case. Nothing learned in earlier rounds enters it
(see 7.1).

## 3. Belief system II: the other teams

### 3.1 Generative model

Each item has a coverage state $z_j \in \lbrace 0, 1\rbrace $ ($1$ = uncovered) and an apparent value
$v_j > 0$, the price the item would have if it were covered:

$$
z_j \sim \operatorname{Bernoulli}(\pi_0), \qquad t_j = (1 - z_j)\, v_j .
$$

Team $k$ is described by $\theta_k = (\alpha_k, \sigma_k, \beta_k, \tau_k, q_k, \rho_k)$. With
probability $\rho_k$ it is absent in a round and has $a = b = 0$ on every item. Otherwise, with
independent standard normal $\varepsilon, \eta$:

$$
\ln a_{kj} = \ln v_j + \alpha_k + \sigma_k\,\varepsilon_{kj},
$$

$$
b_{kj} =
\begin{cases}
v_j \exp(\beta_k + \tau_k\,\eta_{kj}) & z_j = 0, \text{ or } z_j = 1 \text{ and the exclusion is missed (probability } q_k)\\
0 & z_j = 1 \text{ and the exclusion is spotted (probability } 1 - q_k).
\end{cases}
$$

So $\alpha_k > 0$ means team $k$ overcharges on average, and $\beta_k < 0$ means it is a strict
reviewer. The charge rule does not depend on $z_j$: a team that charges for an uncovered item
prices it like a covered one.

### 3.2 What pricing uses

Pricing needs, for each opponent $l$, the probability that our charge $a$ is accepted given the
truth:

$$
G_l(a \mid t) = \mathbb P(b_l \ge a \mid t) = (1 - \rho_l)\,\Phi\!\Big(\frac{\ln t + \beta_l - \ln a}{\tau_l}\Big)
\qquad (t > 0).
$$

`opponents.py::opponent_types` builds the population: one absent component of weight $\rho_l$ and
one noisy component of weight $1 - \rho_l$ per team, all teams weighted equally. A team with less
than 3 effective observations of its limit keeps the default $(\beta, \tau, q) = (0, 0.35, 0.5)$.
Before any round is played, the hand-set mix in `configs/default.toml` is used instead: 15%
absent, 5% accept everything ($b = \infty$), 80% noisy with the same default.

## 4. Optimization

### 4.1 Acceptance limit $b$ (reviewer role)

We choose $b$ to minimize what we expect to pay. For one opponent with charge $A$:

$$
C(b) = \mathbb E\big[\psi(A, b;\, t, c)\big]
= \mathbb E\big[\min(A, c_A)\,\mathbf 1\lbrace A \le b\rbrace \big] + 1.5\,\mathbb E\big[A\,\mathbf 1\lbrace A > b,\ A \le t\rbrace \big],
$$

where $c_A = c$ if $A > t$ and $c_A = \infty$ otherwise. Bounding $\min(A, c) \le A$ (the cap only
makes accepting cheaper) and differentiating with density $f_A$ of the charge:

$$
C'(b) = b\, f_A(b)\,\big[\,1 - 1.5\;\mathbb P(t \ge b \mid A = b)\,\big].
$$

**Exact condition.** The optimal limit satisfies $\mathbb P(t \ge b^\ast \mid A = b^\ast) = 2/3$:
accept a charge as long as it is fair with probability at least $2/3$, *given that someone
charged exactly that amount*. With several opponents the bracket is averaged with weights
$f_{A_l}(b)$.

**Proposition 1 (implemented rule).** Assume the opponent's charge carries no information about
$t$ beyond our belief, $\mathbb P(t \ge a \mid A = a) = S(a)$. Then for every opponent and every
distribution of charges,

$$
b^\ast = \sup\lbrace a : S(a) \ge 2/3\rbrace  = Q(1/3),
\qquad\text{and}\qquad b^\ast = 0 \iff p_0 \ge 1/3 .
$$

*Proof.* For a single incoming charge $a$, accepting costs at most $a$ and rejecting costs
$1.5 a S(a)$ in expectation. Accepting is weakly better iff $S(a) \ge 2/3$. $S$ is
non-increasing, so the set of charges worth accepting is an interval $[0, b^\ast]$, which is
exactly a limit rule, and $S(a) \ge 2/3 \iff \mathbb P(t < a) \le 1/3 \iff a \le Q(1/3)$.
The rule is optimal charge by charge, hence for any mixture of opponents. $\square$

Code: `pricing.py::acceptance_limit`, $b = \lambda_b \cdot Q(q_b)$ with $q_b = 1/3$ and shrink
factor $\lambda_b = 1$. In closed form for $p_0 < 1/3$:

$$
\ln b^\ast = \ln m - s\, z(p_0), \qquad z(p_0) = \Phi^{-1}\!\Big(\frac{2}{3\,(1 - p_0)}\Big),\quad z(0) = 0.431 .
$$

**Correction when charges are informative.** Under the opponent model of section 3 the assumption
of Proposition 1 does not hold: $\ln A = \ln t + \alpha_l + \sigma_l \varepsilon$ is a noisy
signal of $t$. For a covered item and one opponent, conjugate normal updating gives the exact
solution

$$
\ln b^\ast = \ln m \;-\; \alpha_l\,\frac{s^2}{\sigma_l^2} \;-\; s\, z(p_0)\,\sqrt{1 + \frac{s^2}{\sigma_l^2}} .
$$

It reduces to Proposition 1 as $\sigma_l \to \infty$. The second term raises $b$ against teams
that undercharge ($\alpha_l < 0$) and lowers it against teams that overcharge. The third term
lowers $b$ because a charge near our limit is itself evidence of a low $t$. Checked by simulation
(2 million draws, $p_0 = 0$):

| $s$ | $\alpha_l$ | $\sigma_l$ | simulated $\ln(b^\ast/m)$ | formula | Proposition 1 | extra cost of Proposition 1 |
|---|---|---|---|---|---|---|
| 0.4 | −0.14 | 0.40 | −0.100 | −0.104 | −0.172 | 0.2% |
| 0.4 | 0.00 | 0.80 | −0.180 | −0.193 | −0.172 | 0.0% |
| 0.3 | 0.19 | 0.82 | −0.160 | −0.163 | −0.129 | 0.1% |
| 0.5 | −0.36 | 0.42 | +0.170 | +0.175 | −0.215 | 3.5% |

The implemented rule loses little unless an opponent undercharges strongly and precisely relative
to our own uncertainty. The correction is not implemented.

### 4.2 Charge price $a$ (issuer role)

We choose $a$ to maximize expected revenue against the opponent population with weights $w_l$,
$\sum_l w_l = 1$:

$$
R(a) = \sum_l w_l\; \mathbb E_{t}\,\mathbb E_{b_l \mid t}\big[\psi(a, b_l;\, t, c)\big].
$$

Writing the four payoff cells out, with $G_l(a \mid t)$ from 3.2 and $F_+$ the lognormal part of
our belief:

$$
R(a) = (1 - p_0) \sum_l w_l \int_0^\infty
\Big[\underbrace{\mathbf 1\lbrace a \le t\rbrace \; a\,\big(1.5 - 0.5\,G_l(a \mid t)\big)}_{\text{fair: paid } a \text{ or } 1.5a}
+ \underbrace{\mathbf 1\lbrace a > t\rbrace \;\min(a, c(t))\; G_l(a \mid t)}_{\text{inflated: paid only if accepted}}\Big]\, dF_+(t)
\;+\; p_0\, R_0(a).
$$

$R_0(a)$ is the revenue on an uncovered item, $\min(a, c_{\min})$ times the probability that the
opponent misses the exclusion and its limit clears $a$. The code sets $c(t) = 4t$, so
$c(0) = 0$ and $R_0 \equiv 0$ (see 7.3).

The trade-off is visible in the two extreme opponents:

- **Absent** ($G \equiv 0$): $R(a) = 1.5 a S(a)$. The first-order condition is unit elasticity
  of the survival function, which for the lognormal part reads
  $$\frac{\varphi(u^\ast)}{\bar\Phi(u^\ast)} = s, \qquad a^\ast = m\, e^{s u^\ast}.$$
  The inverse Mills ratio is increasing, so $u^\ast$ is unique. Examples: $s = 0.2$ gives
  $a^\ast = 0.78 m$ (the 10% quantile), $s = 0.35$ gives $0.75 m$ (20% quantile), $s = 0.6$
  gives $0.82 m$ (37% quantile). $p_0$ scales $R$ and does not move $a^\ast$.
- **Accept everything** ($G \equiv 1$): $R(a) = a S(a) + \mathbb E[\min(a, c) \mathbf 1\lbrace t < a\rbrace ]$,
  which increases in $a$ until the cap binds. This component pulls $a$ upward.

A noisy opponent sits between: its limit tracks $t$, so overcharging is rarely accepted, while a
fair charge is paid at least once.

**Numerical scheme** (`pricing.py::charge_price`, `expected_revenue`). The expectation is a
deterministic quadrature, no random sampling:

$$
\hat R(a) = \frac{1}{N_t} \sum_{i=1}^{N_t} \sum_l w_l\, \frac{1}{N_z} \sum_{n=1}^{N_z}
\psi\big(a,\ b_l(t_i, \zeta_n);\ t_i,\ 4 t_i\big),
\qquad t_i = Q\!\Big(\frac{i - 0.5}{N_t}\Big),\quad \zeta_n = \Phi^{-1}\!\Big(\frac{n - 0.5}{N_z}\Big),
$$

with $N_t = 100$, $N_z = 15$ and $b_l(t, \zeta) = t e^{\beta_l + \tau_l \zeta}$ for a noisy
opponent, $0$ for an absent one. The maximizer is searched on $N_a = 60$ log-spaced candidates
between the 5% and 95% quantiles of the positive part:

$$
a^\ast = \arg\max_{a \in \mathcal A} \hat R(a), \qquad
\mathcal A = \Big\lbrace  Q_+(0.05) \Big(\tfrac{Q_+(0.95)}{Q_+(0.05)}\Big)^{i/(N_a - 1)} : i = 0, \dots, N_a - 1 \Big\rbrace .
$$

If $m = 0$ or $p_0 = 1$ the charge is $0$.

## 5. Learning between rounds

### 5.1 Observations

After round $g$ the leaderboard publishes, for every item and ordered pair $(i, r)$, whether the
charge was accepted, $y \in \lbrace 0, 1\rbrace $, and the amount paid, $x \ge 0$. Inverting the payoff table
(`opponents.py::_items`):

| Observed | Charge of issuer $i$ | Limit of reviewer $r$ | Threshold |
|---|---|---|---|
| $y = 1,\ x > 0$ | $a_i = x$ (unless the cap bound) | $b_r \ge a_i$ | none |
| $y = 0,\ x > 0$ | $a_i = x / 1.5$ | $b_r < a_i$ | $t \ge a_i$ |
| $y = 0,\ x = 0$ | known only from another row | $b_r < a_i$ | $t < a_i$ |

Neither $t$ nor any $b$ is ever shown. Per item $j$ this yields five kinds of evidence:

- $\mathcal A_j$: teams with a known charge $a_{kj}$;
- $\mathcal F_j$: known charges proven fair, each a lower bound on $t_j$;
- $\mathcal X_j$: known charges proven inflated, each a strict upper bound on $t_j$;
- $[L_{rj}, U_{rj})$ for each reviewer: $L$ is the largest known charge it accepted, $U$ the
  smallest known charge it rejected, so $L_{rj} \le b_{rj} < U_{rj}$ (interval censoring);
- $\mathcal H_j$: teams whose charge was rejected at zero by everyone on an item with proven
  $t_j > 0$. The charge is hidden and satisfies $a_{kj} > \max(t_j, \ell_{kj})$ with
  $\ell_{kj} = \max_r L_{rj}$ (left censoring).

A team is **absent** in round $g$ if it received nothing as issuer and accepted no positive
charge as reviewer. Absent rounds are removed from the charge and limit evidence.

### 5.2 Likelihood of one item

Let $\nu = \ln v_j$. Given the team parameters $\theta$ and $\pi_0$, the joint log density of the
evidence of item $j$ and its latent state $(z, \nu)$ is

$$
\begin{aligned}
\ell_j(z, \nu) ={}& \ln \mathbb P(z) + \ln p_j(\nu)
 + \sum_{k \in \mathcal A_j} \ln \mathcal N\big(\ln a_{kj};\ \nu + \alpha_k,\ \sigma_k^2\big) \\
&+ \sum_{k \in \mathcal H_j} \ln \bar\Phi\!\Big(\frac{\max(\ln \ell_{kj},\ \nu_z) - \nu - \alpha_k}{\sigma_k}\Big)
 + \sum_{a \in \mathcal F_j} \ln h_z^{\mathcal F}(a, \nu) + \sum_{a \in \mathcal X_j} \ln h_z^{\mathcal X}(a, \nu) \\
&+ \sum_{r} \ln \Lambda_{rj}(z, \nu),
\end{aligned}
$$

with $\nu_0 = \nu$ and $\nu_1 = -\infty$ (on an uncovered item every positive charge exceeds
$t = 0$), and

- **prior on the value:** $p_j$ is flat, or $\mathcal N(\ln m_j, \max(s_j, 0.1)^2)$ when our own
  belief about the item is supplied;
- **truth signals**, softened so that a single inconsistent row cannot zero out the posterior,
  with $\epsilon = 0.02$, width $0.02$ and logistic function $\operatorname{expit}$:
  $$h_0^{\mathcal F}(a, \nu) = \epsilon + (1 - 2\epsilon)\operatorname{expit}\!\Big(\frac{\nu - \ln a}{0.02}\Big),\quad
    h_1^{\mathcal F} = \epsilon,\quad
    h_0^{\mathcal X}(a, \nu) = \epsilon + (1 - 2\epsilon)\operatorname{expit}\!\Big(\frac{\ln a - \nu}{0.02}\Big),\quad
    h_1^{\mathcal X} = 1;$$
- **interval-censored limits:** with
  $I_{rj}(\nu) = \Phi\big(\tfrac{\ln U_{rj} - \nu - \beta_r}{\tau_r}\big) - \Phi\big(\tfrac{\ln L_{rj} - \nu - \beta_r}{\tau_r}\big)$,
  $$\Lambda_{rj}(0, \nu) = I_{rj}(\nu), \qquad
    \Lambda_{rj}(1, \nu) = q_r\, I_{rj}(\nu) + (1 - q_r)\,\varsigma_{rj},\qquad
    \varsigma_{rj} = \begin{cases} 1 & L_{rj} = 0 \\ \epsilon & L_{rj} > 0. \end{cases}$$
  A reviewer that spotted the exclusion has $b = 0$ and cannot have accepted a positive charge.

The fair and inflated signals fix $t$ on an absolute scale. Without them the model would be
invariant to shifting every $\nu_j$ up and every $\alpha_k, \beta_k$ down by the same amount.

### 5.3 The update (weighted EM)

Let $G$ be the latest completed round and $\gamma_j = \delta^{ G - g(j)}$ the weight of item $j$.
The update maximizes the exponentially weighted, penalized likelihood

$$
\hat\theta^{(G)},\ \hat\pi_0^{(G)} = \arg\max_{\theta, \pi_0}\;
\sum_j \gamma_j \,\ln \sum_{z \in \lbrace 0,1\rbrace } \int e^{\ell_j(z, \nu)}\, d\nu \;+\; \ln p(\theta),
$$

by EM (`opponents.py::fit`, 40 iterations, started from
$\alpha = \beta = 0$, $\sigma = \tau = 0.3$, $q = 0.5$, $\pi_0 = 0.2$).

**E-step.** Exact posterior of the latent state on a grid of 141 points
$\nu \in [\bar\nu_j - 3.5,\ \bar\nu_j + 3.5]$, centred at the median of the item's log charges and truth signals:

$$
\omega_j(z, \nu) = \frac{e^{\ell_j(z, \nu)}}{\sum_{z'} \sum_{\nu'} e^{\ell_j(z', \nu')}} .
$$

Its marginals are the posterior of the threshold: $\hat p_{0,j} = \sum_\nu \omega_j(1, \nu)$ and
the quantiles of $\omega_j(0, \cdot)$ (stored per item as `p_zero`, `t_median`, `t_q10`, `t_q90`).

**M-step.** With prior strength $\kappa = 2$ pseudo-observations:

- *Uncovered share:* $\displaystyle \pi_0 \leftarrow \operatorname{clip}\Big(\frac{\sum_j \gamma_j \hat p_{0,j} + 1}{\sum_j \gamma_j + 5},\ 0.02,\ 0.6\Big)$,
  the posterior mean under a $\operatorname{Beta}(1, 4)$ prior.
- *Charge rule:* with residual $r_{kj} = \ln a_{kj} - \nu$ and weighted moments
  $M_n = \sum_j \gamma_j \mathbb E_{\omega_j}[r_{kj}^n]$, $n = 0, 1, 2$,
  $$\alpha_k \leftarrow \frac{M_1}{M_0 + \kappa}, \qquad
    \sigma_k^2 \leftarrow \frac{M_2 - 2\alpha_k M_1 + \alpha_k^2 M_0 + \kappa \cdot 0.3^2}{M_0 + \kappa},
    \qquad \sigma_k \in [0.05, 2],$$
  which shrinks $\alpha_k$ towards $0$ and $\sigma_k$ towards $0.3$. For a hidden charge
  ($k \in \mathcal H_j$) the moments are those of a normal truncated below at
  $\vartheta = \max(\ln \ell_{kj}, \nu_z) - \nu$: with $\xi = (\vartheta - \alpha_k)/\sigma_k$ and
  $\lambda = \varphi(\xi)/\bar\Phi(\xi)$,
  $$\mathbb E[r] = \alpha_k + \sigma_k \lambda, \qquad
    \mathbb E[r^2] = \sigma_k^2\,(1 + \xi\lambda - \lambda^2) + \mathbb E[r]^2 .$$
- *Acceptance rule:* no closed form because of the censoring. Nelder–Mead on
  $(\beta_k, \ln \tau_k, \operatorname{logit} q_k)$ maximizes
  $$\sum_j \gamma_j \sum_\nu \Big[\omega_j(0, \nu) \ln I_{kj}(\nu) + \omega_j(1, \nu) \ln\big(q_k I_{kj}(\nu) + (1 - q_k)\varsigma_{kj}\big)\Big]
    - \frac{\kappa}{2}\Big(\frac{\beta_k}{0.5}\Big)^2
    - \frac{\kappa}{2}\Big(\frac{\ln \tau_k - \ln 0.3}{0.7}\Big)^2
    + \kappa \ln\big(q_k (1 - q_k)\big),$$
  inside the box $\beta_k \in [-3, 3]$, $\tau_k \in [0.03, 3]$, $\operatorname{logit} q_k \in [-6, 6]$.
- *Absent rate:* computed directly, outside EM,
  $\displaystyle \rho_k = \frac{\sum_{g \le G} \delta^{G - g} \mathbf 1\lbrace k \text{ absent in } g\rbrace }{\sum_{g \le G} \delta^{G - g}}$.

**Forgetting.** The weights $\gamma_j$ bound the effective memory at
$\sum_{n \ge 0} \delta^n = 1/(1 - \delta) \approx 6.7$ rounds. A team that changes strategy is
followed with a half-life of $\ln 0.5 / \ln 0.85 \approx 4.3$ rounds. The fit is a batch refit on
all rounds, not a recursive update, so the result does not depend on the order of processing.

### 5.4 How the update enters the next round

$$
\underbrace{\mathcal D_{1:G}}_{\text{public transactions}}
\ \xrightarrow{\ \text{EM}\ }\
\big(\hat\theta^{(G)}, \hat\pi_0^{(G)}\big)
\ \xrightarrow{\ \text{3.2}\ }\
\lbrace w_l, G_l\rbrace _{l \ne 0}
\ \xrightarrow{\ \text{4.2}\ }\
a^\ast_{G+1, j} = \arg\max_{a \in \mathcal A} \hat R\big(a;\ p_{0j}, m_j, s_j,\ \hat\theta^{(G)}\big).
$$

Only the charge price depends on what was learned. The acceptance limit stays
$b^\ast_{G+1, j} = Q_j(1/3)$.

## 6. One round, end to end

1. **Before the case drops** (offline, `scripts/update_opponents.py`): fetch $\mathcal D_G$,
   refit $\hat\theta^{(G)}$, write `results/opponent_belief.json`.
2. **Case drops** (60 s): LLM returns $(p_{\text{cov}}, p_{\text{rel}}, n, u, s_{\text{LLM}})$ per item.
3. **Belief:** $(p_0, m, s)$ per item (section 2).
4. **Decide:** $b^\ast = Q(1/3)$ and $a^\ast = \arg\max \hat R(a)$ (section 4). Submit.
5. **Round completes:** transactions become public; go to 1.

## 7. Where the code and the mathematics differ

| # | Issue | Effect | Status |
|---|---|---|---|
| 7.1 | The belief about $t$ is not learned across rounds. The EM posterior of each past $t_j$ (5.3) is saved but never compared with our own $(m_j, s_j, p_{0j})$. | A systematic bias of the LLM in price level, stated uncertainty or coverage persists for the whole tournament and affects both $a$ and $b$. | Open. Natural step: fit $\ln t_j = \ln m_j + d + \lambda s_j \xi$ on past items and use $m e^{d}$, $\lambda s$. |
| 7.2 | Proposition 1 assumes opponents' charges are uninformative about $t$. | $b$ is slightly off; simulated extra cost 0 to 3.5% (4.1). | Open, small. |
| 7.3 | The game states $c \ge 4t$ and "never below an absolute floor". The code uses $c = 4t$, so an uncovered item pays nothing even if accepted, and the miss rate $q_l$ has no influence on $a$. | Revenue from uncovered items is ignored. In rounds 1 to 7, five items show no positive payment at all, including on accepted rows. That fits a cap of $0$ on uncovered items, or every team charging $0$ there; the data cannot tell these apart, so the floor is not identified. | Open. |
| 7.4 | `fit` accepts our own estimates as a prior on $\nu_j$ (5.2) but the update script does not pass them. | The value prior is flat; items with few known charges are weakly identified. | Open, one-line change. |
| 7.5 | We maximize expected payoff, the tournament pays by rank. | Risk neutrality is a proxy; late in the tournament the rank-optimal choice may be more or less aggressive. | By design. |
| 7.6 | `results/opponent_belief.json` was fitted on rounds 1 to 5; transactions for rounds 6 and 7 are on disk. In rounds 1 to 5 our own team appears as absent. | The charge price is optimized against a belief two rounds old. | Rerun `scripts/update_opponents.py`. |
