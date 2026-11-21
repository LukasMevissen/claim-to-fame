"""Turn a belief about t into (a, b) for one line item.

Acceptance limit b (insurer role)
    For an incoming charge a, accepting costs about a (exactly a if a <= t,
    min(a, c) <= a otherwise). Rejecting costs 1.5a if a <= t, else 0.
    So accept iff  a <= 1.5 a P(t >= a)  <=>  P(t >= a) >= 2/3.
    The optimal limit is therefore the 1/3 quantile of our belief about t,
    and b = 0 whenever P(t = 0) > 1/3. See docs/STRATEGY.md.

Charge price a (handyman role)
    Depends on how opponents set b. We maximise expected revenue over a
    configurable opponent population (absent teams with b = 0, generous teams
    that accept everything, and noisy estimators b = t * exp(mu + s Z)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

from .belief import TBelief
from .payoff import insurer_pays

_STD = NormalDist()


@dataclass(frozen=True)
class OpponentType:
    weight: float
    kind: str                 # "absent" | "accept_all" | "noisy"
    mu: float = 0.0           # log-bias of their estimate of t
    s: float = 0.3            # log-noise of their estimate of t
    miss_zero: float = 0.5    # P(they fail to spot an uncovered item)


@dataclass(frozen=True)
class PricingParams:
    acceptance_quantile: float = 1.0 / 3.0
    acceptance_shrink: float = 1.0       # multiply b by this (<1 = stricter insurer)
    charge_floor_quantile: float = 0.05  # never search below this quantile of t | t>0
    charge_ceiling_quantile: float = 0.95
    n_t_grid: int = 100
    n_candidates: int = 60
    n_opp_draws: int = 15
    opponents: tuple[OpponentType, ...] = field(default_factory=lambda: (
        OpponentType(0.15, "absent"),
        OpponentType(0.05, "accept_all"),
        OpponentType(0.80, "noisy", mu=0.0, s=0.35, miss_zero=0.5),
    ))


def acceptance_limit(belief: TBelief, p: PricingParams) -> float:
    return max(0.0, belief.quantile(p.acceptance_quantile) * p.acceptance_shrink)


def _opp_limits(t: float, belief: TBelief, opp: OpponentType, zs: list[float]) -> list[float]:
    if opp.kind == "absent":
        return [0.0]
    if opp.kind == "accept_all":
        return [math.inf]
    if t > 0:
        return [t * math.exp(opp.mu + opp.s * z) for z in zs]
    # Uncovered item: a share of opponents misses it and prices it like a covered one.
    n_miss = round(opp.miss_zero * len(zs))
    return ([belief.median * math.exp(opp.mu + opp.s * z) for z in zs[:n_miss]]
            + [0.0] * (len(zs) - n_miss))


def expected_revenue(a: float, belief: TBelief, p: PricingParams) -> float:
    ts = belief.grid(p.n_t_grid)
    zs = [_STD.inv_cdf((i + 0.5) / p.n_opp_draws) for i in range(p.n_opp_draws)]
    total_w = sum(o.weight for o in p.opponents)
    rev = 0.0
    for t in ts:
        for opp in p.opponents:
            bs = _opp_limits(t, belief, opp, zs)
            rev += opp.weight / total_w * sum(insurer_pays(a, b, t) for b in bs) / len(bs)
    return rev / len(ts)


def charge_price(belief: TBelief, p: PricingParams) -> float:
    if belief.median <= 0:
        return 0.0
    # Priced as if the line is covered, whatever P(t = 0): on an uncovered line a positive
    # charge costs nothing (rejected -> 0) and still pays min(a, c) wherever it is accepted,
    # so a = 0 only gives that chance away and loses everything when we misjudged coverage.
    pos = TBelief(belief.median, belief.sigma, 0.0)
    belief = pos
    lo = pos.quantile(p.charge_floor_quantile)
    hi = pos.quantile(p.charge_ceiling_quantile)
    n = max(p.n_candidates, 2)
    cands = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)] if lo > 0 else [hi]
    return max(cands, key=lambda a: expected_revenue(a, belief, p))


@dataclass(frozen=True)
class Quote:
    charge_price: float      # a, gross line-item total
    acceptance_limit: float  # b, gross line-item total


def quote(belief: TBelief, p: PricingParams | None = None) -> Quote:
    p = p or PricingParams()
    return Quote(round(charge_price(belief, p), 2), round(acceptance_limit(belief, p), 2))
