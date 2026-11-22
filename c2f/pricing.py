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
from functools import lru_cache
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
    charge_min_ratio: float = 0.0        # never charge below this share of our median estimate
    charge_conf_sigma: float = 0.0       # beliefs with sigma below this count as confident ...
    charge_conf_ratio: float = 0.0       # ... and are charged at least this share of the estimate
    n_t_grid: int = 100
    n_candidates: int = 60
    n_opp_draws: int = 15
    opponents: tuple[OpponentType, ...] = field(default_factory=lambda: (
        OpponentType(0.15, "absent"),
        OpponentType(0.05, "accept_all"),
        OpponentType(0.80, "noisy", mu=0.0, s=0.35, miss_zero=0.5),
    ))


def acceptance_limit(belief: TBelief, p: PricingParams) -> float:
    q = belief.quantile(p.acceptance_quantile)
    b = q * p.acceptance_shrink
    if p.acceptance_shrink > 1.0:
        # A factor above 1 never lifts the limit over our own estimate of t (confident lines
        # already sit at about 0.96 x estimate). Replay games 26-46: same saving with the cap.
        b = min(b, max(q, belief.median))
    return max(0.0, b)


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


def expected_revenues(cands: list[float], belief: TBelief, p: PricingParams) -> list[float]:
    """expected_revenue for every candidate at once (numpy). Same model and same numbers as
    expected_revenue up to float summation order; about 100x faster. The pure-Python loop took
    40 s and more per case on the Windows runner (games 52, 53: full and backup stage missed
    the deadline because three stages price in parallel)."""
    import numpy as np

    ts = belief.grid(p.n_t_grid)
    zs = [_STD.inv_cdf((i + 0.5) / p.n_opp_draws) for i in range(p.n_opp_draws)]
    total_w = sum(o.weight for o in p.opponents)
    a = np.asarray(cands, dtype=float)[:, None, None]            # candidates x t x draws
    t = np.asarray(ts, dtype=float)[None, :, None]
    fair = a <= t
    capped = np.minimum(a, 4.0 * t)                              # payment cap c = 4t
    rev = np.zeros(len(cands))
    for opp in p.opponents:
        # limits per t, built with the same arithmetic as _opp_limits
        b = np.asarray([_opp_limits(x, belief, opp, zs) for x in ts], dtype=float)[None, :, :]
        accepted = a <= b
        pay = np.where(fair, np.where(accepted, a, 1.5 * a), np.where(accepted, capped, 0.0))
        rev += opp.weight / total_w * pay.mean(axis=2).sum(axis=1)
    return (rev / len(ts)).tolist()


def _best_candidate(cands: list[float], belief: TBelief, p: PricingParams) -> float:
    try:
        revs = expected_revenues(cands, belief, p)
        return cands[max(range(len(cands)), key=revs.__getitem__)]   # first maximum, like max()
    except Exception:  # noqa: BLE001  numpy missing or failing must not cost a game
        return max(cands, key=lambda a: expected_revenue(a, belief, p))


def charge_price(belief: TBelief, p: PricingParams) -> float:
    if belief.median <= 0:
        return 0.0
    # The rule is scale-free: opponents' limits are multiples of t and the cap is 4t, so the
    # charge is the median times a factor that depends on sigma only. The search for that
    # factor takes seconds per line against the learned opponent population (22 types), and
    # most lines of an invoice share a sigma, so the factor is computed once per sigma.
    return belief.median * _charge_ratio(belief.sigma, p)


@lru_cache(maxsize=4096)
def _charge_ratio(sigma: float, p: PricingParams) -> float:
    """charge / median of a covered belief with this log-sd (see charge_price)."""
    return _charge_price_search(TBelief(1000.0, sigma, 0.0), p) / 1000.0


def _charge_price_search(belief: TBelief, p: PricingParams) -> float:
    # Priced as if the line is covered, whatever P(t = 0): on an uncovered line a positive
    # charge costs nothing (rejected -> 0) and still pays min(a, c) wherever it is accepted,
    # so a = 0 only gives that chance away and loses everything when we misjudged coverage.
    pos = TBelief(belief.median, belief.sigma, 0.0)
    belief = pos
    lo = pos.quantile(p.charge_floor_quantile)
    hi = pos.quantile(p.charge_ceiling_quantile)
    n = max(p.n_candidates, 2)
    cands = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)] if lo > 0 else [hi]
    best = _best_candidate(cands, belief, p)
    # Replay of games 1-12: the opponent model shades a to about 0.8 x estimate, yet our
    # charges were always proven fair with room above, so keep a at least this close.
    # Confident estimates (amount fixed by the policy or by a near-identical earlier case) are
    # charged almost in full. Game 31: estimate 2,950 for a threshold of 2,900-3,110 was charged
    # at 0.87 x while four teams charged 2,687-2,900. Replay of the confident lines in stored
    # runs: +38% at 0.95, rising smoothly up to 1.0 (no cliff), better in 3 of 4 games.
    # Rejected on the replay: a lower floor for uncertain estimates (0.87 is their optimum).
    floor_c = p.charge_conf_ratio * belief.median if belief.sigma < p.charge_conf_sigma else 0.0
    return max(best, p.charge_min_ratio * belief.median, floor_c)


@dataclass(frozen=True)
class Quote:
    charge_price: float      # a, gross line-item total
    acceptance_limit: float  # b, gross line-item total


def quote(belief: TBelief, p: PricingParams | None = None) -> Quote:
    p = p or PricingParams()
    return Quote(round(charge_price(belief, p), 2), round(acceptance_limit(belief, p), 2))
