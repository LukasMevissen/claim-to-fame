"""Belief about the secret fair-value threshold t of one line item.

Model: with probability p_zero the item is not covered or not related (t = 0).
Otherwise t is lognormal around a median gross total with log-sd sigma.
Quantiles are computed analytically, so pricing is deterministic and fast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

_STD = NormalDist()


@dataclass(frozen=True)
class TBelief:
    median: float          # median of t given t > 0 (gross line-item total, EUR)
    sigma: float           # log-sd of t given t > 0
    p_zero: float = 0.0    # P(t = 0): not covered or not related

    def __post_init__(self) -> None:
        if self.median < 0 or self.sigma < 0 or not 0.0 <= self.p_zero <= 1.0:
            raise ValueError(f"invalid belief {self}")

    def survival(self, x: float) -> float:
        """P(t >= x)."""
        if x <= 0:
            return 1.0
        if self.p_zero >= 1.0 or self.median <= 0:
            return 0.0
        if self.sigma == 0:
            pos = 1.0 if x <= self.median else 0.0
        else:
            pos = 1.0 - _STD.cdf((math.log(x) - math.log(self.median)) / self.sigma)
        return (1.0 - self.p_zero) * pos

    def quantile(self, q: float) -> float:
        """Smallest x with P(t <= x) >= q."""
        if q <= self.p_zero or self.median <= 0:
            return 0.0
        q_pos = (q - self.p_zero) / (1.0 - self.p_zero)
        q_pos = min(max(q_pos, 1e-9), 1 - 1e-9)
        if self.sigma == 0:
            return self.median
        return self.median * math.exp(self.sigma * _STD.inv_cdf(q_pos))

    def grid(self, n: int = 200) -> list[float]:
        """n equally weighted representative values of t (midpoint quantiles)."""
        return [self.quantile((i + 0.5) / n) for i in range(n)]
