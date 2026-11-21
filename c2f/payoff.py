"""Exact payoff rules of the Claim to Fame game (see docs/CASE.md).

All amounts are gross line-item totals (incl. VAT), never per unit.
"""

from __future__ import annotations


def insurer_pays(a: float, b: float, t: float, c: float | None = None) -> float:
    """Amount the insurer pays the handyman for one line item.

    a: handyman's charge price, b: insurer's acceptance limit,
    t: secret fair-value threshold (0 if not covered), c: secret payment cap (>= 4t).
    """
    if c is None:
        c = 4.0 * t
    accepted = a <= b
    fair = a <= t
    if fair:
        return a if accepted else 1.5 * a
    return min(a, c) if accepted else 0.0


def net_payoff(a_us: float, b_us: float, a_opp: float, b_opp: float, t: float,
               c: float | None = None) -> float:
    """Our net payoff for one line item against one opponent (both roles)."""
    received = insurer_pays(a_us, b_opp, t, c)
    paid = insurer_pays(a_opp, b_us, t, c)
    return received - paid
