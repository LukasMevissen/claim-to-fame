import math

import pytest

from c2f.agent import PricedItem, price_items, to_belief
from c2f.belief import TBelief
from c2f.config import load_config
from c2f.evaluate import gates, simulate
from c2f.extract import ItemEstimate
from c2f.payoff import insurer_pays, net_payoff
from c2f.pricing import PricingParams, acceptance_limit, charge_price, quote


# ---------- payoff table (docs/CASE.md) ----------

@pytest.mark.parametrize("a,b,t,expected", [
    (80, 100, 100, 80),      # fair, accepted -> a
    (80, 50, 100, 120),      # fair, rejected -> 1.5a
    (150, 200, 100, 150),    # fraudulent, accepted -> min(a, c)
    (500, 1000, 100, 400),   # fraudulent, accepted, capped at c = 4t
    (150, 100, 100, 0),      # fraudulent, rejected -> 0
    (10, 0, 0, 0),           # uncovered item (t = 0), rejected -> 0
])
def test_insurer_pays(a, b, t, expected):
    assert insurer_pays(a, b, t) == pytest.approx(expected)


def test_net_payoff_symmetric_zero():
    assert net_payoff(100, 100, 100, 100, 100) == 0


# ---------- belief ----------

def test_belief_quantile_survival_consistent():
    bel = TBelief(median=200, sigma=0.4, p_zero=0.1)
    for q in (0.2, 0.5, 0.9):
        x = bel.quantile(q)
        assert bel.survival(x) == pytest.approx(1 - q, abs=1e-6)
    assert bel.quantile(0.05) == 0.0   # inside the point mass at 0


# ---------- acceptance limit b ----------

def _insurer_cost(a, accept, bel, n=4000):
    ts = bel.grid(n)
    if accept:
        return sum(a if a <= t else min(a, 4 * t) for t in ts) / n
    return sum(1.5 * a if a <= t else 0 for t in ts) / n


def test_acceptance_limit_is_one_third_quantile():
    bel = TBelief(median=300, sigma=0.3, p_zero=0.05)
    b = acceptance_limit(bel, PricingParams())
    assert b == pytest.approx(bel.quantile(1 / 3))


def test_acceptance_rule_is_optimal_without_cap():
    bel = TBelief(median=300, sigma=0.3, p_zero=0.05)
    b = acceptance_limit(bel, PricingParams())
    # Below b accepting is cheaper, above b rejecting is cheaper (cap only helps accepting).
    for a in (0.6 * b, 0.95 * b):
        assert _insurer_cost(a, True, bel) <= _insurer_cost(a, False, bel)
    for a in (1.05 * b, 1.5 * b):
        assert _insurer_cost(a, False, bel) <= _insurer_cost(a, True, bel) + 1e-9


def test_reject_everything_when_probably_uncovered():
    assert acceptance_limit(TBelief(500, 0.3, p_zero=0.4), PricingParams()) == 0.0


# ---------- charge price a ----------

def test_charge_close_to_t_when_certain():
    a = charge_price(TBelief(250, 0.01, 0.0), PricingParams())
    assert a == pytest.approx(250, rel=0.03)


def test_uncovered_item_charges_zero():
    assert quote(TBelief(500, 0.3, p_zero=1.0)).charge_price == 0.0


def test_quotes_finite_and_nonnegative():
    for bel in (TBelief(0, 0.3, 0), TBelief(50, 1.0, 0.5), TBelief(5000, 0.2, 0.0)):
        q = quote(bel)
        assert math.isfinite(q.charge_price) and q.charge_price >= 0
        assert math.isfinite(q.acceptance_limit) and q.acceptance_limit >= 0


# ---------- config + agent + evaluation ----------

def _items():
    return [
        ItemEstimate("1", "Leckortung", 1, "Pausch", 0.95, 0.95, 250.0, 0.25),
        ItemEstimate("2", "Trocknungsgerät Miete", 5, "Tag", 0.9, 0.9, 35.0, 0.3),
        ItemEstimate("3", "Neue Küchenzeile", 1, "Stk", 0.1, 0.2, 4000.0, 0.4),
    ]


def test_default_config_loads():
    cfg = load_config()
    p = cfg.pricing
    assert len(p.opponents) == 3 and 0 < p.acceptance_quantile < 1


def test_price_items_and_gates():
    priced = price_items(_items(), load_config())
    assert [p.item_id for p in priced] == ["1", "2", "3"]
    assert gates(priced, {"total_s": 10}, 50, {"1": {}, "2": {}, "3": {}}) == []
    assert any("not priced" in e for e in gates(priced, {"total_s": 1}, 50, {"4": {}, **{k: {} for k in "123"}}))
    bad = [PricedItem("x", "", {"median": 1, "p_zero": 0}, -1.0, 1.0)]
    assert gates(bad, {"total_s": 60}, 50, None)


def test_simulation_metrics_and_zero_detection():
    labels = {"1": {"t": 250.0}, "2": {"t": 175.0}, "3": {"t": 0.0, "price_if_covered": 4000.0}}
    m = simulate(price_items(_items(), load_config()), labels)
    assert m["oracle"] > 0
    assert m["zero_items"] == {"tp": 1, "fn": 0, "fp": 0}


def test_beats_naive_in_expectation_over_belief():
    """If t really follows our belief, the decision-theoretic quotes must beat a = b = median.
    (On one fixed draw naive can win by luck; that is why this averages over t.)"""
    cfg = load_config()
    for it, p in zip(_items(), price_items(_items(), cfg)):
        bel = to_belief(it, cfg)
        ours = naive = 0.0
        for i, t in enumerate(bel.grid(40)):
            m = simulate([p], {p.item_id: {"t": t, "price_if_covered": bel.median}}, seed=i)
            ours += m["net_payoff"]
            naive += m["naive_baseline"]
        assert ours > naive, it.item_id
