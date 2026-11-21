"""The opponent belief recovers known behaviour from simulated public transactions."""

import math
import random

import pytest

from c2f.belief import TBelief
from c2f.opponents import Tx, _items, fit, opponent_types
from c2f.payoff import insurer_pays
from c2f.pricing import OpponentType, PricingParams, charge_price, expected_revenue

TEAMS = {  # name: (alpha, sigma, beta, tau, miss, absent_rate)
    "greedy":  (0.30, 0.15, 0.20, 0.15, 0.8, 0.0),
    "honest":  (0.00, 0.10, -0.10, 0.10, 0.2, 0.0),
    "strict":  (-0.10, 0.15, -0.35, 0.10, 0.1, 0.0),
    "lax":     (0.10, 0.20, 0.40, 0.20, 0.9, 0.0),
    "sloppy":  (0.00, 0.40, 0.00, 0.40, 0.5, 0.0),
    "sleepy":  (0.00, 0.20, 0.00, 0.20, 0.5, 1.0),
}


def simulate(n_games=8, n_items=8, pi=0.2, seed=1):
    rng = random.Random(seed)
    txs, truth = [], {}
    for g in range(1, n_games + 1):
        for j in range(1, n_items + 1):
            v = math.exp(rng.uniform(math.log(50), math.log(3000)))
            covered = rng.random() > pi
            t = v if covered else 0.0
            truth[(g, j)] = t
            a, b = {}, {}
            for k, (al, s, be, ta, miss, ab) in TEAMS.items():
                if rng.random() < ab:
                    a[k] = b[k] = 0.0
                    continue
                a[k] = round(v * math.exp(al + s * rng.gauss(0, 1)), 2)
                spotted = (not covered) and rng.random() > miss
                b[k] = 0.0 if spotted else v * math.exp(be + ta * rng.gauss(0, 1))
            for i in TEAMS:
                for r in TEAMS:
                    if i == r:
                        continue
                    acc = a[i] <= b[r]
                    txs.append(Tx(g, j, i, r, acc, insurer_pays(a[i], b[r], t)))
    return txs, truth


@pytest.fixture(scope="module")
def fitted():
    txs, truth = simulate()
    return fit(txs, decay=1.0, n_iter=25), truth


def test_recovers_acceptance_rules(fitted):
    """Strict teams reject inside the range of observed charges, so b is bracketed on both
    sides and recovered. Lenient teams accept nearly every charge they see: b is only
    bounded below, so the fit may sit higher but never clearly lower than the truth."""
    belief, _ = fitted
    for k in ("honest", "strict"):
        tb, (_, _, beta, tau, _, _) = belief.teams[k], TEAMS[k]
        assert tb.beta == pytest.approx(beta, abs=0.1), k
        assert tb.tau == pytest.approx(tau, abs=0.1), k
    for k in ("greedy", "lax"):
        assert belief.teams[k].beta > TEAMS[k][2] - 0.1, k


def test_recovers_charge_rules(fitted):
    belief, _ = fitted
    for k in ("greedy", "honest", "strict", "lax"):
        assert belief.teams[k].alpha == pytest.approx(TEAMS[k][0], abs=0.1), k


def test_orders_teams_by_strictness(fitted):
    belief, _ = fitted
    betas = {k: belief.teams[k].beta for k in ("strict", "honest", "greedy", "lax")}
    assert betas["strict"] < betas["honest"] < betas["greedy"] < betas["lax"]


def test_detects_absent_team(fitted):
    belief, _ = fitted
    assert belief.teams["sleepy"].absent_rate == 1.0
    assert all(belief.teams[k].absent_rate == 0.0 for k in ("greedy", "honest"))


def test_posterior_of_t_is_accurate(fitted):
    belief, truth = fitted
    errs, zero_hits = [], []
    for it in belief.items:
        t = truth[(it["game"], it["item"])]
        if t > 0 and it["p_zero"] < 0.5:
            errs.append(abs(math.log(it["t_median"] / t)))
        zero_hits.append((it["p_zero"] > 0.5) == (t == 0))
    assert sum(errs) / len(errs) < 0.12           # median within ~12 % on average
    assert sum(zero_hits) / len(zero_hits) > 0.85  # coverage classified correctly


def test_interval_for_b_from_transactions():
    txs = [Tx(1, 1, "x", "r", True, 100.0), Tx(1, 1, "y", "r", False, 300.0),  # y fair: a=200
           Tx(1, 1, "z", "r", False, 0.0), Tx(1, 1, "z", "q", True, 500.0)]   # z fraud: a=500
    it = _items(txs, absent=set())[0]
    assert it.charges == {"x": 100.0, "y": 200.0, "z": 500.0}
    assert it.intervals["r"] == (100.0, 200.0)
    assert it.fair == [200.0] and it.fraud == [500.0]


def test_best_response_uses_learned_opponents(fitted):
    """Against the learned population our a should beat a against the default guess."""
    belief, _ = fitted
    learned = PricingParams(opponents=opponent_types(belief, us="none"))
    default = PricingParams()
    bel = TBelief(median=500, sigma=0.3, p_zero=0.1)
    a_learned, a_default = charge_price(bel, learned), charge_price(bel, default)
    assert expected_revenue(a_learned, bel, learned) >= expected_revenue(a_default, bel, learned)
    types = opponent_types(belief, us="none")
    assert any(o.kind == "absent" and o.weight == 1.0 for o in types)   # sleepy
    assert all(isinstance(o, OpponentType) for o in types)
