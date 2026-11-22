"""Overhaul of the case analysis (docs/research/overhaul.md): analyst prompts, decision table,
case memory and the per-sigma cache of the charge search."""

import math

import pytest

from c2f import analyst, decision, memory
from c2f import item_prior as ip
from c2f.belief import TBelief
from c2f.pricing import PricingParams, _charge_price_search, charge_price


# ---- pricing: the cached factor gives the same charge as the full search --------------------

@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.35])
def test_charge_factor_matches_search(sigma):
    p = PricingParams(charge_min_ratio=0.87, charge_conf_sigma=0.12, charge_conf_ratio=0.95)
    for median in (17.3, 420.0, 2950.0):
        direct = _charge_price_search(TBelief(median, sigma, 0.0), p)
        assert charge_price(TBelief(median, sigma, 0.4), p) == pytest.approx(direct, rel=1e-9)
    assert charge_price(TBelief(0.0, sigma, 0.0), p) == 0.0


# ---- analyst: reference blocks and the switch ------------------------------------------------

def _obs(game, text, qty, unit, lo, hi=math.inf, kind=None, policy="P"):
    key, fam = ip.line_key(text, unit)
    kind = kind or ("zero" if lo == 0 else "bracket" if hi < math.inf else "lower")
    return ip.Obs(game, 1, key, fam, text, qty, unit, kind, lo / qty, hi / qty, policy)


def test_full_cover_levels_only_where_rates_differ():
    obs = [_obs(24, "Skilled worker hours", 1, "hr", 35, 42), _obs(46, "Skilled worker hours", 14, "hr", 1260),
           _obs(43, "Vehicle costs", 1, "pc", 82), _obs(44, "Vehicle costs", 1, "pc", 81)]
    lines = {1: "Skilled worker hours 14 hrs", 2: "Vehicle costs 1 pcs", 3: "Never seen 1 pcs"}
    block = analyst.full_cover_levels(lines, obs)
    assert 'line 1 "Skilled worker hours": full rate per hr >=90 (case 46)' in block
    assert "35-42 (case 24)" in block and "at the full rate: >=1260" in block
    assert "Vehicle costs" not in block and "Never seen" not in block     # one level / no history
    assert analyst.full_cover_levels({1: "Never seen 1 pcs"}, obs) == ""


def test_analyst_prompts():
    assert "INVOICED quantity" in analyst.system("x5")
    assert "line_total_gross_eur" in analyst.system("x5")           # same output schema as x3
    from c2f.extract import SYSTEM, SYSTEM_UPPER
    assert analyst.system("rel") == SYSTEM and analyst.system("rel", claude=True) == SYSTEM_UPPER


def test_agent_falls_back_to_x3_reference(monkeypatch):
    from c2f import agent, pricebook
    monkeypatch.setattr(pricebook, "reference", lambda **kw: "NEWEST" if "case" not in kw else "RANKED")
    monkeypatch.setattr(agent, "_with_item_prior", lambda ref, case, llm, errors: ref + "+PRIOR")
    assert agent._reference(object(), {"analyst": "x3"}, []) == ("NEWEST+PRIOR", None, None)
    assert agent._reference(object(), {}, []) == ("NEWEST+PRIOR", None, None)
    errors: list[str] = []
    monkeypatch.setattr(analyst, "reference", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert agent._reference(object(), {"analyst": "x5"}, errors) == ("NEWEST+PRIOR", None, None)
    assert errors and "boom" in errors[0]
    monkeypatch.setattr(analyst, "reference", lambda *a, **kw: "RANKED+LEVELS")
    ref, system, name = agent._reference(object(), {"analyst": "x5"}, [])
    assert (ref, name) == ("RANKED+LEVELS", "x5") and system == analyst.SYSTEM_X5
    # rel keeps the x3 system text of the stage: the Claude stage its upper-end variant
    from c2f.extract import SYSTEM, SYSTEM_UPPER
    assert agent._reference(object(), {"analyst": "rel"}, [])[1] == SYSTEM
    assert agent._reference(object(), {"analyst": "rel"}, [], claude=True)[1] == SYSTEM_UPPER


# ---- decision table ---------------------------------------------------------------------

def _item(t, charges):
    """market summary of one line: opponents charged `charges` (all fair) and accept up to 1.1 t"""
    return {"t_low": t, "t_high": t * 1.02, "charges": {f"o{i}": c for i, c in enumerate(charges)},
            "fair": {f"o{i}": True for i in range(len(charges))},
            "b_bounds": {f"o{i}": [t * 0.6, t * 0.7] for i in range(len(charges))}}


def test_decision_table_learns_the_bias_of_the_estimates():
    # estimates sit 20 % below the threshold in the normal class: charge and accept more
    rows = [decision.Row(g, 0.3, 100.0, 0.05, _item(125.0, [120.0, 122.0, 90.0]), 87.0, 90.0)
            for g in range(1, 9) for _ in range(6)]
    table = decision.fit(rows, min_lines=40, min_games=6)
    rec = table[decision.klass(0.3)]
    assert rec["ka"] is not None and 1.1 <= rec["ka"] <= 1.2      # below 1.25, with the 5 % margin
    assert rec["kb"] >= 1.2                                       # accepts the fair charges of 120-122
    assert rec["table_rev"] > rec["rule_rev"]
    assert table[decision.klass(0.05)]["ka"] is None              # no lines in that class
    a, b = decision.quote(200.0, 0.3, 0.05, table, (174.0, 180.0))
    assert a == pytest.approx(200.0 * rec["ka"]) and b == pytest.approx(200.0 * rec["kb"])
    assert decision.quote(200.0, 0.3, 0.5, table, (174.0, 0.0))[1] == 0.0       # P(t = 0) > 1/3
    assert decision.quote(200.0, 0.05, 0.0, table, (190.0, 192.0)) == (190.0, 192.0)   # silent class
    assert decision.quote(200.0, 0.3, 0.05, table, (174.0, 180.0), use="b")[0] == 174.0


def test_decision_table_needs_enough_lines():
    rows = [decision.Row(1, 0.3, 100.0, 0.0, _item(125.0, [120.0]), 87.0, 90.0)] * 39
    assert decision.fit(rows, min_lines=40, min_games=1)[decision.klass(0.3)]["ka"] is None
    assert decision.klass(0.1) == 0 and decision.klass(0.2) == 1 and decision.klass(0.3) == 2 and decision.klass(0.6) == 3


def test_decision_load_checks_the_analyst(tmp_path):
    import json
    f = tmp_path / "decision.json"
    f.write_text(json.dumps({"analyst": "x5", "table": {"2": {"ka": 1.0, "kb": 1.05, "n": 50}}}))
    assert decision.load(f, "x5")[2]["ka"] == 1.0
    assert decision.load(f, "x3") == {}                 # learned on another prompt: not used
    assert decision.load(tmp_path / "missing.json") == {}


# ---- case memory ------------------------------------------------------------------------

def test_memory_classes_and_trust():
    lines = ["Filling valve 1 pcs", "Connection hose 1 pcs", "Senior installer hours 3 hrs", "Vehicle costs 1 pcs"]
    obs = []
    for g, shift in ((10, 1.0), (20, 1.02), (30, 0.99)):          # the same case drawn three times
        for k, text in enumerate(lines):
            desc, qty, unit = ip.parse_line(text)
            obs.append(_obs(g, desc, qty, unit, 100.0 * (k + 1) * shift, 104.0 * (k + 1) * shift, policy="WD"))
    obs.append(_obs(25, "Vehicle costs", 1, "pc", 80.0, 84.0, policy="OTHER"))       # another case
    mem = memory.Memory(obs, memory.Options(min_pairs=4))
    assert mem.trust["case"][0] > 0.8 and mem.trusted("case")
    precs = mem.precedents({1: lines[0], 2: lines[1], 3: lines[2], 4: lines[3]}, "WD")
    assert [p.cls for p in precs[4]][:3] == ["case", "case", "case"] and precs[4][-1].cls == "type"
    t, sigma, cls = mem.value(precs[4])
    assert cls == "case" and 396 <= t <= 420 and sigma <= 0.1
    assert mem.value([]) is None
