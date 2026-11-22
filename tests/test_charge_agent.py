import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from c2f import charge_agent as ca
from c2f import novelty
from c2f.agent import _agent_charges, price_items
from c2f.case_io import Case
from c2f.config import Config
from c2f.extract import ItemEstimate

POLICY = "RESIDENTIAL DWELLING INSURANCE / Policy wording \"WD-2026\""
CLAIM = frozenset({"lightning", "surge", "conditioning", "unit", "replaced"})


def prec(game, line, key, lo, hi=math.inf, unit="pc", policy=POLICY, claim=CLAIM):
    return ca.Precedent(game, line, key, key, 1.0, unit, "bracket" if math.isfinite(hi) else "lower",
                        lo, hi, policy, novelty.tokens(key), claim)


def history(precs):
    return ca.History(None, precs, {}, {}, ca.acceptance_curve({}), ca.learn_relevance(precs))


# ---------- decision ----------

def test_confident_estimate_is_charged_almost_in_full_and_uncertain_stays_in_bounds():
    h = history([])
    sure = ca.decide(1000.0, 0.05, 1.0, [], h, ca.NEUTRAL)
    assert 0.88 <= sure.ratio <= 1.0
    unsure = ca.decide(1000.0, 0.5, 1.0, [], h, ca.NEUTRAL)
    assert ca.NEUTRAL.min_ratio <= unsure.ratio <= ca.NEUTRAL.max_ratio
    assert unsure.t_q90 / unsure.t_q10 > sure.t_q90 / sure.t_q10


def test_tight_relevant_precedent_moves_the_charge_towards_it():
    h = history([prec(38, 1, "air conditioning unit", 1340.0, 1445.0)])
    base = ca.decide(1000.0, 0.3, 1.0, [], h, ca.NEUTRAL)
    with_prec = ca.decide(1000.0, 0.3, 1.0, [ca.Weight(38, 1, 0.8)], h, ca.NEUTRAL)
    assert with_prec.a > base.a
    assert 1150 <= with_prec.a <= 1445                 # close to, not above, what was proven
    assert with_prec.t_q90 / with_prec.t_q10 < base.t_q90 / base.t_q10
    weak = ca.decide(1000.0, 0.3, 1.0, [ca.Weight(38, 1, 0.1)], h, ca.NEUTRAL)
    assert base.a <= weak.a <= with_prec.a             # less relevance, less pull


def test_precedent_that_contradicts_a_confident_estimate_is_ignored():
    # same line type at a tenth of the price: another indemnity basis, not evidence on the level
    h = history([prec(20, 2, "air conditioning unit", 100.0, 110.0)])
    base = ca.decide(1000.0, 0.1, 1.0, [], h, ca.NEUTRAL)
    odd = ca.decide(1000.0, 0.1, 1.0, [ca.Weight(20, 2, 0.8)], h, ca.NEUTRAL)
    assert odd.a == pytest.approx(base.a, rel=0.03)


def test_quantity_scales_the_precedent_per_unit():
    h = history([prec(30, 1, "labour: senior installer", 93.0, 96.0, unit="hr")])
    base = ca.decide(250.0, 0.3, 3.0, [], h, ca.NEUTRAL)
    d = ca.decide(250.0, 0.3, 3.0, [ca.Weight(30, 1, 0.8)], h, ca.NEUTRAL)   # 3 h at 93-96
    assert base.a < d.a <= 3 * 96
    assert 3 * 93 <= d.t_q50 <= 3 * 96
    assert d.precedents[0]["lo"] == pytest.approx(279.0) and d.precedents[0]["hi"] == pytest.approx(288.0)


def test_risk_loving_never_charges_less_than_risk_neutral():
    h = history([prec(38, 1, "air conditioning unit", 1340.0)])
    for sigma in (0.05, 0.15, 0.3, 0.6):
        for ws in ([], [ca.Weight(38, 1, 0.6)]):
            n = ca.decide(1000.0, sigma, 1.0, ws, h, ca.NEUTRAL)
            lv = ca.decide(1000.0, sigma, 1.0, ws, h, ca.LOVING)
            assert lv.a >= n.a
            assert lv.p_fair <= n.p_fair + 1e-9


def test_zero_estimate_gives_zero_charge():
    assert ca.decide(0.0, 0.3, 1.0, [], history([]), ca.NEUTRAL).a == 0.0


# ---------- which earlier rounds are informative ----------

def test_relevance_is_learned_from_how_earlier_rounds_predicted_later_ones():
    other = "HOUSEHOLD CONTENTS INSURANCE / Policy wording \"HC-2025\""
    precs = []
    for g in range(1, 9):       # same wording: the line repeats at the same level
        precs.append(prec(g, 1, "vehicle costs", 78.0 + g % 2, 84.0))
    for g in range(9, 17):      # another wording: half the level
        precs.append(prec(g, 1, "vehicle costs", 38.0, 42.0, policy=other, claim=frozenset({"frost"})))
    rel = ca.learn_relevance(precs)
    assert rel["same", "same", True] > 0.8
    assert rel["same", "other", False] < 0.2
    h = history(precs)
    ws = ca.rule_weights({1: "Vehicle costs 1 pcs"}, POLICY, " ".join(CLAIM), h, game=17)[1]
    by_game = {w.game: w.w for w in ws}
    assert min(by_game[g] for g in range(1, 9)) > max(by_game.get(g, 0.0) for g in range(9, 17))


def test_rule_weights_skip_other_units_and_other_labour_grades():
    precs = [prec(5, 1, "vehicle costs", 80.0, unit="hr"), prec(6, 1, "labour: helper", 67.0, unit="hr")]
    h = history(precs)
    ws = ca.rule_weights({1: "Vehicle costs 1 pcs", 2: "Service technician hours 6 hrs"}, POLICY, "", h, game=9)
    assert ws[1] == [] and ws[2] == []


# ---------- calibration ----------

def test_calibrate_recovers_the_spread_per_class_of_stated_uncertainty():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        t = 1000 * math.exp(rng.normal(0, 0.04))
        rows.append((i, 1000.0, 0.05, t * 0.99, t * 1.01))
        t = 1000 * math.exp(rng.normal(0, 0.6))
        rows.append((i, 1000.0, 0.3, t * 0.97, t * 1.03))
    p = ca.calibrate(rows, replace(ca.NEUTRAL, tail=0.02))
    sd = {edge: s for edge, _b, s in p.cal}
    assert sd[0.12] <= 0.075
    assert 0.4 <= sd[0.4] <= 0.8
    assert ca.calibrate(rows[:10]) == ca.NEUTRAL       # too few lines: defaults


# ---------- mode "llm" ----------

def test_parse_judgement_keeps_only_valid_references():
    h = history([prec(38, 1, "air conditioning unit", 1340.0, 1445.0)])
    payload = json.dumps({"lines": [
        {"pos": 1, "refs": [{"game": 38, "line": 1, "w": 0.95, "factor": 1.0},
                            {"game": 99, "line": 1, "w": 0.9},            # unknown line
                            {"game": 38, "line": 1, "w": 7},              # weight out of range
                            {"game": "x"}]},
        {"pos": "two", "refs": []},
        {"pos": 2}]})
    out = ca.parse_judgement(payload, h)
    assert [(w.game, w.line, w.w) for w in out[1]] == [(38, 1, 0.95)]
    assert out[2] == []
    with pytest.raises(ValueError):
        ca.parse_judgement("not json", h)


def test_llm_weights_use_measured_rates_and_drop_other_units():
    precs = [prec(38, 1, "air conditioning unit", 1340.0, 1445.0),
             prec(30, 2, "installation", 90.0, 96.0, unit="hr")]
    h = history(precs)
    judgement = {1: [ca.Weight(38, 1, 1.0), ca.Weight(30, 2, 1.0)]}
    ws = ca.llm_weights({1: "Air conditioning unit 1 pcs"}, POLICY, " ".join(CLAIM), h, judgement, None, game=42)
    assert [(w.game, w.line) for w in ws[1]] == [(38, 1)]        # the hourly line cannot be compared
    assert ws[1][0].w < 1.0                                      # the judge's 1.0 is not a probability
    not_named = ca.llm_weights({1: "Air conditioning unit 1 pcs"}, POLICY, " ".join(CLAIM), h, {}, None, game=42)
    assert not_named[1][0].w < ws[1][0].w                        # naming it raises the relevance


# ---------- live path: time limit and fallbacks ----------

INVOICE = ("INVOICE\nPOS. DESCRIPTION AMOUNT UNIT TOTAL\n"
           "1 Air conditioning unit - living room\n1 pcs\n2 Installation of the unit\n1 pcs\n")


def _case(case_id="game41"):
    return Case(case_id, "RESIDENTIAL DWELLING INSURANCE\nPolicy wording \"WD-2026\"\n...",
                "Lightning surge damaged the air conditioning unit; it was replaced.",
                Path("missing.pdf"), INVOICE, None)


@pytest.fixture
def synthetic(monkeypatch):
    h = history([prec(38, 1, "air conditioning unit", 1340.0, 1445.0)])
    monkeypatch.setattr(ca, "shared_history", lambda: h)
    monkeypatch.setattr(ca, "shared_params", lambda setting, model="", gamma_loving=None:
                        ca.SETTINGS.get(setting, ca.NEUTRAL))
    monkeypatch.setattr(ca, "load_judgements", lambda **kw: {})
    monkeypatch.setattr(ca, "save_judgement", lambda *a, **kw: None)
    ca._judge_jobs.clear()
    return h


def test_slow_judge_never_delays_pricing_beyond_the_wait_limit(synthetic, monkeypatch):
    def slow_judge(*args, **kwargs):
        time.sleep(3.0)
        raise TimeoutError("judge timed out")
    monkeypatch.setattr(ca, "judge", slow_judge)
    sec = {"mode": "llm", "judge_wait_seconds": 0.5, "judge_timeout_seconds": 1}
    case = _case("game42")
    ca.begin(case, sec)
    t0 = time.monotonic()
    dec, info = ca.live_charges(case, {1: (1000.0, 0.3), 2: (400.0, 0.3)}, sec, wait=True, time_left=40.0)
    assert time.monotonic() - t0 < 1.5
    assert info["weights"] == "rule" and "judge_error" in info
    assert dec[1].a > 0 and dec[2].a > 0
    # almost no time left: no waiting at all
    case2 = _case("game43")
    ca.begin(case2, {**sec, "judge_wait_seconds": 20})
    t0 = time.monotonic()
    _dec, info = ca.live_charges(case2, {1: (1000.0, 0.3)}, {**sec, "judge_wait_seconds": 20}, wait=True,
                                 time_left=5.0)
    assert time.monotonic() - t0 < 0.5 and info["weights"] == "rule"


def test_judge_answer_in_time_is_used_and_quick_stage_does_not_wait(synthetic, monkeypatch):
    def quick_judge(lines, policy, claim, hist, model, timeout, effort=None, max_chars=60000):
        time.sleep(0.3)
        return {1: [ca.Weight(38, 1, 1.0)], 2: []}, {"model": model, "seconds": 0.3}
    monkeypatch.setattr(ca, "judge", quick_judge)
    sec = {"mode": "llm", "judge_wait_seconds": 5}
    case = _case("game44")
    ca.begin(case, sec)
    t0 = time.monotonic()
    _dec, info = ca.live_charges(case, {1: (1000.0, 0.3)}, sec, wait=False)       # quick safety-net stage
    assert time.monotonic() - t0 < 0.25 and info["weights"] == "rule"
    dec, info = ca.live_charges(case, {1: (1000.0, 0.3)}, sec, wait=True, time_left=40.0)
    assert info["weights"] == "llm" and dec[1].precedents[0]["game"] == 38


def _cfg(**agent):
    return Config({"charge_agent": agent, "llm": {"model": "gpt-6.2-sol"}}, Path("x.toml"), "test")


def _items():
    return [ItemEstimate("1", "Air conditioning unit", 1.0, "pcs", 0.99, 1.0, 1000.0, 0.3),
            ItemEstimate("2", "Installation", 1.0, "pcs", 0.99, 1.0, 400.0, 0.3)]


def test_agent_hook_off_rule_and_failure(synthetic, monkeypatch):
    timing: dict = {}
    assert _agent_charges(_items(), _case(), _cfg(mode="off"), timing, True, 40.0) == {} and timing == {}

    charges = _agent_charges(_items(), _case(), _cfg(mode="rule", risk="neutral"), timing, True, 40.0)
    assert charges[1] > 1000.0                              # the precedent at 1340-1445 lifts line 1
    assert timing["charge_agent"]["lines_set"] == 2 and timing["charge_agent"]["seconds"] < 1.0

    monkeypatch.setattr(ca, "charges", lambda *a, **kw: 1 / 0)
    timing = {}
    assert _agent_charges(_items(), _case(), _cfg(mode="rule"), timing, True, 40.0) == {}   # fixed rule prices
    assert "error" in timing["charge_agent"]


def test_price_items_takes_the_agents_charge_and_keeps_its_own_limit(monkeypatch):
    from c2f import agent
    from c2f.config import load_config
    from c2f.pricing import acceptance_limit
    cfg = load_config()
    calls = []
    monkeypatch.setattr(agent, "quote", lambda bel, p: calls.append(bel) or agent.Quote(1.0, 2.0))
    priced = price_items(_items(), cfg, {1: 1234.5})
    assert priced[0].charge_price == 1234.5
    assert priced[0].acceptance_limit == round(acceptance_limit(agent.to_belief(_items()[0], cfg), cfg.pricing), 2)
    assert (priced[1].charge_price, priced[1].acceptance_limit) == (1.0, 2.0)     # no agent charge: fixed rule
    assert len(calls) == 1                                  # the slow search ran only for line 2


# ---------- shadow run and takeover ----------

LINE = {"t_low": 100.0, "t_high": 110.0, "charges": {"A": 100.0, "B": 110.0},
        "fair": {"A": True, "B": False}, "b_bounds": {"A": [90.0, None], "B": [0.0, 95.0]}}


def _log(tmp_path, rounds):
    """rounds: (game, submitted charge, shadow charge or None)."""
    f = tmp_path / "submissions.jsonl"
    rows = []
    for g, a, shadow in rounds:
        info = {"submitted": "legacy", "shadow_charges": {"1": shadow}} if shadow is not None else {}
        rows.append({"game_id": g, "ok": True, "dry_run": False,
                     "payload": [{"index": 1, "charge_price": a, "acceptance_limit": 0.0}],
                     "timing": {"charge_agent": info} if info else {}})
    f.write_text("\n".join(json.dumps(r) for r in rows))
    return f


def test_line_revenue_prefers_the_fair_higher_charge():
    assert ca.line_revenue(100.0, LINE) > ca.line_revenue(80.0, LINE) > ca.line_revenue(130.0, LINE) > 0


def test_takeover_needs_three_winning_rounds_in_a_row(tmp_path):
    summary = {str(g): {"1": LINE} for g in range(60, 70)}
    win, loss, tie = 100.0, 130.0, 80.0
    # win, win, loss: the streak starts again
    st = ca.takeover_status(_log(tmp_path, [(60, 80.0, win), (61, 80.0, win), (62, 80.0, loss),
                                            (63, 80.0, win), (64, 80.0, win)]), summary, rounds=3)
    assert not st["active"] and st["streak"] == 2 and len(st["rounds"]) == 5
    assert st["rounds"][2]["diff"] < 0 < st["rounds"][0]["diff"]
    # a round without a difference neither counts nor breaks the streak; the third win switches
    st = ca.takeover_status(_log(tmp_path, [(60, 80.0, win), (61, 80.0, tie), (62, 80.0, win),
                                            (63, 80.0, None), (64, 80.0, win), (65, 80.0, loss)]),
                            summary, rounds=3)
    assert st["active"] and st["since_game"] == 65       # live from the game after the third win
    assert [r["game"] for r in st["rounds"]] == [60, 61, 62, 64]     # nothing is scored after the switch
    # rounds whose transactions are not public yet are not scored
    st = ca.takeover_status(_log(tmp_path, [(60, 80.0, win), (61, 80.0, win), (70, 80.0, win)]), summary, rounds=3)
    assert not st["active"] and st["streak"] == 2
    assert not ca.takeover_status(tmp_path / "missing.jsonl", summary)["active"]


def test_shadow_records_the_agents_charges_and_submits_the_fixed_rule_until_takeover(synthetic, monkeypatch):
    status = {"active": False}
    monkeypatch.setattr(ca, "shared_takeover", lambda rounds=3: status)
    cfg = _cfg(mode="rule", risk="neutral", shadow=True, takeover_rounds=3)
    timing: dict = {}
    assert _agent_charges(_items(), _case(), cfg, timing, True, 40.0) == {}         # fixed rule prices
    info = timing["charge_agent"]
    assert info["submitted"] == "legacy" and info["shadow_charges"]["1"] > 1000.0
    assert set(info["shadow_charges"]) == {"1", "2"}

    status["active"] = True
    timing = {}
    charges = _agent_charges(_items(), _case(), cfg, timing, True, 40.0)
    assert charges[1] > 1000.0 and timing["charge_agent"]["submitted"] == "agent"
    assert timing["charge_agent"]["takeover"] is True and "shadow_charges" not in timing["charge_agent"]

    timing = {}                                             # takeover = false: shadow whatever the score
    assert _agent_charges(_items(), _case(), _cfg(mode="rule", shadow=True, takeover=False), timing, True, 40.0) == {}
    assert timing["charge_agent"]["submitted"] == "legacy"


def test_default_config_has_charge_agent_off():
    from c2f.config import load_config
    sec = load_config().section("charge_agent")
    assert sec["mode"] == "off"   # switched off 2026-11-22 (user request)
