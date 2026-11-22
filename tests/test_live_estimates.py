"""Belief behind every submitted quote: reconstruction from (a, b) and the per-game record."""
import json
from types import SimpleNamespace

import pytest

import c2f.__main__ as m
from c2f import live_estimates as le
from c2f.agent import PricedItem
from c2f.belief import TBelief
from c2f.config import load_config
from c2f.pricing import OpponentType, PricingParams, charge_price

RULE = le.rule_of({"pricing": {"charge_min_ratio": 0.87, "charge_conf_sigma": 0.12, "charge_conf_ratio": 0.95,
                               "acceptance_quantile": 0.3333333333, "n_candidates": 60},
                   "belief": {"min_sigma": 0.2, "confident_below": 0.12, "confident_min_sigma": 0.1},
                   "fallback": {"charge_price": 350.0, "acceptance_limit": 0.0}})
FLOOR_ONLY = {s: set() for s in le.sigma_grid(RULE)}     # the charge floor binds for every sigma


def test_reconstructs_belief_from_quotes_on_the_floor():
    a, b = round(0.87 * 120, 2), le.limit_of(120, 0.3, 0.1, RULE, 1.0)
    r = le.reconstruct(a, b, RULE, FLOOR_ONLY)
    assert r["status"] in ("exact", "median_exact") and r["median"] == 120.0
    assert any(s["sigma"] == 0.3 and s["p_zero"] == 0.1 for s in r["solutions"])


def test_confident_estimate_uses_the_higher_floor():
    a, b = round(0.95 * 400, 2), le.limit_of(400, 0.1, 0.05, RULE, 1.0)
    r = le.reconstruct(a, b, RULE, FLOOR_ONLY)
    assert (r["median"], r["sigma"], r["p_zero"]) == (400.0, 0.1, 0.05)


def test_zero_limit_identifies_the_median_only():
    r = le.reconstruct(round(0.87 * 90, 2), 0.0, RULE, FLOOR_ONLY)
    assert r["status"] == "median_only" and r["median"] == 90.0 and r["p_zero"] == ">=1/3" and r["sigma"] is None


def test_fallback_price_and_no_charge_are_not_estimates():
    assert le.reconstruct(350.0, 0.0, RULE, FLOOR_ONLY)["status"] == "fallback_price"
    assert le.reconstruct(0.0, 0.0, RULE, FLOOR_ONLY)["status"] == "no_charge"


def test_search_index_matches_the_pricing_search():
    """The vectorised search picks the same point as pricing.charge_price."""
    opps = (OpponentType(0.2, "absent"), OpponentType(0.1, "accept_all"), OpponentType(0.7, "noisy", mu=-0.1, s=0.3))
    rule = {**RULE, "n_cand": 12, "n_t": 20, "n_draws": 5, "min_ratio": 0.0, "conf_sigma": 0.0}
    params = PricingParams(n_t_grid=20, n_candidates=12, n_opp_draws=5, opponents=opps)
    pops = tuple((o.weight, o.kind, o.mu, o.s) for o in opps)
    for sigma in (0.2, 0.5, 0.9):
        i = le.search_index(sigma, rule, pops)
        assert le._search_points(sigma, rule)[i] * 1000 == pytest.approx(charge_price(TBelief(1000, sigma), params))


def test_reconstruction_on_a_search_point():
    idx = 27
    g = le._search_points(0.5, RULE)[idx]
    a, b = round(g * 200, 2), le.limit_of(200, 0.5, 0.15, RULE, 1.0)
    allowed = {**FLOOR_ONLY, 0.5: {idx}}
    r = le.reconstruct(a, b, RULE, allowed)
    assert r["median"] == 200.0 and any(s["sigma"] == 0.5 and s["p_zero"] == 0.15 for s in r["solutions"])


def _play(monkeypatch, tmp_path, dry_run):
    case = SimpleNamespace(image=None, invoice_text="")
    monkeypatch.setattr(m, "decrypt", lambda z, k, d: d)
    monkeypatch.setattr(m, "load_case", lambda root, cid: case)
    monkeypatch.setattr(m, "invoice_positions", lambda text: [1])
    monkeypatch.setattr(m, "_log", lambda rec: None)
    monkeypatch.delenv("C2F_ROLE", raising=False)

    def fake_price(case, cfg, use_cache, budget, with_image):
        med = 200.0 if with_image else 100.0
        return [PricedItem("1", "Pipe repair", {"p_zero": 0.1, "median": med, "sigma": 0.3}, 0.87 * med, 0.8 * med)], \
            {"model": "full-model" if with_image else "fast-model"}

    monkeypatch.setattr(m, "price_case", fake_price)
    monkeypatch.setattr(m, "price_case_backup", lambda case, cfg, budget: ([], {}))
    client = SimpleNamespace(fetch_key=lambda g, s=None: "k", submit=lambda g, p: p)
    zip_path = tmp_path / "case.zip"
    zip_path.write_bytes(b"")
    st = SimpleNamespace(as_dict=lambda: {"config_sha": "abc"})
    assert m.play_game(client, load_config(), st, 7, None, dry_run, zip_path)


def test_every_stage_is_recorded_per_game(monkeypatch, tmp_path):
    _play(monkeypatch, tmp_path, dry_run=False)
    doc = json.loads((m.LIVE_DIR / "game7__primary.json").read_text(encoding="utf-8"))
    stages = {s["stage"]: s for s in doc["stages"]}
    assert set(stages) == {"fast", "full"}                     # both answers kept, not only the winner
    assert stages["full"]["lines"][0] == {"index": 1, "description": "Pipe repair", "median": 200.0,
                                          "sigma": 0.3, "p_zero": 0.1, "a": 174.0, "b": 160.0}
    assert stages["full"]["model"] == "full-model"
    games = le.live_files(m.LIVE_DIR)
    assert games[7]["source"] == "logged" and games[7]["lines"][0]["median"] in (100.0, 200.0)


def test_dry_run_writes_no_live_record(monkeypatch, tmp_path):
    _play(monkeypatch, tmp_path, dry_run=True)
    assert not m.LIVE_DIR.exists()


def test_charge_agent_answer_is_kept_per_stage():
    priced = [PricedItem("2", "Tiles", {"p_zero": 0.0, "median": 50.0, "sigma": 0.2}, 45.0, 40.0)]
    r = m._stage_record("backup", {"submitted": True}, priced, {"model": "x", "charge_agent": {"shadow_charges": {"2": 47.0}}})
    assert r["charge_agent"] == {"shadow_charges": {"2": 47.0}} and r["lines"][0]["index"] == 2


def test_stage_that_misses_the_deadline_is_still_recorded(tmp_path):
    from concurrent.futures import Future
    priced = [PricedItem("1", "Pipe repair", {"p_zero": 0.1, "median": 200.0, "sigma": 0.3}, 174.0, 160.0)]
    detail = [m._stage_record("fast", {"submitted": True, "since_start_s": 9.0}, priced, {"model": "fast-model"})]
    fut = Future()
    fut.add_done_callback(lambda f: m._record_late(f, "backup", lambda: 61.5, detail,
                                                   lambda: m._save_live(7, False, {}, detail, tmp_path)))
    fut.set_result((priced, {"model": "late-model"}))
    doc = json.loads((tmp_path / "game7__primary.json").read_text(encoding="utf-8"))
    late = doc["stages"][-1]
    assert (late["stage"], late["late"], late["submitted"], late["model"]) == ("backup", True, False, "late-model")
    assert doc["final_stage"] == "fast" and late["lines"][0]["median"] == 200.0
