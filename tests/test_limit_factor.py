import json

from c2f import limit_factor as lf


def _line(charges, fair, lo, hi=None):
    return {"t_low": lo, "t_high": hi, "charges": charges, "fair": fair,
            "b_bounds": {k: [0.0, None] for k in charges}}


def _rec(game, lines, model="m1"):
    return {"game_id": game, "ok": True, "dry_run": False,
            "timing": {"served_by": model, "final_stage": "backup"},
            "payload": [{"index": i, "charge_price": 1.0, "acceptance_limit": b} for i, b in lines.items()]}


def test_cost_counts_penalty_acceptance_and_hidden_charges():
    s = _line({"A": 100.0, "B": 104.0, "C": 150.0, "us": 90.0}, {"A": True, "B": True, "C": False}, 104.0, 150.0)
    assert lf.insurer_cost(100.0, s, 4, us="us") == 100.0 + 1.5 * 104.0
    assert lf.insurer_cost(105.0, s, 4, us="us") == 204.0
    assert lf.insurer_cost(150.0, s, 4, us="us") == 354.0
    assert lf.insurer_cost(160.0, s, 4, us="us") == 354.0 + 160.0      # one team's charge is hidden


def test_factor_rises_when_fair_charges_sit_just_above_the_limit_and_respects_the_cap(tmp_path):
    summary = {str(g): {"1": _line({"A": 103.0, "B": 104.0, "C": 109.0}, {"A": True, "B": True, "C": True}, 109.0)}
               for g in range(1, 7)}
    recs = {g: _rec(g, {1: 100.0}) for g in range(1, 7)}
    q, s, out = tmp_path / "q.jsonl", tmp_path / "s.json", tmp_path / "f.json"
    q.write_text("\n".join(json.dumps(r) for r in recs.values()))
    s.write_text(json.dumps(summary))
    got = lf.update(path=out, quotes=q, summary=s, cap=1.06, us="US")
    assert got["factor"] == 1.04 and lf.load(out, 1.06, 1.0) == 1.04      # 1.09 is beyond the cap
    assert lf.load(out, 1.02, 1.0) == 1.02 and lf.load(tmp_path / "missing.json", 1.06, 1.05) == 1.05


def test_factor_stays_at_one_for_unfair_charges_and_ignores_other_models(tmp_path):
    bad = {"1": _line({"A": 103.0, "B": 104.0}, {"A": False, "B": False}, 0.0, 103.0)}
    good = {"1": _line({"A": 103.0, "B": 104.0}, {"A": True, "B": True}, 104.0)}
    summary = {**{str(g): good for g in range(1, 7)}, **{str(g): bad for g in range(7, 13)}}
    recs = {**{g: _rec(g, {1: 100.0}, "old") for g in range(1, 7)}, **{g: _rec(g, {1: 100.0}, "new") for g in range(7, 13)}}
    ev = lf.evaluate(recs, summary, cap=1.06, us="US")
    assert ev["games"] == list(range(7, 13)) and min(ev["costs"], key=lambda k: (ev["costs"][k], k)) == 1.0
    q, s = tmp_path / "q.jsonl", tmp_path / "s.json"
    q.write_text("\n".join(json.dumps(r) for r in list(recs.values())[:3])); s.write_text(json.dumps(summary))
    assert lf.update(path=tmp_path / "f.json", quotes=q, summary=s) is None     # too few games


def test_logged_beliefs_give_the_base_limit_and_the_estimate_caps_it():
    rec = _rec(1, {1: 999.0})
    rec["estimates"] = [{"item_id": "1", "p_zero": 0.0, "median": 100.0, "sigma": 0.1}]
    q, ceil = lf._base_limits(rec, 1 / 3)[1]
    assert 95.0 < q < 96.5 and ceil == 100.0


def test_key_fetch_retries_on_server_errors():
    from types import SimpleNamespace

    from c2f.api import QuantCoClient
    codes = iter([503, 403, 502, 200])
    c = QuantCoClient.__new__(QuantCoClient)
    c.api = {"key_poll_seconds": 5}
    c._url = lambda ep, g: "x"
    c.s = SimpleNamespace(get=lambda url, timeout: SimpleNamespace(
        status_code=next(codes), text="", json=lambda: {"decryption_key": "k"}))
    import c2f.api as api
    real, api.time.sleep = api.time.sleep, lambda s: None
    try:
        assert c.fetch_key(1) == "k"
    finally:
        api.time.sleep = real
