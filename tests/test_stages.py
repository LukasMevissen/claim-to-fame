"""play_game stage ordering: a slower text-only answer never overwrites the photo answer."""
import time
from types import SimpleNamespace

import c2f.__main__ as m
from c2f.agent import PricedItem
from c2f.config import load_config


FB = load_config().section("fallback")["charge_price"]   # stage 0: fixed fallback charge


def _backup_cfg(**runtime):
    """The live config with a backup stage switched on (the live default has none since
    2026-11-22), so the backup tests do not depend on it."""
    base = load_config()
    def section(name):
        s = base.section(name)
        if name == "llm":
            s = {**s, "backup": "anthropic:test-model", "prefer_backup": True}
        if name == "runtime":
            s = {**s, **runtime}
        return s
    return SimpleNamespace(section=section)


class FakeClient:
    def __init__(self):
        self.calls = []

    def fetch_key(self, game_id, start_time=None):
        return "k"

    def submit(self, game_id, payload):
        self.calls.append(payload)
        return payload


def _run(monkeypatch, tmp_path, text_s, photo_s, image=True):
    case = SimpleNamespace(image=object() if image else None, invoice_text="")
    monkeypatch.setattr(m, "decrypt", lambda z, k, d: d)
    monkeypatch.setattr(m, "load_case", lambda root, cid: case)
    monkeypatch.setattr(m, "invoice_positions", lambda text: [1, 2])
    monkeypatch.setattr(m, "_log", lambda rec: None)

    def fake_price(case, cfg, use_cache, budget, with_image):
        time.sleep(photo_s if with_image else text_s)
        price = 200.0 if with_image else 100.0
        return [PricedItem("1", "", {"p_zero": 0, "median": price}, price, price)], {}

    monkeypatch.setattr(m, "price_case", fake_price)
    monkeypatch.setattr(m, "price_case_backup", lambda case, cfg, budget: ([], {}))
    monkeypatch.delenv("C2F_ROLE", raising=False)
    zip_path = tmp_path / "case.zip"
    zip_path.write_bytes(b"")
    client = FakeClient()
    st = SimpleNamespace(as_dict=lambda: {})
    assert m.play_game(client, load_config(), st, 1, None, False, zip_path)
    return client.calls


def test_photo_after_text_overwrites(monkeypatch, tmp_path):
    calls = _run(monkeypatch, tmp_path, text_s=0.05, photo_s=0.2)
    assert [c[0]["charge_price"] for c in calls] == [FB, 100.0, 200.0]


def test_text_after_photo_is_not_submitted(monkeypatch, tmp_path):
    calls = _run(monkeypatch, tmp_path, text_s=0.2, photo_s=0.05)
    assert [c[0]["charge_price"] for c in calls] == [FB, 200.0]


def test_no_photo_still_runs_fast_and_full(monkeypatch, tmp_path):
    # Without a photo the fast stage is still the safety net for the slower full stage.
    calls = _run(monkeypatch, tmp_path, text_s=0.05, photo_s=0.2, image=False)
    assert [c[0]["charge_price"] for c in calls] == [FB, 100.0, 200.0]


def _run_backup(monkeypatch, tmp_path, role=None):
    def backup(case, cfg, budget):
        time.sleep(0.1)
        return [PricedItem("1", "", {"p_zero": 0, "median": 150.0}, 150.0, 150.0)], {}

    def fake_price(case, cfg, use_cache, budget, with_image):
        if with_image:
            return [], {"model_errors": ["down"]}
        return [PricedItem("1", "", {"p_zero": 0, "median": 100.0}, 100.0, 100.0)], {}

    case = SimpleNamespace(image=None, invoice_text="")
    monkeypatch.setattr(m, "decrypt", lambda z, k, d: d)
    monkeypatch.setattr(m, "load_case", lambda root, cid: case)
    monkeypatch.setattr(m, "invoice_positions", lambda text: [1])
    monkeypatch.setattr(m, "_log", lambda rec: None)
    monkeypatch.setattr(m, "price_case", fake_price)
    monkeypatch.setattr(m, "price_case_backup", backup)
    if role:
        monkeypatch.setenv("C2F_ROLE", role)
    else:
        monkeypatch.delenv("C2F_ROLE", raising=False)
    zip_path = tmp_path / "case.zip"
    zip_path.write_bytes(b"")
    client = FakeClient()
    assert m.play_game(client, _backup_cfg(), SimpleNamespace(as_dict=lambda: {}), 1, None, False, zip_path)
    return [c[0]["charge_price"] for c in client.calls]


def test_backup_stands_when_full_stage_fails(monkeypatch, tmp_path):
    """The other provider's answer outranks the fast stage and is final if full never answers."""
    assert _run_backup(monkeypatch, tmp_path) == [FB, 100.0, 150.0]


def test_standby_submits_only_the_backup_answer(monkeypatch, tmp_path):
    """A second machine never sends fixed or quick prices."""
    assert _run_backup(monkeypatch, tmp_path, role="standby") == [150.0]


def test_preferred_backup_is_not_overwritten_by_a_later_full_stage(monkeypatch, tmp_path):
    """[llm].prefer_backup: the backup provider's answer stands even if full answers later."""
    def backup(case, cfg, budget):
        time.sleep(0.05)
        return [PricedItem("1", "", {"p_zero": 0, "median": 150.0}, 150.0, 150.0)], {}

    def fake_price(case, cfg, use_cache, budget, with_image):
        time.sleep(0.2 if with_image else 0.0)
        v = 200.0 if with_image else 100.0
        return [PricedItem("1", "", {"p_zero": 0, "median": v}, v, v)], {}

    assert _backup_cfg().section("llm").get("prefer_backup") is True
    case = SimpleNamespace(image=None, invoice_text="")
    monkeypatch.setattr(m, "decrypt", lambda z, k, d: d)
    monkeypatch.setattr(m, "load_case", lambda root, cid: case)
    monkeypatch.setattr(m, "invoice_positions", lambda text: [1])
    monkeypatch.setattr(m, "_log", lambda rec: None)
    monkeypatch.setattr(m, "price_case", fake_price)
    monkeypatch.setattr(m, "price_case_backup", backup)
    monkeypatch.delenv("C2F_ROLE", raising=False)
    zip_path = tmp_path / "case.zip"
    zip_path.write_bytes(b"")
    client = FakeClient()
    assert m.play_game(client, _backup_cfg(), SimpleNamespace(as_dict=lambda: {}), 1, None, False, zip_path)
    assert [c[0]["charge_price"] for c in client.calls] == [FB, 100.0, 150.0]


def test_standby_sends_after_the_hold(monkeypatch, tmp_path):
    """Second machine: only the backup answer, and not before [runtime].standby_hold_seconds."""
    case = SimpleNamespace(image=None, invoice_text="")
    monkeypatch.setattr(m, "decrypt", lambda z, k, d: d)
    monkeypatch.setattr(m, "load_case", lambda root, cid: case)
    monkeypatch.setattr(m, "invoice_positions", lambda text: [1])
    monkeypatch.setattr(m, "_log", lambda rec: None)
    monkeypatch.setattr(m, "price_case", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no fast/full stage")))
    monkeypatch.setattr(m, "price_case_backup",
                        lambda case, cfg, budget: ([PricedItem("1", "", {"p_zero": 0, "median": 300.0}, 300.0, 300.0)], {}))
    monkeypatch.setenv("C2F_ROLE", "standby")
    cfg = _backup_cfg(standby_hold_seconds=0.6)
    zip_path = tmp_path / "case.zip"
    zip_path.write_bytes(b"")
    sent = []

    class Client(FakeClient):
        def submit(self, game_id, payload):
            sent.append(time.time())
            return super().submit(game_id, payload)

    client, t0 = Client(), time.time()
    assert m.play_game(client, cfg, SimpleNamespace(as_dict=lambda: {}), 1, t0, False, zip_path)
    assert [c[0]["charge_price"] for c in client.calls] == [300.0]      # no fixed, no fast stage
    assert sent[0] - t0 >= 0.6


def test_no_sync_keeps_the_light_refresh(monkeypatch):
    """--no-sync with [runtime] sync_without_refit: the refresh runs, without the belief refit."""
    import c2f.market as market
    calls = []
    monkeypatch.setattr(market, "refresh", lambda **kw: calls.append(kw) or {})
    args = SimpleNamespace(no_sync=True)
    on = SimpleNamespace(section=lambda name: {"sync_without_refit": True})
    off = SimpleNamespace(section=lambda name: {})
    m._market_refresh(args, 123.0, off)
    m._market_refresh(args, 123.0)
    assert calls == []
    m._market_refresh(args, 123.0, on)
    assert len(calls) == 1 and calls[0]["refit"] is False and calls[0]["deadline"] == 123.0
    m._market_refresh(SimpleNamespace(no_sync=False), 5.0, off)
    assert calls[1]["refit"] is True
