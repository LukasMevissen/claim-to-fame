"""play_game stage ordering: a slower text-only answer never overwrites the photo answer."""
import time
from types import SimpleNamespace

import c2f.__main__ as m
from c2f.agent import PricedItem
from c2f.config import load_config


FB = load_config().section("fallback")["charge_price"]   # stage 0: fixed fallback charge


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
    assert m.play_game(client, load_config(), SimpleNamespace(as_dict=lambda: {}), 1, None, False, zip_path)
    return [c[0]["charge_price"] for c in client.calls]


def test_backup_stands_when_full_stage_fails(monkeypatch, tmp_path):
    """The other provider's answer outranks the fast stage and is final if full never answers."""
    assert _run_backup(monkeypatch, tmp_path) == [FB, 100.0, 150.0]


def test_standby_submits_only_the_backup_answer(monkeypatch, tmp_path):
    """A second machine never sends fixed or quick prices."""
    assert _run_backup(monkeypatch, tmp_path, role="standby") == [150.0]
