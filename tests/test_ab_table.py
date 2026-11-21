"""ab_table rows, market.refresh (sync -> table -> checks) and the scheduler hook."""

import csv
import json
import math
from types import SimpleNamespace

import c2f.__main__ as m
from c2f import leaderboard, market
from c2f.ab_table import ab_table
from c2f.opponents import Tx
from c2f.payoff import insurer_pays

A = {"x": {1: 100.0, 2: 50.0}, "y": {1: 150.0, 2: 40.0}, "z": {1: 90.0, 2: 0.0}, "off": {1: 0.0, 2: 0.0}}
B = {"x": {1: 120.0, 2: 45.0}, "y": {1: 95.0, 2: 60.0}, "z": {1: 200.0, 2: 0.0}, "off": {1: 0.0, 2: 0.0}}
T = {1: 110.0, 2: 0.0}


def play(game=1):
    return [Tx(game, j, i, r, A[i][j] <= B[r][j], insurer_pays(A[i][j], B[r][j], T[j]))
            for j in T for i in A for r in B if i != r]


def rows_by(rows):
    return {(r.team, r.round, r.line_item): r for r in rows}


def test_one_row_per_team_and_item():
    rows = ab_table(play())
    assert len(rows) == len(A) * len(T)


def test_recovered_a_b_and_t_bounds():
    r = rows_by(ab_table(play()))
    x, y, z = r[("x", 1, 1)], r[("y", 1, 1)], r[("z", 1, 1)]
    assert (x.a, x.a_vs_t) == (100.0, "fair")           # rejected by y and paid 1.5a
    assert (y.a, y.a_vs_t) == (150.0, "fraud")          # rejected by x, paid 0; z paid 150
    assert z.a == 90.0
    assert x.b == 90.0 and x.b_high == 150.0            # x accepted z's 90, rejected y's 150
    assert z.b == 150.0 and z.b_high == math.inf        # z accepted everything it saw
    assert (x.t_low, x.t_high, x.cap) == (100.0, 150.0, None)   # same on every team's row
    assert r[("off", 1, 1)].absent and r[("off", 1, 1)].a == 0.0


def _game_file(d, game, txs):
    d.mkdir(parents=True, exist_ok=True)
    items = [{"issuer": x.issuer, "reviewer": x.reviewer, "line_item_index": x.item,
              "accepted": x.accepted, "amount": x.amount} for x in txs]
    (d / f"game{game}.json").write_text(json.dumps({"game_id": game, "items": items}))


def _paths(tmp_path):
    return dict(tx_dir=tmp_path / "tx", quotes=tmp_path / "s.jsonl", table=tmp_path / "ab.csv",
                checks=tmp_path / "checks.csv", belief=tmp_path / "belief.json",
                summary=tmp_path / "summary.json")


def test_refresh_builds_table_and_checks_even_if_sync_fails(tmp_path, monkeypatch):
    p = _paths(tmp_path)
    _game_file(p["tx_dir"], 1, play())
    p["quotes"].write_text(json.dumps({"game_id": 1, "ok": True, "dry_run": False, "payload": [
        {"index": 1, "charge_price": 150.0, "acceptance_limit": 95.0},
        {"index": 2, "charge_price": 40.0, "acceptance_limit": 60.0}]}) + "\n")

    def boom(*a, **k):
        raise ConnectionError("leaderboard down")
    monkeypatch.setattr(leaderboard, "sync", boom)
    logs = []
    rep = market.refresh(refit=False, log=logs.append, **p)
    assert rep["errors"] and "sync" in rep["errors"][0]
    assert rep["rows"] == {1: 8}
    assert not [f for f in rep["findings"] if f.severity == "error"]
    rows = list(csv.DictReader(p["table"].open()))
    item1 = [r for r in rows if r["line_item"] == "1"]
    assert all((r["our_a"], r["our_b"]) == ("150.0", "95.0") for r in item1)  # per round and item
    assert p["checks"].exists()


def test_refresh_is_noop_when_nothing_changed(tmp_path, monkeypatch):
    p = _paths(tmp_path)
    _game_file(p["tx_dir"], 1, play())
    monkeypatch.setattr(market, "_refit", lambda d, path: [path.write_text("{}"), 1][1:])
    rep = market.refresh(fetch=False, log=lambda s: None, **p)
    assert "rows" in rep and rep["belief"] == [1]
    rep = market.refresh(fetch=False, log=lambda s: None, **p)
    assert "rows" not in rep and "belief" not in rep


def test_refit_postponed_close_to_deadline(tmp_path, monkeypatch):
    import time
    p = _paths(tmp_path)
    _game_file(p["tx_dir"], 1, play())
    monkeypatch.setattr(market, "_refit", lambda d, path: (_ for _ in ()).throw(AssertionError))
    rep = market.refresh(fetch=False, deadline=time.time() + 5, log=lambda s: None, **p)
    assert "belief" not in rep and not rep["errors"]


def test_sync_skips_stored_games_and_imports_legacy(tmp_path, monkeypatch):
    d, legacy = tmp_path / "tx", tmp_path / "market"
    legacy.mkdir()
    rows = [{"issuer": x.issuer, "reviewer": x.reviewer, "line_item_index": x.item,
             "accepted": x.accepted, "amount": x.amount} for x in play(2)]
    (legacy / "game_002.json").write_text(json.dumps(rows))
    _game_file(d, 1, play())
    fetched = []
    monkeypatch.setattr(leaderboard, "LEGACY_DIR", legacy)
    monkeypatch.setattr(leaderboard, "completed_games", lambda: [1, 2, 3])
    monkeypatch.setattr(leaderboard, "fetch_game", lambda g, pause, deadline=None: fetched.append(g) or
                        {"game_id": g, "complete": True, "items": rows})
    monkeypatch.setattr(leaderboard.time, "sleep", lambda s: None)
    new = leaderboard.sync(d, log=lambda s: None)
    assert fetched == [3] and new == [2, 3]
    assert json.loads((d / "game2.json").read_text())["complete"] is True


def test_scheduler_hook_never_raises(monkeypatch, capsys):
    def boom(**k):
        raise RuntimeError("anything")
    monkeypatch.setattr(market, "refresh", boom)
    m._market_refresh(SimpleNamespace(no_sync=False), deadline=0.0)
    assert "refresh failed" in capsys.readouterr().out
    m._market_refresh(SimpleNamespace(no_sync=True), deadline=0.0)    # opt-out: not called
