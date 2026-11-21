"""Payoff consistency checks on synthetic transactions generated from the payoff table."""

import json

from c2f.consistency import check, item_facts, our_quotes
from c2f.opponents import Tx
from c2f.payoff import insurer_pays


def play(a: dict, b: dict, t: dict, c: dict | None = None, game: int = 1) -> list[Tx]:
    """Every ordered pair on every item, amounts from the payoff table."""
    out = []
    for j in t:
        for i in a:
            for r in b:
                if i == r:
                    continue
                cap = None if c is None else c[j]
                out.append(Tx(game, j, i, r, a[i][j] <= b[r][j],
                              insurer_pays(a[i][j], b[r][j], t[j], cap)))
    return out


A = {"x": {1: 100.0, 2: 50.0}, "y": {1: 150.0, 2: 40.0}, "z": {1: 90.0, 2: 0.0}}
B = {"x": {1: 120.0, 2: 45.0}, "y": {1: 95.0, 2: 60.0}, "z": {1: 200.0, 2: 0.0}}
T = {1: 110.0, 2: 0.0}


def kinds(found, severity="error"):
    return sorted({f.kind for f in found if f.severity == severity})


def test_clean_game_has_no_errors():
    assert kinds(check(play(A, B, T))) == []
    assert kinds(check(play(A, B, T)), "warn") == []


def test_t_bounds_from_fair_and_fraud():
    it = item_facts(1, 1, [x for x in play(A, B, T) if x.item == 1])
    # x (100) rejected by y (b 95) and paid 150: fair -> t >= 100.
    # y (150) rejected by x (b 120) and paid 0: fraud; accepted by z at 150 -> t <= 150.
    assert it.t_low == 100.0 and it.t_high == 150.0 and it.cap is None


def test_amount_not_in_payoff_set_is_flagged():
    txs = play(A, B, T)
    bad = [Tx(x.game, x.item, x.issuer, x.reviewer, x.accepted, x.amount * 1.1)
           if (x.issuer, x.reviewer, x.item) == ("x", "z", 1) else x for x in txs]
    assert "accept_vs_reject_a" in kinds(check(bad))


def test_issuer_a_must_agree_across_reviewers():
    txs = play(A, B, T) + [Tx(1, 1, "y", "w", True, 140.0), Tx(1, 1, "w", "y", True, 0.0)]
    assert "accepted_amounts_differ" in kinds(check(txs))


def test_fair_and_fraud_rejection_of_one_charge():
    txs = [x if (x.issuer, x.reviewer, x.item) != ("x", "y", 1) else
           Tx(1, 1, "x", "y", False, 0.0) for x in play(A, B, T)]
    txs.append(Tx(1, 1, "x", "w", False, 150.0))
    assert "fair_and_fraud" in kinds(check(txs))


def test_t_bounds_contradiction():
    # fair charge 100 (paid 150 on rejection) but another accepted fraud charge of 80
    txs = [Tx(1, 1, "p", "q", False, 150.0), Tx(1, 1, "q", "p", True, 80.0),
           Tx(1, 1, "q", "r", False, 0.0), Tx(1, 1, "p", "r", False, 150.0),
           Tx(1, 1, "r", "p", True, 0.0), Tx(1, 1, "r", "q", True, 0.0)]
    assert "t_bounds_contradict" in kinds(check(txs))


def test_b_bounds_contradiction():
    # reviewer r accepts a = 100 (exact: wrongly rejected elsewhere) but rejects a = 90 (exact)
    txs = [Tx(1, 1, "p", "r", True, 100.0), Tx(1, 1, "p", "q", False, 150.0),
           Tx(1, 1, "q", "r", False, 135.0), Tx(1, 1, "q", "p", False, 135.0),
           Tx(1, 1, "r", "p", True, 0.0), Tx(1, 1, "r", "q", True, 0.0)]
    assert "b_bounds_contradict" in kinds(check(txs))


# t = 50, c = 200. x is fair; y, z, w overcharge. z (b = 260) accepts w's 250 (paid c = 200)
# and rejects y's 500, which is visible only as 200 (x accepted it, paid min(500, 200)).
# Read as uncapped, z would have accepted 200 and rejected 200: only a cap hit on y explains it.
CAP_A = {"x": {1: 50.0}, "y": {1: 500.0}, "z": {1: 300.0}, "w": {1: 250.0}}
CAP_B = {"x": {1: 600.0}, "y": {1: 40.0}, "z": {1: 260.0}, "w": {1: 0.0}}


def test_cap_hit_reveals_c_and_is_not_an_error():
    txs = play(CAP_A, CAP_B, {1: 50.0}, {1: 200.0})
    found = check(txs)
    assert kinds(found) == []
    it = item_facts(1, 1, txs)
    assert it.cap == 200.0 and "y" in it.capped
    assert it.t_low == 50.0 and it.t_high == 50.0 and not it.t_high_strict   # t <= c/4
    assert "cap_hit" in kinds(found, "info")


def test_paid_above_revealed_cap():
    txs = [x if (x.issuer, x.reviewer) != ("z", "x") else Tx(1, 1, "z", "x", True, 300.0)
           for x in play(CAP_A, CAP_B, {1: 50.0}, {1: 200.0})]
    assert "paid_above_cap" in kinds(check(txs))


def test_incomplete_matrix_and_duplicates():
    txs = play(A, B, T)
    assert "incomplete_matrix" in kinds(check(txs[1:]), "warn")
    dup = txs + [Tx(txs[0].game, txs[0].item, txs[0].issuer, txs[0].reviewer,
                    txs[0].accepted, txs[0].amount + 1)]
    assert "duplicate_conflict" in kinds(check(dup))


def test_our_quotes_last_live_record(tmp_path):
    log = tmp_path / "s.jsonl"
    recs = [
        {"game_id": 1, "ok": True, "dry_run": False, "payload": [{"index": 1, "charge_price": 1, "acceptance_limit": 1}]},
        {"game_id": 1, "ok": True, "dry_run": False, "payload": [{"index": 1, "charge_price": 100.0, "acceptance_limit": 120.0},
                                                                   {"index": 2, "charge_price": 50.0, "acceptance_limit": 45.0}]},
        {"game_id": 1, "ok": True, "dry_run": True, "payload": [{"index": 1, "charge_price": 9, "acceptance_limit": 9}]},
        {"game_id": 1, "ok": False, "dry_run": False, "error": "x"},
        {"game_id": 2, "ok": True, "dry_run": True, "payload": [{"index": 1, "charge_price": 9, "acceptance_limit": 9}]},
    ]
    log.write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n")
    q = our_quotes(log)
    assert q == {(1, 1): (100.0, 120.0), (1, 2): (50.0, 45.0)}


def test_our_quotes_match_transactions():
    q = {(1, 1): (100.0, 120.0), (1, 2): (50.0, 45.0)}
    assert kinds(check(play(A, B, T), q, us="x")) == []
    wrong_a = {(1, 1): (101.0, 120.0), (1, 2): (50.0, 45.0)}
    assert "our_a_mismatch" in kinds(check(play(A, B, T), wrong_a, us="x"))
    wrong_b = {(1, 1): (100.0, 80.0), (1, 2): (50.0, 45.0)}     # we accepted z's 90 with b 80?
    assert "our_b_mismatch" in kinds(check(play(A, B, T), wrong_b, us="x"))
    missing_item = {(1, 1): (100.0, 120.0)}
    assert "our_item_mismatch" in kinds(check(play(A, B, T), missing_item, us="x"))


def test_unlogged_play_is_warned():
    assert "our_unlogged_play" in kinds(check(play(A, B, T), {}, us="x"), "warn")
