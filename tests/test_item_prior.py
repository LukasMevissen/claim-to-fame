import math

import pytest

from c2f import item_prior as ip
from c2f.item_prior import Obs


def obs(game, key, kind, lo=0.0, hi=math.inf, policy="", family=None, line=1):
    return Obs(game, line, key, family, key, 1.0, "pc", kind, lo, hi, policy)


@pytest.mark.parametrize("text, want", [
    ("Service technician hours 6.75 hrs", ("Service technician hours", 6.75, "hr")),
    ("Surge-damaged home electronics (speaker system and TV set) | 2 pcs",
     ("Surge-damaged home electronics (speaker system and TV set)", 2.0, "pc")),
    ("Copper pipe 15x1 6.4 m", ("Copper pipe 15x1", 6.4, "m")),
    ("Two-screw pipe clamp 25 mm 4 pcs", ("Two-screw pipe clamp 25 mm", 4.0, "pc")),
    ("Pipe insulation shell, 100 cm long 1 m", ("Pipe insulation shell, 100 cm long", 1.0, "m")),
    ("Silicone joints 3.6 linear m", ("Silicone joints", 3.6, "m")),
    ("Electricity for drying equipment (separately metered) 2,412.1 kWh",
     ("Electricity for drying equipment (separately metered)", 2412.1, "kwh")),
    ("Vehicle costs 1 flat rate", ("Vehicle costs", 1.0, "flat rate")),
    ("Something without a quantity", ("Something without a quantity", 1.0, "")),
])
def test_parse_line(text, want):
    assert ip.parse_line(text) == want


@pytest.mark.parametrize("desc, unit, want", [
    ("Service technician hours", "hr", ("labour: service technician", "labour")),
    ("Measurement technician", "hr", ("labour: measurement technician", "labour")),
    ("Vehicle costs – return visit", "pc", ("vehicle costs return visit", None)),
    ("Vehicle costs", "hr", ("vehicle costs", None)),
    ("Administrative and claim-processing fee", "flat rate", ("administrative or processing fee", None)),
    ("Processing flat fee", "pc", ("administrative or processing fee", None)),
    ("TV set (surge damaged)", "pc", ("tv set", None)),
    ("Profipress elbow 90° copper 15mm model 02", "pc", ("profipress elbow 90 copper 15mm", None)),
    ("Installation - living-room unit", "pc", ("installation", None)),
])
def test_line_key(desc, unit, want):
    assert ip.line_key(desc, unit) == want


def test_brackets_give_the_level_and_a_range_around_it():
    f = ip.fit("x", [obs(1, "x", "bracket", 86, 94), obs(2, "x", "bracket", 88, 96),
                     obs(3, "x", "bracket", 84, 92)])
    q10, q50, q90 = f.quantiles()
    assert 86 < q50 < 94
    assert 70 < q10 < q50 < q90 < 115
    assert f.p_zero < 0.2 and f.expected_zeros == 0


def test_lower_bounds_put_t_just_above_the_best_fair_charge():
    f = ip.fit("x", [obs(g, "x", "lower", lo) for g, lo in ((1, 91), (2, 93), (3, 96))])
    q10, q50, q90 = f.quantiles()
    assert 93 < q50 < 110          # above the bounds, but not far: the gap is small
    assert q10 > 80 and q90 < 130


def test_zero_history_and_paid_without_fair_charge():
    f = ip.fit("fee", [obs(1, "fee", "zero"), obs(2, "fee", "upper", hi=7), obs(3, "fee", "zero"),
                       obs(4, "fee", "upper", hi=9)])
    assert f.expected_zeros > 3.8      # "upper" lines are almost surely uncovered
    assert f.p_zero > 0.8
    assert "NOT indemnified" in ip.describe(f, 1, "pc")


def test_one_reduced_case_does_not_drag_the_level_down():
    base = [obs(g, "x", "bracket", 88, 96) for g in (1, 2, 3, 4)]
    q_clean = ip.fit("x", base).quantiles()[1]
    q_out = ip.fit("x", base + [obs(5, "x", "bracket", 44, 49)]).quantiles()[1]
    assert abs(math.log(q_out / q_clean)) < 0.05


def test_capped_game_is_detected_and_left_out_of_the_level():
    data = []
    for k, level in (("a", 100), ("b", 40), ("c", 10)):
        data += [obs(g, k, "bracket", level * 0.97, level * 1.03) for g in (1, 2, 3)]
        data += [obs(9, k, "bracket", level * 0.45, level * 0.55)]      # all lines halved
    assert set(ip.deflated_games(data)) == {9}
    m = ip.ItemPrior(data)
    assert all(o.game != 9 for o in m.by_key["a"])
    assert 95 < m.lookup("a").quantiles()[1] < 106


def test_level_follows_the_policy_wording():
    data = ([obs(g, "vehicle costs", "bracket", 76, 84, policy="plain") for g in (1, 2, 3, 4)]
            + [obs(g, "vehicle costs", "bracket", 37, 43, policy="frost") for g in (5, 6)])
    m = ip.ItemPrior(data)
    assert 36 < m.lookup("Vehicle costs", "pc", "frost").quantiles()[1] < 52
    assert 74 < m.lookup("Vehicle costs", "pc", "plain").quantiles()[1] < 86
    pooled = m.lookup("Vehicle costs", "pc", "unknown wording")
    assert pooled.policy == "" and pooled.n_same == 0


def test_unseen_grade_gets_the_family_level_and_unknown_lines_get_nothing():
    data = [obs(g, f"labour: {n}", "bracket", lo, lo * 1.05, family="labour")
            for g, (n, lo) in enumerate((("helper", 62), ("installer", 76), ("service technician", 88),
                                         ("senior installer", 93)), 1)]
    m = ip.ItemPrior(data)
    f = m.lookup("Roofer hours", "hr")
    assert f is not None and f.n == 0
    assert 55 < f.quantiles()[1] < 105
    assert m.lookup("Replacement of a chandelier", "pc") is None


def test_context_lists_only_lines_with_history():
    data = ([obs(g, "labour: service technician", "lower", lo, family="labour") for g, lo in ((1, 81), (2, 88), (3, 93))]
            + [obs(g, "administrative or processing fee", "zero") for g in (1, 2, 3)])
    text = ip.context({1: "Service technician hours 6.75 hrs", 2: "Processing flat fee 1 pcs",
                       3: "Replacement of a chandelier 1 pcs"}, ip.ItemPrior(data))
    assert "line 1" in text and "6.75 hr" in text and "line 2" in text and "line 3" not in text
    assert "NOT indemnified" in text
    assert ip.context({1: "Replacement of a chandelier 1 pcs"}, ip.ItemPrior(data)) == ""


def test_prob_scores_the_revealed_set():
    f = ip.fit("x", [obs(g, "x", "bracket", 88, 96) for g in (1, 2, 3)])
    assert f.prob("bracket", 85, 100) > f.prob("bracket", 40, 50) * 5
    assert f.prob("zero", 0, math.inf) < 0.3


def test_combine_moves_a_close_estimate_and_leaves_a_distant_one():
    f = ip.fit("x", [obs(g, "x", "bracket", 88, 96) for g in (1, 2, 3, 4)])
    m, s = ip.combine(600.0, 0.4, f, 6.75)                 # 6.75 h at about 92 = 620
    assert 600 < m < 630 and s < 0.4
    assert ip.combine(600.0, 0.4, f, 6.75, "replace")[0] == pytest.approx(math.exp(f.level()[0]) * 6.75)
    assert ip.combine(150.0, 0.4, f, 6.75) is None         # looks like a cap: hands off
    zero_only = ip.fit("fee", [obs(1, "fee", "zero"), obs(2, "fee", "zero")])
    assert ip.combine(20.0, 0.4, zero_only, 1) is None     # no price evidence


def test_combine_for_case_changes_only_lines_believed_indemnified(monkeypatch):
    from types import SimpleNamespace

    from c2f.extract import ItemEstimate

    data = [obs(g, "labour: service technician", "bracket", 88, 96, family="labour") for g in (1, 2, 3, 4)]
    monkeypatch.setattr(ip, "shared", lambda: ip.ItemPrior(data))
    case = SimpleNamespace(policy="", invoice_text="INVOICE\nTOTAL\n1 Service technician hours\n6.75 hrs\n"
                                                   "2 Service technician hours\n7 hrs\n")
    items = [ItemEstimate("1", "tech", 1.0, "", 0.98, 1.0, 500.0, 0.4),
             ItemEstimate("2", "tech again", 1.0, "", 0.10, 1.0, 500.0, 0.4)]
    assert ip.combine_for_case(items, case, "fuse") == 1
    assert 560 < items[0].unit_price_gross_eur < 630 and items[0].price_log_sd < 0.4
    assert items[1].unit_price_gross_eur == 500.0
