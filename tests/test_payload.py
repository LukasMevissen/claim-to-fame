from c2f.agent import PricedItem, submission_payload


def _p(item_id, a, b):
    return PricedItem(item_id, "", {}, a, b)


def test_pos_numbers_become_indices():
    out = submission_payload([_p("1", 10.004, 5), _p("Pos. 2", 3, 2), _p("3a", 1, 1)])
    assert [o["index"] for o in out] == [1, 2, 3]
    assert out[0] == {"index": 1, "charge_price": 10.0, "acceptance_limit": 5.0}


def test_unparseable_or_duplicate_ids_fall_back_to_order():
    out = submission_payload([_p("A", 1, 1), _p("A", 1, 1)])
    assert [o["index"] for o in out] == [1, 2]


def test_invalid_money_becomes_zero():
    out = submission_payload([_p("1", float("nan"), -3)])
    assert out[0]["charge_price"] == 0.0 and out[0]["acceptance_limit"] == 0.0


def test_fallback_fills_lines_the_model_missed():
    out = submission_payload([_p("2", 50, 40)], positions=[1, 2, 3], fallback=(20, 20))
    assert out == [
        {"index": 1, "charge_price": 20.0, "acceptance_limit": 20.0},
        {"index": 2, "charge_price": 50.0, "acceptance_limit": 40.0},
        {"index": 3, "charge_price": 20.0, "acceptance_limit": 20.0},
    ]


def test_fallback_covers_every_line_when_model_failed():
    out = submission_payload([], positions=[1, 2], fallback=(20, 10))
    assert [(o["index"], o["charge_price"], o["acceptance_limit"]) for o in out] == [(1, 20, 10), (2, 20, 10)]


def test_invoice_positions_follow_running_sequence():
    from c2f.case_io import invoice_positions

    text = "\n".join([
        "Rechnung Nr. 4711", "12345 Musterstadt", "21.11.2026",
        "Pos. Menge Einheit Bezeichnung",
        "1 2 Stk Kupferrohr 15 mm", "2 1 Std Arbeitszeit Monteur", "3. Anfahrtspauschale",
        "Summe netto",
    ])
    assert invoice_positions(text) == [1, 2, 3]
    assert invoice_positions("") == []


def test_invoice_positions_skip_quantity_lines():
    # Layout of the case 0 invoice: description wraps, quantity on its own line.
    from c2f.case_io import invoice_positions

    text = "\n".join([
        "POS. DESCRIPTION AMOUNT UNIT TOTAL",
        "1 Leak detection and opening of the bathroom wall next", "to the washbasin", "2 hrs",
        "2 Replacement of split copper supply pipe section, 15", "mm, incl. fittings", "3 pcs",
        "3 Closing the wall opening, re-plastering", "5 pcs",
    ])
    assert invoice_positions(text) == [1, 2, 3]
