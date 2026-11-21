import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dashboard", ROOT / "scripts" / "dashboard.py")
dash = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dash)

TODO = """# Team ToDo

intro text

## Alice
- [ ] write tests #qc
- [x] ship it

## Bob
- [ ] review `game 3`
"""


def test_parse_todo_sections_done_and_tags():
    items = dash.parse_todo(TODO)
    assert [(i["owner"], i["text"], i["done"]) for i in items] == [
        ("Alice", "write tests #qc", False),
        ("Alice", "ship it", True),
        ("Bob", "review `game 3`", False),
    ]
    assert items[0]["tags"] == ["qc"]


def test_edit_todo_toggle_only_touches_that_line():
    out = dash.edit_todo(TODO, "toggle", "Alice", "write tests #qc", done=True)
    assert "- [x] write tests #qc" in out
    assert out.replace("- [x] write tests #qc", "- [ ] write tests #qc") == TODO
    with pytest.raises(KeyError):
        dash.edit_todo(TODO, "toggle", "Bob", "write tests #qc", done=True)


def test_edit_todo_add_to_existing_and_new_section():
    out = dash.edit_todo(TODO, "add", "Alice", "  new   task ")
    items = dash.parse_todo(out)
    alice = [i["text"] for i in items if i["owner"] == "Alice"]
    assert alice == ["write tests #qc", "ship it", "new task"]
    out = dash.edit_todo(TODO, "add", "Carol", "label game 0")
    assert dash.parse_todo(out)[-1] == {"owner": "Carol", "text": "label game 0", "done": False,
                                        "tags": [], "source": "TODO.md"}


def test_scores_from_transactions_is_zero_sum(tmp_path):
    rows = [{"issuer": "A", "reviewer": "B", "line_item_index": 1, "accepted": True, "amount": 100.0},
            {"issuer": "B", "reviewer": "A", "line_item_index": 1, "accepted": False, "amount": 30.0}]
    (tmp_path / "game1.json").write_text(json.dumps({"game_id": 1, "items": rows}))
    sc = dash.scores_from_transactions(tmp_path)
    assert sc["games"] == [1]
    assert sc["teams"] == {"A": [70.0], "B": [-70.0]}


def test_labels():
    base = {"absent_rate": 0.0, "over_covered": 0.6, "markup": 1.2}
    assert dash.craftsman_label(base)[0] == "Overcharger"
    assert dash.craftsman_label(base | {"markup": 0.6, "over_covered": 0.2})[0] == "Undercutter"
    assert dash.craftsman_label(base | {"markup": 0.9, "over_covered": 0.2})[0] == "Fair pricer"
    assert dash.craftsman_label(base | {"absent_rate": 0.5})[0] == "Often absent"
    field = {"accept_rate": 0.42, "wrong_accept_share": 0.3, "penalty_rate": 0.1}
    m = {"accept_rate": 0.30, "wrong_accept_share": 0.2, "penalty_rate": 0.2}
    lab, why = dash.insurer_label(m, field)
    assert lab == "Strict gatekeeper" and "fair bills" in why
    assert dash.insurer_label(m | {"accept_rate": 0.5}, field)[0] == "Lenient payer"
