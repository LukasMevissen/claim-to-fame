"""Decision table: which share of the estimate to charge and to accept, learned from the
replay of every earlier game (approach B of the overhaul, docs/research/overhaul.md).

The pricing rule (c2f/pricing.py) turns an estimate m into a = max(search, 0.87 m) and
b = 1/3 quantile x factor. Its constants were set on games 1-12 with another model and
another prompt. What the estimates of the current analyst are worth is an empirical
question, and the answer is on file: every earlier game has our estimate per line, the
uncertainty the model stated, and what the real opponents then charged and accepted.

For each class of stated uncertainty the table holds the pair (ka, kb) that would have
earned most on all earlier games:
    a = ka x m                          (ka maximises the replayed revenue)
    b = kb x m, or 0 if P(t = 0) > 1/3  (kb minimises the replayed cost)
Both are picked cliff-averse: a multiplier is scored by the mean of its replay and the
replay 5 % higher (charges) or lower (limits), because revenue drops to nothing just above
t. A class with too few earlier lines keeps the pricing rule.

Learning is per analyst: the rows must come from the same prompt and model that prices the
current game (c2f/decision.py::rows_from_log for the live log).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .config import REPO_ROOT
from .replay import item_net, t_point

PATH = REPO_ROOT / "results" / "market" / "decision.json"
EDGES = (0.12, 0.25, 0.4)                       # belief sigma: confident | floor | normal | rough
KA = (0.80, 0.87, 0.91, 0.95, 1.0, 1.05, 1.10, 1.20, 1.30)
KB = (0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20, 1.35)
NAMES = ("sigma < 0.12", "0.12-0.25", "0.25-0.4", ">= 0.4")


def klass(sigma: float) -> int:
    return sum(sigma >= e for e in EDGES)


@dataclass(frozen=True)
class Row:
    game: int
    sigma: float       # belief sigma of the line (after the floors of c2f/agent.py::to_belief)
    median: float
    p_zero: float
    item: dict         # market summary of the line (c2f/market.py::summarize)
    a: float = 0.0     # what the pricing rule quoted (for the comparison only)
    b: float = 0.0


def _rev(a: float, s: dict, t: float, cap: float | None) -> float:
    return item_net(a, 0.0, s, t, cap=cap) - item_net(0.0, 0.0, s, t, cap=cap)


def _cost(b: float, s: dict, t: float, cap: float | None) -> float:
    return item_net(0.0, b, s, t, cap=cap)      # negative: what we pay as insurer


def fit(rows: list[Row], min_lines: int = 40, min_games: int = 6, up: float = 1.15,
        cap: float | None = 4.0, margin: float = 0.05) -> dict:
    """{class index: {"ka", "kb", "n", "games"}}; ka / kb None = keep the pricing rule."""
    out = {}
    for c in range(len(EDGES) + 1):
        R = [r for r in rows if klass(r.sigma) == c and r.median > 0]
        games = len({r.game for r in R})
        rec = {"ka": None, "kb": None, "n": len(R), "games": games}
        if len(R) >= min_lines and games >= min_games:
            T = [t_point(r.item, up) for r in R]
            rev = lambda k: sum(_rev(k * r.median, r.item, t, cap) for r, t in zip(R, T))  # noqa: E731
            cost = lambda k: sum(_cost(k * r.median if r.p_zero <= 1 / 3 else 0.0, r.item, t, cap)  # noqa: E731
                                 for r, t in zip(R, T))
            rec["ka"] = max(KA, key=lambda k: rev(k) + rev(k * (1 + margin)))
            rec["kb"] = max(KB, key=lambda k: cost(k) + cost(k / (1 + margin)))
            rec["rule_rev"] = round(sum(_rev(r.a, r.item, t, cap) for r, t in zip(R, T)))
            rec["table_rev"] = round(rev(rec["ka"]))
            rec["rule_cost"] = round(sum(_cost(r.b, r.item, t, cap) for r, t in zip(R, T)))
            rec["table_cost"] = round(cost(rec["kb"]))
        out[c] = rec
    return out


def quote(median: float, sigma: float, p_zero: float, table: dict, rule: tuple[float, float],
          use: str = "ab") -> tuple[float, float]:
    """(a, b) for one line. rule: what the pricing rule quotes; kept where the table is silent.
    use: "a", "b" or "ab": which of the two the table may set."""
    rec = table.get(klass(sigma)) or table.get(str(klass(sigma))) or {}
    a, b = rule
    if "a" in use and rec.get("ka") and median > 0:
        a = rec["ka"] * median
    if "b" in use and rec.get("kb") and median > 0:
        b = rec["kb"] * median if p_zero <= 1 / 3 else 0.0
    return round(a, 2), round(b, 2)


# ---- live: rows from the submission log ---------------------------------------------------

def rows_from_log(quotes: Path, summary: dict, analyst: str | None = None, model: str | None = None) -> list[Row]:
    """Lines of earlier live games whose record stores the estimates (games 44+). Only records
    of the given analyst prompt and model family count: another prompt has another bias."""
    recs: dict[int, dict] = {}
    if not Path(quotes).exists():
        return []
    for line in Path(quotes).read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("dry_run") or not d.get("ok") or "estimates" not in d:
            continue
        t = d.get("timing", {})
        if analyst is not None and (t.get("analyst") or d.get("stamp", {}).get("extractor_version", "x3")) != analyst:
            continue
        if model is not None and model not in str(t.get("model", "")):
            continue
        recs[int(d["game_id"])] = d
    rows = []
    for g, d in sorted(recs.items()):
        items = summary.get(str(g))
        if not items:
            continue
        quoted = {int(p["index"]): p for p in d["payload"]}
        for k, e in enumerate(d["estimates"], 1):
            try:
                pos = int(str(e["item_id"]).split()[0])
            except (ValueError, IndexError):
                pos = k
            s, q = items.get(str(pos)), quoted.get(pos)
            if s is None or q is None:
                continue
            rows.append(Row(g, float(e["sigma"]), float(e["median"]), float(e["p_zero"]), s,
                            float(q["charge_price"]), float(q["acceptance_limit"])))
    return rows


def update(path: Path = PATH, quotes: Path | None = None, summary_path: Path | None = None,
           analyst: str | None = None, model: str | None = None, seed: list[Row] | None = None) -> dict:
    """Refit the table on the live log (plus `seed` rows) and write it. Called after every game."""
    quotes = quotes or REPO_ROOT / "runs" / "submissions.jsonl"
    summary_path = summary_path or REPO_ROOT / "results" / "market" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    rows = (seed or []) + rows_from_log(quotes, summary, analyst, model)
    table = fit(rows)
    doc = {"analyst": analyst, "model": model, "games": sorted({r.game for r in rows}),
           "table": {str(c): v for c, v in table.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    return doc


def load(path: str | Path, analyst: str | None = None) -> dict:
    """The stored table, or {} if there is none or it was learned on another analyst prompt."""
    f = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    try:
        doc = json.loads(f.read_text())
    except (OSError, ValueError):
        return {}
    if analyst is not None and doc.get("analyst") not in (None, analyst):
        return {}
    return {int(c): v for c, v in doc.get("table", {}).items()}


def describe(table: dict) -> str:
    rows = []
    for c in sorted(table):
        v = table[c]
        if v.get("ka") is None:
            rows.append(f"{NAMES[int(c)]}: pricing rule ({v['n']} lines)")
        else:
            rows.append(f"{NAMES[int(c)]}: a = {v['ka']:.2f} m, b = {v['kb']:.2f} m ({v['n']} lines, "
                        f"revenue {v['table_rev'] - v['rule_rev']:+,}, cost {v['table_cost'] - v['rule_cost']:+,} vs rule)")
    return "; ".join(rows)


if __name__ == "__main__":
    doc = update()
    print(describe({int(c): v for c, v in doc["table"].items()}), math.nan if not doc["games"] else "")
