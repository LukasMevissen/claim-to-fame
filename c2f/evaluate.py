"""Quality control of a priced case against labelled thresholds (case 0 by default).

Hard gates (fail the run): every item priced, a and b finite and >= 0, runtime within budget.
Metrics (need labels): simulated tournament net payoff against an evaluation opponent
population that is deliberately different from the pricing assumptions, compared with
an oracle (a = b = t) and a naive baseline (a = b = estimated median); estimation errors.

Labels: data/labels/<case_id>.json
{
  "source": "who/how the thresholds were set",
  "items": {"<item_id>": {"t": 123.45, "price_if_covered": 123.45}, ...}
}
t is the gross line-item total (0 if not covered/unrelated). price_if_covered is optional and
only used to model opponents who miss an exclusion.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path

from .agent import PricedItem
from .config import REPO_ROOT
from .payoff import net_payoff

LABEL_DIR = REPO_ROOT / "data" / "labels"
EVAL_DIR = REPO_ROOT / "results" / "eval"

# Evaluation opponents: (name, weight, a_factor, b_factor, log_noise, miss_zero)
EVAL_OPPONENTS = [
    ("absent", 0.15, None, None, 0.0, 0.0),
    ("naive_good", 0.30, 1.00, 1.00, 0.20, 0.3),
    ("naive_rough", 0.30, 1.00, 1.00, 0.50, 0.6),
    ("greedy", 0.15, 1.40, 1.00, 0.35, 0.5),
    ("stingy", 0.10, 0.90, 0.70, 0.35, 0.3),
]


def load_labels(case_id: str) -> dict | None:
    p = LABEL_DIR / f"{case_id}.json"
    return json.loads(p.read_text())["items"] if p.exists() else None


def gates(priced: list[PricedItem], timing: dict, deadline: float, labels: dict | None) -> list[str]:
    errs = []
    if not priced:
        errs.append("no items priced")
    for p in priced:
        for name, v in (("a", p.charge_price), ("b", p.acceptance_limit)):
            if not math.isfinite(v) or v < 0:
                errs.append(f"item {p.item_id}: invalid {name}={v}")
    ids = [p.item_id for p in priced]
    if len(ids) != len(set(ids)):
        errs.append("duplicate item ids")
    if labels is not None:
        missing = set(labels) - set(ids)
        extra = set(ids) - set(labels)
        if missing:
            errs.append(f"items not priced (count as a=b=0): {sorted(missing)}")
        if extra:
            errs.append(f"items not in labels (id mismatch?): {sorted(extra)}")
    if timing.get("total_s", 0) > deadline:
        errs.append(f"runtime {timing['total_s']}s > budget {deadline}s")
    return errs


def _opp_quotes(t: float, ref: float, rng: random.Random) -> list[tuple[float, float, float]]:
    """(weight, a_opp, b_opp) draws for one item."""
    out = []
    for _name, w, fa, fb, s, miss in EVAL_OPPONENTS:
        for _ in range(20):
            if fa is None:
                out.append((w / 20, 0.0, 0.0))
                continue
            base = t if t > 0 else (ref if rng.random() < miss else 0.0)
            est = base * math.exp(rng.gauss(0, s)) if s else base
            out.append((w / 20, fa * est, fb * est))
    return out


def simulate(priced: list[PricedItem], labels: dict, seed: int = 0) -> dict:
    rng = random.Random(seed)
    ours = naive = oracle = 0.0
    log_err, zero_hits = [], []
    for p in priced:
        lab = labels.get(p.item_id)
        if lab is None:
            continue
        t = float(lab["t"])
        med = p.belief["median"]
        ref = float(lab.get("price_if_covered") or med)
        for w, ao, bo in _opp_quotes(t, ref, rng):
            ours += w * net_payoff(p.charge_price, p.acceptance_limit, ao, bo, t)
            naive_q = 0.0 if p.belief["p_zero"] > 0.5 else med
            naive += w * net_payoff(naive_q, naive_q, ao, bo, t)
            oracle += w * net_payoff(t, t, ao, bo, t)
        if t > 0 and med > 0:
            log_err.append(abs(math.log(med / t)))
        zero_hits.append((t == 0, p.belief["p_zero"] > 1 / 3))
    tp = sum(1 for z, pz in zero_hits if z and pz)
    fn = sum(1 for z, pz in zero_hits if z and not pz)
    fp = sum(1 for z, pz in zero_hits if not z and pz)
    return {
        "net_payoff": round(ours, 2),
        "naive_baseline": round(naive, 2),
        "oracle": round(oracle, 2),
        "share_of_oracle": round(ours / oracle, 3) if oracle else None,
        "median_abs_log_error": round(statistics.median(log_err), 3) if log_err else None,
        "zero_items": {"tp": tp, "fn": fn, "fp": fp},
    }


def write_result(case_id: str, stamp: dict, timing: dict, errs: list[str],
                 metrics: dict | None, priced: list[PricedItem]) -> Path:
    d = EVAL_DIR / case_id
    d.mkdir(parents=True, exist_ok=True)
    label = f"v{stamp['agent_version']}-{stamp['git_sha'][:7]}{'-dirty' if stamp['git_dirty'] else ''}"
    path = d / f"{time.strftime('%Y%m%dT%H%M%S')}{time.time_ns() // 1_000_000 % 1000:03d}__{label}.json"
    path.write_text(json.dumps({
        "case_id": case_id, "stamp": stamp, "timing": timing, "gate_errors": errs,
        "metrics": metrics,
        "items": [{"item_id": p.item_id, "a": p.charge_price, "b": p.acceptance_limit,
                   "belief": p.belief} for p in priced],
    }, indent=2, ensure_ascii=False))
    return path


def previous_best(case_id: str, exclude: Path) -> dict | None:
    best = None
    for f in sorted((EVAL_DIR / case_id).glob("*.json")):
        if f == exclude:
            continue
        r = json.loads(f.read_text())
        m = r.get("metrics") or {}
        if r.get("gate_errors") or m.get("net_payoff") is None:
            continue
        if best is None or m["net_payoff"] > best["metrics"]["net_payoff"]:
            best = r | {"file": f.name}
    return best
