"""Leave-future-out test of the item prior (c2f/item_prior.py).

For every finished game g the prior is fitted on games < g only and compared, line by line,
with what the public transactions of game g revealed about t. Nothing is sent anywhere and
no model is called: the comparison model is the assessment that was stored for game g at the
time (results/extractions/game<g>__x3__<model>.json), which already had the reference cases.

    python scripts/item_prior_backtest.py                    # tables to stdout
    python scripts/item_prior_backtest.py --sweep            # sensitivity to the model parameters
    python scripts/item_prior_backtest.py --replay           # also replay the net payoff (slow)
    python scripts/item_prior_backtest.py --data <dir>       # another checkout's data/ and results/

Outcome of a line: "zero" (nobody paid, or paid without any charge proven fair) or "priced"
(lower bound, with or without an upper bound). Price error = log distance from the estimate
to the revealed set [t_low, t_high): 0 if inside, and for lines without an upper bound only
an estimate below t_low counts as wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c2f import item_prior as ip  # noqa: E402
from c2f.market import load_games, summarize  # noqa: E402


def load_summary(data: Path) -> dict:
    return {str(g): {str(i): v for i, v in summarize(rows).items()}
            for g, rows in load_games(data / "data" / "transactions").items()}


def llm_items(data: Path, game: int, model: str) -> dict[int, dict]:
    f = data / "results" / "extractions" / f"game{game}__x3__{model}.json"
    if not f.exists():
        return {}
    out = {}
    for k, it in enumerate(json.loads(f.read_text(encoding="utf-8"))["items"], 1):
        try:
            pos = int(str(it["item_id"]).split()[0].strip("."))
        except ValueError:
            pos = k
        out[pos] = {"p_zero": 1.0 - it["covered_prob"] * it["related_prob"],
                    "median": it["quantity"] * it["unit_price_gross_eur"],
                    "sigma": min(max(it["price_log_sd"], 0.2), 1.2)}
    return out


def miss(est: float, lo: float, hi: float) -> float:
    if est <= 0:
        return math.inf
    return max(math.log(lo / est), 0.0) + (max(math.log(est / hi), 0.0) if hi < math.inf else 0.0)


def rows_for(data: Path, summary: dict, params: ip.Params, first: int, model: str) -> list[dict]:
    all_obs = ip.observations(summary, data / "results" / "casebook")
    games = sorted({o.game for o in all_obs})
    rows = []
    for g in games:
        if g < first:
            continue
        prior = ip.ItemPrior([o for o in all_obs if o.game < g], params)
        llm = llm_items(data, g, model)
        for o in (o for o in all_obs if o.game == g):
            f = prior.lookup(o.text, o.unit, o.policy)
            r = {"game": g, "line": o.line, "key": o.key, "kind": o.kind, "qty": o.qty,
                 "lo": o.lo * o.qty, "hi": o.hi * o.qty, "has_prior": f is not None,
                 "llm": llm.get(o.line), "fit": f}
            if f is not None:
                r.update(n=f.n, n_priced=f.n_priced, p_zero=f.p_zero, borrowed=f.borrowed_from,
                         logp=math.log(max(f.prob(o.kind, o.lo, o.hi), 1e-12)))
                if f.n_priced or (f.n == 0):
                    q10, q50, q90 = f.quantiles()
                    m, s = f.level()
                    r.update(q10=q10 * o.qty, q50=q50 * o.qty, q90=q90 * o.qty, mu=m + math.log(o.qty), sd=s)
            if r["llm"]:
                b = r["llm"]
                r["llm_logp"] = math.log(max(float(ip.event_lik(
                    o.kind, o.lo * o.qty, o.hi * o.qty, b["p_zero"], np.array(math.log(max(b["median"], 1e-6))),
                    np.array(b["sigma"]), replace(params, eps=0.0))), 1e-12))
            rows.append(r)
    return rows


def metrics(rows: list[dict]) -> dict:
    have = [r for r in rows if r["has_prior"]]
    priced_out = [r for r in have if r["kind"] in ("bracket", "lower") and "q50" in r]
    both = [r for r in priced_out if r["llm"]]
    z = [r for r in have if r["n"] > 0]
    zl = [r for r in z if r["llm"]]
    is_zero = lambda r: float(r["kind"] in ("zero", "upper"))  # noqa: E731
    m = {"lines": len(rows), "with_prior": len(have), "priced_with_prior": len(priced_out)}
    if z:
        m["brier_prior"] = float(np.mean([(r["p_zero"] - is_zero(r)) ** 2 for r in z]))
        base = float(np.mean([is_zero(r) for r in rows]))
        m["brier_base"] = float(np.mean([(base - is_zero(r)) ** 2 for r in z]))
    if zl:
        m["brier_prior_l"] = float(np.mean([(r["p_zero"] - is_zero(r)) ** 2 for r in zl]))
        m["brier_llm"] = float(np.mean([(r["llm"]["p_zero"] - is_zero(r)) ** 2 for r in zl]))
    if priced_out:
        d = [miss(r["q50"], r["lo"], r["hi"]) for r in priced_out]
        m["miss_prior"] = float(np.mean(d))
        m["below_prior"] = float(np.mean([r["q50"] < r["lo"] for r in priced_out]))
        br = [r for r in priced_out if r["kind"] == "bracket"]
        m["above_prior"] = float(np.mean([r["q50"] >= r["hi"] for r in br])) if br else float("nan")
        m["cal_lo_below_q90"] = float(np.mean([r["lo"] <= r["q90"] for r in priced_out]))
        m["cal_hi_above_q10"] = float(np.mean([r["hi"] >= r["q10"] for r in br])) if br else float("nan")
        m["n_bracket"] = len(br)
    if both:
        m["n_both"] = len(both)
        m["miss_prior_b"] = float(np.mean([miss(r["q50"], r["lo"], r["hi"]) for r in both]))
        m["miss_llm"] = float(np.mean([min(miss(r["llm"]["median"], r["lo"], r["hi"]), 5.0) for r in both]))
        m["below_llm"] = float(np.mean([r["llm"]["median"] < r["lo"] for r in both]))
        m["below_prior_b"] = float(np.mean([r["q50"] < r["lo"] for r in both]))
        bb = [r for r in both if r["kind"] == "bracket"]
        if bb:
            m["above_llm"] = float(np.mean([r["llm"]["median"] >= r["hi"] for r in bb]))
            m["above_prior_b"] = float(np.mean([r["q50"] >= r["hi"] for r in bb]))
    sc = [r for r in have if "llm_logp" in r and (r["n"] > 0)]
    if sc:
        m["n_scored"] = len(sc)
        m["logp_prior"] = float(np.mean([r["logp"] for r in sc]))
        m["logp_llm"] = float(np.mean([r["llm_logp"] for r in sc]))
    return m


def fmt(m: dict) -> str:
    g = lambda k, f="{:.3f}": f.format(m[k]) if k in m and m[k] == m[k] else "-"  # noqa: E731
    return "\n".join([
        f"lines {m['lines']}, with a prior {m['with_prior']}, of these priced in the outcome and in the history {m['priced_with_prior']}",
        f"zero or not (Brier, lower is better):  prior {g('brier_prior')}  base rate {g('brier_base')}"
        f"  | on lines the model assessed: prior {g('brier_prior_l')}  model {g('brier_llm')}",
        f"price, prior median:  mean log miss {g('miss_prior')}  below t_low {g('below_prior', '{:.0%}')}"
        f"  above t_high {g('above_prior', '{:.0%}')} (of {m.get('n_bracket', 0)} bracketed)",
        f"80% range:  t_low <= q90 in {g('cal_lo_below_q90', '{:.0%}')}  t_high >= q10 in {g('cal_hi_above_q10', '{:.0%}')}",
        f"same {m.get('n_both', 0)} lines, prior vs stored model estimate:  miss {g('miss_prior_b')} vs {g('miss_llm')}"
        f"  below t_low {g('below_prior_b', '{:.0%}')} vs {g('below_llm', '{:.0%}')}"
        f"  above t_high {g('above_prior_b', '{:.0%}')} vs {g('above_llm', '{:.0%}')}",
        f"log score on {m.get('n_scored', 0)} lines (higher is better):  prior {g('logp_prior')}  model {g('logp_llm')}",
    ])


def fused(r: dict, mode: str) -> dict | None:
    """Stored model belief after ip.combine with the prior (None = unchanged). Coverage stays
    the model's decision: only lines it believes indemnified are touched."""
    b = r.get("llm")
    if not b or r.get("fit") is None or b["p_zero"] > 0.5:
        return None
    out = ip.combine(b["median"], b["sigma"], r["fit"], r["qty"], mode)
    return None if out is None else {**b, "median": out[0], "sigma": out[1]}


def replay(rows: list[dict], summary: dict) -> dict:
    from c2f.belief import TBelief
    from c2f.config import load_config
    from c2f.pricing import quote
    from c2f.replay import game_net

    pr = load_config().pricing
    cache: dict = {}

    def q(b: dict) -> tuple[float, float]:
        k = (round(b["median"], 2), round(b["sigma"], 3), round(b["p_zero"], 3))
        if k not in cache:
            x = quote(TBelief(median=b["median"], sigma=b["sigma"], p_zero=b["p_zero"]), pr)
            cache[k] = (x.charge_price, x.acceptance_limit)
        return cache[k]

    out: dict[str, dict[int, float]] = {}
    games = sorted({r["game"] for r in rows if r["llm"]})
    for name in ("model", "fuse", "replace"):
        out[name] = {}
        for g in games:
            quotes = {}
            for r in (r for r in rows if r["game"] == g and r["llm"]):
                b = (fused(r, name) if name != "model" else None) or r["llm"]
                quotes[r["line"]] = q(b)
            out[name][g] = game_net(quotes, g, summary) or 0.0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT))
    ap.add_argument("--first", type=int, default=5, help="first game that is predicted")
    ap.add_argument("--model", default="gpt-6.2-sol")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--lines", action="store_true", help="print every predicted line")
    a = ap.parse_args()
    data = Path(a.data)
    summary = load_summary(data)
    rows = rows_for(data, summary, ip.DEFAULT, a.first, a.model)
    print(f"games {a.first}..{max(r['game'] for r in rows)}, prior fitted on earlier games only\n")
    print(fmt(metrics(rows)))
    late = [r for r in rows if r["game"] >= 30]
    print("\ngames 30+ only:\n" + fmt(metrics(late)))
    capped = ip.deflated_games(ip.observations(summary, data / "results" / "casebook"))
    print(f"\nwithout the games detected afterwards as capped across the invoice {sorted(capped)}:\n"
          + fmt(metrics([r for r in rows if r["game"] not in capped])))
    if a.lines:
        for r in rows:
            if r["has_prior"] and "q50" in r and r["kind"] in ("bracket", "lower"):
                hi = f"{r['hi']:.0f}" if r["hi"] < math.inf else "-"
                l = f"{r['llm']['median']:.0f}" if r["llm"] else "-"
                print(f"  g{r['game']:>2} {r['line']:>2} {r['key'][:38]:38} t [{r['lo']:.0f}, {hi})"
                      f"  prior {r['q50']:.0f} ({r['q10']:.0f}-{r['q90']:.0f})  model {l}")
    if a.sweep:
        print("\nsensitivity (all games): miss of the median, calibration of the 80% range, log score")
        grid = [("default", {})] + [(f"{k}={v}", {k: v}) for k, vs in (
            ("gap_rate", (5.0, 8.0, 20.0, 40.0)), ("sigma_center", (0.06, 0.2, 0.3)),
            ("eps", (0.0, 0.2)), ("fuzzy", (0.0,)), ("upper_pos", (0.01, 0.2)),
            ("other_policy", (1.0, 0.1)),
            ("deflated_shift", (-9.0,))) for v in vs]
        for name, kw in grid:
            m = metrics(rows_for(data, summary, replace(ip.DEFAULT, **kw), a.first, a.model))
            print(f"  {name:22} n={m['priced_with_prior']:3} miss {m['miss_prior']:.3f}  below {m['below_prior']:.0%}"
                  f"  above {m['above_prior']:.0%}  lo<=q90 {m['cal_lo_below_q90']:.0%}"
                  f"  hi>=q10 {m['cal_hi_above_q10']:.0%}  logp {m['logp_prior']:.3f}  brier {m['brier_prior']:.3f}")
    if a.replay:
        res = replay(rows, summary)
        print("\nreplayed net payoff (c2f/replay.py), quotes from the stored model estimate vs the")
        print("same estimate combined with the prior:")
        for name, per in res.items():
            tot = sum(per.values())
            late_tot = sum(v for g, v in per.items() if g >= 30)
            wins = sum(per[g] > res["model"][g] + 1 for g in per)
            loss = sum(per[g] < res["model"][g] - 1 for g in per)
            print(f"  {name:8} total {tot:9.0f}   games 30+ {late_tot:9.0f}   better in {wins}, worse in {loss} of {len(per)} games")
