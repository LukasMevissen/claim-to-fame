"""Walk-forward backtest of the case-analysis overhaul (docs/research/overhaul.md).

Every past game is assessed again with only the information available before it (cases and
revealed thresholds of earlier games), priced with the live pricing rule and replayed
against the real opponents (c2f/replay.py). Nothing is submitted.

An arm is "<prompt>" or "<prompt>><layer>":
    prompt  base    what the live agent sends today: x3 prompt, newest cases first, PRICE PRIORS
            x5      the analyst prompt of c2f/analyst.py (approach A, learning in context)
    layer   (none)  the model's answer as it stands
            floor<r> charge at least r x estimate (the live rule has 0.87; 0.95 if confident)
            cal     the decision table of c2f/decision.py (approach B, learning in code): a and b
                    as shares of the estimate, refitted before every game on all earlier
                    games' answers of the same prompt and run. cal:a / cal:b set only one.
            mem     the case memory of c2f/memory.py (rejected variant of approach B)
            oracle-price, oracle-cover, scale<f>   diagnostics, not candidates
Model answers are stored (results/overhaul/answers, and the stored `ref+prior` answers of
scripts/item_prior_ab.py count as `base`), so a rerun makes no model calls and layers are
compared on identical answers.

    python scripts/overhaul_backtest.py --cases <dir with game<N>/> --games 30-55 --runs 3
        --arms "base,base>mem,x5,x5>mem" [--model gpt-6.2-sol --effort low] [--no-calls]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c2f import item_prior as ip, pricebook  # noqa: E402
from c2f.agent import _index, price_items  # noqa: E402
from c2f.case_io import load_case  # noqa: E402
from c2f.config import load_config  # noqa: E402
from c2f.env import load_env  # noqa: E402
from c2f.extract import ItemEstimate  # noqa: E402
from c2f.market import load_games, summarize  # noqa: E402
from c2f.replay import item_net, t_point  # noqa: E402

OUT = ROOT / "results" / "overhaul"
ANS = OUT / "answers"
AB = ROOT / "results" / "item_prior" / "ab"          # stored answers of scripts/item_prior_ab.py


def build_summary(data: Path) -> dict:
    return {str(g): {str(i): v for i, v in summarize(rows).items()}
            for g, rows in load_games(data / "data" / "transactions").items()}


def answer_file(game: int, prompt: str, model: str, effort: str, run: int) -> Path:
    f = ANS / f"game{game}__{prompt}__{model}__{effort}__r{run}.json"
    if prompt == "base" and not f.exists():
        old = AB / f"game{game}__ref+prior__{model}__{effort}__r{run}.json"
        if old.exists():
            return old
    return f


def prompt_parts(prompt: str, case, game: int, summary: dict, model: str) -> tuple[str, str]:
    """(system, reference) for one arm, from games before `game` only."""
    from c2f.extract import SYSTEM, SYSTEM_UPPER
    claude = model.startswith("claude")
    if prompt == "base":
        ref = pricebook.reference(before_game=game, max_chars=24000)
        prior = ip.ItemPrior(ip.observations(summary, pricebook.CASEBOOK, before_game=game))
        block = ip.context(pricebook.invoice_lines(case.invoice_text), prior,
                           policy=pricebook.policy_title(case.policy))
        return (SYSTEM_UPPER if claude else SYSTEM), (f"{ref}\n\n{block}" if block else ref)
    from c2f import analyst
    return analyst.system(prompt, claude), analyst.reference(case, prompt, before_game=game, summary=summary)


def assess(game: int, prompt: str, run: int, a, summary: dict) -> dict | None:
    f = answer_file(game, prompt, a.model, a.effort, run)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    if a.no_calls:
        return None
    case = load_case(Path(a.cases) / f"game{game}", f"game{game}")
    system, ref = prompt_parts(prompt, case, game, summary, a.model)
    t0 = time.time()
    try:
        if a.model.startswith("claude"):
            from c2f.anthropic_extract import extract_claude
            items, meta = extract_claude(case, a.model, 150, ref, a.effort, system=system)
        else:
            from c2f import extract as ex
            old, ex.SYSTEM = ex.SYSTEM, system          # extract() reads the module constant
            try:
                items, meta = ex.extract(case, a.model, 150, 0.0, use_cache=False, reference=ref,
                                         effort=a.effort, save=False), {}
            finally:
                ex.SYSTEM = old
    except Exception as e:  # noqa: BLE001
        print(f"game {game} {prompt} r{run}: {e!r}"[:200])
        return None
    d = {"game": game, "arm": prompt, "run": run, "meta": {**meta, "seconds": round(time.time() - t0, 1),
         "ref_chars": len(ref)}, "items": [asdict(i) for i in items]}
    ANS.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    return d


# ---- layers ---------------------------------------------------------------------------

def layer(name: str, items: list[ItemEstimate], game: int, summary: dict, casebook: Path) -> list[ItemEstimate]:
    if not name:
        return items
    g = summary[str(game)]
    if name.startswith("scale"):
        f = float(name[5:])
        for it in items:
            it.unit_price_gross_eur *= f
        return items
    if name in ("oracle-price", "oracle-cover", "oracle"):
        for k, it in enumerate(items, 1):
            s = g.get(str(_index(it.item_id, k)))
            if s is None:
                continue
            t = t_point(s)
            if name in ("oracle-price", "oracle") and t > 0:
                it.unit_price_gross_eur, it.price_log_sd = t / max(it.quantity, 1e-9), 0.05
            if name in ("oracle-cover", "oracle"):
                it.covered_prob, it.related_prob = (0.98 if t > 0 else 0.02), 1.0
        return items
    if name.startswith("mem"):
        from c2f import memory
        digest = json.loads((casebook / f"game{game}.json").read_text(encoding="utf-8"))
        opt = memory.options(name)
        mem = memory.Memory.build(summary, casebook, before_game=game, opt=opt)
        truth = {int(i): t_point(s) for i, s in g.items()} if opt.oracle else None
        memory.apply(items, digest, mem, truth=truth)
        return items
    raise ValueError(f"unknown layer {name!r}")


# ---- scoring ----------------------------------------------------------------------------

def score(items: list[ItemEstimate], game: int, summary: dict, cfg, up: float, cap: float | None,
          table: dict | None = None, use: str = "ab", floor: float = 0.0) -> dict:
    priced = price_items(items, cfg)
    if floor:                                   # fixed charge floor as a share of the estimate
        for it, p in zip(items, priced):
            p.charge_price = round(max(p.charge_price, floor * it.quantity * it.unit_price_gross_eur), 2)
    if table:                                   # decision table on top of the pricing rule
        from c2f import decision
        from c2f.agent import to_belief
        for it, p in zip(items, priced):
            bel = to_belief(it, cfg)
            p.charge_price, p.acceptance_limit = decision.quote(
                bel.median, bel.sigma, bel.p_zero, table, (p.charge_price, p.acceptance_limit), use)
    g = summary[str(game)]
    out = {"net": 0.0, "rev": 0.0, "cost": 0.0, "n": 0, "n_cov": 0, "under": 0, "over": 0, "cover_wrong": 0,
           "miss": []}
    seen = set()
    for k, (it, p) in enumerate(zip(items, priced), 1):
        pos = _index(it.item_id, k)
        s = g.get(str(pos))
        if s is None or pos in seen:
            continue
        seen.add(pos)
        t = t_point(s, up)
        net = item_net(p.charge_price, p.acceptance_limit, s, t, cap=cap)
        out["net"] += net
        est = it.quantity * it.unit_price_gross_eur
        p_cov = it.covered_prob * it.related_prob
        out["n"] += 1
        out["cover_wrong"] += (t == 0 or s["t_low"] == 0) != (p_cov < 0.5)
        if s["t_low"] > 0 and est > 0:
            hi = s["t_high"] or math.inf
            out["n_cov"] += 1
            out["under"] += est < 0.75 * s["t_low"]
            out["over"] += est > hi
            out["miss"].append(max(math.log(s["t_low"] / est), 0.0) + (max(math.log(est / hi), 0.0) if hi < math.inf else 0.0))
    # lines the model skipped earn the API default 0 / 0
    for pos, s in g.items():
        if int(pos) not in seen:
            out["net"] += item_net(0.0, 0.0, s, t_point(s, up), cap=cap)
    return out


_rows_cache: dict[tuple, list] = {}


def _rows(prompt: str, run: int, game: int, a, summary: dict, cfg) -> list:
    """decision.Row per line of one earlier game, from the stored answer of the same prompt and run."""
    from c2f import decision
    from c2f.agent import to_belief
    key = (prompt, run, game)
    if key not in _rows_cache:
        f = answer_file(game, prompt, a.model, a.effort, run)
        rows = []
        if f.exists() and str(game) in summary:
            items = [ItemEstimate(**i) for i in json.loads(f.read_text(encoding="utf-8"))["items"]]
            for k, (it, q) in enumerate(zip(items, price_items(items, cfg)), 1):
                s = summary[str(game)].get(str(_index(it.item_id, k)))
                if s is not None:
                    bel = to_belief(it, cfg)
                    rows.append(decision.Row(game, bel.sigma, bel.median, bel.p_zero, s,
                                             q.charge_price, q.acceptance_limit))
        _rows_cache[key] = rows
    return _rows_cache[key]


def _table(prompt: str, run: int, game: int, a, summary: dict, cfg) -> dict:
    """Decision table for `game`, fitted on the games before it (walk-forward)."""
    from c2f import decision
    rows = [r for g in range(1, game) for r in _rows(prompt, run, g, a, summary, cfg)]
    return decision.fit(rows, up=a.train_up, cap=a.cap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--data", default=str(ROOT))
    ap.add_argument("--games", default="30-55")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--model", default="gpt-6.2-sol")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--arms", default="base,base>mem")
    ap.add_argument("--call-workers", type=int, default=4)
    ap.add_argument("--no-calls", action="store_true", help="only stored answers")
    ap.add_argument("--up", type=float, default=1.15, help="t = up x lower bound where only that is known")
    ap.add_argument("--cap", type=float, default=4.0)
    ap.add_argument("--per-game", action="store_true")
    ap.add_argument("--train-up", type=float, default=1.15, help="the same assumption when the decision table is fitted")
    a = ap.parse_args()
    load_env(ROOT / ".env")
    data = Path(a.data)
    summary = build_summary(data)
    OUT.mkdir(parents=True, exist_ok=True)
    pricebook.CASEBOOK = data / "results" / "casebook"
    pricebook.SUMMARY = OUT / "_summary.json"
    pricebook.SUMMARY.write_text(json.dumps(summary))
    lo, _, hi = a.games.partition("-")
    games = [g for g in range(int(lo), int(hi or lo) + 1)
             if str(g) in summary and (Path(a.cases) / f"game{g}").exists()
             and (pricebook.CASEBOOK / f"game{g}.json").exists()]
    arms = [x.strip() for x in a.arms.split(",")]
    prompts = list(dict.fromkeys(x.split(">")[0] for x in arms))
    jobs = [(g, p, r) for g in games for p in prompts for r in range(a.runs)]
    with ThreadPoolExecutor(a.call_workers) as ex:
        ans = dict(zip(jobs, ex.map(lambda j: assess(*j, a, summary), jobs)))
    cfg = load_config()
    tot = {x: {"net": 0.0, "n": 0, "n_cov": 0, "under": 0, "over": 0, "cover_wrong": 0, "miss": []} for x in arms}
    per: dict[int, dict[str, list[float]]] = {}
    secs: dict[str, list[float]] = {}
    complete = [g for g in games if all(ans[(g, p, r)] for p in prompts for r in range(a.runs))]
    for g in complete:
        for x in arms:
            p, _, lay = x.partition(">")
            for r in range(a.runs):
                d = ans[(g, p, r)]
                table, use, floor = None, "ab", 0.0
                if lay.startswith("floor"):
                    floor, lay = float(lay[5:]), ""
                if lay.startswith("cal"):
                    table = _table(p, r, g, a, summary, cfg)
                    use = lay.partition(":")[2] or "ab"
                    items = [ItemEstimate(**i) for i in d["items"]]
                else:
                    items = layer(lay, [ItemEstimate(**i) for i in d["items"]], g, summary, pricebook.CASEBOOK)
                s = score(items, g, summary, cfg, a.up, a.cap, table, use, floor)
                per.setdefault(g, {}).setdefault(x, []).append(s["net"])
                for k in tot[x]:
                    tot[x][k] += s[k]
                if d.get("meta", {}).get("seconds"):
                    secs.setdefault(p, []).append(d["meta"]["seconds"])
    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    print(f"{a.model}@{a.effort}, games {complete[0]}-{complete[-1]} ({len(complete)} games"
          f"{', missing answers: ' + str(sorted(set(games) - set(complete))) if len(complete) < len(games) else ''}), "
          f"{a.runs} run(s); replay with t = {a.up} x lower bound where open, cap {a.cap} t")
    if a.per_game:
        print("  game " + " ".join(f"{x:>16}" for x in arms))
        for g in complete:
            print(f"  {g:>4} " + " ".join(f"{mean(per[g][x]):16.0f}" for x in arms))
    base = arms[0]
    print(f"{'arm':22s} {'net':>9s} {'vs ' + base:>12s} {'90% interval':>18s} {'better/worse':>12s} | {'cover wrong':>11s} {'est<.75 t_low':>13s} {'est>t_high':>10s} {'log miss':>8s}")
    for x in arms:
        t = tot[x]
        diff = [mean(per[g][x]) - mean(per[g][base]) for g in complete]
        lo_ci, hi_ci = _boot(diff)
        w, l_ = sum(d > 1 for d in diff), sum(d < -1 for d in diff)
        print(f"{x:22s} {t['net'] / a.runs:9.0f} {sum(diff):+12.0f} {f'{lo_ci:+.0f} to {hi_ci:+.0f}':>18s} {f'{w} / {l_}':>12s} | "
              f"{t['cover_wrong'] / max(t['n'], 1):11.1%} {t['under'] / max(t['n_cov'], 1):13.1%} {t['over'] / max(t['n_cov'], 1):10.1%} {mean(t['miss']):8.3f}")
    for x in arms:
        totals = [sum(per[g][x][r] for g in complete) for r in range(a.runs)]
        if len(totals) > 1:
            print(f"  {x}: total per run {[round(v) for v in totals]}, sd {statistics.stdev(totals):.0f}")
    for p, v in secs.items():
        print(f"  {p}: model call seconds median {statistics.median(v):.1f}, max {max(v):.1f} (n={len(v)})")


def _boot(diff: list[float], n: int = 4000, seed: int = 1) -> tuple[float, float]:
    """90 % bootstrap interval of the summed per-game difference (games resampled)."""
    import random
    rng = random.Random(seed)
    if not diff:
        return 0.0, 0.0
    sums = sorted(sum(rng.choice(diff) for _ in diff) for _ in range(n))
    return sums[int(0.05 * n)], sums[int(0.95 * n)]


if __name__ == "__main__":
    main()
