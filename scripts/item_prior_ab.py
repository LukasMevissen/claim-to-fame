"""A/B test of the item prior as context for the case assessment model.

Each past game is assessed twice by the live model, with the same prompt, policy, claim,
invoice, photo and reference cases (earlier games only):
    ref        reference cases only (what the agent sent before)
    ref+prior  the same plus the PRICE PRIORS block of c2f/item_prior.py, fitted on earlier games only
               (any arm name starting with ref+prior, e.g. ref+prior-v1 for stored answers of an
               earlier wording of the block)
    ref>fuse, ref>replace   no extra model call: the stored ref answers, combined afterwards with
               the prior at belief level (c2f/item_prior.py::combine)
Both answers are priced with the live pricing rule and replayed against the real opponents
(c2f/replay.py). Nothing is submitted. Answers are stored, so a rerun costs no tokens.

    python scripts/item_prior_ab.py --cases <dir with game<N>/ folders> --games 30-46
        [--data <checkout with data/transactions and results/casebook>] [--runs 1]
        [--model claude-opus-5-5 --effort medium]     (gpt-* models go through c2f/extract.py)

Decrypt past cases with scripts/analysis/unpack_cases.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
from c2f.replay import game_net, t_point  # noqa: E402

OUT = ROOT / "results" / "item_prior" / "ab"


def assess(game: int, arm: str, run: int, a, summary: dict) -> dict:
    arm = arm.split(">")[0]                   # derived arms reuse the stored answers
    f = OUT / f"game{game}__{arm}__{a.model}__{a.effort}__r{run}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    from c2f.anthropic_extract import extract_claude
    case = load_case(Path(a.cases) / f"game{game}", f"game{game}")
    ref = pricebook.reference(before_game=game, max_chars=24000)
    block = ""
    if arm.startswith("ref+prior"):
        prior = ip.ItemPrior(ip.observations(summary, pricebook.CASEBOOK, before_game=game))
        block = ip.context(pricebook.invoice_lines(case.invoice_text), prior,
                           policy=pricebook.policy_title(case.policy))
        ref = f"{ref}\n\n{block}" if block else ref
    try:
        if a.model.startswith("claude"):
            items, meta = extract_claude(case, a.model, 150, ref, a.effort)
        else:
            from c2f.extract import extract
            items, meta = extract(case, a.model, 150, 0.0, use_cache=False, reference=ref,
                                  effort=a.effort, save=False), {}
    except Exception as e:  # noqa: BLE001
        print(f"game {game} {arm}: {e!r}"[:200])
        return {"game": game, "arm": arm, "error": repr(e)[:300], "items": []}
    d = {"game": game, "arm": arm, "run": run, "meta": meta, "prior_block": block,
         "items": [asdict(i) for i in items]}
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    return d


def _score_job(job) -> dict:
    d, prior_lines, casebook = job
    summary = json.loads((OUT / "_summary.json").read_text())
    if ">" in d["arm"]:                       # belief-level combination with the prior
        mode = d["arm"].split(">")[1]
        g = d["game"]
        digest = json.loads((Path(casebook) / f"game{g}.json").read_text(encoding="utf-8"))
        prior = ip.ItemPrior(ip.observations(summary, casebook, before_game=g))
        items = [ItemEstimate(**i) for i in d["items"]]
        for k, it in enumerate(items, 1):
            text = digest["lines"].get(str(_index(it.item_id, k)))
            if not text or it.covered_prob * it.related_prob < 0.5 or it.quantity <= 0:
                continue
            desc, qty, unit = ip.parse_line(text)
            out = ip.combine(it.quantity * it.unit_price_gross_eur, it.price_log_sd,
                             prior.lookup(desc, unit, digest.get("policy", "")), qty, mode)
            if out is not None:
                it.unit_price_gross_eur, it.price_log_sd = out[0] / it.quantity, out[1]
        d = {**d, "items": [asdict(i) for i in items]}
    return score(d, summary, load_config(), prior_lines)


def score(d: dict, summary: dict, cfg, prior_lines: set[int]) -> dict:
    items = [ItemEstimate(**i) for i in d["items"]]
    priced = price_items(items, cfg)
    g = summary[str(d["game"])]
    quotes = {_index(p.item_id, k): (p.charge_price, p.acceptance_limit) for k, p in enumerate(priced, 1)}
    out = {"net": game_net(quotes, d["game"], summary) or 0.0, "miss": [], "miss_prior": [],
           "zero_err": 0, "zero_err_prior": 0, "below": 0, "below_prior": 0, "n": 0}
    for k, it in enumerate(items, 1):
        pos = _index(it.item_id, k)
        s = g.get(str(pos))
        if s is None:
            continue
        t = t_point(s)
        p_cov = it.covered_prob * it.related_prob
        est = it.quantity * it.unit_price_gross_eur
        out["n"] += 1
        wrong = (t == 0 or s["t_low"] == 0) != (p_cov < 0.5)
        out["zero_err"] += wrong
        out["zero_err_prior"] += wrong and pos in prior_lines
        if s["t_low"] > 0 and est > 0:
            hi = s["t_high"] or math.inf
            m = max(math.log(s["t_low"] / est), 0.0) + (max(math.log(est / hi), 0.0) if hi < math.inf else 0.0)
            out["miss"].append(m)
            out["below"] += est < s["t_low"]
            if pos in prior_lines:
                out["miss_prior"].append(m)
                out["below_prior"] += est < s["t_low"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--data", default=str(ROOT))
    ap.add_argument("--games", default="30-46")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=4, help="processes for the scoring")
    ap.add_argument("--call-workers", type=int, default=3, help="parallel model calls")
    ap.add_argument("--arms", default="ref,ref+prior", help="e.g. 'ref' to store the baseline answers only")
    a = ap.parse_args()
    load_env(ROOT / ".env")
    data = Path(a.data)
    summary = {str(g): {str(i): v for i, v in summarize(rows).items()}
               for g, rows in load_games(data / "data" / "transactions").items()}
    # the reference cases must come from the same data as the prior
    pricebook.CASEBOOK = data / "results" / "casebook"
    pricebook.SUMMARY = OUT / "_summary.json"
    OUT.mkdir(parents=True, exist_ok=True)
    pricebook.SUMMARY.write_text(json.dumps(summary))
    lo, _, hi = a.games.partition("-")
    games = [g for g in range(int(lo), int(hi or lo) + 1)
             if str(g) in summary and (Path(a.cases) / f"game{g}").exists()]
    jobs = [(g, arm, r) for g in games for arm in a.arms.split(",") for r in range(a.runs)]
    with ThreadPoolExecutor(a.call_workers) as ex:
        res = list(ex.map(lambda j: assess(*j, a, summary), jobs))
    arms = a.arms.split(",")
    tot = {arm: {"net": 0.0, "miss": [], "miss_prior": [], "zero_err": 0, "zero_err_prior": 0,
                 "below": 0, "below_prior": 0, "n": 0} for arm in arms}
    per_game: dict[int, dict[str, list[float]]] = {}
    print(f"{a.model}@{a.effort}, {a.runs} run(s) per arm; net = replayed net payoff per game (mean of runs)")
    # lines that have a prior row, from the stored block of any prior arm of the same game
    rows_of: dict[int, set[int]] = {}
    for (g, arm, r), d in zip(jobs, res):
        for m in re.findall(r"^- line (\d+) ", d.get("prior_block", ""), flags=re.M):
            rows_of.setdefault(g, set()).add(int(m))
    todo = [({**d, "arm": arm}, rows_of.get(g, set()), str(pricebook.CASEBOOK))
            for (g, arm, r), d in zip(jobs, res) if d["items"]]
    with ProcessPoolExecutor(a.workers) as ex:       # the pricing rule is slow pure Python
        scores = list(ex.map(_score_job, todo))
    for (d, _, _), s in zip(todo, scores):
        per_game.setdefault(d["game"], {}).setdefault(d["arm"], []).append(s["net"])
        for k, v in s.items():
            tot[d["arm"]][k] += v
    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    base = arms[0]
    print("  game   " + "   ".join(f"{x:>14}" for x in arms))
    wins = {x: 0 for x in arms[1:]}
    losses = {x: 0 for x in arms[1:]}
    for g, by in sorted(per_game.items()):
        print(f"  {g:>4}   " + "   ".join(f"{mean(by.get(x, [])):14.0f}" for x in arms))
        for x in arms[1:]:
            if x in by and base in by:
                wins[x] += mean(by[x]) > mean(by[base]) + 1
                losses[x] += mean(by[x]) < mean(by[base]) - 1
    for arm, t in tot.items():
        print(f"{arm:10} net {t['net'] / a.runs:9.0f} | lines {t['n'] // a.runs}, covered-or-not wrong {t['zero_err'] / a.runs:.1f}"
              f" (lines with a prior row: {t['zero_err_prior'] / a.runs:.1f})"
              f" | priced lines: mean log miss {mean(t['miss']):.3f}, below t_low {t['below'] / max(len(t['miss']), 1):.0%}"
              f" | priced lines with a prior row ({len(t['miss_prior']) // a.runs}): miss {mean(t['miss_prior']):.3f},"
              f" below t_low {t['below_prior'] / max(len(t['miss_prior']), 1):.0%}")
    for x in arms[1:]:
        print(f"{x} better than {base} in {wins[x]} games, worse in {losses[x]}, of {len(per_game)}")
    import statistics
    for x in arms:                                   # run-to-run spread of the total
        totals = [sum(by[x][r] for by in per_game.values() if x in by and len(by[x]) > r) for r in range(a.runs)]
        if len(totals) > 1:
            print(f"{x}: total per run {[round(t) for t in totals]}, sd {statistics.stdev(totals):.0f}")
