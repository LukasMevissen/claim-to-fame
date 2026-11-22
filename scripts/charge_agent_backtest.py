"""Walk-forward backtest of the charge agent (c2f/charge_agent.py).

For every finished game g: the stored case assessment of that game (made live, with the price
book of the games before it), the agent's history cut at g (only games < g - lag), then the
charge per line is scored against the real opponents of game g (c2f/replay.py rules). Only
the charge a differs between the arms, so the difference in net payoff is handyman revenue.

    python scripts/charge_agent_backtest.py                       # rule mode, both settings
    python scripts/charge_agent_backtest.py --llm gpt-4.1         # + llm mode (cached per game)
    python scripts/charge_agent_backtest.py --root <snapshot>     # data of another checkout

Arms: fixed (current rule: max(optimiser, 0.87 m), confident lines 0.95 m),
rule / llm x neutral / loving. Three readings of t where only a lower bound was revealed
(1.0, 1.15 and 1.4 x the bound). Output: table on stdout, results/charge_agent/backtest.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from c2f import charge_agent as ca  # noqa: E402
from c2f import replay  # noqa: E402
from c2f.belief import TBelief  # noqa: E402
from c2f.config import load_config  # noqa: E402
from c2f.pricing import charge_price  # noqa: E402

UPS = (1.0, 1.15, 1.4)


def revenue(a: float, s: dict, up: float, cap: float | None) -> float:
    """Handyman revenue of charge a on a finished line. Where t is bracketed, the mean over
    t spread evenly across the bracket (a single midpoint makes charges inside the bracket a
    coin flip); where only a lower bound is known, t = up x that bound."""
    lo, hi = s["t_low"], s["t_high"]
    if lo > 0 and hi and hi > lo and any(v > 0 for v in s["charges"].values()):
        ts = [lo + (hi - lo) * (i + 0.5) / 9 for i in range(9)]
    else:
        ts = [replay.t_point(s, up)]
    return sum(replay.item_net(a, 0.0, s, t, cap=cap) - replay.item_net(0.0, 0.0, s, t, cap=cap)
               for t in ts) / len(ts)


def fixed_charge(m: float, sigma_stated: float, pricing, conf: bool) -> float:
    """The rule in c2f/pricing.py on the stored estimate. conf: also the confident-line floor
    of origin/main (stated sigma < 0.12 -> at least 0.95 m)."""
    key = f"{m:.2f}|{sigma_stated}|{int(conf)}"
    if key not in _FIXED:
        _FIXED[key] = _fixed_charge(m, sigma_stated, pricing, conf)
    return _FIXED[key]


_FIXED: dict = {}
_FIXED_FILE = REPO / "results/charge_agent/fixed_rule_cache.json"   # the fixed rule is slow to evaluate


def _fixed_charge(m: float, sigma_stated: float, pricing, conf: bool) -> float:
    if m <= 0:
        return 0.0
    sigma = max(sigma_stated, 0.10) if (conf and sigma_stated < 0.12) else min(max(sigma_stated, 0.2), 1.2)
    a = charge_price(TBelief(m, sigma, 0.0), pricing)
    if conf and sigma < 0.12:
        a = max(a, 0.95 * m)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO), help="checkout that holds results/ and runs/")
    ap.add_argument("--games", default="13-999")
    ap.add_argument("--model", default="gpt-6.2-sol", help="assessment whose stored estimates are used")
    ap.add_argument("--llm", default="", help="judge model for llm mode, e.g. gpt-4.1 or gpt-6.2-sol@low")
    ap.add_argument("--llm-timeout", type=float, default=20.0)
    ap.add_argument("--lag", type=int, default=0, help="newest games not yet published at decision time")
    ap.add_argument("--gamma-loving", type=float, default=ca.LOVING.gamma)
    ap.add_argument("--set", nargs="*", default=[], help="override Params, e.g. tau=0.1 temper=0.7")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    summary = json.loads((root / "results/market/summary.json").read_text())
    casebook = root / "results/casebook"
    extractions = root / "results/extractions"
    lo_g, hi_g = (int(x) for x in args.games.split("-"))
    over = {k: float(v) for k, v in (kv.split("=") for kv in args.set)}
    assert "cal" not in over
    # coarser grids than live (3.6 s per line with 22 learned opponents); the 0.87 floor binds on most lines
    pricing = replace(load_config().pricing, n_t_grid=30, n_candidates=24, n_opp_draws=5)
    jdir = REPO / "results/charge_agent/judge"
    jdir.mkdir(parents=True, exist_ok=True)
    if _FIXED_FILE.exists():
        _FIXED.update(json.loads(_FIXED_FILE.read_text()))
    llm_model, _, llm_effort = args.llm.partition("@")

    arms = ["fixed", "fixed+conf", "rule/neutral", "rule/loving"]
    if llm_model:
        arms += ["llm/neutral", "llm/loving"]
    per_game: dict[int, dict] = {}
    rows = []
    judge_meta = []
    t_rule = []
    cals = {}
    for g in sorted(int(k) for k in summary):
        if not lo_g <= g <= hi_g:
            continue
        fs = sorted(extractions.glob(f"game{g}__x*__{args.model}.json"))
        dg = casebook / f"game{g}.json"
        if not fs or not dg.exists():
            continue
        digest = json.loads(dg.read_text(encoding="utf-8"))
        est = {str(it["item_id"]).split()[0].strip("."): it
               for it in json.loads(fs[-1].read_text(encoding="utf-8"))["items"]}
        lines = []
        for pos, text in digest["lines"].items():
            it = est.get(str(pos))
            if it is None or str(pos) not in summary[str(g)]:
                continue
            lines.append(ca.Line(int(pos), text, float(it["quantity"]) * float(it["unit_price_gross_eur"]),
                                 float(it["price_log_sd"])))
        if not lines:
            continue
        hist = ca.load_history(g - args.lag, summary, casebook)
        cal = replace(ca.NEUTRAL, **over)
        if not args.no_calibrate:
            cal = ca.calibrate(ca.estimate_history(summary, extractions, args.model, g - args.lag), cal)
        params = {"neutral": cal, "loving": replace(cal, gamma=args.gamma_loving)}
        last_rel = hist.relevance
        cals[g] = [(round(b, 2), s) for _e, b, s in cal.cal]
        t0 = time.perf_counter()
        dec = {f"rule/{k}": ca.charges(lines, digest.get("policy", ""), digest["description"], hist, p, game=g)
               for k, p in params.items()}
        t_rule.append((time.perf_counter() - t0) / 2)
        if llm_model:
            jf = jdir / f"game{g}__{args.llm.replace('@', '-')}__lag{args.lag}.json"
            weights = None
            if jf.exists():
                j = json.loads(jf.read_text())
                weights = {int(k): [ca.Weight(**w) for w in v] for k, v in j["weights"].items()}
                judge_meta.append(j["meta"])
            else:
                try:
                    weights, meta = ca.judge({ln.pos: ln.text for ln in lines}, digest.get("policy", ""),
                                             digest["description"], hist, llm_model, args.llm_timeout,
                                             llm_effort or None)
                    jf.write_text(json.dumps({"meta": meta, "weights": {k: [asdict(w) for w in v]
                                                                         for k, v in weights.items()}}, indent=1))
                    judge_meta.append(meta)
                except Exception as e:  # noqa: BLE001  live behaviour: fall back to the rule weights
                    print(f"[game {g}] judge failed ({e!r}): rule weights")
                    judge_meta.append({"failed": repr(e)[:120]})
            if weights is not None:
                # relevance per (rule class, judge class), learned from the judgements of earlier games
                pattern = "game{g}__" + args.llm.replace("@", "-") + f"__lag{args.lag}.json"
                jtable = ca.learn_judge(hist, ca.load_judgements(jdir, pattern, g - args.lag))
                weights = ca.llm_weights({ln.pos: ln.text for ln in lines}, digest.get("policy", ""),
                                         digest["description"], hist, weights, jtable, params["neutral"], g)
            for k, p in params.items():
                dec[f"llm/{k}"] = ca.charges(lines, digest.get("policy", ""), digest["description"], hist, p,
                                             weights=weights, game=g)
        pg = {a: {up: 0.0 for up in UPS} for a in arms}
        for ln in lines:
            s = summary[str(g)][str(ln.pos)]
            a_of = {"fixed": fixed_charge(ln.m, ln.sigma, pricing, False),
                    "fixed+conf": fixed_charge(ln.m, ln.sigma, pricing, True)}
            a_of.update({k: d[ln.pos].a for k, d in dec.items()})
            rev = {}
            for arm in arms:
                for up in UPS:
                    r = revenue(a_of[arm], s, up, 4.0)
                    pg[arm][up] += r
                    if up == 1.15:
                        rev[arm] = round(r, 1)
            d = dec["rule/neutral"][ln.pos]
            rows.append({"game": g, "pos": ln.pos, "text": ln.text[:60], "m": round(ln.m, 2), "sigma": ln.sigma,
                         "t_low": s["t_low"], "t_high": s["t_high"],
                         **{f"a[{k}]": round(v, 2) for k, v in a_of.items()},
                         "n_prec": d.n_precedents, "q10": d.t_q10, "q90": d.t_q90, "rev": rev})
        per_game[g] = pg

    _FIXED_FILE.write_text(json.dumps(_FIXED))
    games = sorted(per_game)
    print(f"games {games[0]}-{games[-1]} ({len(games)} games, {len(rows)} lines), assessment {args.model}, "
          f"lag {args.lag}; rule mode {1000 * sum(t_rule) / len(t_rule):.1f} ms per invoice "
          f"(max {1000 * max(t_rule):.1f} ms)")
    print("calibration (bias, sd) per class of stated sd (< 0.12, 0.12-0.2, 0.2-0.4, >= 0.4) at first / middle / last game:",
          cals[games[0]], cals[games[len(games) // 2]], cals[games[-1]])
    print("relevance learned by the last game:", {"/".join(map(str, k)): v for k, v in last_rel.items()})
    rng = random.Random(1)
    print(f"\n{'arm':<14}" + "".join(f"{'t = ' + str(up) + ' x bound':>22}" for up in UPS)
          + f"{'vs fixed+conf at 1.15 (90% interval)':>40}{'games better/worse':>20}")
    base = "fixed+conf"
    table = {}
    for arm in arms:
        tot = {up: sum(per_game[g][arm][up] for g in games) for up in UPS}
        diffs = [per_game[g][arm][1.15] - per_game[g][base][1.15] for g in games]
        boot = sorted(sum(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
        better, worse = sum(d > 1 for d in diffs), sum(d < -1 for d in diffs)
        table[arm] = {"net": {str(up): round(v) for up, v in tot.items()}, "diff": round(sum(diffs)),
                      "ci90": [round(boot[100]), round(boot[1899])], "better": better, "worse": worse}
        print(f"{arm:<14}" + "".join(f"{tot[up]:>22,.0f}" for up in UPS)
              + f"{sum(diffs):>+16,.0f} ({boot[100]:+,.0f} to {boot[1899]:+,.0f})".rjust(40)
              + f"{better:>12} / {worse}")
    print("\nwhere the difference to fixed+conf comes from (t = 1.15 x bound):")
    segs = {"lines with precedents": lambda r: r["n_prec"] > 0, "lines without": lambda r: r["n_prec"] == 0,
            "stated sd < 0.2": lambda r: r["sigma"] < 0.2, "stated sd 0.2-0.4": lambda r: 0.2 <= r["sigma"] < 0.4,
            "stated sd >= 0.4": lambda r: r["sigma"] >= 0.4,
            "t bracketed": lambda r: r["t_low"] > 0 and r["t_high"],
            "t lower bound only": lambda r: r["t_low"] > 0 and not r["t_high"],
            "no fair charge (t ~ 0)": lambda r: r["t_low"] <= 0}
    print(f"{'segment':<26}{'lines':>6}" + "".join(f"{a:>14}" for a in arms[2:]))
    for name, f in segs.items():
        sel = [r for r in rows if f(r)]
        print(f"{name:<26}{len(sel):>6}" + "".join(
            f"{sum(r['rev'][a] - r['rev'][base] for r in sel):>+14,.0f}" for a in arms[2:]))
    if llm_model:
        print("\njudge relevance learned by the last game (rule class, judge class):",
              {"/".join(k): v for k, v in jtable.items()})
    if judge_meta:
        secs = sorted(m["seconds"] for m in judge_meta if "seconds" in m)
        fails = sum("failed" in m for m in judge_meta)
        if secs:
            print(f"\njudge {args.llm}: {len(secs)} calls, median {secs[len(secs) // 2]:.1f} s, "
                  f"p90 {secs[int(0.9 * (len(secs) - 1))]:.1f} s, max {secs[-1]:.1f} s, {fails} failed "
                  f"(timeout {args.llm_timeout:.0f} s -> rule weights)")
            table["judge"] = {"model": args.llm, "calls": len(secs), "median_s": secs[len(secs) // 2],
                              "max_s": secs[-1], "failed": fails}
    if not args.quiet:
        print("\nper game (t = 1.15 x bound):")
        print("game " + "".join(f"{a:>14}" for a in arms))
        for g in games:
            print(f"{g:>4} " + "".join(f"{per_game[g][a][1.15]:>14,.0f}" for a in arms))
    out = Path(args.out) if args.out else REPO / "results/charge_agent/backtest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "table": table,
                               "per_game": {g: {a: {str(u): round(v, 1) for u, v in d.items()}
                                                for a, d in pg.items()} for g, pg in per_game.items()},
                               "lines": rows}, indent=1))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
