"""Backtest charge rules that act on the novelty flag (c2f/novelty.py), game by game.

    python scripts/novelty_backtest.py [--from 4] [--lines]

For every game the detector sees only earlier games. A rule changes the charge a on some
lines; the limit b is untouched, so the change in handyman revenue is the change in net
payoff. Revenue is replayed against each opponent's revealed limit bounds [lo, hi):

    a fair (a <= t):   accepted (a) if a <= lo, rejected (1.5a) if a >= hi, else 1.25a
    a unfair:          min(a, 4t) if a <= lo, else 0

t is not published. Two readings of the bounds [t_low, t_high):
    worst   t = t_low  (the highest charge proven fair; anything above counts as unfair)
    mid     t = midpoint, or 1.15 x t_low when no charge was ever proven too high

Baseline: the charge we submitted. Games we did not price live (4, 5: not played; 21: model
timeout, fixed prices sent) use 0.87 x a saved estimate as the baseline and are marked *.

Rules (S1-S3 act on lines flagged new that we believe covered, S4 on anchored ones):
    S1  new line: a = 1.00 x estimate
    S2  new line: a = 1.20 x estimate
    S3  new line: a = M x estimate, M = the multiple in [0.87, 1.5] that would have earned
        most on the new lines of all earlier games (worst-case reading), once 5 are known
    S4  anchored line: a = max(a, the precedent's highest proven-fair charge)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c2f import novelty  # noqa: E402

US = "GemischtesHack"
FLOOR = 0.87


def load_estimates(g: int) -> dict[int, dict] | None:
    """Saved model estimate per line: live model first, then shadow replay."""
    for f in (*sorted((ROOT / "results/extractions").glob(f"game{g}__x*__gpt-6.2-sol.json")),
              *sorted((ROOT / "results/extractions").glob(f"game{g}__x*.json")),
              ROOT / f"results/shadow/game{g}__gpt-6.2-sol__low.json"):
        if f.exists():
            items = json.loads(f.read_text(encoding="utf-8")).get("items") or []
            if items:
                return {int(it["item_id"]): {"est": it["unit_price_gross_eur"] * (it["quantity"] or 1),
                                             "p": it["covered_prob"] * it["related_prob"]} for it in items}
    return None


def live_quotes() -> dict[int, dict]:
    out = {}
    for line in (ROOT / "runs/submissions.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("ok") and not r.get("dry_run") and r.get("game_id"):
            out[r["game_id"]] = r
    return out


def lines_of(g: int, summary: dict, subs: dict) -> list[dict]:
    est = load_estimates(g) or {}
    tx = json.loads((ROOT / f"data/transactions/game{g}.json").read_text())["items"]
    cb = json.loads((ROOT / f"results/casebook/game{g}.json").read_text(encoding="utf-8"))["lines"]
    flags = novelty.classify({int(k): v for k, v in cb.items()}, before_game=g, summary=summary)
    sub = subs.get(g)
    live = bool(sub) and sub["timing"].get("final_stage") != "fixed"
    quotes = {p["index"]: (p["charge_price"], p["acceptance_limit"]) for p in sub["payload"]} if live else {}
    out = []
    for i, s in summary[str(g)].items():
        i = int(i)
        e = est.get(i)
        if e is None or e["est"] <= 0:
            continue
        a, covered = (quotes[i][0], quotes[i][1] > 0) if i in quotes else (FLOOR * e["est"], e["p"] > 2 / 3)
        bb = {k: list(v) for k, v in s["b_bounds"].items() if k != US}
        if live:                                   # fold in what each team did with our charge
            for t in tx:
                if t["issuer"] == US and t["line_item_index"] == i and a > 0:
                    lo, hi = bb.setdefault(t["reviewer"], [0.0, None])
                    if t["accepted"]:
                        bb[t["reviewer"]][0] = max(lo, a)
                    else:
                        bb[t["reviewer"]][1] = a if hi is None else min(hi, a)
        out.append({"g": g, "i": i, "a": a, "est": e["est"], "covered": covered, "live": live,
                    "tl": s["t_low"], "th": s["t_high"], "bb": bb, "flag": flags.get(i, novelty.Novelty("new")),
                    "text": cb.get(str(i), "")[:60]})
    return out


def revenue(a: float, it: dict, mode: str) -> float:
    tl, th = it["tl"], it["th"]
    t = 0.0 if tl == 0 else (tl if mode == "worst" else ((tl + th) / 2 if th else 1.15 * tl))
    fair = 0 < a <= t
    r = 0.0
    for lo, hi in it["bb"].values():
        if fair:
            r += a if a <= lo else (1.5 * a if hi is not None and a >= hi else 1.25 * a)
        elif a <= lo:
            r += min(a, 4 * t) if t > 0 else a
    return r


def best_multiple(history: list[tuple[float, float]]) -> float:
    """history: (estimate, t_low) of earlier new lines. Multiple that would have earned most."""
    if len(history) < 5:
        return FLOOR
    grid = [FLOOR + 0.01 * k for k in range(0, 64)]
    return max(grid, key=lambda m: sum(m * e for e, tl in history if m * e <= tl))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=4)
    ap.add_argument("--lines", action="store_true", help="print every line with its flag")
    args = ap.parse_args()
    summary = json.loads((ROOT / "results/market/summary.json").read_text())
    subs = live_quotes()
    games = sorted(int(g) for g in summary)
    names = ("S1", "S2", "S3", "S4")
    tot = {m: {n: 0.0 for n in names} for m in ("worst", "mid")}
    history: list[tuple[float, float]] = []
    print("game  lines new anch |" + "".join(f" {n:>7}" for n in names) + "  |" + "".join(f" {n:>7}" for n in names)
          + "  | S3 multiple")
    print("                     |  worst case: t = highest proven-fair  |  mid: t = midpoint / 1.15 x")
    for g in games:
        its = lines_of(g, summary, subs)
        new = [x for x in its if x["covered"] and x["flag"].status == "new"]
        anch = [x for x in its if x["covered"] and x["flag"].status == "anchored"]
        m3 = best_multiple(history)
        if g >= args.start:
            row = {}
            for mode in ("worst", "mid"):
                gain = lambda grp, f: sum(revenue(f(x), x, mode) - revenue(x["a"], x, mode) for x in grp)  # noqa: E731
                row[mode] = {"S1": gain(new, lambda x: 1.0 * x["est"]), "S2": gain(new, lambda x: 1.2 * x["est"]),
                             "S3": gain(new, lambda x: m3 * x["est"]),
                             "S4": gain(anch, lambda x: max(x["a"], x["flag"].t_low))}
                for n in names:
                    tot[mode][n] += row[mode][n]
            mark = " " if its and its[0]["live"] else "*"
            print(f"{g:3d}{mark}  {len(its):4d} {len(new):3d} {len(anch):4d} |"
                  + "".join(f" {row['worst'][n]:+7.0f}" for n in names) + "  |"
                  + "".join(f" {row['mid'][n]:+7.0f}" for n in names) + f"  | {m3:.2f}")
            if args.lines:
                for x in its:
                    if x["covered"]:
                        f = x["flag"]
                        print(f"        {x['i']:2d} {f.status:11s} est={x['est']:7.0f} a={x['a']:8.2f} "
                              f"t[{x['tl']:.0f},{x['th']}] ref={f.ref} {x['text']}")
        history += [(x["est"], x["tl"]) for x in new]
    print("-" * 100)
    print("total                |" + "".join(f" {tot['worst'][n]:+7.0f}" for n in names) + "  |"
          + "".join(f" {tot['mid'][n]:+7.0f}" for n in names))


if __name__ == "__main__":
    main()
