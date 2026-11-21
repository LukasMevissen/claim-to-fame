"""Historial, dinero por línea, backtest walk-forward y autotune (campeón / retador).

  python learn.py           # informe: dinero real vs variantes, y leaderboard reconstruido
  python learn.py --tune    # además actualiza tuned.json si un retador gana con claridad
"""
import glob
import json
import math
import os
import sys

import numpy as np

import config
import decision
from belief import Belief

PARAM_GRID = [{"sd_mult": s, "a_mult": m} for s in (0.8, 1.0, 1.25) for m in (0.92, 1.0, 1.08)]
DEFAULT = {"sd_mult": 1.0, "a_mult": 1.0}
TUNED = os.path.join(config.DATA_DIR, "tuned.json")


# ------------------------------------------------------------------ persistencia
def save_round(game_id, data):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(os.path.join(config.DATA_DIR, f"round_{game_id}.json"), "w") as f:
        json.dump(data, f, indent=1, default=float)


def load_rounds():
    out = []
    for p in sorted(glob.glob(os.path.join(config.DATA_DIR, "round_*.json")), key=os.path.getmtime):
        with open(p) as f:
            out.append(json.load(f))
    return out


def records(rounds):
    """Una fila por línea resuelta: lo que predijimos + la verdad + lo que hicieron los demás."""
    rec = []
    for rd in rounds:
        res = rd.get("results") or {}
        for i, pred in (rd.get("preds") or {}).items():
            r = res.get(str(i)) or res.get(i)
            if not r or r.get("t") is None:
                continue
            others = r.get("others") or {}
            rec.append({"game": rd["game_id"], "index": i, "mid": pred["mid"], "sd": pred["sd"],
                        "p_bad": pred["p_bad"], "t": float(r["t"]),
                        "others_b": [float(o["b"]) for o in others.values() if o.get("b") is not None],
                        "others_a": [float(o["a"]) for o in others.values() if o.get("a") is not None],
                        "our_a": (rd.get("final") or {}).get(str(i), {}).get("charge_price"),
                        "our_b": (rd.get("final") or {}).get(str(i), {}).get("acceptance_limit")})
    return rec


def load_params():
    try:
        with open(TUNED) as f:
            return json.load(f)["params"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return dict(DEFAULT)


# ------------------------------------------------------------------ dinero
def line_money(a, b, t, others_a, others_b):
    """Nuestro dinero en una línea: lo que cobramos como reparador - lo que pagamos como aseguradora."""
    got = decision.revenue(a, t, np.asarray(others_b, float)).sum() if len(others_b) else 0.0
    paid = decision.cost(b, np.asarray(others_a, float), t).sum() if len(others_a) else 0.0
    return float(got - paid)


def decide(belief, line, params, seed=0):
    t, B = belief.sample(line["mid"], line["sd"] * params["sd_mult"], line["p_bad"],
                         n=config.N_SAMPLES, seed=seed)
    d = decision.decide_line(t, B)
    d["charge_price"] = round(d["charge_price"] * params["a_mult"], 2)
    return d


# ------------------------------------------------------------------ backtest
def walk_forward(rec, grid=PARAM_GRID):
    """Para cada línea, aprende solo con las rondas anteriores y decide con cada variante."""
    games = []
    for r in rec:
        if r["game"] not in games:
            games.append(r["game"])
    money = np.zeros((len(grid), len(rec)))
    j = 0
    for g in games:
        past = [r for r in rec if games.index(r["game"]) < games.index(g)]
        bel = Belief(past)
        for r in [r for r in rec if r["game"] == g]:
            for gi, p in enumerate(grid):
                d = decide(bel, r, p, seed=j)
                money[gi, j] = line_money(d["charge_price"], d["acceptance_limit"], r["t"],
                                          r["others_a"], r["others_b"])
            j += 1
    return money


def autotune(rec=None, t_threshold=2.0, verbose=True):
    rec = rec if rec is not None else records(load_rounds())
    if len(rec) < 20:
        return load_params()
    money = walk_forward(rec)
    cur = load_params()
    ci = PARAM_GRID.index(cur) if cur in PARAM_GRID else PARAM_GRID.index(DEFAULT)
    bi = int(np.argmax(money.sum(1)))
    diff = money[bi] - money[ci]
    tstat = diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff)) + 1e-9) if bi != ci else 0.0
    chosen = PARAM_GRID[bi] if tstat > t_threshold else PARAM_GRID[ci]
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(TUNED, "w") as f:
        json.dump({"params": chosen, "n_lines": len(rec), "challenger": PARAM_GRID[bi], "t": tstat}, f)
    if verbose:
        for gi in np.argsort(-money.sum(1))[:5]:
            print(f"  {PARAM_GRID[gi]}  total={money[gi].sum():>12,.0f}")
        print(f"autotune: campeón {PARAM_GRID[ci]} | retador {PARAM_GRID[bi]} t={tstat:.2f} -> {chosen}")
    return chosen


def report():
    rec = records(load_rounds())
    print(f"{len(rec)} líneas resueltas")
    if not rec:
        return
    real = [line_money(r["our_a"], r["our_b"], r["t"], r["others_a"], r["others_b"])
            for r in rec if r["our_a"] is not None]
    print(f"dinero real (nuestros envíos): {sum(real):,.0f}")
    t = np.array([r["t"] for r in rec])
    mid = np.array([r["mid"] for r in rec])
    pos = t > 0
    print(f"líneas con t=0: {(~pos).mean():.0%} | sesgo log(t/mid) mediano: {np.median(np.log(t[pos] / mid[pos])):+.3f}"
          f" | dispersión: {np.std(np.log(t[pos] / mid[pos])):.3f}")
    autotune(rec, verbose=True)


if __name__ == "__main__":
    if "--tune" in sys.argv:
        autotune()
    else:
        report()
