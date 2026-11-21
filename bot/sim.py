"""Simulador del juego completo (todos contra todos, pagos del Deep Dive).

  python sim.py [rondas] [semilla]

Los rivales y el ruido del "LLM" son inventados: sirve para validar la lógica de a y b
y el aprendizaje, no para predecir el leaderboard real.
"""
import math
import sys

import numpy as np

import decision
import learn
from belief import Belief

LINES_PER_ROUND = 6
P_ZERO = 0.25


def make_line(rng):
    base = 300 * math.exp(rng.normal(0, 1.0))
    t = 0.0 if rng.random() < P_ZERO else base
    return {"t": t, "base": base}


def our_signal(rng, line):
    mid = line["base"] * math.exp(rng.normal(0.10, 0.30))          # LLM: sesgo +10 %, ruido 30 %
    p_bad = rng.beta(6, 3) if line["t"] == 0 else rng.beta(1, 8)
    return {"mid": mid, "sd": 0.30, "p_bad": p_bad}


RIVAL_TYPES = {
    "ingenuo": lambda e, s: (e, 1.10 * e),
    "prudente": lambda e, s: (0.90 * e, 0.90 * e),
    "avaricioso": lambda e, s: (1.25 * e, 1.00 * e),
    "laxo": lambda e, s: (e, 1.40 * e),
    "racional": lambda e, s: (0.95 * e, e * math.exp(-0.43 * s)),
}


def make_rivals(rng):
    rivals = []
    for name, fn in RIVAL_TYPES.items():
        for k in range(3):
            rivals.append({"name": f"{name}{k}", "fn": fn, "sd": rng.uniform(0.15, 0.45),
                           "detect": rng.uniform(0.3, 0.9)})
    return rivals


def rival_action(rng, r, line):
    est = line["base"] * math.exp(rng.normal(0, r["sd"]))
    if line["t"] == 0 and rng.random() < r["detect"]:
        return 0.0, 0.0
    return r["fn"](est, r["sd"])


def main(n_rounds=100, seed=0):
    rng = np.random.default_rng(seed)
    rivals = make_rivals(rng)
    history = []
    teams = [r["name"] for r in rivals] + ["NOSOTROS_bot", "NOSOTROS_ingenuo"]
    money = {t: 0.0 for t in teams}
    blocks = []
    for rd in range(n_rounds):
        belief = Belief(history)
        round_bot = 0.0
        for li in range(LINES_PER_ROUND):
            line = make_line(rng)
            sig = our_signal(rng, line)
            acts = {r["name"]: rival_action(rng, r, line) for r in rivals}
            d = learn.decide(belief, sig, learn.DEFAULT, seed=rd * 10 + li)
            acts["NOSOTROS_bot"] = (d["charge_price"], d["acceptance_limit"])
            naive = (0.0, 0.0) if sig["p_bad"] > 0.5 else (sig["mid"], 1.1 * sig["mid"])
            acts["NOSOTROS_ingenuo"] = naive
            t = line["t"]
            for h in teams:                      # h cobra a cada aseguradora i
                for i in teams:
                    if h == i:
                        continue
                    pay = float(decision.revenue(acts[h][0], t, acts[i][1]))
                    money[h] += pay
                    money[i] -= pay
                    if h == "NOSOTROS_bot":
                        round_bot += pay
                    if i == "NOSOTROS_bot":
                        round_bot -= pay
            others = {n: {"a": a, "b": b} for n, (a, b) in acts.items() if not n.startswith("NOSOTROS")}
            history.append({"game": rd, "mid": sig["mid"], "sd": sig["sd"], "p_bad": sig["p_bad"], "t": t,
                            "others_b": [o["b"] for o in others.values()],
                            "others_a": [o["a"] for o in others.values()]})
        blocks.append(round_bot)
    print(f"\n{n_rounds} rondas x {LINES_PER_ROUND} líneas, {len(teams)} equipos (suma cero)")
    print("Dinero medio por ronda del bot, por bloques de 20 rondas:")
    for b in range(0, n_rounds, 20):
        print(f"  rondas {b:>3}-{b + 19:<3}: {np.mean(blocks[b:b + 20]):>9,.0f}")
    print("\nLeaderboard:")
    for t, m in sorted(money.items(), key=lambda x: -x[1]):
        print(f"  {t:<18} {m:>12,.0f}" + ("   <==" if t.startswith("NOSOTROS") else ""))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
