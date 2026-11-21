"""Bot de rondas: encuentra la ronda, descifra, envía en varias pasadas y aprende.

  python bot.py               # bucle continuo
  python bot.py --game 0      # una pasada contra la ronda de prueba 0 (no guarda nada en la plataforma)

Envíos escalonados (se puede reenviar; cuenta el último antes del cierre):
  1. emergencia, en cuanto hay índices: a = 0, b según historial  -> nunca b = 0 por omisión
  2. LLM rápido (solo texto)                                      -> ~10-15 s
  3. LLM grande con imágenes, combinado con el rápido               -> antes del segundo 50
"""
import argparse
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import api
import case
import config
import decision
import learn
import llm
from belief import Belief


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def emergency_b(belief):
    """Sin estimación por línea: regla óptima con la distribución global de t del historial."""
    ts = np.array([r["t"] for r in belief.records])
    return decision.acceptance_limit(ts) if len(ts) >= 20 else config.EMERGENCY_B


def run_round(game_id, deadline):
    rounds = learn.load_rounds()
    belief = Belief(learn.records(rounds))
    params = learn.load_params()
    rec = {"game_id": game_id, "started": time.time(), "submissions": [], "params": params}

    key = api.get_key(game_id)
    root = case.extract(case.find_zip(game_id), key, game_id)
    c = case.load(root)
    log(f"ronda {game_id}: {len(c['indices'])} líneas (regex), {len(c['images'])} imágenes")

    def send(decisions, stage):
        if time.time() > deadline - 1.5:
            log(f"  {stage}: demasiado tarde, no envío")
            return
        try:
            api.submit(game_id, decisions)
            rec["submissions"].append({"stage": stage, "t": time.time(), "decisions": decisions})
            rec["final"] = {str(i): d for i, d in decisions.items()}
            log(f"  enviado [{stage}] {len(decisions)} líneas a {deadline - time.time():.0f}s del cierre")
        except Exception as e:
            log(f"  envío {stage} falló: {e}")

    # 1. emergencia
    eb = emergency_b(belief)
    if c["indices"]:
        send({i: {"charge_price": 0.0, "acceptance_limit": eb} for i in c["indices"]}, "emergencia")

    # 2 y 3. dos LLM en paralelo
    budget = max(5.0, deadline - time.time() - (config.ROUND_S - config.FINAL_SUBMIT_BY_S))
    pool = ThreadPoolExecutor(max_workers=2)
    f_fast = pool.submit(llm.call, c, config.LLM_FAST_MODEL, False, budget)
    f_full = pool.submit(llm.call, c, config.LLM_MODEL, True, budget)
    fast = full = None

    def decide_all(preds, stage):
        out = {}
        for i, p in preds.items():
            out[i] = learn.decide(belief, p, params, seed=i)
            out[i]["_why"] = p.get("why", "")
        rec["preds"] = {str(i): p for i, p in preds.items()}
        rec["decisions_" + stage] = {str(i): d for i, d in out.items()}
        # líneas que el regex vio y el LLM no: mantener la emergencia
        for i in c["indices"]:
            out.setdefault(i, {"charge_price": 0.0, "acceptance_limit": eb})
        return {i: {"charge_price": d["charge_price"], "acceptance_limit": d["acceptance_limit"]}
                for i, d in out.items()}

    try:
        fast = f_fast.result(timeout=budget)
        send(decide_all(fast, "rapido"), "rapido")
    except Exception as e:
        log(f"  LLM rápido falló: {e!r}")
    try:
        full = f_full.result(timeout=max(1.0, deadline - time.time() - (config.ROUND_S - config.FINAL_SUBMIT_BY_S)))
    except Exception as e:
        log(f"  LLM grande falló/tarde: {e!r}")
    if full:
        send(decide_all(llm.combine(fast, full), "completo"), "completo")
    pool.shutdown(wait=False, cancel_futures=True)

    for i, d in sorted((rec.get("decisions_completo") or rec.get("decisions_rapido") or {}).items(),
                       key=lambda x: int(x[0])):
        log(f"   #{i}: a={d['charge_price']:>9,.0f}  b={d['acceptance_limit']:>9,.0f}  "
            f"P(t=0)={d['p_zero']:.2f}  t~{d['t_median']:,.0f}  {d.get('_why', '')}")
    learn.save_round(game_id, rec)


def collect_results():
    """Rellena resultados de rondas pasadas y re-tunea."""
    changed = False
    for rd in learn.load_rounds():
        if rd.get("results") or rd["game_id"] in (0, "0"):
            continue
        try:
            res = api.get_results(rd["game_id"])
        except Exception as e:
            log(f"resultados {rd['game_id']}: {e}")
            continue
        if res:
            rd["results"] = {str(k): v for k, v in res.items()}
            learn.save_round(rd["game_id"], rd)
            changed = True
    if changed:
        threading.Thread(target=lambda: learn.autotune(verbose=False), daemon=True).start()


def loop():
    done = set()
    log("bot arrancado")
    while True:
        try:
            now = time.time()
            for g in api.list_games():
                if g["id"] in (0, "0") or g["id"] in done or not g["start"]:
                    continue
                if g["start"] <= now < g["end"] - 5:
                    done.add(g["id"])
                    run_round(g["id"], g["end"])
            collect_results()
        except Exception:
            traceback.print_exc()
        time.sleep(config.POLL_S)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=None)
    args = ap.parse_args()
    if args.game is not None:
        run_round(args.game, time.time() + config.ROUND_S)
    else:
        loop()
