"""Decisión por línea: acceptance limit b (aseguradora) y charge price a (reparador).

Pagos entre un reparador H (precio a) y una aseguradora I (límite b), valor justo t:
    a <= t, a <= b : I paga a              (justo, aceptado)
    a <= t, a >  b : I paga 1.5 a          (justo, rechazado -> penalización)
    a >  t, a <= b : I paga min(a, c)      (inflado, aceptado)
    a >  t, a >  b : nada                  (fraude cazado)

Los dos papeles son separables: b solo afecta a lo que pagas, a solo a lo que cobras.

b óptimo: para un cargo a < c, aceptar cuesta a; rechazar cuesta 1.5a·P(t >= a).
Se acepta si P(t >= a) > 1/1.5 = 2/3  ->  b* = cuantil 1/3 de tu creencia sobre t.
No depende de lo que hagan los demás. Si P(t = 0) >= 1/3, b* = 0.

a óptimo: sí depende de los demás. Maximiza el cobro esperado frente a los b ajenos:
    R(a) = P(t>=a)·a·[1 + 0.5·P(b_j < a)] + P(t<a)·min(a,c)·P(b_j >= a)
Se resuelve por Monte Carlo con muestras conjuntas de (t, b_j).
"""
import numpy as np

import config


def cap_of(t):
    """Tope de pago c >= 4t, con suelo absoluto (GAME_DESCRIPTION)."""
    return np.maximum(config.CAP_MULT * np.asarray(t, float), config.CAP_FLOOR)


def revenue(a, t, b, cap=None):
    """Lo que cobra el reparador con precio a frente a una aseguradora con límite b."""
    cap = cap_of(t) if cap is None else cap
    fair = a <= t
    acc = a <= b
    return np.where(fair, np.where(acc, a, config.PENALTY * a),
                    np.where(acc, np.minimum(a, cap), 0.0))


def cost(b, a, t, cap=None):
    """Lo que paga la aseguradora con límite b ante un cargo a."""
    return revenue(a, t, b, cap)


def _grid(t, lo_extra=None):
    pos = t[t > 0]
    if len(pos) == 0:
        return np.array([0.0])
    lo, hi = np.quantile(pos, 0.01) * 0.5, np.quantile(pos, 0.995) * 1.5
    if lo_extra:
        hi = max(hi, lo_extra)
    return np.concatenate([[0.0], np.exp(np.linspace(np.log(lo), np.log(hi), config.GRID_SIZE))])


def acceptance_limit(t, cap=None):
    """b*: el mayor a tal que aceptar todos los cargos <= a es mejor que rechazarlos."""
    grid = _grid(t)[1:]
    if len(grid) == 0:
        return 0.0
    c = cap_of(t)[None, :] if cap is None else cap
    fair = t[None, :] >= grid[:, None]
    p_fair = fair.mean(1)
    acc_cost = (grid * p_fair) + (np.minimum(grid[:, None], c) * ~fair).mean(1)
    rej_cost = config.PENALTY * grid * p_fair
    ok = acc_cost <= rej_cost
    if not ok[0]:
        return 0.0
    first_bad = np.argmin(ok) if not ok.all() else len(ok)
    return float(grid[first_bad - 1])


def charge_price(t, others_b, cap=None):
    """a*: maximiza el cobro medio por equipo rival.

    t: (n,) muestras del valor justo. others_b: (n, k) límites de k rivales en la misma
    muestra (correlacionados con t). Devuelve (a, cobro esperado por rival).
    """
    grid = _grid(t)
    best_a, best_r = 0.0, 0.0
    for a in grid[1:]:
        r = revenue(a, t[:, None], others_b, cap).mean()
        if r > best_r:
            best_a, best_r = float(a), float(r)
    return best_a, best_r


def decide_line(t, others_b, cap=None):
    a, ra = charge_price(t, others_b, cap)
    b = acceptance_limit(t, cap)
    return {"charge_price": round(a, 2), "acceptance_limit": round(b, 2),
            "exp_revenue_per_team": round(ra, 2),
            "p_zero": float((t == 0).mean()),
            "t_median": float(np.median(t)),
            "t_q33": float(np.quantile(t, 1 / 3))}
