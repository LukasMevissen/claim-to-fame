"""Creencia sobre cada línea: muestras conjuntas de t (valor justo) y de los límites b
de los rivales, a partir de la salida del LLM y de lo aprendido en rondas anteriores.

  t = 0                              con prob. p0 (no cubierto / no relacionado)
  t = mid · exp(sd · z)              si no, z de los residuos históricos estandarizados

Con pocos datos se mezcla con un prior; cada ronda resuelta pesa más lo empírico.
Los límites de los rivales se remuestrean de rondas pasadas (relativos a NUESTRA
estimación mid de entonces), junto con el residuo de esa misma línea: así se conserva
la correlación "cuando yo me equivoco, ellos también".
"""
import math

import numpy as np

N0 = 10          # peso del prior en "líneas equivalentes"
PRIOR_K = 15     # nº de rivales supuesto si no hay historial
PRIOR_OTHERS_LOGB = (-0.10, 0.30)   # rivales: b ≈ t·exp(N(-0.10, 0.30))
PRIOR_OTHERS_SPOT_ZERO = 0.5        # prob. de que un rival detecte una línea excluida (b = 0)


def logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def line_sd(low, high, default=0.35):
    if not low or not high or low <= 0 or high <= low:
        return default
    return float(np.clip(math.log(high / low) / 2.563, 0.08, 1.2))   # intervalo 10-90 %


class Belief:
    def __init__(self, records=()):
        """records: líneas resueltas con mid, sd, p_bad, t, others_b (lista)."""
        self.records = [r for r in records if r.get("t") is not None and r.get("mid")]
        pos = [r for r in self.records if r["t"] > 0]
        zero = [r for r in self.records if r["t"] == 0]
        self.z_rows = [(math.log(r["t"] / r["mid"]) / r["sd"],
                        np.array(r.get("others_b") or [], float) / r["mid"]) for r in pos]
        self.zero_rows = [np.array(r.get("others_b") or [], float) / r["mid"] for r in zero]
        ks = [len(r.get("others_b") or []) for r in self.records if r.get("others_b")]
        self.k = int(np.median(ks)) if ks else PRIOR_K
        self._fit_p0()

    # ------------------------------------------------ prob. de que la línea valga 0
    def _fit_p0(self):
        self.p0_coef = None
        y = np.array([r["t"] == 0 for r in self.records], float)
        if len(y) >= 25 and 0 < y.sum() < len(y):
            from sklearn.linear_model import LogisticRegression
            X = np.array([[logit(r["p_bad"])] for r in self.records])
            m = LogisticRegression(C=1.0).fit(X, y)
            self.p0_coef = (float(m.intercept_[0]), float(m.coef_[0, 0]))

    def p_zero(self, p_bad):
        if self.p0_coef is None:
            return 0.03 + 0.94 * p_bad          # se fía del LLM pero sin extremos
        a, b = self.p0_coef
        return 1 / (1 + math.exp(-(a + b * logit(p_bad))))

    # ------------------------------------------------ muestreo
    def sample(self, mid, sd, p_bad, n=1500, seed=None):
        rng = np.random.default_rng(seed)
        k = self.k
        t = np.empty(n)
        B = np.empty((n, k))
        is_zero = rng.random(n) < self.p_zero(p_bad)
        n_pos = len(self.z_rows)
        n_zero = len(self.zero_rows)
        for i in range(n):
            if not is_zero[i]:
                if rng.random() < n_pos / (n_pos + N0):
                    z, ratios = self.z_rows[rng.integers(n_pos)]
                    t[i] = mid * math.exp(sd * z)
                    B[i] = mid * rng.choice(ratios, k) if len(ratios) else t[i] * self._prior_b(rng, k)
                else:
                    z = rng.standard_t(5) * math.sqrt(3 / 5)
                    t[i] = mid * math.exp(sd * z)
                    B[i] = t[i] * self._prior_b(rng, k)
            else:
                t[i] = 0.0
                if n_zero and rng.random() < n_zero / (n_zero + N0) and len(self.zero_rows[j := rng.integers(n_zero)]):
                    B[i] = mid * rng.choice(self.zero_rows[j], k)
                else:
                    spot = rng.random(k) < PRIOR_OTHERS_SPOT_ZERO
                    B[i] = np.where(spot, 0.0, mid * self._prior_b(rng, k))
        return t, B

    @staticmethod
    def _prior_b(rng, k):
        mu, s = PRIOR_OTHERS_LOGB
        return np.exp(rng.normal(mu, s, k))
