"""Small numpy-only numerics, so the agent needs no scipy at run time.

erfc: Chebyshev fit from Numerical Recipes (erfcc), fractional error < 1.2e-7 everywhere,
so normal tail probabilities stay accurate in relative terms far into the tails.
"""

from __future__ import annotations

import numpy as np

_SQRT2 = np.sqrt(2.0)


def erfc(x):
    x = np.asarray(x, dtype=float)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = (-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
            + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
            + t * (-0.82215223 + t * 0.17087277)))))))))
    r = t * np.exp(poly)
    return np.where(x >= 0, r, 2.0 - r)


def ndtr(x):
    """Standard normal CDF."""
    return 0.5 * erfc(-np.asarray(x, dtype=float) / _SQRT2)


def norm_interval(lo, hi):
    """P(lo <= Z < hi) for standard normal Z, accurate in both tails."""
    lo, hi = np.broadcast_arrays(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    upper = lo > 0                       # both in the upper tail: use survival functions
    sf = lambda x: 0.5 * erfc(x / _SQRT2)  # noqa: E731
    return np.where(upper, sf(lo) - sf(hi), ndtr(hi) - ndtr(lo))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def minimize_nm(f, x0, bounds, iters: int = 200, tol: float = 1e-6) -> np.ndarray:
    """Nelder-Mead with box bounds (by clipping). Enough for the 3-parameter M-step."""
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    clip = lambda x: np.clip(x, lo, hi)  # noqa: E731
    n = len(x0)
    pts = [clip(np.asarray(x0, dtype=float))]
    for i in range(n):
        p = pts[0].copy()
        p[i] += 0.25 if p[i] + 0.25 <= hi[i] else -0.25
        pts.append(clip(p))
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol * (1 + abs(vals[0])):
            break
        c = np.mean(pts[:-1], axis=0)
        xr = clip(c + (c - pts[-1]))
        fr = f(xr)
        if fr < vals[0]:
            xe = clip(c + 2 * (c - pts[-1]))
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = clip(c + 0.5 * (pts[-1] - c))
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [clip(pts[0] + 0.5 * (p - pts[0])) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    return pts[int(np.argmin(vals))]
