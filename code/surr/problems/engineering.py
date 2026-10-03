"""Engineering emulation functions (block P-ENG) from the Surjanovic & Bingham library, and the six canonical
surfaces of Poznański (2026), M.S. thesis, Univ. of Kansas, App. G (block P-THESIS), ported from
COMPASS/compass/surrogate/canonical.py (read-only source)."""
from __future__ import annotations

import numpy as np

from .base import Problem, register


def borehole(X):
    rw, r, Tu, Hu, Tl, Hl, L, Kw = X.T
    lr = np.log(r / rw)
    return 2 * np.pi * Tu * (Hu - Hl) / (lr * (1 + 2 * L * Tu / (lr * rw**2 * Kw) + Tu / Tl))


def otlcircuit(X):
    Rb1, Rb2, Rf, Rc1, Rc2, beta = X.T
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    den = beta * (Rc2 + 9) + Rf
    return ((Vb1 + 0.74) * beta * (Rc2 + 9) / den + 11.35 * Rf / den + 0.74 * Rf * beta * (Rc2 + 9) / (den * Rc1))


def piston(X):
    M, S, V0, k, P0, Ta, T0 = X.T
    A = P0 * S + 19.62 * M - k * V0 / S
    V = S / (2 * k) * (np.sqrt(A**2 + 4 * k * P0 * V0 * Ta / T0) - A)
    return 2 * np.pi * np.sqrt(M / (k + S**2 * P0 * V0 * Ta / (T0 * V**2)))


def wingweight(X):
    Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = X.T
    L = np.deg2rad(Lam)
    return (0.036 * Sw**0.758 * Wfw**0.0035 * (A / np.cos(L) ** 2) ** 0.6 * q**0.006 * lam**0.04
            * (100 * tc / np.cos(L)) ** -0.3 * (Nz * Wdg) ** 0.49 + Sw * Wp)


def currin(X):
    x1, x2 = X[:, 0], X[:, 1]
    return ((1 - np.exp(-1 / (2 * x2))) * (2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60)
            / (100 * x1**3 + 500 * x1**2 + 4 * x1 + 20))


def park(X):
    x1, x2, x3, x4 = X.T
    return x1 / 2 * (np.sqrt(1 + (x2 + x3**2) * x4 / x1**2) - 1) + (x1 + 3 * x4) * np.exp(1 + np.sin(x3))


_ENG = [
    ("borehole", borehole, [[0.05, 0.15], [100, 50000], [63070, 115600], [990, 1110], [63.1, 116],
                            [700, 820], [1120, 1680], [9855, 12045]], "groundwater flow rate"),
    ("otlcircuit", otlcircuit, [[50, 150], [25, 70], [0.5, 3], [1.2, 2.5], [0.25, 1.2], [50, 300]],
     "output-transformerless push-pull circuit midpoint voltage"),
    ("piston", piston, [[30, 60], [0.005, 0.020], [0.002, 0.010], [1000, 5000], [90000, 110000], [290, 296],
                        [340, 360]], "piston cycle time"),
    ("wingweight", wingweight, [[150, 200], [220, 300], [6, 10], [-10, 10], [16, 45], [0.5, 1], [0.08, 0.18],
                                [2.5, 6], [1700, 2500], [0.025, 0.08]], "light-aircraft wing weight (Raymer)"),
    ("currin", currin, [[0, 1], [1e-3, 1]], "Currin exponential (x2 bounded away from 0)"),
    ("park", park, [[0.01, 1], [0, 1], [0, 1], [0, 1]], "Park (1991), x1 bounded away from 0"),
]
for name, fn, b, desc in _ENG:
    register(Problem(name=name, fn=fn, bounds=np.array(b, float), block="P-ENG", category="engineering",
                     tags=("engineering", "blind"), desc=desc))


# ── Poznański (2026) App. G canonical surfaces on (R, p, T) ─────────────────────────────────
R_LO, R_HI, P_LO, P_HI, T_LO, T_HI = 0.03, 0.09, 5.0e4, 2.0e6, 300.0, 900.0


def _norm(X):
    R, p, T = X[:, 0], np.maximum(X[:, 1], 1.0), X[:, 2]
    return (R, (R - R_LO) / (R_HI - R_LO), (np.log(p) - np.log(P_LO)) / (np.log(P_HI) - np.log(P_LO)),
            (T - T_LO) / (T_HI - T_LO))


def t_smooth(X):
    R, rho, lp, tT = _norm(X); return 1.30 + 0.30 * np.sin(3.0 * rho) + 0.15 * lp - 0.10 * tT


def t_collapse_knee(X):
    R, rho, lp, tT = _norm(X)
    return 1.10 + 0.80 * np.maximum(0.045 / np.maximum(R, 1e-9) - 1.0, 0.0) + 0.12 * lp - 0.06 * tT


def t_clamp_floor(X):
    R, rho, lp, tT = _norm(X)
    return np.maximum(1.05 + 0.45 * np.sin(2.5 * rho) + 0.20 * lp - 0.15 * tT, 1.25)


def t_anisotropic(X):
    R, rho, lp, tT = _norm(X)
    return 1.30 + 0.55 * lp + 0.10 * lp**2 + 0.05 * np.sin(2.0 * rho) + 0.01 * tT


def t_oscillatory(X):
    R, rho, lp, tT = _norm(X)
    return 1.35 + 0.25 * np.sin(9.0 * rho) * np.cos(7.0 * lp) + 0.08 * tT


def t_cliff(X):
    R, rho, lp, tT = _norm(X)
    return 1.05 + 0.60 / (1.0 + np.exp(-14.0 * (rho - 0.5))) + 0.10 * lp


# p is sampled log-uniformly in the thesis; the Problem box is physical, and samplers that want the thesis
# convention apply the log map via `log_axes`.
_TB = np.array([[R_LO, R_HI], [P_LO, P_HI], [T_LO, T_HI]])
for name, fn, tags in [("thesis_smooth", t_smooth, ("smooth",)),
                       ("thesis_collapse_knee", t_collapse_knee, ("knee", "kink", "localized")),
                       ("thesis_clamp_floor", t_clamp_floor, ("kink", "floor")),
                       ("thesis_anisotropic", t_anisotropic, ("anisotropic",)),
                       ("thesis_oscillatory", t_oscillatory, ("oscillatory", "fourier_sparse")),
                       ("thesis_cliff", t_cliff, ("steep", "near_discontinuity"))]:
    p = register(Problem(name=name, fn=fn, bounds=_TB.copy(), block="P-THESIS", category="thesis",
                         tags=tags, desc="Poznański (2026) App. G canonical surface"))
    p.log_axes = (1,)
