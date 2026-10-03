"""The 32 two-dimensional test functions of the author's Math 796 HW1 (Surjanovic & Bingham Virtual Library of
Simulation Experiments, https://www.sfu.ca/~ssurjano/), ported line-by-line from the MATLAB files in
data/reference/math796_hw1/optimization_test_functions/, vectorised over rows of X, plus their d-dimensional
versions where the VLSE definition is scalable (block P-VLSEd).

Domains are the VLSE "usual" domains. Published optima (fstar_ref) are only cross-checks: Problem.optimum() recomputes
them. Note that `powersum` with b = [8, 18] (the HW1 default for d = 2) has no real root (x1 + x2 = 8 and
x1^2 + x2^2 = 18 imply x1 x2 = 23 > 16), so its 2-D minimum is strictly positive.
"""
from __future__ import annotations

import numpy as np

from .base import Problem, box, register

PI = np.pi


# ── bowl-shaped ─────────────────────────────────────────────────────────────
def boha1(X):
    x1, x2 = X[:, 0], X[:, 1]
    return x1**2 + 2 * x2**2 - 0.3 * np.cos(3 * PI * x1) - 0.4 * np.cos(4 * PI * x2) + 0.7


def perm0db(X, b=10.0):
    d = X.shape[1]; j = np.arange(1, d + 1)
    out = 0.0
    for i in range(1, d + 1):
        inner = ((j + b) * (X**i - (1.0 / j) ** i)).sum(1)
        out = out + inner**2
    return out


def rothyp(X):
    return np.cumsum(X**2, axis=1).sum(1)


def sumsqu(X):
    return (np.arange(1, X.shape[1] + 1) * X**2).sum(1)


def trid(X):
    return ((X - 1) ** 2).sum(1) - (X[:, 1:] * X[:, :-1]).sum(1)


# ── many local minima ───────────────────────────────────────────────────────
def ackley(X, a=20.0, b=0.2, c=2 * PI):
    d = X.shape[1]
    return (-a * np.exp(-b * np.sqrt((X**2).sum(1) / d)) - np.exp(np.cos(c * X).sum(1) / d) + a + np.e)


def drop(X):
    r2 = X[:, 0] ** 2 + X[:, 1] ** 2
    return -(1 + np.cos(12 * np.sqrt(r2))) / (0.5 * r2 + 2)


_LANGER_C = np.array([1, 2, 5, 2, 3.0])
_LANGER_A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9.0]])


def langer(X):
    inner = ((X[:, None, :] - _LANGER_A[None]) ** 2).sum(2)           # (n, m)
    return (_LANGER_C * np.exp(-inner / PI) * np.cos(PI * inner)).sum(1)


def griewank(X):
    i = np.arange(1, X.shape[1] + 1)
    return (X**2).sum(1) / 4000 - np.prod(np.cos(X / np.sqrt(i)), axis=1) + 1


def levy(X):
    w = 1 + (X - 1) / 4
    t1 = np.sin(PI * w[:, 0]) ** 2
    t3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * PI * w[:, -1]) ** 2)
    mid = ((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(PI * w[:, :-1] + 1) ** 2)).sum(1)
    return t1 + mid + t3


def levy13(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (np.sin(3 * PI * x1) ** 2 + (x1 - 1) ** 2 * (1 + np.sin(3 * PI * x2) ** 2)
            + (x2 - 1) ** 2 * (1 + np.sin(2 * PI * x2) ** 2))


def rastr(X):
    return 10 * X.shape[1] + (X**2 - 10 * np.cos(2 * PI * X)).sum(1)


def schaffer2(X):
    x1, x2 = X[:, 0], X[:, 1]
    return 0.5 + (np.sin(x1**2 - x2**2) ** 2 - 0.5) / (1 + 0.001 * (x1**2 + x2**2)) ** 2


def shubert(X):
    i = np.arange(1, 6)
    s1 = (i * np.cos((i + 1) * X[:, [0]] + i)).sum(1)
    s2 = (i * np.cos((i + 1) * X[:, [1]] + i)).sum(1)
    return s1 * s2


# ── other ───────────────────────────────────────────────────────────────────
def beale(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2**2) ** 2 + (2.625 - x1 + x1 * x2**3) ** 2


def branin(X, a=1.0, b=5.1 / (4 * PI**2), c=5 / PI, r=6.0, s=10.0, t=1 / (8 * PI)):
    x1, x2 = X[:, 0], X[:, 1]
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def braninmodif(X):
    return branin(X) + 5 * X[:, 0]


def goldpr(X):
    x1, x2 = X[:, 0], X[:, 1]
    f1 = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
    f2 = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
    return f1 * f2


def permdb(X, b=0.5):
    d = X.shape[1]; j = np.arange(1, d + 1)
    out = 0.0
    for i in range(1, d + 1):
        out = out + (((j**i + b) * ((X / j) ** i - 1)).sum(1)) ** 2
    return out


def stybtang(X):
    return (X**4 - 16 * X**2 + 5 * X).sum(1) / 2


# ── plate-shaped ────────────────────────────────────────────────────────────
def booth(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def matya(X):
    x1, x2 = X[:, 0], X[:, 1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def mccorm(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1


_POWERSUM_B = {2: np.array([8, 18.0]), 4: np.array([8, 18, 44, 114.0])}


def powersum(X):
    d = X.shape[1]; b = _POWERSUM_B[d]
    return sum(((X**i).sum(1) - b[i - 1]) ** 2 for i in range(1, d + 1))


def zakharov(X):
    s2 = (0.5 * np.arange(1, X.shape[1] + 1) * X).sum(1)
    return (X**2).sum(1) + s2**2 + s2**4


# ── steep ridges / drops ────────────────────────────────────────────────────
_A5 = np.array([-32, -16, 0, 16, 32.0])
_DJ_A1 = np.tile(_A5, 5)
_DJ_A2 = np.repeat(_A5, 5)


def dejong5(X):
    i = np.arange(1, 26)
    s = (1.0 / (i + (X[:, [0]] - _DJ_A1) ** 6 + (X[:, [1]] - _DJ_A2) ** 6)).sum(1)
    return 1.0 / (0.002 + s)


def easom(X):
    x1, x2 = X[:, 0], X[:, 1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - PI) ** 2) - (x2 - PI) ** 2)


def michal(X, m=10):
    i = np.arange(1, X.shape[1] + 1)
    return -(np.sin(X) * np.sin(i * X**2 / PI) ** (2 * m)).sum(1)


# ── valley-shaped ───────────────────────────────────────────────────────────
def camel3(X):
    x1, x2 = X[:, 0], X[:, 1]
    return 2 * x1**2 - 1.05 * x1**4 + x1**6 / 6 + x1 * x2 + x2**2


def camel6(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2


def dixonpr(X):
    i = np.arange(2, X.shape[1] + 1)
    return (X[:, 0] - 1) ** 2 + (i * (2 * X[:, 1:] ** 2 - X[:, :-1]) ** 2).sum(1)


def rosen(X):
    return (100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (X[:, :-1] - 1) ** 2).sum(1)


# ── registration: (name, fn, category, domain(d), fstar_ref(d), xstar_ref(d), scalable) ─────
def _dom(lo, hi):
    return lambda d: box(lo, hi, d)


_SPECS = [
    # bowl-shaped
    ("boha1", boha1, "bowl_shaped", _dom(-100, 100), lambda d: 0.0, lambda d: np.zeros(d), False),
    ("perm0db", perm0db, "bowl_shaped", lambda d: box(-d, d, d), lambda d: 0.0, lambda d: 1.0 / np.arange(1, d + 1), True),
    ("rothyp", rothyp, "bowl_shaped", _dom(-65.536, 65.536), lambda d: 0.0, lambda d: np.zeros(d), True),
    ("sumsqu", sumsqu, "bowl_shaped", _dom(-10, 10), lambda d: 0.0, lambda d: np.zeros(d), True),
    ("trid", trid, "bowl_shaped", lambda d: box(-d**2, d**2, d), lambda d: -d * (d + 4) * (d - 1) / 6,
     lambda d: np.array([i * (d + 1 - i) for i in range(1, d + 1)], float), True),
    # many local minima
    ("ackley", ackley, "many_local_minima", _dom(-32.768, 32.768), lambda d: 0.0, lambda d: np.zeros(d), True),
    ("drop", drop, "many_local_minima", _dom(-5.12, 5.12), lambda d: -1.0, lambda d: np.zeros(d), False),
    ("griewank", griewank, "many_local_minima", _dom(-600, 600), lambda d: 0.0, lambda d: np.zeros(d), True),
    ("langer", langer, "many_local_minima", _dom(0, 10), lambda d: None, lambda d: None, False),
    ("levy", levy, "many_local_minima", _dom(-10, 10), lambda d: 0.0, lambda d: np.ones(d), True),
    ("levy13", levy13, "many_local_minima", _dom(-10, 10), lambda d: 0.0, lambda d: np.ones(d), False),
    ("rastr", rastr, "many_local_minima", _dom(-5.12, 5.12), lambda d: 0.0, lambda d: np.zeros(d), True),
    ("schaffer2", schaffer2, "many_local_minima", _dom(-100, 100), lambda d: 0.0, lambda d: np.zeros(d), False),
    ("shubert", shubert, "many_local_minima", _dom(-10, 10), lambda d: -186.7309, lambda d: None, False),
    # other
    ("beale", beale, "other", _dom(-4.5, 4.5), lambda d: 0.0, lambda d: np.array([3, 0.5]), False),
    ("branin", branin, "other", lambda d: np.array([[-5, 10], [0, 15.0]]), lambda d: 0.397887,
     lambda d: np.array([PI, 2.275]), False),
    ("braninmodif", braninmodif, "other", lambda d: np.array([[-5, 10], [0, 15.0]]), lambda d: -16.64402,
     lambda d: np.array([-3.68928, 13.62998]), False),
    ("goldpr", goldpr, "other", _dom(-2, 2), lambda d: 3.0, lambda d: np.array([0, -1.0]), False),
    ("permdb", permdb, "other", lambda d: box(-d, d, d), lambda d: 0.0, lambda d: np.arange(1, d + 1, dtype=float), True),
    ("stybtang", stybtang, "other", _dom(-5, 5), lambda d: -39.16599 * d, lambda d: np.full(d, -2.903534), True),
    # plate-shaped
    ("booth", booth, "plate_shaped", _dom(-10, 10), lambda d: 0.0, lambda d: np.array([1, 3.0]), False),
    ("matya", matya, "plate_shaped", _dom(-10, 10), lambda d: 0.0, lambda d: np.zeros(d), False),
    ("mccorm", mccorm, "plate_shaped", lambda d: np.array([[-1.5, 4], [-3, 4.0]]), lambda d: -1.9133,
     lambda d: np.array([-0.54719, -1.54719]), False),
    ("powersum", powersum, "plate_shaped", lambda d: box(0, d, d), lambda d: None, lambda d: None, False),
    ("zakharov", zakharov, "plate_shaped", _dom(-5, 10), lambda d: 0.0, lambda d: np.zeros(d), True),
    # steep ridges / drops
    ("dejong5", dejong5, "steep_ridges_and_drops", _dom(-65.536, 65.536), lambda d: None, lambda d: None, False),
    ("easom", easom, "steep_ridges_and_drops", _dom(-100, 100), lambda d: -1.0, lambda d: np.array([PI, PI]), False),
    ("michal", michal, "steep_ridges_and_drops", _dom(0, PI), lambda d: -1.8013 if d == 2 else None,
     lambda d: np.array([2.20, 1.57]) if d == 2 else None, True),
    # valley-shaped
    ("camel3", camel3, "valley_shaped", _dom(-5, 5), lambda d: 0.0, lambda d: np.zeros(d), False),
    ("camel6", camel6, "valley_shaped", lambda d: np.array([[-3, 3], [-2, 2.0]]), lambda d: -1.0316,
     lambda d: np.array([0.0898, -0.7126]), False),
    ("dixonpr", dixonpr, "valley_shaped", _dom(-10, 10), lambda d: 0.0, None, True),
    ("rosen", rosen, "valley_shaped", _dom(-5, 10), lambda d: 0.0, lambda d: np.ones(d), True),
]

SCALABLE_DIMS = (3, 5, 8, 10)

for name, fn, cat, dom, fs, xs, scalable in _SPECS:
    register(Problem(name=name, fn=fn, bounds=dom(2), block="P-VLSE2", category=cat, tags=("vlse", cat),
                     fstar_ref=fs(2), xstar_ref=None if xs is None else xs(2),
                     desc=f"VLSE {name} (Math 796 HW1), d=2"))
    if scalable:
        for d in SCALABLE_DIMS:
            register(Problem(name=f"{name}_d{d}", fn=fn, bounds=dom(d), block="P-VLSEd", category=cat,
                             tags=("vlse", cat, "scalable"), fstar_ref=fs(d),
                             xstar_ref=None if xs is None else xs(d), desc=f"VLSE {name}, d={d}"))
