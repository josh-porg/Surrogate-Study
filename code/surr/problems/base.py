"""Problem abstraction shared by every test function, feature probe and dataset.

A Problem maps physical inputs X (n, d) inside `bounds` to outputs y (n,). Studies work on the unit cube; `from_unit`
/ `to_unit` convert. Every problem carries tags used by the coverage matrix (docs/PROBLEM_MATRIX.md) and, where known,
the global minimiser for optimizer regret. When the optimum is not known in closed form it is computed numerically
(dense Sobol' scan + multistart L-BFGS-B) and cached — so regret never depends on a transcribed constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

REGISTRY: dict[str, "Problem"] = {}


@dataclass
class Problem:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]          # physical X (n, d) -> y (n,)
    bounds: np.ndarray                               # (d, 2)
    block: str                                       # P-VLSE2, P-VLSEd, P-ENG, P-THESIS, P-FEAT, P-MF
    category: str = ""                               # e.g. VLSE shape class or probe mechanism
    tags: tuple = ()                                 # free-form features used by coverage / ELA checks
    fstar_ref: float | None = None                   # published optimum (checked, not trusted)
    xstar_ref: np.ndarray | None = None
    desc: str = ""
    _fstar: float | None = field(default=None, repr=False)
    _xstar: np.ndarray | None = field(default=None, repr=False)

    @property
    def d(self) -> int:
        return self.bounds.shape[0]

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return np.asarray(self.fn(X), float).reshape(-1)

    # -- unit-cube helpers ----------------------------------------------------
    def from_unit(self, U):
        U = np.atleast_2d(U)
        return self.bounds[:, 0] + U * (self.bounds[:, 1] - self.bounds[:, 0])

    def to_unit(self, X):
        X = np.atleast_2d(X)
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def f_unit(self, U):
        return self(self.from_unit(U))

    # -- gradient (central differences in physical units, step scaled to the box) --------------
    def grad(self, X, rel_h: float = 1e-6):
        X = np.atleast_2d(np.asarray(X, float))
        h = rel_h * (self.bounds[:, 1] - self.bounds[:, 0])
        G = np.empty_like(X)
        for j in range(self.d):
            e = np.zeros(self.d); e[j] = h[j]
            G[:, j] = (self(X + e) - self(X - e)) / (2 * h[j])
        return G

    # -- optimum ------------------------------------------------------------------------------
    def optimum(self, n_scan: int = 2 ** 14, n_starts: int = 32, seed: int = 0):
        """(f*, x*) computed numerically; the published value, if any, is used only as a cross-check."""
        if self._fstar is None:
            from scipy.optimize import minimize
            from scipy.stats import qmc
            U = qmc.Sobol(self.d, scramble=True, seed=seed).random(n_scan)
            X = self.from_unit(U)
            y = self(X)
            starts = X[np.argsort(y)[:n_starts]]
            if self.xstar_ref is not None:
                starts = np.vstack([starts, np.atleast_2d(self.xstar_ref)])
            best = (np.inf, None)
            for x0 in starts:
                r = minimize(lambda x: float(self(x[None])[0]), x0, method="L-BFGS-B",
                             bounds=[tuple(b) for b in self.bounds])
                if r.fun < best[0]:
                    best = (float(r.fun), r.x)
            self._fstar, self._xstar = best
        return self._fstar, self._xstar


def shifted(p: Problem, seed: int, frac: float = 0.2) -> Problem:
    """Instance with the input shifted by a random vector of up to `frac` of each range (BBOB-style instances).
    Removes the centre-of-box bias: many VLSE optima sit at the box centre, where DIRECT and space-filling designs
    sample first (BUGS N-01). Registered on first use; same seed = same instance."""
    name = f"{p.name}@s{seed}"
    if name in REGISTRY:
        return REGISTRY[name]
    w = p.bounds[:, 1] - p.bounds[:, 0]
    s = np.random.default_rng(seed).uniform(-frac, frac, p.d) * w
    xs = None if p.xstar_ref is None else np.asarray(p.xstar_ref) + s
    q = Problem(name=name, fn=lambda X, f=p.fn, s=s: f(X - s), bounds=p.bounds.copy(), block=p.block,
                category=p.category, tags=tuple(p.tags) + ("shifted",), fstar_ref=None, xstar_ref=xs,
                desc=f"{p.desc}; shifted instance seed {seed}")
    q.shift = s
    return register(q)


def register(p: Problem) -> Problem:
    if p.name in REGISTRY:
        raise KeyError(f"duplicate problem {p.name}")
    REGISTRY[p.name] = p
    return p


def box(lo, hi, d):
    return np.tile([float(lo), float(hi)], (d, 1))
