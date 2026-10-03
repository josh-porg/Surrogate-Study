"""Common learner API and registry.

Every learner works on unit-cube inputs U in [0,1]^d (the pipeline maps physical inputs) and standardises y
internally. Metadata on each class records provenance for the paper's taxonomy and the coverage matrix:
`family`, `year` (first publication of the method), `field` (discipline it comes from), `uq` (native predictive
uncertainty), `smooth` (C^2 or better, i.e. usable by a gradient optimizer without upscaling).
"""
from __future__ import annotations

import numpy as np

REGISTRY: dict[str, type["Learner"]] = {}
PLANNED: dict[str, dict] = {}          # declared in the coverage matrix but not implemented yet


def register(cls):
    if cls.name in REGISTRY:
        raise KeyError(f"duplicate learner {cls.name}")
    REGISTRY[cls.name] = cls
    return cls


def planned(name, **meta):
    PLANNED[name] = meta


def make(name, **kw) -> "Learner":
    return REGISTRY[name](**kw)


class Learner:
    name = "base"
    family = ""
    year = None
    field = ""
    ref = ""            # bib key
    uq = False
    smooth = False
    max_n = None        # hard memory/time wall (None = no wall); cells beyond are recorded as infeasible
    max_d = None

    def __init__(self, seed: int = 0, **kw):
        self.seed = seed
        self.kw = kw

    # standardisation of the target ------------------------------------------------------------------------
    def _yfit(self, y):
        y = np.asarray(y, float)
        self._ym, self._ys = y.mean(), (y.std() if y.std() > 1e-12 else 1.0)
        return (y - self._ym) / self._ys

    def _yinv(self, z):
        return np.asarray(z, float) * self._ys + self._ym

    # API ---------------------------------------------------------------------------------------------------
    def fit(self, U, y):
        U = np.atleast_2d(np.asarray(U, float))
        self.d = U.shape[1]
        self._fit(U, self._yfit(y))
        return self

    def predict(self, U):
        return self._yinv(self._predict(np.atleast_2d(np.asarray(U, float))))

    def predict_std(self, U):
        s = self._predict_std(np.atleast_2d(np.asarray(U, float)))
        return None if s is None else np.asarray(s, float) * self._ys

    def grad(self, U, h=1e-5):
        """Default: central differences on the prediction (overridden where analytic gradients exist)."""
        U = np.atleast_2d(np.asarray(U, float))
        G = np.empty_like(U)
        for j in range(U.shape[1]):
            e = np.zeros(U.shape[1]); e[j] = h
            G[:, j] = (self.predict(U + e) - self.predict(U - e)) / (2 * h)
        return G

    def _fit(self, U, z): ...
    def _predict(self, U): ...
    def _predict_std(self, U):
        return None


def pairwise_dist(A, B):
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0.0))
