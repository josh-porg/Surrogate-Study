"""Ensembles of surrogates.

  ens_press   PRESS-weighted average of heterogeneous surrogates           Goel, Haftka, Shyy & Queipo 2007
              w_m ∝ (E_m + α Ē)^β, α = 0.05, β = −1, E_m = k-fold CV RMSE    (MDO / SMO community)
"""
from __future__ import annotations

import numpy as np

from .base import Learner, make, register


@register
class EnsemblePRESS(Learner):
    name, family, year, field, ref, uq = "ens_press", "ensemble of surrogates", 2007, "multidisciplinary design optimization", "goel2007ensemble", True

    def __init__(self, members=("gp", "rbf_tps", "rsm2", "xgb"), alpha=0.05, beta=-1.0, k=5, **kw):
        super().__init__(**kw); self.members, self.alpha, self.beta, self.k = members, alpha, beta, k

    def _fit(self, U, z):
        rng = np.random.default_rng(self.seed); folds = np.array_split(rng.permutation(len(z)), min(self.k, len(z)))
        E = []
        for m in self.members:
            err = []
            for f in folds:
                tr = np.setdiff1d(np.arange(len(z)), f)
                try:
                    err.append(np.mean((make(m, seed=self.seed).fit(U[tr], z[tr]).predict(U[f]) - z[f]) ** 2))
                except Exception:
                    err.append(np.inf)
            E.append(np.sqrt(np.mean(err)))
        E = np.array(E); fin = np.isfinite(E)
        w = np.zeros_like(E); w[fin] = (E[fin] + self.alpha * E[fin].mean()) ** self.beta
        self._w = w / w.sum(); self.cv_rmse_ = dict(zip(self.members, E))
        self._ms = [make(m, seed=self.seed).fit(U, z) if wi > 0 else None for m, wi in zip(self.members, self._w)]

    def _all(self, U):
        return np.vstack([m.predict(U) if m is not None else np.zeros(len(U)) for m in self._ms])

    def _predict(self, U):
        return self._w @ self._all(U)

    def _predict_std(self, U):
        P = self._all(U); mu = self._w @ P
        return np.sqrt(np.maximum(self._w @ (P - mu) ** 2, 0))
