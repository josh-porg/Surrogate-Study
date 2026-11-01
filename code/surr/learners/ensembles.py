"""Ensemble intermediate models added for E15 (the surrogate–optimizer interface study).

  ens_optw  ensemble of surrogates with weights optimised to minimise the k-fold cross-validation MSE of the
            weighted combination (w >= 0, Σ w = 1), same members as ens_press      Acar & Rais-Rohani 2009
  mlp_rpf   MLP ensemble with randomised prior functions: member k = trainable net + β · fixed random net, each
            trained on a bootstrap sample; the prior keeps members apart away from the data
            (uncertainty where plain bootstrapping has none)                     Osband, Aslanides & Cassirer 2018
surr.learners.ensembles.members(m) returns the per-member predictors of any ensemble learner (E15).
"""
from __future__ import annotations

import numpy as np

from .base import Learner, make, register
from .meta import EnsemblePRESS
from .nets import MLPEnsemble, _mlp, _torch, _train


@register
class EnsembleOptW(EnsemblePRESS):
    name, family, year, field, ref, uq = "ens_optw", "ensemble of surrogates", 2009, "multidisciplinary design optimization", "acar2009ensemble", True

    def _fit(self, U, z):
        from scipy.optimize import minimize
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(len(z)), min(self.k, len(z)))
        P = np.full((len(self.members), len(z)), np.nan)                 # out-of-fold predictions
        for i, m in enumerate(self.members):
            for f in folds:
                tr = np.setdiff1d(np.arange(len(z)), f)
                try:
                    P[i, f] = make(m, seed=self.seed).fit(U[tr], z[tr]).predict(U[f])
                except Exception:
                    pass
        ok = np.all(np.isfinite(P), axis=1)
        Q = P[ok]
        obj = lambda w: float(np.mean((w @ Q - z) ** 2))
        k = ok.sum()
        r = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        w = np.zeros(len(self.members)); w[ok] = np.clip(r.x, 0, None)
        self._w = w / w.sum()
        self.cv_mse_ = obj(r.x)
        self._ms = [make(m, seed=self.seed).fit(U, z) if wi > 1e-6 else None for m, wi in zip(self.members, self._w)]

    def member_predictors(self):
        """(weight, predictor) pairs on the original scale (`members` is the tuple of member names)."""
        return [(wi, (lambda U, m=m: self._yinv(m.predict(np.atleast_2d(U))))) for wi, m in zip(self._w, self._ms)
                if m is not None]


@register
class MLPRandomPrior(MLPEnsemble):
    name, family, year, field, ref, uq, smooth = "mlp_rpf", "neural network", 2018, "machine learning", "osband2018randomized", True, True

    def _fit(self, U, z):
        torch = _torch(); rng = np.random.default_rng(self.seed)
        beta = float(self.kw.get("beta", 1.0))
        self._nets, self._priors = [], []
        for k in range(self.K):
            torch.manual_seed(self.seed * 100 + 50 + k)
            prior = _mlp(U.shape[1]).eval()
            for p in prior.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                pz = beta * prior(torch.tensor(U, dtype=torch.float32)).squeeze(-1).numpy()
            idx = rng.integers(0, len(z), len(z))
            net = _train(_mlp(U.shape[1]), U[idx], (z - pz)[idx], self.seed + k, opt=self.kw.get("opt", "lbfgs"))
            self._nets.append(net); self._priors.append((beta, prior))

    def _all(self, U):
        torch = _torch()
        with torch.no_grad():
            X = torch.tensor(U, dtype=torch.float32)
            return np.vstack([n.eval()(X).squeeze(-1).numpy() + b * p(X).squeeze(-1).numpy()
                              for n, (b, p) in zip(self._nets, self._priors)])


def members(m):
    """Uniform-weight member predictors of a fitted ensemble learner, on the original scale; None if the learner
    has no members. Used by E15's Thompson and member-sampling strategies."""
    if m.name == "ens_optw":
        return m.member_predictors()
    if m.name in ("ens_press",):
        return [(wi, (lambda U, s=s: m._yinv(s.predict(np.atleast_2d(U))))) for wi, s in zip(m._w, m._ms)
                if s is not None]
    if m.name in ("rf", "extratrees"):
        T = m._m.estimators_
        return [(1 / len(T), (lambda U, t=t: m._yinv(t.predict(np.atleast_2d(U))))) for t in T]
    if hasattr(m, "_all") and m.name in ("mlp_ens", "mlp_rpf", "mlp_relu", "gbt_committee"):
        K = len(m._all(np.full((1, m.d), 0.5)))
        return [(1 / K, (lambda U, k=k: m._yinv(m._all(np.atleast_2d(U))[k]))) for k in range(K)]
    return None
