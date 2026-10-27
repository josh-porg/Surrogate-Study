"""Multi-fidelity surrogate methods. API: fit(UL, yL, UH, yH) then predict(U) [, predict_std(U)] on the unit cube.
`base` selects the learner used inside (any surr learner name) where the method allows it.

  hf_only               learner on high-fidelity data only (control)
  bridge_mult           f_H ≈ β(x)·f̂_L(x), β learned by `base` on y_H / f̂_L      Chang et al. 1993; Poznański 2026 TMF
  bridge_add            f_H ≈ f̂_L(x) + δ(x)                                       additive correction
  ar1_cokriging         recursive AR1 co-kriging: f_H = ρ f̂_L + δ, δ ~ GP, ρ by ML  Kennedy & O'Hagan 2000; Le Gratiet 2014
  hierarchical_kriging  f_H = β0 f̂_L + Z, β0 by GLS inside the kriging trend      Han & Görtz 2012
  nargp                 GP on augmented inputs (x, f̂_L(x)) with ARD Matérn          Perdikaris et al. 2017 (augmented-input
                        (sklearn lacks active-dims product kernels; documented)      approximation)
  feature_aug           any learner on (x, f̂_L(x)) — default XGBoost                 (rarely benchmarked; see SOURCES §5)
  space_mapping         f_H(x) ≈ s·f̂_L(Ax + b) + c, then GP on the residual          Bandler et al. 1994
  mf_nn                 composite MF network: linear + L2-penalised nonlinear map of  Meng & Karniadakis 2020
                        (x, NN_L(x))
"""
from __future__ import annotations

import numpy as np

from .learners import make
from .learners.kernels import _sk_gp

REGISTRY = {}


def _reg(cls):
    REGISTRY[cls.name] = cls
    return cls


class MFMethod:
    name = "base"

    def __init__(self, base="gp", lf_learner="gp", seed=0):
        self.base, self.lf_learner, self.seed = base, lf_learner, seed

    def _fit_low(self, UL, yL):
        self.low = make(self.lf_learner, seed=self.seed).fit(UL, yL)

    def predict_std(self, U):
        return None


@_reg
class HFOnly(MFMethod):
    name = "hf_only"

    def fit(self, UL, yL, UH, yH):
        self.m = make(self.base, seed=self.seed).fit(UH, yH); return self

    def predict(self, U):
        return self.m.predict(U)

    def predict_std(self, U):
        return self.m.predict_std(U)


@_reg
class BridgeMult(MFMethod):
    name = "bridge_mult"

    def fit(self, UL, yL, UH, yH):
        self._fit_low(UL, yL)
        fl = self.low.predict(UH)
        self._shift = 0.0 if np.min(np.abs(fl)) > 1e-6 * np.abs(fl).max() and np.all(np.sign(fl) == np.sign(fl[0])) \
            else (1.0 - fl.min())                                           # keep the ratio well defined
        self.m = make(self.base, seed=self.seed).fit(UH, (yH + self._shift) / (fl + self._shift)); return self

    def predict(self, U):
        return self.m.predict(U) * (self.low.predict(U) + self._shift) - self._shift


@_reg
class BridgeAdd(MFMethod):
    name = "bridge_add"

    def fit(self, UL, yL, UH, yH):
        self._fit_low(UL, yL)
        self.m = make(self.base, seed=self.seed).fit(UH, yH - self.low.predict(UH)); return self

    def predict(self, U):
        return self.low.predict(U) + self.m.predict(U)


@_reg
class AR1CoKriging(MFMethod):
    """Recursive formulation: GP_L on LF data; ρ maximises the GP log marginal likelihood of y_H − ρ μ_L(X_H)."""
    name = "ar1_cokriging"

    def fit(self, UL, yL, UH, yH):
        from scipy.optimize import minimize_scalar
        self._fit_low(UL, yL)
        mL = self.low.predict(UH); sc = yH.std() or 1.0
        def nll(rho):
            g = _sk_gp(UH.shape[1], 2.5, 0, self.seed).fit(UH, (yH - rho * mL) / sc)
            return -g.log_marginal_likelihood_value_
        r = minimize_scalar(nll, bounds=(-3, 3), method="bounded", options=dict(xatol=1e-3))
        self.rho = float(r.x)
        self._sc = sc
        self.delta = _sk_gp(UH.shape[1], 2.5, 2, self.seed).fit(UH, (yH - self.rho * mL) / sc); return self

    def predict(self, U):
        return self.rho * self.low.predict(U) + self._sc * self.delta.predict(U)

    def predict_std(self, U):
        sL = self.low.predict_std(U); sL = 0 if sL is None else sL
        sd = self._sc * self.delta.predict(U, return_std=True)[1]
        return np.sqrt((self.rho * sL) ** 2 + sd**2)


@_reg
class HierarchicalKriging(MFMethod):
    """Universal kriging with the LF predictor as the single trend basis: β0 by generalised least squares using the
    fitted residual covariance (one GLS / refit iteration)."""
    name = "hierarchical_kriging"

    def fit(self, UL, yL, UH, yH):
        self._fit_low(UL, yL)
        F = self.low.predict(UH)
        beta = float(F @ yH / (F @ F))                                      # OLS start
        for _ in range(2):
            g = _sk_gp(UH.shape[1], 2.5, 1, self.seed).fit(UH, yH - beta * F)
            K = g.kernel_(UH); Ki = np.linalg.pinv(K)
            beta = float((F @ Ki @ yH) / (F @ Ki @ F))                      # GLS
        self.beta, self.g = beta, _sk_gp(UH.shape[1], 2.5, 1, self.seed).fit(UH, yH - beta * F); return self

    def predict(self, U):
        return self.beta * self.low.predict(U) + self.g.predict(U)

    def predict_std(self, U):
        return self.g.predict(U, return_std=True)[1]


class _Augmented(MFMethod):
    def _aug(self, U):
        return np.column_stack([U, (self.low.predict(U) - self._m) / self._s])

    def fit(self, UL, yL, UH, yH):
        self._fit_low(UL, yL)
        f = self.low.predict(UL); self._m, self._s = f.mean(), f.std() or 1.0
        self.m = make(self.inner, seed=self.seed).fit(self._aug(UH), yH); return self

    def predict(self, U):
        return self.m.predict(self._aug(U))

    def predict_std(self, U):
        return self.m.predict_std(self._aug(U))


@_reg
class NARGP(_Augmented):
    name, inner = "nargp", "gp"


@_reg
class FeatureAug(_Augmented):
    name = "feature_aug"

    def __init__(self, base="xgb", **kw):
        super().__init__(base=base, **kw); self.inner = base


@_reg
class SpaceMapping(MFMethod):
    name = "space_mapping"

    def fit(self, UL, yL, UH, yH):
        from scipy.optimize import least_squares
        self._fit_low(UL, yL); d = UH.shape[1]
        def unpack(p):
            return p[: d * d].reshape(d, d), p[d * d: d * d + d], p[-2], p[-1]
        def res(p):
            A, b, s, c = unpack(p)
            return s * self.low.predict(np.clip(UH @ A.T + b, 0, 1)) + c - yH
        p0 = np.concatenate([np.eye(d).ravel(), np.zeros(d), [1.0, 0.0]])
        self.p = least_squares(res, p0, diff_step=1e-3, max_nfev=200 * len(p0)).x
        self.corr = _sk_gp(d, 2.5, 1, self.seed).fit(UH, -res(self.p)); return self

    def _mapped(self, U):
        d = U.shape[1]; A, b = self.p[: d * d].reshape(d, d), self.p[d * d: d * d + d]
        return self.p[-2] * self.low.predict(np.clip(U @ A.T + b, 0, 1)) + self.p[-1]

    def predict(self, U):
        return self._mapped(U) + self.corr.predict(U)


@_reg
class MFNN(MFMethod):
    """Composite multi-fidelity network (Meng & Karniadakis 2020, J. Comput. Phys. 401): a low-fidelity network
    NN_L(x) on the LF data, then f_H(x) = F_lin(x, NN_L(x)) + F_nl(x, NN_L(x)) with F_lin linear and F_nl a small
    tanh network penalised by L2 (so the correlation is linear unless the data demand otherwise). Both stages are
    full-batch Adam with cosine decay; an ensemble of 3 seeds is averaged."""
    name = "mf_nn"

    def fit(self, UL, yL, UH, yH):
        import torch
        self._ym, self._ys = yH.mean(), yH.std() or 1.0
        self._lm, self._ls = yL.mean(), yL.std() or 1.0
        d = UH.shape[1]
        T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
        XL, ZL, XH, ZH = T(UL), T((yL - self._lm) / self._ls), T(UH), T((yH - self._ym) / self._ys)
        self.models = []
        for k in range(3):
            torch.manual_seed(self.seed * 10 + k)
            low = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                      torch.nn.Linear(64, 1)).double()
            self._train(low, lambda: ((low(XL).squeeze(-1) - ZL) ** 2).mean(), 3000)
            with torch.no_grad():
                fl = low(XH)
            A = torch.cat([XH, fl], 1)
            if k == 0:                                                   # λ by 5-fold CV on the HF data (once)
                self.lam = self._cv_lambda(A, ZH, d)
            lin, nl = self._hf_stage(A, ZH, d, self.lam, 3000)
            self.models.append((low.eval(), lin.eval(), nl.eval()))
        return self

    def _hf_stage(self, A, Z, d, lam, epochs):
        import torch
        lin = torch.nn.Linear(d + 1, 1).double()
        nl = torch.nn.Sequential(torch.nn.Linear(d + 1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).double()
        self._train(None, lambda: ((lin(A) + nl(A)).squeeze(-1) - Z).pow(2).mean()
                    + lam * sum((p ** 2).sum() for p in nl.parameters()), epochs,
                    params=list(lin.parameters()) + list(nl.parameters()))
        return lin, nl

    def _cv_lambda(self, A, Z, d):
        """The L2 weight on the nonlinear part decides how far the map departs from linear; Meng & Karniadakis tune
        it. Chosen from {1e-6, 1e-4, 1e-2} by 5-fold CV on the high-fidelity points (shorter training)."""
        import torch
        from sklearn.model_selection import KFold
        n = len(Z)
        best = (np.inf, 1e-4)
        for lam in (1e-6, 1e-4, 1e-2):
            err = 0.0
            for tr, te in KFold(min(5, n), shuffle=True, random_state=self.seed).split(np.arange(n)):
                tr, te = torch.tensor(tr), torch.tensor(te)
                lin, nl = self._hf_stage(A[tr], Z[tr], d, lam, 1000)
                with torch.no_grad():
                    err += float(((lin(A[te]) + nl(A[te])).squeeze(-1) - Z[te]).pow(2).sum())
            if err < best[0]:
                best = (err, lam)
        return best[1]

    @staticmethod
    def _train(net, loss_fn, epochs, params=None):
        import torch
        opt = torch.optim.Adam(params if params is not None else net.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for _ in range(epochs):
            opt.zero_grad(); loss = loss_fn(); loss.backward(); opt.step(); sched.step()

    def predict(self, U):
        import torch
        X = torch.tensor(np.asarray(U), dtype=torch.float64)
        outs = []
        with torch.no_grad():
            for low, lin, nl in self.models:
                A = torch.cat([X, low(X)], 1)
                outs.append((lin(A) + nl(A)).squeeze(-1).numpy())
        return np.mean(outs, 0) * self._ys + self._ym


def make_mf(name, **kw):
    return REGISTRY[name](**kw)
