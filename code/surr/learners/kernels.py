"""Kernel and Gaussian-process learners — the engineering and computer-experiments mainstream.

  gp            ML-II GP, Matérn-5/2 ARD + learned nugget          Sacks et al. 1989 / Rasmussen & Williams 2006
  gp_se         same, squared-exponential kernel                   (smoothness-prior contrast)
  smt_krg       SMT kriging (ONERA/Michigan aerospace toolbox)     Bouhlel et al. 2019 — aerospace standard
  smt_kpls      SMT KPLS kriging (PLS-reduced length-scales)       Bouhlel et al. 2016 — high-d aerospace
  hetgp         most-likely heteroscedastic GP (two-stage)          Goldberg 1998 / Kersting et al. 2007 idea
  local_gp      nearest-neighbour local GP (laGP style)             Gramacy & Apley 2015
  treed_gp      CART partition + GP per leaf (non-Bayesian TGP)     Gramacy & Lee 2008 (approximation)
  krr           kernel ridge regression, CV-tuned                   ML
  svr           ε-support-vector regression, CV-tuned               Vapnik 1995
"""
from __future__ import annotations

import warnings

import numpy as np

from .base import Learner, pairwise_dist, register

warnings.filterwarnings("ignore", module="sklearn.gaussian_process")


def _sk_gp(d, nu=2.5, restarts=3, seed=0, noise_floor=1e-6):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
    base = RBF(np.ones(d) * 0.3, (1e-3, 1e3)) if nu is None else Matern(np.ones(d) * 0.3, (1e-3, 1e3), nu=nu)
    k = C(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-4, (noise_floor, 1e0))
    return GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=restarts, random_state=seed)


@register
class GP(Learner):
    name, family, year, field, ref, uq, smooth = "gp", "gaussian process", 1989, "computer experiments / engineering", "sacks1989design", True, True
    max_n = 5000
    nu = 2.5

    def _fit(self, U, z):
        self._m = _sk_gp(U.shape[1], self.nu, self.kw.get("restarts", 3), self.seed).fit(U, z)

    def _predict(self, U):
        return self._m.predict(U)

    def _predict_std(self, U):
        return self._m.predict(U, return_std=True)[1]


@register
class GP_SE(GP):
    name, nu = "gp_se", None


class _SMT(Learner):
    family, field, uq, smooth = "gaussian process", "aerospace design (ONERA / U. Michigan)", True, True
    max_n = 5000

    def _model(self, d): ...

    def _fit(self, U, z):
        self._m = self._model(U.shape[1])
        self._m.set_training_values(U, z)
        self._m.train()

    def _predict(self, U):
        return self._m.predict_values(U).ravel()

    def _predict_std(self, U):
        return np.sqrt(np.maximum(self._m.predict_variances(U).ravel(), 0))

    def grad(self, U, h=None):                                             # SMT provides analytic derivatives
        U = np.atleast_2d(U)
        return np.column_stack([self._m.predict_derivatives(U, k).ravel() for k in range(U.shape[1])]) * self._ys


@register
class SMT_KRG(_SMT):
    name, year, ref = "smt_krg", 2019, "bouhlel2019smt"

    def _model(self, d):
        from smt.surrogate_models import KRG
        return KRG(theta0=[1e-1] * d, corr="matern52", nugget=1e-8, eval_noise=True, print_global=False)


@register
class SMT_KPLS(_SMT):
    name, year, ref = "smt_kpls", 2016, "bouhlel2016improving"

    def _model(self, d):
        from smt.surrogate_models import KPLS
        return KPLS(n_comp=min(3, d), theta0=[1e-1] * min(3, d), eval_noise=True, print_global=False)


@register
class HetGP(Learner):
    """Two-stage heteroscedastic GP: fit a GP, fit a second GP to log squared residuals, refit the first with
    per-point noise from the second (one 'most-likely' iteration)."""
    name, family, year, field, ref, uq, smooth = "hetgp", "gaussian process", 1998, "statistics / simulation", "goldberg1998regression", True, True
    max_n = 3000

    def _fit(self, U, z):
        from sklearn.gaussian_process import GaussianProcessRegressor
        g1 = _sk_gp(U.shape[1], 2.5, 2, self.seed).fit(U, z)
        r2 = np.log((z - g1.predict(U)) ** 2 + 1e-8)
        self._noise = _sk_gp(U.shape[1], 2.5, 1, self.seed).fit(U, r2)
        a = np.exp(self._noise.predict(U))
        k = g1.kernel_.k1                                                  # signal part, re-optimised below
        self._m = GaussianProcessRegressor(k, alpha=a, n_restarts_optimizer=1, random_state=self.seed).fit(U, z)

    def _predict(self, U):
        return self._m.predict(U)

    def _predict_std(self, U):
        s = self._m.predict(U, return_std=True)[1]
        return np.sqrt(s**2 + np.exp(self._noise.predict(U)))


@register
class LocalGP(Learner):
    """laGP-style: global hyper-parameters from a GP on a subsample, then a GP on the k nearest neighbours of
    each query."""
    name, family, year, field, ref, uq = "local_gp", "gaussian process", 2015, "statistics", "gramacy2015local", True

    def __init__(self, k=40, **kw):
        super().__init__(**kw); self.k = k

    def _fit(self, U, z):
        rng = np.random.default_rng(self.seed)
        sub = rng.choice(len(z), min(len(z), 300), replace=False)
        self._kern = _sk_gp(U.shape[1], 2.5, 1, self.seed).fit(U[sub], z[sub]).kernel_
        self._U, self._z = U, z

    def _local(self, U):
        from sklearn.gaussian_process import GaussianProcessRegressor
        D = pairwise_dist(U, self._U); k = min(self.k, len(self._z))
        mu, sd = np.empty(len(U)), np.empty(len(U))
        for q in range(len(U)):
            nn = np.argpartition(D[q], k - 1)[:k]
            g = GaussianProcessRegressor(self._kern, optimizer=None).fit(self._U[nn], self._z[nn])
            m, s = g.predict(U[q:q + 1], return_std=True); mu[q], sd[q] = m[0], s[0]
        return mu, sd

    def _predict(self, U):
        return self._local(U)[0]

    def _predict_std(self, U):
        return self._local(U)[1]


@register
class TreedGP(Learner):
    """Non-Bayesian treed GP: a shallow CART partition (min leaf size ~max(10, 3d+3)) and an independent GP in each
    leaf. Captures regime changes and non-stationarity; discontinuous across leaf boundaries."""
    name, family, year, field, ref, uq = "treed_gp", "gaussian process", 2008, "statistics (rocket-booster CFD)", "gramacy2008bayesian", True

    def _fit(self, U, z):
        from sklearn.tree import DecisionTreeRegressor
        leaf = max(10, 3 * U.shape[1] + 3)
        self._tree = DecisionTreeRegressor(min_samples_leaf=leaf, max_leaf_nodes=6, random_state=self.seed).fit(U, z)
        ids = self._tree.apply(U)
        self._gps = {}
        for l in np.unique(ids):
            m = ids == l
            self._gps[l] = _sk_gp(U.shape[1], 2.5, 1, self.seed).fit(U[m], z[m])

    def _route(self, U, std=False):
        ids = self._tree.apply(U); out = np.empty(len(U))
        for l, g in self._gps.items():
            m = ids == l
            if m.any():
                out[m] = g.predict(U[m], return_std=True)[1] if std else g.predict(U[m])
        return out

    def _predict(self, U):
        return self._route(U)

    def _predict_std(self, U):
        return self._route(U, std=True)


@register
class KRR(Learner):
    name, family, year, field, ref, smooth = "krr", "kernel", 1970, "machine learning", "rahimi2007random", True
    max_n = 10000

    def _fit(self, U, z):
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV
        g = {"alpha": [1e-4, 1e-3, 1e-2, 1e-1], "gamma": [0.5, 2, 8, 32]}
        self._m = GridSearchCV(KernelRidge(kernel="rbf"), g, cv=min(5, len(z))).fit(U, z)

    def _predict(self, U):
        return self._m.predict(U)


@register
class SVR(Learner):
    name, family, year, field, ref, smooth = "svr", "kernel", 1995, "machine learning", "vapnik1995nature", True
    max_n = 20000

    def _fit(self, U, z):
        from sklearn.svm import SVR as _S
        from sklearn.model_selection import GridSearchCV
        g = {"C": [1, 10, 100], "gamma": [0.5, 2, 8], "epsilon": [0.01, 0.05]}
        self._m = GridSearchCV(_S(kernel="rbf"), g, cv=min(5, len(z))).fit(U, z)

    def _predict(self, U):
        return self._m.predict(U)
