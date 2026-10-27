"""Surrogate-managed optimization loops: they query the TRUE function f as they go (E9b), unlike surr.optimizers,
which optimize a fixed surrogate. API: run(name, f, d, budget, seed) -> dict(X, y) with every evaluated point in
order (the incumbent after k evaluations is the best observed of the first k).

  ego            efficient global optimization: GP (Matérn 5/2, ARD) + expected improvement, maximised over 2000
                 Sobol' candidates then polished by L-BFGS-B from the best 3                Jones, Schonlau & Welch 1998
  turbo          TuRBO-1: one trust region around the incumbent, side length scaled by the GP length-scales,
                 Thompson sampling on 1000 candidates in the region; expand after 3 successes, shrink after
                 ⌈d⌉ failures, restart when it collapses below 0.5^7                             Eriksson et al. 2019
  dycors         DYCORS: cubic RBF with linear tail; candidates perturb the incumbent on a random subset of coordinates
                 (probability decaying with the evaluation count), adaptive step σ; weighted score of the RBF
                 prediction and distance to evaluated points, weights cycling 0.3/0.5/0.8/0.95   Regis & Shoemaker 2013
  tr_model_mgmt  trust-region model management: GP model, step = model minimiser inside the trust region (L-BFGS-B
                 multistart), ratio of actual to predicted decrease sets the radius (×2 if > 0.75, ×0.25 if < 0.25),
                 accept if any decrease                                                         Alexandrov et al. 1998
All start from the same 2d + 2 Sobol' points (seeded). GP hyper-parameters are re-optimised every 10 evaluations and
held fixed in between (O(n³) solves only), the usual cost control; the method's rule is unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

REGISTRY = {}


def _reg(fn):
    REGISTRY[fn.__name__] = fn
    return fn


def _sobol(n, d, seed):
    from surr.samplers import sobol
    return sobol(n, d, seed=seed)


class _GP:
    """sklearn GP on standardised y; hyper-parameters re-optimised every `every` refits."""

    def __init__(self, d, seed, every=10):
        from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
        self.k = C(1.0, (1e-3, 1e3)) * Matern(np.full(d, 0.3), (1e-3, 1e3), nu=2.5) + WhiteKernel(1e-6, (1e-9, 1e-1))
        self.seed, self.every, self.calls = seed, every, 0

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        self.m_, self.s_ = y.mean(), y.std() or 1.0
        opt = self.calls % self.every == 0
        self.gp = GaussianProcessRegressor(self.k, normalize_y=False, optimizer="fmin_l_bfgs_b" if opt else None,
                                           n_restarts_optimizer=1 if opt else 0, random_state=self.seed)
        self.gp.fit(X, (y - self.m_) / self.s_)
        self.k = self.gp.kernel_
        self.calls += 1
        return self

    def predict(self, X, std=False):
        if std:
            mu, s = self.gp.predict(X, return_std=True)
            return mu * self.s_ + self.m_, s * self.s_
        return self.gp.predict(X) * self.s_ + self.m_

    @property
    def lengthscales(self):
        return np.atleast_1d(self.k.k1.k2.length_scale)


def _start(f, d, budget, seed):
    X = _sobol(2 * d + 2, d, seed)[: budget]
    return X, f(X)


@_reg
def ego(f, d, budget, seed=0):
    X, y = _start(f, d, budget, seed)
    gp = _GP(d, seed)
    rng = np.random.default_rng(seed)
    while len(y) < budget:
        gp.fit(X, y)
        best = y.min()

        def nei(u):
            mu, s = gp.predict(np.atleast_2d(u), std=True)
            s = np.maximum(s, 1e-12); z = (best - mu) / s
            return -float(((best - mu) * norm.cdf(z) + s * norm.pdf(z))[0])
        C = _sobol(2000, d, int(rng.integers(1 << 30)))
        mu, s = gp.predict(C, std=True); s = np.maximum(s, 1e-12); z = (best - mu) / s
        ei = (best - mu) * norm.cdf(z) + s * norm.pdf(z)
        cands = [minimize(nei, C[i], bounds=[(0, 1)] * d, method="L-BFGS-B", options=dict(maxiter=30)).x
                 for i in np.argsort(-ei)[:3]]
        x = min(cands, key=nei)
        if np.min(np.linalg.norm(X - x, axis=1)) < 1e-9:                 # duplicate: take the best fresh candidate
            x = C[np.argmax(ei)]
        X = np.vstack([X, x]); y = np.r_[y, f(x[None])]
    return dict(X=X, y=y)


@_reg
def turbo(f, d, budget, seed=0):
    X, y = _start(f, d, budget, seed)
    gp = _GP(d, seed)
    rng = np.random.default_rng(seed)
    L, Lmin, Lmax, succ, fail = 0.8, 0.5 ** 7, 1.6, 0, 0
    tau_s, tau_f = 3, int(np.ceil(d))
    while len(y) < budget:
        gp.fit(X, y)
        c = X[np.argmin(y)]
        w = gp.lengthscales; w = w / np.prod(w) ** (1 / d)
        lo, hi = np.clip(c - w * L / 2, 0, 1), np.clip(c + w * L / 2, 0, 1)
        P = _sobol(1000, d, int(rng.integers(1 << 30)))
        mask = rng.random((1000, d)) < min(1.0, 20 / d)                  # perturb a subset of coordinates
        mask[np.arange(1000), rng.integers(0, d, 1000)] = True
        C = np.where(mask, lo + (hi - lo) * P, c)
        sample = gp.gp.sample_y(C, 1, random_state=int(rng.integers(1 << 30)))[:, 0]
        x = C[np.argmin(sample)]
        fx = f(x[None])
        if fx[0] < y.min() - 1e-3 * abs(y.min()):
            succ, fail = succ + 1, 0
        else:
            succ, fail = 0, fail + 1
        if succ == tau_s:
            L, succ = min(2 * L, Lmax), 0
        elif fail == tau_f:
            L, fail = L / 2, 0
        X = np.vstack([X, x]); y = np.r_[y, fx]
        if L < Lmin and len(y) < budget:                                 # restart: fresh design, keep history
            L = 0.8
            Xn = _sobol(2 * d + 2, d, int(rng.integers(1 << 30)))[: budget - len(y)]
            X = np.vstack([X, Xn]); y = np.r_[y, f(Xn)]
    return dict(X=X, y=y)


def _rbf_fit(X, y):
    """Cubic RBF with linear tail (the DYCORS / Stochastic RBF surrogate)."""
    n, d = X.shape
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    P = np.column_stack([np.ones(n), X])
    A = np.block([[D ** 3, P], [P.T, np.zeros((d + 1, d + 1))]])
    coef = np.linalg.lstsq(A + 1e-10 * np.eye(len(A)), np.r_[y, np.zeros(d + 1)], rcond=None)[0]
    lam, c = coef[:n], coef[n:]
    return lambda Z: (np.sqrt(((Z[:, None] - X[None]) ** 2).sum(-1)) ** 3) @ lam + np.column_stack([np.ones(len(Z)), Z]) @ c


@_reg
def dycors(f, d, budget, seed=0):
    X, y = _start(f, d, budget, seed)
    rng = np.random.default_rng(seed)
    n0 = len(y)
    sigma, smin, succ, fail = 0.2, 0.2 * 0.5 ** 6, 0, 0
    tau_s, tau_f = 3, max(d, 5)
    weights = (0.3, 0.5, 0.8, 0.95)
    k = 0
    while len(y) < budget:
        s = _rbf_fit(X, y)
        xb = X[np.argmin(y)]
        p = min(20 / d, 1.0) * (1 - np.log(len(y) - n0 + 1) / np.log(budget - n0 + 1))
        p = max(p, 1.0 / d)
        mask = rng.random((100 * d, d)) < p
        mask[np.arange(100 * d), rng.integers(0, d, 100 * d)] = True
        C = np.clip(xb + mask * rng.normal(0, sigma, (100 * d, d)), 0, 1)
        sv = s(C); dist = np.sqrt(((C[:, None] - X[None]) ** 2).sum(-1)).min(1)
        vs = (sv - sv.min()) / (np.ptp(sv) + 1e-12)
        vd = (dist.max() - dist) / (np.ptp(dist) + 1e-12)
        wgt = weights[k % 4]; k += 1
        x = C[np.argmin(wgt * vs + (1 - wgt) * vd)]
        fx = f(x[None])
        if fx[0] < y.min() - 1e-3 * abs(y.min()):
            succ, fail = succ + 1, 0
        else:
            succ, fail = 0, fail + 1
        if succ >= tau_s:
            sigma, succ = min(2 * sigma, 0.2), 0
        if fail >= tau_f:
            sigma, fail = max(sigma / 2, smin), 0
        X = np.vstack([X, x]); y = np.r_[y, fx]
    return dict(X=X, y=y)


@_reg
def tr_model_mgmt(f, d, budget, seed=0):
    X, y = _start(f, d, budget, seed)
    gp = _GP(d, seed)
    rng = np.random.default_rng(seed)
    delta = 0.2
    c = X[np.argmin(y)]; fc = y.min()
    while len(y) < budget:
        gp.fit(X, y)
        lo, hi = np.clip(c - delta, 0, 1), np.clip(c + delta, 0, 1)
        starts = [c] + list(lo + (hi - lo) * rng.random((4, d)))
        res = [minimize(lambda u: float(gp.predict(u[None])[0]), s, bounds=list(zip(lo, hi)), method="L-BFGS-B",
                        options=dict(maxiter=50)) for s in starts]
        x = min(res, key=lambda r: r.fun).x
        if np.min(np.linalg.norm(X - x, axis=1)) < 1e-9:
            x = np.clip(c + delta * rng.uniform(-1, 1, d), 0, 1)        # model stalled: sample the region
        pred = float(gp.predict(c[None])[0] - gp.predict(x[None])[0])
        fx = f(x[None])
        X = np.vstack([X, x]); y = np.r_[y, fx]
        rho = (fc - fx[0]) / pred if pred > 1e-12 else -1.0
        if rho > 0.75:
            delta = min(2 * delta, 0.5)
        elif rho < 0.25:
            delta *= 0.25
        if fx[0] < fc:
            c, fc = x, fx[0]
        if delta < 1e-4:                                                 # converged locally: restart elsewhere
            delta = 0.2; c = _sobol(1, d, int(rng.integers(1 << 30)))[0]
            fc = float(f(c[None])[0]); X = np.vstack([X, c]); y = np.r_[y, fc]
    return dict(X=X[:budget], y=y[:budget])


def run(name, f, d, budget, seed=0):
    return REGISTRY[name](f, d, budget, seed=seed)
