"""Acquisition (scout) rules for sequential design. Each rule scores a candidate pool and returns the index of the
next point. Signature: rule(model, U, y, pool, rng, **kw) -> int, where `model` is a fitted surr learner on (U, y)
(rules that need no model ignore it).

  max_variance  argmax σ(x)                                      MacKay 1992 (ALM)
  imse_alc      argmax Σ_ref cov(ref, x)^2 / var(x) (GP only)     Cohn 1996 (ALC) / Sacks 1989 (IMSE)
  eigf          (μ(x) − y_nn(x))^2 + σ^2(x)                       Lam 2008
  mepe          α·e_LOO(nn(x))^2 + (1−α)·σ^2(x), α adaptive        Liu, Xu & Wang 2017
  lola_voronoi  Voronoi volume × local-linear nonlinearity        Crombecq et al. 2011
  qbc_committee committee spread (pass a committee learner)       Seung et al. 1992
  seq_maximin   argmax min-distance to the design (model-free)    —
  random        uniform pick (control)                            —
  ei            expected improvement (minimisation)               Jones et al. 1998
  lcb           lower confidence bound μ − κσ                     Srinivas et al. 2010
  u_function    argmin |μ − T| / σ (contour of level T)           Echard et al. 2011
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

REGISTRY = {}


def _reg(fn):
    REGISTRY[fn.__name__] = fn
    return fn


def _sd(model, X):
    s = model.predict_std(X)
    if s is None:
        raise ValueError(f"{model.name} has no native uncertainty; wrap it in a committee or use a model-free rule")
    return np.maximum(s, 1e-12)


def _mindist(pool, U):
    d2 = ((pool[:, None, :] - U[None]) ** 2).sum(2)
    return np.sqrt(d2.min(1)), d2.argmin(1)


@_reg
def max_variance(model, U, y, pool, rng, **kw):
    return int(np.argmax(_sd(model, pool)))


@_reg
def qbc_committee(model, U, y, pool, rng, **kw):
    return max_variance(model, U, y, pool, rng)


@_reg
def imse_alc(model, U, y, pool, rng, n_cand=150, n_ref=200, **kw):
    """Active learning Cohn: variance reduction integrated over a reference set; needs the GP posterior covariance."""
    gp = getattr(model, "_m", None)
    if gp is None or not hasattr(gp, "kernel_"):
        raise ValueError("imse_alc needs a scikit-learn GP model")
    cand = rng.choice(len(pool), min(n_cand, len(pool)), replace=False)
    ref = pool[rng.choice(len(pool), min(n_ref, len(pool)), replace=False)]
    _, C = gp.predict(np.vstack([pool[cand], ref]), return_cov=True)
    nc = len(cand)
    gain = (C[:nc, nc:] ** 2).sum(1) / np.maximum(np.diag(C)[:nc], 1e-12)
    return int(cand[np.argmax(gain)])


@_reg
def eigf(model, U, y, pool, rng, **kw):
    _, nn = _mindist(pool, U)
    return int(np.argmax((model.predict(pool) - y[nn]) ** 2 + _sd(model, pool) ** 2))


@_reg
def mepe(model, U, y, pool, rng, alpha=None, **kw):
    """LOO errors approximated by refits without each design point are expensive; use the fast surrogate:
    e_LOO(x_i) ≈ |y_i − μ_{-i}(x_i)| from a nearest-neighbour cross-check (cheap proxy, documented in BUGS)."""
    from .learners import make
    loo = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        loo[i] = abs(y[i] - make(model.name, seed=0).fit(U[m], y[m]).predict(U[i:i + 1])[0]) if len(y) <= 40 else np.nan
    if np.isnan(loo).any():                                                # large n: k-NN residual proxy
        from sklearn.neighbors import KNeighborsRegressor
        k = KNeighborsRegressor(3).fit(U, y); loo = np.abs(y - k.predict(U))
    _, nn = _mindist(pool, U)
    s2 = _sd(model, pool) ** 2
    a = 0.5 if alpha is None else alpha
    return int(np.argmax(a * loo[nn] ** 2 + (1 - a) * s2))


@_reg
def lola_voronoi(model, U, y, pool, rng, n_mc=4000, **kw):
    n, d = U.shape
    P = rng.random((n_mc, d))
    _, cell = _mindist(P, U)
    vol = np.bincount(cell, minlength=n) / n_mc                           # Monte-Carlo Voronoi volumes
    D = ((U[:, None] - U[None]) ** 2).sum(2)
    nonlin = np.empty(n)
    k = min(n - 1, 2 * d + 2)
    for i in range(n):
        nb = np.argsort(D[i])[1:k + 1]
        A = np.column_stack([np.ones(k), U[nb] - U[i]])
        c, *_ = np.linalg.lstsq(A, y[nb], rcond=None)
        nonlin[i] = abs(c[0] - y[i]) + np.abs(A @ c - y[nb]).mean()       # local-linear misfit
    score = vol / vol.sum() + nonlin / max(nonlin.sum(), 1e-12)
    top = int(np.argmax(score))
    mine = np.where(cell == top)[0]
    if len(mine) == 0:
        return seq_maximin(model, U, y, pool, rng)
    _, pc = _mindist(pool, U)
    cand = np.where(pc == top)[0]
    if len(cand) == 0:
        return seq_maximin(model, U, y, pool, rng)
    dd, _ = _mindist(pool[cand], U)
    return int(cand[np.argmax(dd)])


@_reg
def seq_maximin(model, U, y, pool, rng, **kw):
    return int(np.argmax(_mindist(pool, U)[0]))


@_reg
def random(model, U, y, pool, rng, **kw):
    return int(rng.integers(len(pool)))


@_reg
def ei(model, U, y, pool, rng, xi=0.0, **kw):
    mu, s = model.predict(pool), _sd(model, pool)
    imp = y.min() - mu - xi; z = imp / s
    return int(np.argmax(imp * norm.cdf(z) + s * norm.pdf(z)))


@_reg
def lcb(model, U, y, pool, rng, kappa=2.0, **kw):
    return int(np.argmin(model.predict(pool) - kappa * _sd(model, pool)))


@_reg
def u_function(model, U, y, pool, rng, level=None, **kw):
    T = np.median(y) if level is None else level
    return int(np.argmin(np.abs(model.predict(pool) - T) / _sd(model, pool)))


def acquire(name, model, U, y, pool, rng, exclude=None, **kw):
    """Score the pool and return the chosen index, never re-picking indices in `exclude`."""
    if exclude is not None and len(exclude):
        keep = np.setdiff1d(np.arange(len(pool)), exclude)
        return int(keep[REGISTRY[name](model, U, y, pool[keep], rng, **kw)])
    return REGISTRY[name](model, U, y, pool, rng, **kw)
