"""Initial (one-shot) designs on the unit cube. Each returns an (n, d) array; all are seeded.

  random       i.i.d. uniform                                   (the thesis control)
  lhs          Latin hypercube                                  McKay, Beckman & Conover 1979
  maximin_lhs  best of 64 LHS by minimum pairwise distance      Morris & Mitchell 1995
  sobol        scrambled Sobol'                                 Sobol' 1967
  halton       scrambled Halton                                 Halton 1960
  cvt          centroidal Voronoi tessellation (Lloyd on MC)    Du, Faber & Gunzburger 1999
  tensor_grid  full factorial grid, m = round(n^(1/d))          (thesis original B-spline design)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

REGISTRY = {}


def _reg(fn):
    REGISTRY[fn.__name__] = fn
    return fn


@_reg
def random(n, d, seed=0):
    return np.random.default_rng(seed).random((n, d))


@_reg
def lhs(n, d, seed=0):
    return qmc.LatinHypercube(d, seed=seed).random(n)


@_reg
def maximin_lhs(n, d, seed=0, tries=64):
    from scipy.spatial.distance import pdist
    rng = np.random.default_rng(seed)
    best, bd = None, -1
    for _ in range(tries):
        X = qmc.LatinHypercube(d, seed=rng.integers(2**31)).random(n)
        m = pdist(X).min() if n > 1 else 0
        if m > bd:
            best, bd = X, m
    return best


@_reg
def sobol(n, d, seed=0):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                    # non-power-of-2 balance warning
        return qmc.Sobol(d, scramble=True, seed=seed).random(n)


@_reg
def halton(n, d, seed=0):
    return qmc.Halton(d, scramble=True, seed=seed).random(n)


@_reg
def cvt(n, d, seed=0, n_mc=None, iters=30):
    rng = np.random.default_rng(seed)
    P = rng.random((n_mc or max(2000, 50 * n), d))
    C = sobol(n, d, seed)
    for _ in range(iters):
        lab = np.argmin(((P[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        for k in range(n):
            m = lab == k
            if m.any():
                C[k] = P[m].mean(0)
    return C


@_reg
def tensor_grid(n, d, seed=0):
    m = max(2, int(round(n ** (1 / d))))
    g = np.linspace(0, 1, m)
    return np.array(np.meshgrid(*[g] * d, indexing="ij")).reshape(d, -1).T


def make(name, n, d, seed=0):
    return REGISTRY[name](n, d, seed=seed)
