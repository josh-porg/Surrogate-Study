"""Feature probes (block P-FEAT): problems built so that a specific method family *should* win or lose, plus
"native prior draws" (one random function from each family's own prior) and blind composites where no a-priori
winner is obvious. Which method each probe is expected to favour or hurt is declared in code/surr/coverage.yaml
and checked by code/surr/check_coverage.py; the intent of every probe is in its `desc`.

All probes live on the unit cube [0,1]^d. Random structure (directions, frequencies, partitions, network weights)
is fixed by a seed so a probe is the same function in every run.
"""
from __future__ import annotations

import numpy as np

from .base import Problem, box, register

DIMS = (2, 5, 10)


def _rng(name, d):
    return np.random.default_rng(abs(hash((name, d))) % (2**32) if False else (sum(map(ord, name)) * 1009 + d))


def _reg(name, fn, d, category, tags, desc):
    return register(Problem(name=f"{name}_d{d}", fn=fn, bounds=box(0, 1, d), block="P-FEAT", category=category,
                            tags=tags, desc=desc))


def build(d):
    r = lambda n: _rng(n, d)

    # smooth, globally analytic — favours spectral/kernel/polynomial methods, hurts piecewise-constant trees
    _reg("smooth_trig", lambda X: np.sin(2 * X.sum(1) / np.sqrt(d)) + 0.5 * np.cos(3 * X[:, 0]), d,
         "smooth", ("smooth", "analytic"), "smooth analytic trig")

    # exact quadratic — response-surface/PCE recover it exactly; local methods (IDW, kNN, trees) cannot
    A = r("quad").normal(size=(d, d)); A = A @ A.T / d; bq = r("quad_b").normal(size=d)
    _reg("quadratic", lambda X, A=A, bq=bq: ((X - .5) @ A * (X - .5)).sum(1) + (X - .5) @ bq, d,
         "polynomial", ("polynomial", "smooth"), "exact random quadratic")

    # additive g1(x1)+...+gd(xd) — favours GAM/additive P-splines, TT (rank 2), PPR, sparse grids
    w = r("add").uniform(1, 4, d)
    _reg("additive", lambda X, w=w: np.sin(w * np.pi * X).sum(1) / np.sqrt(d), d,
         "additive", ("additive", "separable"), "sum of 1-D sines")

    # low-rank product — favours functional SVD / tensor train; hurts additive models
    w2 = r("prod").uniform(1, 3, d)
    _reg("lowrank_product", lambda X, w=w2: np.prod(1 + 0.5 * np.sin(w * np.pi * X), axis=1), d,
         "lowrank", ("low_rank", "product"), "rank-1 product of 1-D factors")

    # rotated ridge g(w'x) — favours active subspaces/PPR/ridge/KPLS; hurts axis-aligned trees, additive models
    v = r("ridge").normal(size=d); v /= np.linalg.norm(v)
    _reg("rotated_ridge", lambda X, v=v: np.tanh(4 * ((X - .5) @ v)) + 0.3 * ((X - .5) @ v) ** 2, d,
         "ridge", ("ridge", "rotated", "low_intrinsic_dim"), "1-D ridge along an oblique direction")

    # axis-aligned jump — favours trees; hurts global smooth bases (Gibbs / ringing)
    _reg("axis_step", lambda X: np.where(X[:, 0] > 0.37, 1.0, 0.0) + 0.3 * X[:, -1], d,
         "discontinuity", ("discontinuity", "axis_aligned"), "jump across x1 = 0.37")

    # oblique jump — hurts trees (staircase) *and* smooth bases; unclear winner
    _reg("oblique_step", lambda X, v=v: np.where((X - .5) @ v > 0.05, 1.0, 0.0) + 0.3 * X[:, -1], d,
         "discontinuity", ("discontinuity", "oblique", "blind"), "jump across an oblique hyperplane")

    # kink / floor — favours MARS, piecewise-linear, trees; hurts smooth kernels and polynomials
    _reg("kink_floor", lambda X: np.maximum(np.sin(3 * X[:, 0]) + 0.5 * X[:, 1:].sum(1) / max(d - 1, 1), 0.6), d,
         "kink", ("kink", "floor"), "smooth field clamped to a floor")

    # plateau + small localized bump — favours adaptive sampling; space-filling designs miss it
    c = r("bump").uniform(0.2, 0.8, d)
    _reg("plateau_bump", lambda X, c=c: -np.exp(-((X - c) ** 2).sum(1) / (2 * 0.06**2)), d,
         "localized", ("localized", "plateau", "needle"), "flat plane with one narrow well (Easom-like)")

    # pole-like knee near a boundary — favours rational/physics dictionaries; hurts polynomials/PCE
    _reg("pole_knee", lambda X: 1.0 / (X[:, 0] + 0.05) + 0.5 * X[:, 1:].sum(1) / max(d - 1, 1), d,
         "pole", ("knee", "pole", "localized"), "1/(x1+0.05) hyperbolic knee at the boundary")

    # Fourier-sparse — favours compressed sensing in Fourier, spectral-mixture GP, Fourier features;
    # hurts MLP (spectral bias), trees, low-order polynomials
    K = r("four").integers(1, 6, size=(4, d)); ph = r("four_p").uniform(0, 2 * np.pi, 4)
    _reg("fourier_sparse", lambda X, K=K, ph=ph: np.cos(2 * np.pi * X @ K.T + ph).sum(1) / 2, d,
         "oscillatory", ("fourier_sparse", "oscillatory"), "4 random Fourier modes, frequencies 1-5")

    # multiscale: smooth trend + localized high-frequency packet — favours wavelets/THB/adaptive; hurts stationary GP
    _reg("multiscale", lambda X: X.sum(1) / d + 0.3 * np.exp(-((X - .7) ** 2).sum(1) / 0.01)
         * np.sin(40 * X[:, 0]), d, "multiscale", ("multiscale", "nonstationary", "localized"),
         "smooth trend plus a localized high-frequency packet")

    # nonstationary length-scale — favours treed/deep/warped/local GP; hurts stationary GP
    _reg("nonstationary", lambda X: np.sin(1.0 / (0.08 + X[:, 0])) + 0.2 * X[:, 1:].sum(1) / max(d - 1, 1), d,
         "nonstationary", ("nonstationary",), "chirp: oscillation frequency grows toward x1 = 0")

    # regimes — two different smooth functions on either side of a known boundary
    _reg("regime", lambda X: np.where(X[:, 0] + X[:, 1 % d] < 1.0, np.sin(4 * X[:, 0]), 1.5 + (X[:, 0] - .5) ** 2),
         d, "regime", ("regime", "discontinuity", "partitioned"), "two smooth regimes split by x1 + x2 = 1")

    # monotone — favours shape-constrained models at small n
    _reg("monotone", lambda X: np.log1p(5 * X).sum(1) + X[:, 0] * X[:, -1], d,
         "monotone", ("monotone", "smooth"), "increasing in every input")

    # native prior draws: one function from each family's own prior ---------------------------------------
    rr = r("gp"); W = rr.standard_t(df=5, size=(256, d)) / 0.3; bb = rr.uniform(0, 2 * np.pi, 256)
    aa = rr.normal(size=256) * np.sqrt(2 / 256)                       # Matérn-5/2-like spectral draw, ℓ≈0.3
    _reg("gp_draw", lambda X, W=W, bb=bb, aa=aa: np.cos(X @ W.T + bb) @ aa, d,
         "prior_draw", ("prior_draw", "gp_native"), "random Fourier-feature draw from a Matérn-like GP prior")

    rt = r("tree"); thr = rt.uniform(0.1, 0.9, (64, 2)); ax = rt.integers(0, d, (64, 2)); lv = rt.normal(size=64)
    _reg("tree_draw", lambda X, thr=thr, ax=ax, lv=lv: sum(
        lv[k] * ((X[:, ax[k, 0]] > thr[k, 0]) & (X[:, ax[k, 1]] > thr[k, 1])) for k in range(64)) / 8, d,
        "prior_draw", ("prior_draw", "tree_native", "discontinuity", "axis_aligned"),
        "sum of 64 random depth-2 axis-aligned stumps")

    rn = r("nn"); W1 = rn.normal(size=(d, 32)) * 2; b1 = rn.normal(size=32); W2 = rn.normal(size=32) / np.sqrt(32)
    _reg("nn_draw", lambda X, W1=W1, b1=b1, W2=W2: np.tanh((X - .5) @ W1 + b1) @ W2, d,
         "prior_draw", ("prior_draw", "nn_native", "smooth"), "random 32-unit tanh network")

    # blind composite: random mixture of several mechanisms, no designed winner -----------------------------
    rc = r("mix"); wts = rc.dirichlet(np.ones(5))
    parts = [f"smooth_trig_d{d}", f"kink_floor_d{d}", f"fourier_sparse_d{d}", f"rotated_ridge_d{d}", f"axis_step_d{d}"]
    from .base import REGISTRY as _R
    fns = [_R[p] for p in parts]
    _reg("composite", lambda X, f=fns, w=wts: sum(wi * fi(X) for wi, fi in zip(w, f)), d,
         "blind", ("blind", "mixed"), "random convex mix of smooth, kink, Fourier, ridge and jump probes")


for _d in DIMS:
    build(_d)


def with_dummies(base: Problem, k: int) -> Problem:
    """Append k uninformative inputs to any problem (tests robustness to irrelevant features)."""
    name = f"{base.name}+dummy{k}"
    from .base import REGISTRY as _R
    if name in _R:
        return _R[name]
    d0 = base.d
    return register(Problem(name=name, fn=lambda X: base(X[:, :d0]),
                            bounds=np.vstack([base.bounds, box(0, 1, k)]), block=base.block,
                            category=base.category, tags=tuple(base.tags) + ("uninformative_inputs",),
                            desc=f"{base.desc} + {k} dummy inputs"))


for _d in (2, 5):
    with_dummies(__import__("surr.problems.base", fromlist=["REGISTRY"]).REGISTRY[f"smooth_trig_d{_d}"], 8)
    with_dummies(__import__("surr.problems.base", fromlist=["REGISTRY"]).REGISTRY[f"rotated_ridge_d{_d}"], 8)
