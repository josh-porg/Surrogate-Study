"""Upscalers: turn a fitted learner into an optimizer-ready representation (Poznański 2026, Sec. 14.3 pipeline and
its alternatives; error decomposition M.15, costs MATHEMATICS §11.3). API: build(name, model, d, **kw) -> object
with predict(U) and grad(U).

  none              the learner itself (finite-difference gradients unless it has its own)
  tensor_spline     dense m^d grid of model predictions → cubic tensor interpolant (thesis production path)
  smoothing_spline  dense grid → separable cubic smoothing spline per axis (GCV), then cubic tensor interpolant:
                    a low-pass filter over a rough intermediate model (e.g. trees)
  pspline_direct    P-spline / GAM fitted to the *scattered training data* directly (skips the intermediate);
                    tensor-product terms for d <= 3, additive terms above                Eilers & Marx 1996; Wood 2003
  smolyak_grid      Smolyak sparse grid of model predictions (feasible in high d)       Smolyak 1963
  tt_grid           tensor-train cross of model predictions on a Chebyshev grid        Oseledets & Tyrtyshnikov 2010
  pou_quadratics    partition of unity of K local quadratics fitted to the model's predictions: a smooth "summed
                    quadratics" surrogate with exact gradients; stochastic gradient = the gradient of one local
                    quadratic drawn by its weight at x                    Babuška & Melenk 1997; Lancaster & Šalkauskas 1981
  subsample_ensemble  mean of K smooth cubic-RBF models, each fitted to a random half of the training data; stochastic
                    gradient = the gradient of one random member (model uncertainty becomes gradient noise, which
                    can carry SGD-type optimizers out of shallow local minima)           Breiman 1996 (subagging);
                                                                                          STUDY_PLAN E9 arm (b)
The last two expose stochastic_grad(U, rng) for the first-order optimizers (E8 stochastic arm).
Planned: mba, thb (see coverage.yaml).
"""
from __future__ import annotations

import numpy as np

REGISTRY = {}


def _reg(fn):
    REGISTRY[fn.__name__] = fn
    return fn


class _Grid:
    def __init__(self, axes, vals):
        from scipy.interpolate import RegularGridInterpolator
        self.axes = axes
        self.rgi = RegularGridInterpolator(axes, vals, method="cubic", bounds_error=False, fill_value=None)

    def predict(self, U):
        return self.rgi(np.clip(np.atleast_2d(U), 0, 1))

    def grad(self, U, h=1e-5):
        U = np.atleast_2d(U); G = np.empty_like(U)
        for j in range(U.shape[1]):
            e = np.zeros(U.shape[1]); e[j] = h
            G[:, j] = (self.predict(U + e) - self.predict(U - e)) / (2 * h)
        return G


def _grid_values(model, d, m, max_points):
    if m**d > max_points:
        raise MemoryError(f"tensor grid {m}^{d} = {m**d:.2e} points exceeds cap {max_points:.0e} (infeasible cell)")
    axes = [np.linspace(0, 1, m)] * d
    G = np.array(np.meshgrid(*axes, indexing="ij")).reshape(d, -1).T
    return axes, model.predict(G).reshape([m] * d)


@_reg
def none(model, d, **kw):
    return model


@_reg
def tensor_spline(model, d, m=24, max_points=2_000_000, **kw):
    axes, V = _grid_values(model, d, m, max_points)
    return _Grid(axes, V)


@_reg
def smoothing_spline(model, d, m=24, max_points=2_000_000, lam=None, **kw):
    from scipy.interpolate import make_smoothing_spline
    axes, V = _grid_values(model, d, m, max_points)
    for ax in range(d):                                                    # separable smoothing, axis by axis
        V = np.apply_along_axis(lambda v: make_smoothing_spline(axes[ax], v, lam=lam)(axes[ax]), ax, V)
    return _Grid(axes, V)


@_reg
def pspline_direct(model, d, U=None, y=None, n_splines=10, **kw):
    """Ignores `model`; fits a GAM to (U, y). Gradients by finite differences on the GAM."""
    from pygam import LinearGAM, s, te
    if U is None:
        raise ValueError("pspline_direct needs the training data U, y")
    if d <= 3:
        terms = te(*range(d), n_splines=[max(4, n_splines // (d - 1 or 1))] * d) if d > 1 else s(0, n_splines=n_splines)
    else:
        terms = s(0, n_splines=n_splines)
        for j in range(1, d):
            terms = terms + s(j, n_splines=n_splines)
    gam = LinearGAM(terms).gridsearch(U, y, progress=False)

    class _G:
        def predict(self, V):
            return gam.predict(np.clip(np.atleast_2d(V), 0, 1))
        grad = _Grid.grad
    return _G()


class _SelfDesigned:
    """Wraps a fitted surr.selfdesign interpolant built on the intermediate model's predictions."""

    def __init__(self, sd):
        self.sd = sd

    def predict(self, U):
        return self.sd.predict(np.clip(np.atleast_2d(U), 0, 1))

    grad = _Grid.grad


@_reg
def smolyak_grid(model, d, max_points=20_000, **kw):
    """Smolyak sparse grid (Clenshaw–Curtis, combination technique) of the intermediate model's predictions: the
    upscaling grid that stays feasible in high d (mixed smoothness)."""
    from .selfdesign import make
    return _SelfDesigned(make("smolyak").fit(model.predict, d, max_points))


@_reg
def tt_grid(model, d, max_points=20_000, seed=0, **kw):
    """Tensor-train cross approximation of the intermediate model on a Chebyshev grid (compressed full grid; exact
    when the model is low-rank in the given axes)."""
    from .selfdesign import make
    return _SelfDesigned(make("tt_cross").fit(model.predict, d, max_points, seed=seed))


def _quad_features(X):
    """[1, x, upper-triangular x x^T] and the gradient of each feature, shapes (n, P) and (n, P, d)."""
    n, d = X.shape
    iu = np.triu_indices(d)
    F = np.hstack([np.ones((n, 1)), X, (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]])
    G = np.zeros((n, F.shape[1], d))
    G[:, 1:1 + d, :] = np.eye(d)
    for t, (i, j) in enumerate(zip(*iu)):
        G[:, 1 + d + t, i] += X[:, j]
        G[:, 1 + d + t, j] += X[:, i]
    return F, G


class _PoUQuadratics:
    def __init__(self, C, h, coefs):
        self.C, self.h, self.coefs = C, h, coefs                       # coefs: (K, P)

    def _w(self, U):
        D2 = ((U[:, None, :] - self.C[None]) ** 2).sum(-1)
        L = -0.5 * D2 / self.h ** 2
        W = np.exp(L - L.max(1, keepdims=True))
        return W / W.sum(1, keepdims=True)

    def _q(self, U):
        F, G = _quad_features(U)
        return F @ self.coefs.T, np.einsum("npd,kp->nkd", G, self.coefs)     # values (n, K), grads (n, K, d)

    def predict(self, U):
        U = np.atleast_2d(U)
        return (self._w(U) * self._q(U)[0]).sum(1)

    def grad(self, U):
        U = np.atleast_2d(U)
        W = self._w(U); q, gq = self._q(U)
        gw = -(U[:, None, :] - self.C[None]) / self.h ** 2                    # ∇ log of the Gaussian kernel
        gW = W[:, :, None] * (gw - (W[:, :, None] * gw).sum(1, keepdims=True))
        return (W[:, :, None] * gq).sum(1) + (q[:, :, None] * gW).sum(1)

    def stochastic_grad(self, U, rng):
        """Gradient of one local quadratic drawn with probability w_k(x): the weighted average of the local
        gradients in expectation (the smooth blending term ∇w_k is left out, a small bias where pieces overlap)."""
        U = np.atleast_2d(U)
        W = self._w(U); _, gq = self._q(U)
        k = np.array([rng.choice(len(w), p=w) for w in W])
        return gq[np.arange(len(U)), k]


@_reg
def pou_quadratics(model, d, K=None, n_fit=None, seed=0, **kw):
    """K = min(64, max(8, 4d)) centres (k-means on a Sobol' sample of n_fit = max(2000, 200d) model predictions);
    each local quadratic is a Gaussian-weighted ridge least-squares fit to the predictions; bandwidth h = the median
    distance from a sample point to its nearest centre."""
    from sklearn.cluster import KMeans
    from .samplers import sobol
    K = K or min(64, max(8, 4 * d)); n_fit = n_fit or max(2000, 200 * d)
    S = sobol(n_fit, d, seed=70_000 + seed); v = model.predict(S)
    C = KMeans(K, n_init=1, random_state=seed).fit(S).cluster_centers_
    D2 = ((S[:, None, :] - C[None]) ** 2).sum(-1)
    h = float(np.median(np.sqrt(D2.min(1)))) * 1.5 + 1e-9
    F, _ = _quad_features(S)
    coefs = []
    for k in range(K):
        w = np.exp(-0.5 * D2[:, k] / h ** 2); sw = np.sqrt(w)[:, None]
        A = F * sw
        coefs.append(np.linalg.solve(A.T @ A + 1e-8 * len(S) * np.eye(F.shape[1]), A.T @ (v * sw[:, 0])))
    return _PoUQuadratics(C, h, np.array(coefs))


class _CubicRBF:
    """Cubic RBF with linear tail (interpolating, tiny ridge) with its analytic gradient."""

    def __init__(self, X, y):
        n, d = X.shape
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        P = np.column_stack([np.ones(n), X])
        A = np.block([[D ** 3 + 1e-8 * np.eye(n), P], [P.T, np.zeros((d + 1, d + 1))]])
        c = np.linalg.lstsq(A, np.r_[y, np.zeros(d + 1)], rcond=None)[0]
        self.X, self.lam, self.b = X, c[:n], c[n:]

    def predict(self, U):
        R = np.sqrt(((U[:, None] - self.X[None]) ** 2).sum(-1))
        return R ** 3 @ self.lam + self.b[0] + U @ self.b[1:]

    def grad(self, U):
        Dv = U[:, None, :] - self.X[None]
        R = np.sqrt((Dv ** 2).sum(-1))
        return np.einsum("nm,nmd->nd", 3 * R * self.lam[None], Dv) + self.b[1:]


class _SubsampleEnsemble:
    def __init__(self, members):
        self.members = members

    def predict(self, U):
        U = np.atleast_2d(U)
        return np.mean([m.predict(U) for m in self.members], 0)

    def grad(self, U):
        U = np.atleast_2d(U)
        return np.mean([m.grad(U) for m in self.members], 0)

    def stochastic_grad(self, U, rng):
        """Gradient of one member drawn uniformly: unbiased for the ensemble-mean gradient."""
        return self.members[int(rng.integers(len(self.members)))].grad(np.atleast_2d(U))


@_reg
def subsample_ensemble(model, d, U=None, y=None, K=8, frac=0.5, seed=0, **kw):
    """Ignores `model` (like pspline_direct): K cubic-RBF members on random halves of the training data (U, y)."""
    if U is None:
        raise ValueError("subsample_ensemble needs the training data U, y")
    rng = np.random.default_rng(80_000 + seed)
    m = max(d + 2, int(frac * len(U)))
    return _SubsampleEnsemble([_CubicRBF(U[i], y[i]) for i in (rng.choice(len(U), m, replace=False)
                                                                 for _ in range(K))])


def build(name, model, d, **kw):
    return REGISTRY[name](model, d, **kw)
