"""Upscalers: turn a fitted learner into an optimizer-ready representation (Poznański 2026, Sec. 14.3 pipeline and
its alternatives; error decomposition M.15, costs MATHEMATICS §11.3). API: build(name, model, d, **kw) -> object
with predict(U) and grad(U).

  none              the learner itself (finite-difference gradients unless it has its own)
  tensor_spline     dense m^d grid of model predictions → cubic tensor interpolant (thesis production path)
  smoothing_spline  dense grid → separable cubic smoothing spline per axis (GCV), then cubic tensor interpolant:
                    a low-pass filter over a rough intermediate model (e.g. trees)
  pspline_direct    P-spline / GAM fitted to the *scattered training data* directly (skips the intermediate);
                    tensor-product terms for d <= 3, additive terms above                Eilers & Marx 1996; Wood 2003
Planned: mba, thb, smolyak_grid, tt_grid (see coverage.yaml).
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


def build(name, model, d, **kw):
    return REGISTRY[name](model, d, **kw)
