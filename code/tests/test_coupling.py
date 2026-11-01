"""E15 coupling strategies: the tree-ensemble MILP finds the model's global minimum (no worse than a dense grid,
and the recovered point attains the MILP objective); every strategy returns a point in the box."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surr import coupling as C, problems as P, samplers as S  # noqa: E402
from surr.learners import make  # noqa: E402
from surr.learners.ensembles import members  # noqa: E402


def test_milp():
    p = P.get("branin")
    U = S.make("sobol", 60, 2, seed=0); y = p.f_unit(U)
    g = np.linspace(0, 1, 401); G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    out = {}
    for L in ("xgb", "lgbm", "rf"):
        m = make(L, seed=0).fit(U, y)
        x = C.milp_trees(m, U, y, 2, 1000, 0)
        fm, fg = float(m.predict(x[None])[0]), float(m.predict(G).min())
        assert fm <= fg + 1e-9 * (1 + abs(fg)), (L, fm, fg)
        out[L] = (fm, fg, C.milp_trees.last)
    return out


def _grid(k=401):
    g = np.linspace(0, 1, k)
    return np.array(np.meshgrid(g, g)).reshape(2, -1).T


def test_milp_relu():
    """The ReLU MILP objective equals the network ensemble's prediction at the recovered point (the encoding is
    exact) and is no worse than a dense grid."""
    p = P.get("branin")
    U = S.make("sobol", 60, 2, seed=0); y = p.f_unit(U)
    m = make("mlp_relu", seed=0, opt="lbfgs").fit(U, y)
    x = C.milp_relu(m, U, y, 2, 1000, 0)
    info = C.milp_relu.last
    fm, fg = float(m.predict(x[None])[0]), float(m.predict(_grid()).min())
    obj = float(m._yinv(info["milp_obj"]))
    tol = 1e-4 * (1 + abs(fg)) * max(1.0, m._ys)
    assert abs(obj - fm) <= tol, (obj, fm)                               # float32 network vs float64 program
    assert fm <= fg + tol, (fm, fg)
    return fm, fg, info


def test_gp_bnb():
    """The GP branch-and-bound mean matches sklearn's predictor, certifies on a 2-D problem, and its value is no
    worse than a dense grid (up to the certified tolerance)."""
    r = np.linspace(0, 20, 200001)
    for k, (ka, dk, k2) in C._KAPPA.items():                             # the remainder N(r) must be increasing
        assert np.diff(2 - 2 * ka(r) + 2 * r * r * dk(r) - k2 * r * r).min() >= -1e-12, k
    p = P.get("branin")
    U = S.make("sobol", 40, 2, seed=0); y = p.f_unit(U)
    out = {}
    rng = np.random.default_rng(0)
    for L in ("gp", "gp_se"):
        m = make(L, seed=0).fit(U, y)
        X, al, cst, il, (kap, dkr, k20) = C._gp_parts(m)
        G = _grid(101)
        mz = cst * np.array([al @ kap(np.sqrt((((g - X) * il) ** 2).sum(1))) for g in G])
        assert np.allclose(m._yinv(mz), m.predict(G), rtol=1e-8, atol=1e-8 * m._ys), L
        x0, e = np.array([0.37, 0.61]), 1e-6                             # analytic gradient vs central differences
        Z = (x0 - X) * il; rr = np.sqrt((Z ** 2).sum(1))
        g_an = cst * ((al * dkr(rr)) @ (Z * il))
        f = lambda x: cst * (al @ kap(np.sqrt((((x - X) * il) ** 2).sum(1))))
        g_fd = np.array([(f(x0 + e * v) - f(x0 - e * v)) / (2 * e) for v in np.eye(2)])
        assert np.allclose(g_an, g_fd, rtol=1e-4), (L, g_an, g_fd)       # FD loses digits to the α cancellation
        Hn = np.sqrt(al @ (cst * kap(np.sqrt(((((X[:, None] - X[None]) * il) ** 2).sum(-1))))) @ al)
        for _ in range(300):                                             # the centred bound is a valid lower bound
            c = rng.random(2); h = rng.random(2) * 0.2
            lo, hi = np.clip(c - h, 0, 1), np.clip(c + h, 0, 1); cc, hh = 0.5 * (lo + hi), 0.5 * (hi - lo)
            Z = (cc - X) * il; rr = np.sqrt((Z ** 2).sum(1))
            fc, gc = cst * (al @ kap(rr)), cst * ((al * dkr(rr)) @ (Z * il))
            rho = np.sqrt(((hh * il) ** 2).sum())
            lb = fc - np.abs(gc) @ hh - Hn * np.sqrt(max(cst * (2 - 2 * kap(rho) + 2 * rho ** 2 * dkr(rho)
                                                               - k20 * rho ** 2), 0))
            pts = lo + rng.random((400, 2)) * (hi - lo)
            assert lb <= min(f(q) for q in pts) + 1e-9, (L, lb)
        x = C.gp_bnb(m, U, y, 2, 1000, 0)
        info = C.gp_bnb.last
        fm, fg = float(m.predict(x[None])[0]), float(m.predict(_grid()).min())
        assert info["status"] == "certified", (L, info)
        assert fm <= fg + 2e-4 * m._ys, (L, fm, fg)
        out[L] = (fm, fg, info)
    return out


def test_strategies():
    p = P.get("branin")
    U = S.make("sobol", 30, 2, seed=0); y = p.f_unit(U)
    out = {}
    for L in ("gp", "rf", "mlp_ens", "mlp_relu"):
        m = make(L, seed=0).fit(U, y)
        mem = members(m)
        for name, fn in C.STRATEGIES.items():
            if name == "milp_trees" and L != "rf":
                continue
            x = fn(m, U, y, 2, 200, 0, mem=mem)
            if x is None:
                continue
            assert x.shape == (2,) and np.all((x >= 0) & (x <= 1)), (L, name, x)
            out[(L, name)] = float(p.f_unit(x[None])[0])
    return out


if __name__ == "__main__":
    for k, v in test_milp().items():
        print(f"{k}: model value at MILP point {v[0]:.4f} <= dense-grid min {v[1]:.4f}; {v[2]}")
    v = test_milp_relu()
    print(f"mlp_relu: model value at MILP point {v[0]:.4f} <= dense-grid min {v[1]:.4f}; {v[2]}")
    for k, v in test_gp_bnb().items():
        print(f"{k}: B&B value {v[0]:.4f} <= dense-grid min {v[1]:.4f}; {v[2]}")
    r = test_strategies()
    print("true f at each strategy's pick (Branin f* = 0.398):")
    for k, v in sorted(r.items()):
        print(f"  {k[0]:8s} {k[1]:17s} {v:.4f}")
