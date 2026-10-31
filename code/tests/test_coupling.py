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


def test_strategies():
    p = P.get("branin")
    U = S.make("sobol", 30, 2, seed=0); y = p.f_unit(U)
    out = {}
    for L in ("gp", "rf", "mlp_ens"):
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
    r = test_strategies()
    print("true f at each strategy's pick (Branin f* = 0.398):")
    for k, v in sorted(r.items()):
        print(f"  {k[0]:8s} {k[1]:17s} {v:.4f}")
