"""E9b managed loops: each respects its budget of true evaluations and beats random search on a smooth 2-D problem."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surr import managed as M, problems as P  # noqa: E402


def test_loops(budget=40):
    p = P.get("branin")
    f = p.f_unit
    rnd = min(f(np.random.default_rng(s).random((budget, 2))).min() for s in range(5))
    out = {}
    for name in M.REGISTRY:
        r = M.run(name, f, 2, budget, seed=0)
        assert len(r["y"]) <= budget, (name, len(r["y"]))
        assert np.allclose(f(r["X"]), r["y"])
        out[name] = float(r["y"].min())
    fstar = 0.397887
    for name, v in out.items():
        assert v - fstar < rnd - fstar, (name, v, rnd)
    return out, rnd


if __name__ == "__main__":
    out, rnd = test_loops()
    print("best of 5 random searches:", round(rnd, 4))
    for k, v in out.items():
        print(f"{k:14s} best f = {v:.4f}  (f* = 0.3979)")
