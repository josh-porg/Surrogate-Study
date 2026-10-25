"""E13 learners: analytic gradients agree with finite differences of each learner's own prediction; the derivative
covariance blocks agree with finite differences of the kernel; gradient data improves a smooth fit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surr import problems as P, samplers as S  # noqa: E402
from surr.learners import make  # noqa: E402
from surr.learners.gradient import _se_blocks  # noqa: E402


def test_kernel_derivative_blocks():
    rng = np.random.default_rng(0)
    A, B = rng.random((3, 2)), rng.random((4, 2)); ell = np.array([0.3, 0.7]); h = 1e-6
    K = _se_blocks(A, B, ell, 1.3, True, True)
    k = lambda a, b: _se_blocks(a, b, ell, 1.3, False, False)
    for j in range(2):
        e = np.zeros(2); e[j] = h
        fd_b = (k(A, B + e) - k(A, B - e)) / (2 * h)                     # ∂/∂x'_j
        assert np.allclose(K[:3, 4 * (j + 1):4 * (j + 2)], fd_b, atol=1e-6)
        fd_a = (k(A + e, B) - k(A - e, B)) / (2 * h)                     # ∂/∂x_j
        assert np.allclose(K[3 * (j + 1):3 * (j + 2), :4], fd_a, atol=1e-6)
        for i in range(2):
            e2 = np.zeros(2); e2[i] = h
            fd = (k(A + e2, B + e) - k(A + e2, B - e) - k(A - e2, B + e) + k(A - e2, B - e)) / (4 * h * h)
            assert np.allclose(K[3 * (i + 1):3 * (i + 2), 4 * (j + 1):4 * (j + 2)], fd, atol=1e-4)


def test_learners():
    p = P.get("smooth_trig_d2"); d = 2
    U = S.make("sobol", 24, d, seed=0); y = p.f_unit(U)
    G = p.grad(p.from_unit(U)) * (p.bounds[:, 1] - p.bounds[:, 0])
    T = S.sobol(2048, d, seed=1); yt = p.f_unit(T)
    nr = lambda m: float(np.sqrt(np.mean((m.predict(T) - yt) ** 2)) / yt.std())
    out = {}
    for name, kw in (("gek", {}), ("hermite_rbf", {}), ("ge_pce", {}), ("sobolev_mlp", {"epochs": 800})):
        mg = make(name, seed=0, **kw).fit(U, y, G)
        mv = make(name, seed=0, **kw).fit(U, y)
        X = T[:20]; h = 1e-5
        fd = np.stack([(mg.predict(X + h * np.eye(d)[j]) - mg.predict(X - h * np.eye(d)[j])) / (2 * h) for j in range(d)], 1)
        assert np.allclose(mg.grad(X), fd, rtol=1e-3, atol=1e-4 * np.abs(fd).max()), name
        out[name] = (nr(mv), nr(mg))
    for name, (v, g) in out.items():
        assert g < v, (name, v, g)                                       # gradients help on a smooth function
    return out


if __name__ == "__main__":
    test_kernel_derivative_blocks(); print("kernel derivative blocks = finite differences: ok")
    for k, (v, g) in test_learners().items():
        print(f"{k:12s} NRMSE value-only {v:.4f}  with gradients {g:.4f}")
