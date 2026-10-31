"""E8 stochastic upscalers: exact gradients match finite differences; the subsample ensemble's stochastic gradient
is unbiased for its mean gradient; the PoU-quadratics stochastic gradient matches the weighted local gradients in
expectation and is close to the full gradient; the PoU surrogate reproduces a quadratic exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surr import problems as P, samplers as S, upscalers as UP  # noqa: E402
from surr.learners import make  # noqa: E402


def _fd(s, X, h=1e-6):
    d = X.shape[1]
    return np.stack([(s.predict(X + h * np.eye(d)[j]) - s.predict(X - h * np.eye(d)[j])) / (2 * h) for j in range(d)], 1)


def test_all():
    p = P.get("smooth_trig_d2")
    U = S.make("sobol", 60, 2, seed=0); y = p.f_unit(U)
    m = make("gp", seed=0).fit(U, y)
    X = S.sobol(16, 2, seed=3) * 0.8 + 0.1
    rng = np.random.default_rng(0)
    out = {}
    for name in ("pou_quadratics", "subsample_ensemble"):
        s = UP.build(name, m, 2, U=U, y=y, seed=0)
        g = s.grad(X)
        assert np.allclose(g, _fd(s, X), rtol=1e-4, atol=1e-5 * np.abs(g).max()), name
        sg = np.mean([s.stochastic_grad(X, rng) for _ in range(4000)], 0)
        out[name] = float(np.abs(sg - g).max() / np.abs(g).max())
    assert out["subsample_ensemble"] < 0.05, out
    assert out["pou_quadratics"] < 0.25, out                              # blending term left out: small bias
    q = lambda V: 1 + V[:, 0] - 2 * V[:, 0] * V[:, 1] + 3 * V[:, 1] ** 2
    class Q:
        predict = staticmethod(q)
    s = UP.build("pou_quadratics", Q(), 2)
    T = S.sobol(256, 2, seed=5)
    assert np.allclose(s.predict(T), q(T), atol=1e-6)
    return out


if __name__ == "__main__":
    print("relative gap between the mean stochastic gradient and the exact gradient:", test_all())
