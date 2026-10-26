"""Self-designing surrogates (E14): Smolyak reproduces polynomials of low total degree exactly, respects the budget;
TT-cross recovers a rank-1 / low-rank function and respects the budget."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surr import samplers as S  # noqa: E402
from surr.selfdesign import bary_matrix, make  # noqa: E402


def test_bary():
    x = np.linspace(0, 1, 7)
    for m in (1, 2, 3, 5, 9):
        B = bary_matrix(x, m)
        assert np.allclose(B.sum(1), 1)


def test_smolyak():
    T = S.sobol(512, 3, seed=1)
    f = lambda U: 1 + U[:, 0] - 2 * U[:, 1] * U[:, 2] + 3 * U[:, 0] ** 2
    m = make("smolyak").fit(f, 3, budget=100)
    assert m.n <= 100, m.n
    assert np.allclose(m.predict(T), f(T), atol=1e-10), np.abs(m.predict(T) - f(T)).max()
    g = lambda U: np.exp(-((U - 0.3) ** 2).sum(1))
    e = [np.sqrt(np.mean((make("smolyak").fit(g, 3, budget=b).predict(T) - g(T)) ** 2)) for b in (30, 100, 400)]
    assert e[2] < e[1] < e[0], e
    return e


def test_tt():
    T = S.sobol(512, 5, seed=1)
    f = lambda U: np.prod(np.cos(U), axis=1) + 0.5 * np.sin(U.sum(1))      # rank ≤ 3 (sum of separable terms)
    m = make("tt_cross").fit(f, 5, budget=1500, seed=0)
    assert m.n <= 1500
    err = np.sqrt(np.mean((m.predict(T) - f(T)) ** 2)) / f(T).std()
    assert err < 1e-3, err
    return m.n, err


if __name__ == "__main__":
    test_bary(); print("barycentric partition of unity: ok")
    print("smolyak exact on a degree-2 polynomial; errors at budgets 30/100/400:", test_smolyak())
    print("tt_cross (evaluations, NRMSE):", test_tt())
