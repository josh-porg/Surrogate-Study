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


def _smolyak_predict_uncached(s, U):
    """Smolyak.predict as it was before the per-call matrix cache (2026-11-01), kept here as the reference."""
    from surr.selfdesign import _m
    U = np.atleast_2d(U)
    out = np.zeros(len(U))
    for l, c, V in s.terms:
        T = V
        first = True
        for i, li in enumerate(l):
            B = bary_matrix(U[:, i], _m(li))
            T = np.tensordot(B, T, axes=([1], [0])) if first else np.einsum("nj,nj...->n...", B, T)
            first = False
        out += c * T
    return out


def test_smolyak_cache_bit_identical():
    f = lambda U: np.sin(3 * U).sum(1) + U.prod(1)
    n = 0
    for d, budget in ((2, 2000), (5, 20_000), (10, 20_000)):
        s = make("smolyak").fit(f, d, budget)
        rng = np.random.default_rng(d)
        for U in (rng.random((1, d)), rng.random((257, d)), S.sobol(64, d, seed=3), np.zeros((1, d)), np.ones((2, d))):
            a, b = s.predict(U), _smolyak_predict_uncached(s, U)
            assert np.array_equal(a, b), (d, np.abs(a - b).max())
            n += 1
    return n


def test_selfdesigned_batched_grad_within_rounding():
    """The batched central-difference gradient of the self-designed upscalers (STUDY D-23) agrees with _Grid.grad's
    per-axis loop to rounding: the predictions differ at ~1e-14 relative (batch-size-dependent rounding in numpy's
    contractions), so the gradients (divided by 2h = 2e-5) may differ by ~1e-9. Returns the largest difference."""
    from surr import upscalers as UP

    class M:
        def predict(self, U):
            U = np.atleast_2d(U); return np.sin(3 * U).sum(1) + U.prod(1) + np.exp(-((U - 0.3) ** 2).sum(1))

    rng = np.random.default_rng(0)
    worst = 0.0
    for name in ("smolyak_grid", "tt_grid"):
        for d in (2, 5):
            s = UP.build(name, M(), d, max_points=3000, **({"seed": 0} if name == "tt_grid" else {}))
            for n in (1, 7, 50):
                U = rng.random((n, d))
                a, b = s.grad(U), UP._Grid.grad(s, U)
                worst = max(worst, float(np.abs(a - b).max()))
                assert np.allclose(a, b, rtol=0, atol=1e-8), (name, d, n, np.abs(a - b).max())
    return worst


if __name__ == "__main__":
    test_bary(); print("barycentric partition of unity: ok")
    print("smolyak per-call cache bit-identical to the uncached predictor:", test_smolyak_cache_bit_identical(), "cases")
    print("smolyak exact on a degree-2 polynomial; errors at budgets 30/100/400:", test_smolyak())
    print("tt_cross (evaluations, NRMSE):", test_tt())
    print("batched self-designed gradient vs per-axis loop, largest difference:",
          test_selfdesigned_batched_grad_within_rounding())
