"""Spectral / sparse-basis / change-of-basis learners (UQ discipline and family F10).

  pce_lars     sparse polynomial chaos, Legendre total degree p, LARS + CV       UQ (UQLab-style); Lüthen et al. 2021
  cs_fourier   compressed sensing in a Fourier basis, OMP with CV sparsity        Candès 2006 / Donoho 2006
  as_gp        active-subspace GP: gradients from a GP, top-k eigenvectors of     Constantine 2015
               E[∇f∇fᵀ], then a GP on the k projected coordinates
  poly_ridge   polynomial ridge approximation g(Wᵀx), W on the Grassmannian       Hokanson & Constantine 2018
               (variable projection, multistart)
"""
from __future__ import annotations

import itertools

import numpy as np
from numpy.polynomial import legendre as L

from .base import Learner, register


def _multi_indices(d, p, q=1.0):
    """Total-degree (q=1) or hyperbolic (q<1) multi-indices with ||α||_q <= p."""
    out = []
    for a in itertools.product(range(p + 1), repeat=d):
        if sum(x ** q for x in a) ** (1 / q) <= p + 1e-9:
            out.append(a)
    return np.array(out)


def _legendre_design(U, A):
    """Orthonormal (w.r.t. uniform measure) tensor Legendre basis: Phi[i, a] = Π_j sqrt(2α_j+1) P_{α_j}(2u_ij − 1)."""
    X = 2 * U - 1
    P = np.stack([L.legval(X, np.eye(A.max() + 1)[k]) * np.sqrt(2 * k + 1) for k in range(A.max() + 1)], -1)
    return np.prod(np.stack([P[:, j, A[:, j]] for j in range(U.shape[1])], 0), axis=0)   # (n, |A|)


@register
class PCE_LARS(Learner):
    name, family, year, field, ref, smooth = "pce_lars", "polynomial chaos", 2002, "uncertainty quantification", "luthen2021sparse", True

    def __init__(self, p=None, q=0.75, **kw):
        super().__init__(**kw); self.p, self.q = p, q

    def _fit(self, U, z):
        from sklearn.linear_model import LassoLarsCV
        d, n = U.shape[1], len(z)
        p = self.p or max(2, min(8, int(np.floor((n / 2) ** (1 / max(d, 1)) + 1)) if d > 2 else 10))
        A = _multi_indices(d, p, self.q)
        while len(A) > 2000 and p > 2:
            p -= 1; A = _multi_indices(d, p, self.q)
        self._A = A
        Phi = _legendre_design(U, A)
        self._m = LassoLarsCV(cv=min(5, n), fit_intercept=True, max_n_alphas=200).fit(Phi[:, 1:], z)

    def _predict(self, U):
        return self._m.predict(_legendre_design(U, self._A)[:, 1:])


class _CS(Learner):
    family, year, field, ref, smooth = "compressed sensing", 2006, "signal processing", "donoho2006compressed", True

    def __init__(self, K=None, **kw):
        super().__init__(**kw); self.K = K

    def _fit(self, U, z):
        from sklearn.linear_model import OrthogonalMatchingPursuitCV
        import warnings
        warnings.filterwarnings("ignore", message="Orthogonal matching pursuit ended prematurely")
        self._ks = self._freqs(U.shape[1])
        D = self._design(U)
        self._m = OrthogonalMatchingPursuitCV(cv=min(5, len(z)), max_iter=min(len(z) - 2, D.shape[1])).fit(D, z)

    def _predict(self, U):
        return self._m.predict(self._design(U))


@register
class CSFourier(_CS):
    """Periodic Fourier basis cos/sin(2π k·u), integer k with ||k||_inf <= K (one of each ±k pair); OMP with CV
    sparsity. Native basis for functions that are sparse in Fourier on the unit cube."""
    name = "cs_fourier"

    def _freqs(self, d):
        K = self.K or (6 if d <= 2 else 3 if d <= 3 else 2 if d <= 5 else 1)
        ks = [k for k in itertools.product(range(-K, K + 1), repeat=d) if next((x for x in k if x != 0), 1) > 0]
        return np.array(sorted(ks, key=lambda k: sum(map(abs, k))), float)

    def _design(self, U):
        ang = 2 * np.pi * U @ self._ks.T
        return np.hstack([np.cos(ang), np.sin(ang[:, 1:])])


@register
class CSCosine(_CS):
    """Tensor cosine basis Π_j cos(π k_j u_j), k_j >= 0, Σk_j <= K: orthogonal on [0,1]^d for *non-periodic*
    functions (even extension), so no Gibbs ringing at the box edges."""
    name = "cs_cosine"

    def _freqs(self, d):
        K = self.K or (12 if d <= 2 else 6 if d <= 3 else 3)
        return np.array(sorted([k for k in itertools.product(range(K + 1), repeat=d) if sum(k) <= K], key=sum), float)

    def _design(self, U):
        return np.prod(np.cos(np.pi * U[:, None, :] * self._ks[None]), axis=2)


@register
class ActiveSubspaceGP(Learner):
    name, family, year, field, ref, uq, smooth = "as_gp", "ridge / active subspace", 2015, "uncertainty quantification", "constantine2015active", True, True

    def __init__(self, k=None, **kw):
        super().__init__(**kw); self.k = k

    def _fit(self, U, z):
        from .kernels import _sk_gp
        g = _sk_gp(U.shape[1], 2.5, 1, self.seed).fit(U, z)
        h = 1e-4; G = np.empty_like(U)
        for j in range(U.shape[1]):
            e = np.zeros(U.shape[1]); e[j] = h
            G[:, j] = (g.predict(U + e) - g.predict(U - e)) / (2 * h)
        lam, V = np.linalg.eigh(G.T @ G / len(z)); lam, V = lam[::-1], V[:, ::-1]
        k = self.k or int(np.clip(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.99) + 1, 1, min(4, U.shape[1])))
        self._W = V[:, :k]
        self._g = _sk_gp(k, 2.5, 2, self.seed).fit(U @ self._W, z)

    def _predict(self, U):
        return self._g.predict(U @ self._W)

    def _predict_std(self, U):
        return self._g.predict(U @ self._W, return_std=True)[1]


@register
class PolyRidge(Learner):
    """Fit s(x) = g(Wᵀx), W ∈ R^{d×k} orthonormal, g a total-degree-p polynomial in k variables. Variable
    projection: for fixed W, g by least squares; W optimised over an unconstrained parametrisation (QR-normalised)."""
    name, family, year, field, ref, smooth = "poly_ridge", "ridge / active subspace", 2018, "uncertainty quantification", "hokanson2018data", True

    def __init__(self, k=1, p=3, **kw):
        super().__init__(**kw); self.k, self.p = k, p

    def _gfit(self, T, z):
        A = _multi_indices(T.shape[1], self.p)
        Phi = np.column_stack([np.prod(T ** a, axis=1) for a in A])
        c, *_ = np.linalg.lstsq(Phi, z, rcond=None)
        return A, c, Phi

    def _fit(self, U, z):
        from scipy.optimize import minimize
        d, k = U.shape[1], min(self.k, U.shape[1]); rng = np.random.default_rng(self.seed)
        Uc = U - 0.5
        def W_of(v):
            Q, _ = np.linalg.qr(v.reshape(d, k)); return Q
        def loss(v):
            A, c, Phi = self._gfit(Uc @ W_of(v), z); return np.mean((Phi @ c - z) ** 2)
        starts = [np.linalg.lstsq(np.column_stack([np.ones(len(z)), Uc]), z, rcond=None)[0][1:, None].repeat(k, 1)
                  + 0.01 * rng.normal(size=(d, k))] + [rng.normal(size=(d, k)) for _ in range(4)]
        best = min((minimize(loss, s.ravel(), method="L-BFGS-B") for s in starts), key=lambda r: r.fun)
        self._W = W_of(best.x); self._A, self._c, _ = self._gfit(Uc @ self._W, z)

    def _predict(self, U):
        T = (U - 0.5) @ self._W
        return np.column_stack([np.prod(T ** a, axis=1) for a in self._A]) @ self._c
