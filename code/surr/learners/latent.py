"""Planned methods that work on fixed scattered designs (2026-10-11): learned kernels, low-rank / tensor formats and
latent sparse regression.

  kernel_flow_gp  GP whose ARD Matérn length-scales are learned by Kernel Flows: minimise
                  ρ = 1 − ‖u_coarse‖²/‖u_fine‖² (relative RKHS-norm loss from halving random batches) by SGD,
                  then noise/amplitude by ML-II with the scales fixed                          Owhadi & Yoo 2019
  lowrank_cp      f(x) ≈ Σ_r Π_j g_rj(x_j), 1-D Legendre factors, ALS + ridge, rank on hold-out — the scattered-data
                  form of the functional SVD (d = 2: Townsend & Trefethen's Chebfun2 idea) / CP decomposition
  ft_als          functional tensor train f(x) = G_1(x_1) G_2(x_2) ⋯ G_d(x_d), each core a Legendre expansion,
                  fitted by alternating least squares on scattered data; TT-rank on hold-out
                                                                              Gorodetsky, Karaman & Marzouk 2019 (C3)
  sindy_ae_static encoder x → z (2-D latent) and a sparse polynomial dictionary y ≈ Θ(z)ξ trained jointly, with
                  sequential thresholding of ξ (and a decoder reconstruction term); the static analogue of the
                  SINDy autoencoder                                                     after Champion et al. 2019
TT-cross (adaptive sampling) and Smolyak need their own designs and live with the sampling / upscaling studies.
"""
from __future__ import annotations

import numpy as np

from .base import Learner, register
from .nets import _torch


# ── kernel flows ─────────────────────────────────────────────────────────────────────────────────────────────
@register
class KernelFlowGP(Learner):
    name, family, year, field, ref, uq, smooth = "kernel_flow_gp", "gaussian process", 2019, "applied mathematics", "owhadi2019kernel", True, True
    max_n = 3000

    def _fit(self, U, z, steps=300):
        torch = _torch(); torch.manual_seed(self.seed)
        X = torch.tensor(U, dtype=torch.float64); Y = torch.tensor(z, dtype=torch.float64)
        n, d = U.shape
        logl = torch.nn.Parameter(torch.log(torch.full((d,), 0.3, dtype=torch.float64)))
        opt = torch.optim.Adam([logl], lr=0.05)
        rng = np.random.default_rng(self.seed)
        nf = min(n, 64)

        def K(A, B):
            r = torch.sqrt(((A[:, None, :] - B[None, :, :]) / logl.exp()).pow(2).sum(-1) + 1e-12) * np.sqrt(5)
            return (1 + r + r ** 2 / 3) * torch.exp(-r)

        for _ in range(steps):
            f = torch.tensor(rng.choice(n, nf, replace=False)); c = f[: nf // 2]
            Kf = K(X[f], X[f]) + 1e-6 * torch.eye(nf, dtype=torch.float64)
            Kc = K(X[c], X[c]) + 1e-6 * torch.eye(len(c), dtype=torch.float64)
            yf, yc = Y[f], Y[c]
            rho = 1 - (yc @ torch.linalg.solve(Kc, yc)) / (yf @ torch.linalg.solve(Kf, yf))
            opt.zero_grad(); rho.backward(); opt.step()
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
        ell = logl.detach().exp().numpy()
        k = C(1.0, (1e-3, 1e3)) * Matern(ell, "fixed", nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e0))
        self._m = GaussianProcessRegressor(k, n_restarts_optimizer=1, random_state=self.seed).fit(U, z)
        self.length_scales = ell

    def _predict(self, U):
        return self._m.predict(U)

    def _predict_std(self, U):
        return self._m.predict(U, return_std=True)[1]


# ── low-rank functional formats ──────────────────────────────────────────────────────────────────────────────
def _leg(t, p):
    return np.polynomial.legendre.legvander(2 * t - 1, p)                # (n, p+1)


def _holdout(n, seed):
    idx = np.random.default_rng(seed).permutation(n); h = max(2, n // 5)
    return idx[h:], idx[:h]


@register
class LowRankCP(Learner):
    """Sum of R separable terms Π_j g_rj(x_j), g_rj in Legendre degree p; ALS (each factor by ridge LS with the
    others fixed); (R, p, ridge λ) on hold-out with ≤ n/3 unknowns. d = 2 is the functional SVD."""
    name, family, year, field, ref, smooth = "lowrank_cp", "low-rank / tensor", 2013, "numerical analysis", "townsend2013extension", True

    def _fit(self, U, z):
        n, d = U.shape
        tr, va = _holdout(n, self.seed)
        cands = [(R, p, lam) for R in (1, 2, 3, 5, 8) for p in (3, 5, 8) for lam in (1e-6, 1e-3, 1e-1)
                 if R * d * (p + 1) <= n / 3] or [(1, 3, 1e-3)]
        fits = {c: self._als(U[tr], z[tr], *c) for c in cands}
        err = {c: np.mean((self._eval(f, U[va]) - z[va]) ** 2) for c, f in fits.items()}
        best = min(err, key=err.get)
        # ALS is non-convex: warm-start the full-data refit from the validated solution so it stays in that basin
        self._c = self._als(U, z, *best, init=fits[best]["C"])

    def _als(self, U, z, R, p, lam=1e-6, sweeps=25, init=None):
        n, d = U.shape; rng = np.random.default_rng(self.seed)
        B = [_leg(U[:, j], p) for j in range(d)]
        C = [c.copy() for c in init] if init is not None else             [rng.normal(size=(p + 1, R)) * 0.1 + np.eye(p + 1, R) for _ in range(d)]
        c0 = z.mean(); z = z - c0                          # explicit intercept: the base class centres y, and a
        for _ in range(sweeps):                            # constant would otherwise cost a whole rank
            for j in range(d):
                other = np.ones((n, R))
                for k in range(d):
                    if k != j:
                        other *= B[k] @ C[k]
                A = (B[j][:, :, None] * other[:, None, :]).reshape(n, -1)          # unknowns: C_j (p+1, R)
                C[j] = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ z).reshape(p + 1, R)
            f = LowRankCP._eval(dict(C=C, p=p, c0=0.0), U); shift = (z - f).mean(); c0 += shift; z = z - shift
        return dict(C=C, p=p, c0=c0)

    @staticmethod
    def _eval(c, U):
        out = np.ones((len(U), c["C"][0].shape[1]))
        for j in range(U.shape[1]):
            out *= _leg(U[:, j], c["p"]) @ c["C"][j]
        return out.sum(1) + c.get("c0", 0.0)

    def _predict(self, U):
        return self._eval(self._c, U)


@register
class FunctionalTT(Learner):
    """Functional tensor train: f(x) = G_1(x_1) ⋯ G_d(x_d), G_j(t) = Σ_b φ_b(t) A_j[:, :, b] (Legendre degree p),
    ranks (1, r, …, r, 1); ALS sweeps left-to-right and back, each core a ridge LS problem; (r, p, λ) on hold-out."""
    name, family, year, field, ref, smooth = "ft_als", "low-rank / tensor", 2019, "uncertainty quantification", "gorodetsky2019continuous", True

    def _fit(self, U, z):
        n, d = U.shape
        tr, va = _holdout(n, self.seed)
        size = lambda r, p: (p + 1) * (2 * r + max(d - 2, 0) * r * r)
        cands = [(r, p, lam) for r in (1, 2, 3, 4, 6) for p in (3, 5, 8) for lam in (1e-6, 1e-3, 1e-1)
                 if size(r, p) <= n / 3] or [(1, 3, 1e-3)]
        fits = {c: self._als(U[tr], z[tr], *c) for c in cands}
        err = {c: np.mean((self._eval(f, U[va]) - z[va]) ** 2) for c, f in fits.items()}
        best = min(err, key=err.get)
        self._c = self._als(U, z, *best, init=fits[best]["A"])       # warm start (see LowRankCP)

    def _als(self, U, z, r, p, lam=1e-6, sweeps=10, init=None):
        n, d = U.shape; rng = np.random.default_rng(self.seed)
        ranks = [1] + [r] * (d - 1) + [1]
        if init is not None:
            A = [a.copy() for a in init]
        else:
            A = [rng.normal(size=(ranks[j], ranks[j + 1], p + 1)) * 0.3 for j in range(d)]
            for j in range(d):
                A[j][..., 0] += np.eye(ranks[j], ranks[j + 1])
        Phi = [_leg(U[:, j], p) for j in range(d)]
        core = lambda j: np.einsum("abq,nq->nab", A[j], Phi[j])                # (n, r_j, r_{j+1})
        order = list(range(d)) + list(range(d - 2, 0, -1))
        c0 = z.mean(); z = z - c0                          # explicit intercept (see LowRankCP)
        for _ in range(sweeps):
            for j in order:
                L = np.ones((n, 1, 1))
                for k in range(j):
                    L = np.einsum("nab,nbc->nac", L, core(k))
                Rm = np.ones((n, 1, 1))
                for k in range(d - 1, j, -1):
                    Rm = np.einsum("nab,nbc->nac", core(k), Rm)
                # f = L[:, 0, a] A_j[a, b, q] Φ_j[n, q] R[:, b, 0]  -> linear in A_j
                M = np.einsum("na,nb,nq->nabq", L[:, 0, :], Rm[:, :, 0], Phi[j]).reshape(n, -1)
                v = np.linalg.solve(M.T @ M + lam * np.eye(M.shape[1]), M.T @ z)
                A[j] = v.reshape(ranks[j], ranks[j + 1], p + 1)
            shift = (z - FunctionalTT._eval(dict(A=A, p=p, c0=0.0), U)).mean(); c0 += shift; z = z - shift
        return dict(A=A, p=p, c0=c0)

    @staticmethod
    def _eval(c, U):
        out = np.ones((len(U), 1, 1))
        for j in range(U.shape[1]):
            out = np.einsum("nab,nbc->nac", out, np.einsum("abq,nq->nab", c["A"][j], _leg(U[:, j], c["p"])))
        return out[:, 0, 0] + c.get("c0", 0.0)

    def _predict(self, U):
        return self._eval(self._c, U)


# ── latent sparse regression ─────────────────────────────────────────────────────────────────────────────────
@register
class SINDyAEStatic(Learner):
    """Encoder x → z ∈ R^k (k = min(d, 2)), decoder z → x, and y ≈ Θ(z)ξ with Θ the polynomials of total degree ≤ 3
    in z. Loss = MSE(y) + 0.1·MSE(x̂) + λ‖ξ‖₁; every 1000 epochs coefficients with |ξ| < 0.05·max|ξ| are fixed to
    zero (sequential thresholding, as in SINDy). Adam, full batch, 4000 epochs."""
    name, family, year, field, ref, smooth = "sindy_ae_static", "latent / ROM", 2019, "dynamical systems / ROM", "champion2019data", True

    def _fit(self, U, z, epochs=4000):
        torch = _torch(); torch.manual_seed(self.seed); nn = torch.nn
        d = U.shape[1]; k = min(d, 2)
        self._enc = nn.Sequential(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, k))
        dec = nn.Sequential(nn.Linear(k, 32), nn.SiLU(), nn.Linear(32, d))
        self._exps = [(i, j) for i in range(4) for j in range(4) if i + j <= 3] if k == 2 else [(i,) for i in range(4)]
        self._xi = nn.Parameter(torch.zeros(len(self._exps), dtype=torch.float32))
        mask = torch.ones(len(self._exps))
        X = torch.tensor(U, dtype=torch.float32); Y = torch.tensor(z, dtype=torch.float32)
        opt = torch.optim.Adam(list(self._enc.parameters()) + list(dec.parameters()) + [self._xi], lr=3e-3)
        for e in range(epochs):
            opt.zero_grad()
            Z = self._enc(X)
            yhat = self._theta(Z) @ (self._xi * mask)
            loss = ((yhat - Y) ** 2).mean() + 0.1 * ((dec(Z) - X) ** 2).mean() + 1e-4 * self._xi.abs().sum()
            loss.backward(); opt.step()
            if (e + 1) % 1000 == 0 and e + 1 < epochs:
                with torch.no_grad():
                    mask = (self._xi.abs() >= 0.05 * self._xi.abs().max()).float()
        self._mask = mask
        self.n_terms = int(mask.sum())

    def _theta(self, Z):
        torch = _torch()
        return torch.stack([torch.prod(torch.stack([Z[:, i] ** e for i, e in enumerate(ex)]), 0)
                            for ex in self._exps], 1)

    def _predict(self, U):
        torch = _torch()
        with torch.no_grad():
            Z = self._enc(torch.tensor(U, dtype=torch.float32))
            return (self._theta(Z) @ (self._xi * self._mask)).numpy().astype(float)
