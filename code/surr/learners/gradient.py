"""Gradient-enhanced learners (E13). Each also runs without gradients (use_grad=False, or fit without G), which is
its value-only counterpart for the matched-cost comparison. API: fit(U, y, G=None) with G the (n, d) gradient of y
with respect to the unit-cube inputs.

  gek          gradient-enhanced kriging: GP on values and derivatives jointly; squared-exponential ARD kernel
               (closed-form derivative covariances), ML-II over length-scales, signal, value and gradient noise
               Morris, Mitchell & Ylvisaker 1993; Han, Görtz & Zimmermann 2013
  hermite_rbf  Hermite (generalised) Gaussian RBF interpolation of values and derivatives; one shape parameter by
               5-fold CV on the values; tiny nugget                                    Wu 1992; Fasshauer 2007 ch. 36
  ge_pce       Legendre polynomial chaos, total degree, least squares with derivative rows; degree by 5-fold CV
               Roderick, Anitescu & Fischer 2010; Peng, Hampton & Doostan 2016
  sobolev_mlp  MLP ensemble (3 × 2×64 SiLU) trained on values + λ·gradients (Sobolev training); λ = 0 is the
               value-only network with the same architecture and budget                 Czarnecki et al. 2017

Cost walls (bound by cost, not by failure): GEK and Hermite RBF solve an n(d+1) system — capped at 2000 rows with
gradients; ge_pce needs fewer basis terms than half the rows.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .base import Learner, register

GE_ROWS = 2000


def _rows(n, d, use_grad):
    return n * (d + 1) if use_grad else n


# ── squared-exponential kernel with derivative blocks ─────────────────────────────────────────────────────────
def _se_blocks(A, B, ell, s2, grads_a, grads_b):
    """Joint covariance of [f(A); ∂f(A)] and [f(B); ∂f(B)] (derivative blocks only where requested), for
    k(x, x') = s2 exp(−½ Σ_j (x_j − x'_j)² / ℓ_j²)."""
    D = A[:, None, :] - B[None, :, :]                                    # Δ = x − x'
    il2 = 1.0 / ell ** 2
    k = s2 * np.exp(-0.5 * (D ** 2 * il2).sum(-1))
    Dl = D * il2                                                         # Δ_j / ℓ_j²
    d = A.shape[1]
    rows = [[k] + ([k * Dl[..., j] for j in range(d)] if grads_b else [])]
    if grads_a:
        for i in range(d):
            row = [-k * Dl[..., i]]                                      # cov(∂_i f(x), f(x'))
            if grads_b:
                row += [k * ((i == j) * il2[i] - Dl[..., i] * Dl[..., j]) for j in range(d)]
            rows.append(row)
    return np.block(rows)


class _JointGP:
    """GP conditioned on values (and optionally gradients) with the SE kernel."""

    def __init__(self, U, z, g, ell, s2, nv, ng):
        self.U, self.ell, self.s2, self.ug = U, ell, s2, g is not None
        n, d = U.shape
        K = _se_blocks(U, U, ell, s2, self.ug, self.ug)
        noise = np.full(K.shape[0], nv)
        if self.ug:
            noise[n:] = ng
        K[np.diag_indices_from(K)] += noise + 1e-10 * s2
        self.cf = cho_factor(K, lower=True)
        self.t = np.concatenate([z, g.T.ravel()]) if self.ug else z
        self.alpha = cho_solve(self.cf, self.t)

    def nll(self):
        L = self.cf[0]
        return 0.5 * self.t @ self.alpha + np.log(np.diag(L)).sum() + 0.5 * len(self.t) * np.log(2 * np.pi)

    def mean(self, X):
        return _se_blocks(X, self.U, self.ell, self.s2, False, self.ug) @ self.alpha

    def std(self, X):
        Ks = _se_blocks(X, self.U, self.ell, self.s2, False, self.ug)
        v = cho_solve(self.cf, Ks.T)
        return np.sqrt(np.maximum(self.s2 - (Ks * v.T).sum(1), 1e-300))

    def grad(self, X):
        n = len(X); d = X.shape[1]
        Kg = _se_blocks(X, self.U, self.ell, self.s2, True, self.ug)[n:]      # rows ∂_i f(X), stacked by i
        return (Kg @ self.alpha).reshape(d, n).T


class _GELearner(Learner):
    family, uq, smooth = "gradient-enhanced", False, True

    def fit(self, U, y, G=None):
        U = np.atleast_2d(np.asarray(U, float)); self.d = U.shape[1]
        z = self._yfit(y)
        use = G is not None and self.kw.get("use_grad", True) not in (False, "False", "0")
        g = np.asarray(G, float) / self._ys if use else None
        if _rows(len(U), self.d, use) > self.kw.get("max_rows", GE_ROWS) and self.cost_wall:
            raise MemoryError(f"{self.name}: {_rows(len(U), self.d, use)} rows exceeds the cost wall")
        self._fit_g(U, z, g)
        return self

    cost_wall = True

    def grad(self, U, h=1e-5):
        return self._grad(np.atleast_2d(np.asarray(U, float))) * self._ys


@register
class GEK(_GELearner):
    name, year, field, ref, uq = "gek", 1993, "computer experiments / aerospace", "morris1993bayesian", True

    @staticmethod
    def feasible(n, d):
        return n * (d + 1) <= GE_ROWS

    def _fit_g(self, U, z, g):
        from scipy.optimize import minimize
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
        d = U.shape[1]
        # start from the value-only ML-II fit (cheap), then optimise the joint likelihood
        k0 = C(1.0, (1e-3, 1e3)) * RBF(np.full(d, 0.3), (1e-3, 1e3)) + WhiteKernel(1e-4, (1e-8, 1.0))
        gp0 = GaussianProcessRegressor(k0, random_state=self.seed).fit(U, z)
        kp = gp0.kernel_.get_params()
        th0 = np.r_[np.log(np.atleast_1d(kp["k1__k2__length_scale"])), np.log(kp["k1__k1__constant_value"]),
                    np.log(max(kp["k2__noise_level"], 1e-8)), np.log(max(kp["k2__noise_level"], 1e-8))]
        if g is None:
            th0 = th0[:-1]

        def unpack(th):
            ell, s2, nv = np.exp(th[:d]), np.exp(th[d]), np.exp(th[d + 1])
            ng = np.exp(th[d + 2]) if g is not None else 0.0
            return ell, s2, nv, ng

        def f(th):
            try:
                return _JointGP(U, z, g, *unpack(th)).nll()
            except np.linalg.LinAlgError:
                return 1e25
        lo = np.r_[np.full(d, np.log(1e-3)), np.log(1e-3), np.log(1e-8)] ; hi = np.r_[np.full(d, np.log(1e3)), np.log(1e3), 0.0]
        if g is not None:
            lo, hi = np.r_[lo, np.log(1e-8)], np.r_[hi, 0.0]
        r = minimize(f, np.clip(th0, lo, hi), method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options=dict(maxiter=self.kw.get("maxiter", 40)))
        self._gp = _JointGP(U, z, g, *unpack(r.x))
        self.theta = r.x

    def _predict(self, U):
        return self._gp.mean(U)

    def _predict_std(self, U):
        return self._gp.std(U)

    def _grad(self, U):
        return self._gp.grad(U)


@register
class HermiteRBF(_GELearner):
    name, year, field, ref = "hermite_rbf", 1992, "approximation theory", "wu1992hermite"

    @staticmethod
    def feasible(n, d):
        return n * (d + 1) <= GE_ROWS

    def _fit_g(self, U, z, g):
        from sklearn.model_selection import KFold
        best = (np.inf, None)
        for ell in np.geomspace(0.05, 2.0, 8):                          # shape parameter by 5-fold CV on values
            err = 0.0
            for tr, te in KFold(5, shuffle=True, random_state=self.seed).split(U):
                try:
                    m = _JointGP(U[tr], z[tr], None if g is None else g[tr], np.full(U.shape[1], ell), 1.0, 1e-8, 1e-8)
                    err += np.sum((m.mean(U[te]) - z[te]) ** 2)
                except np.linalg.LinAlgError:
                    err = np.inf; break
            if err < best[0]:
                best = (err, ell)
        self._gp = _JointGP(U, z, g, np.full(U.shape[1], best[1]), 1.0, 1e-8, 1e-8)
        self.ell = best[1]

    def _predict(self, U):
        return self._gp.mean(U)

    def _grad(self, U):
        return self._gp.grad(U)


# ── gradient-enhanced polynomial chaos ────────────────────────────────────────────────────────────────────────
def _legendre(x, p):
    """P_0..P_p and derivatives at x ∈ [−1, 1] (three-term recurrence), shape (len(x), p + 1)."""
    P = np.zeros((len(x), p + 1)); dP = np.zeros_like(P)
    P[:, 0] = 1.0
    if p >= 1:
        P[:, 1] = x; dP[:, 1] = 1.0
    for k in range(1, p):
        P[:, k + 1] = ((2 * k + 1) * x * P[:, k] - k * P[:, k - 1]) / (k + 1)
        dP[:, k + 1] = dP[:, k - 1] + (2 * k + 1) * P[:, k]
    return P, dP


def _multi_indices(d, p):
    out = [()]
    for _ in range(d):
        out = [a + (k,) for a in out for k in range(p + 1)]
    return np.array([a for a in out if sum(a) <= p])


class _PCEBasis:
    def __init__(self, d, p):
        self.d, self.p, self.A = d, p, _multi_indices(d, p)

    def values(self, U):
        X = 2 * U - 1
        Ps = [_legendre(X[:, j], self.p) for j in range(self.d)]
        V = np.ones((len(U), len(self.A)))
        for j in range(self.d):
            V *= Ps[j][0][:, self.A[:, j]]
        return V, Ps

    def grads(self, U, Ps=None):
        """(d, n, P): ∂Ψ/∂u_j = 2 ∂Ψ/∂x_j."""
        if Ps is None:
            _, Ps = self.values(U)
        out = np.empty((self.d, len(U), len(self.A)))
        for j in range(self.d):
            M = np.ones((len(U), len(self.A)))
            for i in range(self.d):
                M *= (2 * Ps[i][1] if i == j else Ps[i][0])[:, self.A[:, i]]
            out[j] = M
        return out


@register
class GEPCE(_GELearner):
    name, year, field, ref = "ge_pce", 2010, "uncertainty quantification", "roderick2010polynomial"
    cost_wall = False

    def _design(self, B, U, z, g):
        V, Ps = B.values(U)
        if g is None:
            return V, z
        Gd = B.grads(U, Ps)
        return np.vstack([V, *Gd]), np.concatenate([z, g.T.ravel()])

    def _fit_g(self, U, z, g):
        from math import comb
        from sklearn.model_selection import KFold
        n, d = U.shape
        rows = _rows(n, d, g is not None)
        degs = [p for p in range(1, 13) if comb(d + p, p) <= rows / 2] or [1]
        best = (np.inf, degs[0])
        for p in degs:
            B = _PCEBasis(d, p); err = 0.0
            for tr, te in KFold(5, shuffle=True, random_state=self.seed).split(U):
                M, t = self._design(B, U[tr], z[tr], None if g is None else g[tr])
                c = np.linalg.lstsq(np.vstack([M, 1e-6 * np.eye(M.shape[1])]), np.r_[t, np.zeros(M.shape[1])],
                                    rcond=None)[0]
                err += np.sum((B.values(U[te])[0] @ c - z[te]) ** 2)
            if err < best[0]:
                best = (err, p)
        self.B = _PCEBasis(d, best[1])
        M, t = self._design(self.B, U, z, g)
        self.c = np.linalg.lstsq(np.vstack([M, 1e-6 * np.eye(M.shape[1])]), np.r_[t, np.zeros(M.shape[1])],
                                 rcond=None)[0]
        self.degree = best[1]

    def _predict(self, U):
        return self.B.values(U)[0] @ self.c

    def _grad(self, U):
        return np.einsum("jnp,p->nj", self.B.grads(U), self.c)


# ── Sobolev-trained network ensemble ──────────────────────────────────────────────────────────────────────────
@register
class SobolevMLP(_GELearner):
    name, year, field, ref, uq = "sobolev_mlp", 2017, "machine learning", "czarnecki2017sobolev", True
    cost_wall = False

    def _nets(self):
        import torch
        return [torch.nn.Sequential(torch.nn.Linear(self.d, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64),
                                    torch.nn.SiLU(), torch.nn.Linear(64, 1)) for _ in range(self.kw.get("K", 3))]

    def _fit_g(self, U, z, g):
        import torch
        torch.manual_seed(self.seed)
        lam = float(self.kw.get("lam", 1.0))
        X = torch.tensor(U, dtype=torch.float64); Z = torch.tensor(z, dtype=torch.float64)
        Gt = torch.tensor(g, dtype=torch.float64) if g is not None else None
        epochs = int(self.kw.get("epochs", 2000))
        self.nets = []
        for net in self._nets():
            net = net.double()
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-6)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            for _ in range(epochs):
                opt.zero_grad()
                Xr = X.clone().requires_grad_(Gt is not None and lam > 0)
                out = net(Xr).squeeze(-1)
                loss = ((out - Z) ** 2).mean()
                if Gt is not None and lam > 0:
                    dg = torch.autograd.grad(out.sum(), Xr, create_graph=True)[0]
                    loss = loss + lam * ((dg - Gt) ** 2).mean()
                loss.backward(); opt.step(); sched.step()
            self.nets.append(net.eval())

    def _all(self, U):
        import torch
        with torch.no_grad():
            X = torch.tensor(U, dtype=torch.float64)
            return np.stack([n(X).squeeze(-1).numpy() for n in self.nets])

    def _predict(self, U):
        return self._all(U).mean(0)

    def _predict_std(self, U):
        return self._all(U).std(0)

    def _grad(self, U):
        import torch
        X = torch.tensor(U, dtype=torch.float64, requires_grad=True)
        out = sum(n(X).squeeze(-1) for n in self.nets) / len(self.nets)
        return torch.autograd.grad(out.sum(), X)[0].numpy()
