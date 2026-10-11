"""Rational surrogates in several variables (2026-10-11).

Established (baselines):
  paaa          p-AAA for scattered data: multivariate barycentric form on a tensor grid of greedily chosen nodes,
                interpolation at grid-compatible samples, linearised least squares solved by SVD
                (Carracedo Rodriguez, Balicki & Gugercin 2023; scattered-data form: Balicki & Gugercin 2025)
  rat_ls        least-squares multivariate rational p(x)/q(x) in a total-degree Legendre basis, linearised LS by SVD
                plus Sanathanan–Koerner reweighting (Austin et al. 2021, the Apprentice approach)

New pairings (E12; no prior use found in the 2026-10-11 novelty search, data/literature/notes/aaa_multivariate.md):
  ridge_aaa     univariate AAA along the active-subspace direction of a GP (rational ridge function)
  additive_aaa  additive model with a univariate AAA term per input, fitted by backfitting
  gp_ratmean    GP on the residuals of a rational ridge trend (rational mean function + GP)

Run walls are set by cost, not by expected failure (STUDY §9): cheap runs in regimes where a method is expected to
fail are kept on purpose as low-cost evidence.

Noise caveat (stated in the paper): AAA-type methods interpolate; on noisy data they create spurious pole–zero pairs
(Froissart doublets). Here the number of support points / iterations is chosen on a 20 % hold-out, which acts as the
least-squares stopping rule.
"""
from __future__ import annotations

from itertools import product
from math import comb

import numpy as np

from .base import Learner, register

_EPS = 1e-12


# ── univariate AAA with hold-out stopping (used by the pairings) ─────────────────────────────────────────────
def aaa(x, y, mmax):
    """Plain AAA (Nakatsukasa, Sète & Trefethen 2018) on distinct real nodes; returns (support, values, weights)."""
    J = np.ones(len(x), bool); zs, fs = [], []; R = np.full(len(x), y.mean())
    for _ in range(min(mmax, len(x) - 1)):
        j = int(np.argmax(np.abs(y - R) * J)); zs.append(x[j]); fs.append(y[j]); J[j] = False
        C = 1.0 / (x[J][:, None] - np.array(zs)[None, :])
        A = (y[J][:, None] - np.array(fs)[None, :]) * C
        w = np.linalg.svd(A, full_matrices=False)[2][-1]
        R = y.copy(); R[J] = (C @ (w * fs)) / (C @ w)
    return np.array(zs), np.array(fs), (w if zs else np.array([]))


def aaa_eval(p, x):
    zs, fs, w = p
    if len(zs) == 0:
        return np.zeros(len(x))
    D = x[:, None] - zs[None, :]
    hit = np.abs(D) < _EPS
    D[hit] = 1.0
    C = 1.0 / D
    den = C @ w
    r = (C @ (w * fs)) / np.where(np.abs(den) < _EPS, _EPS, den)
    rows, cols = np.nonzero(hit)
    r[rows] = fs[cols]
    return r


def aaa_poles(p):
    """Poles of the barycentric form: finite eigenvalues of the arrowhead pencil (E, B) (Nakatsukasa et al. 2018)."""
    from scipy.linalg import eigvals
    zs, _, w = p
    m = len(zs)
    if m < 2:
        return np.array([])
    E = np.zeros((m + 1, m + 1)); E[0, 1:] = w; E[1:, 0] = 1.0; E[1:, 1:] = np.diag(zs)
    B = np.eye(m + 1); B[0, 0] = 0.0
    ev = eigvals(E, B)
    return ev[np.isfinite(ev)]


def pole_free(p, lo, hi, tol=1e-6, yspan=None):
    """Froissart-doublet cleanup: True when no (numerically) real pole lies in [lo, hi] (widened by 1 %) and, if
    `yspan` is given, the approximant stays within 3 data spans of the data mean on a dense grid of the interval —
    which also rejects complex poles close enough to the axis to spike between samples."""
    pl = aaa_poles(p)
    pad = 0.01 * (hi - lo + _EPS)
    real = pl[np.abs(pl.imag) <= tol * max(1.0, hi - lo)]
    if np.any((real.real >= lo - pad) & (real.real <= hi + pad)):
        return False
    if yspan is not None:
        g = aaa_eval(p, np.linspace(lo, hi, 2001))
        return bool(np.all(np.isfinite(g)) and np.max(np.abs(g - yspan[0])) <= 3 * yspan[1] + _EPS)
    return True


def _dedupe(x, y):
    xs, inv = np.unique(np.round(x, 12), return_inverse=True)
    return xs, np.bincount(inv, weights=y) / np.bincount(inv)


def binned(x, y, nb=None):
    """Quantile-bin means of (x, y). Partial residuals / projected samples scatter around the 1-D signal (the other
    inputs act as noise), and AAA's greedy step would interpolate that scatter; averaging within bins removes it."""
    nb = nb or int(np.clip(len(x) // 5, 8, 60))
    q = np.quantile(x, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, nb - 1)
    keep = np.bincount(b, minlength=nb) > 0
    cnt = np.bincount(b, minlength=nb)[keep]
    return (np.bincount(b, weights=x, minlength=nb)[keep] / cnt, np.bincount(b, weights=y, minlength=nb)[keep] / cnt)


def aaa_holdout(x, y, seed, mmax=40, bins=False):
    """AAA whose number of support points is chosen on a 20 % hold-out (the least-squares stopping rule);
    `bins=True` fits the quantile-bin means instead of the raw scatter."""
    x, y = binned(x, y) if bins else _dedupe(x, y)
    if len(x) < 6:
        return aaa(x, y, max(len(x) - 1, 1))
    idx = np.random.default_rng(seed).permutation(len(x)); h = max(2, len(x) // 5)
    va, tr = idx[:h], idx[h:]
    lo, hi = x.min(), x.max(); ys = (y.mean(), np.ptp(y))
    best, bm = np.inf, 1
    for m in range(1, min(mmax, len(tr) - 1) + 1):
        p = aaa(x[tr], y[tr], m)
        if not pole_free(p, lo, hi, yspan=ys):             # spurious pole / spike inside the data range: reject
            continue
        e = np.mean((aaa_eval(p, x[va]) - y[va]) ** 2)
        if e < best:
            best, bm = e, m
    for m in range(bm, 0, -1):                             # refit on all data; step down until pole-free
        p = aaa(x, y, m)
        if pole_free(p, lo, hi, yspan=ys):
            return p
    return aaa(x, y, 1)


# ── established: p-AAA for scattered data ────────────────────────────────────────────────────────────────────
def _cauchy(x, nodes):
    D = x[:, None] - np.asarray(nodes)[None, :]
    hit = np.abs(D) < _EPS
    D[hit] = 1.0
    C = 1.0 / D
    rows = hit.any(1)
    C[rows] = hit[rows].astype(float)                      # at a node the basis becomes the indicator
    return C


def _khatri_rao(mats):
    out = mats[0]
    for M in mats[1:]:
        out = (out[:, :, None] * M[:, None, :]).reshape(len(out), -1)
    return out


@register
class PAAA(Learner):
    """p-AAA, scattered-data form. r = Σ β_I Π_j 1/(x_j − λ_{j,i_j}) / Σ α_I Π_j ..., nodes per variable added
    greedily at the worst-fit sample; samples on the node grid are interpolated (β_I = α_I f); the rest enter a
    linearised least-squares problem solved by the SVD. Iteration count chosen on a 20 % hold-out. The number of
    unknowns is the product of the node counts, so in d variables only ≈ (n/2)^(1/d) nodes per variable fit: at
    d ≥ 5 and our n that is 1–3 nodes (expected to fail, cheap, so it runs)."""
    name, family, year, field, ref, smooth = "paaa", "rational", 2023, "model reduction / numerical analysis", "carracedo2023paaa", True
    max_n = 1000                                           # cost wall: SVD of n x 2N with N <= n/2, per iteration

    def _fit(self, U, z):
        idx = np.random.default_rng(self.seed).permutation(len(z)); h = max(2, len(z) // 5)
        va, tr = idx[:h], idx[h:]
        hist = self._run(U[tr], z[tr])
        errs = [np.mean((self._eval(p, U[va]) - z[va]) ** 2) for p in hist]
        k = int(np.argmin(errs))
        self._p = self._run(U, z, iters=k + 1)[-1]

    def _run(self, U, z, iters=None):
        n, d = U.shape
        nodes = [[] for _ in range(d)]
        hist = []
        r = np.full(n, z.mean())
        cap = max(1, int((n / 2) ** (1 / d)))
        for _ in range(iters or cap):
            j = int(np.argmax(np.abs(z - r)))
            grown = [list(nd) + ([U[j, v]] if not np.any(np.isclose(nd, U[j, v], atol=_EPS)) else [])
                     for v, nd in enumerate(nodes)]
            if np.prod([len(g) for g in grown]) > n / 2:
                break
            nodes = grown
            p = self._solve(U, z, nodes)
            hist.append(p); r = self._eval(p, U)
        return hist or [dict(nodes=None, mean=z.mean())]

    def _solve(self, U, z, nodes):
        d = U.shape[1]
        Cs = [_cauchy(U[:, v], nodes[v]) for v in range(d)]
        Phi = _khatri_rao(Cs)                                  # n x N
        N = Phi.shape[1]
        # samples lying on the node grid: their multi-index -> linear index (constrained coefficients)
        on = np.all([np.isclose(U[:, v][:, None], np.asarray(nodes[v])[None, :], atol=_EPS).any(1)
                     for v in range(d)], axis=0)
        cidx, H = [], []
        dims = [len(nd) for nd in nodes]
        for s in np.nonzero(on)[0]:
            mi = [int(np.argmin(np.abs(np.asarray(nodes[v]) - U[s, v]))) for v in range(d)]
            li = int(np.ravel_multi_index(mi, dims))
            if li not in cidx:
                cidx.append(li); H.append(z[s])
        cidx = np.array(cidx, int); H = np.array(H)
        uidx = np.setdiff1d(np.arange(N), cidx)
        Aa = z[:, None] * Phi
        if len(cidx):
            Aa[:, cidx] -= Phi[:, cidx] * H[None, :]
        A = np.hstack([Aa, -Phi[:, uidx]])
        v = np.linalg.svd(A, full_matrices=False)[2][-1]
        alpha = v[:N]; beta = np.zeros(N); beta[uidx] = v[N:]
        if len(cidx):
            beta[cidx] = alpha[cidx] * H
        return dict(nodes=[list(nd) for nd in nodes], alpha=alpha, beta=beta)

    @staticmethod
    def _eval(p, U):
        if p.get("nodes") is None:
            return np.full(len(U), p["mean"])
        Phi = _khatri_rao([_cauchy(U[:, v], p["nodes"][v]) for v in range(U.shape[1])])
        den = Phi @ p["alpha"]
        return (Phi @ p["beta"]) / np.where(np.abs(den) < _EPS, _EPS, den)

    def _predict(self, U):
        return self._eval(self._p, U)


# ── established: least-squares multivariate rational p/q ─────────────────────────────────────────────────────
def _legendre_basis(U, deg):
    X = 2 * U - 1
    d = X.shape[1]
    P1 = [np.polynomial.legendre.legvander(X[:, j], deg) for j in range(d)]     # (n, deg+1) per input
    idx = [a for a in product(range(deg + 1), repeat=d) if sum(a) <= deg] if d <= 6 else _total_degree(d, deg)
    return np.column_stack([np.prod([P1[j][:, a[j]] for j in range(d)], axis=0) for a in idx])


def _total_degree(d, deg):
    out = [(0,) * d]
    for t in range(1, deg + 1):
        def rec(prefix, left, k):
            if k == d - 1:
                yield prefix + (left,); return
            for i in range(left + 1):
                yield from rec(prefix + (i,), left - i, k + 1)
        out += list(rec((), t, 0))
    return out


@register
class RatLS(Learner):
    """r = p/q with p, q total-degree (m, k) Legendre expansions; min ‖y·q − p‖ with ‖coefficients‖ = 1 (SVD),
    then 3 Sanathanan–Koerner reweightings (rows / |q_prev|). (m, k) chosen on a 20 % hold-out among degree pairs
    whose total coefficient count is ≤ n/2 and ≤ 400 (cost wall)."""
    name, family, year, field, ref, smooth = "rat_ls", "rational", 2021, "computational physics / numerical analysis", "austin2021practical", True

    PAIRS = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 2), (4, 4), (5, 3), (6, 4), (8, 4)]

    def _fit(self, U, z):
        n, d = U.shape
        ok = [(m, k) for m, k in self.PAIRS if comb(d + m, m) + comb(d + k, k) <= min(n / 2, 400)] or [(1, 1)]
        idx = np.random.default_rng(self.seed).permutation(n); h = max(2, n // 5)
        va, tr = idx[:h], idx[h:]
        err = {}
        for m, k in ok:
            c = self._solve(U[tr], z[tr], m, k)
            err[(m, k)] = np.mean((self._eval(c, U[va]) - z[va]) ** 2)
        m, k = min(err, key=err.get)
        self._c = self._solve(U, z, m, k)

    def _solve(self, U, z, m, k, sk=3):
        P, Q = _legendre_basis(U, m), _legendre_basis(U, k)
        w = np.ones(len(z))
        for _ in range(sk + 1):
            A = np.hstack([-P, z[:, None] * Q]) * w[:, None]
            v = np.linalg.svd(A, full_matrices=False)[2][-1]
            a, b = v[:P.shape[1]], v[P.shape[1]:]
            w = 1.0 / np.maximum(np.abs(Q @ b), 1e-8)
        return dict(m=m, k=k, a=a, b=b)

    @staticmethod
    def _eval(c, U):
        den = _legendre_basis(U, c["k"]) @ c["b"]
        return (_legendre_basis(U, c["m"]) @ c["a"]) / np.where(np.abs(den) < _EPS, _EPS, den)

    def _predict(self, U):
        return self._eval(self._c, U)


# ── new pairings (E12) ───────────────────────────────────────────────────────────────────────────────────────
def _active_direction(U, z, seed):
    from .kernels import _sk_gp
    g = _sk_gp(U.shape[1], 2.5, 1, seed).fit(U, z)
    h = 1e-4; G = np.empty_like(U)
    for j in range(U.shape[1]):
        e = np.zeros(U.shape[1]); e[j] = h
        G[:, j] = (g.predict(U + e) - g.predict(U - e)) / (2 * h)
    lam, V = np.linalg.eigh(G.T @ G / len(z))
    return V[:, -1], g


@register
class RidgeAAA(Learner):
    """Rational ridge function r(wᵀx): w = leading active-subspace direction of an ML-II GP fitted to the data,
    r = univariate AAA on the projected samples with hold-out stopping. NEW (E12)."""
    name, family, year, field, ref, smooth = "ridge_aaa", "rational", 2026, "this study (E12)", "", True

    def _fit(self, U, z):
        self._w, _ = _active_direction(U, z, self.seed)
        self._p = aaa_holdout(U @ self._w, z, self.seed, bins=True)

    def _predict(self, U):
        return aaa_eval(self._p, U @ self._w)


@register
class AdditiveAAA(Learner):
    """Additive model whose 1-D terms are AAA rational functions where that helps: start from a penalised-spline
    GAM (one s(x_j) per input), then for each input fit pole-free AAA (largest support size <= 12 with no real pole or
    spike in the data range, Froissart cleanup) to the binned partial residual and keep it only if it beats the spline
    term on the bins by > 5 %; two backfitting sweeps. NEW (E12).
    Why hybrid: at ~4 % residual noise (the partial residuals of a sine sum after the spline fit) plain AAA places
    real poles inside the interval at every degree >= 4 (E12 smoke test, 2026-10-11), so a pure-AAA additive model
    collapses to constants; `n_rational` records how many terms ended up rational."""
    name, family, year, field, ref, smooth = "additive_aaa", "rational", 2026, "this study (E12)", "", True

    def _fit(self, U, z, sweeps=2):
        from pygam import LinearGAM, s as _s
        d = U.shape[1]
        tt = _s(0, n_splines=12)
        for j in range(1, d):
            tt = tt + _s(j, n_splines=12)
        self._gm = LinearGAM(tt).gridsearch(U, z, lam=np.logspace(-3, 3, 7), progress=False)
        F = np.column_stack([self._gm.partial_dependence(term=j, X=U) for j in range(d)])
        self._c = z.mean(); self._shift = list(F.mean(0)); F = F - F.mean(0)
        self._terms = [None] * d                              # None = keep the spline term
        for _ in range(sweeps):
            for j in range(d):
                res = z - self._c - F.sum(1) + F[:, j]
                xs, ys = binned(U[:, j], res)
                spl = self._gm.partial_dependence(term=j, X=np.column_stack([xs] * d)) - self._shift[j]                     if self._terms[j] is None else aaa_eval(self._terms[j], xs) - self._shift[j]
                base = np.sqrt(np.mean((spl - ys) ** 2))
                best = None
                for mm in range(min(12, len(xs) - 1), 1, -1):
                    t = aaa(xs, ys, mm)
                    if pole_free(t, xs.min(), xs.max(), yspan=(ys.mean(), np.ptp(ys))):
                        best = t; break
                if best is not None:
                    raw = aaa_eval(best, xs)
                    if np.sqrt(np.mean((raw - raw.mean() - (ys - ys.mean())) ** 2)) < 0.95 * base:
                        self._terms[j] = best
                        r = aaa_eval(best, U[:, j]); self._shift[j] = r.mean(); F[:, j] = r - r.mean()
        self.n_rational = sum(t is not None for t in self._terms)

    def _predict(self, U):
        out = np.full(len(U), self._c)
        for j, t in enumerate(self._terms):
            v = self._gm.partial_dependence(term=j, X=U) if t is None else aaa_eval(t, U[:, j])
            out += v - self._shift[j]
        return out


@register
class GPRatMean(Learner):
    """GP with a rational mean: trend = rational ridge function (ridge_aaa), then an ML-II GP on the residuals. The
    GP posterior std is the UQ. NEW (E12)."""
    name, family, year, field, ref, uq, smooth = "gp_ratmean", "rational", 2026, "this study (E12)", "", True, True
    max_n = 5000

    def _fit(self, U, z):
        from .kernels import _sk_gp
        self._w, _ = _active_direction(U, z, self.seed)
        self._p = aaa_holdout(U @ self._w, z, self.seed, bins=True)
        self._g = _sk_gp(U.shape[1], 2.5, 1, self.seed).fit(U, z - aaa_eval(self._p, U @ self._w))

    def _predict(self, U):
        return aaa_eval(self._p, U @ self._w) + self._g.predict(U)

    def _predict_std(self, U):
        return self._g.predict(U, return_std=True)[1]
