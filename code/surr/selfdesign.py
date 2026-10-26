"""Self-designing surrogates: methods that choose their own evaluation points (they cannot fit a given design), so
they are studied as design + interpolant pairs against learners on a Sobol' design of the same size (E14).
API: fit(f, d, budget, seed) evaluates f (vectorised, unit cube) at most `budget` distinct points, then predict(U).

  smolyak    Smolyak sparse grid, combination technique, nested Clenshaw–Curtis points, tensor barycentric Lagrange
             interpolation; the level is the largest whose grid fits the budget          Smolyak 1963; Barthelmann,
             Novak & Ritter 2000
  tt_cross   tensor-train cross approximation on an m^d Chebyshev grid: alternating left/right sweeps with maxvol
             index selection, fixed rank r; the TT is interpolated along each axis; m, r and the number of sweeps are
             chosen to fit the budget                                                       Oseledets & Tyrtyshnikov
             2010; Savostyanov & Oseledets 2011
Both interpolate: noise is not smoothed (the known weakness; measured, not avoided — cheap runs).
"""
from __future__ import annotations

from itertools import product
from math import comb

import numpy as np

REGISTRY = {}


def _reg(cls):
    REGISTRY[cls.name] = cls
    return cls


# ── 1-D Chebyshev-extreme (Clenshaw–Curtis) nodes and barycentric interpolation ───────────────────────────────
def cc_nodes(m):
    if m == 1:
        return np.array([0.5])
    return (1 - np.cos(np.pi * np.arange(m) / (m - 1))) / 2


def bary_matrix(x, m):
    """(len(x), m) matrix of Lagrange basis values at x for the m Chebyshev-extreme nodes on [0, 1]."""
    x = np.asarray(x, float)
    if m == 1:
        return np.ones((len(x), 1))
    t = cc_nodes(m)
    w = (-1.0) ** np.arange(m); w[0] *= 0.5; w[-1] *= 0.5
    D = x[:, None] - t[None, :]
    hit = np.isclose(D, 0.0, atol=1e-14)
    D[hit] = 1.0
    B = w / D
    with np.errstate(divide="ignore", invalid="ignore"):          # rows at a node are overwritten just below
        B /= B.sum(axis=1, keepdims=True)
    rows = hit.any(axis=1)
    B[rows] = hit[rows].astype(float)
    return B


class _Counter:
    """Distinct-point evaluation counter with a cache (a self-designing method may revisit grid points)."""

    def __init__(self, f, budget):
        self.f, self.budget, self.cache = f, budget, {}

    def __call__(self, U):
        U = np.atleast_2d(U)
        keys = [u.tobytes() for u in np.round(U, 15)]
        new = [i for i, k in enumerate(keys) if k not in self.cache]
        if len(self.cache) + len({keys[i] for i in new}) > self.budget:
            raise _Budget()
        if new:
            vals = self.f(U[new])
            for i, v in zip(new, vals):
                self.cache[keys[i]] = float(v)
        return np.array([self.cache[k] for k in keys])

    @property
    def n(self):
        return len(self.cache)


class _Budget(Exception):
    pass


# ── Smolyak ───────────────────────────────────────────────────────────────────────────────────────────────────
def _m(level):
    return 1 if level == 1 else 2 ** (level - 1) + 1


def _smolyak_indices(d, q):
    """Multi-indices l (l_i >= 1) with q − d + 1 <= |l| <= q, and their combination coefficients."""
    out = []
    for l in product(range(1, q - d + 2), repeat=d):
        s = sum(l)
        if q - d + 1 <= s <= q:
            out.append((l, (-1) ** (q - s) * comb(d - 1, q - s)))
    return out


def smolyak_points(d, q):
    pts = set()
    for l, _ in _smolyak_indices(d, q):
        for p in product(*[range(_m(li)) for li in l]):
            pts.add(tuple(np.round([cc_nodes(_m(li))[j] for li, j in zip(l, p)], 15)))
    return np.array(sorted(pts))


@_reg
class Smolyak:
    name = "smolyak"

    def fit(self, f, d, budget, seed=0):
        q = d
        while len(smolyak_points(d, q + 1)) <= budget and q < d + 12:
            q += 1
        self.d, self.q = d, q
        F = _Counter(f, budget)
        self.terms = []
        for l, c in _smolyak_indices(d, q):
            axes = [cc_nodes(_m(li)) for li in l]
            G = np.array(np.meshgrid(*axes, indexing="ij")).reshape(d, -1).T
            self.terms.append((l, c, F(G).reshape([_m(li) for li in l])))
        self.n = F.n
        return self

    def predict(self, U):
        U = np.atleast_2d(U)
        out = np.zeros(len(U))
        for l, c, V in self.terms:
            T = V
            first = True
            for i, li in enumerate(l):
                B = bary_matrix(U[:, i], _m(li))
                T = np.tensordot(B, T, axes=([1], [0])) if first else np.einsum("nj,nj...->n...", B, T)
                first = False
            out += c * T
        return out


# ── TT-cross ──────────────────────────────────────────────────────────────────────────────────────────────────
def maxvol(A, tol=1.05, iters=100):
    """Rows of a tall (n, r) matrix forming a dominant r × r submatrix (Goreinov et al. 2010)."""
    n, r = A.shape
    if n <= r:
        return np.arange(n)
    import scipy.linalg as sl
    idx = np.array(sl.qr(A.T, pivoting=True, mode="r")[1][:r], dtype=int)   # pivoted-QR start, then swaps
    try:
        B = A @ np.linalg.inv(A[idx])
    except np.linalg.LinAlgError:
        B = A @ np.linalg.pinv(A[idx])
    for _ in range(iters):
        i, j = np.unravel_index(np.argmax(np.abs(B)), B.shape)
        if abs(B[i, j]) <= tol:
            break
        idx[j] = i
        bj, bi = B[:, j].copy(), B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])
    return idx


@_reg
class TTCross:
    name = "tt_cross"

    def fit(self, f, d, budget, seed=0, m=None, r=None, sweeps=None):
        rng = np.random.default_rng(seed)
        # largest grid / rank that fits two sweeps in the budget (each sweep ~ Σ_k r_{k-1} m r_k evaluations)
        best = None
        for mm in (5, 7, 9, 13, 17, 25, 33):
            for rr in range(1, 9):
                cost = 2 * sum(min(rr, mm ** k, mm ** (d - k)) * mm * min(rr, mm ** (k + 1), mm ** (d - k - 1))
                               for k in range(d))
                if cost <= budget and (best is None or mm * rr ** 1.5 > best[0] * best[1] ** 1.5):
                    best = (mm, rr)
        m, r = (m, r) if m and r else (best or (3, 1))
        self.m, self.d = m, d
        nodes = cc_nodes(m)
        F = _Counter(lambda U: f(U), budget)
        ranks = [1] + [min(r, m ** k, m ** (d - k)) for k in range(1, d)] + [1]
        # right index sets J[k]: (ranks[k+1], d-k-1) multi-indices of dims k+1..d-1
        J = [rng.integers(0, m, size=(ranks[k + 1], d - k - 1)) for k in range(d)]
        I = [np.zeros((1, 0), int)] + [None] * d
        self.cores = None
        for sweep in range(sweeps or 50):
            try:
                cores = []
                for k in range(d):
                    Ik, Jk = I[k], J[k]
                    idx = np.array([np.r_[a, j, b] for a in Ik for j in range(m) for b in Jk], int)
                    C = F(nodes[idx]).reshape(len(Ik) * m, len(Jk))
                    if k == d - 1:
                        cores.append(C.reshape(len(Ik), m, 1)); break
                    Q, _ = np.linalg.qr(C)
                    rows = maxvol(Q)
                    G = Q @ np.linalg.pinv(Q[rows])
                    cores.append(G.reshape(len(Ik), m, len(rows)))
                    pairs = [(a, j) for a in Ik for j in range(m)]
                    I[k + 1] = np.array([np.r_[pairs[t][0], pairs[t][1]] for t in rows], int)
                self.cores = cores
                # right-to-left: refresh J from the new left sets (columns maxvol on the unfolding transposes)
                for k in range(d - 1, 0, -1):
                    Ik, Jk = I[k], J[k]
                    idx = np.array([np.r_[a, j, b] for a in Ik for j in range(m) for b in Jk], int)
                    C = F(nodes[idx]).reshape(len(Ik), m * len(Jk))
                    Q, _ = np.linalg.qr(C.T)
                    cols = maxvol(Q)
                    pairs = [(j, b) for j in range(m) for b in Jk]
                    J[k - 1] = np.array([np.r_[pairs[t][0], pairs[t][1]] for t in cols], int)
            except _Budget:
                break
        if self.cores is None:
            raise RuntimeError("budget too small for one TT-cross sweep")
        self.n = F.n
        return self

    def predict(self, U):
        U = np.atleast_2d(U)
        v = np.ones((len(U), 1))
        for k, G in enumerate(self.cores):
            B = bary_matrix(U[:, k], self.m)
            Gk = np.einsum("nj,ajb->nab", B, G)
            v = np.einsum("na,nab->nb", v, Gk)
        return v[:, 0]


def make(name):
    return REGISTRY[name]()
