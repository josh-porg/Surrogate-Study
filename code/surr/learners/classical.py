"""Classical, early and largely discontinued surrogate methods, implemented from their original descriptions.

  rsm2            Box & Wilson (1951)              quadratic response surface            chemical engineering
  ok_variogram    Krige (1951) / Matheron (1963)    ordinary kriging, fitted variogram    mining geostatistics
  gmdh            Ivakhnenko (1968/1971)            self-organising polynomial network     cybernetics (USSR)
  idw             Shepard (1968)                    inverse-distance weighting             geography / cartography
  rbf_mq          Hardy (1971)                      multiquadric RBF (Rippa LOOCV tuning)   geodesy / topography
  rbf_tps         Duchon (1977)                     thin-plate spline                       approximation theory
  ppr             Friedman & Stuetzle (1981)        projection pursuit regression           statistics
  mls             Lancaster & Šalkauskas (1981)     moving least squares                    computer graphics / meshfree
  knn             Fix & Hodges (1951)               k-nearest-neighbour average             statistics
  cart            Breiman et al. (1984)             single regression tree                  statistics
  mars            Friedman (1991)                   multivariate adaptive regression splines statistics
  lipschitz       Sukharev (1978) / Beliakov (2006) central Lipschitz interpolant           optimal recovery
  delaunay_linear (thesis `linear` baseline)        piecewise-linear on Delaunay simplices  computational geometry
"""
from __future__ import annotations

import itertools

import numpy as np

from .base import Learner, pairwise_dist, register


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@register
class RSM2(Learner):
    """Full quadratic response surface by (ridge-stabilised) least squares."""
    name, family, year, field, ref, smooth = "rsm2", "polynomial", 1951, "chemical engineering", "box1951experimental", True

    @staticmethod
    def _feat(U):
        n, d = U.shape
        cols = [np.ones(n)] + [U[:, i] for i in range(d)]
        cols += [U[:, i] * U[:, j] for i in range(d) for j in range(i, d)]
        return np.column_stack(cols)

    def _fit(self, U, z):
        F = self._feat(U)
        lam = 1e-8 * np.trace(F.T @ F) / F.shape[1]
        self._b = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ z)
        # OLS predictive variance (for UQ-less use as a baseline only)
        r = z - F @ self._b
        self._s2 = r @ r / max(len(z) - F.shape[1], 1)
        self._G = np.linalg.pinv(F.T @ F + lam * np.eye(F.shape[1]))

    def _predict(self, U):
        return self._feat(U) @ self._b

    def _predict_std(self, U):
        F = self._feat(U)
        return np.sqrt(self._s2 * (1 + np.einsum("ij,jk,ik->i", F, self._G, F)))


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@register
class OrdinaryKrigingVariogram(Learner):
    """Geostatistical ordinary kriging: empirical semivariogram in distance bins, weighted least-squares fit of a
    spherical or exponential model (Cressie weights), then the ordinary-kriging system with a Lagrange multiplier.
    Isotropic, no likelihood optimisation — the classical mining workflow, distinct from ML-II GP regression."""
    name, family, year, field, ref, uq, smooth = ("ok_variogram", "kriging", 1963, "mining geostatistics",
                                                  "matheron1963principles", True, False)
    max_n = 5000

    def __init__(self, model="spherical", n_bins=15, **kw):
        super().__init__(**kw); self.model, self.n_bins = model, n_bins

    def _gamma(self, h, nug, sill, rng):
        h = np.asarray(h, float)
        if self.model == "spherical":
            g = np.where(h < rng, nug + (sill - nug) * (1.5 * h / rng - 0.5 * (h / rng) ** 3), sill)
        else:
            g = nug + (sill - nug) * (1 - np.exp(-3 * h / rng))
        return np.where(h > 0, g, 0.0)

    def _fit(self, U, z):
        from scipy.optimize import minimize
        D = pairwise_dist(U, U)
        iu = np.triu_indices(len(z), 1)
        h, g = D[iu], 0.5 * (z[iu[0]] - z[iu[1]]) ** 2
        edges = np.linspace(0, h.max() * 0.6, self.n_bins + 1)
        hb, gb, nb = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (h >= a) & (h < b)
            if m.sum() >= 3:
                hb.append(h[m].mean()); gb.append(g[m].mean()); nb.append(m.sum())
        hb, gb, nb = map(np.asarray, (hb, gb, nb))

        def loss(t):
            nug, sill, rng = np.exp(t)
            gm = self._gamma(hb, nug, nug + sill, rng)
            return np.sum(nb * (gb / np.maximum(gm, 1e-12) - 1) ** 2)          # Cressie (1985) WLS
        t0 = np.log([max(gb.min(), 1e-3), max(gb.max(), 1e-2), max(hb.mean(), 1e-2)])
        t = minimize(loss, t0, method="Nelder-Mead").x
        nug, ps, rng = np.exp(t)
        self._par = (nug, nug + ps, rng)
        n = len(z)
        G = self._gamma(D, *self._par)
        A = np.zeros((n + 1, n + 1)); A[:n, :n] = G; A[:n, n] = A[n, :n] = 1.0
        self._Ainv = np.linalg.pinv(A)
        self._U, self._z = U, z

    def _weights(self, U):
        g0 = self._gamma(pairwise_dist(self._U, U), *self._par)            # (n, m)
        rhs = np.vstack([g0, np.ones((1, U.shape[0]))])
        return self._Ainv @ rhs, rhs

    def _predict(self, U):
        W, _ = self._weights(U)
        return W[:-1].T @ self._z

    def _predict_std(self, U):
        W, rhs = self._weights(U)
        return np.sqrt(np.maximum((W * rhs).sum(0), 0.0))                 # kriging variance


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@register
class GMDH(Learner):
    """Group Method of Data Handling (combinatorial-multilayer). Each neuron is Ivakhnenko's quadratic polynomial
    of two inputs, fitted on subsample A; neurons are ranked by the *external* regularity criterion (MSE on
    subsample B); the best F survive as inputs to the next layer; layering stops when the best external error stops
    improving. Final model refits the selected structure on all data."""
    name, family, year, field, ref, smooth = "gmdh", "polynomial network", 1968, "cybernetics (USSR)", "ivakhnenko1971polynomial", True

    def __init__(self, F=8, max_layers=6, frac_a=0.6, **kw):
        super().__init__(**kw); self.F, self.max_layers, self.frac_a = F, max_layers, frac_a

    @staticmethod
    def _q(a, b):
        return np.column_stack([np.ones_like(a), a, b, a * b, a * a, b * b])

    def _fit(self, U, z):
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(z)); na = max(int(self.frac_a * len(z)), 7)
        A, B = idx[:na], idx[na:] if len(z) - na >= 3 else idx[:na]
        cur = U.copy()
        self._layers, best_err = [], np.inf
        for _ in range(self.max_layers):
            cands = []
            for i, j in itertools.combinations(range(cur.shape[1]), 2):
                Q = self._q(cur[:, i], cur[:, j])
                c, *_ = np.linalg.lstsq(Q[A], z[A], rcond=None)
                err = np.mean((Q[B] @ c - z[B]) ** 2)
                cands.append((err, i, j))
            if not cands:
                break
            cands.sort()
            keep = cands[: self.F]
            if keep[0][0] >= best_err * (1 - 1e-3):
                break
            best_err = keep[0][0]
            neurons = []
            for err, i, j in keep:                                         # refit on all data
                Q = self._q(cur[:, i], cur[:, j])
                c, *_ = np.linalg.lstsq(Q, z, rcond=None)
                neurons.append((i, j, c))
            self._layers.append(neurons)
            cur = np.column_stack([self._q(cur[:, i], cur[:, j]) @ c for i, j, c in neurons])
        if not self._layers:                                               # d == 1: plain quadratic
            Q = self._q(U[:, 0], U[:, 0] * 0)
            self._c1, *_ = np.linalg.lstsq(Q, z, rcond=None)

    def _predict(self, U):
        if not self._layers:
            return self._q(U[:, 0], U[:, 0] * 0) @ self._c1
        cur = U
        for neurons in self._layers:
            cur = np.column_stack([self._q(cur[:, i], cur[:, j]) @ c for i, j, c in neurons])
        return cur[:, 0]                                                   # best neuron of the last layer


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@register
class IDW(Learner):
    """Shepard's inverse-distance weighting (global, power p) — interpolates, flat spots at data points."""
    name, family, year, field, ref = "idw", "local", 1968, "geography / cartography", "shepard1968two"

    def __init__(self, power=2.0, **kw):
        super().__init__(**kw); self.p = power

    def _fit(self, U, z):
        self._U, self._z = U, z

    def _predict(self, U):
        D = pairwise_dist(U, self._U)
        W = 1.0 / np.maximum(D, 1e-12) ** self.p
        exact = D < 1e-12
        W[exact.any(1)] = exact[exact.any(1)].astype(float)
        return (W @ self._z) / W.sum(1)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
class _RBF(Learner):
    """Dense RBF with a linear polynomial tail. Shape parameter ε and smoothing λ chosen by Rippa's (1999)
    closed-form leave-one-out error e_i = c_i / (M^{-1})_ii over a small grid."""
    family, smooth = "rbf", True
    kernel = "multiquadric"
    max_n = 5000

    def _phi(self, r, eps):
        if self.kernel == "multiquadric":
            return np.sqrt(1 + (eps * r) ** 2)
        if self.kernel == "tps":
            return np.where(r > 0, r**2 * np.log(np.maximum(r, 1e-300)), 0.0)
        if self.kernel == "gaussian":
            return np.exp(-(eps * r) ** 2)
        if self.kernel == "cubic":
            return r**3
        raise ValueError(self.kernel)

    def _system(self, D, eps, lam):
        n, d = self._U.shape
        P = np.column_stack([np.ones(n), self._U])
        M = np.zeros((n + d + 1, n + d + 1))
        M[:n, :n] = self._phi(D, eps) + lam * np.eye(n)
        M[:n, n:] = P; M[n:, :n] = P.T
        return M

    def _fit(self, U, z):
        self._U = U
        n = len(z)
        D = pairwise_dist(U, U)
        h = np.median(np.sort(D + np.eye(n) * 1e9, 1)[:, 0])               # typical spacing
        eps_grid = [None] if self.kernel in ("tps", "cubic") else [0.25 / h, 0.5 / h, 1 / h, 2 / h]
        lam_grid = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
        rhs = np.concatenate([z, np.zeros(U.shape[1] + 1)])
        best = (np.inf, None)
        for eps in eps_grid:
            for lam in lam_grid:
                M = self._system(D, eps, lam)
                try:
                    Minv = np.linalg.inv(M)
                except np.linalg.LinAlgError:
                    continue
                c = Minv @ rhs
                loo = c[:n] / np.diag(Minv)[:n]
                e = np.mean(loo**2)
                if np.isfinite(e) and e < best[0]:
                    best = (e, (eps, lam, c))
        self._eps, self._lam, self._c = best[1]

    def _predict(self, U):
        n = self._U.shape[0]
        A = self._phi(pairwise_dist(U, self._U), self._eps)
        return A @ self._c[:n] + np.column_stack([np.ones(len(U)), U]) @ self._c[n:]


@register
class RBF_MQ(_RBF):
    name, year, field, ref, kernel = "rbf_mq", 1971, "geodesy / topography", "hardy1971multiquadric", "multiquadric"


@register
class RBF_TPS(_RBF):
    name, year, field, ref, kernel = "rbf_tps", 1977, "approximation theory", "duchon1977splines", "tps"


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@register
class PPR(Learner):
    """Projection pursuit regression: s(x) = Σ_j g_j(w_j·x), each g_j a smoothing spline (GCV), each w_j found by
    minimising the stage residual over the unit sphere (multistart), with backfitting passes."""
    name, family, year, field, ref, smooth = "ppr", "ridge functions", 1981, "statistics", "friedman1981projection", True

    def __init__(self, n_terms=3, backfit=1, **kw):
        super().__init__(**kw); self.J, self.backfit = n_terms, backfit

    @staticmethod
    def _smooth(t, r):
        from scipy.interpolate import make_smoothing_spline
        o = np.argsort(t); ts, rs = t[o], r[o]
        keep = np.concatenate([[True], np.diff(ts) > 1e-9])
        ts, rs = ts[keep], rs[keep]
        if len(ts) < 5:
            c = np.polyfit(ts, rs, min(2, len(ts) - 1)); return lambda s: np.polyval(c, s)
        sp = make_smoothing_spline(ts, rs)                                 # GCV-chosen penalty
        lo, hi = ts[0], ts[-1]
        return lambda s: sp(np.clip(s, lo, hi))

    def _stage(self, U, r, rng):
        from scipy.optimize import minimize
        d = U.shape[1]
        def loss(w):                                                       # fast degree-5 smoother for the search
            w = w / (np.linalg.norm(w) + 1e-12); t = (U - .5) @ w
            c = np.polyfit(t, r, min(5, len(t) - 1)); return np.mean((r - np.polyval(c, t)) ** 2)
        starts = [np.eye(d)[k] for k in range(min(d, 3))] + [rng.normal(size=d)]
        best = min((minimize(loss, w0, method="Nelder-Mead", options=dict(maxiter=40 * d, xatol=1e-3, fatol=1e-6)) for w0 in starts),
                   key=lambda res: res.fun)
        def loss_spline(w):                                                # short refinement with the real smoother
            w = w / (np.linalg.norm(w) + 1e-12); t = (U - .5) @ w
            return np.mean((r - self._smooth(t, r)(t)) ** 2)
        ref = minimize(loss_spline, best.x, method="Nelder-Mead", options=dict(maxiter=15 * d, xatol=1e-4))
        w = ref.x / (np.linalg.norm(ref.x) + 1e-12)
        return w, self._smooth((U - .5) @ w, r)

    def _fit(self, U, z):
        rng = np.random.default_rng(self.seed)
        self._terms = []
        r = z.copy()
        for _ in range(self.J):
            w, g = self._stage(U, r, rng); self._terms.append([w, g]); r = r - g((U - .5) @ w)
        for _ in range(self.backfit):
            for k in range(len(self._terms)):
                w, g = self._terms[k]
                partial = z - sum(gg((U - .5) @ ww) for j, (ww, gg) in enumerate(self._terms) if j != k)
                self._terms[k] = list(self._stage(U, partial, rng))

    def _predict(self, U):
        return sum(g((U - .5) @ w) for w, g in self._terms)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@register
class MLS(Learner):
    """Moving least squares: at each query a weighted (Gaussian) local polynomial (quadratic if enough neighbours,
    else linear); radius from the k-th neighbour distance."""
    name, family, year, field, ref, smooth = "mls", "local", 1981, "computer graphics / meshfree", "lancaster1981surfaces", True

    def __init__(self, k=None, **kw):
        super().__init__(**kw); self.k = k

    def _fit(self, U, z):
        self._U, self._z = U, z
        d = U.shape[1]
        self._quad = len(z) >= 3 * (1 + d + d * (d + 1) // 2)
        p = (1 + d + d * (d + 1) // 2) if self._quad else (1 + d)
        self._k = self.k or min(len(z), max(2 * p, 8))

    def _basis(self, V):
        cols = [np.ones(len(V))] + [V[:, i] for i in range(V.shape[1])]
        if self._quad:
            cols += [V[:, i] * V[:, j] for i in range(V.shape[1]) for j in range(i, V.shape[1])]
        return np.column_stack(cols)

    def _predict(self, U):
        D = pairwise_dist(U, self._U)
        out = np.empty(len(U))
        for q in range(len(U)):
            nn = np.argpartition(D[q], self._k - 1)[: self._k]
            h = D[q, nn].max() + 1e-12
            w = np.exp(-(D[q, nn] / (0.5 * h)) ** 2)
            B = self._basis(self._U[nn] - U[q])
            BW = B * w[:, None]
            c = np.linalg.lstsq(BW.T @ B + 1e-10 * np.eye(B.shape[1]), BW.T @ self._z[nn], rcond=None)[0]
            out[q] = c[0]
        return out


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@register
class KNN(Learner):
    name, family, year, field, ref = "knn", "local", 1951, "statistics", "breiman1984classification"

    def _fit(self, U, z):
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.model_selection import GridSearchCV
        ks = [k for k in (1, 2, 3, 5, 8, 12) if k < len(z)]
        self._m = GridSearchCV(KNeighborsRegressor(weights="distance"), {"n_neighbors": ks},
                               cv=min(5, len(z))).fit(U, z)

    def _predict(self, U):
        return self._m.predict(U)


@register
class CART(Learner):
    name, family, year, field, ref = "cart", "tree", 1984, "statistics", "breiman1984classification"

    def _fit(self, U, z):
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import GridSearchCV
        self._m = GridSearchCV(DecisionTreeRegressor(random_state=self.seed),
                               {"min_samples_leaf": [1, 2, 4, 8]}, cv=min(5, len(z))).fit(U, z)

    def _predict(self, U):
        return self._m.predict(U)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@register
class MARS(Learner):
    """Friedman's MARS: forward stepwise addition of reflected hinge pairs max(0, ±(x_v - t)) (and products with
    existing terms up to `max_degree`), knots at data quantiles; backward deletion by generalised cross-validation
    GCV = RSS / (n (1 - C(M)/n)^2), C(M) = M + penalty·(M-1)/2."""
    name, family, year, field, ref = "mars", "adaptive splines", 1991, "statistics", "friedman1991multivariate"

    def __init__(self, max_terms=21, max_degree=2, penalty=3.0, n_knots=12, **kw):
        super().__init__(**kw)
        self.max_terms, self.max_degree, self.penalty, self.n_knots = max_terms, max_degree, penalty, n_knots

    @staticmethod
    def _eval_term(term, U):
        v = np.ones(len(U))
        for (var, t, s) in term:
            v = v * np.maximum(0.0, s * (U[:, var] - t))
        return v

    def _design(self, terms, U):
        return np.column_stack([self._eval_term(t, U) for t in terms])

    def _gcv(self, rss, n, M):
        C = M + self.penalty * (M - 1) / 2
        return rss / (n * max(1 - C / n, 1e-3) ** 2)

    def _fit(self, U, z):
        n, d = U.shape
        knots = [np.unique(np.quantile(U[:, v], np.linspace(0.05, 0.95, self.n_knots))) for v in range(d)]
        terms = [()]
        B = self._design(terms, U)
        while len(terms) + 2 <= self.max_terms:
            best = (np.inf, None)
            for p, parent in enumerate(terms):
                if len(parent) >= self.max_degree:
                    continue
                used = {v for v, _, _ in parent}
                pv = self._eval_term(parent, U)
                for v in range(d):
                    if v in used:
                        continue
                    for t in knots[v]:
                        h1 = pv * np.maximum(0, U[:, v] - t); h2 = pv * np.maximum(0, t - U[:, v])
                        Bc = np.column_stack([B, h1, h2])
                        c, *_ = np.linalg.lstsq(Bc, z, rcond=None)
                        rss = np.sum((z - Bc @ c) ** 2)
                        if rss < best[0]:
                            best = (rss, (parent + ((v, t, 1.0),), parent + ((v, t, -1.0),)))
            if best[1] is None:
                break
            terms += list(best[1]); B = self._design(terms, U)
        # backward pruning by GCV
        cur = list(range(len(terms)))
        def score(idx):
            Bc = B[:, idx]; c, *_ = np.linalg.lstsq(Bc, z, rcond=None)
            return self._gcv(np.sum((z - Bc @ c) ** 2), n, len(idx))
        best_set, best_g = list(cur), score(cur)
        while len(cur) > 1:
            trials = [(score([i for i in cur if i != k]), k) for k in cur if k != 0]
            g, k = min(trials)
            cur = [i for i in cur if i != k]
            if g < best_g:
                best_g, best_set = g, list(cur)
        self._terms = [terms[i] for i in best_set]
        self._c, *_ = np.linalg.lstsq(self._design(self._terms, U), z, rcond=None)

    def _predict(self, U):
        return self._design(self._terms, U) @ self._c


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@register
class Lipschitz(Learner):
    """Central (optimal-recovery) Lipschitz interpolant: s(x) = ½[min_i(y_i + L|x−x_i|) + max_i(y_i − L|x−x_i|)].
    L is the largest observed slope, inflated 20%. The half-width of the bracket is a worst-case error bound,
    reported as the 'uncertainty'."""
    name, family, year, field, ref, uq = "lipschitz", "optimal recovery", 1978, "approximation theory", "beliakov2006interpolation", True

    def _fit(self, U, z):
        D = pairwise_dist(U, U); np.fill_diagonal(D, np.inf)
        self._L = 1.2 * np.max(np.abs(z[:, None] - z[None, :]) / D)
        self._U, self._z = U, z

    def _bounds(self, U):
        D = pairwise_dist(U, self._U) * self._L
        return np.max(self._z - D, axis=1), np.min(self._z + D, axis=1)

    def _predict(self, U):
        lo, hi = self._bounds(U); return 0.5 * (lo + hi)

    def _predict_std(self, U):
        lo, hi = self._bounds(U); return 0.5 * np.maximum(hi - lo, 0)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
@register
class DelaunayLinear(Learner):
    """Piecewise-linear interpolation on the Delaunay triangulation (the `linear` baseline of Poznański 2026,
    App. G), nearest-neighbour fill outside the convex hull."""
    name, family, year, field, ref = "delaunay_linear", "local", 1934, "computational geometry", "poznanski2026towards"
    max_d = 6

    def _fit(self, U, z):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        self._lin = LinearNDInterpolator(U, z); self._near = NearestNDInterpolator(U, z)

    def _predict(self, U):
        out = self._lin(U); bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = self._near(U[bad])
        return out
