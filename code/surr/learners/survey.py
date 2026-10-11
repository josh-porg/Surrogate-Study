"""Survey extension (2026-10-11): method classes the review must cover that E2a/E3 did not yet include.

  gam               additive model, one penalised spline per input (pyGAM)              Hastie & Tibshirani 1986
  pspline_te        tensor-product P-spline, all inputs jointly (d <= 3)                 Eilers & Marx 1996
  symreg            symbolic regression by genetic programming (gplearn)                Koza 1992
  sparse_gp         inducing-point GP, collapsed variational bound (SGPR)               Titsias 2009
  deep_gp           2-layer doubly-stochastic variational deep GP                       Damianou & Lawrence 2013
  bnn_laplace       network + last-layer Laplace (evidence-tuned Bayesian linear head)  MacKay 1992; Kristiadi 2020
  bnn_vi            mean-field variational BNN (Bayes by Backprop)                      Blundell et al. 2015
  bart              Bayesian additive regression trees (stochtree, GFR warm start)      Chipman et al. 2010
  moe               Gaussian-gated mixture of local quadratic experts, EM               Xu, Jordan & Hinton 1995
  natural_neighbor  Sibson natural-neighbour interpolation (Monte-Carlo discrete, d<=3) Sibson 1981; Park 2006
  warped_gp         GP with per-input Kumaraswamy warping learned by ML-II              Snoek et al. 2014
  aaa_rational      AAA barycentric rational approximant (1-D only)                     Nakatsukasa et al. 2018

All follow the package conventions: inputs on [0,1]^d, target standardised by the base class, single-threaded.
"""
from __future__ import annotations

import numpy as np

from .base import Learner, register
from .nets import _mlp, _torch, _train


# ── additive and spline models ───────────────────────────────────────────────────────────────────────────────
@register
class GAM(Learner):
    name, family, year, field, ref, uq, smooth = "gam", "additive / spline", 1986, "statistics", "hastie1986generalized", True, True

    def _fit(self, U, z):
        from pygam import LinearGAM, s
        terms = s(0, n_splines=12)
        for j in range(1, U.shape[1]):
            terms = terms + s(j, n_splines=12)
        self._m = LinearGAM(terms).gridsearch(U, z, lam=np.logspace(-3, 3, 7), progress=False)

    def _predict(self, U):
        return self._m.predict(U)

    def _predict_std(self, U):
        lo, hi = self._m.prediction_intervals(U, width=0.6827).T
        return (hi - lo) / 2


@register
class PSplineTensor(Learner):
    name, family, year, field, ref, uq, smooth = "pspline_te", "additive / spline", 1996, "statistics", "eilers1996flexible", True, True
    max_d = 3

    def _fit(self, U, z):
        from pygam import LinearGAM, te
        d = U.shape[1]
        k = {1: 20, 2: 12, 3: 8}[d]
        self._m = LinearGAM(te(*range(d), n_splines=[k] * d)).gridsearch(U, z, lam=np.logspace(-3, 3, 7),
                                                                          progress=False)

    def _predict(self, U):
        return self._m.predict(U)

    def _predict_std(self, U):
        lo, hi = self._m.prediction_intervals(U, width=0.6827).T
        return (hi - lo) / 2


# ── symbolic regression ──────────────────────────────────────────────────────────────────────────────────────
@register
class SymReg(Learner):
    """Genetic-programming symbolic regression: evolves expression trees; returns a closed-form formula."""
    name, family, year, field, ref, smooth = "symreg", "symbolic regression", 1992, "evolutionary computation", "koza1992genetic", True

    def _fit(self, U, z):
        from gplearn.genetic import SymbolicRegressor
        self._m = SymbolicRegressor(population_size=1000, generations=20, tournament_size=20, stopping_criteria=1e-4,
                                    function_set=("add", "sub", "mul", "div", "sin", "cos", "sqrt", "log", "abs"),
                                    const_range=(-5.0, 5.0), parsimony_coefficient=1e-3, p_crossover=0.7,
                                    p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1,
                                    max_samples=1.0, random_state=self.seed, n_jobs=1).fit(U, z)
        self.formula = str(self._m._program)

    def _predict(self, U):
        return self._m.predict(U)


# ── GP extensions (gpytorch) ─────────────────────────────────────────────────────────────────────────────────
def _fit_gpytorch(model, lik, loss_fn, params, iters=300, lr=0.05):
    torch = _torch()
    model.train(); lik.train()
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(iters):
        opt.zero_grad(); loss = loss_fn(); loss.backward(); opt.step()
    model.eval(); lik.eval()


@register
class SparseGP(Learner):
    name, family, year, field, ref, uq, smooth = "sparse_gp", "gaussian process", 2009, "machine learning", "titsias2009variational", True, True

    def _fit(self, U, z):
        torch = _torch(); import gpytorch as gp
        torch.manual_seed(self.seed)
        X, Y = torch.tensor(U, dtype=torch.float64), torch.tensor(z, dtype=torch.float64)
        m = min(self.kw.get("m", 64), len(z))
        Z = X[np.random.default_rng(self.seed).choice(len(z), m, replace=False)].clone()

        class M(gp.models.ExactGP):
            def __init__(s, x, y, lik):
                super().__init__(x, y, lik)
                s.mean = gp.means.ConstantMean()
                base = gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=2.5, ard_num_dims=x.shape[1]))
                s.cov = gp.kernels.InducingPointKernel(base, inducing_points=Z, likelihood=lik)

            def forward(s, x):
                return gp.distributions.MultivariateNormal(s.mean(x), s.cov(x))

        lik = gp.likelihoods.GaussianLikelihood().double()
        self._m = M(X, Y, lik).double(); self._lik = lik
        mll = gp.mlls.ExactMarginalLogLikelihood(lik, self._m)
        _fit_gpytorch(self._m, lik, lambda: -mll(self._m(X), Y), self._m.parameters())

    def _post(self, U):
        torch = _torch(); import gpytorch as gp
        with torch.no_grad(), gp.settings.fast_pred_var():
            return self._lik(self._m(torch.tensor(U, dtype=torch.float64)))

    def _predict(self, U):
        return self._post(U).mean.numpy()

    def _predict_std(self, U):
        return self._post(U).stddev.numpy()


@register
class WarpedGP(Learner):
    """ARD Matérn-5/2 GP on Kumaraswamy-warped inputs w_j(u) = 1 - (1 - u^a_j)^b_j, warps learned with the kernel."""
    name, family, year, field, ref, uq, smooth = "warped_gp", "gaussian process", 2014, "machine learning", "snoek2014input", True, True
    max_n = 3000

    def _fit(self, U, z):
        torch = _torch(); import gpytorch as gp
        torch.manual_seed(self.seed)
        X, Y = torch.tensor(U, dtype=torch.float64), torch.tensor(z, dtype=torch.float64)
        d = U.shape[1]

        class M(gp.models.ExactGP):
            def __init__(s, x, y, lik):
                super().__init__(x, y, lik)
                s.la = torch.nn.Parameter(torch.zeros(d, dtype=torch.float64))   # log a, log b (identity warp at 0)
                s.lb = torch.nn.Parameter(torch.zeros(d, dtype=torch.float64))
                s.mean = gp.means.ConstantMean()
                s.cov = gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=2.5, ard_num_dims=d))

            def warp(s, x):
                x = x.clamp(1e-6, 1 - 1e-6)
                return 1 - (1 - x ** s.la.exp()) ** s.lb.exp()

            def forward(s, x):
                w = s.warp(x)
                return gp.distributions.MultivariateNormal(s.mean(w), s.cov(w))

        lik = gp.likelihoods.GaussianLikelihood().double()
        self._m = M(X, Y, lik).double(); self._lik = lik
        mll = gp.mlls.ExactMarginalLogLikelihood(lik, self._m)
        _fit_gpytorch(self._m, lik, lambda: -mll(self._m(X), Y), self._m.parameters())

    def _post(self, U):
        torch = _torch(); import gpytorch as gp
        with torch.no_grad(), gp.settings.fast_pred_var():
            return self._lik(self._m(torch.tensor(U, dtype=torch.float64)))

    def _predict(self, U):
        return self._post(U).mean.numpy()

    def _predict_std(self, U):
        return self._post(U).stddev.numpy()


@register
class DeepGP(Learner):
    """Two-layer deep GP (hidden width min(d, 3)), doubly-stochastic variational inference (Salimbeni & Deisenroth
    2017), 32 inducing points per layer, Matérn-5/2 ARD kernels."""
    name, family, year, field, ref, uq, smooth = "deep_gp", "gaussian process", 2013, "machine learning", "damianou2013deep", True, True
    max_n = 3000

    def _fit(self, U, z):
        torch = _torch(); import gpytorch as gp
        from gpytorch.models.deep_gps import DeepGP as _DGP, DeepGPLayer
        torch.manual_seed(self.seed)
        X, Y = torch.tensor(U, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)
        d, h = U.shape[1], min(U.shape[1], 3)

        class Layer(DeepGPLayer):
            def __init__(s, din, dout):
                shape = torch.Size([dout]) if dout else torch.Size([])
                Z = torch.rand(*(shape + torch.Size([32, din])))
                vd = gp.variational.CholeskyVariationalDistribution(32, batch_shape=shape)
                vs = gp.variational.VariationalStrategy(s, Z, vd, learn_inducing_locations=True)
                super().__init__(vs, din, dout)
                s.mean = gp.means.LinearMean(din, batch_shape=shape) if dout else gp.means.ConstantMean()
                s.cov = gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=2.5, batch_shape=shape, ard_num_dims=din),
                                               batch_shape=shape)

            def forward(s, x):
                return gp.distributions.MultivariateNormal(s.mean(x), s.cov(x))

        class M(_DGP):
            def __init__(s):
                super().__init__()
                s.l1, s.l2 = Layer(d, h), Layer(h, None)
                s.lik = gp.likelihoods.GaussianLikelihood()

            def forward(s, x):
                return s.l2(s.l1(x))

        self._m = M()
        mll = gp.mlls.DeepApproximateMLL(gp.mlls.VariationalELBO(self._m.lik, self._m, len(z)))
        with gp.settings.num_likelihood_samples(8):
            _fit_gpytorch(self._m, self._m.lik, lambda: -mll(self._m(X), Y), self._m.parameters(), iters=400, lr=0.02)

    def _post(self, U):
        torch = _torch(); import gpytorch as gp
        with torch.no_grad(), gp.settings.num_likelihood_samples(32), torch.random.fork_rng():
            torch.manual_seed(self.seed + 17)                    # common random numbers: repeatable predictions
            o = self._m.lik(self._m(torch.tensor(U, dtype=torch.float32)))
            mu, var = o.mean, o.variance                          # (samples, n)
            m = mu.mean(0)
            return m.numpy().astype(float), (var.mean(0) + mu.var(0)).sqrt().numpy().astype(float)

    def _predict(self, U):
        return self._post(U)[0]

    def _predict_std(self, U):
        return self._post(U)[1]


# ── Bayesian neural networks ─────────────────────────────────────────────────────────────────────────────────
@register
class BNNLaplace(Learner):
    """MLP trained with L-BFGS (as mlp_ens); the trained (MAP) network is the predictive mean, and a Gaussian
    (Laplace) posterior over its last layer gives the uncertainty, with prior and noise precisions set by evidence
    maximisation (MacKay's fixed point, with the MAP weights and residuals held fixed) — last-layer Laplace."""
    name, family, year, field, ref, uq, smooth = "bnn_laplace", "neural network", 1992, "machine learning", "mackay1992practical", True, True

    def _fit(self, U, z):
        torch = _torch(); torch.manual_seed(self.seed)
        self._net = _train(_mlp(U.shape[1]), U, z, self.seed, opt="lbfgs")
        P = self._phi(U); n, k = P.shape
        last = self._net[-1]
        w = np.r_[last.weight.detach().numpy().ravel(), last.bias.detach().numpy()].astype(float)
        sse = max(float(np.sum((z - self._predict(U)) ** 2)), 1e-12)
        lam0 = np.clip(np.linalg.eigvalsh(P.T @ P), 0, None)
        a, b = 1.0, n / sse
        for _ in range(100):                                       # MacKay evidence fixed point for (alpha, beta)
            g = np.sum(b * lam0 / (a + b * lam0))
            a_new, b_new = g / max(w @ w, 1e-12), max(n - g, 1.0) / sse
            done = abs(a_new - a) / a < 1e-8 and abs(b_new - b) / b < 1e-8
            a, b = a_new, b_new
            if done:
                break
        self._S = np.linalg.inv(a * np.eye(k) + b * P.T @ P); self._b = b

    def _phi(self, U):
        torch = _torch()
        with torch.no_grad():
            h = self._net[:-1](torch.tensor(U, dtype=torch.float32)).numpy().astype(float)
        return np.column_stack([h, np.ones(len(h))])

    def _predict(self, U):
        torch = _torch()
        with torch.no_grad():
            return self._net(torch.tensor(U, dtype=torch.float32)).squeeze(-1).numpy().astype(float)

    def _predict_std(self, U):
        P = self._phi(U)
        return np.sqrt(1.0 / self._b + np.einsum("ij,jk,ik->i", P, self._S, P))


@register
class BNNVI(Learner):
    """Mean-field Gaussian variational BNN (Bayes by Backprop with the local reparameterisation trick), 2 x 64 SiLU,
    N(0, 1) prior, learned homoscedastic noise; full-batch Adam with KL annealing over the first third."""
    name, family, year, field, ref, uq, smooth = "bnn_vi", "neural network", 2015, "machine learning", "blundell2015weight", True, True

    def _fit(self, U, z):
        torch = _torch(); torch.manual_seed(self.seed); nn = torch.nn
        d = U.shape[1]

        class BLin(nn.Module):
            def __init__(s, i, o):
                super().__init__()
                s.mu = nn.Parameter(torch.randn(i, o) * (1 / np.sqrt(i))); s.rho = nn.Parameter(torch.full((i, o), -5.0))
                s.bmu = nn.Parameter(torch.zeros(o)); s.brho = nn.Parameter(torch.full((o,), -5.0))

            def forward(s, x, whole=False):
                sw, sb = nn.functional.softplus(s.rho), nn.functional.softplus(s.brho)
                if whole:                                            # one weight draw for the whole batch
                    return x @ (s.mu + sw * torch.randn_like(sw)) + s.bmu + sb * torch.randn_like(sb)
                m = x @ s.mu + s.bmu; v = (x ** 2) @ sw ** 2 + sb ** 2      # local reparameterisation (training)
                return m + v.sqrt() * torch.randn_like(m)

            def kl(s):
                out = 0.0
                for mu, rho in ((s.mu, s.rho), (s.bmu, s.brho)):
                    sg = nn.functional.softplus(rho)
                    out = out + (0.5 * ((sg ** 2 + mu ** 2) / ps ** 2 - 1) - torch.log(sg / ps)).sum()
                return out

        ps = float(self.kw.get("prior_sd", 1.0))
        self._layers = nn.ModuleList([BLin(d, 64), BLin(64, 64), BLin(64, 1)])
        self._lognoise = nn.Parameter(torch.tensor(np.log(0.1)))
        X, Y = torch.tensor(U, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)
        opt = torch.optim.Adam(list(self._layers.parameters()) + [self._lognoise], lr=float(self.kw.get("lr", 3e-3)))
        E = int(self.kw.get("epochs", 3000))
        for e in range(E):
            opt.zero_grad()
            f = self._forward(X); s2 = torch.exp(2 * self._lognoise)
            nll = 0.5 * ((Y - f) ** 2 / s2 + torch.log(s2)).sum()
            kl = sum(l.kl() for l in self._layers)
            loss = (nll + float(self.kw.get("kl_weight", 1.0)) * min(1.0, 3 * e / E) * kl) / len(z)
            loss.backward(); opt.step()

    def _forward(self, X, whole=False):
        torch = _torch()
        h = X
        for l in self._layers[:-1]:
            h = torch.nn.functional.silu(l(h, whole))
        return self._layers[-1](h, whole).squeeze(-1)

    def _samples(self, U, T=50):
        torch = _torch()
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(self.seed + 31)                       # common random numbers: repeatable predictions
            X = torch.tensor(U, dtype=torch.float32)
            return np.vstack([self._forward(X, whole=True).numpy() for _ in range(T)]).astype(float)

    def _predict(self, U):
        return self._samples(U).mean(0)

    def _predict_std(self, U):
        S = self._samples(U)
        return np.sqrt(S.var(0) + float(np.exp(2 * self._lognoise.detach().numpy())))


# ── Bayesian trees ───────────────────────────────────────────────────────────────────────────────────────────
@register
class BART(Learner):
    name, family, year, field, ref, uq = "bart", "tree ensemble", 2010, "Bayesian statistics", "chipman2010bart", True

    def _fit(self, U, z):
        from stochtree import BARTModel
        self._m = BARTModel()
        self._m.sample(X_train=U, y_train=z, num_gfr=10, num_burnin=0, num_mcmc=100,
                       general_params={"random_seed": int(self.seed), "num_threads": 1})

    def _draws(self, U):
        return self._m.predict(X=U)["y_hat"]                       # (n, draws)

    def _predict(self, U):
        return self._draws(U).mean(1)

    def _predict_std(self, U):
        return self._draws(U).std(1)


# ── mixture of experts ───────────────────────────────────────────────────────────────────────────────────────
def _quad_features(U):
    d = U.shape[1]
    cols = [np.ones(len(U)), *U.T]
    if d <= 5:
        cols += [U[:, i] * U[:, j] for i in range(d) for j in range(i, d)]
    return np.column_stack(cols)


@register
class MoE(Learner):
    """Gaussian-gated mixture of experts (Xu, Jordan & Hinton 1995): p(x, y) = Σ_k π_k N(x | μ_k, Σ_k)
    N(y | w_kᵀφ(x), σ_k²) with diagonal Σ_k and quadratic φ (linear for d > 5), fitted by EM from a k-means start;
    K ∈ {2, 4, 8} chosen by 3-fold CV. Gate g_k(x) ∝ π_k N(x | μ_k, Σ_k)."""
    name, family, year, field, ref, uq, smooth = "moe", "mixture of experts", 1995, "machine learning / neural computation", "xu1995alternative", True, True

    def _fit(self, U, z):
        ks = [k for k in (2, 4, 8) if len(z) >= 6 * k * _quad_features(U[:1]).shape[1] // 4] or [2]
        if len(ks) > 1:
            from sklearn.model_selection import KFold
            err = {}
            for K in ks:
                e = 0.0
                for tr, te in KFold(3, shuffle=True, random_state=self.seed).split(U):
                    p = self._em(U[tr], z[tr], K); e += np.sum((self._mean(p, U[te]) - z[te]) ** 2)
                err[K] = e
            K = min(err, key=err.get)
        else:
            K = ks[0]
        self._p = self._em(U, z, K)

    def _em(self, U, z, K, iters=100):
        from sklearn.cluster import KMeans
        n, d = U.shape; F = _quad_features(U); q = F.shape[1]
        lab = KMeans(K, n_init=3, random_state=self.seed).fit_predict(U)
        R = np.eye(K)[lab] * 0.9 + 0.1 / K
        reg = 1e-3
        for _ in range(iters):
            Nk = R.sum(0) + 1e-9; pi = Nk / n
            mu = (R.T @ U) / Nk[:, None]
            var = np.stack([(R[:, k:k + 1] * (U - mu[k]) ** 2).sum(0) / Nk[k] for k in range(K)]) + 1e-3
            W = np.stack([np.linalg.solve(F.T @ (R[:, k:k + 1] * F) + reg * np.eye(q), F.T @ (R[:, k] * z))
                          for k in range(K)])
            s2 = np.array([(R[:, k] * (z - F @ W[k]) ** 2).sum() / Nk[k] for k in range(K)]) + 1e-6
            lg = (np.log(pi) - 0.5 * np.log(var).sum(1))[None, :] - 0.5 * (((U[:, None, :] - mu) ** 2) / var).sum(2)
            ly = -0.5 * np.log(s2)[None, :] - 0.5 * (z[:, None] - F @ W.T) ** 2 / s2
            L = lg + ly; L -= L.max(1, keepdims=True); Rn = np.exp(L); Rn /= Rn.sum(1, keepdims=True)
            if np.abs(Rn - R).max() < 1e-6:
                R = Rn; break
            R = Rn
        return dict(pi=pi, mu=mu, var=var, W=W, s2=s2)

    @staticmethod
    def _gate(p, U):
        lg = (np.log(p["pi"]) - 0.5 * np.log(p["var"]).sum(1))[None, :] \
             - 0.5 * (((U[:, None, :] - p["mu"]) ** 2) / p["var"]).sum(2)
        lg -= lg.max(1, keepdims=True); G = np.exp(lg)
        return G / G.sum(1, keepdims=True)

    def _mean(self, p, U):
        return (self._gate(p, U) * (_quad_features(U) @ p["W"].T)).sum(1)

    def _predict(self, U):
        return self._mean(self._p, U)

    def _predict_std(self, U):
        G = self._gate(self._p, U); M = _quad_features(U) @ self._p["W"].T
        m = (G * M).sum(1)
        return np.sqrt(np.maximum((G * (self._p["s2"] + M ** 2)).sum(1) - m ** 2, 0))


# ── natural-neighbour interpolation ──────────────────────────────────────────────────────────────────────────
@register
class NaturalNeighbor(Learner):
    """Sibson interpolation: weight_i = share of the query's would-be Voronoi cell taken from data point i. Volumes
    by Monte Carlo on a fixed offset set (common random numbers, so predictions are deterministic) inside a ball of
    radius 2 × the distance to the (2d+2)-th nearest data point, clipped to the unit box (discrete Sibson, Park et al.
    2006). Exact interpolation, C¹ away from the data; cost grows quickly with d, hence d ≤ 3."""
    name, family, year, field, ref = "natural_neighbor", "local / geometric", 1981, "computational geometry", "sibson1981brief"
    max_d = 3

    def _fit(self, U, z):
        from scipy.spatial import cKDTree
        self._U, self._z, self._tree = U, z, cKDTree(U)
        d = U.shape[1]; M = {1: 400, 2: 1500, 3: 3000}[d]
        r = np.random.default_rng(12345)
        v = r.normal(size=(M, d)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        self._off = v * r.random(M)[:, None] ** (1 / d)              # uniform in the unit ball

    def _predict(self, U):
        k = min(2 * U.shape[1] + 2, len(self._z))
        dk, ik = self._tree.query(U, k=k)
        out = np.empty(len(U))
        for t, x in enumerate(U):
            if dk[t, 0] < 1e-12:
                out[t] = self._z[ik[t, 0]]; continue
            P = x + 2 * dk[t, -1] * self._off
            P = P[np.all((P >= 0) & (P <= 1), axis=1)]
            dp, ip = self._tree.query(P)
            own = np.linalg.norm(P - x, axis=1) < dp
            if not own.any():
                out[t] = self._z[ik[t, 0]]; continue
            w = np.bincount(ip[own], minlength=len(self._z)).astype(float)
            out[t] = w @ self._z / w.sum()
        return out


# ── rational approximation ───────────────────────────────────────────────────────────────────────────────────
@register
class AAARational(Learner):
    """AAA (adaptive Antoulas–Anderson) barycentric rational approximant; greedy support-point selection with a
    least-squares weight fit (SVD). Defined for 1-D inputs only — used on 1-D and multi-fidelity problems."""
    name, family, year, field, ref, smooth = "aaa_rational", "rational", 2018, "numerical analysis", "nakatsukasa2018aaa", True
    max_d = 1

    def _fit(self, U, z, tol=1e-10, mmax=None):
        x = U[:, 0]; mmax = mmax or min(len(x) // 2, 60)
        J = np.ones(len(x), bool); zs, fs = [], []; R = np.full(len(x), z.mean())
        for _ in range(mmax):
            j = np.argmax(np.abs(z - R) * J); zs.append(x[j]); fs.append(z[j]); J[j] = False
            C = 1.0 / (x[J][:, None] - np.array(zs)[None, :])
            A = (z[J][:, None] - np.array(fs)[None, :]) * C
            w = np.linalg.svd(A, full_matrices=False)[2][-1].conj()
            R = z.copy(); R[J] = (C @ (w * fs)) / (C @ w)
            if np.max(np.abs(z - R)) <= tol * np.max(np.abs(z)):
                break
        self._zs, self._fs, self._w = np.array(zs), np.array(fs), w

    def _predict(self, U):
        x = U[:, 0]; D = x[:, None] - self._zs[None, :]
        hit = np.isclose(D, 0)
        D[hit] = 1.0
        C = 1.0 / D
        r = (C @ (self._w * self._fs)) / (C @ self._w)
        rows, cols = np.nonzero(hit)
        r[rows] = self._fs[cols]
        return r
