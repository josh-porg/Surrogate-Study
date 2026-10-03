"""Neural-network learners (PyTorch, CPU).

  mlp_ens     deep ensemble of K MLPs (random init + bootstrap)        Lakshminarayanan et al. 2017
  mc_dropout  single MLP, dropout kept on at prediction (T passes)     Gal & Ghahramani 2016
  kan         Kolmogorov–Arnold network with B-spline edge functions   Liu et al. 2024 (own minimal implementation)
  dkl         deep kernel learning: MLP feature map + exact GP (ML-II) Wilson et al. 2016 (gpytorch)
  tabpfn      TabPFN v2 in-context regressor                           Hollmann et al. 2025
"""
from __future__ import annotations

import numpy as np

from .base import Learner, register


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


class _BB:
    """Full-batch gradient descent with Barzilai–Borwein (BB2) step sizes (Barzilai & Borwein 1988), the method of
    the author's Math 796 HW2; first step uses lr."""

    def __init__(self, params, lr):
        self.p, self.lr, self.prev = list(params), lr, None

    def zero_grad(self):
        for q in self.p:
            q.grad = None

    def step(self):
        torch = _torch()
        x = torch.cat([q.detach().reshape(-1) for q in self.p]); g = torch.cat([q.grad.reshape(-1) for q in self.p])
        a = self.lr
        if self.prev is not None:
            dx, dg = x - self.prev[0], g - self.prev[1]
            a = float(torch.clamp(torch.abs(dx @ dg) / torch.clamp(dg @ dg, min=1e-16), 1e-6, 1.0))
        self.prev = (x.clone(), g.clone())
        with torch.no_grad():
            for q in self.p:
                q -= a * q.grad


def _optimizer(name, params, lr, wd):
    torch = _torch()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    if name == "sgd":
        return torch.optim.SGD(params, lr=10 * lr, momentum=0.9, weight_decay=wd)
    if name == "lbfgs":
        return torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
    if name == "bb":
        return _BB(params, lr)
    raise ValueError(name)


def _train(net, X, y, seed, epochs=3000, lr=3e-3, wd=1e-5, patience=200, val_frac=0.15, opt="adam"):
    """Full-batch training with early stopping on a held-out split (no split when n < 20). `opt` ∈ {adam, adamw,
    sgd, lbfgs, bb} — the training-optimizer factor screened in E2 (L-BFGS often wins at small full-batch n).
    L-BFGS takes up to 20 inner iterations per 'epoch', so its epoch budget is divided by 20."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    n = len(y); idx = rng.permutation(n)
    nv = int(val_frac * n) if n >= 20 else 0
    va, tr = idx[:nv], idx[nv:]
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    optim = _optimizer(opt, net.parameters(), lr, wd)
    if opt == "lbfgs":
        epochs, patience = max(epochs // 20, 10), max(patience // 20, 5)
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        def closure():
            optim.zero_grad()
            loss = torch.mean((net(Xt[tr]).squeeze(-1) - yt[tr]) ** 2)
            loss.backward(); return loss
        if opt == "lbfgs":
            optim.step(closure)
        else:
            closure(); optim.step()
        if nv:
            net.eval()
            with torch.no_grad():
                v = torch.mean((net(Xt[va]).squeeze(-1) - yt[va]) ** 2).item()
            if v < best - 1e-7:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait > patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


def _mlp(d, width=64, depth=3, p_drop=0.0, act="silu"):
    torch = _torch(); nn = torch.nn
    A = {"silu": nn.SiLU, "tanh": nn.Tanh}[act]
    layers, w = [], d
    for _ in range(depth):
        layers += [nn.Linear(w, width), A()] + ([nn.Dropout(p_drop)] if p_drop else []); w = width
    layers += [nn.Linear(w, 1)]
    return nn.Sequential(*layers)


@register
class MLPEnsemble(Learner):
    name, family, year, field, ref, uq, smooth = "mlp_ens", "neural network", 2017, "machine learning", "lakshminarayanan2017simple", True, True

    def __init__(self, K=5, **kw):
        super().__init__(**kw); self.K = K

    def _fit(self, U, z):
        torch = _torch(); rng = np.random.default_rng(self.seed); self._nets = []
        for k in range(self.K):
            torch.manual_seed(self.seed * 100 + k)
            idx = rng.integers(0, len(z), len(z)) if self.K > 1 else np.arange(len(z))
            self._nets.append(_train(_mlp(U.shape[1]), U[idx], z[idx], self.seed + k, opt=self.kw.get("opt", "adam")))

    def _all(self, U):
        torch = _torch()
        with torch.no_grad():
            X = torch.tensor(U, dtype=torch.float32)
            return np.vstack([n.eval()(X).squeeze(-1).numpy() for n in self._nets])

    def _predict(self, U):
        return self._all(U).mean(0)

    def _predict_std(self, U):
        return self._all(U).std(0)


@register
class MCDropout(Learner):
    name, family, year, field, ref, uq, smooth = "mc_dropout", "neural network", 2016, "machine learning", "gal2016dropout", True, True

    def _fit(self, U, z):
        torch = _torch(); torch.manual_seed(self.seed)
        self._net = _train(_mlp(U.shape[1], p_drop=0.1), U, z, self.seed, opt=self.kw.get("opt", "adam"))

    def _passes(self, U, T=50):
        torch = _torch(); self._net.train()
        with torch.no_grad():
            X = torch.tensor(U, dtype=torch.float32)
            return np.vstack([self._net(X).squeeze(-1).numpy() for _ in range(T)])

    def _predict(self, U):
        return self._passes(U).mean(0)

    def _predict_std(self, U):
        return self._passes(U).std(0)


class _KANLayer:
    """One KAN layer: every edge (i -> j) carries a learnable cubic B-spline on a uniform grid over [-1, 1] plus a
    SiLU residual, phi(x) = w_b * silu(x) + Σ_g c_g B_g(x)."""

    def __init__(self, din, dout, G=8, k=3, seed=0):
        torch = _torch(); g = torch.Generator().manual_seed(seed)
        h = 2.0 / G
        self.grid = torch.arange(-k, G + k + 1, dtype=torch.float32) * h - 1.0
        self.k = k
        self.coef = torch.nn.Parameter(torch.randn(din, dout, G + k, generator=g) * 0.1)
        self.wb = torch.nn.Parameter(torch.randn(din, dout, generator=g) / np.sqrt(din))

    def basis(self, x):                                                    # Cox–de Boor, x: (n, din)
        torch = _torch(); t = self.grid; x = x.unsqueeze(-1)
        B = ((x >= t[:-1]) & (x < t[1:])).float()
        for p in range(1, self.k + 1):
            B = ((x - t[: -(p + 1)]) / (t[p:-1] - t[: -(p + 1)]) * B[..., :-1]
                 + (t[p + 1:] - x) / (t[p + 1:] - t[1:-p]) * B[..., 1:])
        return B                                                           # (n, din, G+k)

    def __call__(self, x, squash=True):
        torch = _torch()
        if squash:
            x = torch.tanh(x)                                              # keep hidden activations on the grid
        spl = torch.einsum("nig,iog->no", self.basis(x), self.coef)
        return spl + torch.nn.functional.silu(x) @ self.wb

    def parameters(self):
        return [self.coef, self.wb]


class _KAN:
    def __init__(self, widths, seed):
        torch = _torch()
        self.layers = [_KANLayer(a, b, seed=seed + i) for i, (a, b) in enumerate(zip(widths[:-1], widths[1:]))]
        self._mod = torch.nn.Module()
        for i, L in enumerate(self.layers):
            self._mod.register_parameter(f"c{i}", L.coef); self._mod.register_parameter(f"w{i}", L.wb)

    def __call__(self, x):
        for i, L in enumerate(self.layers):
            x = L(x, squash=i > 0)                                         # inputs already in [-1, 1]
        return x

    def parameters(self):
        return self._mod.parameters()

    def train(self): ...
    def eval(self):
        return self

    def state_dict(self):
        return self._mod.state_dict()

    def load_state_dict(self, s):
        self._mod.load_state_dict(s)


@register
class KAN(Learner):
    name, family, year, field, ref, smooth = "kan", "neural network (spline edges)", 2024, "machine learning for science", "liu2024kan", True

    def _fit(self, U, z):
        torch = _torch(); torch.manual_seed(self.seed)
        self._net = _train(_KAN([U.shape[1], 2 * U.shape[1] + 1, 1], self.seed), 2 * U - 1, z, self.seed,
                           epochs=2000, lr=1e-2)

    def _predict(self, U):
        torch = _torch()
        with torch.no_grad():
            return self._net(torch.tensor(2 * U - 1, dtype=torch.float32)).squeeze(-1).numpy()


@register
class DKL(Learner):
    """Deep kernel learning, ML-II (the variant Ober et al. 2021 show can over-fit): feature map MLP d->32->16->2,
    RBF kernel on the 2-D features, exact GP marginal likelihood, full batch."""
    name, family, year, field, ref, uq, smooth = "dkl", "gaussian process + network", 2016, "machine learning", "wilson2016deep", True, True
    max_n = 5000

    def _fit(self, U, z):
        torch = _torch(); import gpytorch
        torch.manual_seed(self.seed)
        X, y = torch.tensor(U, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)
        fe = torch.nn.Sequential(torch.nn.Linear(U.shape[1], 32), torch.nn.SiLU(), torch.nn.Linear(32, 16),
                                 torch.nn.SiLU(), torch.nn.Linear(16, 2))

        class M(gpytorch.models.ExactGP):
            def __init__(s, X, y, lik):
                super().__init__(X, y, lik); s.fe = fe
                s.mean = gpytorch.means.ConstantMean()
                s.cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            def forward(s, x):
                h = s.fe(x); return gpytorch.distributions.MultivariateNormal(s.mean(h), s.cov(h))

        lik = gpytorch.likelihoods.GaussianLikelihood()
        self._m = M(X, y, lik); self._lik = lik
        self._m.train(); lik.train()
        opt = torch.optim.Adam(self._m.parameters(), lr=1e-2)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, self._m)
        for _ in range(400):
            opt.zero_grad(); loss = -mll(self._m(X), y); loss.backward(); opt.step()
        self._m.eval(); lik.eval()

    def _dist(self, U):
        torch = _torch(); import gpytorch
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self._lik(self._m(torch.tensor(U, dtype=torch.float32)))

    def _predict(self, U):
        return self._dist(U).mean.numpy()

    def _predict_std(self, U):
        return self._dist(U).stddev.numpy()


@register
class TabPFN(Learner):
    name, family, year, field, ref, uq = "tabpfn", "foundation model (in-context)", 2025, "machine learning (tabular)", "hollmann2025tabpfn", True
    max_n = 10000

    def _fit(self, U, z):
        import os
        if not (os.environ.get("TABPFN_TOKEN") or os.environ.get("TABPFN_MODEL_CACHE_DIR")):
            raise RuntimeError("TabPFN v2 weights need the Prior Labs licence: accept it at ux.priorlabs.ai and "
                               "set TABPFN_TOKEN (author action) — skipped")
        from tabpfn import TabPFNRegressor
        self._m = TabPFNRegressor(device="cpu", random_state=self.seed).fit(U, z)

    def _predict(self, U):
        return self._m.predict(U)
