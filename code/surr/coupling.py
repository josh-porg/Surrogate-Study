"""E15 — the surrogate–optimizer interface: different ways an optimizer can CONSUME one fitted surrogate.
Each strategy gets the fitted learner m, the training data and a budget of surrogate evaluations, and returns the
point it would recommend. Evaluations of the mean, of a member and of a smoothed estimate all count against the
budget (gradients from the learner's own grad() are free, as in E9's "model" arm).

  Presentation (what the optimizer sees)
    mean_lbfgsb        L-BFGS-B multistart on the mean, learner gradient (E9 baseline)
    mean_cmaes         CMA-ES on the mean (E9 baseline)
    thompson_members   one random ensemble member per restart, L-BFGS-B on that member; the restart optima are
                       ranked by the mean (Thompson sampling as a fit-once optimizer)        Russo et al. 2018
    member_sgd         Adam with the gradient of one random member per step (model uncertainty as gradient noise)
    smoothed_adam      Adam on the Gaussian-smoothed mean, gradient from random perturbations (works on trees),
                       σ annealed 0.1 → 0.01                                                 Nesterov & Spokoiny 2017
    langevin           Adam with the mean gradient plus annealed Gaussian noise (the control for member_sgd)
                                                                                         Raginsky et al. 2017
    cbo                consensus-based optimization on the mean (particles, weighted consensus, exploration noise)
                                                                                         Pinnau et al. 2017
  Exact structure-exploiting optimization
    milp_trees         global minimum of a tree ensemble (RF / XGBoost / LightGBM) as a mixed-integer linear
                       program (HiGHS via scipy.optimize.milp, time-limited)                Mišić 2020
    milp_relu          global minimum of a ReLU-network ensemble (mlp_relu) as a big-M MILP  Fischetti & Jo 2018
    gp_bnb             certified global minimum of a GP mean by spatial branch-and-bound with interval bounds on the
                       kernel terms (records the certified gap)           Franey et al. 2011; cf. Schweidtmann 2021
  Exploitation and validity
    guard_distance     L-BFGS-B on the mean + a penalty for leaving the data (distance to the nearest training point
                       beyond the design's 75th-percentile nearest-neighbour distance)    Schweidtmann et al. 2022;
                                                                                         Thebelt et al. 2021
    pessimistic        L-BFGS-B on μ + σ (risk-averse: guards against the optimizer's curse) Smith & Winkler 2006
Every strategy's result is also scored for the optimizer's curse: surrogate value minus true value at the pick.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .optimizers import _starts


class _Budget:
    def __init__(self, n):
        self.n, self.used = n, 0

    def take(self, k):
        if self.used + k > self.n:
            raise StopIteration
        self.used += k


def _lbfgsb(f, g, x0, budget, maxiter=200):
    """L-BFGS-B from x0 counting f evaluations; g = gradient callable or None (finite differences, counted)."""
    d = len(x0)

    def fun(x):
        budget.take(1)
        return float(f(np.clip(x, 0, 1)[None])[0])

    def jac(x):
        if g is not None:
            return g(np.clip(x, 0, 1)[None])[0]
        budget.take(2 * d)
        e = np.eye(d) * 1e-5
        return np.array([(f(np.clip(x + e[j], 0, 1)[None])[0] - f(np.clip(x - e[j], 0, 1)[None])[0]) / 2e-5
                         for j in range(d)])
    best = [np.inf, x0]
    try:
        r = minimize(lambda x: fun(x), x0, jac=jac, method="L-BFGS-B", bounds=[(0, 1)] * d,
                     options=dict(maxiter=maxiter), callback=None)
        best = [r.fun, r.x]
    except StopIteration:
        pass
    return np.clip(best[1], 0, 1)


def _multistart(f, g, d, budget, seed, k=None):
    B = _Budget(budget)
    k = k or max(1, min(8, budget // (30 * d)))
    cands = []
    for x0 in _starts(d, k, seed):
        if B.used >= budget:
            break
        cands.append(_lbfgsb(f, g, x0, B))
    cands = np.array(cands)
    return cands[np.argmin(f(cands))]


def mean_lbfgsb(m, U, y, d, budget, seed, **kw):
    return _multistart(m.predict, m.grad, d, budget, seed)


def mean_cmaes(m, U, y, d, budget, seed, **kw):
    from . import optimizers as O
    return O.run("cmaes", m.predict, d, budget=budget, seed=seed)["x"]


def thompson_members(m, U, y, d, budget, seed, mem=None, **kw):
    if not mem:
        return None
    rng = np.random.default_rng(seed)
    w = np.array([a for a, _ in mem]); w = w / w.sum()
    k = max(2, min(8, budget // (30 * d)))
    per = budget // k
    cands = []
    for x0 in _starts(d, k, seed):
        f = mem[int(rng.choice(len(mem), p=w))][1]
        cands.append(_lbfgsb(f, None, x0, _Budget(per)))
    cands = np.array(cands)
    return cands[np.argmin(m.predict(cands))]


def _adam_loop(gradient, f, d, budget, seed, lr=0.02, steps_per_start=None, cost_per_step=1):
    """Projected Adam, multistarted from the shared Sobol' starts; `gradient(x, t, rng)` returns a direction;
    each step costs `cost_per_step` surrogate evaluations; the incumbent is the best mean value seen."""
    rng = np.random.default_rng(seed)
    k = max(1, min(4, budget // (100 * max(1, cost_per_step))))
    B = _Budget(budget)
    best = (np.inf, None)
    try:
        for x0 in _starts(d, k, seed):
            x, mo, v = x0.copy(), np.zeros(d), np.zeros(d)
            steps = (budget // k) // (cost_per_step + 1)
            for t in range(1, steps + 1):
                B.take(1)
                fx = float(f(x[None])[0])
                if fx < best[0]:
                    best = (fx, x.copy())
                B.take(cost_per_step)
                gr = gradient(x, t / steps, rng)
                mo = 0.9 * mo + 0.1 * gr; v = 0.999 * v + 0.001 * gr * gr
                x = np.clip(x - lr * (mo / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8), 0, 1)
    except StopIteration:
        pass
    return best[1] if best[1] is not None else np.full(d, 0.5)


def _scale(m, d, seed):
    return float(np.std(m.predict(_starts(d, 4 * d + 4, seed + 11)))) + 1e-12


def member_sgd(m, U, y, d, budget, seed, mem=None, **kw):
    if not mem:
        return None
    w = np.array([a for a, _ in mem]); w = w / w.sum()
    sc = _scale(m, d, seed); e = np.eye(d) * 1e-4

    def gr(x, frac, rng):
        f = mem[int(rng.choice(len(mem), p=w))][1]
        return np.array([(f((x + e[j])[None])[0] - f((x - e[j])[None])[0]) / 2e-4 for j in range(d)]) / sc
    return _adam_loop(gr, m.predict, d, budget, seed, cost_per_step=2 * d)


def smoothed_adam(m, U, y, d, budget, seed, **kw):
    sc = _scale(m, d, seed)

    def gr(x, frac, rng):
        sig = 0.1 * (0.1 ** frac)                                         # 0.1 → 0.01
        Z = rng.normal(size=(d, d))
        fx = m.predict(x[None])[0]
        vals = m.predict(np.clip(x + sig * Z, 0, 1))
        return ((vals - fx)[:, None] * Z).mean(0) / sig / sc
    return _adam_loop(gr, m.predict, d, budget, seed, cost_per_step=d + 1)


def langevin(m, U, y, d, budget, seed, **kw):
    sc = _scale(m, d, seed)

    def gr(x, frac, rng):
        T = 1.0 * (1 - frac) ** 2                                         # annealed temperature
        return m.grad(x[None])[0] / sc + np.sqrt(2 * T) * rng.normal(size=d)
    return _adam_loop(gr, m.predict, d, budget, seed, cost_per_step=0)


def cbo(m, U, y, d, budget, seed, **kw):
    """Anisotropic consensus-based optimization (Carrillo et al. 2021 variant of Pinnau et al. 2017): N particles,
    dx = −λ(x − v)dt + σ |x − v| ⊙ dW, v = softmin-weighted mean with α on normalised values."""
    rng = np.random.default_rng(seed)
    N = min(50, 10 * d); lam, sig, dt, alpha = 1.0, 0.8, 0.1, 30.0
    X = _starts(d, N, seed + 3).copy()
    B = _Budget(budget); best = (np.inf, X[0])
    try:
        while True:
            B.take(N)
            fx = m.predict(X)
            i = int(np.argmin(fx))
            if fx[i] < best[0]:
                best = (fx[i], X[i].copy())
            z = (fx - fx.min()) / (np.ptp(fx) + 1e-12)
            wts = np.exp(-alpha * z); v = (wts[:, None] * X).sum(0) / wts.sum()
            X = np.clip(X - lam * (X - v) * dt + sig * np.abs(X - v) * np.sqrt(dt) * rng.normal(size=X.shape), 0, 1)
    except StopIteration:
        pass
    return best[1]


def guard_distance(m, U, y, d, budget, seed, **kw):
    from scipy.spatial import cKDTree
    tree = cKDTree(U)
    nn = tree.query(U, k=2)[0][:, 1]
    delta = float(np.quantile(nn, 0.75))
    spread = float(np.ptp(m.predict(U))) + 1e-12
    lam = 10 * spread / delta ** 2

    def f(X):
        dist = tree.query(np.atleast_2d(X), k=1)[0]
        return m.predict(X) + lam * np.maximum(dist - delta, 0) ** 2
    return _multistart(f, None, d, budget, seed)


def pessimistic(m, U, y, d, budget, seed, **kw):
    if m.predict_std(U[:2]) is None:
        return None
    f = lambda X: m.predict(X) + m.predict_std(X)
    return _multistart(f, None, d, budget, seed)


# ── exact MILP for tree ensembles (Mišić 2020) ────────────────────────────────────────────────────────────────
def _trees(m):
    """Leaves of every tree as (weight, [(feature, threshold, go_left)], value) on the standardised scale; split
    rule: x_f <= thr is left for sklearn, x_f < thr is 'yes' for XGBoost/LightGBM (ties have measure zero)."""
    out = []
    if m.name in ("rf", "extratrees"):
        T = m._m.estimators_
        for t in T:
            tr = t.tree_

            def walk(node, path):
                if tr.children_left[node] == -1:
                    out.append((1 / len(T), path, float(tr.value[node].ravel()[0]))); return
                f, th = int(tr.feature[node]), float(tr.threshold[node])
                walk(tr.children_left[node], path + [(f, th, True)])
                walk(tr.children_right[node], path + [(f, th, False)])
            walk(0, [])
        return out
    if m.name in ("xgb", "lgbm"):
        import pandas as pd
        est = m._m.best_estimator_ if hasattr(m._m, "best_estimator_") else m._m
        if m.name == "xgb":
            df = est.get_booster().trees_to_dataframe()
            for tid, g in df.groupby("Tree"):
                rows = g.set_index("ID")

                def walk(nid, path):
                    r = rows.loc[nid]
                    if r.Feature == "Leaf":
                        out.append((1.0, path, float(r.Gain))); return
                    f = int(str(r.Feature).lstrip("f")); th = float(r.Split)
                    walk(r.Yes, path + [(f, th, True)]); walk(r.No, path + [(f, th, False)])
                walk(f"{tid}-0", [])
        else:
            df = est.booster_.trees_to_dataframe()
            for tid, g in df.groupby("tree_index"):
                rows = g.set_index("node_index")

                def walk(nid, path):
                    r = rows.loc[nid]
                    if pd.isna(r.split_feature):
                        out.append((1.0, path, float(r.value))); return
                    f = int(str(r.split_feature).replace("Column_", "")); th = float(r.threshold)
                    walk(r.left_child, path + [(f, th, True)]); walk(r.right_child, path + [(f, th, False)])
                walk(g.node_index.iloc[0], [])
        return out
    return None


def milp_trees(m, U, y, d, budget, seed, time_limit=120, max_leaves=30_000, **kw):
    """Binary x_{f,j} = [x_f <= θ_{f,j}] for each feature's sorted thresholds (monotone in j); continuous leaf
    indicators y_l with Σ_l y_l = 1 per tree; a leaf is reachable only if every split on its path agrees; minimise
    Σ_l w_l v_l y_l. The recovered point is the midpoint of the active threshold interval in each feature."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    leaves = _trees(m)
    if leaves is None:
        return None
    if len(leaves) > max_leaves:
        raise MemoryError(f"{len(leaves)} leaves exceed the MILP cost wall")
    thr = [sorted({th for _, p, _ in leaves for f, th, _ in p if f == j}) for j in range(d)]
    xi = {}
    for j in range(d):
        for k, th in enumerate(thr[j]):
            xi[(j, th)] = len(xi)
    nx, nl = len(xi), len(leaves)
    counts = _leaf_counts(m)                                              # leaves are listed tree by tree
    if sum(counts) != nl:
        raise RuntimeError("tree/leaf bookkeeping mismatch")
    tree_of = np.repeat(np.arange(len(counts)), counts)
    A = lil_matrix((len(counts), nx + nl)); lo, hi = [], []
    r = 0
    for t in range(len(counts)):                                          # Σ y_l = 1 per tree
        for l in np.where(tree_of == t)[0]:
            A[r, nx + l] = 1.0
        lo.append(1.0); hi.append(1.0); r += 1
    cons_rows = []                                                        # y_l <= x (left) or <= 1 − x (right)
    for l, (w, p, v) in enumerate(leaves):
        for f, th, left in p:
            cons_rows.append((l, xi[(f, th)], left))
    A2 = lil_matrix((len(cons_rows) + nx, nx + nl)); lo2, hi2 = [], []
    for i, (l, j, left) in enumerate(cons_rows):
        A2[i, nx + l] = 1.0
        if left:
            A2[i, j] = -1.0; lo2.append(-np.inf); hi2.append(0.0)        # y_l − x_j <= 0
        else:
            A2[i, j] = 1.0; lo2.append(-np.inf); hi2.append(1.0)         # y_l + x_j <= 1
    i = len(cons_rows)
    for j in range(d):                                                    # x_{j,k} <= x_{j,k+1}
        for k in range(len(thr[j]) - 1):
            A2[i, xi[(j, thr[j][k])]] = 1.0; A2[i, xi[(j, thr[j][k + 1])]] = -1.0
            lo2.append(-np.inf); hi2.append(0.0); i += 1
    A2 = A2[:i]
    c = np.r_[np.zeros(nx), [w * v for w, p, v in leaves]]
    integrality = np.r_[np.ones(nx), np.zeros(nl)]
    res = milp(c, constraints=[LinearConstraint(A[:r].tocsr(), lo, hi), LinearConstraint(A2.tocsr(), lo2, hi2)],
               integrality=integrality, bounds=Bounds(0, 1), options=dict(time_limit=time_limit, disp=False))
    if res.x is None:
        milp_trees.last = dict(status=int(res.status), mip_gap=float("nan"), leaves=nl, binaries=nx)
        return None
    xv = res.x[:nx] > 0.5
    point = np.empty(d)
    for j in range(d):
        ths = thr[j]
        if not ths:
            point[j] = 0.5; continue
        on = [k for k, th in enumerate(ths) if xv[xi[(j, th)]]]           # x_j <= θ_k for these k
        upper = ths[on[0]] if on else 1.0
        lower = ths[on[0] - 1] if on and on[0] > 0 else (ths[-1] if not on else 0.0)
        point[j] = np.clip(0.5 * (lower + upper), 0, 1)
    milp_trees.last = dict(status=int(res.status), mip_gap=float(getattr(res, "mip_gap", np.nan) or 0.0),
                           leaves=nl, binaries=nx)
    return point


def _leaf_counts(m):
    if m.name in ("rf", "extratrees"):
        return [int((t.tree_.children_left == -1).sum()) for t in m._m.estimators_]
    est = m._m.best_estimator_ if hasattr(m._m, "best_estimator_") else m._m
    if m.name == "xgb":
        df = est.get_booster().trees_to_dataframe()
        return df[df.Feature == "Leaf"].groupby("Tree").size().tolist()
    df = est.booster_.trees_to_dataframe()
    return df[df.split_feature.isna()].groupby("tree_index").size().tolist()


def _relu_layers(net):
    """(W, b) per Linear layer of a ReLU nn.Sequential, float64."""
    import torch
    L = [mod for mod in net if isinstance(mod, torch.nn.Linear)]
    return [(mod.weight.detach().double().numpy(), mod.bias.detach().double().numpy()) for mod in L]


def milp_relu(m, U, y, d, budget, seed, time_limit=120, **kw):
    """Global minimum of the mean of a ReLU-network ensemble over [0, 1]^d as a mixed-integer linear program
    (Fischetti & Jo 2018 big-M encoding). Pre-activation bounds [l, u] by interval arithmetic layer by layer;
    a unit with u <= 0 is fixed at 0, one with l >= 0 is linear (h = a), and an unstable unit gets a binary δ:
        h >= a,  h <= a − l(1 − δ),  h <= u δ,  h >= 0.
    Objective: (1/K) Σ_k (w_out,k · h_k + b_out,k), the ensemble mean in standardised units. HiGHS, time-limited;
    status and relative gap are recorded (a time-limited solve returns its incumbent, not a certificate)."""
    if m.name != "mlp_relu":
        return None
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    nets = [_relu_layers(n) for n in m._nets]
    K = len(nets)
    lb, ub, integ = [0.0] * d, [1.0] * d, [0] * d                       # variables: x, then per unit h (and δ)
    rows, rlo, rhi = [], [], []                                          # sparse rows as {col: coef}
    c, nb, const = {}, 0, 0.0
    for layers in nets:
        prev, plo, phi = list(range(d)), np.zeros(d), np.ones(d)
        for li, (W, b) in enumerate(layers):
            lo = b + np.minimum(W, 0) @ phi + np.maximum(W, 0) @ plo      # interval bounds on a = W h_prev + b
            hi = b + np.maximum(W, 0) @ phi + np.minimum(W, 0) @ plo
            if li == len(layers) - 1:                                    # output layer: linear, into the objective
                for j, col in enumerate(prev):
                    c[col] = c.get(col, 0.0) + W[0, j] / K
                const += b[0] / K
                break
            cur = []
            for u in range(W.shape[0]):
                h = len(lb); cur.append(h)
                neg_a = {col: -W[u, j] for j, col in enumerate(prev) if W[u, j] != 0.0}
                if hi[u] <= 0:                                           # always inactive
                    lb.append(0.0); ub.append(0.0); integ.append(0); continue
                lb.append(0.0); ub.append(float(hi[u])); integ.append(0)
                if lo[u] >= 0:                                           # always active: h − W h_prev = b
                    rows.append({**neg_a, h: 1.0}); rlo.append(b[u]); rhi.append(b[u]); continue
                dl = len(lb); lb.append(0.0); ub.append(1.0); integ.append(1); nb += 1
                rows.append({**neg_a, h: 1.0}); rlo.append(b[u]); rhi.append(np.inf)             # h >= a
                rows.append({**neg_a, h: 1.0, dl: -lo[u]}); rlo.append(-np.inf); rhi.append(b[u] - lo[u])
                rows.append({h: 1.0, dl: -hi[u]}); rlo.append(-np.inf); rhi.append(0.0)           # h <= u δ
            prev, plo, phi = cur, np.maximum(lo, 0), np.maximum(hi, 0)
    nv = len(lb)
    A = lil_matrix((len(rows), nv))
    for i, r in enumerate(rows):
        for k, v in r.items():
            A[i, k] = v
    cv = np.zeros(nv)
    for k, v in c.items():
        cv[k] = v
    res = milp(cv, constraints=[LinearConstraint(A.tocsr(), rlo, rhi)], integrality=np.array(integ),
               bounds=Bounds(np.array(lb), np.array(ub)), options=dict(time_limit=time_limit, disp=False))
    if res.x is None:                                                     # no incumbent within the time limit
        milp_relu.last = dict(status=int(res.status), mip_gap=float("nan"), binaries=nb, milp_obj=float("nan"))
        return None
    milp_relu.last = dict(status=int(res.status), mip_gap=float(getattr(res, "mip_gap", np.nan) or 0.0),
                          binaries=nb, milp_obj=float(res.fun + const))
    return np.clip(res.x[:d], 0, 1)


_S3, _S5 = np.sqrt(3.0), np.sqrt(5.0)
# per kernel: κ(r), κ'(r)/r, κ''(0)   (r = length-scale-scaled distance; κ'(r)/r is finite at 0 for C² kernels)
_KAPPA = {1.5: (lambda r: (1 + _S3 * r) * np.exp(-_S3 * r), lambda r: -3.0 * np.exp(-_S3 * r), -3.0),
          2.5: (lambda r: (1 + _S5 * r + 5 * r * r / 3) * np.exp(-_S5 * r),
                lambda r: -(5.0 / 3.0) * (1 + _S5 * r) * np.exp(-_S5 * r), -5.0 / 3.0),
          "se": (lambda r: np.exp(-0.5 * r * r), lambda r: -np.exp(-0.5 * r * r), -1.0)}


def _gp_parts(m):
    """(X, α, c, 1/ℓ, (κ, κ'/r, κ''(0))) with mean_z(x) = c Σ_i α_i κ(‖(x − X_i)/ℓ‖) for the sklearn GP learners;
    None otherwise (including Matérn-1/2, which is not differentiable)."""
    from sklearn.gaussian_process.kernels import RBF, Matern
    g = m._m
    if getattr(g, "normalize_y", False):
        return None
    k1 = g.kernel_.k1                                                    # C * base (+ White: zero off the diagonal)
    cst, base = k1.k1.constant_value, k1.k2
    il = 1.0 / np.broadcast_to(np.asarray(base.length_scale, float), (g.X_train_.shape[1],))
    if isinstance(base, Matern):
        kk = _KAPPA.get(float(base.nu))
    elif isinstance(base, RBF):
        kk = _KAPPA["se"]
    else:
        kk = None
    if kk is None:
        return None
    return g.X_train_, np.asarray(g.alpha_, float).reshape(-1), cst, il, kk


def gp_bnb(m, U, y, d, budget, seed, time_limit=120, max_nodes=200_000, tol=1e-4, **kw):
    """Deterministic global minimisation of a GP posterior mean over [0, 1]^d by spatial branch-and-bound. Each box
    takes the larger of two valid lower bounds (standardised units):
      interval  κ decreases in the scaled distance r, so with r_i ∈ [rmin_i, rmax_i] over the box
                μ >= c (Σ_{α_i>0} α_i κ(rmax_i) + Σ_{α_i<0} α_i κ(rmin_i))   (Franey, Ranjan & Chipman 2011, for EI);
                tight far from the data, loose near it when the α_i are large and of mixed sign;
      centred   μ(x) >= μ(c) − Σ_j |∂_j μ(c)| h_j − ‖μ‖_H √N(ρ), with h the half-widths, ρ = ‖h / ℓ‖ and
                N(r) = c [2 − 2κ(r) + 2 r κ'(r) − κ''(0) r²] = max over the box of ‖k_x − k_c − (x − c)·∇k_c‖²_H
                (Cauchy–Schwarz in the RKHS: the first-order Taylor remainder of μ is <μ, k_x − k_c − v·∇k_c>);
                ‖μ‖²_H = αᵀ K_f α with the noise-free kernel matrix, so the cancellation between the α_i is kept.
                N is increasing (checked in tests) and √N = O(r²) for Matérn-5/2 and SE, O(r^1.5) for Matérn-3/2.
    Best-first; upper bounds from box centres and an initial L-BFGS-B multistart; boxes split along the longest side
    in length-scale units; pruned when LB >= UB − tol. Stops at gap <= tol (certified), the node cap or the time
    limit, and records the remaining certified gap."""
    import heapq
    import time
    parts = _gp_parts(m) if m.name in ("gp", "gp_se") else None
    if parts is None:
        return None
    X, al, cst, il, (kap, dkr, k20) = parts
    pos = al > 0
    D = np.sqrt(((((X[:, None, :] - X[None, :, :]) * il) ** 2).sum(-1)))
    Hn = float(np.sqrt(max(al @ (cst * kap(D)) @ al, 0.0)))             # ‖μ‖_H
    del D

    def mu_grad(x):
        Z = (x - X) * il; r = np.sqrt((Z ** 2).sum(1))
        return float(cst * (al @ kap(r))), cst * ((al * dkr(r)) @ (Z * il))

    def mu(x):
        return mu_grad(np.asarray(x, float).reshape(-1))[0]

    def remainder(rho):
        return Hn * np.sqrt(max(cst * (2 - 2 * kap(rho) + 2 * rho * rho * dkr(rho) - k20 * rho * rho), 0.0))

    def lower(lo, hi, fc=None, gc=None):
        dmin = np.maximum(np.maximum(lo - X, 0), X - hi) * il
        dmax = np.maximum(np.abs(X - lo), np.abs(X - hi)) * il
        kmin, kmax = kap(np.sqrt((dmax ** 2).sum(1))), kap(np.sqrt((dmin ** 2).sum(1)))
        lb = float(cst * (al[pos] @ kmin[pos] + al[~pos] @ kmax[~pos]))
        if fc is None:
            fc, gc = mu_grad(0.5 * (lo + hi))
        h = 0.5 * (hi - lo)
        return max(lb, fc - float(np.abs(gc) @ h) - remainder(float(np.sqrt(((h * il) ** 2).sum()))))

    t0 = time.perf_counter()
    best, best_x = np.inf, None
    for x0 in _starts(d, 2 * d + 2, seed):                               # incumbent: local searches on the mean
        r = minimize(lambda x: mu_grad(x), x0, jac=True, method="L-BFGS-B", bounds=[(0, 1)] * d,
                     options=dict(maxiter=100))
        if r.fun < best:
            best, best_x = float(r.fun), np.clip(r.x, 0, 1)
    lo0, hi0 = np.zeros(d), np.ones(d)
    heap, cnt, nodes, status = [(lower(lo0, hi0), 0, lo0, hi0)], 1, 0, "certified"
    while heap:
        lb, _, lo, hi = heap[0]
        if lb >= best - tol:                                             # best-first: every open box is pruned
            heap = []; break
        if nodes >= max_nodes or time.perf_counter() - t0 > time_limit:
            status = "limit"; break
        heapq.heappop(heap); nodes += 1
        j = int(np.argmax((hi - lo) * il)); mid = 0.5 * (lo[j] + hi[j])
        for a, b in ((lo[j], mid), (mid, hi[j])):
            l2, h2 = lo.copy(), hi.copy(); l2[j], h2[j] = a, b
            cx = 0.5 * (l2 + h2); fc, gc = mu_grad(cx)
            if fc < best:
                best, best_x = fc, cx
            lb2 = lower(l2, h2, fc, gc)
            if lb2 < best - tol:
                heapq.heappush(heap, (lb2, cnt, l2, h2)); cnt += 1
    gap = max(best - heap[0][0], 0.0) if heap else 0.0
    gp_bnb.last = dict(status=status, bnb_gap=float(gap), nodes=nodes, bnb_obj=float(best),
                       wall_bnb=time.perf_counter() - t0)
    return best_x


STRATEGIES = {f.__name__: f for f in (mean_lbfgsb, mean_cmaes, thompson_members, member_sgd, smoothed_adam, langevin,
                                       cbo, milp_trees, milp_relu, gp_bnb, guard_distance, pessimistic)}
