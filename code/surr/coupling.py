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


STRATEGIES = {f.__name__: f for f in (mean_lbfgsb, mean_cmaes, thompson_members, member_sgd, smoothed_adam, langevin,
                                       cbo, milp_trees, guard_distance, pessimistic)}
