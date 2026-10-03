"""Optimizers run on a surrogate (or any callable) over the unit box. Uniform interface:

    run(name, f, d, budget, seed, grad=None, x0=None) -> dict(x, f, nfev, nit)

`f` maps (n, d) -> (n,) (vectorised); `grad` maps (n, d) -> (n, d) and is used by gradient methods when given,
otherwise they use finite differences. `budget` caps function evaluations (approximately for methods that only
expose iteration limits). Local methods are multistarted from Sobol' points inside the budget.

  ipopt_multistart  IPOPT (interior point, L-BFGS Hessian) via CasADi   Wächter & Biegler 2006
  slsqp             SQP                                                 Kraft 1988
  lbfgsb            L-BFGS-B                                            Byrd et al. 1995
  trust_constr      trust-region interior point (SciPy)                 Conn, Gould & Toint 2000
  nelder_mead       Nelder–Mead simplex (bounded)                       Nelder & Mead 1965
  cobyla            COBYLA                                              Powell 1994
  direct            DIRECT                                              Jones et al. 1993
  differential_evolution  DE/rand/1/bin                                 Storn & Price 1997
  cmaes             CMA-ES                                              Hansen & Ostermeier 2001
  pso               particle swarm (inertia 0.72, c1=c2=1.49)            Kennedy & Eberhart 1995
  basin_hopping     basin hopping + L-BFGS-B                            Wales & Doye 1997
"""
from __future__ import annotations

import numpy as np
from scipy import optimize as so
from scipy.stats import qmc

REGISTRY = {}


def _reg(fn):
    REGISTRY[fn.__name__] = fn
    return fn


class _Counter:
    def __init__(self, f, budget):
        self.f, self.budget, self.n, self.best = f, budget, 0, (np.inf, None)

    def __call__(self, x):
        x = np.clip(np.asarray(x, float), 0, 1)
        self.n += 1
        v = float(self.f(x[None])[0])
        if v < self.best[0]:
            self.best = (v, x.copy())
        return v


def _starts(d, k, seed):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return qmc.Sobol(d, scramble=True, seed=seed).random(k)


def _local(method, f, d, budget, seed, grad=None, n_starts=None, **opts):
    F = _Counter(f, budget)
    jac = (lambda x: grad(np.clip(x, 0, 1)[None])[0]) if grad is not None else None
    k = n_starts or max(1, min(8, budget // (30 * d)))
    for x0 in _starts(d, k, seed):
        if F.n >= budget:
            break
        try:
            so.minimize(F, x0, method=method, jac=jac if method in ("SLSQP", "L-BFGS-B", "trust-constr") else None,
                        bounds=[(0, 1)] * d, options=dict(maxiter=max(10, (budget - F.n) // max(k, 1)), **opts))
        except Exception:
            pass
    return dict(x=F.best[1], f=F.best[0], nfev=F.n)


@_reg
def slsqp(f, d, budget, seed=0, grad=None):
    return _local("SLSQP", f, d, budget, seed, grad)


@_reg
def lbfgsb(f, d, budget, seed=0, grad=None):
    return _local("L-BFGS-B", f, d, budget, seed, grad)


@_reg
def trust_constr(f, d, budget, seed=0, grad=None):
    return _local("trust-constr", f, d, budget, seed, grad)


@_reg
def nelder_mead(f, d, budget, seed=0, grad=None):
    return _local("Nelder-Mead", f, d, budget, seed)


@_reg
def cobyla(f, d, budget, seed=0, grad=None):
    return _local("COBYLA", f, d, budget, seed)


@_reg
def direct(f, d, budget, seed=0, grad=None):
    F = _Counter(f, budget)
    so.direct(F, [(0, 1)] * d, maxfun=budget, locally_biased=True)
    return dict(x=F.best[1], f=F.best[0], nfev=F.n)


@_reg
def differential_evolution(f, d, budget, seed=0, grad=None):
    F = _Counter(f, budget); pop = 15
    so.differential_evolution(F, [(0, 1)] * d, popsize=pop, maxiter=max(1, budget // (pop * d) - 1),
                              seed=seed, polish=False, tol=0)
    return dict(x=F.best[1], f=F.best[0], nfev=F.n)


@_reg
def cmaes(f, d, budget, seed=0, grad=None):
    import cma
    F = _Counter(f, budget)
    es = cma.CMAEvolutionStrategy(np.full(d, 0.5), 0.3, {"bounds": [0, 1], "seed": seed + 1, "maxfevals": budget,
                                                         "verbose": -9})
    while not es.stop():
        X = es.ask(); es.tell(X, [F(x) for x in X])
    return dict(x=F.best[1], f=F.best[0], nfev=F.n)


@_reg
def pso(f, d, budget, seed=0, grad=None, n_part=None):
    rng = np.random.default_rng(seed); F = _Counter(f, budget)
    m = n_part or min(40, 10 + 2 * d)
    X = rng.random((m, d)); V = 0.1 * rng.normal(size=(m, d))
    P = X.copy(); Pf = np.array([F(x) for x in X]); g = P[np.argmin(Pf)].copy()
    while F.n + m <= budget:
        r1, r2 = rng.random((m, d)), rng.random((m, d))
        V = 0.72 * V + 1.49 * r1 * (P - X) + 1.49 * r2 * (g - X)
        X = np.clip(X + V, 0, 1)
        fx = np.array([F(x) for x in X])
        better = fx < Pf; P[better], Pf[better] = X[better], fx[better]
        g = P[np.argmin(Pf)].copy()
    return dict(x=F.best[1], f=F.best[0], nfev=F.n)


@_reg
def basin_hopping(f, d, budget, seed=0, grad=None):
    F = _Counter(f, budget)
    jac = (lambda x: grad(np.clip(x, 0, 1)[None])[0]) if grad is not None else None
    so.basinhopping(F, _starts(d, 1, seed)[0], niter=max(1, budget // (40 * d)), stepsize=0.2, seed=seed,
                    minimizer_kwargs=dict(method="L-BFGS-B", jac=jac, bounds=[(0, 1)] * d,
                                          options=dict(maxiter=30)))
    return dict(x=F.best[1], f=F.best[0], nfev=F.n)


@_reg
def ipopt_multistart(f, d, budget, seed=0, grad=None):
    """IPOPT through CasADi with a black-box callback; gradients from `grad` if given, else CasADi finite
    differences; limited-memory Hessian (the thesis optimizer class)."""
    import casadi as ca
    F = _Counter(f, budget)

    class CB(ca.Callback):
        def __init__(s):
            ca.Callback.__init__(s); s.construct("f", {"enable_fd": grad is None})
        def get_n_in(s): return 1
        def get_n_out(s): return 1
        def get_sparsity_in(s, i): return ca.Sparsity.dense(d, 1)
        def eval(s, arg): return [F(np.array(arg[0]).ravel())]
        def has_jacobian(s): return grad is not None
        def get_jacobian(s, name, inames, onames, opts):
            class J(ca.Callback):
                def __init__(t):
                    ca.Callback.__init__(t); t.construct(name, {})
                def get_n_in(t): return 2
                def get_n_out(t): return 1
                def get_sparsity_in(t, i): return ca.Sparsity.dense(d, 1) if i == 0 else ca.Sparsity.dense(1, 1)
                def get_sparsity_out(t, i): return ca.Sparsity.dense(1, d)
                def eval(t, arg): return [grad(np.clip(np.array(arg[0]).ravel(), 0, 1)[None])[0][None]]
            s._jac = J(); return s._jac

    cb = CB(); x = ca.MX.sym("x", d)
    solver = ca.nlpsol("s", "ipopt", {"x": x, "f": cb(x)},
                       {"ipopt.print_level": 0, "print_time": 0, "ipopt.hessian_approximation": "limited-memory",
                        "ipopt.max_iter": 200, "ipopt.sb": "yes"})
    for x0 in _starts(d, max(1, min(8, budget // (40 * d))), seed):
        if F.n >= budget:
            break
        try:
            solver(x0=x0, lbx=0, ubx=1)
        except Exception:
            pass
    return dict(x=F.best[1], f=F.best[0], nfev=F.n)


# ── first-order stochastic-approximation family (projected onto the box, multistarted) ───────────────────────
#  adam              Adam                                    Kingma & Ba 2015
#  sgd_momentum      heavy-ball SGD                          Polyak 1964
#  bb_gd             Barzilai–Borwein step gradient descent  Barzilai & Borwein 1988
#  spsa              simultaneous-perturbation SA (2 evals)  Spall 1992
#  kiefer_wolfowitz  finite-difference SA (2d evals)         Kiefer & Wolfowitz 1952
#  robbins_monro     SA with a_k = a/k on a gradient oracle  Robbins & Monro 1951
# Gradient source: `grad` if given, else central differences (2d evals per step). With `stochastic_grad` (a callable
# returning a *random* gradient estimate, e.g. from one ensemble member), they run as genuine SGD.

def _fd_grad(F, x, h=1e-4):
    g = np.empty_like(x)
    for j in range(len(x)):
        e = np.zeros_like(x); e[j] = h
        g[j] = (F(np.clip(x + e, 0, 1)) - F(np.clip(x - e, 0, 1))) / (2 * h)
    return g


def _scale(F, d, seed):
    """Spread of f over 2d+2 Sobol points (counted against the budget): makes step sizes scale-free, the automatic
    stand-in for the per-problem gain tuning these methods need in practice."""
    v = [F(x) for x in _starts(d, 2 * d + 2, seed + 7)]
    return float(np.std(v)) + 1e-12


def _first_order(step_rule, f, d, budget, seed, grad=None, stochastic_grad=None, n_starts=None, lr=0.05):
    F = _Counter(f, budget); rng = np.random.default_rng(seed)
    sc = _scale(F, d, seed)                                              # gains act on f / spread(f)
    G0 = (lambda x: stochastic_grad(x[None], rng)[0]) if stochastic_grad else \
         (lambda x: grad(x[None])[0]) if grad is not None else (lambda x: _fd_grad(F, x))
    G = lambda x: G0(x) / sc
    k = n_starts or max(1, min(8, budget // (50 * (2 * d + 1))))
    for x0 in _starts(d, k, seed):
        x, state = x0.copy(), {}
        for t in range(1, 10_000):
            if F.n >= budget * (len(state) * 0 + 1) and F.n >= budget:
                break
            F(x)
            g = G(x)
            x = np.clip(step_rule(x, g, t, state, lr), 0, 1)
            if F.n >= budget:
                break
    return dict(x=F.best[1], f=F.best[0], nfev=F.n)


def _adam(x, g, t, s, lr, b1=0.9, b2=0.999):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g; s["v"] = b2 * s.get("v", 0) + (1 - b2) * g * g
    return x - lr * (s["m"] / (1 - b1**t)) / (np.sqrt(s["v"] / (1 - b2**t)) + 1e-8)


def _momentum(x, g, t, s, lr, beta=0.9):
    gn = g / (np.linalg.norm(g) + 1e-12) if np.linalg.norm(g) > 1 else g     # crude scale guard
    s["v"] = beta * s.get("v", 0) + gn
    return x - lr * s["v"]


def _bb(x, g, t, s, lr):
    if "xp" in s:
        dx, dg = x - s["xp"], g - s["gp"]
        a = abs(dx @ dg) / max(dg @ dg, 1e-16)                                  # BB2 step
        a = float(np.clip(a, 1e-6, 1.0))
    else:
        a = lr / (np.linalg.norm(g) + 1e-12)
    s["xp"], s["gp"] = x.copy(), g.copy()
    return x - a * g


def _rm(x, g, t, s, lr):
    return x - (lr / t) * g


@_reg
def adam(f, d, budget, seed=0, grad=None, stochastic_grad=None):
    return _first_order(_adam, f, d, budget, seed, grad, stochastic_grad, lr=0.02)


@_reg
def sgd_momentum(f, d, budget, seed=0, grad=None, stochastic_grad=None):
    return _first_order(_momentum, f, d, budget, seed, grad, stochastic_grad, lr=0.01)


@_reg
def bb_gd(f, d, budget, seed=0, grad=None, stochastic_grad=None):
    return _first_order(_bb, f, d, budget, seed, grad, stochastic_grad, lr=0.05)


@_reg
def robbins_monro(f, d, budget, seed=0, grad=None, stochastic_grad=None):
    return _first_order(_rm, f, d, budget, seed, grad, stochastic_grad, lr=0.1)


@_reg
def kiefer_wolfowitz(f, d, budget, seed=0, grad=None, stochastic_grad=None):
    """Classic KW: central finite differences with shrinking width c_k = c/k^(1/6) and step a_k = a/k."""
    F = _Counter(f, budget)
    for x0 in _starts(d, max(1, min(4, budget // (100 * d))), seed):
        x = x0.copy()
        for k in range(1, 10_000):
            if F.n + 2 * d > budget:
                break
            ck, ak = 0.05 / k ** (1 / 6), 0.1 / k
            g = _fd_grad(F, x, h=ck)
            x = np.clip(x - ak * g / (np.linalg.norm(g) + 1e-12) * min(np.linalg.norm(g), 1.0) * 5, 0, 1)
    return dict(x=F.best[1], f=F.best[0], nfev=F.n)


@_reg
def spsa(f, d, budget, seed=0, grad=None, stochastic_grad=None, a=0.1, c=0.05, A=None, alpha=0.602, gamma=0.101):
    """Spall's SPSA with the standard gain sequences a_k = a/(k+1+A)^0.602, c_k = c/(k+1)^0.101; Rademacher
    perturbations; 2 evaluations per iteration regardless of d."""
    rng = np.random.default_rng(seed); F = _Counter(f, budget)
    sc = _scale(F, d, seed)
    k_starts = max(1, min(4, budget // 200))
    A = A if A is not None else 0.1 * budget / (2 * k_starts)
    for x0 in _starts(d, k_starts, seed):
        x = x0.copy()
        for k in range(10_000):
            if F.n + 3 > budget * 1.0 and F.n >= budget - 2:
                break
            ak, ck = a / (k + 1 + A) ** alpha, c / (k + 1) ** gamma
            delta = rng.choice([-1.0, 1.0], d)
            gh = (F(np.clip(x + ck * delta, 0, 1)) - F(np.clip(x - ck * delta, 0, 1))) / (2 * ck) * delta / sc
            x = np.clip(x - ak * gh, 0, 1)
            if F.n >= budget:
                break
        F(x)
    return dict(x=F.best[1], f=F.best[0], nfev=F.n)


def run(name, f, d, budget=2000, seed=0, grad=None):
    return REGISTRY[name](f, d, budget, seed=seed, grad=grad)
