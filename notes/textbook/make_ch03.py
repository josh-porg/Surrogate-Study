"""Numbers and figures for Chapter 3 (polynomial chaos and sparse regression). Run: python notes/textbook/make_ch03.py"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as L
from scipy.stats import qmc

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
OUT = HERE / "figures"
sys.path.insert(0, str(HERE.parents[1] / "code"))
from surr.learners.basis import _legendre_design  # noqa: E402  (the study's own orthonormal Legendre basis)

INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def indices(d, p, q=1.0):
    """Multi-indices alpha in N^d with ||alpha||_q <= p (q = 1: total degree), by recursion (no p^d enumeration)."""
    out = []

    def rec(prefix, budget):
        j = len(prefix)
        if j == d:
            out.append(tuple(prefix)); return
        for a in range(p + 1):
            b = budget - a ** q
            if b < -1e-9:
                break
            rec(prefix + [a], b)
    rec([], p ** q)
    return np.array(out)


def sobol(n, d, seed=0):
    return qmc.Sobol(d, scramble=True, seed=seed).random(n)


# 1. basis sizes ---------------------------------------------------------------------------------------------------
print("1. number of basis terms: total degree C(d+p,p) | hyperbolic q=0.75 | q=0.5")
for d in (2, 5, 10, 20):
    row = []
    for p in (2, 3, 5):
        row.append(f"p={p}: {math.comb(d + p, p):>6} | {len(indices(d, p, 0.75)):>5} | {len(indices(d, p, 0.5)):>4}")
    print(f"  d={d:2d}  " + "   ".join(row))

# 2. coefficient decay -----------------------------------------------------------------------------------------------
# orthonormal Legendre coefficients c_k = <f, sqrt(2k+1) P_k> under the uniform measure on [-1, 1], by Gauss-Legendre
# quadrature on [-1, 0] and [0, 1] separately so the kink and the jump at 0 are integrated exactly as piecewise-smooth
xg, wg = L.leggauss(400)
xs = np.concatenate([(xg - 1) / 2, (xg + 1) / 2]); ws = np.concatenate([wg, wg]) / 2 / 2   # uniform measure: dx/2
K = 60
Pn = np.stack([L.legval(xs, np.eye(K + 1)[k]) * np.sqrt(2 * k + 1) for k in range(K + 1)])
funcs = {"$e^{x}$ (entire)": np.exp, r"$1/(1+25x^2)$ (analytic, poles near $[-1,1]$)": lambda x: 1 / (1 + 25 * x * x),
         "$|x|$ (kink)": np.abs, r"$\mathrm{sign}(x)$ (jump)": np.sign}
coef = {k: Pn @ (ws * f(xs)) for k, f in funcs.items()}
print("2. coefficient decay (even functions have zero odd coefficients and vice versa: only the nonzero parity is used)")
for name, c in coef.items():
    a = np.abs(c); k = np.arange(K + 1)
    par = 0 if a[0::2].sum() > a[1::2].sum() else 1                     # the parity that carries the function
    ks = k[(k % 2 == par) & (k >= 10) & (a > 1e-12)]
    at = {kk: a[kk + (kk % 2 != par)] for kk in (10, 20, 40)}
    if len(ks) < 5:
        print(f"  {name:48s} reaches the 1e-12 round-off floor at degree {int(k[(a < 1e-12) & (k > 0)][0])} (faster than any geometric rate)")
        continue
    lg = np.log(a[ks])
    geo = np.exp(np.polyfit(ks, lg, 1)[0]); alg = np.polyfit(np.log(ks), lg, 1)[0]
    print(f"  {name:48s} |c| near k=10,20,40: {at[10]:.1e} {at[20]:.1e} {at[40]:.1e}; fit: ratio {geo:.3f}/degree, "
          f"power slope {alg:.2f}")
rho = 1 / 5 + math.sqrt(1 + 1 / 25)
print(f"  Runge: Bernstein-ellipse prediction |c_k| ~ rho^-k with rho = {rho:.4f}, ratio {1 / rho:.4f}/step")
FLOOR = 1e-12                                     # quadrature round-off: values below it are not information
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
for i, ((name, c), f) in enumerate(zip(coef.items(), funcs.values())):
    a = np.abs(c); k = np.arange(K + 1); m = a > FLOOR
    axes[0].semilogy(k[m], a[m], "o", ms=3, color=C[i], label=name)
    # L2 error of keeping degrees 0..k (Parseval): the squared coefficients beyond k, summed directly up to K, plus the
    # part beyond K, R = E[f^2] - sum_{j<=K} c_j^2. R matters for slowly decaying f (else the curve falls to zero at
    # K); for fast decay it is pure cancellation noise (~1e-16) and is dropped, else it would fake a 1e-8 floor
    Ef2 = float(ws @ f(xs) ** 2)
    R = Ef2 - float((a ** 2).sum()); R = R if R > 1e-10 * Ef2 else 0.0
    tail = np.cumsum((a ** 2)[::-1])[::-1]
    err = np.sqrt(np.r_[tail[1:], 0.0] + R)
    ok = err > FLOOR
    axes[1].semilogy(k[ok], err[ok], color=C[i], lw=1.5)
axes[0].set_xlabel("degree $k$", fontsize=9, color=MUTED); axes[0].set_ylabel("$|c_k|$", fontsize=9, color=MUTED)
axes[1].set_xlabel("highest degree kept $k$", fontsize=9, color=MUTED)
axes[1].set_ylabel("$L^2$ error of the truncation", fontsize=9, color=MUTED)
axes[0].set_ylim(FLOOR, 2); axes[1].set_ylim(FLOOR, 2)
for ax in axes:
    style(ax)
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "ch03_coeff_decay.pdf")

# 3. sparse recovery: d = 10, total degree 3 (P = 286 terms), a function with 8 active terms --------------------------
from sklearn.linear_model import LassoLarsCV, RidgeCV
d, p = 10, 3
A = indices(d, p, 1.0); P = len(A)
rng = np.random.default_rng(1)
active = np.r_[0, rng.choice(np.arange(1, P), 7, replace=False)]
ctrue = np.zeros(P); ctrue[active] = rng.choice([-1, 1], 8) * rng.uniform(0.5, 2.0, 8)
Ut = sobol(4096, d, seed=99); Pt = _legendre_design(Ut, A); yt = Pt @ ctrue
res = {"least squares (min-norm if n < P)": [], "ridge (CV)": [], "LASSO path by LARS (CV)": []}
ns = (30, 60, 120, 240, 480)
print(f"3. sparse recovery, d={d}, total degree {p}: P={P} terms, 8 active; NRMSE on 4096 test points")
for n in ns:
    U = sobol(n, d, seed=n); Ph = _legendre_design(U, A); y = Ph @ ctrue + 0.01 * rng.normal(size=n)
    w_ls = np.linalg.lstsq(Ph, y, rcond=None)[0]
    w_r = RidgeCV(alphas=np.logspace(-8, 2, 41), fit_intercept=False).fit(Ph, y).coef_
    las = LassoLarsCV(cv=5, fit_intercept=True, max_n_alphas=400).fit(Ph[:, 1:], y)
    pred = {"least squares (min-norm if n < P)": Pt @ w_ls, "ridge (CV)": Pt @ w_r, "LASSO path by LARS (CV)": las.predict(Pt[:, 1:])}
    for k, v in pred.items():
        res[k].append(float(np.sqrt(np.mean((v - yt) ** 2)) / yt.std()))
    sel = np.flatnonzero(np.abs(las.coef_) > 1e-6) + 1
    hit = len(set(sel) & set(active[1:]))
    print(f"  n={n:3d}  LS {res['least squares (min-norm if n < P)'][-1]:.3f}  ridge {res['ridge (CV)'][-1]:.3f}"
          f"  LASSO {res['LASSO path by LARS (CV)'][-1]:.3f}   LASSO kept {len(sel)} terms, {hit}/7 true non-constant terms")
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for i, (k, v) in enumerate(res.items()):
    ax.loglog(ns, np.maximum(v, 1e-4), "o-", color=C[i], lw=1.5, ms=4, label=k)
ax.axvline(P, color=MUTED, lw=0.8, ls=(0, (3, 3))); ax.text(P * 1.04, 0.5, f"$n = P = {P}$", fontsize=8, color=MUTED)
ax.set_xlabel("number of samples $n$", fontsize=9, color=MUTED); ax.set_ylabel("NRMSE (test)", fontsize=9, color=MUTED)
style(ax); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "ch03_sparse.pdf")

# 4. Sobol' indices from the coefficients: Ishigami function -----------------------------------------------------------
a_, b_ = 7.0, 0.1
ish = lambda X: np.sin(X[:, 0]) + a_ * np.sin(X[:, 1]) ** 2 + b_ * X[:, 2] ** 4 * np.sin(X[:, 0])
V = a_ ** 2 / 8 + b_ * np.pi ** 4 / 5 + b_ ** 2 * np.pi ** 8 / 18 + 0.5
exact = dict(S1=(b_ * np.pi ** 4 / 5 + b_ ** 2 * np.pi ** 8 / 50 + 0.5) / V, S2=a_ ** 2 / 8 / V, S3=0.0,
             ST1=(b_ * np.pi ** 4 / 5 + b_ ** 2 * np.pi ** 8 / 18 + 0.5) / V, ST2=a_ ** 2 / 8 / V,
             ST3=(b_ ** 2 * np.pi ** 8 / 18 - b_ ** 2 * np.pi ** 8 / 50) / V)
print(f"4. Ishigami: exact mean {a_ / 2:.4f}, variance {V:.4f};", {k: round(v, 4) for k, v in exact.items()})
A3 = indices(3, 14, 1.0)
for n in (100, 200, 400, 800):
    U = sobol(n, 3, seed=7); y = ish(-np.pi + 2 * np.pi * U)
    Ph = _legendre_design(U, A3)
    m = LassoLarsCV(cv=5, fit_intercept=True, max_n_alphas=600).fit(Ph[:, 1:], y)
    # with an intercept the constant term is not orthonormal-centred, so read the mean from the model on a big sample
    Ub = sobol(2 ** 16, 3, seed=8); mean_hat = float(m.predict(_legendre_design(Ub, A3)[:, 1:]).mean())
    cc = m.coef_; Ak = A3[1:]
    var = float((cc ** 2).sum())
    S = {f"S{i + 1}": float((cc[(Ak[:, i] > 0) & (Ak.sum(1) == Ak[:, i])] ** 2).sum() / var) for i in range(3)}
    S.update({f"ST{i + 1}": float((cc[Ak[:, i] > 0] ** 2).sum() / var) for i in range(3)})
    Yt = ish(-np.pi + 2 * np.pi * Ub)
    nr = float(np.sqrt(np.mean((m.predict(_legendre_design(Ub, A3)[:, 1:]) - Yt) ** 2)) / Yt.std())
    print(f"  n={n:3d}: kept {int((np.abs(cc) > 1e-8).sum())}/{len(cc)} terms, NRMSE {nr:.3g}, mean {mean_hat:.4f}, "
          f"variance {var:.4f}, " + ", ".join(f"{k} {v:.4f}" for k, v in S.items()))

# Monte Carlo (pick-freeze, Saltelli/Jansen) for comparison: N base samples cost N (d + 2) evaluations
rng = np.random.default_rng(5)
for N in (1000, 10000):
    Am, Bm = -np.pi + 2 * np.pi * rng.random((N, 3)), -np.pi + 2 * np.pi * rng.random((N, 3))
    fA, fB = ish(Am), ish(Bm); Vt = np.var(np.r_[fA, fB])
    S1 = []
    for i in range(3):
        ABi = Am.copy(); ABi[:, i] = Bm[:, i]; fABi = ish(ABi)
        S1.append(float(np.mean(fB * (fABi - fA)) / Vt))
    print(f"  Monte Carlo pick-freeze, {N * 5} evaluations: S1..S3 = " + ", ".join(f"{s:.3f}" for s in S1))
