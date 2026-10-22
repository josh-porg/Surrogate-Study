"""Numbers and figures for Chapter 2 (Gaussian processes and kernels). Run: python notes/textbook/make_ch02.py"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def matern(r, nu):
    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        a = np.sqrt(3) * r; return (1 + a) * np.exp(-a)
    if nu == 2.5:
        a = np.sqrt(5) * r; return (1 + a + a * a / 3) * np.exp(-a)
    return np.exp(-0.5 * r * r)                                          # nu = inf: squared exponential


def K(x, z, ell, nu, amp=1.0):
    return amp * matern(np.abs(x[:, None] - z[None, :]) / ell, nu)


# 1. prior samples
x = np.linspace(0, 1, 400); rng = np.random.default_rng(3)
fig, axes = plt.subplots(1, 4, figsize=(10, 2.4), sharey=True)
for ax, (nu, name) in zip(axes, ((0.5, r"Matérn $\nu=1/2$"), (1.5, r"Matérn $\nu=3/2$"), (2.5, r"Matérn $\nu=5/2$"),
                                 (np.inf, "squared exponential"))):
    L = np.linalg.cholesky(K(x, x, 0.2, nu) + 1e-8 * np.eye(len(x)))
    for i in range(3):
        ax.plot(x, L @ rng.normal(size=len(x)), color=C[i], lw=1.2)
    ax.set_title(name, fontsize=9, color=INK); style(ax)
axes[0].set_ylabel("$f(x)$", fontsize=9, color=MUTED)
fig.tight_layout(); fig.savefig(OUT / "ch02_prior_samples.pdf")

# 2. posterior on Forrester, Matern 5/2: a fixed sensible length-scale vs what ML-II picks from 6 points
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as Ck, Matern, WhiteKernel
forr = lambda t: (6 * t - 2) ** 2 * np.sin(12 * t - 4)
xd = np.array([0.0, 0.18, 0.4, 0.6, 0.8, 1.0]); yd = forr(xd)
fits = {
    r"length-scale fixed at 0.15": GaussianProcessRegressor(Ck(1.0, (1e-3, 1e4)) * Matern(0.15, "fixed", nu=2.5),
                                                            normalize_y=True, alpha=1e-8).fit(xd[:, None], yd),
    "length-scale by ML-II (10 restarts)": GaussianProcessRegressor(
        Ck(1.0, (1e-3, 1e4)) * Matern(0.2, (1e-2, 10), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-2)),
        normalize_y=True, n_restarts_optimizer=10, random_state=0).fit(xd[:, None], yd)}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
for ax, (title, gpf) in zip(axes, fits.items()):
    m, s = gpf.predict(x[:, None], return_std=True)
    ax.fill_between(x, m - 2 * s, m + 2 * s, color="#cde2fb", label=r"mean $\pm 2$ sd")
    ax.plot(x, forr(x), color=INK, lw=1.0, ls=(0, (4, 3)), label="true $f$")
    ax.plot(x, m, color=C[0], lw=1.6, label="posterior mean")
    ax.plot(xd, yd, "o", color=C[1], ms=5, label="data")
    ax.set_title(title, fontsize=9, color=INK); ax.set_xlabel("$x$", fontsize=9, color=MUTED); style(ax)
    print(title, "->", gpf.kernel_, " NRMSE", round(float(np.sqrt(np.mean((m - forr(x)) ** 2)) / forr(x).std()), 3),
          " log ML", round(float(gpf.log_marginal_likelihood_value_), 2))
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "ch02_posterior.pdf")

# 3. profile log marginal likelihood landscape: slow + fast component + noise
rng = np.random.default_rng(7); n = 40
xs = np.sort(rng.random(n)); ys = np.sin(2 * np.pi * xs) + 0.3 * np.sin(20 * np.pi * xs) + 0.05 * rng.normal(size=n)
ys = (ys - ys.mean()) / ys.std()


def prof_lml(ell, g):
    """log ML with the signal variance profiled out; g = noise/signal variance ratio."""
    A = K(xs, xs, ell, 2.5) + g * np.eye(n)
    Lc = np.linalg.cholesky(A); a = np.linalg.solve(Lc, ys)
    s2 = a @ a / n
    return -0.5 * n * np.log(s2) - np.log(np.diag(Lc)).sum() - 0.5 * n * (np.log(2 * np.pi) + 1)


le = np.linspace(np.log10(0.005), np.log10(2.0), 120); lg = np.linspace(-6, 1, 120)
Z = np.array([[prof_lml(10 ** a, 10 ** b) for a in le] for b in lg])
from scipy.ndimage import maximum_filter
peaks = np.argwhere((Z == maximum_filter(Z, size=9)) & (Z > Z.max() - 40))
fig, ax = plt.subplots(figsize=(5.4, 3.6))
cs = ax.contourf(le, lg, np.clip(Z, Z.max() - 60, None), levels=24, cmap="Blues")
ax.contour(le, lg, np.clip(Z, Z.max() - 60, None), levels=12, colors="white", linewidths=0.4)
for (i, j) in peaks:
    ax.plot(le[j], lg[i], "o", color=C[1], ms=6, mec="white")
    ax.annotate(f"$\\ell$={10**le[j]:.3f}\n noise/signal={10**lg[i]:.1e}\n log ML={Z[i, j]:.1f}", (le[j], lg[i]),
                textcoords="offset points", xytext=(8, 6), fontsize=7, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
ax.set_xlabel(r"$\log_{10}$ length-scale $\ell$", fontsize=9, color=MUTED)
ax.set_ylabel(r"$\log_{10}$ (noise / signal variance)", fontsize=9, color=MUTED)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=8)
fig.colorbar(cs, ax=ax, shrink=0.8).ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(OUT / "ch02_lml_landscape.pdf")
print("local maxima of profile log ML (ell, noise ratio, value):",
      [(round(10 ** le[j], 4), float(f"{10 ** lg[i]:.2e}"), round(float(Z[i, j]), 2)) for i, j in peaks])

# 4. condition numbers of K
xc = np.linspace(0, 1, 50)
print("cond(K), 50 equispaced points on [0,1]:")
for ell in (0.02, 0.05, 0.1, 0.2, 0.5):
    print(f"  ell={ell:4}: SE {np.linalg.cond(K(xc, xc, ell, np.inf)):.2e}  Matern5/2 {np.linalg.cond(K(xc, xc, ell, 2.5)):.2e}"
          f"  Matern1/2 {np.linalg.cond(K(xc, xc, ell, 0.5)):.2e}")
