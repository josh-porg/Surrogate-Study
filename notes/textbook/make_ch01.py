"""Numbers and figures for Chapter 1 of the private textbook (notes/textbook). Run: python notes/textbook/make_ch01.py"""
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures"
rng = np.random.default_rng(0)
print("ball/cube volume fraction (ball radius 1/2 inside unit cube):")
for d in (1, 2, 3, 5, 10, 20):
    v = math.pi ** (d / 2) / math.gamma(d / 2 + 1) * 0.5 ** d
    print(f"  d={d:2d}  {v:.3g}")
print("distance concentration: (max-min)/min of distances from a random point to 1000 uniform points:")
for d in (2, 10, 100, 1000):
    X = rng.random((1000, d)); q = rng.random(d); D = np.linalg.norm(X - q, axis=1)
    print(f"  d={d:4d}  {(D.max() - D.min()) / D.min():.3g}")
print("grid points for 10 per axis:", {d: 10 ** d for d in (1, 2, 3, 5, 10)})
# finite difference error vs h
x0 = 1.0; true = math.cos(x0)
hs = np.logspace(-15, -1, 200)
fwd = np.abs((np.sin(x0 + hs) - np.sin(x0)) / hs - true)
cen = np.abs((np.sin(x0 + hs) - np.sin(x0 - hs)) / (2 * hs) - true)
print("best forward h %.1e err %.1e; best central h %.1e err %.1e" % (hs[fwd.argmin()], fwd.min(), hs[cen.argmin()], cen.min()))
print("central at h=1e-5: %.1e ; at h=1e-10: %.1e" % (abs((math.sin(x0+1e-5)-math.sin(x0-1e-5))/2e-5-true), abs((math.sin(x0+1e-10)-math.sin(x0-1e-10))/2e-10-true)))
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.loglog(hs, fwd, color="#eb6834", lw=1.5, label="forward difference")
ax.loglog(hs, cen, color="#2a78d6", lw=1.5, label="central difference")
ax.set_xlabel("step size $h$"); ax.set_ylabel(r"absolute error in $f'(1)$, $f=\sin$")
ax.grid(True, which="major", color="#e4e3dd", lw=0.6); ax.legend(frameon=False, fontsize=8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "ch01_fd_error.pdf")
# conditioning: monomial vs Legendre Vandermonde on [-1,1], n = 100 equispaced points
x = np.linspace(-1, 1, 100)
print("condition numbers (n=100 points on [-1,1]):")
for p in (5, 10, 15, 20):
    Vm = np.vander(x, p + 1, increasing=True); Vl = np.polynomial.legendre.legvander(x, p)
    print(f"  degree {p:2d}: monomial {np.linalg.cond(Vm):.2e}  Legendre {np.linalg.cond(Vl):.2e}  monomial normal eq {np.linalg.cond(Vm.T @ Vm):.2e}")
# convergence rate illustration n^{-s/d}
n = np.logspace(1, 5, 50)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for (s, d), c in zip(((2, 1), (2, 2), (2, 5), (2, 10), (4, 10)), ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")):
    ax.loglog(n, n ** (-s / d), color=c, lw=1.5 if (s, d) != (4, 10) else 2.2, ls="-" if (s, d) != (4, 10) else (0, (4, 3)),
              label=f"$s={s}$, $d={d}$")
ax.set_xlabel("number of samples $n$"); ax.set_ylabel(r"error bound $\propto n^{-s/d}$")
ax.grid(True, which="major", color="#e4e3dd", lw=0.6); ax.legend(frameon=False, fontsize=8)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "ch01_rates.pdf")
