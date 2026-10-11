"""BUGS N-14 diagnosis: why do KRR (and XGBoost at d = 10) match or beat the ML-II GP on `gp_draw`?

`gp_draw` is a 256-feature random-Fourier draw with independent Student-t(5) frequencies / 0.3, i.e. the spectral
draw of a *product* of 1-D Matérn-5/2 kernels with ℓ = 0.3 and unit variance. For a draw from a known prior the GP
posterior mean under the true kernel is the Bayes-optimal predictor, so it bounds what any learner can do:
- our `gp` far behind the oracle            → GP hyper-parameter fitting is the problem;
- oracle no better than KRR / XGBoost       → the function is not learnable at this n (no GP bug);
- oracle far ahead and `gp` close to it     → nothing wrong; look elsewhere (e.g. scoring).

    python code/analysis/n14_gp_draw.py        (clean data, Sobol' designs, seeds 0-4; about 10 min on the laptop)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")
from surr import problems as P, samplers as S  # noqa: E402
from surr.learners import make  # noqa: E402

ELL = 0.3


def matern52_product(A, B, ell=ELL):
    K = np.ones((len(A), len(B)))
    for j in range(A.shape[1]):
        r = np.sqrt(5) * np.abs(A[:, j, None] - B[None, :, j]) / ell
        K *= (1 + r + r * r / 3) * np.exp(-r)
    return K


def oracle(U, y, T, nugget=1e-8):
    K = matern52_product(U, U) + nugget * np.eye(len(U))
    return matern52_product(T, U) @ np.linalg.solve(K, y)


def main():
    cases = [(2, (20, 50, 200)), (5, (50, 200, 500)), (10, (100, 500))]
    learners = ["gp", "krr", "xgb"]
    print(f"{'problem':12s} {'n':>4s}  {'oracle':>7s} " + " ".join(f"{L:>7s}" for L in learners) +
          "   gp fitted: amp, median ℓ, noise")
    for d, ns in cases:
        p = P.get(f"gp_draw_d{d}")
        for n in ns:
            res = {k: [] for k in ["oracle"] + learners}; fitted = []
            for seed in range(5):
                U = S.make("sobol", n, d, seed=seed); y = p.f_unit(U)
                T = S.sobol(4096, d, seed=10_000 + seed); yt = p.f_unit(T); sc = yt.std()
                res["oracle"].append(np.sqrt(np.mean((oracle(U, y, T) - yt) ** 2)) / sc)
                for L in learners:
                    m = make(L, seed=seed).fit(U, y)
                    res[L].append(np.sqrt(np.mean((m.predict(T) - yt) ** 2)) / sc)
                    if L == "gp":
                        kp = m._m.kernel_.get_params()
                        fitted.append((kp["k1__k1__constant_value"], np.median(kp["k1__k2__length_scale"]),
                                       kp["k2__noise_level"]))
            f = np.array(fitted)
            print(f"gp_draw_d{d:<4d} {n:4d}  " + " ".join(f"{np.mean(res[k]):7.3f}" for k in ["oracle"] + learners) +
                  f"   {np.median(f[:, 0]):.2g}, {np.median(f[:, 1]):.2g}, {np.median(f[:, 2]):.1e}", flush=True)


if __name__ == "__main__":
    main()
