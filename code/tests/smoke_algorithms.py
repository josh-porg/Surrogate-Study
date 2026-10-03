"""Smoke test for samplers, acquisitions, optimizers, multi-fidelity methods and upscalers (1 seed, small sizes)."""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from surr import acquisitions as A, multifidelity as MF, optimizers as O, problems as P, samplers as S, upscalers as Up  # noqa
from surr.learners import make  # noqa: E402


def nrmse(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / truth.std())


def section(t):
    print(f"\n== {t}")


rng = np.random.default_rng(0)
test = S.sobol(4096, 5, seed=99)

section("samplers: min pairwise distance, n=32, d=2 (larger = better fill)")
from scipy.spatial.distance import pdist  # noqa: E402
for name in S.REGISTRY:
    X = S.make(name, 32, 2, seed=1)
    print(f"  {name:14s} n={len(X):3d}  min-dist={pdist(X).min():.4f}")

section("acquisitions: GP scout on plateau_bump_d2, 12 -> 30 points, NRMSE of the final GP")
p = P.get("plateau_bump_d2"); pool = S.sobol(1024, 2, seed=5); yp = p.f_unit(pool); yt = p.f_unit(test[:, :2])
for name in A.REGISTRY:
    t = time.time(); sel = list(range(12))
    try:
        for _ in range(18):
            m = make("gp", seed=0, restarts=0).fit(pool[sel], yp[sel])
            sel.append(A.acquire(name, m, pool[sel], yp[sel], pool, rng, exclude=sel))
        e = nrmse(make("gp", seed=0).fit(pool[sel], yp[sel]).predict(test[:, :2]), yt)
        print(f"  {name:14s} NRMSE={e:.3f}  {time.time() - t:5.1f}s")
    except Exception as ex:
        print(f"  {name:14s} ERROR {ex!r}"[:120])

section("optimizers on the true function (unit cube), budget 600: regret f(x)-f*")
for pname in ("branin", "rastr", "rosen_d5"):
    p = P.get(pname); fs, _ = p.optimum()
    for name in O.REGISTRY:
        t = time.time()
        try:
            r = O.run(name, p.f_unit, p.d, budget=600, seed=0)
            print(f"  {pname:9s} {name:22s} regret={r['f'] - fs:10.4g} nfev={r['nfev']:4d} {time.time() - t:5.1f}s")
        except Exception as ex:
            print(f"  {pname:9s} {name:22s} ERROR {ex!r}"[:120])

section("multi-fidelity: 60 LF + 10 HF points (d=2), NRMSE on the HF function")
for pname in ("mf_linear__smooth_trig_d2", "mf_nonlinear__smooth_trig_d2", "mf_shifted__smooth_trig_d2",
              "mf_weak__smooth_trig_d2", "mf2_forrester", "mf2_branin"):
    mp = P.MF_REGISTRY[pname]; d = mp.d
    UL = S.sobol(60, d, seed=2); UH = S.maximin_lhs(10, d, seed=3)
    yt = mp.high(test[:, :d]); row = []
    for name in MF.REGISTRY:
        try:
            m = MF.make_mf(name, seed=0).fit(UL, mp.low(UL), UH, mp.high(UH))
            row.append(f"{name}={nrmse(m.predict(test[:, :d]), yt):.3f}")
        except Exception as ex:
            row.append(f"{name}=ERR({type(ex).__name__})")
    print(f"  {pname:30s} r={mp.r:+.2f}  " + "  ".join(row))

section("upscalers: GP and XGB intermediates on smooth_trig_d2 (n=40), value NRMSE and gradient NRMSE")
p = P.get("smooth_trig_d2"); U = S.sobol(40, 2, seed=1); y = p.f_unit(U); T = test[:2048, :2]; yt = p.f_unit(T)
gt = p.grad(p.from_unit(T)) * (p.bounds[:, 1] - p.bounds[:, 0])
for lname in ("gp", "xgb"):
    m = make(lname, seed=0).fit(U, y)
    for name in Up.REGISTRY:
        try:
            s = Up.build(name, m, 2, U=U, y=y)
            g = s.grad(T)
            print(f"  {lname:4s} {name:17s} value={nrmse(s.predict(T), yt):.3f}  grad={np.sqrt(np.mean((g - gt) ** 2)) / gt.std():.3f}")
        except Exception as ex:
            print(f"  {lname:4s} {name:17s} ERROR {ex!r}"[:120])
