"""Smoke test: fit every registered learner on a few probes, report clean-truth NRMSE and fit time.
Not a benchmark (1 seed, tiny n) — only checks that every learner runs and is sane."""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from surr import learners as Lr, problems as P  # noqa: E402

probs = sys.argv[1:] or ["smooth_trig_d2", "axis_step_d2", "fourier_sparse_d2", "rotated_ridge_d5"]
n_train = {2: 40, 5: 80}
test = qmc.Sobol(5, seed=99).random(4096)
rows = []
for pname in probs:
    p = P.get(pname); d = p.d
    U = qmc.Sobol(d, seed=1).random(n_train.get(d, 10 * d)); y = p.f_unit(U)
    Ut = test[:, :d]; yt = p.f_unit(Ut); sd = yt.std()
    for name in sorted(Lr.REGISTRY):
        L = Lr.REGISTRY[name]
        if (L.max_d and d > L.max_d):
            rows.append((pname, name, "infeasible(d)", "", "")); continue
        t = time.time()
        try:
            m = Lr.make(name, seed=0).fit(U, y)
            e = np.sqrt(np.mean((m.predict(Ut) - yt) ** 2)) / sd
            s = m.predict_std(Ut[:5]) if L.uq else None
            ok = "uq" if (s is not None and np.all(np.isfinite(s))) else ""
            rows.append((pname, name, f"{e:.3f}", f"{time.time() - t:.2f}s", ok))
        except Exception as ex:
            rows.append((pname, name, "ERROR", f"{time.time() - t:.2f}s", repr(ex)[:90]))
for r in rows:
    print(f"{r[0]:18s} {r[1]:16s} {r[2]:>14s} {r[3]:>8s} {r[4]}")
