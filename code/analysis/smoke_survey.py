"""Smoke test for the survey-extension learners (learners/survey.py): fit, predict, std, gradient on two problems,
with gp and xgb as reference. Output: results/smoke_survey.txt.

    python code/analysis/smoke_survey.py [learner ...]
"""
from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
warnings.filterwarnings("ignore")
from surr import problems as P, samplers as S  # noqa: E402
from surr.learners import REGISTRY, make  # noqa: E402

NEW = ["gam", "pspline_te", "symreg", "sparse_gp", "warped_gp", "deep_gp", "bnn_laplace", "bnn_vi", "bart", "moe",
       "natural_neighbor", "aaa_rational"]
CASES = [("smooth_trig_d2", 60), ("borehole", 200), ("axis_step_d2", 100)]


def main():
    names = sys.argv[1:] or ["gp", "xgb"] + NEW
    lines = []
    for pn, n in CASES:
        p = P.get(pn); d = p.d
        U = S.make("sobol", n, d, seed=0); y = p.f_unit(U)
        T = S.sobol(2048, d, seed=10_000); yt = p.f_unit(T); sc = yt.std()
        for L in names:
            cls = REGISTRY[L]
            if (cls.max_d and d > cls.max_d) or (cls.max_n and n > cls.max_n):
                lines.append(f"{pn:16s} {L:17s} infeasible (d/n wall)"); continue
            try:
                t = time.time(); m = make(L, seed=0).fit(U, y); tf = time.time() - t
                pr = m.predict(T); e = np.sqrt(np.mean((pr - yt) ** 2)) / sc
                sd = m.predict_std(T[:200]) if cls.uq else None
                g = m.grad(T[:50])
                ok = np.isfinite(pr).all() and np.isfinite(g).all() and (sd is None or np.isfinite(sd).all())
                rep = np.allclose(m.predict(T[:50]), pr[:50])
                lines.append(f"{pn:16s} {L:17s} nrmse {e:7.4f}  fit {tf:6.1f}s  finite {ok}  repeatable {rep}"
                             + (f"  std~{np.median(sd):.3g}" if sd is not None else ""))
            except Exception as ex:
                lines.append(f"{pn:16s} {L:17s} ERROR {type(ex).__name__}: {str(ex)[:150]}")
                traceback.print_exc()
            print(lines[-1], flush=True)
    (ROOT / "results" / "smoke_survey.txt").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
