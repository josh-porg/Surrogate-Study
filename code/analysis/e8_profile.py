"""E8 cost profile: wall time per upscaler build and per optimizer on one E8 cell, to bound the Kaggle cost.

    python code/analysis/e8_profile.py levy_d5 catboost [upscaler ...]

Runs studies.task_e8 on the n = 20d, noise-free, instance-0 cell (optionally only the named upscalers) and prints the
build time and each optimizer's wall time, largest first. Writes nothing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surr import studies as ST  # noqa: E402
from surr.problems import get  # noqa: E402


def main(problem, learner, only):
    d = get(problem).d
    kw = {"opt": "lbfgs"} if learner == "mlp_ens" else {}
    cfg = dict(kind="e8", exp="E8_upscale", problem=problem, shift_seed=0, learner=learner, kw=kw, n=20 * d,
               noise="none", level=0.0, seed=0, **({"only_upscalers": only} if only else {}))
    t0 = time.perf_counter()
    out = ST.task_e8(cfg)
    print(f"{problem} {learner}: total {time.perf_counter() - t0:.0f}s, fit {out['t_fit']:.1f}s")
    rows = []
    for tag, rec in out["upscalers"].items():
        if "t_build" not in rec:
            print(f"  {tag}: {rec}"); continue
        rows.append((rec["t_build"], tag, "build"))
        rows += [(t, tag, k) for k, t in rec.get("t_opt", {}).items()]
    for t, tag, k in sorted(rows, reverse=True):
        print(f"  {t:8.1f}s  {tag:24s} {k}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
