"""E8 analysis (RQ4, H5): does upscaling a fitted surrogate (a smooth surface built on the learner's own predictions)
make it a better surface to optimize, and at what accuracy cost? Interim-safe: uses whatever records exist and says
how many tasks each learner has.

Design (runner.exp_E8_upscale): intermediate learner (gp, catboost, xgb, rf, mlp_ens, rbf_tps, pce_lars) fitted on an
E3-style design, then every upscaler (none; tensor cubic grid m = 12 / 24; separable smoothing spline; P-spline
direct; Smolyak; TT grid; partition-of-unity quadratics; subsample ensemble), then optimizers on each at 20d / 50d /
200d surrogate evaluations, scored on the true function (regret normalised by std(f); success = regret < 1e-2).
Unit for tests = problem.
1. Coverage: tasks per learner (interim runs are dominated by whichever learners ran first).
2. Accuracy: NRMSE ratio upscaled / intermediate, the error the upscaler adds, gradient NRMSE.
3. Optimization: success at 200d with the best optimizer per (learner, upscaler), and the paired regret ratio
   upscaled / none for the same optimizer (the H5 question, above all for trees).
4. Cost: build time and evaluation time per point.

    python code/analysis/e8_analysis.py
Writes results/E8_report.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "code" / "analysis"))
from e3_learners import md  # noqa: E402

FLOOR, OK = 1e-8, 1e-2
TASK = ["problem", "inst", "n", "noise", "learner"]


def load():
    best = {}
    for f in sorted((ROOT / "results" / "raw" / "E8_upscale").glob("*.jsonl")):       # not failed_run_1/
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] not in best or r.get("status") == "ok":
                best[r["key"]] = r
    acc, opt, status = [], [], {}
    for r in best.values():
        status[r.get("status")] = status.get(r.get("status"), 0) + 1
        if r.get("status") != "ok":
            continue
        c = r["cfg"]
        t = dict(problem=c["problem"], inst=c["shift_seed"], n=c["n"], noise=c["level"], learner=c["learner"])
        for up, u in r["upscalers"].items():
            if "nrmse" not in u:
                acc.append(dict(t, up=up, failed=True)); continue
            acc.append(dict(t, up=up, failed=False, nrmse=u["nrmse"], added=u["added"], grad=u["grad_nrmse"],
                            t_build=u["t_build"], t_eval_us=u["t_eval_us"], inter=r["intermediate_nrmse"]))
            if not r.get("fstar_confirmed", True):
                continue
            for o, reg in u.get("regret", {}).items():
                if isinstance(reg, list):
                    for b, v in zip((20, 50, 200), reg):
                        opt.append(dict(t, up=up, opt=o, budget=b, regret=max(v, FLOOR)))
    return pd.DataFrame(acc), pd.DataFrame(opt), status


def wil(lr, by):
    g = lr.groupby(by).mean()
    return float(stats.wilcoxon(g).pvalue) if len(g) > 5 and (g != 0).any() else np.nan


def main():
    A, O, status = load()
    if A.empty:
        print("no E8 records yet"); return
    from surr.runner import EXPERIMENTS
    plan = pd.Series([c["learner"] for c in EXPERIMENTS["E8_upscale"]()]).value_counts()
    have = A.drop_duplicates(TASK).learner.value_counts()
    out = [f"# E8 — upscaling for optimization (INTERIM unless every learner is complete; status {status})\n",
           "## 1. Coverage (tasks done / planned)\n",
           md(pd.DataFrame({"done": have, "planned": plan}).fillna(0).astype(int)), "",
           "Note: `pspline_direct` and `subsample_ensemble` ignore the learner and refit the training data (they are "
           "replacements, not upscalers: identical results across learners). The upscalers proper, built on the "
           "learner's predictions, are the tensor and smoothing splines, Smolyak, TT and pou_quadratics. The m = 24 "
           "splines are infeasible at d = 5, so their rows cover d = 2 only." + chr(10)]
    F = A[~A.failed]
    acc = F.groupby(["learner", "up"]).apply(lambda g: pd.Series(dict(
        nrmse_ratio=np.exp(np.log(g.nrmse / g.inter).mean()), added=g.added.median(), grad_nrmse=g.grad.median())),
        include_groups=False)
    fails = A.groupby(["learner", "up"]).failed.mean().rename("failed")
    out += ["## 2. Accuracy: NRMSE upscaled / intermediate (geometric mean), median added error and gradient NRMSE\n",
            md(acc.join(fails, how="outer"), ".3g"), ""]
    if not O.empty:
        G = O[O.budget == 200]
        sr = G.groupby(["learner", "up", "opt"]).regret.apply(lambda x: (x < OK).mean())
        bestopt = sr.groupby(["learner", "up"]).agg(["max", "idxmax"])
        bestopt["best optimizer"] = bestopt["idxmax"].map(lambda t: t[2])
        S = bestopt["max"].unstack("up")
        out += ["## 3. Success at 200d, best optimizer per learner × upscaler\n", md(S, ".2f"), ""]
        base = G[G.up == "none"].set_index(TASK + ["opt"]).regret.rename("none")
        J = G[G.up != "none"].join(base, on=TASK + ["opt"]).dropna(subset=["none"])
        J["lr"] = np.log(J.regret / J["none"])
        P = J.groupby(["learner", "up"]).apply(lambda g: pd.Series(dict(
            ratio=np.exp(g.lr.mean()), better=(g.lr < 0).mean(), p=wil(g.lr, g.problem))), include_groups=False)
        out += ["### Paired regret ratio, upscaled / none, same optimizer and task (< 1: upscaling helps)\n",
                md(P, ".3g"), ""]
    C = F.groupby("up")[["t_build", "t_eval_us"]].median()
    out += ["## 4. Cost: median build time [s] and evaluation time per point [µs]\n", md(C, ".3g"), ""]
    (ROOT / "results" / "E8_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
