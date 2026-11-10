"""E14 analysis: do self-designing surrogates (Smolyak sparse grid, TT-cross) beat ordinary learners on a Sobol' design?

Design (runner.exp_E14_selfdesign): 28 problems (favoured structure: additive, low-rank product, smooth; hurt: rotated
ridge, steps) × budget {50, 200, 1000} × noise {0, 1 %} × 5 seeds; smolyak and tt_cross choose their own points (they
may use fewer than the budget: n_used); gp, pce_lars, catboost, rbf_tps fit a Sobol' design of the full budget.
Unit for tests = problem.
1. Each self-designing method vs the best baseline in the cell, and vs gp.
2. By problem structure and by budget: where they win.
3. Budget actually used.

    python code/analysis/e14_analysis.py
Writes results/E14_report.md.
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

CELL = ["problem", "budget", "level", "seed"]
FL = 1e-10
SELF = ("smolyak", "tt_cross")
BASE = ("gp", "pce_lars", "catboost", "rbf_tps")


def structure(p):
    for k in ("additive", "lowrank", "smooth_trig", "rotated_ridge", "axis_step", "kink"):
        if p.startswith(k):
            return k
    return "benchmark"


def load():
    rows, seen = [], {}
    for f in sorted((ROOT / "results" / "raw" / "E14_selfdesign").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                seen[r["key"]] = r
    for r in seen.values():
        c = r["cfg"]
        rows.append(dict(problem=c["problem"], struct=structure(c["problem"]), budget=c["budget"], level=c["level"],
                         seed=c["seed"], method=c["method"], nrmse=r["nrmse"], n_used=r.get("n_used", c["budget"])))
    D = pd.DataFrame(rows)
    # Smolyak without noise is deterministic and ran once (seed 0): copy it to the other seeds' cells
    s0 = D[(D.method == "smolyak") & (D.level == 0.0)]
    D = pd.concat([D] + [s0.assign(seed=s) for s in range(1, 5)], ignore_index=True)
    return D


def wil(lr, by):
    g = lr.groupby(by).mean()
    return float(stats.wilcoxon(g).pvalue) if len(g) > 5 and (g != 0).any() else np.nan


def main():
    D = load()
    if D.empty:
        print("no E14 records yet"); return
    B = D[D.method.isin(BASE)]
    best = B.groupby(CELL).nrmse.min().rename("best"); gp = B[B.method == "gp"].set_index(CELL).nrmse.rename("gp")
    S = D[D.method.isin(SELF)].join(best, on=CELL).join(gp, on=CELL).dropna(subset=["best"])
    S["lr_best"] = np.log((S.nrmse + FL) / (S.best + FL)); S["lr_gp"] = np.log((S.nrmse + FL) / (S.gp + FL))
    out = [f"# E14 — self-designing surrogates vs learners on a Sobol' design ({len(D)} fits, "
           f"{D.problem.nunique()} problems)\n",
           "Ratio < 1: the self-designing method has lower test NRMSE. 'best baseline' = the best of gp, pce_lars, "
           "catboost, rbf_tps in the cell (an oracle choice, so it favours the baselines). Wilcoxon over problems.\n"]
    T = S.groupby("method").apply(lambda g: pd.Series(dict(
        vs_best=np.exp(g.lr_best.mean()), beats_best=(g.lr_best < 0).mean(), p_best=wil(g.lr_best, g.problem),
        vs_gp=np.exp(g.lr_gp.mean()), beats_gp=(g.lr_gp < 0).mean(), p_gp=wil(g.lr_gp, g.problem),
        n_used_share=(g.n_used / g.budget).median())), include_groups=False)
    out += ["## 1. Overall\n", md(T, ".3g"), ""]
    A = S.groupby(["struct", "method"]).lr_best.agg(lambda x: f"{np.exp(x.mean()):.2f} ({(x < 0).mean():.0%})")
    out += ["## 2. By problem structure: ratio to the best baseline (share of cells won)\n",
            md(A.unstack("method")), ""]
    C = S.groupby(["method", "budget", "level"]).lr_best.agg(
        ratio=lambda x: np.exp(x.mean()), wins=lambda x: (x < 0).mean())
    out += ["### By budget and noise\n", md(C, ".3g"), ""]
    W = D.groupby(["problem", "method"]).nrmse.apply(lambda x: np.exp(np.log(x + FL).mean())).unstack("method")
    out += ["## 3. Best method per problem (geometric mean over budgets, noise, seeds)\n",
            md(W.idxmin(axis=1).value_counts().to_frame("problems won")), "",
            md(W.assign(best=W.idxmin(axis=1)), ".3g"), ""]
    (ROOT / "results" / "E14_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
