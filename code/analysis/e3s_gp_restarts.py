"""E3s_gp_restarts analysis (BUGS N-14): how much of E3's GP results is ML-II optimizer failure?

The run refits the four sklearn GP-family learners (gp, gp_se, hetgp, as_gp) on E3's own cells (seeds 0-4) with 5
ML-II restarts instead of E3's 1. Each 5-restart fit is paired with its E3 twin on the same cell. Unit for tests =
problem (cells averaged in log space).
1. Paired gain: error ratio 5 restarts / 1 restart, share improved, share of fits rescued (> 2× better), fit time.
2. Distance to the per-cell best of all E3 methods, before and after.
3. Head-to-head with the best tree-ensemble config in the cell, before and after: does the GP-vs-tree margin move?
4. By problem category.

    python code/analysis/e3s_gp_restarts.py
Writes results/E3s_gp_restarts_report.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "code" / "analysis"))
from e3_learners import load, md  # noqa: E402
from e3s_tuning import CELL, load_tuned, problem_wilcoxon  # noqa: E402


def main():
    T = load_tuned("E3s_gp_restarts", extra={"restarts"})
    if T.empty:
        print("no E3s_gp_restarts records yet"); return
    E, _ = load()
    E = E[(E.src == "E3") & (E.seed < 5)]
    best = E.groupby(CELL).nrmse.min().rename("best")
    tree = E[E.fam == "tree ensembles"].groupby(CELL).nrmse.min().rename("tree_best")
    base = E.set_index(CELL + ["cfg"])[["nrmse", "t_fit"]]
    J = T.join(base, on=CELL + ["base"]).join(best, on=CELL).join(tree, on=CELL).dropna(subset=["nrmse", "nrmse_t"])
    fl = 1e-12
    J["lr"] = np.log((J.nrmse_t + fl) / (J.nrmse + fl))
    out = [f"# E3s_gp_restarts — 5 ML-II restarts vs E3's 1 ({len(J)} paired cells, {J.problem.nunique()} problems)\n",
           "Ratio < 1: the restarts lower the error. Wilcoxon over problems (cells averaged in log space).\n"]
    rows = {}
    for b, g in J.groupby("base"):
        rows[b] = dict(cells=len(g), ratio=np.exp(g.lr.mean()), improved=(g.lr < -1e-9).mean(),
                       rescued_2x=(g.lr < -np.log(2)).mean(), worse_10pct=(g.lr > np.log(1.1)).mean(),
                       fit_time_x=np.exp(np.log((g.t_fit_t + 1e-3) / (g.t_fit + 1e-3)).mean()),
                       p_wilcoxon=problem_wilcoxon(g.lr, g.problem))
    out += ["## 1. Paired gain from restarts\n", md(pd.DataFrame(rows).T, ".3g"), ""]
    rows = {}
    for b, g in J.groupby("base"):
        ref = np.minimum(g.best, g.nrmse_t)
        rows[b] = dict(x_best_before=np.exp(np.log(g.nrmse / ref).mean()),
                       x_best_after=np.exp(np.log(g.nrmse_t / ref).mean()),
                       best_in_cell_before=(g.nrmse <= ref + 1e-15).mean(),
                       best_in_cell_after=(g.nrmse_t <= ref + 1e-15).mean())
    out += ["## 2. Distance to the per-cell best (× best)\n", md(pd.DataFrame(rows).T, ".3g"), ""]
    rows = {}
    for b, g in J.dropna(subset=["tree_best"]).groupby("base"):
        rows[b] = dict(beats_best_tree_before=(g.nrmse < g.tree_best).mean(),
                       beats_best_tree_after=(g.nrmse_t < g.tree_best).mean(),
                       ratio_to_best_tree_before=np.exp(np.log(g.nrmse / g.tree_best).mean()),
                       ratio_to_best_tree_after=np.exp(np.log(g.nrmse_t / g.tree_best).mean()))
    out += ["## 3. Head-to-head with the best tree-ensemble config in the cell\n", md(pd.DataFrame(rows).T, ".3g"), ""]
    C = J.groupby(["cat", "base"]).lr.mean().unstack("base").apply(np.exp)
    out += ["## 4. Restart / E3 ratio by problem category\n", md(C, ".3g"), ""]
    (ROOT / "results" / "E3s_gp_restarts_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
