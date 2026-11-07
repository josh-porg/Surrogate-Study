"""E3s_tuning analysis: does a larger tuning budget change the tree / network results of E3?

The run re-fits, on E3's own cells (seeds 0-4), XGBoost and CatBoost with a 30-trial random search (3-fold CV) and the
L-BFGS MLP ensemble with 10 members instead of 5. Each tuned fit is paired with its E3 twin (same problem, n, noise,
level, seed), so every comparison is within cell. Unit for tests = problem (cells averaged in log space).
1. Paired gain: geometric-mean NRMSE ratio tuned / E3 config, share of cells improved, fit-time ratio, Wilcoxon over
   problems.
2. Distance to the per-cell best of all E3 methods (× best), before and after tuning.
3. Head-to-head with the GP family: share of cells where the learner beats the best GP-family config, before/after.
4. By problem category: where the tuning helps.

    python code/analysis/e3s_tuning.py
Writes results/E3s_tuning_report.md.
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
from e3_learners import FAM, cfg_name, load, md  # noqa: E402

CELL = ["problem", "n", "noise", "level", "seed"]
EXTRA = {"tune", "K"}                         # the keys the tuning run adds to the E3 config


def load_tuned():
    from surr.problems import REGISTRY
    rows, seen = [], set()
    for f in sorted((ROOT / "results" / "raw" / "E3s_tuning").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in seen or r.get("status") != "ok":
                seen.add(r["key"]); continue
            seen.add(r["key"]); c = r["cfg"]
            base = {**c, "kw": {k: v for k, v in (c.get("kw") or {}).items() if k not in EXTRA}}
            rows.append(dict(problem=c["problem"], n=c["n"], noise=c["noise"], level=c["level"], seed=c["seed"],
                             cat=REGISTRY[c["problem"]].category, base=cfg_name(base), tuned=cfg_name(c),
                             nrmse_t=r["nrmse"], t_fit_t=r.get("t_fit", np.nan)))
    return pd.DataFrame(rows)


def problem_wilcoxon(x, by):
    g = x.groupby(by).mean()
    return float(stats.wilcoxon(g).pvalue) if len(g) > 5 and (g != 0).any() else np.nan


def main():
    T = load_tuned()
    if T.empty:
        print("no E3s_tuning records yet"); return
    E, _ = load()
    E = E[(E.src == "E3") & (E.seed < 5)]
    best = E.groupby(CELL).nrmse.min().rename("best")
    gpbest = E[E.fam == "GP family"].groupby(CELL).nrmse.min().rename("gp_best")
    base = E.set_index(CELL + ["cfg"])[["nrmse", "t_fit"]]
    J = T.join(base, on=CELL + ["base"]).join(best, on=CELL).join(gpbest, on=CELL).dropna(subset=["nrmse", "nrmse_t"])
    fl = 1e-12
    J["lr"] = np.log((J.nrmse_t + fl) / (J.nrmse + fl))
    out = [f"# E3s_tuning — a larger tuning budget for trees and networks ({len(J)} paired cells, "
           f"{J.problem.nunique()} problems)\n",
           "Pairs: each tuned fit against its E3 twin on the same cell. Ratio < 1: tuning lowers the error. Wilcoxon over "
           "problems (cells averaged in log space).\n"]
    rows = {}
    for b, g in J.groupby("base"):
        rows[f"{b} → {g.tuned.iloc[0]}"] = dict(
            cells=len(g), ratio=np.exp(g.lr.mean()), improved=(g.lr < 0).mean(),
            worse_by_10pct=(g.lr > np.log(1.1)).mean(), fit_time_x=np.exp(np.log((g.t_fit_t + 1e-3) / (g.t_fit + 1e-3)).mean()),
            p_wilcoxon=problem_wilcoxon(g.lr, g.problem))
    out += ["## 1. Paired gain from tuning\n", md(pd.DataFrame(rows).T, ".3g"), ""]
    rows = {}
    for b, g in J.groupby("base"):
        ref = np.minimum(g.best, g.nrmse_t)            # one reference for both: the cell's best including the tuned fit
        xb, xt = np.log(g.nrmse / ref), np.log(g.nrmse_t / ref)
        rows[b] = dict(x_best_before=np.exp(xb.mean()), x_best_after=np.exp(xt.mean()),
                       best_in_cell_before=(g.nrmse <= ref + 1e-15).mean(),
                       best_in_cell_after=(g.nrmse_t <= ref + 1e-15).mean())
    out += ["## 2. Distance to the per-cell best (all E3 methods plus the tuned fit; × best)\n",
            md(pd.DataFrame(rows).T, ".3g"), ""]
    rows = {}
    for b, g in J.dropna(subset=["gp_best"]).groupby("base"):
        rows[b] = dict(beats_best_GP_before=(g.nrmse < g.gp_best).mean(), beats_best_GP_after=(g.nrmse_t < g.gp_best).mean(),
                       ratio_to_best_GP_before=np.exp(np.log(g.nrmse / g.gp_best).mean()),
                       ratio_to_best_GP_after=np.exp(np.log(g.nrmse_t / g.gp_best).mean()))
    out += ["## 3. Head-to-head with the best GP-family config in the same cell\n", md(pd.DataFrame(rows).T, ".3g"), ""]
    C = J.groupby(["cat", "base"]).lr.mean().unstack("base").apply(np.exp)
    out += ["## 4. Tuned / E3 ratio by problem category\n", md(C, ".3g"), ""]
    (ROOT / "results" / "E3s_tuning_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
