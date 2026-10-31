"""E15 analysis — the surrogate–optimizer interface. Interim-safe: uses whatever records exist.

For each fitted surrogate the coupling strategies are paired (same fit, same budget), so comparisons are within task.
1. Success rate (regret < 1e-2) and geometric-mean regret per strategy, overall and per surrogate family.
2. Paired gain over the E9 baseline (mean + L-BFGS-B): geometric-mean regret ratio and share of tasks improved, per
   strategy × surrogate (does member sampling / smoothing / MILP / guarding beat optimizing the mean?).
3. The optimizer's curse: surrogate value minus true value at the pick (normalised), per strategy — negative means
   the surrogate promised more than it delivered; and the distance from the pick to the nearest training point.
4. Exact optimization of tree ensembles: MILP vs mean + L-BFGS-B / CMA-ES on the same tree model.

    python code/analysis/e15_analysis.py
Writes results/E15_report.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
from e3_learners import cfg_name, md  # noqa: E402

FLOOR, OK = 1e-8, 1e-2
FAMILY = {"gp": "GP", "hetgp": "GP", "rf": "trees", "xgb": "trees", "lgbm": "trees", "catboost": "trees",
          "gbt_committee": "trees", "mlp_ens": "networks", "mlp_rpf": "networks", "ens_press": "ensembles of surrogates",
          "ens_optw": "ensembles of surrogates"}


def load():
    rows, seen = [], set()
    for f in sorted((ROOT / "results" / "raw" / "E15_coupling").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in seen or r.get("status") != "ok" or not r.get("fstar_confirmed", True):
                seen.add(r["key"]); continue
            seen.add(r["key"]); c = r["cfg"]
            for s, v in r["strategies"].items():
                if "regret" in v:
                    rows.append(dict(task=r["key"], problem=c["problem"], surr=cfg_name(c),
                                     family=FAMILY.get(c["learner"], "other"), strategy=s,
                                     regret=max(v["regret"], FLOOR), curse=v["curse"], dist_data=v["dist_data"]))
    return pd.DataFrame(rows)


def main():
    D = load()
    if D.empty:
        print("no E15 records yet"); return
    out = [f"# E15 — the surrogate–optimizer interface ({D.task.nunique()} tasks, {D.problem.nunique()} problems)\n",
           f"Success = normalised regret < {OK:g}; geometric means floored at {FLOOR:g}.\n"]
    g = D.assign(succ=D.regret < OK, lr=np.log(D.regret))
    T = g.groupby(["strategy", "family"]).succ.mean().unstack("family")
    T["all"] = g.groupby("strategy").succ.mean()
    out += ["## 1. Success rate per strategy and surrogate family\n", md(T.sort_values("all", ascending=False), ".2f"), ""]
    base = D[D.strategy == "mean_lbfgsb"].set_index("task").regret
    D2 = D[D.strategy != "mean_lbfgsb"].copy()
    D2["lr"] = np.log(D2.regret.values / base.reindex(D2.task).values)
    G = D2.dropna(subset=["lr"]).groupby(["strategy", "family"]).lr.agg(
        ratio=lambda x: np.exp(x.mean()), improved=lambda x: (x < 0).mean(), n="size")
    out += ["## 2. Paired against mean + L-BFGS-B on the same fit (ratio < 1: the strategy finds lower true regret)\n",
            md(G, ".2f"), ""]
    C = g.groupby("strategy").agg(curse_median=("curse", "median"), share_overpromised=("curse", lambda x: (x < 0).mean()),
                                  dist_to_data=("dist_data", "median"))
    out += ["## 3. Optimizer's curse (surrogate − true value at the pick, ÷ std f; < 0 = promised more than "
            "delivered) and distance from the pick to the data\n", md(C, ".3f"), ""]
    tr = D[D.family == "trees"]
    if "milp_trees" in set(tr.strategy):
        P = tr.pivot_table(index="task", columns="strategy", values="regret")
        rows = {}
        for s in ("mean_lbfgsb", "mean_cmaes", "smoothed_adam", "cbo"):
            if s in P:
                q = P[["milp_trees", s]].dropna()
                rows[s] = dict(milp_over=np.exp(np.log(q.milp_trees / q[s]).mean()),
                               milp_better=(q.milp_trees < q[s]).mean(), tasks=len(q))
        out += ["## 4. Exact MILP on tree ensembles vs other strategies on the same tree model (< 1: MILP better)\n",
                md(pd.DataFrame(rows).T, ".2f"), ""]
    (ROOT / "results" / "E15_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
