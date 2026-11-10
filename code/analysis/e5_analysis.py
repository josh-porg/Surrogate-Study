"""E5 analysis: does adaptive sampling beat a space-filling design, and does the scout have to match the deliverable?

Design (runner.exp_E5_sampling): 13 scouts (GP variance / IMSE / EIGF / MEPE, committee and ensemble variances,
LOLA-Voronoi, sequential maximin, random, Sobol' control) × 22 problems × noise {0, 1 %} × 10 seeds; inside each task
six deliverables (gp, catboost, rf, mlp_ens, rbf_tps, pce_lars) are fitted on the scout's design at 10d, 20d, 50d.
Unit for tests = problem (cells averaged in log space).
1. Each scout vs the Sobol' control, over all deliverables and sizes (thesis bug B-02: is a GP scout just a
   space-filling design?).
2. Scout × deliverable matrix (B-03: the thesis only paired a scout with a GP deliverable).
3. Matched pairs: a scout's own learner as the deliverable vs other deliverables.
4. By budget, and the design cost (t_design).

    python code/analysis/e5_analysis.py
Writes results/E5_report.md.
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

CELL = ["problem", "level", "seed", "deliv", "size"]
FL = 1e-10
OWN = {"gp": "gp", "rf": "rf", "mlp_ens": "mlp_ens[lbfgs]"}   # scouts whose own learner is also a deliverable


def load():
    rows, seen = [], {}
    for f in sorted((ROOT / "results" / "raw" / "E5_sampling").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                seen[r["key"]] = r
    for r in seen.values():
        c = r["cfg"]
        scout = c["rule"] if c["scout"] == "none" else f'{c["scout"]}:{c["rule"]}'
        for k, v in r["fits"].items():
            deliv, size = k.split("@")
            rows.append(dict(problem=c["problem"], level=c["level"], seed=c["seed"], scout=scout,
                             scout_learner=c["scout"], deliv=deliv, size=size, nrmse=v, t_design=r.get("t_design")))
    return pd.DataFrame(rows)


def wil(lr, by):
    g = lr.groupby(by).mean()
    return float(stats.wilcoxon(g).pvalue) if len(g) > 5 and (g != 0).any() else np.nan


def main():
    D = load()
    if D.empty:
        print("no E5 records yet"); return
    D = D[np.isfinite(D.nrmse)]
    ref = D[D.scout == "sobol"].set_index(CELL).nrmse.rename("ref")
    J = D[D.scout != "sobol"].join(ref, on=CELL).dropna(subset=["ref"])
    J["lr"] = np.log((J.nrmse + FL) / (J.ref + FL))
    out = [f"# E5 — adaptive sampling vs a Sobol' design ({D.groupby(['problem','level','seed','scout']).ngroups} "
           f"designs, {len(D)} deliverable fits, {D.problem.nunique()} problems)\n",
           "Ratio < 1: the scout's design gives a lower test NRMSE than the Sobol' design of the same size, for the "
           "same deliverable. Wilcoxon over problems.\n"]
    T = J.groupby("scout").apply(lambda g: pd.Series(dict(
        ratio=np.exp(g.lr.mean()), wins=(g.lr < 0).mean(), worse_10pct=(g.lr > np.log(1.1)).mean(),
        p=wil(g.lr, g.problem), t_design_s=g.t_design.median())), include_groups=False).sort_values("ratio")
    out += ["## 1. Each scout vs Sobol', all deliverables and sizes\n", md(T, ".3g"), ""]
    M = J.groupby(["scout", "deliv"]).lr.mean().apply(np.exp).unstack("deliv")
    out += ["## 2. Scout × deliverable (ratio to Sobol')\n", md(M.loc[T.index], ".3g"), ""]
    # 3. matched vs unmatched: for scouts with a learner that is also a deliverable
    rows = {}
    for s, own in OWN.items():
        g = J[J.scout_learner == s]
        if g.empty:
            continue
        rows[s] = dict(own_deliverable=np.exp(g[g.deliv == own].lr.mean()),
                       other_deliverables=np.exp(g[g.deliv != own].lr.mean()))
    out += ["## 3. Does the deliverable need to match the scout? (variance-rule scouts; ratio to Sobol')\n",
            md(pd.DataFrame(rows).T, ".3g"), ""]
    S = J.groupby(["scout", "size"]).lr.mean().apply(np.exp).unstack("size")
    out += ["## 4. By budget (ratio to Sobol')\n", md(S.loc[T.index], ".3g"), ""]
    N = J.groupby(["scout", "level"]).lr.mean().apply(np.exp).unstack("level")
    out += ["### By noise level\n", md(N.loc[T.index], ".3g"), ""]
    P = J.groupby(["problem", "scout"]).lr.mean().apply(np.exp).unstack("scout")
    out += ["### Best scout per problem (ratio to Sobol'; > 1 means Sobol' was best)\n",
            md(pd.DataFrame({"best": P.idxmin(axis=1), "ratio": P.min(axis=1)}), ".3g"), ""]
    (ROOT / "results" / "E5_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
