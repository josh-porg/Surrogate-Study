"""E9b analysis: surrogate-managed optimization loops vs model-free baselines, at 10d / 20d / 50d true evaluations.

Design (runner.exp_E9b_managed): E9's problems × shifted instances 0-9 × noise {0, 1 %}; methods EGO, TuRBO, DYCORS,
trust-region model management, random and Sobol' (ego_bootstrap on d ≤ 5, instances 0-4). Regret is normalised by
std(f) as in E9; success = regret < 1e-2; instances with an unconfirmed f* are excluded. Unit for tests = problem.
1. Success and geometric-mean regret per method and budget.
2. Paired comparisons with Sobol' and between the managed loops (Wilcoxon over problems, regret ratios).
3. By dimension, noise level and problem category.
4. Loop cost (wall time of the method's own computation).
5. Bootstrapped-kriging EGO vs plain EGO on its subset.
6. Best method per problem (no universal winner?).
The comparison with E9's fit-once-then-optimize approach is section 7 of results/E9_report.md (e9_analysis.py).

    python code/analysis/e9b_analysis.py
Writes results/E9b_report.md.
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
INST = ["problem", "inst", "noise"]


def load():
    from surr.problems import REGISTRY, get
    best = {}
    for f in sorted((ROOT / "results" / "raw" / "E9b_managed").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] not in best or r.get("status") == "ok":
                best[r["key"]] = r
    rows, bad, unconf = [], 0, 0
    for r in best.values():
        c = r["cfg"]
        if r.get("status") != "ok":
            bad += 1; continue
        if not r.get("fstar_confirmed", True):
            unconf += 1; continue
        for k, reg in zip(r["checkpoints"], r["regret"]):
            rows.append(dict(problem=c["problem"], d=get(c["problem"]).d, cat=REGISTRY[c["problem"]].category,
                             inst=c["shift_seed"], noise=c["level"], method=c["method"], budget=k,
                             regret=max(reg, FLOOR), t_loop=r.get("t_loop", np.nan)))
    return pd.DataFrame(rows), bad, unconf


def wil(lr, by):
    g = lr.groupby(by).mean()
    return float(stats.wilcoxon(g).pvalue) if len(g) > 5 and (g != 0).any() else np.nan


def paired(D, a, b, budget):
    X = D[D.budget == budget].pivot_table(index=INST, columns="method", values="regret")
    if a not in X or b not in X:
        return None
    X = X[[a, b]].dropna()
    lr = np.log(X[a] / X[b])
    return dict(instances=len(X), regret_ratio=np.exp(lr.mean()), a_better=(lr < 0).mean(),
                p=wil(lr, X.index.get_level_values("problem")))


def main():
    D, bad, unconf = load()
    if D.empty:
        print("no E9b records yet"); return
    main_methods = [m for m in D.method.unique() if m != "ego_bootstrap"]
    out = [f"# E9b — surrogate-managed optimization loops ({D.groupby(INST + ['method']).ngroups} runs, "
           f"{D.problem.nunique()} problems; {bad} errors and {unconf} unconfirmed-f* runs dropped)\n",
           f"Regret normalised by std(f); success = regret < {OK:g}. Budgets in true evaluations ÷ d.\n"]
    M = D[D.method.isin(main_methods)]
    S = M.groupby(["method", "budget"]).regret.agg(success=lambda x: (x < OK).mean(),
                                                    geo_regret=lambda x: np.exp(np.log(x).mean())).unstack("budget")
    S.columns = [f"{a}@{b}d" for a, b in S.columns]
    out += ["## 1. Success and geometric-mean regret\n", md(S.sort_values(S.columns[2], ascending=False), ".3g"), ""]
    rows = {}
    for m in main_methods:
        if m != "sobol":
            for b in (10, 50):
                r = paired(D, m, "sobol", b)
                if r:
                    rows[(m, f"vs sobol @{b}d")] = r
    loops = [m for m in ("ego", "turbo", "dycors", "tr_model_mgmt") if m in main_methods]
    for i, a in enumerate(loops):
        for b2 in loops[i + 1:]:
            r = paired(D, a, b2, 50)
            if r:
                rows[(a, f"vs {b2} @50d")] = r
    out += ["## 2. Paired comparisons (regret ratio < 1: the first method is better; Wilcoxon over problems)\n",
            md(pd.DataFrame(rows).T, ".3g"), ""]
    F = M[M.budget == 50]
    for by, title in (("d", "dimension"), ("noise", "noise level"), ("cat", "problem category")):
        T = F.groupby([by, "method"]).regret.apply(lambda x: (x < OK).mean()).unstack("method")
        out += [f"## 3. Success at 50d by {title}\n", md(T, ".2f"), ""]
    C = M[M.budget == 50].drop_duplicates(INST + ["method"]).groupby(["method", "d"]).t_loop.median().unstack("d")
    out += ["## 4. Median loop wall time [s] (the method's own computation; true evaluations are cheap here)\n",
            md(C, ".3g"), ""]
    if "ego_bootstrap" in set(D.method):
        rows = {f"ego_bootstrap vs ego @{b}d": paired(D, "ego_bootstrap", "ego", b) for b in (10, 20, 50)}
        out += ["## 5. Bootstrapped-kriging EGO vs plain EGO (its subset: d ≤ 5, instances 0-4)\n",
                md(pd.DataFrame({k: v for k, v in rows.items() if v}).T, ".3g"), ""]
    W = F.groupby(["problem", "method"]).regret.apply(lambda x: np.exp(np.log(x).mean())).unstack("method")
    win = W.idxmin(axis=1)
    out += ["## 6. Best method per problem at 50d (geometric-mean regret)\n",
            md(win.value_counts().to_frame("problems won")), "",
            md(W.assign(best=win).sort_index(), ".3g"), ""]
    (ROOT / "results" / "E9b_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
