"""E13 analysis: are gradients worth their cost? Gradient-enhanced surrogates vs value-only ones at matched cost.

Design (runner.exp_E13_gradient): 24 problems with usable gradients × n_base ∈ {5d, 10d, 20d} × noise on values and
gradients ∈ {0, 1 %, 10 %} × 5 seeds. Gradient learners (gek, hermite_rbf, ge_pce, sobolev_mlp) see n_base points with
gradients; value-only learners (the same four without derivative data, plus gp and pce_lars) see cost_mult × n_base
points, cost_mult ∈ {1 (same points), 2 (an adjoint gradient ≈ one extra evaluation), d + 1 (finite differences)}.
Unit for tests = problem.
1. The information in a gradient: each method with vs without gradients on the SAME points (cost_mult 1).
2. Matched cost: each gradient learner vs the best value-only learner given 2× and (d+1)× as many points.
3. Where gradients pay: by noise level and by n_base, against the best value-only learner at 2× cost.
4. Gradient accuracy of the surrogates (grad_nrmse), with vs without gradient data.

    python code/analysis/e13_analysis.py
Writes results/E13_report.md.
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

CELL = ["problem", "n_base", "level", "seed"]
FL = 1e-10


def load():
    from surr.problems import get
    rows, seen, dmap = [], set(), {}
    for f in sorted((ROOT / "results" / "raw" / "E13_gradient").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in seen or r.get("status") != "ok":
                seen.add(r["key"]); continue
            seen.add(r["key"]); c = r["cfg"]
            if c["problem"] not in dmap:
                dmap[c["problem"]] = get(c["problem"]).d
            d = dmap[c["problem"]]
            rows.append(dict(problem=c["problem"], d=d, n_base=c["n_base"], m=c["n_base"] // d, level=c["level"],
                             seed=c["seed"], learner=c["learner"], grad=bool(c["use_grad"]),
                             cost=("1" if c["cost_mult"] == 1 else "2" if c["cost_mult"] == 2 else "d+1"),
                             nrmse=r["nrmse"], grad_nrmse=r.get("grad_nrmse", np.nan)))
    D = pd.DataFrame(rows)
    # a 2-D problem has d + 1 = 3; a cost_mult of 2 is then "2" and 3 is "d+1" — both kept
    return D


def wil(x, by):
    g = x.groupby(by).mean()
    return float(stats.wilcoxon(g).pvalue) if len(g) > 5 and (g != 0).any() else np.nan


def main():
    D = load()
    if D.empty:
        print("no E13 records yet"); return
    G = D[D.grad]; V = D[~D.grad]
    out = [f"# E13 — gradient-enhanced surrogates at matched cost ({len(D)} fits, {D.problem.nunique()} problems)\n",
           "Ratio < 1: the gradient-enhanced fit has lower test NRMSE. Wilcoxon over problems (cells averaged in log "
           "space). Value-only learners at cost 2 model an adjoint gradient (≈ one extra solve); at d + 1, finite "
           "differences.\n"]
    # 1. same points
    v1 = V[V.cost == "1"].set_index(CELL + ["learner"]).nrmse.rename("v")
    J = G.join(v1, on=CELL + ["learner"]).dropna(subset=["v"])
    J["lr"] = np.log((J.nrmse + FL) / (J.v + FL))
    R = {L: dict(cells=len(g), ratio=np.exp(g.lr.mean()), better=(g.lr < 0).mean(), p=wil(g.lr, g.problem))
         for L, g in J.groupby("learner")}
    out += ["## 1. The information in a gradient: the same method with vs without gradients on the same points\n",
            md(pd.DataFrame(R).T, ".3g"), ""]
    # 2. matched cost against the best value-only learner (chosen per cell: optimistic for value-only)
    R = {}
    for cost in ("2", "d+1"):
        vb = V[V.cost == cost].groupby(CELL).nrmse.min().rename("vbest")
        for L, g in G.groupby("learner"):
            h = g.join(vb, on=CELL).dropna(subset=["vbest"])
            lr = np.log((h.nrmse + FL) / (h.vbest + FL))
            R[(L, f"value-only at {cost}x")] = dict(cells=len(h), ratio=np.exp(lr.mean()), better=(lr < 0).mean(),
                                                    p=wil(lr, h.problem))
        gb = G.groupby(CELL).nrmse.min().rename("gbest")
        h = vb.to_frame().join(gb).dropna(); lr = np.log((h.gbest + FL) / (h.vbest + FL))
        R[("best gradient learner", f"value-only at {cost}x")] = dict(
            cells=len(h), ratio=np.exp(lr.mean()), better=(lr < 0).mean(),
            p=wil(pd.Series(lr.values, index=h.index), h.index.get_level_values("problem")))
    out += ["## 2. Matched cost: gradient learners vs the best value-only learner per cell (oracle choice among six "
            "value-only learners, so this favours value-only)\n", md(pd.DataFrame(R).T, ".3g"), ""]
    # 2b. the fair pairing: the SAME method without gradients, given 2x or (d+1)x as many points
    R = {}
    for cost in ("2", "d+1"):
        vs = V[V.cost == cost].set_index(CELL + ["learner"]).nrmse.rename("v")
        for L, g in G.groupby("learner"):
            h = g.join(vs, on=CELL + ["learner"]).dropna(subset=["v"])
            lr = np.log((h.nrmse + FL) / (h.v + FL))
            R[(L, f"same method, value-only at {cost}x")] = dict(
                cells=len(h), ratio_geo=np.exp(lr.mean()), ratio_median=np.exp(lr.median()), better=(lr < 0).mean(),
                p=wil(lr, h.problem))
    out += ["## 2b. Matched cost, same method: with gradients on n points vs without on 2n or (d+1)n points "
            "(the median is shown because exactly recovered polynomial problems dominate geometric means)\n",
            md(pd.DataFrame(R).T, ".3g"), ""]
    # 3. by noise and by n_base, best gradient learner vs best value-only at 2x
    vb = V[V.cost == "2"].groupby(CELL).nrmse.min().rename("vbest"); gb = G.groupby(CELL).nrmse.min().rename("gbest")
    H = vb.to_frame().join(gb).dropna().reset_index(); H["lr"] = np.log((H.gbest + FL) / (H.vbest + FL))
    H["m"] = H.n_base // H.problem.map(D.groupby("problem").d.first())
    S = H.groupby(["level", "m"]).lr.agg(ratio=lambda x: np.exp(x.mean()), better=lambda x: (x < 0).mean(), cells="size")
    out += ["## 3. Best gradient learner vs best value-only learner at 2× cost, by noise level and n_base/d\n",
            md(S, ".3g"), ""]
    P = H.groupby("problem").lr.mean().apply(np.exp).sort_values()
    out += ["Per problem (ratio, all noise levels and sizes):\n", md(P.to_frame("ratio"), ".3g"), ""]
    # 4. gradient accuracy
    A = D[D.cost == "1"].groupby(["learner", "grad"]).grad_nrmse.median().unstack("grad")
    A.columns = ["value-only (same points)" if not c else "with gradients" for c in A.columns]
    out += ["## 4. Median gradient NRMSE of the surrogate (lower is better)\n", md(A, ".3g"), ""]
    (ROOT / "results" / "E13_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
