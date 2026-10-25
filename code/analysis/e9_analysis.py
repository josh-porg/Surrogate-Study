"""E9 analysis (RQ5/RQ6, H6/H7): optimizing on the surrogate. Interim-safe: uses whatever records exist.

Regret = (f(x̂) − f*) / std(f) on the true function at the incumbent after 10d … 500d surrogate evaluations.
Success = regret < 1e-2 (1 % of the function's spread). Unit of analysis for tests = problem.
1. Overall: success rate and geometric-mean regret (floored at 1e-8) per surrogate (best optimizer per surrogate,
   and averaged over optimizers), at the full budget; the "truth" row is the optimizer-only baseline.
2. H6 — does accuracy predict optimization? Per (problem, instance, n, noise, optimizer): Spearman correlation
   between the surrogates' NRMSE ranking and their regret ranking; distribution per optimizer class.
3. Gradient source: surrogate's own gradient vs finite differences inside the budget, paired, per gradient method
   and per surrogate.
4. The design-only baseline: share of tasks where optimizing the surrogate beats simply taking the best training
   point (no optimizer).
5. Anytime: success rate vs evaluations for the truth and the three best surrogates.
6. Interaction: best surrogate × optimizer pair per problem; how many distinct pairs win.

    python code/analysis/e9_analysis.py
Writes results/E9_report.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
from e3_learners import cfg_name, md  # noqa: E402

LOCAL_GRAD = {"slsqp", "lbfgsb", "trust_constr", "ipopt_multistart", "basin_hopping"}
FIRST_ORDER = {"adam", "sgd_momentum", "bb_gd", "robbins_monro", "kiefer_wolfowitz", "spsa"}
LOCAL_DFO = {"nelder_mead", "cobyla"}
GLOBAL = {"direct", "differential_evolution", "cmaes", "pso"}
FLOOR, OK = 1e-8, 1e-2


def opt_class(o):
    o = o.split("|")[0]
    return ("local gradient" if o in LOCAL_GRAD else "first-order SA" if o in FIRST_ORDER else
            "local derivative-free" if o in LOCAL_DFO else "global")


def load():
    rows, tasks, seen = [], [], set()
    for f in sorted((ROOT / "results" / "raw" / "E9_optim").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in seen or r.get("status") != "ok":
                seen.add(r["key"]); continue
            seen.add(r["key"])
            c = r["cfg"]
            surr = "truth" if c["learner"] == "truth" else cfg_name(c)
            t = dict(problem=c["problem"], inst=c["shift_seed"], n=c.get("n", 0), noise=c.get("level", 0.0), surr=surr,
                     nrmse=r.get("nrmse", 0.0), design_best=r.get("design_best_regret", np.nan),
                     confirmed=r.get("fstar_confirmed", True))
            tasks.append(t)
            for opt, v in r["runs"].items():
                if "regret" not in v:
                    continue
                for k, (reg, dist) in enumerate(zip(v["regret"], v["dist"])):
                    rows.append({**t, "opt": opt, "cp": r["checkpoints"][k], "regret": max(reg, FLOOR), "dist": dist})
    return pd.DataFrame(rows), pd.DataFrame(tasks)


def main():
    df, T = load()
    if df.empty:
        print("no E9 records yet"); return
    df = df[df.confirmed]
    full = df[df.cp == df.cp.max()]
    out = [f"# E9 — optimization on the surrogate ({len(T)} tasks, {df.problem.nunique()} problems)\n",
           f"Regret normalised by std(f); success = regret < {OK:g}. Instances with an unconfirmed f* are excluded.\n"]

    # 1. per surrogate
    g = full.groupby(["surr", "opt"]).regret
    per = pd.DataFrame({"success": g.apply(lambda s: (s < OK).mean()), "geo_regret": np.exp(np.log(g.apply(list).apply(np.array).apply(np.mean) * 0 + 1)) })
    s1 = full.assign(succ=full.regret < OK, lr=np.log(full.regret)).groupby(["surr", "opt"]).agg(
        success=("succ", "mean"), lr=("lr", "mean")).reset_index()
    best = s1.sort_values("success", ascending=False).groupby("surr").head(1).set_index("surr")
    avg = s1.groupby("surr").agg(success_avg=("success", "mean"), geo_regret_avg=("lr", lambda x: np.exp(x.mean())))
    tab = avg.join(best[["opt", "success"]].rename(columns={"opt": "best optimizer", "success": "success_best"}))
    nr = T.groupby("surr").nrmse.median()
    tab["median NRMSE"] = nr
    out += ["## 1. Surrogates at the full budget (500d)\n", md(tab.sort_values("success_best", ascending=False), ".3g"), ""]

    # 2. H6
    sub = full[full.surr != "truth"]
    cors = []
    for key, grp in sub.groupby(["problem", "inst", "n", "noise", "opt"]):
        if grp.surr.nunique() >= 5 and grp.nrmse.nunique() > 1 and grp.regret.nunique() > 1:
            cors.append(dict(opt=key[-1], cls=opt_class(key[-1]),
                             rho=stats.spearmanr(grp.nrmse, grp.regret).statistic))
    C = pd.DataFrame(cors)
    if len(C):
        h6 = C.groupby("cls").rho.describe()[["count", "mean", "25%", "50%", "75%"]]
        out += ["## 2. H6: does surrogate accuracy predict optimization regret? (Spearman ρ of NRMSE vs regret "
                "rankings across surrogates, per problem × instance × n × noise × optimizer)\n", md(h6, ".2f"),
                f"\nOverall median ρ = {C.rho.median():.2f}; share of settings with ρ > 0.5: {(C.rho > 0.5).mean():.0%}.\n"]

    # 3. gradient source
    gr = full[full.opt.str.endswith(("|fd", "|model"))].copy()
    if len(gr):
        gr["method"] = gr.opt.str.split("|").str[0]; gr["src"] = gr.opt.str.split("|").str[1]
        pv = gr.pivot_table(index=["problem", "inst", "n", "noise", "surr", "method"], columns="src", values="regret")
        pv = pv.dropna()
        lr = np.log(pv["model"] / pv["fd"])
        t3 = pd.DataFrame({"model / fd regret (geo)": np.exp(lr.groupby(level="method").mean()),
                           "model better (share)": (lr < 0).groupby(level="method").mean(),
                           "pairs": lr.groupby(level="method").size()})
        out += ["## 3. Surrogate gradient vs finite differences (< 1: the surrogate's gradient gives lower regret)\n",
                md(t3, ".2f"), "",
                "By surrogate:\n", md(pd.DataFrame({"model / fd (geo)": np.exp(lr.groupby(level="surr").mean())}), ".2f"), ""]

    # 4. design-only baseline
    bestopt = full[full.surr != "truth"].groupby(["problem", "inst", "n", "noise", "surr"]).regret.min()
    db = T[T.surr != "truth"].set_index(["problem", "inst", "n", "noise", "surr"]).design_best.reindex(bestopt.index)
    out += [f"## 4. Optimizing the surrogate vs taking the best training point\n",
            f"Best optimizer on the surrogate beats the best training point in {(bestopt < db).mean():.0%} of tasks; "
            f"geometric-mean regret {np.exp(np.log(bestopt).mean()):.3g} vs {np.exp(np.log(np.maximum(db, FLOOR)).mean()):.3g}.\n"]

    # 5. anytime
    top = tab.drop(index="truth", errors="ignore").sort_values("success_best", ascending=False).index[:3].tolist()
    any_ = df[df.surr.isin(["truth"] + top)].assign(succ=lambda x: x.regret < OK).groupby(["surr", "cp"]).succ.mean().unstack("cp")
    out += ["## 5. Success rate vs evaluations (×d), averaged over optimizers\n", md(any_, ".2f"), ""]

    # 6. winning pairs per problem
    pr = full[full.surr != "truth"].assign(lr=lambda x: np.log(x.regret)).groupby(["problem", "surr", "opt"]).lr.mean()
    wins = pr.groupby(level="problem").idxmin().apply(lambda t: f"{t[1]} + {t[2]}")
    out += [f"## 6. Best surrogate + optimizer per problem ({wins.nunique()} distinct pairs over {len(wins)} problems)\n",
            md(wins.to_frame("best pair (geo-mean regret)"), ""), ""]
    (ROOT / "results" / "E9_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
