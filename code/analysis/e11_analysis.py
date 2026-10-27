"""E11 analysis (RQ8) on R-FEA: which surrogate wins on the real FEA data, how the free scalar model is best used,
and whether the synthetic-study picture transfers. Interim-safe: uses whatever records exist.

Metric: relative RMSE of the sized wall t_fem (the thesis metric), mean over the 30 paired splits.
1. Ranking per training size and use of the scalar model (direct / tmf bridge / aug feature); the scalar model alone.
2. Paired comparison of the three uses per learner (geometric-mean ratio over splits, Wilcoxon over splits).
3. Transfer: where each learner ranks on R-FEA vs its E3 overall rank (Spearman), and the E3 data-driven type pick.

    python code/analysis/e11_analysis.py
Writes results/E11_report.md.
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


def load():
    rows, seen = [], set()
    for f in sorted((ROOT / "results" / "raw" / "E11_fea").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            if r.get("status") == "ok":
                c = r["cfg"]
                rows.append(dict(cfg=cfg_name(c), use=c["use_lf"], n=c["n_train"] or "all", seed=c["seed"],
                                 rel=r["rel_rmse"], scalar=r["rel_rmse_scalar_only"]))
    return pd.DataFrame(rows)


def active_learning():
    """E11_al: relative RMSE and optimizer proxy per scout × deliverable × design size, against the random scout."""
    rows, seen = [], set()
    for f in sorted((ROOT / "results" / "raw" / "E11_al").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in seen or r.get("status") != "ok":
                continue
            seen.add(r["key"])
            for k, v in r["fits"].items():
                if isinstance(v, dict):
                    deliv, n = k.split("@")
                    rows.append(dict(scout=f"{r['cfg']['scout']}:{r['cfg']['rule']}", deliv=deliv, n=int(n),
                                     seed=r["cfg"]["seed"], **v))
    if not rows:
        return ["## Retrospective active learning: no records yet\n"]
    D = pd.DataFrame(rows)
    rnd = D[D.scout == "none:random"].set_index(["deliv", "n", "seed"]).rel_rmse
    D["vs_random"] = np.log(D.rel_rmse.values / rnd.reindex(pd.MultiIndex.from_frame(D[["deliv", "n", "seed"]])).values)
    T = np.exp(D.groupby(["scout", "n"]).vs_random.mean()).unstack("n")
    out = ["## Retrospective active learning on the FEA cache (relative RMSE vs the random scout, geometric mean over "
           "deliverables and splits; < 1 = better than random)\n", md(T, ".2f"), ""]
    P = D.groupby(["scout", "deliv"]).pick_pctile.mean().unstack("deliv")
    out += ["Optimizer proxy at all design sizes: mean true percentile of the run the surrogate would pick (0 = best):\n",
            md(P, ".3f"), ""]
    return out


def main():
    df = load()
    if df.empty:
        print("no E11 records yet"); return
    out = [f"# E11 — real FEA transfer ({len(df)} fits)\n",
           f"Relative RMSE of t_fem on the held-out 20 %. The free scalar model alone: "
           f"{df.groupby('seed').scalar.first().mean():.3f}.\n"]
    for n, g in df.groupby("n", sort=False):
        T = g.groupby(["cfg", "use"]).rel.mean().unstack("use")
        T["best use"] = T.idxmin(axis=1)
        T["best"] = T[[c for c in ("direct", "tmf", "aug") if c in T]].min(axis=1)
        out += [f"## n_train = {n}\n", md(T.sort_values("best").head(15), ".3f"), ""]
    # paired uses
    pv = df.pivot_table(index=["cfg", "n", "seed"], columns="use", values="rel")
    rows = []
    for a, b in (("tmf", "direct"), ("aug", "direct"), ("aug", "tmf")):
        if {a, b} <= set(pv.columns):
            lr = np.log(pv[a] / pv[b]).dropna()
            per = lr.groupby(level="cfg").mean()
            rows.append({"comparison": f"{a} / {b}", "geo ratio (all fits)": np.exp(lr.mean()),
                         "learners where row better": (per < 0).mean(),
                         "Wilcoxon p (per learner means)": stats.wilcoxon(per).pvalue if len(per) > 5 else np.nan})
    if rows:
        out += ["## Use of the scalar model, paired (< 1: the first is more accurate)\n",
                md(pd.DataFrame(rows).set_index("comparison"), ".3g"), ""]
    # transfer from E3
    e3 = ROOT / "results" / "E3_learners_report.md"
    try:
        from e3_learners import load as load3
        d3, _ = load3()
        d3 = d3[d3.n <= 1000]
        cm = np.log(d3.groupby(["problem", "n", "noise", "level", "cfg"]).nrmse.mean()).unstack("cfg")
        r3 = cm.sub(cm.min(axis=1), axis=0).mean()
        rf = np.log(df[df.n == "all"].groupby("cfg").rel.mean())
        j = pd.concat([r3.rename("E3"), rf.rename("R-FEA")], axis=1).dropna()
        rho = stats.spearmanr(j["E3"], j["R-FEA"]).statistic
        out += [f"## Transfer: E3 overall ranking vs R-FEA ranking (n = all, best use per learner not taken)\n",
                f"Spearman ρ = {rho:.2f} over {len(j)} learners.\n"]
    except Exception as ex:                                              # E3 raw data may be absent on other machines
        out += [f"(transfer section skipped: {ex})\n"]
    out += active_learning()
    (ROOT / "results" / "E11_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")         # reports use Greek letters; the Windows console is cp1252
    main()
