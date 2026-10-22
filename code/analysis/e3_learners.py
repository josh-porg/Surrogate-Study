"""E3 analysis — learner study (STUDY_PLAN E3, RQ1 / H1). Interim-safe: runs on whatever records exist.

Cell = problem × n × noise kind × level. Per cell: mean clean-truth NRMSE over seeds per config (≥ 5 seeds), rank among
the configs present. Reports overall mean rank by method family; best-of-family head-to-head (GP family vs tree
ensembles vs networks) by samples per dimension, noise and problem category; the thesis pair GP vs XGBoost as a
geometric-mean NRMSE ratio; gradient accuracy and calibration. Writes results/E3_learners_report.md.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw" / "E3_learners"
RAW_X = ROOT / "results" / "raw" / "E3x_learners"        # D-9 extension: the problems E3's core set left out
sys.path.insert(0, str(ROOT / "code"))

FAMILY = {
    "GP family": {"gp", "gp_se", "hetgp", "as_gp", "local_gp", "smt_krg", "smt_kpls", "treed_gp", "ok_variogram"},
    "tree ensembles": {"xgb", "lgbm", "catboost", "rf", "extratrees", "cart", "gbt_committee"},
    "networks": {"mlp_ens", "kan", "dkl"},
    "kernel / RBF": {"rbf_tps", "rbf_mq", "krr", "svr", "mls"},
    "basis / regression": {"pce_lars", "cs_fourier", "cs_cosine", "rsm2", "mars", "gmdh", "ppr"},
    "other": {"delaunay_linear", "lipschitz", "ens_press"},
}
FAM = {L: f for f, s in FAMILY.items() for L in s}
H2H = ("GP family", "tree ensembles", "networks")


def cfg_name(c):
    kw = c.get("kw") or {}
    return c["learner"] + ("[" + ",".join(f"{k}={v}" for k, v in sorted(kw.items())) + "]" if kw else "")


def load():
    from surr.problems import REGISTRY
    P = {n: p for n, p in REGISTRY.items() if "@s" not in n}       # E3 core + E3x extension problems
    rows, seen = [], set()
    for f in sorted(RAW.glob("*.jsonl")) + sorted(RAW_X.glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in seen or r["status"] != "ok":
                seen.add(r["key"]); continue
            seen.add(r["key"]); c = r["cfg"]; p = P[c["problem"]]
            rows.append(dict(problem=c["problem"], cat=p.category, d=p.d, n=c["n"], m=round(c["n"] / p.d),
                             noise=c["noise"], level=c["level"], seed=c["seed"], cfg=cfg_name(c),
                             fam=FAM.get(c["learner"], "other"), learner=c["learner"],
                             **{k: r.get(k, np.nan) for k in ("nrmse", "grad_nrmse", "cover95", "nlpd", "t_fit")}))
    return pd.DataFrame(rows), len(seen)


def md(df, floatfmt=".3g"):
    fmt = lambda v: format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    head = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    cols = [df[c].tolist() for c in df.columns]                          # per column, so ints stay ints
    lines += ["| " + " | ".join([str(i)] + [fmt(c[k]) for c in cols]) + " |" for k, i in enumerate(df.index)]
    return "\n".join(lines)


def main():
    df, n_rec = load()
    skip = set(sys.argv[sys.argv.index("--exclude") + 1].split(",")) if "--exclude" in sys.argv else set()
    df = df[~df.learner.isin(skip)]
    key = ["problem", "n", "noise", "level"]
    g = df.groupby(key + ["cfg"])
    cm = g.nrmse.mean()[g.size() >= 5].unstack("cfg")                    # cell × config mean NRMSE
    # complete cells only: every config eligible at that n (caps: smt_krg <= 500, O(n^3)/networks <= 1000) present,
    # so ranks and head-to-heads never compare a learner against a partial field
    band = pd.cut(cm.index.get_level_values("n"), [0, 500, 1000, 10 ** 9]).astype(str)
    cnt = cm.notna().sum(axis=1)
    n_all = len(cm)
    cm = cm[cnt.values == cnt.groupby(band).transform("max").values]
    rk = cm.rank(axis=1)
    fam_of = df.drop_duplicates("cfg").set_index("cfg").fam
    meta = df.drop_duplicates(key).set_index(key)[["cat", "d", "m"]].loc[cm.index]

    out = [f"# E3 — learner study ({n_rec} of 826 020 task records, {len(df)} ok fits; {len(cm)} complete cells of "
           f"{n_all})" + (f"; excluded: {', '.join(sorted(skip))}" if skip else "") + "\n",
           "Clean-truth NRMSE; cells = problem × n × noise × level, mean over ≥ 5 seeds. Ranks are within cell among "
           "the configs present (O(n³)/network learners stop at n = 1000, `smt_krg` at 500, so large-n cells have "
           "fewer entrants). Interim numbers move as the remaining (mostly small-n) tasks arrive.\n"]

    o = pd.DataFrame({"family": fam_of.reindex(cm.columns), "mean rank": rk.mean(), "cells": cm.notna().sum(),
                      "median NRMSE": cm.median(), "win share": (rk == 1).sum() / rk.notna().sum()})
    out += ["## Overall (mean rank within cell)\n", md(o.sort_values("mean rank").head(20)), ""]

    best = {f: cm[[c for c in cm.columns if fam_of[c] == f]].min(axis=1) for f in H2H}
    bf = pd.DataFrame(best).dropna()
    win = bf.idxmin(axis=1)
    mm = meta.loc[bf.index]
    out += [f"## Best-of-family head-to-head ({len(bf)} cells where all three families have an entrant)\n",
            "Share of cells each family's best config wins, by samples per dimension m = n/d:\n"]
    t = pd.crosstab(mm.m.values, win.values, normalize="index").reindex(columns=H2H).fillna(0)
    t["cells"] = pd.Series(mm.m.values).value_counts().sort_index().values
    out += [md(t, ".2f"), "", "By noise:\n"]
    nz = [f"{a} {b:g}" for a, b in zip(bf.index.get_level_values("noise"), bf.index.get_level_values("level"))]
    out += [md(pd.crosstab(np.array(nz), win.values, normalize="index").reindex(columns=H2H).fillna(0), ".2f"), "",
            "By problem category:\n",
            md(pd.crosstab(mm.cat.values, win.values, normalize="index").reindex(columns=H2H).fillna(0), ".2f"), ""]

    if {"gp", "xgb"} <= set(cm.columns):
        pair = cm[["gp", "xgb"]].dropna()
        lr = np.log(pair.xgb / pair.gp); pm = meta.loc[pair.index]
        tab = pd.DataFrame({"geo-mean NRMSE ratio xgb/gp": np.exp(lr.groupby(pm.m.values).mean()),
                            "GP better (share)": (lr > 0).groupby(pm.m.values).mean(),
                            "cells": lr.groupby(pm.m.values).size()})
        out += ["## Thesis pair: GP vs XGBoost (ratio > 1 means GP more accurate)\n", md(tab, ".2f"), "",
                "By noise level (additive only):\n"]
        add = pair.index.get_level_values("noise").isin(["none", "additive"])
        lv = pair.index.get_level_values("level")[add]
        out += [md(pd.DataFrame({"ratio": np.exp(lr[add].groupby(lv).mean()),
                                 "GP better": (lr[add] > 0).groupby(lv).mean()}), ".2f"), ""]

    s = df.groupby("cfg").agg(family=("fam", "first"), grad_nrmse=("grad_nrmse", "median"),
                              cover95=("cover95", "median"), nlpd=("nlpd", "median"), t_fit_s=("t_fit", "median"))
    out += ["## Gradient accuracy and calibration (medians over all fits; cover95 target 0.95)\n",
            md(s.sort_values("grad_nrmse")), ""]
    (ROOT / "results" / "E3_learners_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
