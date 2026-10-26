"""E12 analysis (H9): where rational surrogates and the AAA pairings win, and how they degrade with noise.
Interim-safe: uses whatever records exist.

Per cell (problem × n × noise level), mean NRMSE over seeds per config. Reports: geometric-mean NRMSE relative to the
cell's best, per config, split by problem group (rational structure: P-RAT + pole/knee probes; controls without it)
and by noise level; win counts; best rational config vs best reference learner per cell (share of cells the
rational side wins, and the geometric-mean ratio) by noise level — the accuracy-vs-noise curve of H9.

    python code/analysis/e12_analysis.py
Writes results/E12_report.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "code" / "analysis"))
from e3_learners import cfg_name, md  # noqa: E402

REFERENCE = {"gp", "as_gp", "gam", "ppr", "pce_lars", "mlp_ens", "xgb", "catboost"}
CONTROLS = {"smooth_trig_d2", "smooth_trig_d5", "additive_d5", "rotated_ridge_d5", "axis_step_d2", "borehole",
            "otlcircuit"}


def load():
    rows, seen = [], set()
    for f in sorted((ROOT / "results" / "raw" / "E12_rational").glob("*.jsonl")):
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
                rows.append(dict(problem=c["problem"], n=c["n"], level=c["level"], seed=c["seed"], cfg=cfg_name(c),
                                 learner=c["learner"], nrmse=r["nrmse"], grad_nrmse=r.get("grad_nrmse", np.nan)))
    return pd.DataFrame(rows), len(seen)


def main():
    df, n_rec = load()
    if df.empty:
        print("no E12 records yet"); return
    df["group"] = np.where(df.problem.isin(CONTROLS), "control", "rational structure")
    key = ["problem", "n", "level"]
    g = df.groupby(key + ["cfg"])
    cm = np.log(g.nrmse.mean()[g.size() >= 3]).unstack("cfg")
    R = cm.sub(cm.min(axis=1), axis=0).clip(upper=np.log(1000))      # capped at 1000x: spurious poles reach 1e14x
    grp = df.drop_duplicates("problem").set_index("problem").group
    lv = cm.index.get_level_values("level")
    gi = grp.reindex(cm.index.get_level_values("problem")).values
    out = [f"# E12 — rational surrogates ({n_rec} task records, {len(df)} ok fits, {len(cm)} cells)\n",
           "× best = geometric-mean NRMSE over the per-cell best config (1 = best everywhere); each cell capped at 1000×.\n"]
    cols = {}
    for G in ("rational structure", "control"):
        for L in sorted(set(lv)):
            m = (gi == G) & (lv == L)
            if m.any():
                cols[f"{G[:8]} {L:g}"] = np.exp(R[m].mean())
    T = pd.DataFrame(cols)
    T["cells"] = cm.notna().sum()
    T = T.sort_values(T.columns[0])
    out += ["## Each config, × best by problem group and noise level\n", md(T, ".2f"), ""]
    win = R.idxmin(axis=1)
    out += ["## Wins (cells where the config is best)\n",
            md(win.value_counts().to_frame("cells"), ".0f"), ""]
    lrn = df.drop_duplicates("cfg").set_index("cfg").learner
    rat = [c for c in cm.columns if lrn[c] not in REFERENCE]
    ref = [c for c in cm.columns if lrn[c] in REFERENCE]
    if rat and ref:
        diff = cm[rat].min(axis=1) - cm[ref].min(axis=1)
        tab = pd.DataFrame({"rational best wins (share)": (diff < 0).groupby([gi, lv]).mean(),
                            "best rational / best reference (geo)": np.exp(diff.groupby([gi, lv]).mean()),
                            "cells": diff.groupby([gi, lv]).size()})
        tab.index.names = ["group", "noise level"]
        out += ["## Best rational config vs best reference learner, per cell (< 1: rational more accurate)\n",
                md(tab.reset_index().set_index(["group", "noise level"]), ".2f"), ""]
    (ROOT / "results" / "E12_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
