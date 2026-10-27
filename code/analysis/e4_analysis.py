"""E4 analysis (H3): which mechanisms move the tree-vs-GP comparison, and does debiasing the aircraft data change
which surrogate class wins. Interim-safe: uses whatever records exist.

Mechanisms (E4_mechanisms): per learner and transform, the geometric-mean NRMSE ratio to the same learner without the
transform (paired on problem × n × noise × seed); the catboost / gp ratio per transform (does a mechanism reproduce
"GBT > GP"?); GP hyper-parameter bound hits per transform.
R-AIR (E4_rair): per target and setting, mean NRMSE overall and macro-averaged over engine types; the rank
correlation of learner rankings between settings; the winner per setting and metric.

    python code/analysis/e4_analysis.py
Writes results/E4_report.md.
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


def _read(exp):
    rows, seen = [], set()
    for f in sorted((ROOT / "results" / "raw" / exp).glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            if r.get("status") == "ok":
                rows.append({**r["cfg"], "cfg": cfg_name(r["cfg"]),
                             **{k: v for k, v in r.items() if k not in ("cfg", "key", "status", "kind")}})
    return pd.DataFrame(rows)


def mechanisms():
    df = _read("E4_mechanisms")
    if df.empty:
        return ["## Mechanisms: no records yet\n"]
    key = ["problem", "n", "noise", "level", "seed", "cfg"]
    base = df[df.transform == "none"].set_index(key).nrmse
    out = [f"## Mechanism probes ({len(df)} fits)\n",
           "Geometric-mean NRMSE relative to the same learner without the transform (paired); > 1 = the transform hurts.\n"]
    rat = {}
    for tr, g in df[df.transform != "none"].groupby("transform"):
        g = g.set_index(key).nrmse
        j = base.reindex(g.index)
        ok = j.notna()
        lr = np.log(g[ok] / j[ok])
        rat[tr] = np.exp(lr.groupby(level="cfg").mean())
    T = pd.DataFrame(rat)
    out += [md(T, ".2f"), ""]
    cm = df.groupby(["transform", "problem", "n", "noise", "level", "seed", "cfg"]).nrmse.mean().unstack("cfg")
    if {"catboost", "gp"} <= set(cm.columns):
        lr = np.log(cm.catboost / cm.gp).dropna()
        tab = pd.DataFrame({"catboost / gp (geo mean)": np.exp(lr.groupby(level="transform").mean()),
                            "share catboost better": (lr < 0).groupby(level="transform").mean(),
                            "fits": lr.groupby(level="transform").size()})
        out += ["### Does a mechanism reproduce \"trees beat GPs\"? (< 1 = catboost more accurate)\n", md(tab, ".2f"), ""]
    if "hp_at_bound" in df:
        gp = df[df.learner.isin(["gp", "hetgp"])]
        tab = gp.groupby("transform").agg(any_hp_at_bound=("hp_at_bound", lambda s: (s > 0).mean()),
                                          ls_at_bound=("ls_at_bound", "mean"), noise_at_bound=("noise_at_bound", "mean"))
        out += ["### GP hyper-parameters at their bounds (share of fits / mean count)\n", md(tab, ".2f"), ""]
    return out


def rair():
    df = _read("E4_rair")
    if df.empty:
        return ["## R-AIR: no records yet\n"]
    out = [f"## R-AIR bias arm ({len(df)} fits; {df.seed.nunique()} paired splits)\n"]
    for t, g in df.groupby("target"):
        m = g.groupby(["cfg", "setting"])[["nrmse", "nrmse_macro"]].mean().unstack("setting")
        out += [f"### {t}\n"]
        rows = {}
        for metric in ("nrmse", "nrmse_macro"):
            M = m[metric]
            for st in M.columns:
                rows[(metric, st)] = dict(winner=M[st].idxmin(), best=M[st].min(),
                                          catboost=M[st].get("catboost", np.nan), gp=M[st].get("gp", np.nan))
        out += [md(pd.DataFrame(rows).T, ".3g"), ""]
        M = m["nrmse_macro"]
        if {"biased", "weighted", "balanced"} <= set(M.columns):
            rho = {f"{a} vs {b}": stats.spearmanr(M[a], M[b], nan_policy="omit").statistic
                   for a, b in (("biased", "weighted"), ("biased", "balanced"), ("weighted", "balanced"))}
            out += ["Spearman correlation of learner rankings (macro NRMSE): " +
                    ", ".join(f"{k} {v:.2f}" for k, v in rho.items()) + "\n"]
            top = M.rank()
            out += ["Top 8 by macro NRMSE per setting (rank):\n",
                    md(top.sort_values("biased").head(8), ".0f"), ""]
    return out


def main():
    out = ["# E4 — mechanisms and the R-AIR bias arm\n"] + mechanisms() + rair()
    (ROOT / "results" / "E4_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")         # reports use Greek letters; the Windows console is cp1252
    main()
