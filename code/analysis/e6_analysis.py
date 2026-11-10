"""E6 analysis: when does multi-fidelity modelling beat spending the budget on high-fidelity points?

Design (runner.exp_E6_mf): 27 bi-fidelity pairs (constructed linear / weak / nonlinear / shifted pairs on four bases, and
11 from the mf2 collection) × n_H ∈ {3d, 10d, 30d} × n_L / n_H ∈ {4, 16} × 10 seeds; eight MF methods, HF-only on the
same n_H points, and HF-only at matched cost (n_H + n_L / κ points). Unit for tests = pair (cells averaged in log space).
1. Each MF method vs HF-only on the same HF points (the value of the low-fidelity data, free) and vs HF-only at
   matched cost (the decision a user faces).
2. By pair family: where MF pays.
3. By the low/high correlation r: does r predict the gain?
4. No universal winner: best method per pair, and the oracle-vs-default gap.

    python code/analysis/e6_analysis.py
Writes results/E6_report.md.
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

CELL = ["pair", "nH", "ratio", "seed"]
FL = 1e-10


def family(name):
    return "mf2" if name.startswith("mf2_") else name.split("__")[0].replace("mf_", "")


def load():
    from surr.problems.mf import MF_REGISTRY
    best = {}
    for f in sorted((ROOT / "results" / "raw" / "E6_mf").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["key"] not in best or r.get("status") == "ok":
                best[r["key"]] = r
    rows, bad = [], 0
    for r in best.values():
        c = r["cfg"]
        if r.get("status") != "ok":
            bad += 1; continue
        p = MF_REGISTRY[c["mfproblem"]]
        rows.append(dict(pair=c["mfproblem"], fam=family(c["mfproblem"]), d=p.d, r_lh=float(p.r) if p.r is not None
                         else np.nan, nH=c["nH"], m=c["nH"] // p.d, ratio=c["ratio"], seed=c["seed"],
                         method=c["method"] + ("@matched" if c.get("matched") else ""), nrmse=r["nrmse"],
                         t_fit=r.get("t_fit", np.nan)))
    return pd.DataFrame(rows), bad


def wil(lr, by):
    g = lr.groupby(by).mean()
    return float(stats.wilcoxon(g).pvalue) if len(g) > 5 and (g != 0).any() else np.nan


def versus(D, ref):
    base = D[D.method == ref].set_index(CELL).nrmse.rename("ref")
    J = D[~D.method.str.startswith("hf_only")].join(base, on=CELL).dropna(subset=["ref"])
    J["lr"] = np.log((J.nrmse + FL) / (J.ref + FL))
    return J


def summ(J, by):
    return J.groupby(by).apply(lambda g: pd.Series(dict(
        cells=len(g), ratio=np.exp(g.lr.mean()), wins=(g.lr < 0).mean(), p=wil(g.lr, g.pair))), include_groups=False)


def main():
    D, bad = load()
    if D.empty:
        print("no E6 records yet"); return
    out = [f"# E6 — multi-fidelity methods vs high-fidelity only ({len(D)} fits, {D.pair.nunique()} pairs; "
           f"{bad} timeouts/errors dropped)\n",
           "Ratio < 1: the MF method has lower test NRMSE. 'same HF' = HF-only on the same n_H points (the low-fidelity "
           "data is free); 'matched' = HF-only given the low-fidelity budget as extra HF points (n_H + n_L / κ).\n"]
    S = versus(D, "hf_only"); M = versus(D, "hf_only@matched")
    T = pd.concat({"same HF": summ(S, "method"), "matched cost": summ(M, "method")}, axis=1)
    out += ["## 1. Each MF method vs HF-only\n", md(T.sort_values(("matched cost", "ratio")), ".3g"), ""]
    # best MF method per cell (oracle) at matched cost
    mb = M.groupby(CELL).agg(lr=("lr", "min"), fam=("fam", "first"), m=("m", "first"))
    out += [f"Oracle (best MF method per cell) vs matched HF-only: ratio {np.exp(mb.lr.mean()):.3g}, wins "
            f"{(mb.lr < 0).mean():.3g}.\n"]
    out += ["## 2. By pair family, matched cost (ratio; wins in brackets)\n"]
    F = M.groupby(["fam", "method"]).lr.agg(lambda x: f"{np.exp(x.mean()):.2f} ({(x < 0).mean():.0%})").unstack("method")
    out += [md(F), ""]
    out += ["### By n_H / d and n_L / n_H, matched cost (best MF method per cell)\n",
            md(mb.reset_index().groupby(["m", "ratio"]).lr.agg(
                ratio=lambda x: np.exp(x.mean()), wins=lambda x: (x < 0).mean(), cells="size"), ".3g"), ""]
    # 3. correlation
    P = M.groupby(["pair", "method"]).lr.mean().unstack("method")
    rr = D.groupby("pair").r_lh.first()
    rho = {m: stats.spearmanr(rr.loc[P.index], P[m], nan_policy="omit")[0] for m in P.columns}
    out += ["## 3. Does the low/high correlation r predict the gain? (Spearman between r and log ratio at matched "
            "cost, over pairs; negative = higher r, bigger gain)\n", md(pd.Series(rho).to_frame("spearman"), ".3g"), ""]
    # 4. winners
    W = D.groupby(["pair", "method"]).nrmse.apply(lambda x: np.exp(np.log(x + FL).mean())).unstack("method")
    win = W.idxmin(axis=1)
    out += ["## 4. Best method per pair (geometric-mean NRMSE over all cells)\n",
            md(win.value_counts().to_frame("pairs won")), "",
            md(pd.DataFrame({"best": win, "r": rr.loc[W.index], "best/matched-HF": W.min(axis=1) /
                             W["hf_only@matched"]}), ".3g"), ""]
    (ROOT / "results" / "E6_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
