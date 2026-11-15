"""E2b analysis: the survey-extension learners screened cell-for-cell against E2a.

E2b ran the learners added on 2026-10-11 (learners/survey.py) on exactly the E2a grid (same problems, n, noise and
seeds, so the designs, noise draws and test sets are identical). This ranks every E2a and E2b config together within
each problem × n × noise cell (mean NRMSE over seeds), as the E2a screen does. E2a/E2b are a fairness and smoke check
only (STUDY: never used for inference).
1. Each E2b config: mean rank among all configs, cells won (best or within the best's 95 % CI), ratio to the best
   E2a config in the cell, and failures.
2. Best E2b config per screening problem, with its ratio to the best E2a config.

    python code/analysis/e2b_learners.py
Writes results/E2b_learners_report.md.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
from e2a_learners import cfg_name  # noqa: E402
from e3_learners import md  # noqa: E402

CELL = ["problem", "n", "noise"]


def load():
    rows, status = [], Counter()
    for exp in ("E2a_learners", "E2b_learners"):
        seen = set()
        for f in sorted((ROOT / "results" / "raw" / exp).glob("*.jsonl")):
            for line in f.read_text(encoding="utf8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r["key"] in seen:
                    continue
                seen.add(r["key"]); c = r["cfg"]
                status[(exp, r.get("status"))] += 1
                rows.append(dict(src=exp[:3], problem=c["problem"], n=c["n"], noise=c["noise"], seed=c["seed"],
                                 cfg=cfg_name(c), ok=r.get("status") == "ok",
                                 nrmse=r.get("nrmse", np.nan) if r.get("status") == "ok" else np.nan))
    return pd.DataFrame(rows), status


def main():
    D, status = load()
    B = D[D.src == "E2b"]
    if B.empty:
        print("no E2b records yet"); return
    fail = B.groupby("cfg").ok.apply(lambda x: 1 - x.mean())
    O = D[D.ok]
    M = O.groupby(CELL + ["cfg", "src"]).nrmse.agg(["mean", "std", "count"]).reset_index()
    M = M[M["count"] >= 3]
    M["rank"] = M.groupby(CELL)["mean"].rank()
    best = M.loc[M.groupby(CELL)["mean"].idxmin()].set_index(CELL)
    hw = stats.t.ppf(0.975, best["count"] - 1) * best["std"] / np.sqrt(best["count"])
    M = M.join((best["mean"] + hw).rename("won_below"), on=CELL)
    a_best = M[M.src == "E2a"].groupby(CELL)["mean"].min().rename("e2a_best")
    M = M.join(a_best, on=CELL)
    Mb = M[M.src == "E2b"]
    T = Mb.groupby("cfg").apply(lambda g: pd.Series(dict(
        cells=len(g), mean_rank=g["rank"].mean(), of_configs=M.groupby(CELL).size().mean(),
        cells_won=int((g["mean"] <= g.won_below).sum()),
        ratio_to_best_E2a=np.exp(np.log(g["mean"] / g.e2a_best).mean()),
        beats_best_E2a=(g["mean"] < g.e2a_best).mean())), include_groups=False)
    T["failed_share"] = fail.reindex(T.index)
    out = ["# E2b — survey-extension learners on the E2a screening grid\n",
           f"Tasks: {dict(status)}. Ranks are among all E2a and E2b configs in each problem × n × noise cell "
           "(mean NRMSE over ≥ 3 seeds). Screening only, never used for inference.\n",
           "## 1. Each E2b config\n", md(T.sort_values("mean_rank"), ".3g"), ""]
    P = Mb.assign(lr=np.log(Mb["mean"] / Mb.e2a_best)).groupby(["problem", "cfg"]).lr.mean().unstack("cfg")
    out += ["## 2. Best E2b config per problem (ratio to the best E2a config, averaged over that problem's cells)\n",
            md(pd.DataFrame({"best E2b config": P.idxmin(axis=1), "ratio": np.exp(P.min(axis=1))}), ".3g"), ""]
    (ROOT / "results" / "E2b_learners_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
