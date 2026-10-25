"""E3 inference (STUDY_PLAN E3; M §9) — the statistics the paper leads with.

1. Family champions chosen on independent data: the best-ranked E3-shortlisted config of each family in E2a (mean
   within-cell rank, the E2a screening rule). Selecting on E2a and testing on E3 avoids the winner's-curse bias of
   taking each family's best on the same data (bigger families get more draws).
2. Unit of analysis = problem. Per cell (problem × n × noise × level, mean over seeds) the log NRMSE ratio of two
   champions; averaged per problem, so the 82 problems are the independent replicates (cells of one problem are not).
3. Pairwise Wilcoxon signed-rank on the per-problem log ratios, Holm-corrected over all champion pairs (as in the
   thesis, Poznański 2026, App. G; Benavoli, Corani & Mangili 2016 for why not mean-rank post-hoc tests).
4. Effect size: geometric-mean NRMSE ratio with a cluster bootstrap over problems (2 000 resamples, 95 % CI).
5. Bayesian signed-rank test with a region of practical equivalence (Benavoli et al. 2017, JMLR 18:77): ROPE =
   ratio within ±10 %; reports P(row better), P(equivalent), P(column better). Dirichlet-process posterior with prior
   weight 0.5 on a pseudo-observation at 0, 4 000 draws.
6. Performance profiles (Dolan & Moré 2002) of the champions: ρ(τ) = share of cells where the champion's NRMSE is
   within a factor τ of the best champion's, overall and by samples per dimension.

    python code/analysis/e3_stats.py [--exclude smt_krg,smt_kpls]
Writes results/E3_stats_report.md and results/figures/E3_performance_profiles.png.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
from e3_learners import FAM, cfg_name, load, md  # noqa: E402

FAMILIES = ["GP family", "tree ensembles", "networks", "kernel / RBF", "basis / regression", "other"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # dataviz reference slots 1-6 (light)
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]
ROPE = np.log(1.10)


def e2a_champions(allowed, exclude):
    """Best E2a mean-rank config per family among the E3 shortlist (`allowed`)."""
    rows, seen = [], set()
    for f in sorted((ROOT / "results" / "raw" / "E2a_learners").glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            r = json.loads(line)
            if r["key"] in seen or r["status"] != "ok":
                continue
            seen.add(r["key"]); c = r["cfg"]
            rows.append(dict(cell=(c["problem"], c["n"], c["noise"]), cfg=cfg_name(c), learner=c["learner"],
                             nrmse=r["nrmse"]))
    df = pd.DataFrame(rows)
    df = df[df.cfg.isin(allowed) & ~df.learner.isin(exclude)]
    cm = df.groupby(["cell", "cfg"]).nrmse.mean().unstack()
    mr = cm.rank(axis=1).mean()
    fam = df.drop_duplicates("cfg").set_index("cfg").learner.map(lambda L: FAM.get(L, "other"))
    out = {}
    for f in FAMILIES:
        c = [k for k in mr.index if fam[k] == f]
        if c:
            out[f] = min(c, key=lambda k: mr[k])
    return out, mr


def bayes_signed_rank(z, rope=ROPE, prior=0.5, draws=4000, seed=0):
    z = np.concatenate([[0.0], np.asarray(z, float)])
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.r_[prior, np.ones(len(z) - 1)], size=draws)            # (draws, n+1)
    s = (z[:, None] + z[None, :]) / 2
    L, E, R = (s < -rope).astype(float), (np.abs(s) <= rope).astype(float), (s > rope).astype(float)
    p = np.stack([np.einsum("ki,ij,kj->k", w, M, w) for M in (L, E, R)], 1)    # posterior mass per draw
    win = np.bincount(p.argmax(1), minlength=3) / draws                          # P(each region is most probable)
    return win


def main():
    exclude = set(sys.argv[sys.argv.index("--exclude") + 1].split(",")) if "--exclude" in sys.argv else set()
    df, n_rec = load()
    df = df[~df.learner.isin(exclude)]
    champ, e2a_rank = e2a_champions(set(df.cfg.unique()), exclude)
    names = {f: c for f, c in champ.items()}
    key = ["problem", "n", "noise", "level"]
    g = df[df.cfg.isin(names.values())].groupby(key + ["cfg"])
    cm = g.nrmse.mean()[g.size() >= 5].unstack("cfg")
    meta = df.drop_duplicates(key).set_index(key)[["d", "m", "cat"]]

    out = [f"# E3 inference — E2a-selected family champions ({n_rec} records"
           + (f"; excluded: {', '.join(sorted(exclude))}" if exclude else "") + ")\n",
           "Champions chosen on E2a (independent screening data), tested on E3. Unit of analysis: problem (per-problem "
           "mean of per-cell log NRMSE ratios). Ratios are row/column NRMSE: < 1 means the row champion is more "
           "accurate.\n", "## Champions\n",
           md(pd.DataFrame({"champion": pd.Series(names), "E2a mean rank": pd.Series(
               {f: e2a_rank[c] for f, c in names.items()})})), ""]

    fams = [f for f in FAMILIES if f in names]
    res = []
    for a, b in itertools.combinations(fams, 2):
        ca, cb = names[a], names[b]
        pair = cm[[ca, cb]].dropna()
        lr = np.log(pair[ca] / pair[cb])
        prob = lr.groupby(level="problem").mean()
        stat_p = stats.wilcoxon(prob).pvalue if len(prob) > 5 and np.any(prob != 0) else np.nan
        rng = np.random.default_rng(1); probs = prob.index.to_numpy(); per = lr.groupby(level="problem")
        sums, cnts = per.sum(), per.size()
        boots = []
        for _ in range(2000):
            pick = rng.choice(probs, len(probs))
            boots.append(sums[pick].sum() / cnts[pick].sum())
        lo, hi = np.exp(np.percentile(boots, [2.5, 97.5]))
        pl, pe, pr = bayes_signed_rank(prob.values)
        res.append(dict(row=a, col=b, problems=len(prob), cells=len(lr), ratio=np.exp(lr.mean()), ci_lo=lo, ci_hi=hi,
                        p=stat_p, P_row_better=pl, P_equiv=pe, P_col_better=pr))
    R = pd.DataFrame(res)
    R["p_holm"] = _holm(R.p.fillna(1).values)
    R = R.drop(columns="p").set_index(["row", "col"])
    out += ["## Pairwise comparisons (Wilcoxon signed-rank, Holm; cluster-bootstrap CI; Bayesian signed-rank, "
            "ROPE ±10 %)\n", md(R, ".3g"), ""]

    # performance profiles on cells where every champion is present
    full = cm[[names[f] for f in fams]].dropna()
    ratio = full.div(full.min(axis=1), axis=0)
    taus = np.logspace(0, np.log10(20), 400)
    m_of = meta.loc[full.index].m.values
    groups = [("all cells", np.ones(len(full), bool)), ("m = 5–10", m_of <= 10),
              ("m = 20–50", (m_of >= 20) & (m_of <= 50)), ("m = 100–300", m_of >= 100)]
    prof = {}
    for title, mask in groups:
        prof[title] = {f: [(ratio[names[f]].values[mask] <= t).mean() for t in taus] for f in fams}
    tab = pd.DataFrame({f: {f"ρ({t:g})": (ratio[names[f]] <= t).mean() for t in (1, 1.1, 1.5, 2, 5)} for f in fams}).T
    out += [f"## Performance profiles ({len(full)} cells with every champion)\n",
            "ρ(τ) = share of cells within a factor τ of the best champion; ρ(1) = win share.\n", md(tab, ".2f"), ""]
    _plot(prof, taus, fams, names)
    out += _all_methods(df, key, meta)
    (ROOT / "results" / "E3_stats_report.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


def _all_methods(df, key, meta):
    """Method-level view (not family claims): every config, and the best method per problem category. Category
    winners are *selected on seeds 0-4 and scored on seeds 5-9* (split-sample), so the reported margins are not
    inflated by picking the best on the same data."""
    out = []
    g = df.groupby(key + ["cfg"])
    cm = g.nrmse.mean()[g.size() >= 5].unstack("cfg")
    band = pd.cut(cm.index.get_level_values("n"), [0, 1000, 10 ** 9]).astype(str)
    cnt = cm.notna().sum(axis=1)
    cm = cm[cnt.values == cnt.groupby(band).transform("max").values].dropna(axis=1, how="all")
    ratio = cm.div(cm.min(axis=1), axis=0)
    fam = df.drop_duplicates("cfg").set_index("cfg").fam
    tab = pd.DataFrame({"family": fam.reindex(cm.columns),
                        **{f"ρ({t:g})": (ratio <= t).sum() / ratio.notna().sum() for t in (1, 1.5, 2, 5)}})
    out += [f"## All methods: performance profiles ({len(cm)} complete cells, every config)\n",
            md(tab.sort_values("ρ(2)", ascending=False), ".2f"), ""]

    half = lambda s: df[df.seed.isin(s)].groupby(key + ["cfg"]).nrmse.mean().unstack("cfg")
    A, B = half(range(5)), half(range(5, 10))
    cat = meta.cat
    rows = []
    for c in sorted(cat.unique()):
        idx = [i for i in A.index if cat.get(i) == c and i[1] <= 1000]     # n <= 1000: every learner runs there
        la, lb = np.log(A.loc[idx]), np.log(B.loc[idx])
        sa, sb = la.mean(), lb.mean()                                    # mean log NRMSE per config in the category
        sa, sb = sa[la.notna().mean() > 0.9], sb[lb.notna().mean() > 0.9]  # configs present in (almost) every cell
        both = sa.index.intersection(sb.index); sa, sb = sa[both], sb[both]   # ... in both seed halves
        best_b = sb.idxmin()
        row = {"category": c, "cells": len(idx)}
        for label, members in [("overall", None), ("GP", "GP family"), ("tree", "tree ensembles"),
                               ("network", "networks")]:
            cand = sa.index if members is None else [k for k in sa.index if fam.get(k) == members]
            if not len(cand):
                continue
            pick = min(cand, key=lambda k: sa[k])
            row[f"{label} pick (seeds 0-4)"] = pick
            row[f"{label}: ×best on 5-9"] = float(np.exp(sb[pick] - sb[best_b]))
        row["best on seeds 5-9"] = best_b
        row["overall pick still best"] = row["overall pick (seeds 0-4)"] == best_b
        rows.append(row)
    T = pd.DataFrame(rows).set_index("category")
    out += ["## Best method per problem category — selected on seeds 0-4, scored on held-out seeds 5-9\n",
            "`×best` = geometric-mean NRMSE of the pick over that of the held-out best (1.00 = the pick is the best).\n",
            md(T, ".2f"), "",
            f"Overall pick confirmed as held-out best in {T['overall pick still best'].mean():.0%} of categories.\n"]
    return out


def _holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def _plot(prof, taus, fams, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#1f1f1e", "#5f5e58", "#e4e3dd"
    fig, axes = plt.subplots(1, len(prof), figsize=(3.1 * len(prof), 3.0), sharey=True, constrained_layout=True)
    for ax, (title, P) in zip(axes, prof.items()):
        for i, f in enumerate(fams):
            ax.plot(taus, P[f], color=COLORS[FAMILIES.index(f)], lw=2, ls=STYLES[FAMILIES.index(f)],
                    marker=MARKERS[FAMILIES.index(f)], markevery=[0, 133, 266, 399], ms=5,
                    label=f"{f} ({names[f]})")
        ax.set_xscale("log"); ax.set_xlim(1, 20); ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"]); ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=9, color=ink)
        ax.grid(True, color=grid, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelsize=8)
        ax.set_xlabel("τ  (NRMSE / best champion's)", fontsize=8, color=muted)
    axes[0].set_ylabel("share of cells within τ of best", fontsize=8, color=muted)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3,
               fontsize=7.5, frameon=False, labelcolor=ink)
    out = ROOT / "results" / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / "E3_performance_profiles.png", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
