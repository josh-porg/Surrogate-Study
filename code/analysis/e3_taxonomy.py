"""Data-driven taxonomy of methods and problems from E3 (n <= 1000, where every learner runs).

The pre-assigned labels (method "families", problem "categories") were a priori; this lets the results define them.
Profile of a method in a cell = log(NRMSE / best NRMSE in that cell) (0 = it is the best there).
1. Methods are clustered by the correlation of their profiles across cells (who wins and loses on the same cells);
   the clusters are compared with the nominal families (adjusted Rand index).
2. Problems are clustered by their mean method profile (which methods do well on them); the number of clusters is
   chosen by silhouette; the clusters are compared with the nominal categories.
3. Selection gap with data-driven problem types: clusters and per-cluster best methods are learned on seeds 0-4 and
   scored on held-out seeds 5-9, against single-best and the per-cell oracle.

    python code/analysis/e3_taxonomy.py
Writes results/E3_taxonomy.md and results/figures/E3_method_dendrogram.png, E3_problem_clusters.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
from e3_learners import load, md  # noqa: E402

KEY = ["problem", "n", "noise", "level"]
FAMCOL = {"GP family": "#2a78d6", "tree ensembles": "#eb6834", "networks": "#1baf7a", "kernel / RBF": "#eda100",
          "basis / regression": "#e87ba4", "other": "#5f5e58"}


def matrices(df, seeds):
    M = np.log(df[df.seed.isin(seeds)].groupby(KEY + ["cfg"]).nrmse.mean()).unstack("cfg")
    return M


def main():
    df, _ = load()
    df = df[df.n <= 1000]
    fam = df.drop_duplicates("cfg").set_index("cfg").fam
    cat = df.drop_duplicates("problem").set_index("problem").cat
    A, B = matrices(df, range(5)), matrices(df, range(5, 10))
    # drop cells still being computed (E3x runs slowest learners first, so new problems are partly filled) before
    # choosing methods; otherwise incomplete cells would knock whole methods out of the analysis
    full = (A.notna().mean(axis=1) > 0.9) & (B.notna().mean(axis=1) > 0.9)
    A, B = A[full.values], B[full.reindex(B.index).fillna(False).values] if len(B) == len(full) else B
    A, B = A.loc[A.index.intersection(B.index)], B.loc[A.index.intersection(B.index)]
    cols = A.columns[(A.notna().mean() > 0.9) & (B.notna().mean() > 0.9)]
    A, B = A[cols].dropna(), B[cols].dropna()
    idx = A.index.intersection(B.index); A, B = A.loc[idx], B.loc[idx]
    PA = A.sub(A.min(axis=1), axis=0).clip(upper=np.log(10))           # log ratio to best, capped at 10x so a few
    RA = A.rank(axis=1, pct=True)                                      # 1000x polynomial cells do not dominate
    out = [f"# E3 data-driven taxonomy ({len(idx)} cells, {len(cols)} methods, n <= 1000)\n",
           "Profile = log(NRMSE / best NRMSE in the cell). Everything is learned on seeds 0-4; selection results are "
           "scored on held-out seeds 5-9.\n"]

    # 1. method clusters
    C = RA.corr().clip(-1, 1)                                          # behaviour = within-cell rank
    Dm = squareform(1 - C.values, checks=False)
    Zm = linkage(Dm, "average")
    best_k, best_s = None, -1
    for k in range(3, 13):
        lab = fcluster(Zm, k, "maxclust")
        s = silhouette_score(1 - C.values, lab, metric="precomputed")
        if s > best_s:
            best_k, best_s = k, s
    mlab = pd.Series(fcluster(Zm, best_k, "maxclust"), index=C.index)
    ari_m = adjusted_rand_score(fam.reindex(C.index), mlab)
    out += [f"## 1. Method groups from behaviour ({best_k} groups, silhouette {best_s:.2f}; agreement with the "
            f"nominal families: adjusted Rand {ari_m:.2f}, where 1 = identical, 0 = chance)\n"]
    rows = []
    for g in sorted(mlab.unique()):
        mem = list(mlab[mlab == g].index)
        rows.append(dict(group=g, members=", ".join(mem), nominal_families=", ".join(sorted(set(fam[m] for m in mem)))))
    out += [md(pd.DataFrame(rows).set_index("group")), ""]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.2))
    dn = dendrogram(Zm, labels=list(C.index), ax=ax, leaf_font_size=7, color_threshold=None, above_threshold_color="#9e9d96")
    for t in ax.get_xticklabels():
        t.set_color(FAMCOL.get(fam.get(t.get_text()), "#1f1f1e"))
    ax.set_ylabel("1 - correlation of within-cell ranks", fontsize=8, color="#5f5e58")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=f) for f, c in FAMCOL.items()]
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(); (ROOT / "results" / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / "E3_method_dendrogram.png", dpi=200, facecolor="white")

    # 2. problem clusters from mean method profiles
    probs = PA.groupby(level="problem").mean()
    Zp = linkage(probs.values, "ward")
    best_kp, best_sp = None, -1
    for k in range(3, 16):
        lab = fcluster(Zp, k, "maxclust")
        s = silhouette_score(probs.values, lab)
        if s > best_sp:
            best_kp, best_sp = k, s
    plab = pd.Series(fcluster(Zp, best_kp, "maxclust"), index=probs.index)
    ari_p = adjusted_rand_score(cat.reindex(probs.index), plab)
    out += [f"## 2. Problem types from which methods win ({best_kp} types, silhouette {best_sp:.2f}; agreement with "
            f"the nominal categories: adjusted Rand {ari_p:.2f})\n"]
    rows = []
    dim = df.drop_duplicates("problem").set_index("problem").d
    for g in sorted(plab.unique()):
        mem = list(plab[plab == g].index)
        top = probs.loc[mem].mean().sort_values().head(4)
        dims = pd.Series([dim[m] for m in mem]).value_counts().sort_index()
        rows.append(dict(type=g, n_problems=len(mem), dims=" ".join(f"d{k}:{v}" for k, v in dims.items()),
                         best_methods=", ".join(f"{m} ({np.exp(v):.2f}x)" for m, v in top.items()),
                         problems=", ".join(mem)))
    out += ["Best methods are listed with their geometric-mean NRMSE relative to the per-cell best.\n",
            md(pd.DataFrame(rows).set_index("type")), ""]

    # 3. selection gap with data-driven types (types and picks from seeds 0-4; scored on 5-9)
    vbs = B.min(axis=1)
    prob_of = pd.Series(idx.get_level_values("problem"), index=idx)

    def score(choice):
        r = np.exp(np.array([B.loc[i, m] for i, m in choice.items()]) - vbs.values)
        return dict(geo_x_oracle=float(np.exp(np.log(r).mean())), within_1_1=float(np.mean(r <= 1.1)),
                    within_2=float(np.mean(r <= 2)), worst10=float(np.quantile(r, 0.9)))

    pol = {}
    sbs = A.mean().idxmin(); pol[f"single best ({sbs})"] = score(pd.Series(sbs, index=idx))
    nominal = {c: A.loc[cat.reindex(prob_of).values == c].mean().idxmin() for c in cat.unique()}
    pol[f"best per nominal category ({cat.nunique()})"] = score(pd.Series([nominal[cat[p]] for p in prob_of], index=idx))
    for k in sorted({best_kp, cat.nunique()}):
        lab_k = pd.Series(fcluster(Zp, k, "maxclust"), index=probs.index)
        dtype = {g: A.loc[lab_k.reindex(prob_of).values == g].mean().idxmin() for g in lab_k.unique()}
        pol[f"best per data-driven type ({k} types)"] = score(pd.Series([dtype[lab_k[p]] for p in prob_of], index=idx))
    pp = {p: A.loc[prob_of.values == p].mean().idxmin() for p in prob_of.unique()}
    pol["best per problem"] = score(pd.Series([pp[p] for p in prob_of], index=idx))
    pol["oracle per cell"] = score(B.idxmin(axis=1))
    out += ["## 3. Selection gap (held-out seeds 5-9)\n", md(pd.DataFrame(pol).T, ".2f"),
            f"\n{B.idxmin(axis=1).nunique()} of {len(cols)} methods are the best in at least one cell.\n"]
    (ROOT / "results" / "E3_taxonomy.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
