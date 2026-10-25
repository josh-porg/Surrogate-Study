"""Inference for the paper's main claim: no universal winner, but the winner is predictable from the problem.

Data: E3 + E3x, cells (problem × n × noise × level) at n <= 1000 where every method runs (and, separately, n <= 500
with smt_krg). Unit of analysis = problem: per-cell quantities are averaged per problem, so problems are the
independent replicates (cells of one problem are not).

1. Universality. Kendall's W of the method rankings across problems (1 = every problem ranks the methods the same
   way; 0 = no agreement), with Friedman's test and a problem bootstrap CI. W is then computed within each
   data-driven problem type: if the winner is predictable from the type, W rises inside types.
2. Selection policies, scored on held-out seeds 5-9 against the per-cell oracle (best method on seeds 5-9).
   Picks use seeds 0-4 only and are leave-one-problem-out (LOPO): a problem's own data never chooses its method,
   except "best per problem", which is the upper bound needing every method run on that problem. Data-driven types
   are re-clustered without the problem, and the problem is assigned to the nearest type centroid by its seeds 0-4
   profile. That assignment uses the problem's own method profile, so it is a ceiling for a feature-based selector
   (E10), not a deployable rule.
   Pairwise: Wilcoxon signed-rank on per-problem mean log ratios with Holm; cluster-bootstrap CI of the geometric-
   mean ratio; Bayesian signed-rank with ROPE ±10 % (Benavoli et al. 2017).
3. Is the grouping real? Permutation test: the held-out gain of per-type picks over the single best, against 500
   random partitions of the problems into groups of the same sizes (same LOPO procedure).
4. Benchmark dependence. Per-method mean log ratio to the best on the original E3 problems vs the E3x additions:
   Spearman correlation of the two rankings (bootstrap CI), and per-method Mann-Whitney tests (Holm).

    python code/analysis/e3_selection_stats.py
Writes results/E3_selection_stats.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
from e3_learners import load, md  # noqa: E402
from e3_stats import _holm, bayes_signed_rank  # noqa: E402

KEY = ["problem", "n", "noise", "level"]
CAP = np.log(10)
RNG = np.random.default_rng(7)


def prepare(df):
    half = lambda s: np.log(df[df.seed.isin(s)].groupby(KEY + ["cfg"]).nrmse.mean()).unstack("cfg")
    A, B, F = half(range(5)), half(range(5, 10)), half(range(10))
    idx = A.index.intersection(B.index)
    A, B, F = A.loc[idx], B.loc[idx], F.loc[idx]
    full = (A.notna().mean(axis=1) > 0.9) & (B.notna().mean(axis=1) > 0.9)
    A, B, F = A[full], B[full], F[full]
    cols = A.columns[(A.notna().mean() > 0.9) & (B.notna().mean() > 0.9)]
    A, B, F = A[cols].dropna(), B[cols].dropna(), F[cols].dropna()
    idx = A.index.intersection(B.index).intersection(F.index)
    return A.loc[idx], B.loc[idx], F.loc[idx]


def kendall_w(R):
    """R: problems × methods matrix of ranks (rows = raters)."""
    m, n = R.shape
    S = ((R.sum(axis=0) - m * (n + 1) / 2) ** 2).sum()
    return 12 * S / (m ** 2 * (n ** 3 - n))


def universality(F, types):
    prob = F.sub(F.min(axis=1), axis=0).groupby(level="problem").mean()     # per-problem mean log ratio to best
    R = prob.rank(axis=1).values
    m, n = R.shape
    W = kendall_w(R)
    chi2 = m * (n - 1) * W
    p = stats.chi2.sf(chi2, n - 1)
    boots = [kendall_w(R[RNG.integers(0, m, m)]) for _ in range(1000)]
    rows = [dict(group="all problems", problems=m, W=W, ci_lo=np.percentile(boots, 2.5),
                 ci_hi=np.percentile(boots, 97.5), friedman_p=p, top_method=prob.mean().idxmin())]
    for g in sorted(types.unique()):
        mem = types.index[types == g]
        Rg = prob.loc[mem].rank(axis=1).values
        bg = [kendall_w(Rg[RNG.integers(0, len(mem), len(mem))]) for _ in range(1000)]
        rows.append(dict(group=f"type {g}", problems=len(mem), W=kendall_w(Rg), ci_lo=np.percentile(bg, 2.5),
                         ci_hi=np.percentile(bg, 97.5),
                         friedman_p=stats.chi2.sf(len(mem) * (n - 1) * kendall_w(Rg), n - 1),
                         top_method=prob.loc[mem].mean().idxmin()))
    return pd.DataFrame(rows).set_index("group")


def profiles(A):
    return A.sub(A.min(axis=1), axis=0).clip(upper=CAP).groupby(level="problem").mean()


def lopo_scores(A, B, cat, ks):
    """Per-problem held-out mean log(NRMSE_pick / NRMSE_oracle) for each policy."""
    vbs = B.min(axis=1)
    loss = B.sub(vbs, axis=0)                                           # cell × method held-out log ratio
    prob_of = A.index.get_level_values("problem")
    P = profiles(A)
    probs = list(P.index)
    cell_mean = A.groupby(level="problem").mean()                        # problem × method, seeds 0-4
    ncell = A.groupby(level="problem").size()
    tot = cell_mean.mul(ncell, axis=0)                                   # sums, so groups pool cells as in the taxonomy

    def pick_of(group):
        return (tot.loc[group].sum() / ncell.loc[group].sum()).idxmin()

    def score(p, m):
        return float(loss.loc[prob_of == p, m].mean())

    out = {k: {} for k in ["single best", "nominal category", *[f"data-driven ({k} types)" for k in ks],
                           "best per problem"]}
    for p in probs:
        rest = [q for q in probs if q != p]
        out["single best"][p] = score(p, pick_of(rest))
        same = [q for q in rest if cat[q] == cat[p]]
        out["nominal category"][p] = score(p, pick_of(same) if same else pick_of(rest))
        Z = linkage(P.loc[rest].values, "ward")
        for k in ks:
            lab = pd.Series(fcluster(Z, k, "maxclust"), index=rest)
            cent = P.loc[rest].groupby(lab.values).mean()
            g = cent.index[np.argmin(((cent.values - P.loc[p].values) ** 2).sum(axis=1))]
            out[f"data-driven ({k} types)"][p] = score(p, pick_of(list(lab.index[lab == g])))
        out["best per problem"][p] = score(p, cell_mean.loc[p].idxmin())
    return pd.DataFrame(out), ncell


def policy_tables(S, ncell):
    w = ncell.loc[S.index]
    summ = pd.DataFrame({
        "geo × oracle (cell-weighted)": np.exp(S.mul(w, axis=0).sum() / w.sum()),
        "geo × oracle (per problem)": np.exp(S.mean()),
        "problems within 2×": (S <= np.log(2)).mean(),
        "worst problem decile ×": np.exp(S.quantile(0.9))})
    res = []
    cols = list(S.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            z = S[a] - S[b]
            p = stats.wilcoxon(z).pvalue if np.any(z != 0) else 1.0
            boots = [np.exp(z.values[RNG.integers(0, len(z), len(z))].mean()) for _ in range(2000)]
            pl, pe, pr = bayes_signed_rank(z.values)
            res.append(dict(row=a, col=b, ratio=np.exp(z.mean()), ci_lo=np.percentile(boots, 2.5),
                            ci_hi=np.percentile(boots, 97.5), p=p, P_row_better=pl, P_equiv=pe, P_col_better=pr))
    R = pd.DataFrame(res)
    R["p_holm"] = _holm(R.p.values)
    return summ, R.drop(columns="p").set_index(["row", "col"])


def permutation(A, B, k, n_perm=500):
    """Gain of per-group picks over single best for the data-driven grouping (LOPO) vs random groupings of the same
    sizes. Random groupings are scored the same way (pick = best of the other members of the problem's group)."""
    vbs = B.min(axis=1)
    loss = B.sub(vbs, axis=0).groupby(level="problem").mean()
    cell_mean = A.groupby(level="problem").mean(); ncell = A.groupby(level="problem").size()
    tot = cell_mean.mul(ncell, axis=0)
    P = profiles(A); probs = np.array(P.index)
    lab = pd.Series(fcluster(linkage(P.values, "ward"), k, "maxclust"), index=probs)

    def gain(labels):
        g = []
        for p in probs:
            mates = [q for q in probs if labels[q] == labels[p] and q != p]
            rest = [q for q in probs if q != p]
            pk = lambda grp: (tot.loc[grp].sum() / ncell.loc[grp].sum()).idxmin()
            g.append(loss.loc[p, pk(rest)] - loss.loc[p, pk(mates) if mates else pk(rest)])
        return float(np.mean(g))

    obs = gain(lab)
    null = []
    for _ in range(n_perm):
        null.append(gain(pd.Series(RNG.permutation(lab.values), index=probs)))
    null = np.array(null)
    return obs, null, (1 + (null >= obs).sum()) / (1 + n_perm)


def suites(F, src):
    prob = F.sub(F.min(axis=1), axis=0).clip(upper=CAP).groupby(level="problem").mean()
    a = prob.loc[[p for p in prob.index if src[p] == "E3"]]
    b = prob.loc[[p for p in prob.index if src[p] == "E3x"]]
    rho = stats.spearmanr(a.mean(), b.mean()).statistic
    boots = []
    for _ in range(1000):
        ia, ib = RNG.integers(0, len(a), len(a)), RNG.integers(0, len(b), len(b))
        boots.append(stats.spearmanr(a.iloc[ia].mean(), b.iloc[ib].mean()).statistic)
    rows = []
    for m in prob.columns:
        rows.append(dict(method=m, rank_E3=None, rank_E3x=None, x_best_E3=np.exp(a[m].mean()),
                         x_best_E3x=np.exp(b[m].mean()), p=stats.mannwhitneyu(a[m], b[m]).pvalue))
    T = pd.DataFrame(rows).set_index("method")
    T["rank_E3"] = T.x_best_E3.rank().astype(int); T["rank_E3x"] = T.x_best_E3x.rank().astype(int)
    T["p_holm"] = _holm(T.p.values)
    T["better on"] = np.where(T.p_holm < 0.05, np.where(T.x_best_E3 < T.x_best_E3x, "E3", "E3x"), "—")
    return len(a), len(b), rho, np.percentile(boots, [2.5, 97.5]), T.drop(columns="p").sort_values("rank_E3")


def section(df, label, cat, src, ks=(3, 26), perm=True):
    A, B, F = prepare(df)
    P = profiles(A)
    types = pd.Series(fcluster(linkage(P.values, "ward"), 3, "maxclust"), index=P.index)
    out = [f"# {label}: {len(F)} cells, {F.index.get_level_values('problem').nunique()} problems, {F.shape[1]} methods\n"]
    U = universality(F, types)
    out += ["## 1. Is there a universal ranking? Kendall's W across problems\n",
            "W = 1: every problem ranks the methods identically; W = 0: no agreement. Friedman p tests W = 0. "
            "Types = the 3 data-driven problem types (seeds 0-4 profiles).\n", md(U, ".3g"), ""]
    S, ncell = lopo_scores(A, B, cat, ks)
    summ, R = policy_tables(S, ncell)
    out += ["## 2. Selection policies (leave-one-problem-out picks on seeds 0-4; scored on held-out seeds 5-9)\n",
            "× oracle = geometric-mean NRMSE over the per-cell best on seeds 5-9.\n", md(summ, ".3g"), "",
            "Pairwise (ratio = row / column, < 1 means the row policy is closer to the oracle; Wilcoxon over "
            "problems with Holm; problem-bootstrap 95 % CI; Bayesian signed-rank, ROPE ±10 %):\n", md(R, ".3g"), ""]
    if perm:
        obs, null, p = permutation(A, B, 3)
        out += ["## 3. Are the data-driven types real structure? Permutation test (3 types, 500 random partitions of "
                "equal sizes, LOPO)\n",
                f"Held-out gain of per-type picks over the single best: ×{np.exp(obs):.2f} (log {obs:.3f}); random "
                f"groupings: median ×{np.exp(np.median(null)):.2f}, 95th percentile ×{np.exp(np.quantile(null, 0.95)):.2f}; "
                f"permutation p = {p:.3g}.\n"]
    na, nb, rho, ci, T = suites(F, src)
    out += [f"## 4. Does the benchmark suite decide the winner? Original E3 problems ({na}) vs E3x additions ({nb})\n",
            f"Spearman correlation of the two method rankings: {rho:.2f} (bootstrap 95 % CI {ci[0]:.2f}–{ci[1]:.2f}). "
            "× best = geometric-mean NRMSE over the per-cell best (capped at 10×); Mann-Whitney over problems, Holm.\n",
            md(T, ".3g"), ""]
    return out


def main():
    df, _ = load()
    cat = df.drop_duplicates("problem").set_index("problem").cat
    src = df.drop_duplicates("problem").set_index("problem").src
    out = section(df[df.n <= 1000], "n ≤ 1000 (all methods except smt_krg, which stops at n = 500)", cat, src)
    out += ["\n---\n"] + section(df[df.n <= 500], "n ≤ 500 (with smt_krg)", cat, src, perm=False)
    (ROOT / "results" / "E3_selection_stats.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
