"""E10 prototype — choose the surrogate from the training data alone (RQ7), on the E3 + E3x results.

A practitioner has only the samples (X, y), d and n before choosing; noise kind, level and the problem's category are
unknown. Features are computed from exactly the training set each E3 fit saw (design, noise draw and seed rebuilt as
in runner.task_learner), so they carry no information the practitioner would lack:
  size        d, n, log(n/d)
  y shape     skewness, kurtosis, share of tied values (plateaus)
  fits (5-fold CV R², so noise and over-fitting count against them)
              linear, quadratic (ridge), 5-NN, random forest (30 trees), and the gaps quadratic − linear,
              forest − quadratic (trees beat polynomials → rugged), 5-NN − quadratic
  roughness   mean |y_i − y_nn(i)| / std(y) over nearest neighbours; nearest-better ratio (multimodality: mean distance
              to the nearest better point / mean nearest-neighbour distance); top-10 % dispersion ratio (ELA)
  anisotropy  share of the largest |β| in the linear fit; number of inputs carrying 90 % of Σβ²
Selector: an extra-trees regressor predicting every method's log NRMSE ratio to the cell's best (seeds 0-4); for a
held-out cell each of its seeds 5-9 datasets gets a pick (argmin), the cell takes the majority pick, scored on
seeds 5-9 against the per-cell oracle — the same scoring as code/analysis/e3_selection_stats.py.
Validation: leave-problem-out (20 grouped folds), leave-category-out (nominal categories as groups), and
cross-suite (train on E3 problems → test on E3x additions, and back).

    python code/analysis/e10_selector.py --features [--workers 8]    feature cache, resumable (results/E10_features.csv)
    python code/analysis/e10_selector.py                             selector evaluation → results/E10_selector.md
"""
from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "code" / "analysis"))
FEAT = ROOT / "results" / "E10_features.csv"
KEYCOLS = ["problem", "n", "noise", "level", "seed"]


def features(problem, n, noise, level, seed):
    import warnings
    warnings.filterwarnings("ignore")
    from scipy import stats
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from surr import noise as N, problems as P, samplers as S
    p = P.get(problem); d = p.d
    U = S.make("sobol", n, d, seed=seed)                                 # as runner.task_learner (sampler "sobol")
    T = S.sobol(4096, d, seed=10_000 + seed)
    scale = float(p.f_unit(T).std())
    y = N.apply(noise, level, p.f_unit(U), U, scale, seed=20_000 + seed)
    z = (y - y.mean()) / (y.std() + 1e-300)
    kf = KFold(5, shuffle=True, random_state=0)
    r2 = lambda pred: float(1 - np.mean((pred - z) ** 2) / np.mean(z ** 2))
    f = dict(problem=problem, n=n, noise=noise, level=level, seed=seed, d=d, log_m=float(np.log(n / d)),
             skew=float(stats.skew(z)), kurt=float(stats.kurtosis(z)),
             ties=float(1 - len(np.unique(np.round(z, 9))) / len(z)))
    f["r2_lin"] = r2(cross_val_predict(LinearRegression(), U, z, cv=kf))
    f["r2_quad"] = r2(cross_val_predict(make_pipeline(PolynomialFeatures(2), Ridge(1e-3)), U, z, cv=kf))
    f["r2_knn"] = r2(cross_val_predict(KNeighborsRegressor(5), U, z, cv=kf))
    f["r2_rf"] = r2(cross_val_predict(RandomForestRegressor(30, random_state=0, n_jobs=1), U, z, cv=kf))
    f["gap_quad_lin"], f["gap_rf_quad"], f["gap_knn_quad"] = (f["r2_quad"] - f["r2_lin"], f["r2_rf"] - f["r2_quad"],
                                                              f["r2_knn"] - f["r2_quad"])
    nn = NearestNeighbors(n_neighbors=2).fit(U)
    dist, idx = nn.kneighbors(U)
    f["nn_dy"] = float(np.mean(np.abs(z - z[idx[:, 1]])))
    D = np.sqrt(((U[:, None] - U[None]) ** 2).sum(-1)) if n <= 1000 else None
    better = z[None, :] < z[:, None]                                     # [i, j]: j better than i
    Db = np.where(better, D, np.inf).min(axis=1)
    ok = np.isfinite(Db)
    f["nbc_ratio"] = float(np.mean(Db[ok]) / (np.mean(dist[:, 1]) + 1e-300))
    top = z <= np.quantile(z, 0.1)
    iu = np.triu_indices(top.sum(), 1)
    f["disp_top10"] = float(D[np.ix_(top, top)][iu].mean() / D[np.triu_indices(n, 1)].mean()) if top.sum() > 2 else 1.0
    b = np.abs(LinearRegression().fit(U, z).coef_)
    s = np.sort(b ** 2)[::-1]
    f["aniso_top"] = float(b.max() / (b.sum() + 1e-300))
    f["eff_dim90"] = int(np.searchsorted(np.cumsum(s) / (s.sum() + 1e-300), 0.9) + 1)
    return f


def build(workers):
    import pandas as pd
    from e3_learners import load
    df, _ = load()
    df = df[df.n <= 1000]
    want = df[KEYCOLS].drop_duplicates()
    have = set()
    if FEAT.exists():
        old = pd.read_csv(FEAT)
        have = set(map(tuple, old[KEYCOLS].astype(str).values))
    todo = [r for r in want.itertuples(index=False) if tuple(map(str, r)) not in have]
    print(f"E10 features: {len(want)} datasets, {len(have)} cached, {len(todo)} to compute", flush=True)
    new = not FEAT.exists()
    with FEAT.open("a", newline="", encoding="utf8") as fh, ProcessPoolExecutor(workers) as ex:
        w = None
        futs = [ex.submit(features, *r) for r in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                row = fu.result()
            except Exception as e:
                print("  failed:", repr(e)[:200], flush=True); continue
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
            w.writerow(row); fh.flush()
            if k % 500 == 0:
                print(f"  {k}/{len(todo)}", flush=True)


def evaluate():
    import pandas as pd
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import GroupKFold
    from e3_learners import load, md
    from e3_selection_stats import lopo_scores, policy_tables, prepare
    df, _ = load()
    df = df[df.n <= 1000]
    cat = df.drop_duplicates("problem").set_index("problem").cat
    src = df.drop_duplicates("problem").set_index("problem").src
    A, B, _ = prepare(df)
    methods = list(A.columns)
    F = pd.read_csv(FEAT)
    F["level"] = F.level.astype(float)
    ck = ["problem", "n", "noise", "level"]
    F = F.set_index(ck)
    cells = A.index
    F = F[F.index.isin(cells)]
    xcols = [c for c in F.columns if c not in ("seed",)]
    R = A.sub(A.min(axis=1), axis=0).clip(upper=np.log(10))            # targets: log ratio to cell best (seeds 0-4)
    tr = F[F.seed < 5]; te = F[F.seed >= 5]
    Ytr = R.loc[tr.index].values
    vbs = B.min(axis=1)

    def run(groups_of, folds):
        """Per-cell held-out log loss of the selector's majority pick."""
        g_tr = np.array([groups_of(i) for i in tr.index]); g_te = np.array([groups_of(i) for i in te.index])
        loss = pd.Series(np.nan, index=cells)
        for train_groups, test_groups in folds(np.unique(g_tr)):
            mtr = np.isin(g_tr, train_groups); mte = np.isin(g_te, test_groups)
            if not mte.any():
                continue
            mdl = ExtraTreesRegressor(300, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0)
            mdl.fit(tr[xcols].values[mtr], Ytr[mtr])
            pred = pd.DataFrame(mdl.predict(te[xcols].values[mte]), index=te.index[mte], columns=methods)
            picks = pred.idxmin(axis=1)
            score = pred.groupby(level=ck).sum()
            for cell, grp in picks.groupby(level=ck):
                vc = grp.value_counts()
                top = list(vc.index[vc == vc.max()])
                pick = min(top, key=lambda m: score.loc[cell, m])
                loss[cell] = B.loc[cell, pick] - vbs[cell]
        return loss

    def kfold(k):
        def f(groups):
            gk = GroupKFold(min(k, len(groups)))
            for a, b in gk.split(groups, groups=groups):
                yield groups[a], groups[b]
        return f

    def one_out(groups):
        for g in groups:
            yield np.array([x for x in groups if x != g]), np.array([g])

    def suites(groups):
        yield np.array(["E3"]), np.array(["E3x"])
        yield np.array(["E3x"]), np.array(["E3"])

    prob = lambda i: i[0]
    variants = {"selector, leave-problem-out (20 folds)": run(prob, kfold(20)),
                "selector, leave-category-out": run(lambda i: cat[i[0]], one_out),
                "selector, cross-suite (E3 ↔ E3x)": run(lambda i: src[i[0]], suites)}
    S, ncell = lopo_scores(A, B, cat, (26,))
    prob_of = cells.get_level_values("problem")
    for name, loss in variants.items():
        S[name] = loss.groupby(prob_of).mean().reindex(S.index)
    S = S.dropna()
    summ, Rt = policy_tables(S, ncell)
    # feature importances from one fit on all training rows
    mdl = ExtraTreesRegressor(300, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0).fit(tr[xcols].values, Ytr)
    imp = pd.Series(mdl.feature_importances_, index=xcols).sort_values(ascending=False)
    out = [f"# E10 prototype — selecting the surrogate from the training data ({len(cells)} cells, {S.shape[0]} problems, "
           f"{len(methods)} methods)\n",
           "Policies other than the selector are those of results/E3_selection_stats.md (leave-one-problem-out picks on "
           "seeds 0-4; 'data-driven types' assign a problem by its own method profile, a ceiling, not deployable). The "
           "selector sees only (X, y), d and n of each dataset. All scored on held-out seeds 5-9.\n",
           md(summ, ".3g"), "", "Pairwise (Wilcoxon over problems, Holm; bootstrap CI; Bayesian signed-rank, ROPE ±10 %):\n",
           md(Rt, ".3g"), "", "## Feature importances (impurity, one fit on all seeds 0-4 rows)\n",
           md(imp.to_frame("importance"), ".3f"), ""]
    (ROOT / "results" / "E10_selector.md").write_text("\n".join(out), encoding="utf8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")         # reports use Greek letters; the Windows console is cp1252
    if "--features" in sys.argv:
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
        build(w)
    else:
        evaluate()
