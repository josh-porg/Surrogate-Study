"""Task kinds and helpers for the later studies (registered in surr.runner):

  E4 mechanisms (H3)   input / target transforms applied inside `runner.task_learner` (cfg["transform"]):
                       smooth:<ℓ>   training targets replaced by a Gaussian-kernel (Nadaraya–Watson) smooth of the noisy
                                    targets with bandwidth ℓ in unit coordinates; scored against the unsmoothed truth
                                    (Grinsztajn, Oyallon & Varoquaux 2022, §5.2)
                       rotate       inputs seen by the learner are a random rotation of the design, scaled to stay in
                                    the unit box: V = 0.5 + (U − 0.5) Rᵀ / √d (Grinsztajn §5.4)
                       uninf:<k>    k uninformative uniform inputs appended to design and test points (Grinsztajn §5.3)
                       plus cfg["diag"]: GP hyper-parameter pathology (length-scales or noise at their bounds)
  E4 R-AIR bias arm    `task_rair`: aircraft data, biased / debiased-weights / debiased-balanced (STUDY_PLAN E4)
  E9 optimization      `task_e9`: one fitted surrogate, then every optimizer (and gradient source) on it, with a shared
                       evaluation memo and the incumbent recorded at budget checkpoints (anytime regret)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


# ── E4: transforms ────────────────────────────────────────────────────────────────────────────────────────────
def designs(cfg, p, S):
    """(U, T, Ul, Tl): true-coordinate design and test points, and what the learner sees. Without a transform
    Ul is U and Tl is T, drawn exactly as in E3 (same seeds), so E4 baselines match E3 cells."""
    d, n, seed = p.d, cfg["n"], cfg["seed"]
    tr = cfg.get("transform") or "none"
    if tr.startswith("uninf:"):
        k = int(tr.split(":")[1])
        Ul = S.make(cfg.get("sampler", "sobol"), n, d + k, seed=seed)
        Tl = S.sobol(cfg.get("n_test", 4096), d + k, seed=10_000 + seed)
        return Ul[:, :d], Tl[:, :d], Ul, Tl
    U = S.make(cfg.get("sampler", "sobol"), n, d, seed=seed)
    T = S.sobol(cfg.get("n_test", 4096), d, seed=10_000 + seed)
    if tr == "rotate":
        Q, R = np.linalg.qr(np.random.default_rng(40_000 + seed).normal(size=(d, d)))
        Q = Q * np.sign(np.diag(R))                                      # Haar-distributed rotation
        rot = lambda Z: 0.5 + (Z - 0.5) @ Q.T / np.sqrt(d)
        return U, T, rot(U), rot(T)
    return U, T, U, T


def smooth_targets(cfg, U, y):
    tr = cfg.get("transform") or "none"
    if not tr.startswith("smooth:"):
        return y
    ell = float(tr.split(":")[1])
    D2 = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    W = np.exp(-0.5 * D2 / ell ** 2)
    return W @ y / W.sum(axis=1)


def gp_diag(m):
    """Length-scales / noise at their optimisation bounds (sklearn GP learners; empty for others)."""
    g = getattr(m, "_m", None)
    k = getattr(g, "kernel_", None)
    if k is None:
        return {}
    out, hit = {}, 0
    for name, hp in [(h.name, h) for h in k.hyperparameters]:
        v = np.atleast_1d(k.get_params()[name]).astype(float)
        lo, hi = np.asarray(hp.bounds, float).T if np.ndim(hp.bounds) == 2 else hp.bounds
        at = (v <= np.asarray(lo) * 1.01) | (v >= np.asarray(hi) / 1.01)
        hit += int(at.sum())
        if name.endswith("length_scale"):
            out["ls_min"], out["ls_max"] = float(v.min()), float(v.max())
            out["ls_at_bound"] = int(at.sum())
        if name.endswith("noise_level"):
            out["noise"] = float(v[0]); out["noise_at_bound"] = int(at.sum())
    out["hp_at_bound"] = hit
    return out


# ── E4: R-AIR bias arm ────────────────────────────────────────────────────────────────────────────────────────
RAIR = ROOT / "data" / "local" / "r_air" / "cleaned_with_dates.csv"
RAIR_TARGETS = ("Empty_Weight_Fraction", "Fuel_Weight_Fraction", "Payload_Weight_Fraction")
RAIR_ENGINES = ("Jet", "Piston", "Propjet")
_RAIR_CACHE = {}


def _eis_year(v):
    s = str(v).strip()
    if s.endswith("s") and s[:4].isdigit():
        return int(s[:4]) + 5.0                                          # decade midpoint (author's decade_midpoint)
    try:
        return float(s[:4])
    except ValueError:
        return np.nan


def rair_data():
    """The author's processing (aircraftPerformancePrediction: compute_weight_ratios, build_strata) without the
    TensorFlow dependency: payload = AUW − MEW − FW with impossible rows dropped (< 0, > AUW, > 0.5·AUW), weight
    fractions as targets, AUW/MEW/FW/SHP/THR dropped from the inputs, engine type one-hot, EIS as a year."""
    if "df" in _RAIR_CACHE:
        return _RAIR_CACHE["df"], _RAIR_CACHE["X"]
    import pandas as pd
    df = pd.read_csv(RAIR)
    df = df[df["Engine Type"].isin(RAIR_ENGINES)].copy()                 # 4 malformed rows (company in that column)
    num = ["THR", "SHP", "Length", "Height", "Wing Span", "FW", "MEW", "AUW", "Vmax", "Vcruise", "Vstall", "Hmax",
           "Hmax (One)", "ROC", "ROC (One)", "Vlo", "Slo", "Vl", "Sl", "Range"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")             # the malformed rows made these text columns
    df["EIS"] = df["Entry into Service"].map(_eis_year)
    df = df.dropna(subset=["Range", "EIS", "FW", "MEW", "AUW"])
    pay = df.AUW - df.MEW - df.FW
    df = df[(pay >= 0) & (pay <= 0.5 * df.AUW)].copy()
    df["Empty_Weight_Fraction"] = df.MEW / df.AUW
    df["Fuel_Weight_Fraction"] = df.FW / df.AUW
    df["Payload_Weight_Fraction"] = (df.AUW - df.MEW - df.FW) / df.AUW
    for e in RAIR_ENGINES:
        df[e] = (df["Engine Type"] == e).astype(float)
    df["Multi Engine"] = df["Multi Engine"].astype(str).str.lower().eq("true").astype(float)
    df["TP mods"] = df["TP mods"].astype(str).str.lower().eq("true").astype(float)
    cols = [*RAIR_ENGINES, "Multi Engine", "TP mods", "Length", "Height", "Wing Span", "Vmax", "Vcruise", "Vstall", "Hmax",
            "Hmax (One)", "ROC", "ROC (One)", "Vlo", "Slo", "Vl", "Sl", "Range", "EIS"]
    df = df.reset_index(drop=True)
    _RAIR_CACHE["df"], _RAIR_CACHE["X"] = df, cols
    return df, cols


def rair_strata(df):
    """Author's strata: EIS decade bin × fuel-fraction quartile × range quartile × engine type."""
    import pandas as pd
    eis = pd.cut(df.EIS, bins=[1930, 1950, 1970, 1990, 2010, 2030]).astype(str)
    fuel = pd.qcut(df.Fuel_Weight_Fraction, q=4, duplicates="drop").astype(str)
    rng_ = pd.qcut(df.Range, q=4, duplicates="drop").astype(str)
    return eis + "_" + fuel + "_" + rng_ + "_" + df["Engine Type"].astype(str)


def task_rair(cfg):
    """One learner on one R-AIR target, split seed and bias setting. The 80/20 split depends only on the seed, so the
    three settings share the same test set (paired). Scores on the untouched test set: overall NRMSE, per engine
    type (same scale), their macro average, and an inverse-engine-frequency weighted NRMSE."""
    import warnings
    warnings.filterwarnings("ignore")
    from surr.learners import REGISTRY, make
    df, cols = rair_data()
    L = REGISTRY[cfg["learner"]]
    d = len(cols)
    if L.max_d and d > L.max_d:
        return dict(status="infeasible")
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(df)); cut = int(0.8 * len(df))
    tr, te = df.iloc[perm[:cut]].reset_index(drop=True), df.iloc[perm[cut:]].reset_index(drop=True)
    setting = cfg["setting"]
    r2 = np.random.default_rng(1000 + cfg["seed"])
    if setting == "weighted":                                            # inverse-frequency strata weights, applied
        s = rair_strata(tr)                                              # by weighted bootstrap (every learner alike)
        w = s.map(1.0 / s.value_counts(normalize=True)).values; w = w / w.sum()
        fit = tr.iloc[r2.choice(len(tr), len(tr), p=w)].reset_index(drop=True)
    elif setting == "balanced":                                          # equal engine-type counts
        k = tr["Engine Type"].value_counts().min()
        fit = tr.groupby("Engine Type", group_keys=False).apply(
            lambda g: g.sample(k, random_state=int(r2.integers(1 << 31)))).reset_index(drop=True)
    else:
        fit = tr
    if L.max_n and len(fit) > L.max_n:
        return dict(status="infeasible")
    med = tr[cols].median()
    lo, hi = tr[cols].min(), tr[cols].max()
    unit = lambda Z: np.clip(((Z[cols].fillna(med) - lo) / (hi - lo).replace(0, 1)).values.astype(float), 0, 1)
    U, Ut = unit(fit), unit(te)
    y, yt = fit[cfg["target"]].values.astype(float), te[cfg["target"]].values.astype(float)
    t0 = time.perf_counter()
    m = make(cfg["learner"], seed=cfg["seed"], **cfg.get("kw", {})).fit(U, y)
    t_fit = time.perf_counter() - t0
    pred = m.predict(Ut)
    scale = float(yt.std())
    err = pred - yt
    out = dict(status="ok", n_fit=len(fit), nrmse=float(np.sqrt(np.mean(err ** 2)) / scale), t_fit=t_fit)
    eng = te["Engine Type"].values
    per = {}
    for e in RAIR_ENGINES:
        q = eng == e
        if q.any():
            per[e] = float(np.sqrt(np.mean(err[q] ** 2)) / scale)
            out[f"nrmse_{e}"] = per[e]
    out["nrmse_macro"] = float(np.mean(list(per.values())))
    freq = {e: (eng == e).mean() for e in per}
    w = np.array([1.0 / freq[e] for e in eng])
    out["nrmse_weighted"] = float(np.sqrt(np.sum(w * err ** 2) / w.sum()) / scale)
    return out


# ── E9: optimization on the surrogate ─────────────────────────────────────────────────────────────────────────
GRAD_OPTS = ("slsqp", "lbfgsb", "trust_constr", "basin_hopping", "ipopt_multistart", "adam", "sgd_momentum", "bb_gd",
             "robbins_monro")
E9_MULTS = (10, 20, 50, 100, 200, 500)                                   # checkpoints, × d evaluations of the surrogate


def task_e9(cfg):
    """Fit the surrogate once (E3 design, noise and seeds), then run every optimizer on it with a budget of 500·d
    surrogate evaluations; gradient methods once with finite differences inside the budget ("fd") and once with the
    surrogate's own gradient ("model": analytic where the learner has one, its default otherwise). The incumbent is
    recorded at 10d … 500d evaluations and scored on the TRUE function: normalised regret (f(x̂) − f*) / std(f) over
    the E3 test set, and distance to x*. learner = "truth" runs the optimizers on f itself (optimizer-only baseline)."""
    import warnings
    warnings.filterwarnings("ignore")
    from surr import noise as N, optimizers as O, problems as P, samplers as S
    from surr.learners import REGISTRY, make
    from surr.problems.base import shifted
    import json
    p0 = P.get(cfg["problem"]); p = shifted(p0, cfg["shift_seed"])
    d, seed = p.d, cfg["seed"]
    table = _fstar()
    ref = table.get(p.name)
    if ref is None or "fstar" not in ref:
        return dict(status="error", error=f"no f* for {p.name}")
    fstar, xstar = ref["fstar"], np.asarray(ref["xstar"], float)
    T = S.sobol(4096, d, seed=10_000 + seed)
    scale = float(p.f_unit(T).std())
    out = dict(status="ok", fstar=fstar, fstar_confirmed=bool(ref.get("confirmed")), scale=scale)
    if cfg["learner"] == "truth":
        f = p.f_unit
        gfun = lambda U: p.grad(p.from_unit(U)) * (p.bounds[:, 1] - p.bounds[:, 0])
    else:
        L = REGISTRY[cfg["learner"]]
        n = cfg["n"]
        if (L.max_n and n > L.max_n) or (L.max_d and d > L.max_d):
            return dict(status="infeasible")
        U = S.make("sobol", n, d, seed=seed)
        y = N.apply(cfg.get("noise", "none"), cfg.get("level", 0.0), p.f_unit(U), U, scale, seed=20_000 + seed)
        t0 = time.perf_counter()
        m = make(cfg["learner"], seed=seed, **cfg.get("kw", {})).fit(U, y)
        out["t_fit"] = time.perf_counter() - t0
        out["nrmse"] = float(np.sqrt(np.mean((m.predict(T) - p.f_unit(T)) ** 2)) / scale)
        f, gfun = m.predict, m.grad
        out["design_best_regret"] = float((p.f_unit(U).min() - fstar) / scale)   # best training point (no optimizer)
    budget = E9_MULTS[-1] * d
    cps = tuple(k * d for k in E9_MULTS)
    memo = {}
    runs = {}
    for name in sorted(O.REGISTRY):
        for gsrc in (("fd", "model") if name in GRAD_OPTS else ("none",)):
            t0 = time.perf_counter()
            try:
                r = O.run(name, f, d, budget=budget, seed=seed, grad=gfun if gsrc == "model" else None, memo=memo,
                          checkpoints=cps)
            except Exception as ex:
                runs[f"{name}|{gsrc}"] = dict(error=f"{type(ex).__name__}: {ex}"[:200]); continue
            X = np.array([r["trace"][c] for c in cps])
            ft = p.f_unit(X)
            runs[f"{name}|{gsrc}"] = dict(
                nfev=int(r["nfev"]), wall=time.perf_counter() - t0,
                regret=[float((v - fstar) / scale) for v in ft],
                dist=[float(np.linalg.norm(x - xstar) / np.sqrt(d)) for x in X],
                s_at_x=float(f(X[-1:])[0]))                               # surrogate's belief at the final incumbent
    out["checkpoints"] = list(E9_MULTS)
    out["runs"] = runs
    out["memo_size"] = len(memo)
    return out


_FSTAR = {}


def _fstar():
    if not _FSTAR:
        import json
        _FSTAR.update(json.loads((ROOT / "results" / "fstar_table.json").read_text(encoding="utf8")))
    return _FSTAR
