"""Experiment runner: config rows → content-hashed tasks → parallel, resumable, shardable execution.

Usage
  python -m surr.runner E0_timing --workers 10                 run everything not yet done
  python -m surr.runner E3 --shard 3/8 --workers 4             run shard 3 of 8 (for Kaggle / ACCESS nodes)
  python -m surr.runner E0_timing --dry                        count tasks and estimate cost from timings

Each task writes one JSON line to results/raw/<exp>/<shard-or-local>.jsonl; finished task keys are read back at
start so killed runs resume where they stopped. Workers are separate processes with all BLAS/OpenMP pools pinned to
one thread, so `--workers` equals cores used. A per-task wall-clock cap records "timeout" instead of hanging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import TimeoutError as _TO, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"


def key_of(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


# ── task kinds ────────────────────────────────────────────────────────────────────────────────────────────────
def task_learner(cfg):
    """Fit one learner on one problem / design / noise draw; score against clean truth."""
    import warnings
    warnings.filterwarnings("ignore")
    from scipy.stats import kendalltau
    from surr import noise as N, problems as P, samplers as S
    from surr.learners import REGISTRY, make

    p = P.get(cfg["problem"])
    if cfg.get("shift_seed") is not None:
        from surr.problems.base import shifted
        p = shifted(p, cfg["shift_seed"])
    d, n, seed = p.d, cfg["n"], cfg["seed"]
    L = REGISTRY[cfg["learner"]]
    if (L.max_n and n > L.max_n) or (L.max_d and d > L.max_d):
        return dict(status="infeasible")
    if hasattr(L, "feasible") and not L.feasible(n, d):                 # cost wall (bound by cost, not by failure)
        return dict(status="infeasible", error="cost wall")
    from surr import studies as ST
    U, T, Ul, Tl = ST.designs(cfg, p, S)              # Ul/Tl: what the learner sees (= U/T unless E4 transforms)
    y_clean = p.f_unit(U)
    yt = p.f_unit(T)
    scale = float(yt.std())
    y = N.apply(cfg.get("noise", "none"), cfg.get("level", 0.0), y_clean, U, scale, seed=20_000 + seed)
    y = ST.smooth_targets(cfg, U, y)                  # E4 smoothing probe (no-op otherwise)
    if L.max_d and Ul.shape[1] > L.max_d:
        return dict(status="infeasible")

    t0 = time.perf_counter()
    m = make(cfg["learner"], seed=seed, **cfg.get("kw", {})).fit(Ul, y)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = m.predict(Tl)
    t_pred = (time.perf_counter() - t0) / len(T)
    err = pred - yt
    out = dict(status="ok", nrmse=float(np.sqrt(np.mean(err**2)) / scale), max_err=float(np.max(np.abs(err)) / scale),
               mae=float(np.mean(np.abs(err)) / scale), tau=float(kendalltau(pred, yt)[0]),
               t_fit=t_fit, t_pred_us=1e6 * t_pred, scale=scale)
    q = yt <= np.quantile(yt, 0.1)                                       # near-optimal sublevel set (M.17)
    out["tau_low10"] = float(kendalltau(pred[q], yt[q])[0]) if q.sum() > 5 else float("nan")
    out["nrmse_low10"] = float(np.sqrt(np.mean(err[q] ** 2)) / scale)
    if cfg.get("noisy_cv"):                                              # practitioner's view (thesis B-01)
        yt_noisy = N.apply(cfg.get("noise", "none"), cfg.get("level", 0.0), yt, T, scale, seed=30_000 + seed)
        out["nrmse_vs_noisy"] = float(np.sqrt(np.mean((pred - yt_noisy) ** 2)) / scale)
    if cfg.get("diag"):
        out.update(ST.gp_diag(m))
    if L.uq:
        s = m.predict_std(Tl)
        if s is not None:
            s = np.maximum(s, 1e-12)
            out["nlpd"] = float(np.mean(0.5 * np.log(2 * np.pi * s**2) + 0.5 * (err / s) ** 2))
            out["cover95"] = float(np.mean(np.abs(err) <= 1.96 * s))
    if cfg.get("grad") and not cfg.get("transform"):                     # derivative accuracy (M.4), 256 points
        G = T[:256]
        gt = p.grad(p.from_unit(G)) * (p.bounds[:, 1] - p.bounds[:, 0])
        gm = m.grad(G)
        out["grad_nrmse"] = float(np.sqrt(np.mean((gm - gt) ** 2)) / (np.std(gt) + 1e-12))
    return out


COMPASS = ROOT.parent / "COMPASS"                                     # read-only: imported, never modified


def task_thesis_canon(cfg):
    """Poznański (2026) App. G canonical-surface CV, run with the thesis' own COMPASS code. Each fold is scored
    three ways: against the noisy held-out targets (thesis protocol, B-01), against the clean truth at the same
    points, and on a large independent clean test set. Relative RMSE as in COMPASS `_rel_errors`."""
    import warnings
    warnings.filterwarnings("ignore")
    if str(COMPASS) not in sys.path:
        sys.path.insert(0, str(COMPASS))
    from compass.surrogate import canonical as C
    from compass.surrogate.strategies import make as cmake

    s, name, sd, logt = cfg["surface"], cfg["strategy"], cfg["seed"], cfg["log_target"]
    X, y = C.sample(s, cfg.get("n", 140), noise=cfg.get("noise", 0.03), seed=sd)
    yc = np.asarray(C.CANONICAL[s].fn(X), float)
    Xt = C.grid_pool(cfg.get("n_test", 4096), seed=50_000 + sd); yt = np.asarray(C.CANONICAL[s].fn(Xt), float)
    rel = lambda a, b: float(np.sqrt(np.mean(((b - a) / np.maximum(np.abs(a), 1e-9)) ** 2)))
    rng = np.random.default_rng(sd)                                      # identical to benchmark.cv_folds
    folds = np.array_split(rng.permutation(len(X)), cfg.get("k", 5))
    noisy, clean, big = [], [], []
    for i, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        try:
            m = cmake(name, log_target=logt).fit(X[tr], y[tr])
            p = np.asarray(m.predict(X[te]), float)
            noisy.append(rel(y[te], p)); clean.append(rel(yc[te], p)); big.append(rel(yt, np.asarray(m.predict(Xt), float)))
        except Exception:
            noisy.append(float("nan")); clean.append(float("nan")); big.append(float("nan"))
    return dict(status="ok", fold_rmse_noisy=noisy, fold_rmse_clean=clean, fold_rmse_cleantest=big)


def task_fstar_refine(cfg):
    """Heavier second search for one unconfirmed reference optimum (code/analysis/fstar_table.py: refine); the
    first-pass record comes from the results/fstar_table.json shipped with the code."""
    sys.path.insert(0, str(ROOT / "code" / "analysis"))
    import fstar_table as ft
    k = cfg["name"] if cfg["seed"] is None else f"{cfg['name']}@s{cfg['seed']}"
    prev = ft.load()[k]
    return dict(status="ok", record=ft.refine(cfg["name"], cfg["seed"], prev))


def task_rair(cfg):
    from surr.studies import task_rair as t
    return t(cfg)


def task_e9(cfg):
    from surr.studies import task_e9 as t
    return t(cfg)


def task_e13(cfg):
    from surr.studies import task_e13 as t
    return t(cfg)


def task_rfea(cfg):
    from surr.studies import task_rfea as t
    return t(cfg)


def task_e14(cfg):
    from surr.studies import task_e14 as t
    return t(cfg)


def task_rfea_al(cfg):
    from surr.studies import task_rfea_al as t
    return t(cfg)


def task_e6(cfg):
    from surr.studies import task_e6 as t
    return t(cfg)


def task_e8(cfg):
    from surr.studies import task_e8 as t
    return t(cfg)


def task_e5(cfg):
    from surr.studies import task_e5 as t
    return t(cfg)


KINDS = {"learner": task_learner, "thesis_canon": task_thesis_canon, "fstar_refine": task_fstar_refine,
         "rair": task_rair, "e9": task_e9, "e13": task_e13, "rfea": task_rfea, "rfea_al": task_rfea_al, "e14": task_e14, "e6": task_e6, "e8": task_e8, "e5": task_e5}


def _init_worker():
    sys.path.insert(0, str(ROOT / "code"))
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _run_one(task):
    cfg = task["cfg"]
    t0 = time.perf_counter()
    try:
        res = KINDS[cfg["kind"]](cfg)
    except MemoryError as ex:
        res = dict(status="infeasible", error=str(ex)[:300])
    except Exception as ex:
        res = dict(status="error", error=f"{type(ex).__name__}: {ex}"[:300], tb=traceback.format_exc()[-800:])
    res["wall"] = time.perf_counter() - t0
    return dict(key=task["key"], cfg=cfg, **res)


# ── experiment definitions (each returns a list of cfg dicts) ─────────────────────────────────────────────────
def exp_E0_timing():
    from surr.learners import REGISTRY
    rows = []
    for prob in ("smooth_trig", "axis_step"):
        for d in (2, 5, 10):
            for n in (20, 50, 100, 200, 500, 1000, 2000):
                for L in sorted(REGISTRY):
                    if L == "tabpfn":
                        continue
                    rows.append(dict(kind="learner", exp="E0_timing", problem=f"{prob}_d{d}", learner=L, n=n,
                                     seed=0, sampler="sobol", noise="none", level=0.0))
    return rows


def exp_E1_canon():
    """App. G Tables canonical_a/b with the thesis code: 13 strategies × 6 surfaces × lin/log × 30 seeds
    (seeds 0–7 reproduce the published 8 repeats; 0–29 give the corrected-power version)."""
    strategies = ["linear", "rbf", "gp", "sparse_physics", "gp_physics_mean", "gbt", "mlp", "gp_grid", "gp_bspline",
                  "mlp_grid", "gbt_grid", "nn", "gp_regime"]
    surfaces = ["smooth", "collapse_knee", "clamp_floor", "anisotropic", "oscillatory", "cliff"]
    return [dict(kind="thesis_canon", exp="E1_canon", surface=s, strategy=g, log_target=lt, seed=sd, n=140,
                 noise=0.03, k=5)
            for s in surfaces for g in strategies for lt in (False, True) for sd in range(30)]


SCREEN_PROBLEMS = ["smooth_trig_d5", "axis_step_d5", "rotated_ridge_d5", "fourier_sparse_d2", "kink_floor_d2",
                   "plateau_bump_d2", "nonstationary_d2", "gp_draw_d5", "tree_draw_d5", "composite_d5",
                   "thesis_collapse_knee", "borehole"]


def exp_E2a_learners():
    """E2 learner screening: every learner (and the network training-optimizer variants) on 12 screening problems
    × n ∈ {10d, 50d, 200d} × noise {none, additive 3 %} × 10 seeds, with gradient and calibration metrics."""
    from surr.learners import REGISTRY
    from surr.problems import get
    configs = [(L, {}) for L in sorted(REGISTRY) if L not in ("tabpfn", "mlp_ens", "kan")]
    configs += [("mlp_ens", {"opt": o}) for o in ("adam", "adamw", "sgd", "lbfgs", "bb")]
    configs += [("kan", {"opt": o}) for o in ("adam", "lbfgs")]
    rows = []
    for pn in SCREEN_PROBLEMS:
        d = get(pn).d
        for mult in (10, 50, 200):
            n = min(mult * d, 1000)
            for noise, lvl in (("none", 0.0), ("additive", 0.03)):
                for L, kw in configs:
                    for sd in range(10):
                        rows.append(dict(kind="learner", exp="E2a_learners", problem=pn, learner=L, kw=kw, n=n,
                                         seed=sd, sampler="sobol", noise=noise, level=lvl, grad=True))
    return rows


E2B_CONFIGS = [("gam", {}), ("pspline_te", {}), ("symreg", {}), ("sparse_gp", {}), ("warped_gp", {}), ("deep_gp", {}),
               ("bnn_laplace", {}), ("bnn_vi", {}), ("bnn_vi", {"kl_weight": 0.1}), ("bnn_vi", {"kl_weight": 0.01}),
               ("bart", {}), ("moe", {}), ("natural_neighbor", {}),
               ("kernel_flow_gp", {}), ("lowrank_cp", {}), ("ft_als", {}), ("sindy_ae_static", {}),
               ("paaa", {}), ("rat_ls", {}), ("rat_ls", {"reg": "tik"})]


def exp_E2b_learners():
    """E2b: the survey-extension learners (learners/survey.py, added 2026-10-11) on exactly the E2a screening grid, so
    they are compared cell-for-cell with the E2a records (same problems, n, noise, seeds; the designs and noise draws
    depend only on those). bnn_vi is screened as the textbook ELBO and two tempered KL weights. aaa_rational is 1-D
    only and is studied in E6."""
    from surr.problems import get
    rows = []
    for pn in SCREEN_PROBLEMS:
        d = get(pn).d
        for mult in (10, 50, 200):
            n = min(mult * d, 1000)
            for noise, lvl in (("none", 0.0), ("additive", 0.03)):
                for L, kw in E2B_CONFIGS:
                    for sd in range(10):
                        rows.append(dict(kind="learner", exp="E2b_learners", problem=pn, learner=L, kw=kw, n=n,
                                         seed=sd, sampler="sobol", noise=noise, level=lvl, grad=True))
    return rows


_PROF = ("aaa", "cleanup", "svd", "caaa", "fh")     # 1-D profile methods (rational.fit_profile)
E12_LEARNERS = ([("aaa_rational", {})] + [("aaa_rational", {"profile": q}) for q in _PROF]      # 1-D problems only
                + [("paaa", {}), ("rat_ls", {}), ("rat_ls", {"reg": "tik"}), ("rat_ls", {"reg": "pos"})]
                + [("ridge_aaa", {"profile": q}) for q in _PROF]
                + [("additive_aaa", {"profile": q}) for q in _PROF]
                + [("gp_ratmean", {"profile": q}) for q in ("aaa", "cleanup", "caaa", "fh")]
                + [("gp", {}), ("as_gp", {}), ("gam", {}), ("ppr", {}), ("pce_lars", {}), ("mlp_ens", {"opt": "lbfgs"}),
                   ("xgb", {}), ("catboost", {})])
E12_PROBLEMS = ["forrester", "runge", "near_pole", "abs_kink", "step1d", "ratio2d", "resonance2d", "ridge_pole_d3",
                "ridge_pole_d5", "ridge_pole_d10", "pole_knee_d2", "pole_knee_d5", "pole_knee_d10",
                "thesis_collapse_knee", "smooth_trig_d2", "smooth_trig_d5", "additive_d5", "rotated_ridge_d5",
                "axis_step_d2", "borehole", "otlcircuit"]


def exp_E12_rational():
    """E12 (H9): rational surrogates — the established multivariate methods (p-AAA, LS rational) and the new AAA
    pairings (ridge, additive, GP rational mean), with univariate AAA on 1-D problems, against reference learners.
    Problems: P-RAT (known rational structure) + pole/knee probes + controls without rational structure. Factors:
    n ∈ {10d, 50d, 200d} (≤ 1000) × noise {0, 0.1 %, 1 %, 10 % additive} (the noise sweep is the key factor: AAA-type
    methods interpolate) × 10 seeds. Cells a learner cannot run in cheaply are recorded as infeasible (cost walls)."""
    from surr.problems import get
    rows = []
    for pn in E12_PROBLEMS:
        d = get(pn).d
        for mult in (10, 50, 200):
            n = min(mult * d, 1000)
            for noise, lvl in (("none", 0.0), ("additive", 0.001), ("additive", 0.01), ("additive", 0.10)):
                for L, kw in E12_LEARNERS:
                    for sd in range(10):
                        rows.append(dict(kind="learner", exp="E12_rational", problem=pn, learner=L, kw=kw, n=n,
                                         seed=sd, sampler="sobol", noise=noise, level=lvl, grad=True))
    return rows


CAP1000 = {"gp", "gp_se", "hetgp", "smt_krg", "smt_kpls", "as_gp", "dkl", "ens_press", "mlp_ens", "mc_dropout",
           "kan", "local_gp", "ok_variogram", "rbf_mq", "rbf_tps", "krr", "svr", "cs_fourier"}


# smt_krg (full ML-II kriging, d hyper-parameters) timed out at n = 1000 in 131/131 Kaggle fits (900 s cap, E3 wave 1 of
# the SMT track); capped at n <= 500 on 2026-10-09 (STUDY §9 deviation D-1). Its n > 500 timeout records stay in the data.
CAP500 = {"smt_krg"}
# Cut at E2a only because their family was already represented; reinstated 2026-10-11 by the author for the survey
# (STUDY §9 D-2) so every implemented method class is in the E3 accuracy study. E2a configs, unchanged.
E3_REINSTATED = ["idw", "knn", "qrf", "ngboost", "mc_dropout", "poly_ridge"]


def exp_E3_learners(seeds=range(10)):
    """E3 stage 1 (10 seeds): shortlist (results/E2a_shortlist.json) × core problems (problems.core_set) ×
    n ∈ {5d,10d,20d,50d,100d,300d} capped at 2000 (1000 for O(n^3)/network learners, CAP1000) × noise: additive
    {0, 1 %, 10 %} on all problems + {multiplicative, heteroscedastic, heavytail, numerical} × {1 %, 10 %} on a fixed
    quarter of the problems. Gradient and calibration metrics on."""
    from surr.problems import core_set
    short = json.loads((ROOT / "results" / "E2a_shortlist.json").read_text()) + E3_REINSTATED
    configs = []
    for k in short:
        if "[" in k:
            base, kv = k[:-1].split("[")
            configs.append((base, dict(x.split("=") for x in kv.split(","))))
        else:
            configs.append((k, {}))
    probs = core_set()
    other = {p.name for i, p in enumerate(sorted(probs, key=lambda p: p.name)) if i % 4 == 0}
    rows = []
    for p in probs:
        noises = [("none", 0.0), ("additive", 0.01), ("additive", 0.10)]
        if p.name in other:
            noises += [(k, lv) for k in ("multiplicative", "heteroscedastic", "heavytail", "numerical") for lv in (0.01, 0.10)]
        for n in sorted({min(m * p.d, 2000) for m in (5, 10, 20, 50, 100, 300)}):
            for L, kw in configs:
                if L in CAP1000 and n > 1000 or L in CAP500 and n > 500:
                    continue
                for noise, lvl in noises:
                    for sd in seeds:
                        rows.append(dict(kind="learner", exp="E3_learners", problem=p.name, learner=L, kw=kw, n=n,
                                         seed=sd, sampler="sobol", noise=noise, level=lvl, grad=True))
    return rows


def exp_E3x_learners(seeds=range(10)):
    """E3 extension (D-9, 2026-10-13): every registered single-fidelity problem that E3's core set left out — the
    designed P-FEAT probes at d = 10, the remaining VLSE benchmark functions (2-D and scalable d = 3/5/8/10) and the
    P-RAT problems — at the full E3 design, with the same learners. A separate experiment so E3's finished task list
    (and the sensitivity runs derived from it) stay unchanged; the extra noise types go to every fourth extension
    problem by name, as in E3. Analysis reads E3 and E3x together."""
    from surr.problems import REGISTRY, core_set
    core = {p.name for p in core_set()}
    probs = [p for n, p in sorted(REGISTRY.items()) if "@s" not in n and n not in core and p.block != "P-MF"]
    rows = [{**r, "exp": "E3x_learners"} for r in _e3_rows(probs, seeds)]
    return rows


def exp_Efstar_refine():
    """The f* table's refinement pass as runner tasks (so it runs on Kaggle): one task per instance that the first
    pass left unconfirmed and no refinement has covered yet. Keys depend only on (name, seed)."""
    sys.path.insert(0, str(ROOT / "code" / "analysis"))
    import fstar_table as ft
    table = ft.load()
    return [dict(kind="fstar_refine", exp="Efstar_refine", name=v["name"], seed=v["seed"])
            for v in table.values() if "fstar" in v and not v.get("confirmed") and not v.get("refined")]


def _e3_rows(probs, seeds):
    short = json.loads((ROOT / "results" / "E2a_shortlist.json").read_text()) + E3_REINSTATED
    configs = []
    for k in short:
        if "[" in k:
            base, kv = k[:-1].split("[")
            configs.append((base, dict(x.split("=") for x in kv.split(","))))
        else:
            configs.append((k, {}))
    other = {p.name for i, p in enumerate(sorted(probs, key=lambda p: p.name)) if i % 4 == 0}
    rows = []
    for p in probs:
        noises = [("none", 0.0), ("additive", 0.01), ("additive", 0.10)]
        if p.name in other:
            noises += [(k, lv) for k in ("multiplicative", "heteroscedastic", "heavytail", "numerical") for lv in (0.01, 0.10)]
        for n in sorted({min(m * p.d, 2000) for m in (5, 10, 20, 50, 100, 300)}):
            for L, kw in configs:
                if L in CAP1000 and n > 1000 or L in CAP500 and n > 500:
                    continue
                for noise, lvl in noises:
                    for sd in seeds:
                        rows.append(dict(kind="learner", exp="E3_learners", problem=p.name, learner=L, kw=kw, n=n,
                                         seed=sd, sampler="sobol", noise=noise, level=lvl, grad=True))
    return rows


def exp_E3s_gp_restarts():
    """E3 sensitivity (BUGS N-14): the four sklearn GP-family learners refit on the identical E3 grid (same designs,
    noise draws and test sets via the same seeds) with 5 ML-II restarts instead of the E3 cost setting. The 1-restart
    E3 records stay primary; this run measures how much of the GP results is optimizer failure. Seeds 0-4 only (paired
    with the same E3 seeds): 5 restarts cost ≈ 2.5x the E3 GP fits (292 CPU-h for 10 seeds at E3 settings)."""
    rows = []
    for r in exp_E3_learners(seeds=range(5)):
        if r["learner"] in ("gp", "gp_se", "hetgp", "as_gp"):
            rows.append({**r, "exp": "E3s_gp_restarts", "kw": {**r["kw"], "restarts": 5}})
    return rows


def exp_E3s_tuning():
    """E3 sensitivity, the symmetric counterpart of E3s_gp_restarts: the leading tree and network configs with a
    larger tuning budget on the identical E3 grid (seeds 0-4): XGBoost and CatBoost with a 30-trial random search
    (3-fold CV) instead of the small grid / fixed config; the L-BFGS MLP ensemble with 10 members instead of 5."""
    up = {"xgb": {"tune": "rs30"}, "catboost": {"tune": "rs30"}}
    rows = []
    for r in exp_E3_learners(seeds=range(5)):
        if r["learner"] in up:
            rows.append({**r, "exp": "E3s_tuning", "kw": {**r["kw"], **up[r["learner"]]}})
        elif r["learner"] == "mlp_ens" and r["kw"].get("opt") == "lbfgs":
            rows.append({**r, "exp": "E3s_tuning", "kw": {**r["kw"], "K": 10}})
    return rows


# ── later studies (2026-10-25) ─────────────────────────────────────────────────────────────────────────────────
E4_LEARNERS = [("gp", {}), ("hetgp", {}), ("smt_kpls", {}), ("catboost", {}), ("xgb", {}), ("rf", {}), ("extratrees", {}),
               ("mlp_ens", {"opt": "lbfgs"}), ("rbf_tps", {}), ("pce_lars", {})]
E4_TRANSFORMS = ["none", "smooth:0.05", "smooth:0.1", "smooth:0.2", "rotate", "uninf:2", "uninf:5", "uninf:15"]


def exp_E4_mechanisms():
    """E4 (H3): Grinsztajn mechanism probes, one factor at a time around the E3 baseline. The 19 P-FEAT probes at
    d = 5 × n ∈ {20d, 100d} × noise {none, additive 3 %} × transform (none; target smoothing ℓ ∈ {0.05, 0.1, 0.2};
    random rotation; 2 / 5 / 15 uninformative inputs) × 10 learners spanning the behaviour groups × 5 seeds, with
    GP hyper-parameter bound checks. "none" reproduces the E3 design and noise draws exactly."""
    from surr.problems import REGISTRY
    probs = [n for n, q in sorted(REGISTRY.items()) if q.block == "P-FEAT" and n.endswith("_d5")]
    rows = []
    for pn in probs:
        for n in (100, 500):
            for noise, lvl in (("none", 0.0), ("additive", 0.03)):
                for tr in E4_TRANSFORMS:
                    for L, kw in E4_LEARNERS:
                        for sd in range(5):
                            rows.append(dict(kind="learner", exp="E4_mechanisms", problem=pn, learner=L, kw=kw, n=n,
                                             seed=sd, sampler="sobol", noise=noise, level=lvl, transform=tr,
                                             diag=True))
    return rows


def exp_E4_rair():
    """E4 R-AIR bias arm (local only: real data). Every E3 learner config × 3 weight-fraction targets × setting
    {biased, weighted (inverse-frequency strata, weighted bootstrap), balanced (equal engine-type counts)} × 20
    random 80/20 splits; the test set depends only on the split seed, so settings are paired."""
    short = json.loads((ROOT / "results" / "E2a_shortlist.json").read_text()) + E3_REINSTATED
    configs = []
    for k in short:
        if "[" in k:
            base, kv = k[:-1].split("[")
            configs.append((base, dict(x.split("=") for x in kv.split(","))))
        else:
            configs.append((k, {}))
    from surr.studies import RAIR_TARGETS
    return [dict(kind="rair", exp="E4_rair", target=t, setting=st, learner=L, kw=kw, seed=sd)
            for t in RAIR_TARGETS for st in ("biased", "weighted", "balanced") for L, kw in configs for sd in range(20)]


E9_LEARNERS = [("gp", {}), ("hetgp", {}), ("as_gp", {}), ("smt_krg", {}), ("smt_kpls", {}), ("catboost", {}), ("lgbm", {}),
               ("rf", {}), ("mlp_ens", {"opt": "lbfgs"}), ("rbf_tps", {}), ("krr", {}), ("pce_lars", {}), ("rsm2", {}),
               ("ens_press", {}), ("mars", {}), ("idw", {})]
# spans the three data-driven E3 types (rugged / polynomial-like / smooth-kinked) and d = 2-10
E9_PROBLEMS = ["ackley_d5", "ackley_d10", "rastr", "rastr_d5", "levy_d5", "levy_d10", "michal_d5", "schaffer2",
               "axis_step_d5", "plateau_bump_d5", "tree_draw_d5", "regime_d5", "oblique_step_d10", "composite_d5",
               "sumsqu_d5", "trid_d5", "rothyp_d10", "quadratic_d5", "booth", "griewank_d5",
               "branin", "goldpr", "camel6", "rosen_d5", "stybtang", "zakharov_d5", "dixonpr_d5", "smooth_trig_d5",
               "smooth_trig_d10", "rotated_ridge_d5", "kink_floor_d5", "gp_draw_d2", "borehole", "wingweight", "piston",
               "thesis_collapse_knee", "pole_knee_d5", "monotone_d5"]


def exp_E9_optim():
    """E9 (RQ5/RQ6, H6/H7): surrogate × every optimizer (17; the 9 gradient methods with finite-difference and with
    surrogate gradients) on 38 problems × shifted instances 0-9 × n ∈ {10d, 50d} × noise {none, additive 1 %}, budget
    500·d surrogate evaluations with the incumbent scored at 10d…500d; plus the optimizers on the true function. One
    task = one fitted surrogate (shared memo). 10 instances, not the planned 30 (cost; STUDY D-10)."""
    from surr.problems import get
    rows = []
    for pn in E9_PROBLEMS:
        d = get(pn).d
        for s in range(10):
            rows.append(dict(kind="e9", exp="E9_optim", problem=pn, shift_seed=s, learner="truth", seed=s))
            for mult in (10, 50):
                for noise, lvl in (("none", 0.0), ("additive", 0.01)):
                    for L, kw in E9_LEARNERS:
                        if L == "smt_krg" and mult * d > 500:
                            continue
                        rows.append(dict(kind="e9", exp="E9_optim", problem=pn, shift_seed=s, learner=L, kw=kw,
                                         n=mult * d, noise=noise, level=lvl, seed=s))
    return rows


def exp_E11_fea():
    """E11 (RQ8) on R-FEA, local only (real data): every E3 learner config × use of the free scalar model {direct,
    tmf, aug} × training size {60, 140 (the thesis' canonical n), all ≈ 498} × 30 random 80/20 splits."""
    short = json.loads((ROOT / "results" / "E2a_shortlist.json").read_text()) + E3_REINSTATED
    configs = []
    for k in short:
        if "[" in k:
            base, kv = k[:-1].split("[")
            configs.append((base, dict(x.split("=") for x in kv.split(","))))
        else:
            configs.append((k, {}))
    return [dict(kind="rfea", exp="E11_fea", learner=L, kw=kw, use_lf=u, n_train=nt, seed=sd)
            for L, kw in configs for u in ("direct", "tmf", "aug") for nt in (60, 140, 0) for sd in range(30)]


def exp_E11_al():
    """E11 retrospective active learning on R-FEA (local only): 9 scouts (GP variance / IMSE / EIGF / MEPE, QRF and
    GBT-committee variance, LOLA-Voronoi, maximin, random control) × 20 splits; deliverables at 30 / 60 / 140 runs."""
    from surr.studies import E11_AL_SCOUTS
    return [dict(kind="rfea_al", exp="E11_al", scout=s, rule=r, seed=sd) for s, r in E11_AL_SCOUTS for sd in range(20)]


E14_PROBLEMS = ["smooth_trig_d2", "additive_d2", "lowrank_product_d2", "rotated_ridge_d2", "axis_step_d2", "kink_floor_d2",
                "branin", "goldpr", "camel6", "rastr", "smooth_trig_d5", "additive_d5", "lowrank_product_d5",
                "rotated_ridge_d5", "axis_step_d5", "quadratic_d5", "rosen_d5", "ackley_d5", "borehole", "wingweight",
                "piston", "smooth_trig_d10", "additive_d10", "lowrank_product_d10", "rotated_ridge_d10", "sumsqu_d10",
                "ackley_d10", "otlcircuit"]


def exp_E14_selfdesign():
    """E14 (added 2026-10-26; the coverage matrix's last planned methods): self-designing surrogates (Smolyak sparse
    grid, TT-cross) against gp, pce_lars, catboost and rbf_tps on a Sobol' design of the same size. 28 problems
    (favoured: additive, low-rank product, smooth; hurt: rotated ridge, steps) × budget {50, 200, 1000} × noise
    {0, 1 %} × 5 seeds (Smolyak's design is deterministic: seeds vary only its noise draws)."""
    rows = []
    for pn in E14_PROBLEMS:
        for b in (50, 200, 1000):
            for lvl in (0.0, 0.01):
                for sd in range(5):
                    for meth in ("smolyak", "tt_cross", "gp", "pce_lars", "catboost", "rbf_tps"):
                        if meth == "smolyak" and lvl == 0.0 and sd > 0:
                            continue                                     # identical runs
                        rows.append(dict(kind="e14", exp="E14_selfdesign", problem=pn, method=meth, budget=b,
                                         level=lvl, seed=sd))
    return rows


E13_PROBLEMS = ["ackley_d5", "rastr_d5", "levy_d5", "michal_d5", "schaffer2", "sumsqu_d5", "trid_d5", "rothyp_d10",
                "quadratic_d5", "griewank_d5", "branin", "goldpr", "camel6", "rosen_d5", "stybtang", "zakharov_d5",
                "dixonpr_d5", "smooth_trig_d5", "smooth_trig_d10", "rotated_ridge_d5", "kink_floor_d5", "borehole",
                "wingweight", "piston"]
E13_WITH_GRAD = [("gek", {}), ("hermite_rbf", {}), ("ge_pce", {}), ("sobolev_mlp", {})]
E13_VALUE_ONLY = [("gek", {"use_grad": False}), ("hermite_rbf", {"use_grad": False}), ("ge_pce", {"use_grad": False}),
                  ("sobolev_mlp", {"lam": 0}), ("gp", {}), ("pce_lars", {})]


def exp_E13_gradient():
    """E13: gradient-enhanced surrogates vs value-only ones at matched cost. 24 problems with smooth-enough gradients
    (step functions excluded) × n_base ∈ {5d, 10d, 20d} × noise level ∈ {0, 1 %, 10 %} on values and gradients × 5
    seeds. Gradient learners use n_base points with gradients; value-only learners (the same methods without the
    derivative data, plus gp and pce_lars) use cost_mult × n_base points, cost_mult ∈ {1, 2, d + 1} (an adjoint
    gradient ≈ 1 extra evaluation; finite differences d), capped at 2000."""
    from surr.problems import get
    rows = []
    for pn in E13_PROBLEMS:
        d = get(pn).d
        for mult in (5, 10, 20):
            for lvl in (0.0, 0.01, 0.10):
                for sd in range(5):
                    base = dict(kind="e13", exp="E13_gradient", problem=pn, n_base=mult * d, level=lvl, seed=sd)
                    rows += [{**base, "learner": L, "kw": kw, "use_grad": True, "cost_mult": 1} for L, kw in E13_WITH_GRAD]
                    rows += [{**base, "learner": L, "kw": kw, "use_grad": False, "cost_mult": c}
                             for L, kw in E13_VALUE_ONLY for c in sorted({1, 2, d + 1})]
    return rows


E6_METHODS = ["hf_only", "bridge_mult", "bridge_add", "ar1_cokriging", "hierarchical_kriging", "nargp", "feature_aug",
              "space_mapping"]


def exp_E6_mf():
    """E6 (RQ3, H4): every MF method on every bi-fidelity pair in P-MF (constructed linear / weak / nonlinear /
    shifted pairs on four bases, and the mf2 collection) × n_H ∈ {3d, 10d, 30d} × n_L / n_H ∈ {4, 16} × 10 seeds,
    plus HF-only at matched cost (the low-fidelity budget spent on extra HF points, κ = 10)."""
    from surr.problems.mf import MF_REGISTRY
    rows = []
    for pn, p in sorted(MF_REGISTRY.items()):
        for mult in (3, 10, 30):
            for ratio in (4, 16):
                for sd in range(10):
                    base = dict(kind="e6", exp="E6_mf", mfproblem=pn, nH=mult * p.d, ratio=ratio, seed=sd)
                    rows += [{**base, "method": m} for m in E6_METHODS]
                    rows.append({**base, "method": "hf_only", "matched": True})
    return rows


E8_LEARNERS = [("gp", {}), ("catboost", {}), ("xgb", {}), ("rf", {}), ("mlp_ens", {"opt": "lbfgs"}), ("rbf_tps", {}),
               ("pce_lars", {})]
E8_PROBLEMS = ["rastr", "schaffer2", "booth", "branin", "goldpr", "camel6", "stybtang", "gp_draw_d2",
               "thesis_collapse_knee", "ackley_d5", "levy_d5", "michal_d5", "axis_step_d5", "plateau_bump_d5",
               "tree_draw_d5", "regime_d5", "composite_d5", "sumsqu_d5", "trid_d5", "quadratic_d5", "rosen_d5",
               "zakharov_d5", "smooth_trig_d5", "rotated_ridge_d5", "kink_floor_d5", "pole_knee_d5", "monotone_d5"]


def exp_E8_upscale():
    """E8 (RQ4, H5): intermediate learner × every upscaler (none; tensor cubic grid m = 12 / 24; separable smoothing
    spline m = 24; P-spline direct) on 27 problems at d = 2 and 5 (grids are infeasible above; recorded) × n ∈ {20d,
    100d} × noise {0, 1 %} × shifted instances 0-9. One task = one fitted intermediate + all upscalers + six
    optimizers on each (the key question: does smoothing turn GBT into a good optimizer-facing surrogate?)."""
    from surr.problems import get
    rows = []
    for pn in E8_PROBLEMS:
        d = get(pn).d
        for mult in (20, 100):
            for noise, lvl in (("none", 0.0), ("additive", 0.01)):
                for L, kw in E8_LEARNERS:
                    for s in range(10):
                        rows.append(dict(kind="e8", exp="E8_upscale", problem=pn, shift_seed=s, learner=L, kw=kw,
                                         n=mult * d, noise=noise, level=lvl, seed=s))
    return rows


E5_SCOUTS = [("gp", "max_variance"), ("gp", "imse_alc"), ("gp", "eigf"), ("gp", "mepe"), ("gbt_committee", "max_variance"),
             ("qrf", "max_variance"), ("ngboost", "max_variance"), ("mlp_ens", "max_variance"), ("rf", "max_variance"),
             ("none", "lola_voronoi"), ("none", "seq_maximin"), ("none", "random"), ("none", "sobol")]
E5_PROBLEMS = ["thesis_smooth", "thesis_collapse_knee", "thesis_clamp_floor", "thesis_anisotropic", "thesis_oscillatory",
               "thesis_cliff", "smooth_trig_d5", "axis_step_d5", "kink_floor_d2", "nonstationary_d2", "plateau_bump_d2",
               "multiscale_d2", "regime_d5", "rotated_ridge_d5", "branin", "goldpr", "camel6", "rosen_d5", "ackley_d5",
               "levy_d5", "michal_d5", "stybtang_d5"]


def exp_E5_sampling():
    """E5 (RQ2, H2): scout (13: GP variance / IMSE / EIGF / MEPE, committee and ensemble variances, LOLA-Voronoi,
    sequential maximin, random, Sobol' control) × 22 problems (P-THESIS, P-FEAT probes, VLSE) × noise {0, 1 %} × 10
    seeds (the plan said 30; cost); every deliverable fitted at 10d / 20d / 50d inside each task."""
    rows = []
    for pn in E5_PROBLEMS:
        for scout, rule in E5_SCOUTS:
            for lvl in (0.0, 0.01):
                for sd in range(10):
                    rows.append(dict(kind="e5", exp="E5_sampling", problem=pn, scout=scout, rule=rule,
                                     scout_kw={"opt": "lbfgs"} if scout == "mlp_ens" else {}, level=lvl, seed=sd))
    return rows


EXPERIMENTS = {"E0_timing": exp_E0_timing, "E1_canon": exp_E1_canon, "E2a_learners": exp_E2a_learners,
               "E3_learners": exp_E3_learners, "E3s_gp_restarts": exp_E3s_gp_restarts, "E3s_tuning": exp_E3s_tuning,
               "E2b_learners": exp_E2b_learners, "E12_rational": exp_E12_rational, "E3x_learners": exp_E3x_learners,
               "Efstar_refine": exp_Efstar_refine, "E4_mechanisms": exp_E4_mechanisms, "E4_rair": exp_E4_rair,
               "E9_optim": exp_E9_optim, "E13_gradient": exp_E13_gradient, "E6_mf": exp_E6_mf,
               "E8_upscale": exp_E8_upscale, "E5_sampling": exp_E5_sampling, "E11_fea": exp_E11_fea,
               "E11_al": exp_E11_al, "E14_selfdesign": exp_E14_selfdesign}


# ── driver ────────────────────────────────────────────────────────────────────────────────────────────────────
def _done_keys(folder: Path):
    keys = set()
    for f in folder.glob("done_keys*.txt"):                              # compact resume list shipped to remote shards
        keys |= set(f.read_text(encoding="utf8").split())
    for f in folder.glob("*.jsonl"):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass
    return keys


# peak RSS per worker [GB], measured on the laptop. smt_krg d=10: 0.52 GB at n=200, 0.76 at n=500 (n=1000
# extrapolated). At n=1000, d=10 (2026-10-13): gp 0.45, as_gp 0.45, gp_ratmean 0.46, mlp_ens 0.35, rat_ls/paaa/
# additive_aaa 0.11-0.12. The old blanket 2 GB for CAP1000 held E12 to one worker. Learners not listed are < 0.5 GB.
MEM_GB = {**{L: 0.6 for L in CAP1000}, "smt_krg": 2.0, "smt_kpls": 2.0, "treed_gp": 2.0}


def _memory_capped_workers(workers, todo):
    """Fewer workers when the heaviest pending learner would not fit in RAM: 10 SMT kriging fits at n=1000 on the
    16.8 GB laptop pushed it into swap (2026-10-04). Leaves 2 GB for the OS and the driver."""
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
    except ImportError:
        return workers
    per = max((MEM_GB.get(t["cfg"].get("learner"), 0.5) for t in todo), default=0.5)
    cap = max(1, int((avail - 2.0) // per))
    if cap < workers:
        print(f"  memory cap: {workers} -> {cap} workers ({avail:.1f} GB free, up to {per:g} GB per task)", flush=True)
    return min(workers, cap)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-task wall-clock cap [s]")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only-learners", default="", help="comma list: keep only tasks whose learner is listed")
    ap.add_argument("--skip-learners", default="", help="comma list: drop tasks whose learner is listed")
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop cleanly after this wall time")
    ap.add_argument("--limit", type=int, default=0, help="run at most this many tasks (testing)")
    ap.add_argument("--min-free-disk-gb", type=float, default=5.0, help="stop cleanly when the results disk gets this full")
    a = ap.parse_args(argv)

    cfgs = EXPERIMENTS[a.exp]()
    tasks = [dict(key=key_of(c), cfg=c) for c in cfgs]
    if a.only_learners:
        keep = set(a.only_learners.split(",")); tasks = [t for t in tasks if t["cfg"].get("learner") in keep]
    if a.skip_learners:
        drop = set(a.skip_learners.split(",")); tasks = [t for t in tasks if t["cfg"].get("learner") not in drop]
    i, k = map(int, a.shard.split("/"))
    tasks = [t for t in tasks if int(t["key"], 16) % k == i - 1]
    folder = RAW / a.exp; folder.mkdir(parents=True, exist_ok=True)
    done = _done_keys(folder)
    todo = [t for t in tasks if t["key"] not in done]
    print(f"{a.exp}: {len(tasks)} tasks in shard {a.shard}, {len(done & {t['key'] for t in tasks})} done, "
          f"{len(todo)} to run on {a.workers} workers", flush=True)
    if a.dry or not todo:
        return
    # slow learners first so the pool drains evenly
    slow = {"ppr", "ens_press", "local_gp", "mlp_ens", "dkl", "gbt_committee", "mepe", "ngboost", "kan", "mc_dropout"}
    todo.sort(key=lambda t: (t["cfg"].get("learner") not in slow, -t["cfg"].get("n", 0)))
    if a.limit:
        todo = todo[-a.limit:]
    a.workers = _memory_capped_workers(a.workers, todo)
    out = folder / f"part_{a.shard.replace('/', 'of')}_{os.getpid()}.jsonl"
    t_start, n_done = time.time(), 0
    from pebble import ProcessPool                                       # kills a worker that exceeds the cap
    with ProcessPool(max_workers=a.workers, max_tasks=50, initializer=_init_worker) as ex, \
            out.open("a", encoding="utf8") as fh:
        futs = {ex.schedule(_run_one, args=(t,), timeout=a.timeout): t for t in todo}
        for fu in as_completed(futs):
            t = futs[fu]
            try:
                r = fu.result()
            except _TO:
                r = dict(key=t["key"], cfg=t["cfg"], status="timeout")
            except Exception as e:                                       # worker crash (e.g. OOM kill)
                r = dict(key=t["key"], cfg=t["cfg"], status="crash", error=repr(e)[:300])
            fh.write(json.dumps(r) + "\n"); fh.flush()
            n_done += 1
            low_disk = n_done % 20 == 0 and shutil.disk_usage(folder).free < a.min_free_disk_gb * 1e9
            if low_disk or (a.max_hours and time.time() - t_start > 3600 * a.max_hours):   # graceful stop
                why = "free disk below %g GB" % a.min_free_disk_gb if low_disk else f"max-hours {a.max_hours} reached"
                print(f"  {why} after {n_done} tasks; stopping (resume re-runs the rest)", flush=True)
                for f2 in futs:
                    f2.cancel()
                ex.stop()
                break
            if n_done % 50 == 0 or n_done == len(todo):
                el = time.time() - t_start
                print(f"  {n_done}/{len(todo)}  elapsed {el / 60:.1f} min  "
                      f"eta {el / n_done * (len(todo) - n_done) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "code"))
    main()
