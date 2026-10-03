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
    U = S.make(cfg.get("sampler", "sobol"), n, d, seed=seed)
    y_clean = p.f_unit(U)
    T = S.sobol(cfg.get("n_test", 4096), d, seed=10_000 + seed)
    yt = p.f_unit(T)
    scale = float(yt.std())
    y = N.apply(cfg.get("noise", "none"), cfg.get("level", 0.0), y_clean, U, scale, seed=20_000 + seed)

    t0 = time.perf_counter()
    m = make(cfg["learner"], seed=seed).fit(U, y)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = m.predict(T)
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
    if L.uq:
        s = m.predict_std(T)
        if s is not None:
            s = np.maximum(s, 1e-12)
            out["nlpd"] = float(np.mean(0.5 * np.log(2 * np.pi * s**2) + 0.5 * (err / s) ** 2))
            out["cover95"] = float(np.mean(np.abs(err) <= 1.96 * s))
    if cfg.get("grad"):                                                  # derivative accuracy (M.4), 256 points
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


KINDS = {"learner": task_learner, "thesis_canon": task_thesis_canon}


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


EXPERIMENTS = {"E0_timing": exp_E0_timing, "E1_canon": exp_E1_canon}


# ── driver ────────────────────────────────────────────────────────────────────────────────────────────────────
def _done_keys(folder: Path):
    keys = set()
    for f in folder.glob("*.jsonl"):
        for line in f.read_text(encoding="utf8").splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-task wall-clock cap [s]")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="run at most this many tasks (testing)")
    a = ap.parse_args(argv)

    cfgs = EXPERIMENTS[a.exp]()
    tasks = [dict(key=key_of(c), cfg=c) for c in cfgs]
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
            if n_done % 50 == 0 or n_done == len(todo):
                el = time.time() - t_start
                print(f"  {n_done}/{len(todo)}  elapsed {el / 60:.1f} min  "
                      f"eta {el / n_done * (len(todo) - n_done) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "code"))
    main()
