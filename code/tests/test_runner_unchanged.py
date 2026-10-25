"""Regression check for runner.task_learner after the E4 transform hooks (2026-10-25): without a transform the task
gives exactly what the pre-hook code path gave (re-implemented inline below), and transform="none" equals no key.
(Stored Kaggle records are not used: some learners, e.g. CART and MARS, differ in the last digits between Kaggle
and the laptop for the same code — platform numerics, BUGS N-23. Old vs new runner on those records: identical.)"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from surr import noise as N, problems as P, samplers as S  # noqa: E402
from surr.learners import make  # noqa: E402
from surr.runner import task_learner  # noqa: E402


def _old_nrmse(cfg):
    p = P.get(cfg["problem"]); d, n, seed = p.d, cfg["n"], cfg["seed"]
    U = S.make(cfg.get("sampler", "sobol"), n, d, seed=seed)
    T = S.sobol(cfg.get("n_test", 4096), d, seed=10_000 + seed)
    yt = p.f_unit(T); scale = float(yt.std())
    y = N.apply(cfg.get("noise", "none"), cfg.get("level", 0.0), p.f_unit(U), U, scale, seed=20_000 + seed)
    m = make(cfg["learner"], seed=seed, **cfg.get("kw", {})).fit(U, y)
    return float(np.sqrt(np.mean((m.predict(T) - yt) ** 2)) / scale)


def test_no_transform_is_the_old_path():
    bad, k = [], 0
    for L in ("rsm2", "pce_lars", "rbf_tps", "cart", "rf", "mars", "gp"):
        for pn, n, noise, lvl in (("smooth_trig_d2", 40, "none", 0.0), ("axis_step_d5", 100, "additive", 0.1),
                                  ("borehole", 80, "heteroscedastic", 0.1)):
            cfg = dict(kind="learner", problem=pn, learner=L, kw={}, n=n, seed=3, sampler="sobol", noise=noise,
                       level=lvl, grad=True)
            a = task_learner(cfg)["nrmse"]; b = _old_nrmse(cfg); c = task_learner({**cfg, "transform": "none"})["nrmse"]
            k += 1
            if not (a == b == c):
                bad.append((L, pn, a, b, c))
    assert not bad, bad
    return k


if __name__ == "__main__":
    print(f"no-transform path identical to the pre-hook code: ok ({test_no_transform_is_the_old_path()} cases)")
