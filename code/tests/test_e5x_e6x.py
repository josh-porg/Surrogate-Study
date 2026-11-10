"""Smoke and behaviour tests for the E5x scouts, the kriging-believer batch and the E6x multi-fidelity methods.

    python code/tests/test_e5x_e6x.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from surr import acquisitions as A, studies  # noqa: E402
from surr.learners import make  # noqa: E402


def _batch(rule, scout, kb, d=2, n=12, seed=0):
    """One batch of d+3 points from a GP scout on Branin-like data, with or without the believer refit."""
    rng = np.random.default_rng(seed)
    U = rng.random((n, d)); y = np.sin(6 * U[:, 0]) + U[:, 1] ** 2
    pool = rng.random((2000, d))
    model = make(scout, seed=0).fit(U, y)
    batch = []
    for _ in range(d + 3):
        Ub = np.vstack([U, pool[batch]]) if batch else U
        yb = np.r_[y, model.predict(pool[batch])] if batch else y
        mb = studies._believer(model, scout, {}, 0, Ub, yb) if kb and batch else model
        batch.append(A.acquire(rule, mb, Ub, yb, pool, rng, exclude=np.array(batch, int)))
    P = pool[batch]
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(2)); np.fill_diagonal(D, np.inf)
    return D.min()


def test_believer_spreads_batch():
    old, new = _batch("max_variance", "gp", False), _batch("max_variance", "gp", True)
    assert new > 3 * old, (old, new)
    return old, new


def test_believer_gp_is_exact_update():
    """Fixed-kernel refit = the GP posterior conditioned on the extra points: the variance at a fantasised point
    collapses to the noise level."""
    rng = np.random.default_rng(1)
    U = rng.random((15, 2)); y = np.sin(8 * U[:, 0]) * np.cos(5 * U[:, 1])      # wiggly: short length-scales
    m = make("gp", seed=0).fit(U, y)
    x = np.array([[0.02, 0.98]])                                                  # a corner, far from the design
    m2 = studies._believer(m, "gp", {}, 0, np.vstack([U, x]), np.r_[y, m.predict(x)])
    noise = m._m.kernel_.k2.noise_level                    # sklearn's predictive sd includes the WhiteKernel term
    latent = lambda mm: (mm.predict_std(x)[0] / mm._ys) ** 2 - noise
    v = latent(m)                                          # one observation with noise σ²: v → v σ² / (v + σ²)
    assert np.isclose(latent(m2), v * noise / (v + noise), rtol=0.05), (v, latent(m2), v * noise / (v + noise))
    assert np.allclose(m2._m.kernel_.theta, m._m.kernel_.theta)
    return float(latent(m)), float(latent(m2))


def test_new_scouts_run():
    out = {}
    for scout, rule in [("none", "mipt"), ("gp", "tead"), ("gp", "sfcvt"), ("mlp_ens", "masa"), ("gp", "imse_alc")]:
        cfg = dict(kind="e5", exp="E5x_sampling", problem="branin", scout=scout, rule=rule,
                   scout_kw={"opt": "lbfgs"} if scout == "mlp_ens" else {}, level=0.0, seed=0, batch="kb")
        r = studies.task_e5(cfg)
        assert r["status"] == "ok" and all(np.isfinite(v) for v in r["fits"].values()), r
        out[rule] = (round(r["fits"]["gp@50d"], 4), round(r["t_design"], 1))
    return out


def test_e6x_methods_run():
    out = {}
    for m in ("ar1_rhox", "smt_mfk", "mf_transfer_nn", "cv_select", "ar1_cokriging"):
        r = studies.task_e6(dict(kind="e6", exp="E6x_mf", mfproblem="mf_linear__smooth_trig_d2", nH=20, ratio=4,
                                 seed=0, method=m))
        assert r["status"] == "ok" and np.isfinite(r["nrmse"]), r
        out[m] = (round(r["nrmse"], 4), r.get("choice"))
    return out


if __name__ == "__main__":
    print("min distance in a batch, scout not refitted vs believer:", test_believer_spreads_batch())
    print("GP sd at a fantasised point before / after:", test_believer_gp_is_exact_update())
    print("E5x scouts on Branin (gp NRMSE at 50d, design s):", test_new_scouts_run())
    print("E6x methods (NRMSE, choice):", test_e6x_methods_run())
    print("ok")
