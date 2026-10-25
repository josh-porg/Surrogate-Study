"""Guarantees for the optimizer harness (code/surr/optimizers.py) that the E9 results rely on.

1. Shared starts: _starts(d, k, seed) is a prefix of _starts(d, K, seed) for k < K, so every multistart method starts
   from the same points.
2. The evaluation memo changes nothing but wall time: every optimizer returns bit-identical x, f and nfev with the
   memo on and off, on real fitted surrogates (smooth GP, a network ensemble, a tree model).

    python code/tests/test_optimizers_shared.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from surr import optimizers as O, problems as P, samplers as S  # noqa: E402
from surr.learners import make  # noqa: E402


def test_prefix():
    for d in (2, 5, 10):
        for seed in (0, 3, 17):
            big = O._starts(d, 16, seed)
            for k in (1, 2, 3, 5, 8):
                assert np.array_equal(O._starts(d, k, seed), big[:k]), (d, seed, k)


def test_memo_identical(budget=300):
    p = P.get("smooth_trig_d2")
    U = S.make("sobol", 40, 2, seed=1); y = p.f_unit(U)
    models = {L: make(L, seed=0, **kw).fit(U, y) for L, kw in
              (("gp", {}), ("mlp_ens", {"opt": "lbfgs", "K": 2}), ("xgb", {}))}
    names = sorted(O.REGISTRY)
    bad = []
    for L, m in models.items():
        for name in names:
            out = {}
            for memo in (False, True):
                O.MEMO = memo
                out[memo] = O.run(name, m.predict, 2, budget=budget, seed=5)
            a, b = out[False], out[True]
            if not (a["nfev"] == b["nfev"] and a["f"] == b["f"] and np.array_equal(a["x"], b["x"])):
                bad.append((L, name, a["f"], b["f"], a["nfev"], b["nfev"]))
    O.MEMO = True
    assert not bad, bad
    return len(models) * len(names)


def test_shared_memo_across_methods(budget=300):
    """One memo shared by all optimizers on a surrogate (the E9 task layout) gives exactly the results of no memo,
    in forward and in reverse method order — including surrogates whose predict uses fixed random streams."""
    p = P.get("smooth_trig_d2")
    U = S.make("sobol", 40, 2, seed=1); y = p.f_unit(U)
    models = {L: make(L, seed=0, **kw).fit(U, y) for L, kw in
              (("gp", {}), ("xgb", {}), ("mc_dropout", {}), ("bnn_vi", {"kl_weight": 0.1, "epochs": 500}),
               ("deep_gp", {}), ("local_gp", {}))}
    names = sorted(O.REGISTRY)
    bad = []
    for L, m in models.items():
        O.MEMO = False
        ref = {nm: O.run(nm, m.predict, 2, budget=budget, seed=5) for nm in names}
        O.MEMO = True
        for order in (names, names[::-1]):
            shared = {}
            got = {nm: O.run(nm, m.predict, 2, budget=budget, seed=5, memo=shared) for nm in order}
            for nm in names:
                a, b = ref[nm], got[nm]
                if not (a["nfev"] == b["nfev"] and a["f"] == b["f"] and np.array_equal(a["x"], b["x"])):
                    bad.append((L, nm, "reverse" if order is not names else "forward", a["f"], b["f"]))
    assert not bad, bad
    return len(models) * len(names) * 2


def test_checkpoints_do_not_change_results(budget=300):
    """Recording the incumbent at checkpoints (E9 anytime curves) leaves every optimizer's result bit-identical, and
    the checkpoint at the budget is the final incumbent when the run stops exactly there (DIRECT and others can
    overshoot the budget; their trace then shows the incumbent at the budget, the fairer number for E9)."""
    p = P.get("smooth_trig_d2")
    U = S.make("sobol", 40, 2, seed=1); y = p.f_unit(U)
    m = make("gp", seed=0).fit(U, y)
    bad = []
    for nm in sorted(O.REGISTRY):
        a = O.run(nm, m.predict, 2, budget=budget, seed=5)
        b = O.run(nm, m.predict, 2, budget=budget, seed=5, checkpoints=(20, 100, budget))
        if not (a["nfev"] == b["nfev"] and a["f"] == b["f"] and np.array_equal(a["x"], b["x"])):
            bad.append((nm, "changed"))
        last = b["trace"][budget]
        if b["nfev"] == budget and not np.array_equal(last, b["x"]) and float(m.predict(last[None])[0]) != b["f"]:
            bad.append((nm, "trace end != final"))
    assert not bad, bad
    return len(O.REGISTRY)


if __name__ == "__main__":
    test_prefix(); print("prefix property: ok")
    n = test_memo_identical(); print(f"memo on/off bit-identical: ok ({n} optimizer x surrogate pairs)")
    n = test_checkpoints_do_not_change_results(); print(f"checkpoint trace leaves results bit-identical: ok ({n} optimizers)")
    n = test_shared_memo_across_methods(); print(f"shared memo across methods, both orders, bit-identical: ok ({n} runs)")
