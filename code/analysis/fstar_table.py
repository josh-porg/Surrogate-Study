"""Reference optima f* for every analytic problem instance (unshifted and BBOB-style shifted, seeds 0-29): the
denominator-free part of every E9 regret f(x̂) − f*. Computed once, read by every surrogate × optimizer task.

Three independent global searches on the TRUE function, all on the unit box:
  de      differential evolution (popsize 30, vectorised), polished with L-BFGS-B
  ms      L-BFGS-B from 64 scrambled Sobol' starts
  direct  DIRECT (not locally biased, 2000·d evaluations), polished with L-BFGS-B from its best point
f* = the smallest value found; an instance is "confirmed" when ≥ 2 searches agree within 1e-6 × the function's
spread over 4096 Sobol' points (and the unshifted value is checked against the literature f* where one is stored).
Unconfirmed instances are listed, not silently used.

    python code/analysis/fstar_table.py [--workers 2]
Writes results/fstar_table.json and results/fstar_table.md. Resumable (skips instances already in the JSON).
"""
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "fstar_table.json"
SEEDS = range(30)


def _instances():
    sys.path.insert(0, str(ROOT / "code"))
    from surr.problems import REGISTRY
    base = [n for n, p in REGISTRY.items() if "@s" not in n and p.block not in ("P-MF",) and p.d <= 13]
    return [(n, None) for n in base] + [(n, s) for n in base for s in SEEDS]


def solve(name, seed):
    sys.path.insert(0, str(ROOT / "code"))
    warnings.filterwarnings("ignore")
    from scipy import optimize as so
    from scipy.stats import qmc
    from surr.problems import get
    from surr.problems.base import shifted
    p = get(name) if seed is None else shifted(get(name), seed)
    d = p.d
    F = lambda x: float(p.f_unit(np.clip(np.atleast_2d(x), 0, 1))[0])
    Fv = lambda X: p.f_unit(np.clip(np.asarray(X).T, 0, 1))              # DE vectorised: (d, S) -> (S,)
    spread = float(np.ptp(p.f_unit(qmc.Sobol(d, scramble=True, seed=99).random(4096))))
    pol = lambda x0: so.minimize(F, x0, method="L-BFGS-B", bounds=[(0, 1)] * d, options=dict(maxiter=500))
    res = {}
    r = so.differential_evolution(Fv, [(0, 1)] * d, popsize=30, maxiter=300, tol=1e-12, seed=12345, polish=False,
                                  vectorized=True, updating="deferred")
    q = pol(r.x); res["de"] = (min(r.fun, q.fun), (q.x if q.fun < r.fun else r.x).tolist())
    best = (np.inf, None)
    for x0 in qmc.Sobol(d, scramble=True, seed=54321).random(64):
        q = pol(x0)
        if q.fun < best[0]:
            best = (float(q.fun), q.x.tolist())
    res["ms"] = best
    r = so.direct(F, [(0, 1)] * d, maxfun=2000 * d, locally_biased=False)
    q = pol(r.x); res["direct"] = (float(min(r.fun, q.fun)), (q.x if q.fun < r.fun else r.x).tolist())
    fstar = min(v[0] for v in res.values())
    tol = 1e-6 * max(spread, 1e-12)
    agree = [k for k, v in res.items() if v[0] - fstar <= tol]
    x = min(res.values(), key=lambda v: v[0])[1]
    out = dict(name=name, seed=seed, d=d, fstar=fstar, xstar=x, spread=spread, agree=agree,
               confirmed=len(agree) >= 2, methods={k: v[0] for k, v in res.items()})
    if seed is None and p.fstar_ref is not None:
        out["fstar_lit"] = float(p.fstar_ref); out["lit_ok"] = bool(fstar <= p.fstar_ref + tol)
    return out


def refine(name, seed, prev):
    """Heavier second pass for unconfirmed instances (hard multimodal landscapes): IPOP-CMA-ES (9 restarts, doubling
    population) and a large DE (popsize 80, 2000 generations), each polished with L-BFGS-B; agreement is then re-checked
    over all five searches."""
    sys.path.insert(0, str(ROOT / "code"))
    warnings.filterwarnings("ignore")
    import cma
    from scipy import optimize as so
    from surr.problems import get
    from surr.problems.base import shifted
    p = get(name) if seed is None else shifted(get(name), seed)
    d = p.d
    F = lambda x: float(p.f_unit(np.clip(np.atleast_2d(x), 0, 1))[0])
    Fv = lambda X: p.f_unit(np.clip(np.asarray(X).T, 0, 1))
    pol = lambda x0: so.minimize(F, x0, method="L-BFGS-B", bounds=[(0, 1)] * d, options=dict(maxiter=1000))
    m = dict(prev["methods"])
    xs = {"prev": prev["xstar"]}
    x, es = cma.fmin2(F, np.full(d, 0.5), 0.3, {"bounds": [0, 1], "seed": 777, "verbose": -9,
                                                "maxfevals": 4000 * d * 10}, restarts=9, incpopsize=2)
    q = pol(x); m["ipop_cma"] = float(min(F(x), q.fun)); xs["ipop_cma"] = (q.x if q.fun < F(x) else x).tolist()
    r = so.differential_evolution(Fv, [(0, 1)] * d, popsize=80, maxiter=2000, tol=1e-14, seed=2468, polish=False,
                                  vectorized=True, updating="deferred")
    q = pol(r.x); m["de_large"] = float(min(r.fun, q.fun)); xs["de_large"] = (q.x if q.fun < r.fun else r.x).tolist()
    fstar = min(m.values())
    tol = 1e-6 * max(prev["spread"], 1e-12)
    agree = [k for k, v in m.items() if v - fstar <= tol]
    best = min(m, key=m.get)
    return {**prev, "fstar": fstar, "xstar": xs.get(best, prev["xstar"]), "methods": m, "agree": agree,
            "confirmed": len(agree) >= 2, "refined": True}


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    if "--refine" in sys.argv:
        done = json.loads(OUT.read_text())
        todo = [(k, v) for k, v in done.items() if "fstar" in v and not v.get("confirmed") and not v.get("refined")]
        print(f"refining {len(todo)} unconfirmed instances on {workers} workers", flush=True)
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(refine, v["name"], v["seed"], v): k for k, v in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                k = futs[fu]
                try:
                    done[k] = fu.result()
                except Exception as e:
                    done[k]["refine_error"] = repr(e)[:300]
                if i % 10 == 0 or i == len(todo):
                    OUT.write_text(json.dumps(done)); print(f"  {i}/{len(todo)}", flush=True)
        unc = [k for k, v in done.items() if not v.get("confirmed")]
        print(f"after refinement: {len(unc)} unconfirmed", flush=True)
        (ROOT / "results" / "fstar_table.md").write_text(
            f"# f* reference table\n\n{len(done)} instances; {len(done) - len(unc)} confirmed by >= 2 independent "
            f"searches (5 searches for the {len(todo)} refined); {len(unc)} unconfirmed (regret on these is reported "
            f"against the best known value and flagged).\n\nUnconfirmed: {', '.join(unc)}\n\nNote: mccorm's stored "
            f"literature f* (-1.9133) is the computed optimum (-1.913223) rounded to 4 decimals.\n", encoding="utf8")
        return
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    key = lambda n, s: n if s is None else f"{n}@s{s}"
    todo = [(n, s) for n, s in _instances() if key(n, s) not in done]
    print(f"{len(done)} done, {len(todo)} to solve on {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(solve, n, s): (n, s) for n, s in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            n, s = futs[fu]
            try:
                done[key(n, s)] = fu.result()
            except Exception as e:
                done[key(n, s)] = dict(name=n, seed=s, error=repr(e)[:300])
            if i % 20 == 0 or i == len(todo):
                OUT.write_text(json.dumps(done))
                print(f"  {i}/{len(todo)}", flush=True)
    rows = [v for v in done.values() if "fstar" in v]
    unc = [k for k, v in done.items() if not v.get("confirmed")]
    lit = [k for k, v in done.items() if v.get("lit_ok") is False]
    (ROOT / "results" / "fstar_table.md").write_text(
        f"# f* reference table\n\n{len(rows)} instances solved; {len(rows) - len(unc)} confirmed by >= 2 searches; "
        f"{len(unc)} unconfirmed; {len(lit)} below-literature mismatches.\n\nUnconfirmed: {', '.join(unc[:200])}\n\n"
        f"Literature mismatches: {', '.join(lit)}\n", encoding="utf8")
    print(f"{len(rows)} solved, {len(unc)} unconfirmed, {len(lit)} literature mismatches", flush=True)


if __name__ == "__main__":
    main()
