"""E1 analysis — Poznański (2026) App. G Tables canonical_a/b: reproduction gate and corrected version.

(1) Reproduction: seeds 0-7, thesis protocol (relative RMSE vs noisy held-out folds, lin/log chosen per cell by the
    better mean), mean ± 95 % Student-t CI over all folds × seeds; compared cell by cell with the published values.
(2) Corrected: seeds 0-29, scored on a 4096-point clean test set, lin/log chosen by the *noisy CV* mean only
    (what a practitioner can see), so no test-set selection (B-01, B-05, B-08).
Writes results/E1_canon_report.md.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw" / "E1_canon"

# Published values, transcribed from Maters-Project-V3/chapters/Appendix_Spline_Study.tex Tables canonical_a/b
# (relative RMSE %, mean ± 95% CI, 8 repeats). nn column from canonical_b.
PUB = {
    "smooth":        dict(linear=(4.5, .3), rbf=(3.7, .1), gp=(3.0, .1), sparse_physics=(3.5, .2), gp_physics_mean=(3.5, .2), gbt=(3.7, .2), mlp=(4.3, .2), gp_grid=(3.0, .1), gp_bspline=(3.0, .1), mlp_grid=(4.2, .2), gbt_grid=(3.9, .2), nn=(3.3, .1)),
    "collapse_knee": dict(linear=(4.9, .3), rbf=(3.8, .2), gp=(3.1, .1), sparse_physics=(3.6, .2), gp_physics_mean=(3.6, .2), gbt=(3.6, .2), mlp=(5.3, .4), gp_grid=(3.1, .1), gp_bspline=(3.1, .1), mlp_grid=(5.3, .4), gbt_grid=(3.9, .2), nn=(3.5, .2)),
    "clamp_floor":   dict(linear=(4.3, .2), rbf=(3.7, .2), gp=(3.2, .1), sparse_physics=(4.5, .2), gp_physics_mean=(4.4, .2), gbt=(3.9, .1), mlp=(4.6, .3), gp_grid=(3.2, .1), gp_bspline=(3.2, .1), mlp_grid=(4.6, .3), gbt_grid=(4.0, .2), nn=(3.4, .2)),
    "anisotropic":   dict(linear=(4.0, .2), rbf=(3.6, .1), gp=(3.0, .1), sparse_physics=(3.0, .1), gp_physics_mean=(3.0, .1), gbt=(3.5, .1), mlp=(4.6, .2), gp_grid=(3.0, .1), gp_bspline=(3.0, .1), mlp_grid=(4.5, .2), gbt_grid=(3.5, .2), nn=(3.1, .1)),
    "oscillatory":   dict(linear=(7.5, .5), rbf=(5.7, .3), gp=(3.8, .2), sparse_physics=(10.4, .4), gp_physics_mean=(4.0, .2), gbt=(8.5, .4), mlp=(9.1, .4), gp_grid=(3.8, .2), gp_bspline=(3.8, .2), mlp_grid=(9.2, .4), gbt_grid=(9.8, .4), nn=(9.0, .5)),
    "cliff":         dict(linear=(4.7, .2), rbf=(3.9, .2), gp=(3.1, .1), sparse_physics=(6.3, .2), gp_physics_mean=(3.9, .5), gbt=(3.7, .1), mlp=(6.8, .5), gp_grid=(3.1, .1), gp_bspline=(3.1, .1), mlp_grid=(6.7, .4), gbt_grid=(3.6, .2), nn=(3.3, .1)),
}


def mean_ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return float("nan"), float("nan")
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def load():
    recs, seen = defaultdict(dict), set()
    for f in sorted(RAW.glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            r = json.loads(line)
            if r["key"] in seen or r.get("status") != "ok":
                continue
            seen.add(r["key"]); c = r["cfg"]
            recs[(c["surface"], c["strategy"], c["log_target"])][c["seed"]] = r
    return recs


def cell(recs, s, g, seeds, metric, select_by):
    """Pick lin/log by mean of `select_by` over the given seeds; return mean±CI of `metric` for that choice."""
    best = None
    for lt in (False, True):
        rs = [recs.get((s, g, lt), {}).get(sd) for sd in seeds]
        if any(r is None for r in rs):
            continue
        sel = np.nanmean(np.concatenate([r[select_by] for r in rs]))
        val = np.concatenate([r[metric] for r in rs])
        if best is None or sel < best[0]:
            best = (sel, mean_ci(val), lt)
    return (best[1], best[2]) if best else ((float("nan"), float("nan")), None)


def main():
    recs = load()
    surfaces = list(PUB)
    strategies = sorted({k[1] for k in recs}, key=lambda g: list(PUB["smooth"]).index(g) if g in PUB["smooth"] else 99)
    L = ["# E1 — Revisiting Poznański (2026) App. G canonical tables", "",
         "Relative RMSE in %, mean ± 95 % CI. **Repro** = thesis protocol (noisy-fold CV, seeds 0–7). "
         "**Clean** = same fits scored on a 4096-point clean test set, 30 seeds, lin/log chosen on noisy CV only.", ""]
    # (1) reproduction gate
    L += ["## 1. Reproduction gate (published vs reproduced)", "",
          "| surface | strategy | published | reproduced | within CIs |", "|---|---|---|---|---|"]
    n_ok = n_tot = 0
    for s in surfaces:
        for g in strategies:
            if g not in PUB[s]:
                continue
            (m, h), _ = cell(recs, s, g, range(8), "fold_rmse_noisy", "fold_rmse_noisy")
            if not np.isfinite(m):
                continue
            pm, ph = PUB[s][g]
            ok = abs(100 * m - pm) <= (100 * h + ph) + 0.05                  # intervals overlap (rounding slack)
            n_ok += ok; n_tot += 1
            L.append(f"| {s} | `{g}` | {pm:.1f} ± {ph:.1f} | {100 * m:.1f} ± {100 * h:.1f} | {'yes' if ok else '**no**'} |")
    L += ["", f"**{n_ok}/{n_tot} cells reproduced within overlapping CIs.**", ""]
    # (2) corrected table and ranks
    L += ["## 2. Corrected: clean-truth error (30 seeds)", "",
          "| surface | " + " | ".join(f"`{g}`" for g in strategies) + " |", "|---|" + "---|" * len(strategies)]
    ranks_pub, ranks_cln = [], []
    for s in surfaces:
        vals, row = {}, []
        for g in strategies:
            (m, h), _ = cell(recs, s, g, range(30), "fold_rmse_cleantest", "fold_rmse_noisy")
            vals[g] = (m, h); row.append(f"{100 * m:.2f} ± {100 * h:.2f}" if np.isfinite(m) else "—")
        best = min((v[0] for v in vals.values() if np.isfinite(v[0])), default=np.nan)
        bh = [v[1] for v in vals.values() if v[0] == best][0] if np.isfinite(best) else 0
        row = [f"**{r}**" if np.isfinite(vals[g][0]) and vals[g][0] - vals[g][1] <= best + bh else r
               for g, r in zip(strategies, row)]
        L.append(f"| {s} | " + " | ".join(row) + " |")
        common = [g for g in strategies if g in PUB[s] and np.isfinite(vals[g][0])]
        ranks_pub.append(stats.rankdata([PUB[s][g][0] for g in common]))
        ranks_cln.append(stats.rankdata([vals[g][0] for g in common]))
    L += ["", "Bold = best or within the best's CI.", ""]
    rho = [stats.spearmanr(a, b)[0] for a, b in zip(ranks_pub, ranks_cln)]
    L += ["## 3. Does the ranking change?", "",
          "Spearman correlation between the published ranking and the clean-truth ranking, per surface: "
          + ", ".join(f"{s} {r:+.2f}" for s, r in zip(surfaces, rho)) + ".", ""]
    (ROOT / "results" / "E1_canon_report.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L[:12])); print(f"... reproduced {n_ok}/{n_tot}; report -> results/E1_canon_report.md")


if __name__ == "__main__":
    main()
