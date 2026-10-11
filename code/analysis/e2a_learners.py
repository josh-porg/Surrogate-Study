"""E2a analysis — learner screening (STUDY_PLAN E2).

For each screening problem × n × noise cell: mean clean-truth NRMSE over seeds per learner config, rank within the
cell; aggregate to mean rank per problem and overall. Keep rule (STUDY_PLAN E2): top 5 by mean rank overall, every
thesis method's counterpart, and any config that is best (or within the best's 95 % CI) in at least one cell
("surprise rule"). Also reports the network training-optimizer comparison and gradient / calibration metrics.
Writes results/E2a_learners_report.md and results/E2a_shortlist.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw" / "E2a_learners"
# this package's counterparts of the Poznański (2026) App. G strategies (linear, rbf, gp, gbt, mlp/nn, gbt scout)
THESIS_COUNTERPARTS = {"gp", "rbf_tps", "delaunay_linear", "xgb", "mlp_ens", "gbt_committee"}


def cfg_name(c):
    kw = c.get("kw") or {}
    return c["learner"] + ("[" + ",".join(f"{k}={v}" for k, v in sorted(kw.items())) + "]" if kw else "")


def load():
    rows, seen = [], set()
    for f in sorted(RAW.glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            r = json.loads(line)
            if r["key"] not in seen:
                seen.add(r["key"]); rows.append(r)
    return rows


def main():
    rows = load()
    status = defaultdict(int)
    for r in rows:
        status[r["status"]] += 1
    cell = defaultdict(lambda: defaultdict(list))          # (problem, n, noise) -> config -> [nrmse]
    extra = defaultdict(lambda: defaultdict(list))          # config -> metric -> values
    for r in rows:
        if r["status"] != "ok":
            continue
        c = r["cfg"]; name = cfg_name(c)
        cell[(c["problem"], c["n"], c["noise"])][name].append(r["nrmse"])
        for m in ("grad_nrmse", "tau_low10", "cover95", "t_fit"):
            if m in r and r[m] is not None and np.isfinite(r[m]):
                extra[name][m].append(r[m])
    ranks, wins = defaultdict(list), defaultdict(int)
    per_problem = defaultdict(lambda: defaultdict(list))
    for (pn, n, noise), byc in cell.items():
        means = {k: np.mean(v) for k, v in byc.items() if len(v) >= 3}
        if len(means) < 5:
            continue
        names = list(means); r = stats.rankdata([means[k] for k in names])
        best = min(means, key=means.get)
        v = np.asarray(byc[best]); hw = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        for k, rk in zip(names, r):
            ranks[k].append(rk); per_problem[pn][k].append(rk)
            if means[k] <= means[best] + hw:
                wins[k] += 1
    overall = sorted(ranks, key=lambda k: np.mean(ranks[k]))
    # keep rule (revised 2026-10-04: "won any cell" kept 37/45 — too loose to shrink E3):
    #   overall top 6 ∪ top 2 per screening problem ∪ thesis counterparts ∪ best config of every method family
    #   (so classical / discontinued families stay represented); network optimizer variants: best + adam only
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from surr.learners import REGISTRY
    keep = set(overall[:6])
    for pn, byc in per_problem.items():
        keep |= set(sorted(byc, key=lambda k: np.mean(byc[k]))[:2])
    keep |= {k for k in ranks if k.split("[")[0] in THESIS_COUNTERPARTS and "[" not in k}
    fam_best = {}
    for k in overall:
        fam = REGISTRY[k.split("[")[0]].family
        fam_best.setdefault(fam, k)
    keep |= set(fam_best.values())
    for base in ("mlp_ens", "kan"):
        vs = sorted([k for k in keep if k.startswith(base + "[")], key=lambda k: np.mean(ranks[k]))
        keep -= set(vs[1:])
        if base == "mlp_ens":
            keep.add("mlp_ens[opt=adam]")                                    # thesis-style default kept
    L = ["# E2a — learner screening", "",
         f"Tasks: {dict(status)}. Cells with ≥ 5 configs × ≥ 3 seeds: {sum(1 for _ in cell)}.", "",
         "## Overall mean rank (lower = better) and cells won (best or within best's 95 % CI)", "",
         "| config | mean rank | cells won | median grad NRMSE | median τ (best 10 %) | 95 % coverage | median fit s | keep |",
         "|---|---|---|---|---|---|---|---|"]
    for k in overall:
        e = extra[k]
        med = lambda m: f"{np.median(e[m]):.3g}" if e.get(m) else "—"
        L.append(f"| `{k}` | {np.mean(ranks[k]):.1f} | {wins[k]} | {med('grad_nrmse')} | {med('tau_low10')} | "
                 f"{med('cover95')} | {med('t_fit')} | {'**yes**' if k in keep else ''} |")
    L += ["", "## Best config per screening problem (mean rank over its n × noise cells)", ""]
    for pn, byc in sorted(per_problem.items()):
        top = sorted(byc, key=lambda k: np.mean(byc[k]))[:5]
        L.append(f"- **{pn}**: " + ", ".join(f"`{k}` ({np.mean(byc[k]):.1f})" for k in top))
    L += ["", "## Network training optimizer (mlp_ens / kan)", ""]
    for base in ("mlp_ens", "kan"):
        vs = sorted([k for k in ranks if k.startswith(base + "[")], key=lambda k: np.mean(ranks[k]))
        if vs:
            L.append(f"- {base}: " + " < ".join(f"`{k}` ({np.mean(ranks[k]):.1f})" for k in vs))
    (ROOT / "results" / "E2a_learners_report.md").write_text("\n".join(L) + "\n", encoding="utf8")
    (ROOT / "results" / "E2a_shortlist.json").write_text(json.dumps(sorted(keep), indent=1), encoding="utf8")
    print("\n".join(L[:8 + min(len(overall), 50)]))
    print(f"\nshortlist ({len(keep)}):", sorted(keep))


if __name__ == "__main__":
    main()
