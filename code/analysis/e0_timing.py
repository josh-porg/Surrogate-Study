"""E0 analysis: measured cost model per learner and status table.

Outputs
  results/E0_timing_table.md   median fit time [s] by learner × n (per d), statuses, power-law exponent b in t ≈ a·n^b
  results/E0_cost_model.json   {learner: {d: [a, b]}} used to budget later experiments (budget_cpu_hours)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw" / "E0_timing"


def load():
    rows, seen = [], set()
    for f in sorted(RAW.glob("*.jsonl")):
        for line in f.read_text(encoding="utf8").splitlines():
            r = json.loads(line)
            if r["key"] not in seen:
                seen.add(r["key"]); rows.append(r)
    return rows


def fit_powerlaw(ns, ts):
    ns, ts = np.asarray(ns, float), np.maximum(np.asarray(ts, float), 1e-4)
    if len(ns) < 2:
        return [float(ts[0]) if len(ts) else float("nan"), 0.0]
    b, loga = np.polyfit(np.log(ns), np.log(ts), 1)
    return [float(np.exp(loga)), float(b)]


def main():
    rows = load()
    by = defaultdict(list)
    for r in rows:
        c = r["cfg"]; d = int(c["problem"].rsplit("_d", 1)[1])
        by[(c["learner"], d, c["n"])].append(r)
    learners = sorted({k[0] for k in by}); ns = sorted({k[2] for k in by}); ds = sorted({k[1] for k in by})
    model, L = {}, ["# E0 timing — median fit time [s] (2 problems, 1 seed)", "",
                    "`T` = timeout (600 s), `I` = infeasible (memory/dimension wall), `E` = error.", ""]
    for d in ds:
        L += [f"## d = {d}", "", "| learner | " + " | ".join(f"n={n}" for n in ns) + " | b (t∝n^b) | NRMSE smooth / step @ max ok n |",
              "|---|" + "---|" * (len(ns) + 2)]
        for lr in learners:
            cells, okn, okt = [], [], []
            for n in ns:
                rs = by.get((lr, d, n), [])
                st = {r["status"] for r in rs}
                if "ok" in st:
                    t = np.median([r["t_fit"] for r in rs if r["status"] == "ok"])
                    cells.append(f"{t:.2g}"); okn.append(n); okt.append(t)
                else:
                    cells.append({"timeout": "T", "infeasible": "I"}.get(next(iter(st), ""), "E") if st else "·")
            a, b = fit_powerlaw(okn, okt) if okn else [float("nan")] * 2
            model.setdefault(lr, {})[d] = [a, b]
            acc = ""
            if okn:
                big = [r for r in by[(lr, d, okn[-1])] if r["status"] == "ok"]
                acc = " / ".join(f"{r['nrmse']:.3f}" for r in sorted(big, key=lambda r: r["cfg"]["problem"], reverse=True))
            L.append(f"| `{lr}` | " + " | ".join(cells) + f" | {b:.2f} | {acc} |")
        L.append("")
    errs = [r for r in rows if r["status"] == "error"]
    if errs:
        L += ["## Errors", ""] + [f"- `{r['cfg']['learner']}` {r['cfg']['problem']} n={r['cfg']['n']}: {r.get('error', '')}"
                                  for r in errs[:40]]
    (ROOT / "results" / "E0_timing_table.md").write_text("\n".join(L) + "\n", encoding="utf8")
    (ROOT / "results" / "E0_cost_model.json").write_text(json.dumps(model, indent=1), encoding="utf8")
    st = defaultdict(int)
    for r in rows:
        st[r["status"]] += 1
    print(f"{len(rows)} tasks: {dict(st)}; table -> results/E0_timing_table.md")


def budget_cpu_hours(cells, model_path=ROOT / "results" / "E0_cost_model.json", overhead=1.3):
    """cells: iterable of (learner, d, n, count). Predicted CPU-hours from the power-law cost model."""
    model = json.loads(Path(model_path).read_text())
    tot = 0.0
    for lr, d, n, cnt in cells:
        dd = min(model[lr], key=lambda k: abs(int(k) - d))
        a, b = model[lr][dd]
        if np.isfinite(a):
            tot += cnt * a * n**b
    return overhead * tot / 3600


if __name__ == "__main__":
    main()
