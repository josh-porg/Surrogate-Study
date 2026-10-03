"""Multi-fidelity problem pairs (block P-MF).

1. Constructed pairs from feature probes, one per fidelity condition in coverage.yaml (M.2 in MATHEMATICS.md):
     mf_linear     f_L = a f_H + b φ + c, φ smooth discrepancy, (a, b) tuned to a target Pearson r
     mf_nonlinear  f_L = tanh(1.5 f̃_H) + b φ  (monotone but nonlinear in f_H)
     mf_shifted    f_L(x) = f_H(0.9 x + 0.05)  (input misalignment — space mapping's case)
     mf_weak       mf_linear with r ≈ 0.5
2. The `mf2` bi-fidelity benchmark collection (van Rijn & Schmitt 2020): Forrester, Bohachevsky, Booth, Borehole,
   Branin, Currin, Hartmann6, Himmelblau, Park91a/b, Six-hump camel.
Each MFProblem exposes .high(U) and .low(U) on the unit cube, a nominal cost ratio, and the measured r.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import REGISTRY as _P

MF_REGISTRY: dict[str, "MFProblem"] = {}


@dataclass
class MFProblem:
    name: str
    high: callable
    low: callable
    d: int
    condition: str
    cost_ratio: float = 10.0
    desc: str = ""
    r: float = field(default=np.nan)

    def measure_r(self, n=4096, seed=0):
        U = np.random.default_rng(seed).random((n, self.d))
        self.r = float(np.corrcoef(self.high(U), self.low(U))[0, 1])
        return self.r


def _reg(p: MFProblem):
    p.measure_r(); MF_REGISTRY[p.name] = p
    return p


def _phi(U):
    return np.sin(3 * U[:, 0] + 1) + 0.5 * np.cos(2 * U[:, -1])


def _linear_pair(base, r_target, seed=0):
    """Find b so that corr(f_H, a f_H + b φ) = r_target (a = 1 after standardisation)."""
    U = np.random.default_rng(seed).random((4096, base.d))
    h, p = base.f_unit(U), _phi(U)
    hs, ps = (h - h.mean()) / h.std(), (p - p.mean()) / p.std()
    rho = np.corrcoef(hs, ps)[0, 1]
    # corr(hs, hs + b ps) = (1 + b rho) / sqrt(1 + b^2 + 2 b rho) = r  -> solve for b numerically
    bs = np.linspace(-20, 20, 40001)                                     # negative b lets r go below corr(h, phi)
    rr = (1 + bs * rho) / np.sqrt(1 + bs**2 + 2 * bs * rho)
    b = bs[np.argmin(np.abs(rr - r_target))]
    m, s, pm, psd = h.mean(), h.std(), p.mean(), p.std()
    return lambda V, b=b: base.f_unit(V) + b * s * (_phi(V) - pm) / psd     # same a = 1 as the solve above


for base_name in ("smooth_trig_d2", "smooth_trig_d5", "kink_floor_d2", "rotated_ridge_d5"):
    base = _P[base_name]
    _reg(MFProblem(f"mf_linear__{base_name}", base.f_unit, _linear_pair(base, 0.95), base.d, "mf_linear",
                   desc="AR1-compatible low fidelity, target r = 0.95"))
    _reg(MFProblem(f"mf_weak__{base_name}", base.f_unit, _linear_pair(base, 0.5), base.d, "mf_weak",
                   desc="weakly correlated low fidelity, target r = 0.5"))
    U0 = np.random.default_rng(1).random((4096, base.d)); h0 = base.f_unit(U0); hm, hsd = h0.mean(), h0.std()
    _reg(MFProblem(f"mf_nonlinear__{base_name}", base.f_unit,
                   lambda V, f=base.f_unit, hm=hm, hsd=hsd: np.tanh(1.5 * (f(V) - hm) / hsd) + 0.1 * _phi(V),
                   base.d, "mf_nonlinear", desc="monotone nonlinear transform of f_H plus small discrepancy"))
    _reg(MFProblem(f"mf_shifted__{base_name}", base.f_unit, lambda V, f=base.f_unit: f(0.9 * V + 0.05),
                   base.d, "mf_shifted", desc="low fidelity = f_H at affinely shifted inputs"))


def _mf2():
    try:
        import mf2
    except Exception:
        return
    for name in ("forrester", "bohachevsky", "booth", "borehole", "branin", "currin", "hartmann6", "himmelblau",
                 "park91a", "park91b", "six_hump_camelback"):
        f = getattr(mf2, name)
        lo, hi = np.asarray(f.l_bound, float), np.asarray(f.u_bound, float)
        to_phys = lambda U, lo=lo, hi=hi: lo + U * (hi - lo)
        _reg(MFProblem(f"mf2_{name}", lambda U, f=f, t=to_phys: np.asarray(f.high(t(U))).ravel(),
                       lambda U, f=f, t=to_phys: np.asarray(f.low(t(U))).ravel(), f.ndim, "mf2",
                       desc=f"mf2 {name} (van Rijn & Schmitt 2020)"))


_mf2()
