"""Problem library. Importing this package registers every problem in `REGISTRY`.

Conditions (noise types, sample-size regimes, dimension regimes, fidelity structures) are not functions but
experimental settings; they are listed in CONDITIONS so the coverage matrix can reference them.
"""
from .base import REGISTRY, Problem, register  # noqa: F401
from . import vlse, engineering, probes, mf, rational  # noqa: F401  (registration side effects)
from .mf import MF_REGISTRY  # noqa: F401

CONDITIONS = {
    "noise_none": "deterministic outputs",
    "noise_additive": "homoscedastic Gaussian noise",
    "noise_heavytail": "Student-t (nu=2) noise / 5% outliers",
    "noise_hetero": "noise sd varies across the domain",
    "noise_numerical": "repeatable high-frequency low-amplitude noise (mesh/tolerance)",
    "n_tiny": "n <= 2d training points",
    "n_small": "n in [5d, 20d]",
    "n_large": "n in [1e3, 1e5]",
    "d_high": "d >= 10 (up to 50 with low intrinsic dimension)",
    "mf_linear": "low fidelity linearly related to high fidelity (AR1-compatible)",
    "mf_nonlinear": "low fidelity nonlinearly related",
    "mf_shifted": "low fidelity is the high fidelity at shifted/scaled inputs",
    "mf_weak": "low fidelity weakly correlated (r < 0.6)",
    "real_fea": "Poznanski (2026) FEA cache",
    "real_air": "aircraft performance data",
}


CORE_VLSE2 = ["boha1", "sumsqu", "ackley", "levy", "booth", "zakharov", "easom", "michal", "camel6", "rosen",
              "branin", "goldpr"]                                       # 2 per VLSE shape class
CORE_SCALABLE = ["rosen", "ackley", "zakharov", "stybtang"]             # at d = 5 and 10
CORE_D10_FAMILIES = ["smooth_trig", "axis_step", "rotated_ridge", "additive", "fourier_sparse", "kink_floor",
                     "gp_draw", "composite"]


def core_set():
    """E3 core problems (STUDY_PLAN E3): ~70 problems keeping every coverage-matrix case."""
    fams = sorted({n.rsplit("_d", 1)[0] for n, p in REGISTRY.items() if p.block == "P-FEAT" and "+dummy" not in n})
    names = [f"{f}_d{d}" for f in fams for d in (2, 5)] + [f"{f}_d10" for f in CORE_D10_FAMILIES]
    names += [n for n, p in REGISTRY.items() if p.block in ("P-THESIS", "P-ENG")]
    names += CORE_VLSE2 + [f"{f}_d{d}" for f in CORE_SCALABLE for d in (5, 10)]
    names += [n for n in REGISTRY if "+dummy" in n]
    return [REGISTRY[n] for n in names]


def get(name: str) -> Problem:
    return REGISTRY[name]


def by_block(block: str):
    return [p for p in REGISTRY.values() if p.block == block]
