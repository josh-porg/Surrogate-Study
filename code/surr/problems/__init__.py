"""Problem library. Importing this package registers every problem in `REGISTRY`.

Conditions (noise types, sample-size regimes, dimension regimes, fidelity structures) are not functions but
experimental settings; they are listed in CONDITIONS so the coverage matrix can reference them.
"""
from .base import REGISTRY, Problem, register  # noqa: F401
from . import vlse, engineering, probes, mf  # noqa: F401  (registration side effects)
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


def get(name: str) -> Problem:
    return REGISTRY[name]


def by_block(block: str):
    return [p for p in REGISTRY.values() if p.block == block]
