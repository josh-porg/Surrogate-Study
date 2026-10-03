"""Observation-noise models (MATHEMATICS §1.1). All are applied to clean values y at inputs U and are seeded.
`level` is the noise standard deviation as a fraction of sd(f) over the domain (`scale`), except `multiplicative`
where it is relative to |y| (the Poznański 2026 App. G convention)."""
from __future__ import annotations

import numpy as np


def apply(kind: str, level: float, y, U, scale: float, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    if level <= 0 or kind == "none":
        return y.copy()
    s = level * scale
    if kind == "additive":
        return y + rng.normal(0, s, len(y))
    if kind == "multiplicative":
        return y * (1 + rng.normal(0, level, len(y)))
    if kind == "heteroscedastic":                                        # sd grows 0.2s -> 1.8s across the box
        return y + rng.normal(0, s, len(y)) * (0.2 + 1.6 * U.mean(1))
    if kind == "heavytail":                                              # Student-t, nu = 2, scaled to s
        return y + s * rng.standard_t(2, len(y)) / np.sqrt(2)
    if kind == "numerical":                                              # repeatable high-frequency noise
        return y + s * np.sqrt(2) * np.sin(97.0 * U @ (np.arange(1, U.shape[1] + 1) * 1.37) + 0.3)
    raise ValueError(kind)
