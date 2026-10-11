"""P-RAT: test functions for the rational-surrogate study E12 (block "P-RAT"; deliberately outside core_set, so E3's
task list is unchanged). Each has a known rational structure (or a known lack of it, as a control).

  1-D  forrester     (6x-2)^2 sin(12x-4)                      Forrester et al. 2008 — smooth, oscillatory control
       runge         1/(1+25(2x-1)^2)                          Runge 1901 — complex poles at ±i/5 near the interval
       near_pole     1/(1.05-x)                                 real pole 0.05 outside the domain
       abs_kink      |x - 0.3|                                  non-analytic point inside (Newman: rational wins)
       step1d        tanh(50(x - 0.6))                          near-discontinuity
  2-D  ratio2d       (1 + x1 x2)/(1.1 - x1 + 0.2 x2)            genuinely bivariate rational, pole curve outside
       resonance2d   1/((x1 + x2 - 1)^2 + 0.01)                 Lorentzian ridge: rational along x1 + x2
  d    ridge_pole_dD 1/(1.05 - mean(x))   D = 3, 5, 10          pole along a ridge direction (ridge_aaa's home)
"""
from __future__ import annotations

import numpy as np

from .base import Problem, box, register


def _reg(name, fn, d, category, tags, desc):
    return register(Problem(name=name, fn=fn, bounds=box(0, 1, d), block="P-RAT", category=category, tags=tags,
                            desc=desc))


_reg("forrester", lambda X: (6 * X[:, 0] - 2) ** 2 * np.sin(12 * X[:, 0] - 4), 1, "rational_control",
     ("smooth", "oscillatory"), "Forrester et al. (2008) 1-D test function")
_reg("runge", lambda X: 1 / (1 + 25 * (2 * X[:, 0] - 1) ** 2), 1, "rational", ("pole", "complex_pole"),
     "Runge function on [0,1]")
_reg("near_pole", lambda X: 1 / (1.05 - X[:, 0]), 1, "rational", ("pole",), "real pole just outside the domain")
_reg("abs_kink", lambda X: np.abs(X[:, 0] - 0.3), 1, "rational", ("kink",),
     "|x-0.3|: Newman (1964) rational rate beats polynomials")
_reg("step1d", lambda X: np.tanh(50 * (X[:, 0] - 0.6)), 1, "rational_control", ("discontinuity",),
     "steep tanh step")
_reg("ratio2d", lambda X: (1 + X[:, 0] * X[:, 1]) / (1.1 - X[:, 0] + 0.2 * X[:, 1]), 2, "rational", ("pole",),
     "bivariate rational, pole curve outside the box")
_reg("resonance2d", lambda X: 1 / ((X[:, 0] + X[:, 1] - 1) ** 2 + 0.01), 2, "rational", ("pole", "ridge"),
     "Lorentzian resonance along x1 + x2 = 1")
for _d in (3, 5, 10):
    _reg(f"ridge_pole_d{_d}", lambda X: 1 / (1.05 - X.mean(1)), _d, "rational", ("pole", "ridge"),
         "real pole along the (1,...,1) ridge direction, just outside the box")
