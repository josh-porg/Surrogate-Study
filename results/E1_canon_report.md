# E1 — Revisiting Poznański (2026) App. G canonical tables

Relative RMSE in %, mean ± 95 % CI. **Repro** = thesis protocol (noisy-fold CV, seeds 0–7). **Clean** = same fits scored on a 4096-point clean test set, 30 seeds, lin/log chosen on noisy CV only.

## 1. Reproduction gate (published vs reproduced)

| surface | strategy | published | reproduced | within CIs |
|---|---|---|---|---|
| smooth | `linear` | 4.5 ± 0.3 | 4.5 ± 0.3 | yes |
| smooth | `rbf` | 3.7 ± 0.1 | 3.7 ± 0.1 | yes |
| smooth | `gp` | 3.0 ± 0.1 | 3.0 ± 0.1 | yes |
| smooth | `sparse_physics` | 3.5 ± 0.2 | 3.5 ± 0.2 | yes |
| smooth | `gp_physics_mean` | 3.5 ± 0.2 | 3.4 ± 0.2 | yes |
| smooth | `gbt` | 3.7 ± 0.2 | 3.7 ± 0.2 | yes |
| smooth | `mlp` | 4.3 ± 0.2 | 4.3 ± 0.2 | yes |
| smooth | `gp_grid` | 3.0 ± 0.1 | 3.0 ± 0.1 | yes |
| smooth | `gp_bspline` | 3.0 ± 0.1 | 3.0 ± 0.1 | yes |
| smooth | `mlp_grid` | 4.2 ± 0.2 | 4.2 ± 0.2 | yes |
| smooth | `gbt_grid` | 3.9 ± 0.2 | 3.9 ± 0.2 | yes |
| smooth | `nn` | 3.3 ± 0.1 | 3.3 ± 0.1 | yes |
| collapse_knee | `linear` | 4.9 ± 0.3 | 4.9 ± 0.3 | yes |
| collapse_knee | `rbf` | 3.8 ± 0.2 | 3.8 ± 0.2 | yes |
| collapse_knee | `gp` | 3.1 ± 0.1 | 3.1 ± 0.1 | yes |
| collapse_knee | `sparse_physics` | 3.6 ± 0.2 | 3.6 ± 0.2 | yes |
| collapse_knee | `gp_physics_mean` | 3.6 ± 0.2 | 3.4 ± 0.2 | yes |
| collapse_knee | `gbt` | 3.6 ± 0.2 | 3.6 ± 0.2 | yes |
| collapse_knee | `mlp` | 5.3 ± 0.4 | 5.3 ± 0.4 | yes |
| collapse_knee | `gp_grid` | 3.1 ± 0.1 | 3.1 ± 0.1 | yes |
| collapse_knee | `gp_bspline` | 3.1 ± 0.1 | 3.1 ± 0.1 | yes |
| collapse_knee | `mlp_grid` | 5.3 ± 0.4 | 5.3 ± 0.4 | yes |
| collapse_knee | `gbt_grid` | 3.9 ± 0.2 | 3.9 ± 0.2 | yes |
| collapse_knee | `nn` | 3.5 ± 0.2 | 3.5 ± 0.2 | yes |
| clamp_floor | `linear` | 4.3 ± 0.2 | 4.3 ± 0.2 | yes |
| clamp_floor | `rbf` | 3.7 ± 0.2 | 3.7 ± 0.2 | yes |
| clamp_floor | `gp` | 3.2 ± 0.1 | 3.2 ± 0.1 | yes |
| clamp_floor | `sparse_physics` | 4.5 ± 0.2 | 4.5 ± 0.2 | yes |
| clamp_floor | `gp_physics_mean` | 4.4 ± 0.2 | 3.8 ± 0.2 | **no** |
| clamp_floor | `gbt` | 3.9 ± 0.1 | 3.9 ± 0.1 | yes |
| clamp_floor | `mlp` | 4.6 ± 0.3 | 4.6 ± 0.3 | yes |
| clamp_floor | `gp_grid` | 3.2 ± 0.1 | 3.2 ± 0.1 | yes |
| clamp_floor | `gp_bspline` | 3.2 ± 0.1 | 3.2 ± 0.1 | yes |
| clamp_floor | `mlp_grid` | 4.6 ± 0.3 | 4.6 ± 0.3 | yes |
| clamp_floor | `gbt_grid` | 4.0 ± 0.2 | 4.0 ± 0.2 | yes |
| clamp_floor | `nn` | 3.4 ± 0.2 | 3.4 ± 0.2 | yes |
| anisotropic | `linear` | 4.0 ± 0.2 | 4.0 ± 0.2 | yes |
| anisotropic | `rbf` | 3.6 ± 0.1 | 3.6 ± 0.1 | yes |
| anisotropic | `gp` | 3.0 ± 0.1 | 3.0 ± 0.1 | yes |
| anisotropic | `sparse_physics` | 3.0 ± 0.1 | 3.0 ± 0.1 | yes |
| anisotropic | `gp_physics_mean` | 3.0 ± 0.1 | 3.0 ± 0.1 | yes |
| anisotropic | `gbt` | 3.5 ± 0.1 | 3.5 ± 0.1 | yes |
| anisotropic | `mlp` | 4.6 ± 0.2 | 4.6 ± 0.2 | yes |
| anisotropic | `gp_grid` | 3.0 ± 0.1 | 3.0 ± 0.1 | yes |
| anisotropic | `gp_bspline` | 3.0 ± 0.1 | 3.0 ± 0.1 | yes |
| anisotropic | `mlp_grid` | 4.5 ± 0.2 | 4.5 ± 0.2 | yes |
| anisotropic | `gbt_grid` | 3.5 ± 0.2 | 3.5 ± 0.2 | yes |
| anisotropic | `nn` | 3.1 ± 0.1 | 3.1 ± 0.1 | yes |
| oscillatory | `linear` | 7.5 ± 0.5 | 7.5 ± 0.5 | yes |
| oscillatory | `rbf` | 5.7 ± 0.3 | 5.7 ± 0.3 | yes |
| oscillatory | `gp` | 3.8 ± 0.2 | 3.8 ± 0.2 | yes |
| oscillatory | `sparse_physics` | 10.4 ± 0.4 | 10.4 ± 0.4 | yes |
| oscillatory | `gp_physics_mean` | 4.0 ± 0.2 | 4.0 ± 0.2 | yes |
| oscillatory | `gbt` | 8.5 ± 0.4 | 8.5 ± 0.4 | yes |
| oscillatory | `mlp` | 9.1 ± 0.4 | 9.1 ± 0.4 | yes |
| oscillatory | `gp_grid` | 3.8 ± 0.2 | 3.8 ± 0.2 | yes |
| oscillatory | `gp_bspline` | 3.8 ± 0.2 | 3.8 ± 0.2 | yes |
| oscillatory | `mlp_grid` | 9.2 ± 0.4 | 9.2 ± 0.4 | yes |
| oscillatory | `gbt_grid` | 9.8 ± 0.4 | 9.9 ± 0.4 | yes |
| oscillatory | `nn` | 9.0 ± 0.5 | 9.0 ± 0.5 | yes |
| cliff | `linear` | 4.7 ± 0.2 | 4.7 ± 0.2 | yes |
| cliff | `rbf` | 3.9 ± 0.2 | 3.9 ± 0.2 | yes |
| cliff | `gp` | 3.1 ± 0.1 | 3.1 ± 0.1 | yes |
| cliff | `sparse_physics` | 6.3 ± 0.2 | 6.3 ± 0.2 | yes |
| cliff | `gp_physics_mean` | 3.9 ± 0.5 | 3.3 ± 0.2 | yes |
| cliff | `gbt` | 3.7 ± 0.1 | 3.7 ± 0.1 | yes |
| cliff | `mlp` | 6.8 ± 0.5 | 6.8 ± 0.5 | yes |
| cliff | `gp_grid` | 3.1 ± 0.1 | 3.1 ± 0.1 | yes |
| cliff | `gp_bspline` | 3.1 ± 0.1 | 3.1 ± 0.1 | yes |
| cliff | `mlp_grid` | 6.7 ± 0.4 | 6.7 ± 0.4 | yes |
| cliff | `gbt_grid` | 3.6 ± 0.2 | 3.8 ± 0.2 | yes |
| cliff | `nn` | 3.3 ± 0.1 | 3.3 ± 0.1 | yes |

**71/72 cells reproduced within overlapping CIs.**

## 2. Corrected: clean-truth error (30 seeds)

| surface | `linear` | `rbf` | `gp` | `sparse_physics` | `gp_physics_mean` | `gbt` | `mlp` | `gp_grid` | `gp_bspline` | `mlp_grid` | `gbt_grid` | `nn` | `gp_regime` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth | 3.50 ± 0.04 | 2.56 ± 0.04 | **0.98 ± 0.09** | 1.98 ± 0.03 | 1.80 ± 0.06 | 2.39 ± 0.03 | 3.20 ± 0.06 | **0.98 ± 0.09** | **0.99 ± 0.09** | 3.09 ± 0.06 | 2.73 ± 0.06 | 1.47 ± 0.04 | **0.98 ± 0.09** |
| collapse_knee | 4.24 ± 0.05 | 2.82 ± 0.04 | **1.13 ± 0.03** | 2.14 ± 0.03 | 1.81 ± 0.07 | 2.36 ± 0.04 | 4.39 ± 0.09 | **1.18 ± 0.03** | **1.15 ± 0.03** | 4.33 ± 0.09 | 2.68 ± 0.08 | 1.86 ± 0.07 | **1.13 ± 0.03** |
| clamp_floor | 3.37 ± 0.04 | 2.68 ± 0.04 | **1.38 ± 0.03** | 3.42 ± 0.03 | 2.54 ± 0.13 | 2.68 ± 0.04 | 3.67 ± 0.07 | **1.39 ± 0.03** | **1.38 ± 0.03** | 3.57 ± 0.07 | 2.92 ± 0.04 | 1.85 ± 0.04 | **1.38 ± 0.03** |
| anisotropic | 2.98 ± 0.03 | 2.50 ± 0.03 | **0.83 ± 0.04** | **0.76 ± 0.04** | **0.80 ± 0.04** | 2.13 ± 0.03 | 3.63 ± 0.05 | **0.83 ± 0.04** | **0.83 ± 0.04** | 3.52 ± 0.05 | 2.13 ± 0.05 | 1.24 ± 0.04 | **0.83 ± 0.04** |
| oscillatory | 7.13 ± 0.07 | 5.40 ± 0.09 | **2.40 ± 0.05** | 10.03 ± 0.05 | 2.76 ± 0.13 | 8.03 ± 0.11 | 9.08 ± 0.09 | **2.49 ± 0.05** | **2.40 ± 0.05** | 9.10 ± 0.09 | 9.39 ± 0.06 | 8.83 ± 0.16 | **2.40 ± 0.05** |
| cliff | 4.10 ± 0.05 | 2.95 ± 0.04 | **1.04 ± 0.03** | 5.53 ± 0.05 | 1.47 ± 0.09 | 2.32 ± 0.03 | 5.93 ± 0.10 | **1.09 ± 0.03** | **1.04 ± 0.03** | 5.87 ± 0.10 | 2.47 ± 0.05 | 1.59 ± 0.05 | **1.04 ± 0.03** |

Bold = best or within the best's CI.

## 3. Does the ranking change?

Spearman correlation between the published ranking and the clean-truth ranking, per surface: smooth +0.99, collapse_knee +0.96, clamp_floor +0.91, anisotropic +0.96, oscillatory +0.99, cliff +0.93.

