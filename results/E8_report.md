# E8 — upscaling for optimization (INTERIM unless every learner is complete; status {'ok': 252})

## 1. Coverage (tasks done / planned)

|  | done | planned |
|---|---|---|
| catboost | 0 | 1120 |
| gp | 0 | 1120 |
| mlp_ens | 252 | 1120 |
| pce_lars | 0 | 1120 |
| rbf_tps | 0 | 1120 |
| rf | 0 | 1120 |
| xgb | 0 | 1120 |

## 2. Accuracy: NRMSE upscaled / intermediate (geometric mean), median added error and gradient NRMSE

|  | nrmse_ratio | added | grad_nrmse | failed |
|---|---|---|---|---|
| ('mlp_ens', 'none') | 1 | 0 | 0.999 | 0 |
| ('mlp_ens', 'pou_quadratics') | 1.79 | 0.262 | 0.952 | 0 |
| ('mlp_ens', 'pspline_direct') | 3.72 | 0.44 | 0.875 | 0 |
| ('mlp_ens', 'smolyak_grid') | 1.09 | 0.031 | 1 | 0 |
| ('mlp_ens', 'smoothing_spline[m=24]') | nan | nan | nan | 1 |
| ('mlp_ens', 'subsample_ensemble') | 2.5 | 0.216 | 0.983 | 0 |
| ('mlp_ens', 'tensor_spline[m=12]') | 1.07 | 0.00509 | 0.996 | 0 |
| ('mlp_ens', 'tensor_spline[m=24]') | nan | nan | nan | 1 |
| ('mlp_ens', 'tt_grid') | 1.09 | 0.056 | 0.998 | 0 |

## 3. Success at 200d, best optimizer per learner × upscaler

| learner | none | pou_quadratics | pspline_direct | smolyak_grid | subsample_ensemble | tensor_spline[m=12] | tt_grid |
|---|---|---|---|---|---|---|---|
| mlp_ens | 0.51 | 0.50 | 0.26 | 0.47 | 0.46 | 0.51 | 0.49 |

### Paired regret ratio, upscaled / none, same optimizer and task (< 1: upscaling helps)

|  | ratio | better | p |
|---|---|---|---|
| ('mlp_ens', 'pou_quadratics') | 0.292 | 0.453 | 0.208 |
| ('mlp_ens', 'pspline_direct') | 4.35 | 0.348 | 0.135 |
| ('mlp_ens', 'smolyak_grid') | 0.972 | 0.377 | 0.679 |
| ('mlp_ens', 'subsample_ensemble') | 1.69 | 0.336 | 0.252 |
| ('mlp_ens', 'tensor_spline[m=12]') | 0.587 | 0.388 | 0.0181 |
| ('mlp_ens', 'tt_grid') | 0.73 | 0.403 | 0.169 |

## 4. Cost: median build time [s] and evaluation time per point [µs]

| up | t_build | t_eval_us |
|---|---|---|
| none | 1.05e-05 | 6.05 |
| pou_quadratics | 0.0485 | 16.3 |
| pspline_direct | 0.48 | 7.35 |
| smolyak_grid | 62.6 | 532 |
| subsample_ensemble | 0.188 | 262 |
| tensor_spline[m=12] | 16.4 | 18.9 |
| tt_grid | 0.85 | 29.9 |
