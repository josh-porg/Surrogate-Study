# E8 — upscaling for optimization (INTERIM unless every learner is complete; status {'ok': 7760, 'timeout': 80})

## 1. Coverage (tasks done / planned)

|  | done | planned |
|---|---|---|
| catboost | 1120 | 1120 |
| gp | 1120 | 1120 |
| mlp_ens | 1080 | 1120 |
| pce_lars | 1120 | 1120 |
| rbf_tps | 1120 | 1120 |
| rf | 1080 | 1120 |
| xgb | 1120 | 1120 |

Note: `pspline_direct` and `subsample_ensemble` ignore the learner and refit the training data (they are replacements, not upscalers: identical results across learners). The upscalers proper, built on the learner's predictions, are the tensor and smoothing splines, Smolyak, TT and pou_quadratics. The m = 24 splines are infeasible at d = 5, so their rows cover d = 2 only.

## 2. Accuracy: NRMSE upscaled / intermediate (geometric mean), median added error and gradient NRMSE

|  | nrmse_ratio | added | grad_nrmse | failed |
|---|---|---|---|---|
| ('catboost', 'none') | 1 | 0 | 5.71 | 0 |
| ('catboost', 'pou_quadratics') | 1.47 | 0.263 | 0.787 | 0 |
| ('catboost', 'pspline_direct') | 0.964 | 0.272 | 0.822 | 0 |
| ('catboost', 'smolyak_grid') | 1.01 | 0.0331 | 2 | 0 |
| ('catboost', 'smoothing_spline[m=24]') | 1.01 | 0.11 | 0.645 | 0.679 |
| ('catboost', 'subsample_ensemble') | 0.892 | 0.242 | 0.669 | 0 |
| ('catboost', 'tensor_spline[m=12]') | 1.14 | 0.0967 | 0.777 | 0.0357 |
| ('catboost', 'tensor_spline[m=24]') | 1.02 | 0.104 | 0.802 | 0.679 |
| ('catboost', 'tt_grid') | 1.05 | 0.075 | 0.99 | 0 |
| ('gp', 'none') | 1 | 0 | 0.298 | 0 |
| ('gp', 'pou_quadratics') | 2.71 | 0.253 | 0.778 | 0 |
| ('gp', 'pspline_direct') | 3.38 | 0.268 | 0.822 | 0 |
| ('gp', 'smolyak_grid') | 1.01 | 0.000403 | 0.298 | 0 |
| ('gp', 'smoothing_spline[m=24]') | 1.24 | 0.00529 | 0.194 | 0.679 |
| ('gp', 'subsample_ensemble') | 3.13 | 0.174 | 0.669 | 0 |
| ('gp', 'tensor_spline[m=12]') | 1.17 | 0.00261 | 0.299 | 0.0357 |
| ('gp', 'tensor_spline[m=24]') | 1.03 | 0.0004 | 0.181 | 0.679 |
| ('gp', 'tt_grid') | 1.06 | 0.00583 | 0.334 | 0 |
| ('mlp_ens', 'none') | 1 | 0 | 0.389 | 0 |
| ('mlp_ens', 'pou_quadratics') | 2.29 | 0.232 | 0.775 | 0 |
| ('mlp_ens', 'pspline_direct') | 2.61 | 0.292 | 0.823 | 0 |
| ('mlp_ens', 'smolyak_grid') | 1.03 | 0.000346 | 0.389 | 0 |
| ('mlp_ens', 'smoothing_spline[m=24]') | 1.03 | 0.00855 | 0.389 | 0.667 |
| ('mlp_ens', 'subsample_ensemble') | 2.31 | 0.174 | 0.63 | 0 |
| ('mlp_ens', 'tensor_spline[m=12]') | 1.07 | 0.00296 | 0.397 | 0 |
| ('mlp_ens', 'tensor_spline[m=24]') | 1.01 | 0.000648 | 0.392 | 0.667 |
| ('mlp_ens', 'tt_grid') | 1.03 | 0.00463 | 0.389 | 0 |
| ('pce_lars', 'none') | 1 | 0 | 0.597 | 0 |
| ('pce_lars', 'pou_quadratics') | 37.1 | 0.136 | 0.771 | 0 |
| ('pce_lars', 'pspline_direct') | 61.2 | 0.226 | 0.822 | 0 |
| ('pce_lars', 'smolyak_grid') | 1.27 | 1.14e-14 | 0.597 | 0 |
| ('pce_lars', 'smoothing_spline[m=24]') | 121 | 0.00618 | 0.213 | 0.679 |
| ('pce_lars', 'subsample_ensemble') | 56.6 | 0.195 | 0.669 | 0 |
| ('pce_lars', 'tensor_spline[m=12]') | 21.1 | 5.63e-05 | 0.56 | 0.0357 |
| ('pce_lars', 'tensor_spline[m=24]') | 56.9 | 0.000305 | 0.233 | 0.679 |
| ('pce_lars', 'tt_grid') | 1.17 | 2.92e-15 | 0.597 | 0 |
| ('rbf_tps', 'none') | 1 | 0 | 0.609 | 0 |
| ('rbf_tps', 'pou_quadratics') | 1.27 | 0.178 | 0.771 | 0 |
| ('rbf_tps', 'pspline_direct') | 1.12 | 0.26 | 0.822 | 0 |
| ('rbf_tps', 'smolyak_grid') | 1 | 0.0057 | 0.609 | 0 |
| ('rbf_tps', 'smoothing_spline[m=24]') | 1.01 | 0.00427 | 0.464 | 0.679 |
| ('rbf_tps', 'subsample_ensemble') | 1.04 | 0.0921 | 0.669 | 0 |
| ('rbf_tps', 'tensor_spline[m=12]') | 1.01 | 0.000491 | 0.588 | 0.0357 |
| ('rbf_tps', 'tensor_spline[m=24]') | 0.999 | 0.00101 | 0.466 | 0.679 |
| ('rbf_tps', 'tt_grid') | 1.03 | 0.0227 | 0.609 | 0 |
| ('rf', 'none') | 1 | 0 | 6.14 | 0 |
| ('rf', 'pou_quadratics') | 1.33 | 0.252 | 0.804 | 0 |
| ('rf', 'pspline_direct') | 0.674 | 0.33 | 0.823 | 0 |
| ('rf', 'smolyak_grid') | 1.02 | 0.0387 | 1.91 | 0 |
| ('rf', 'smoothing_spline[m=24]') | 1.01 | 0.102 | 0.739 | 0.667 |
| ('rf', 'subsample_ensemble') | 0.597 | 0.35 | 0.63 | 0 |
| ('rf', 'tensor_spline[m=12]') | 1.1 | 0.105 | 0.912 | 0 |
| ('rf', 'tensor_spline[m=24]') | 1.02 | 0.0903 | 0.854 | 0.667 |
| ('rf', 'tt_grid') | 1.05 | 0.0705 | 1 | 0 |
| ('xgb', 'none') | 1 | 0 | 1.33 | 0 |
| ('xgb', 'pou_quadratics') | 1.34 | 0.292 | 0.793 | 0 |
| ('xgb', 'pspline_direct') | 0.771 | 0.292 | 0.822 | 0 |
| ('xgb', 'smolyak_grid') | 1.01 | 0.0529 | 2.91 | 0 |
| ('xgb', 'smoothing_spline[m=24]') | 0.934 | 0.166 | 0.705 | 0.679 |
| ('xgb', 'subsample_ensemble') | 0.714 | 0.305 | 0.669 | 0 |
| ('xgb', 'tensor_spline[m=12]') | 1.08 | 0.157 | 0.929 | 0.0357 |
| ('xgb', 'tensor_spline[m=24]') | 1.01 | 0.154 | 0.925 | 0.679 |
| ('xgb', 'tt_grid') | 1.04 | 0.115 | 1.09 | 0 |

## 3. Success at 200d, best optimizer per learner × upscaler

| learner | none | pou_quadratics | pspline_direct | smolyak_grid | smoothing_spline[m=24] | subsample_ensemble | tensor_spline[m=12] | tensor_spline[m=24] | tt_grid |
|---|---|---|---|---|---|---|---|---|---|
| catboost | 0.27 | 0.40 | 0.35 | 0.25 | 0.41 | 0.47 | 0.27 | 0.33 | 0.24 |
| gp | 0.54 | 0.44 | 0.35 | 0.55 | 0.56 | 0.47 | 0.57 | 0.57 | 0.55 |
| mlp_ens | 0.55 | 0.45 | 0.35 | 0.53 | 0.44 | 0.48 | 0.55 | 0.44 | 0.53 |
| pce_lars | 0.53 | 0.45 | 0.35 | 0.53 | 0.53 | 0.47 | 0.54 | 0.54 | 0.53 |
| rbf_tps | 0.42 | 0.46 | 0.35 | 0.43 | 0.46 | 0.47 | 0.43 | 0.47 | 0.40 |
| rf | 0.30 | 0.35 | 0.35 | 0.30 | 0.47 | 0.48 | 0.30 | 0.39 | 0.26 |
| xgb | 0.23 | 0.37 | 0.35 | 0.21 | 0.38 | 0.47 | 0.26 | 0.26 | 0.22 |

### Paired regret ratio, upscaled / none, same optimizer and task (< 1: upscaling helps)

|  | ratio | better | p |
|---|---|---|---|
| ('catboost', 'pou_quadratics') | 0.0496 | 0.63 | 0.00312 |
| ('catboost', 'pspline_direct') | 0.0985 | 0.594 | 0.00172 |
| ('catboost', 'smolyak_grid') | 0.549 | 0.56 | 2.36e-07 |
| ('catboost', 'smoothing_spline[m=24]') | 0.264 | 0.642 | 0.0273 |
| ('catboost', 'subsample_ensemble') | 0.0396 | 0.689 | 9.16e-06 |
| ('catboost', 'tensor_spline[m=12]') | 0.34 | 0.637 | 3.08e-06 |
| ('catboost', 'tensor_spline[m=24]') | 0.456 | 0.615 | 0.0195 |
| ('catboost', 'tt_grid') | 0.445 | 0.602 | 3.46e-07 |
| ('gp', 'pou_quadratics') | 1.17 | 0.341 | 0.943 |
| ('gp', 'pspline_direct') | 5.06 | 0.296 | 0.0719 |
| ('gp', 'smolyak_grid') | 1.01 | 0.319 | 0.309 |
| ('gp', 'smoothing_spline[m=24]') | 1.05 | 0.411 | 0.734 |
| ('gp', 'subsample_ensemble') | 2.04 | 0.318 | 0.136 |
| ('gp', 'tensor_spline[m=12]') | 1.01 | 0.317 | 0.809 |
| ('gp', 'tensor_spline[m=24]') | 0.959 | 0.364 | 0.82 |
| ('gp', 'tt_grid') | 0.989 | 0.347 | 0.171 |
| ('mlp_ens', 'pou_quadratics') | 0.741 | 0.379 | 0.386 |
| ('mlp_ens', 'pspline_direct') | 3.14 | 0.364 | 0.239 |
| ('mlp_ens', 'smolyak_grid') | 0.682 | 0.381 | 0.00271 |
| ('mlp_ens', 'smoothing_spline[m=24]') | 0.751 | 0.424 | 0.203 |
| ('mlp_ens', 'subsample_ensemble') | 1.17 | 0.39 | 0.878 |
| ('mlp_ens', 'tensor_spline[m=12]') | 0.607 | 0.377 | 2.59e-05 |
| ('mlp_ens', 'tensor_spline[m=24]') | 0.77 | 0.404 | 0.0977 |
| ('mlp_ens', 'tt_grid') | 0.656 | 0.39 | 0.000828 |
| ('pce_lars', 'pou_quadratics') | 1.69 | 0.282 | 0.794 |
| ('pce_lars', 'pspline_direct') | 9.86 | 0.287 | 0.0552 |
| ('pce_lars', 'smolyak_grid') | 0.996 | 0.209 | 0.824 |
| ('pce_lars', 'smoothing_spline[m=24]') | 1.13 | 0.349 | 1 |
| ('pce_lars', 'subsample_ensemble') | 3.97 | 0.324 | 0.125 |
| ('pce_lars', 'tensor_spline[m=12]') | 1.16 | 0.257 | 0.696 |
| ('pce_lars', 'tensor_spline[m=24]') | 1.03 | 0.314 | 1 |
| ('pce_lars', 'tt_grid') | 0.994 | 0.204 | 0.622 |
| ('rbf_tps', 'pou_quadratics') | 0.779 | 0.356 | 0.894 |
| ('rbf_tps', 'pspline_direct') | 2.29 | 0.311 | 0.0627 |
| ('rbf_tps', 'smolyak_grid') | 1.04 | 0.345 | 0.658 |
| ('rbf_tps', 'smoothing_spline[m=24]') | 1.05 | 0.39 | 0.359 |
| ('rbf_tps', 'subsample_ensemble') | 0.923 | 0.398 | 0.558 |
| ('rbf_tps', 'tensor_spline[m=12]') | 1.06 | 0.285 | 0.228 |
| ('rbf_tps', 'tensor_spline[m=24]') | 0.992 | 0.371 | 0.82 |
| ('rbf_tps', 'tt_grid') | 1.23 | 0.321 | 0.000529 |
| ('rf', 'pou_quadratics') | 0.0655 | 0.598 | 0.00594 |
| ('rf', 'pspline_direct') | 0.103 | 0.587 | 0.00388 |
| ('rf', 'smolyak_grid') | 0.602 | 0.527 | 2.59e-05 |
| ('rf', 'smoothing_spline[m=24]') | 0.183 | 0.682 | 0.00781 |
| ('rf', 'subsample_ensemble') | 0.0384 | 0.687 | 6.66e-06 |
| ('rf', 'tensor_spline[m=12]') | 0.364 | 0.623 | 6.41e-07 |
| ('rf', 'tensor_spline[m=24]') | 0.36 | 0.623 | 0.00781 |
| ('rf', 'tt_grid') | 0.547 | 0.567 | 8.37e-05 |
| ('xgb', 'pou_quadratics') | 0.0669 | 0.612 | 0.00664 |
| ('xgb', 'pspline_direct') | 0.117 | 0.593 | 0.00147 |
| ('xgb', 'smolyak_grid') | 0.811 | 0.464 | 0.0027 |
| ('xgb', 'smoothing_spline[m=24]') | 0.193 | 0.618 | 0.0273 |
| ('xgb', 'subsample_ensemble') | 0.0472 | 0.686 | 7.01e-06 |
| ('xgb', 'tensor_spline[m=12]') | 0.354 | 0.591 | 1.14e-05 |
| ('xgb', 'tensor_spline[m=24]') | 0.678 | 0.521 | 0.0547 |
| ('xgb', 'tt_grid') | 0.661 | 0.519 | 4.01e-05 |

## 4. Cost: median build time [s] and evaluation time per point [µs]

| up | t_build | t_eval_us |
|---|---|---|
| none | 9.86e-06 | 4.12 |
| pou_quadratics | 0.0432 | 15.3 |
| pspline_direct | 0.433 | 7.05 |
| smolyak_grid | 57.7 | 440 |
| smoothing_spline[m=24] | 1.56 | 0.515 |
| subsample_ensemble | 0.0243 | 67 |
| tensor_spline[m=12] | 10.8 | 17.2 |
| tensor_spline[m=24] | 0.00886 | 0.517 |
| tt_grid | 0.91 | 27.5 |
