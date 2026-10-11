# E2a — learner screening

Tasks: {'ok': 32300, 'infeasible': 60, 'timeout': 40}. Cells with ≥ 5 configs × ≥ 3 seeds: 72.

## Overall mean rank (lower = better) and cells won (best or within best's 95 % CI)

| config | mean rank | cells won | median grad NRMSE | median τ (best 10 %) | 95 % coverage | median fit s | keep |
|---|---|---|---|---|---|---|---|
| `smt_krg` | 7.4 | 18 | 0.466 | 0.691 | 0.91 | 7.03 | **yes** |
| `hetgp` | 8.0 | 2 | 0.386 | 0.756 | 0.948 | 4.96 | **yes** |
| `smt_kpls` | 9.0 | 10 | 0.452 | 0.613 | 0.889 | 3.86 | **yes** |
| `gp` | 9.6 | 9 | 0.525 | 0.707 | 0.961 | 1.45 | **yes** |
| `ens_press` | 11.5 | 4 | 1.83 | 0.526 | 0.961 | 5.58 | **yes** |
| `local_gp` | 12.9 | 4 | 0.815 | 0.517 | 0.953 | 1.44 | **yes** |
| `gp_se` | 14.5 | 8 | 0.482 | 0.584 | 0.956 | 1.06 | **yes** |
| `treed_gp` | 15.1 | 7 | 0.482 | 0.713 | 0.94 | 0.896 | **yes** |
| `as_gp` | 16.6 | 14 | 0.59 | 0.619 | 0.956 | 3.31 | **yes** |
| `mlp_ens[opt=lbfgs]` | 17.4 | 9 | 0.777 | 0.567 | 0.904 | 16.9 | **yes** |
| `catboost` | 18.0 | 12 | 9.15 | 0.477 | — | 0.868 | **yes** |
| `rbf_tps` | 18.1 | 4 | 0.61 | 0.401 | — | 0.0472 | **yes** |
| `krr` | 18.8 | 5 | 0.739 | 0.329 | — | 0.611 | **yes** |
| `mlp_ens[opt=adamw]` | 19.1 | 0 | 0.777 | 0.538 | 0.853 | 19.1 |  |
| `mc_dropout` | 19.2 | 1 | 519 | 0.444 | 0.937 | 4.72 |  |
| `rbf_mq` | 19.2 | 5 | 0.607 | 0.452 | — | 0.155 | **yes** |
| `mlp_ens[opt=adam]` | 19.4 | 0 | 0.773 | 0.54 | 0.839 | 19.9 | **yes** |
| `ok_variogram` | 19.8 | 6 | 0.753 | 0.327 | 0.998 | 0.0828 | **yes** |
| `dkl` | 19.8 | 4 | 0.922 | 0.634 | 0.98 | 4.87 | **yes** |
| `extratrees` | 20.9 | 4 | 6.39 | 0.439 | 0.994 | 0.593 | **yes** |
| `lgbm` | 23.5 | 0 | 1.55 | 0.439 | — | 2.72 | **yes** |
| `ppr` | 23.5 | 2 | 0.878 | 0.532 | — | 3.21 | **yes** |
| `xgb` | 23.8 | 0 | 1.21 | 0.483 | — | 1.23 | **yes** |
| `mlp_ens[opt=bb]` | 24.0 | 0 | 0.783 | 0.426 | 0.713 | 24.2 |  |
| `svr` | 24.2 | 5 | 0.801 | 0.271 | — | 2.2 | **yes** |
| `cs_cosine` | 24.4 | 0 | 0.821 | 0.263 | — | 0.0573 | **yes** |
| `kan[opt=lbfgs]` | 24.9 | 0 | 1.01 | 0.275 | — | 1.61 | **yes** |
| `rf` | 25.2 | 8 | 7.22 | 0.397 | 0.988 | 0.976 | **yes** |
| `kan[opt=adam]` | 25.5 | 0 | 0.995 | 0.244 | — | 2.56 |  |
| `mlp_ens[opt=sgd]` | 25.7 | 1 | 0.792 | 0.437 | 0.751 | 20.8 |  |
| `ngboost` | 26.0 | 8 | 4.32 | 0.445 | 0.317 | 6.1 |  |
| `pce_lars` | 26.2 | 1 | 0.957 | 0.278 | — | 0.0521 | **yes** |
| `gbt_committee` | 26.8 | 0 | 4.51 | 0.422 | 0.822 | 1.55 | **yes** |
| `mls` | 27.0 | 0 | 1 | 0.394 | — | 0.0002 | **yes** |
| `qrf` | 27.0 | 7 | 4.59 | 0.372 | 0.995 | 0.62 |  |
| `mars` | 30.2 | 2 | 0.969 | 0.43 | — | 0.749 | **yes** |
| `knn` | 30.5 | 2 | 0.983 | 0.283 | — | 0.148 |  |
| `lipschitz` | 31.2 | 2 | 1.08 | 0.253 | 1 | 0.00137 | **yes** |
| `delaunay_linear` | 32.2 | 2 | 1.25 | 0.264 | — | 0.0115 | **yes** |
| `rsm2` | 32.7 | 0 | 0.951 | 0.251 | — | 0.00101 | **yes** |
| `cart` | 33.4 | 7 | 1.03 | 0.273 | — | 0.101 | **yes** |
| `poly_ridge` | 35.1 | 0 | 0.993 | 0.177 | — | 0.119 |  |
| `idw` | 36.9 | 4 | 0.981 | 0.194 | — | 0.000191 |  |
| `cs_fourier` | 37.3 | 6 | 1.59 | 0.0169 | — | 0.199 | **yes** |
| `gmdh` | 40.4 | 0 | 1 | 0.26 | — | 0.0318 | **yes** |

## Best config per screening problem (mean rank over its n × noise cells)

- **axis_step_d5**: `rf` (1.8), `treed_gp` (2.0), `qrf` (2.7), `ngboost` (3.7), `cart` (4.8)
- **borehole**: `gp_se` (1.3), `gp` (3.0), `hetgp` (3.0), `smt_kpls` (3.3), `mlp_ens[opt=lbfgs]` (5.8)
- **composite_d5**: `smt_krg` (2.0), `gp` (3.3), `hetgp` (3.3), `mlp_ens[opt=lbfgs]` (5.0), `as_gp` (6.0)
- **fourier_sparse_d2**: `cs_fourier` (1.3), `smt_kpls` (4.8), `ok_variogram` (5.3), `smt_krg` (8.3), `gp_se` (8.8)
- **gp_draw_d5**: `krr` (2.3), `svr` (3.5), `hetgp` (6.7), `gp` (7.0), `rbf_mq` (7.0)
- **kink_floor_d2**: `as_gp` (2.3), `gp` (3.0), `local_gp` (3.2), `hetgp` (5.2), `treed_gp` (7.0)
- **nonstationary_d2**: `dkl` (5.0), `extratrees` (5.0), `smt_krg` (5.2), `ngboost` (5.3), `catboost` (6.3)
- **plateau_bump_d2**: `smt_kpls` (2.8), `rbf_mq` (4.2), `rbf_tps` (5.0), `ok_variogram` (7.0), `delaunay_linear` (8.7)
- **rotated_ridge_d5**: `as_gp` (1.2), `mlp_ens[opt=lbfgs]` (2.2), `gp` (5.7), `hetgp` (6.2), `gp_se` (7.0)
- **smooth_trig_d5**: `as_gp` (1.8), `gp_se` (2.5), `smt_kpls` (3.0), `smt_krg` (5.3), `gp` (5.5)
- **thesis_collapse_knee**: `smt_krg` (1.2), `gp` (3.2), `hetgp` (3.5), `local_gp` (5.5), `treed_gp` (9.0)
- **tree_draw_d5**: `catboost` (1.0), `lgbm` (2.7), `xgb` (3.3), `ens_press` (4.7), `ngboost` (5.7)

## Network training optimizer (mlp_ens / kan)

- mlp_ens: `mlp_ens[opt=lbfgs]` (17.4) < `mlp_ens[opt=adamw]` (19.1) < `mlp_ens[opt=adam]` (19.4) < `mlp_ens[opt=bb]` (24.0) < `mlp_ens[opt=sgd]` (25.7)
- kan: `kan[opt=lbfgs]` (24.9) < `kan[opt=adam]` (25.5)
