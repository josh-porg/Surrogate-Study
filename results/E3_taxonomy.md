# E3 data-driven taxonomy (2229 cells, 39 methods, n <= 1000)

Profile = log(NRMSE / best NRMSE in the cell). Everything is learned on seeds 0-4; selection results are scored on held-out seeds 5-9.

## 1. Method groups from behaviour (8 groups, silhouette 0.36; agreement with the nominal families: adjusted Rand 0.39, where 1 = identical, 0 = chance)

| group | members | nominal_families |
|---|---|---|
| 1 | cs_cosine, cs_fourier | basis / regression |
| 2 | cart, catboost, extratrees, gbt_committee, kan[opt=lbfgs], lgbm, ngboost, qrf, rf, xgb | networks, other, tree ensembles |
| 3 | idw, knn, krr, lipschitz, mls, ok_variogram, rbf_mq, rbf_tps, smt_kpls, svr | GP family, kernel / RBF, other |
| 4 | dkl, mc_dropout, mlp_ens[opt=adam], mlp_ens[opt=lbfgs] | networks, other |
| 5 | as_gp, gp, gp_se, hetgp, local_gp, poly_ridge, treed_gp | GP family, other |
| 6 | ppr | basis / regression |
| 7 | ens_press, gmdh, rsm2 | basis / regression, other |
| 8 | mars, pce_lars | basis / regression |

## 2. Problem types from which methods win (3 types, silhouette 0.30; agreement with the nominal categories: adjusted Rand 0.08)

Best methods are listed with their geometric-mean NRMSE relative to the per-cell best.

| type | n_problems | dims | best_methods | problems |
|---|---|---|---|---|
| 1 | 29 | d2:12 d5:11 d10:6 | smt_kpls (1.30x), catboost (1.33x), ens_press (1.34x), hetgp (1.36x) | ackley, ackley_d10, ackley_d5, axis_step_d2, composite_d10, composite_d2, composite_d5, easom, fourier_sparse_d10, fourier_sparse_d5, gp_draw_d10, gp_draw_d2, gp_draw_d5, levy, lowrank_product_d5, michal, multiscale_d2, nn_draw_d5, oblique_step_d2, oblique_step_d5, plateau_bump_d2, plateau_bump_d5, regime_d2, regime_d5, rosen_d10, stybtang_d10, stybtang_d5, tree_draw_d2, tree_draw_d5 |
| 2 | 25 | d2:11 d3:5 d5:5 d7:1 d10:3 | hetgp (1.59x), gp (1.60x), gp_se (1.91x), local_gp (1.92x) | additive_d10, additive_d2, additive_d5, axis_step_d10, axis_step_d5, branin, camel6, currin, fourier_sparse_d2, goldpr, kink_floor_d10, kink_floor_d2, kink_floor_d5, lowrank_product_d2, nn_draw_d2, nonstationary_d2, nonstationary_d5, piston, pole_knee_d2, rosen_d5, thesis_anisotropic, thesis_clamp_floor, thesis_cliff, thesis_collapse_knee, thesis_oscillatory |
| 3 | 26 | d2:9 d3:1 d4:1 d5:7 d6:1 d8:1 d10:6 | smt_kpls (2.01x), gp_se (2.12x), hetgp (2.18x), gp (2.20x) | boha1, booth, borehole, monotone_d2, monotone_d5, multiscale_d5, otlcircuit, park, pole_knee_d5, quadratic_d2, quadratic_d5, rosen, rotated_ridge_d10, rotated_ridge_d2, rotated_ridge_d2+dummy8, rotated_ridge_d5, smooth_trig_d10, smooth_trig_d2, smooth_trig_d2+dummy8, smooth_trig_d5, sumsqu, thesis_smooth, wingweight, zakharov, zakharov_d10, zakharov_d5 |

## 3. Selection gap (held-out seeds 5-9)

|  | geo_x_oracle | within_1_1 | within_2 | worst10 |
|---|---|---|---|---|
| single best (smt_kpls) | 2.56 | 0.26 | 0.72 | 4.22 |
| best per nominal category (26) | 1.49 | 0.53 | 0.87 | 2.41 |
| best per data-driven type (3 types) | 2.46 | 0.30 | 0.78 | 3.79 |
| best per data-driven type (26 types) | 1.37 | 0.56 | 0.92 | 1.85 |
| best per problem | 1.18 | 0.66 | 0.95 | 1.45 |
| oracle per cell | 1.00 | 1.00 | 1.00 | 1.00 |

38 of 39 methods are the best in at least one cell.
