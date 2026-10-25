# E3 data-driven taxonomy (4647 cells, 38 methods, n <= 1000)

Profile = log(NRMSE / best NRMSE in the cell). Everything is learned on seeds 0-4; selection results are scored on held-out seeds 5-9.

## 1. Method groups from behaviour (11 groups, silhouette 0.35; agreement with the nominal families: adjusted Rand 0.47, where 1 = identical, 0 = chance)

| group | members | nominal_families |
|---|---|---|
| 1 | cart, catboost, extratrees, gbt_committee, lgbm, ngboost, qrf, rf, xgb | other, tree ensembles |
| 2 | cs_cosine, cs_fourier | basis / regression |
| 3 | as_gp, gp, gp_se, hetgp, local_gp, treed_gp | GP family |
| 4 | dkl, mlp_ens[opt=adam], mlp_ens[opt=lbfgs] | networks |
| 5 | mc_dropout, poly_ridge | other |
| 6 | ppr | basis / regression |
| 7 | idw, knn, lipschitz, ok_variogram | GP family, other |
| 8 | krr, pce_lars, svr | basis / regression, kernel / RBF |
| 9 | mls, rbf_mq, rbf_tps, smt_kpls | GP family, kernel / RBF |
| 10 | ens_press, gmdh, rsm2 | basis / regression, other |
| 11 | mars | basis / regression |

## 2. Problem types from which methods win (3 types, silhouette 0.39; agreement with the nominal categories: adjusted Rand 0.08)

Best methods are listed with their geometric-mean NRMSE relative to the per-cell best.

| type | n_problems | dims | best_methods | problems |
|---|---|---|---|---|
| 1 | 71 | d2:20 d3:5 d5:18 d8:9 d10:19 | catboost (1.25x), ens_press (1.33x), lgbm (1.33x), xgb (1.37x) | ackley, ackley_d10, ackley_d3, ackley_d5, ackley_d8, additive_d10, additive_d5, axis_step_d2, composite_d10, composite_d2, composite_d5, dejong5, dixonpr_d10, dixonpr_d8, drop, easom, fourier_sparse_d10, fourier_sparse_d5, gp_draw_d10, gp_draw_d2, gp_draw_d5, langer, levy, levy13, levy_d10, levy_d3, levy_d5, levy_d8, lowrank_product_d10, lowrank_product_d5, michal, michal_d10, michal_d3, michal_d5, michal_d8, multiscale_d2, nn_draw_d10, nn_draw_d5, nonstationary_d2, nonstationary_d5, oblique_step_d10, oblique_step_d2, oblique_step_d5, perm0db_d10, perm0db_d5, perm0db_d8, permdb_d10, permdb_d5, permdb_d8, plateau_bump_d10, plateau_bump_d2, plateau_bump_d5, rastr, rastr_d10, rastr_d3, rastr_d5, rastr_d8, regime_d10, regime_d2, regime_d5, rosen_d10, rosen_d8, schaffer2, shubert, stybtang_d10, stybtang_d5, stybtang_d8, thesis_oscillatory, tree_draw_d10, tree_draw_d2, tree_draw_d5 |
| 2 | 24 | d2:7 d3:3 d5:5 d8:4 d10:5 | pce_lars (1.67x), rsm2 (2.09x), smt_kpls (3.60x), gp_se (4.60x) | boha1, booth, griewank_d10, griewank_d5, griewank_d8, matya, quadratic_d10, quadratic_d2, quadratic_d5, rothyp, rothyp_d10, rothyp_d3, rothyp_d5, rothyp_d8, sumsqu, sumsqu_d10, sumsqu_d3, sumsqu_d5, sumsqu_d8, trid, trid_d10, trid_d3, trid_d5, trid_d8 |
| 3 | 74 | d1:5 d2:27 d3:13 d4:1 d5:11 d6:1 d7:1 d8:2 d10:13 | smt_kpls (1.91x), hetgp (1.97x), gp (1.97x), gp_se (2.15x) | abs_kink, additive_d2, axis_step_d10, axis_step_d5, beale, borehole, branin, braninmodif, camel3, camel6, currin, dixonpr, dixonpr_d3, dixonpr_d5, forrester, fourier_sparse_d2, goldpr, griewank, griewank_d3, kink_floor_d10, kink_floor_d2, kink_floor_d5, lowrank_product_d2, mccorm, monotone_d10, monotone_d2, monotone_d5, multiscale_d10, multiscale_d5, near_pole, nn_draw_d2, nonstationary_d10, otlcircuit, park, perm0db, perm0db_d3, permdb, permdb_d3, piston, pole_knee_d10, pole_knee_d2, pole_knee_d5, powersum, ratio2d, resonance2d, ridge_pole_d10, ridge_pole_d3, ridge_pole_d5, rosen, rosen_d3, rosen_d5, rotated_ridge_d10, rotated_ridge_d2, rotated_ridge_d2+dummy8, rotated_ridge_d5, runge, smooth_trig_d10, smooth_trig_d2, smooth_trig_d2+dummy8, smooth_trig_d5, step1d, stybtang, stybtang_d3, thesis_anisotropic, thesis_clamp_floor, thesis_cliff, thesis_collapse_knee, thesis_smooth, wingweight, zakharov, zakharov_d10, zakharov_d3, zakharov_d5, zakharov_d8 |

## 3. Selection gap (held-out seeds 5-9)

|  | geo_x_oracle | within_1_1 | within_2 | worst10 |
|---|---|---|---|---|
| single best (pce_lars) | 2.79 | 0.32 | 0.65 | 12.09 |
| best per nominal category (26) | 1.65 | 0.50 | 0.86 | 2.80 |
| best per data-driven type (3 types) | 2.37 | 0.37 | 0.79 | 3.66 |
| best per data-driven type (26 types) | 1.42 | 0.56 | 0.91 | 1.91 |
| best per problem | 1.19 | 0.69 | 0.95 | 1.48 |
| oracle per cell | 1.00 | 1.00 | 1.00 | 1.00 |

37 of 38 methods are the best in at least one cell.
