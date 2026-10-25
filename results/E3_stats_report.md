# E3 inference — E2a-selected family champions (2031977 records)

Champions chosen on E2a (independent screening data), tested on E3. Unit of analysis: problem (per-problem mean of per-cell log NRMSE ratios). Ratios are row/column NRMSE: < 1 means the row champion is more accurate.

## Champions

|  | champion | E2a mean rank |
|---|---|---|
| GP family | smt_krg | 7.09 |
| tree ensembles | catboost | 16.6 |
| networks | mlp_ens[opt=lbfgs] | 16.4 |
| kernel / RBF | rbf_tps | 16.4 |
| basis / regression | ppr | 21.5 |
| other | ens_press | 11 |

## Pairwise comparisons (Wilcoxon signed-rank, Holm; cluster-bootstrap CI; Bayesian signed-rank, ROPE ±10 %)

|  | problems | cells | ratio | ci_lo | ci_hi | P_row_better | P_equiv | P_col_better | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| ('GP family', 'tree ensembles') | 171 | 3984 | 0.367 | 0.313 | 0.425 | 1 | 0 | 0 | 1.17e-20 |
| ('GP family', 'networks') | 171 | 3987 | 0.383 | 0.254 | 0.499 | 1 | 0 | 0 | 2.47e-22 |
| ('GP family', 'kernel / RBF') | 171 | 3987 | 0.445 | 0.391 | 0.509 | 1 | 0 | 0 | 4.29e-21 |
| ('GP family', 'basis / regression') | 171 | 3987 | 0.374 | 0.322 | 0.432 | 1 | 0 | 0 | 9.53e-27 |
| ('GP family', 'other') | 171 | 3987 | 0.608 | 0.555 | 0.663 | 1 | 0 | 0 | 2.17e-19 |
| ('tree ensembles', 'networks') | 171 | 4708 | 1.04 | 0.58 | 1.53 | 0 | 0 | 1 | 0.000132 |
| ('tree ensembles', 'kernel / RBF') | 171 | 4708 | 1.19 | 1.05 | 1.34 | 0 | 0 | 1 | 0.000565 |
| ('tree ensembles', 'basis / regression') | 171 | 5187 | 0.933 | 0.836 | 1.04 | 0.888 | 0 | 0.112 | 0.258 |
| ('tree ensembles', 'other') | 171 | 4708 | 1.66 | 1.47 | 1.88 | 0 | 0 | 1 | 3.58e-15 |
| ('networks', 'kernel / RBF') | 171 | 4711 | 1.14 | 0.804 | 2 | 1 | 0 | 0.0005 | 0.0109 |
| ('networks', 'basis / regression') | 171 | 4711 | 0.948 | 0.65 | 1.7 | 1 | 0 | 0 | 1.5e-06 |
| ('networks', 'other') | 171 | 4711 | 1.59 | 1.16 | 2.73 | 0 | 0 | 1 | 2.46e-08 |
| ('kernel / RBF', 'basis / regression') | 171 | 4711 | 0.83 | 0.743 | 0.926 | 1 | 0 | 0 | 7.42e-05 |
| ('kernel / RBF', 'other') | 171 | 4711 | 1.39 | 1.28 | 1.52 | 0 | 0 | 1 | 3.58e-15 |
| ('basis / regression', 'other') | 171 | 4711 | 1.68 | 1.49 | 1.9 | 0 | 0 | 1 | 2.73e-17 |

## Performance profiles (3984 cells with every champion)

ρ(τ) = share of cells within a factor τ of the best champion; ρ(1) = win share.

|  | ρ(1) | ρ(1.1) | ρ(1.5) | ρ(2) | ρ(5) |
|---|---|---|---|---|---|
| GP family | 0.46 | 0.61 | 0.87 | 0.93 | 1.00 |
| tree ensembles | 0.16 | 0.28 | 0.44 | 0.53 | 0.75 |
| networks | 0.13 | 0.17 | 0.41 | 0.56 | 0.81 |
| kernel / RBF | 0.09 | 0.28 | 0.47 | 0.59 | 0.80 |
| basis / regression | 0.04 | 0.13 | 0.39 | 0.55 | 0.77 |
| other | 0.12 | 0.33 | 0.61 | 0.72 | 0.90 |

## All methods: performance profiles (1978 complete cells, every config)

| cfg | family | ρ(1) | ρ(1.5) | ρ(2) | ρ(5) |
|---|---|---|---|---|---|
| smt_krg | GP family | 0.20 | 0.75 | 0.86 | 0.94 |
| hetgp | GP family | 0.04 | 0.60 | 0.78 | 0.89 |
| smt_kpls | GP family | 0.06 | 0.60 | 0.73 | 0.89 |
| gp | GP family | 0.03 | 0.59 | 0.72 | 0.88 |
| as_gp | GP family | 0.12 | 0.52 | 0.67 | 0.85 |
| local_gp | GP family | 0.01 | 0.51 | 0.66 | 0.86 |
| gp_se | GP family | 0.04 | 0.46 | 0.63 | 0.83 |
| ens_press | other | 0.01 | 0.43 | 0.60 | 0.83 |
| pce_lars | basis / regression | 0.09 | 0.41 | 0.56 | 0.78 |
| treed_gp | GP family | 0.03 | 0.39 | 0.55 | 0.81 |
| krr | kernel / RBF | 0.01 | 0.33 | 0.50 | 0.75 |
| mlp_ens[opt=adam] | networks | 0.00 | 0.31 | 0.50 | 0.78 |
| dkl | networks | 0.02 | 0.29 | 0.50 | 0.81 |
| catboost | tree ensembles | 0.06 | 0.36 | 0.48 | 0.71 |
| ppr | basis / regression | 0.02 | 0.31 | 0.48 | 0.74 |
| rbf_tps | kernel / RBF | 0.01 | 0.33 | 0.48 | 0.74 |
| svr | kernel / RBF | 0.01 | 0.31 | 0.46 | 0.74 |
| mc_dropout | other | 0.00 | 0.29 | 0.45 | 0.74 |
| extratrees | tree ensembles | 0.03 | 0.35 | 0.44 | 0.68 |
| rbf_mq | kernel / RBF | 0.02 | 0.31 | 0.44 | 0.76 |
| mlp_ens[opt=lbfgs] | networks | 0.03 | 0.26 | 0.44 | 0.77 |
| xgb | tree ensembles | 0.00 | 0.28 | 0.43 | 0.66 |
| ngboost | other | 0.01 | 0.28 | 0.43 | 0.66 |
| lgbm | tree ensembles | 0.01 | 0.29 | 0.42 | 0.67 |
| rf | tree ensembles | 0.03 | 0.30 | 0.40 | 0.62 |
| gbt_committee | tree ensembles | 0.00 | 0.22 | 0.39 | 0.64 |
| ok_variogram | GP family | 0.01 | 0.27 | 0.39 | 0.66 |
| mls | kernel / RBF | 0.00 | 0.23 | 0.38 | 0.68 |
| cs_cosine | basis / regression | 0.00 | 0.24 | 0.38 | 0.67 |
| qrf | other | 0.02 | 0.25 | 0.37 | 0.60 |
| kan[opt=lbfgs] | networks | 0.01 | 0.21 | 0.36 | 0.66 |
| rsm2 | basis / regression | 0.03 | 0.22 | 0.34 | 0.60 |
| mars | basis / regression | 0.00 | 0.15 | 0.32 | 0.65 |
| lipschitz | other | 0.01 | 0.21 | 0.30 | 0.52 |
| cart | tree ensembles | 0.00 | 0.18 | 0.29 | 0.55 |
| knn | other | 0.00 | 0.22 | 0.29 | 0.52 |
| delaunay_linear | other | 0.01 | 0.20 | 0.28 | 0.51 |
| poly_ridge | other | 0.02 | 0.17 | 0.28 | 0.56 |
| idw | other | 0.02 | 0.17 | 0.26 | 0.42 |
| cs_fourier | basis / regression | 0.03 | 0.11 | 0.22 | 0.49 |
| gmdh | basis / regression | 0.01 | 0.13 | 0.20 | 0.37 |

## Best method per problem category — selected on seeds 0-4, scored on held-out seeds 5-9

`×best` = geometric-mean NRMSE of the pick over that of the held-out best (1.00 = the pick is the best).

| category | cells | overall pick (seeds 0-4) | overall: ×best on 5-9 | GP pick (seeds 0-4) | GP: ×best on 5-9 | tree pick (seeds 0-4) | tree: ×best on 5-9 | network pick (seeds 0-4) | network: ×best on 5-9 | best on seeds 5-9 | overall pick still best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| additive | 96 | smt_krg | 1.00 | smt_krg | 1.00 | catboost | 2.82 | kan[opt=lbfgs] | 1.99 | smt_krg | True |
| blind | 88 | smt_krg | 1.00 | smt_krg | 1.00 | catboost | 1.42 | mlp_ens[opt=adam] | 1.30 | smt_krg | True |
| bowl_shaped | 518 | pce_lars | 1.00 | smt_kpls | 41.54 | catboost | 433.76 | mlp_ens[opt=lbfgs] | 141.61 | pce_lars | True |
| discontinuity | 136 | rf | 1.00 | treed_gp | 1.14 | rf | 1.00 | mlp_ens[opt=adam] | 2.02 | rf | True |
| engineering | 133 | gp_se | 1.00 | gp_se | 1.00 | catboost | 5.68 | mlp_ens[opt=lbfgs] | 1.55 | hetgp | False |
| kink | 88 | gp | 1.00 | gp | 1.00 | catboost | 2.08 | mlp_ens[opt=lbfgs] | 1.77 | gp | True |
| localized | 88 | smt_kpls | 1.00 | smt_kpls | 1.00 | extratrees | 1.20 | dkl | 9737.26 | smt_kpls | True |
| lowrank | 48 | smt_kpls | 1.00 | smt_kpls | 1.00 | catboost | 2.65 | dkl | 3.34 | smt_kpls | True |
| many_local_minima | 782 | pce_lars | 1.00 | smt_kpls | 1.15 | catboost | 1.46 | mlp_ens[opt=adam] | 1.41 | pce_lars | True |
| monotone | 88 | smt_kpls | 1.00 | smt_kpls | 1.00 | catboost | 3.78 | mlp_ens[opt=lbfgs] | 1.60 | smt_kpls | True |
| multiscale | 88 | smt_kpls | 1.00 | smt_kpls | 1.00 | catboost | 12.18 | mlp_ens[opt=adam] | 2.58 | smt_kpls | True |
| nonstationary | 128 | hetgp | 1.00 | hetgp | 1.00 | xgb | 2.75 | dkl | 3.82 | gp | False |
| oscillatory | 96 | cs_fourier | 1.00 | smt_kpls | 9.96 | extratrees | 14.90 | dkl | 16.22 | cs_fourier | True |
| other | 450 | pce_lars | 1.00 | smt_kpls | 1.86 | catboost | 4.20 | mlp_ens[opt=lbfgs] | 4.61 | pce_lars | True |
| plate_shaped | 233 | pce_lars | 1.00 | smt_kpls | 1.35 | catboost | 17.54 | mlp_ens[opt=lbfgs] | 2.47 | pce_lars | True |
| pole | 96 | hetgp | 1.00 | hetgp | 1.00 | extratrees | 2.66 | mlp_ens[opt=lbfgs] | 2.63 | hetgp | True |
| polynomial | 48 | pce_lars | 1.00 | smt_kpls | 6.53 | catboost | 420.82 | mlp_ens[opt=lbfgs] | 58.60 | pce_lars | True |
| prior_draw | 240 | hetgp | 1.03 | hetgp | 1.03 | catboost | 1.17 | mlp_ens[opt=adam] | 1.35 | smt_kpls | False |
| rational | 226 | as_gp | 1.00 | as_gp | 1.00 | extratrees | 3.00 | mlp_ens[opt=lbfgs] | 1.37 | as_gp | True |
| rational_control | 36 | smt_kpls | 1.00 | smt_kpls | 1.00 | extratrees | 4.00 | dkl | 3.90 | smt_kpls | True |
| regime | 96 | as_gp | 1.00 | as_gp | 1.00 | extratrees | 1.12 | dkl | 1.14 | as_gp | True |
| ridge | 115 | as_gp | 1.00 | as_gp | 1.00 | catboost | 9.25 | mlp_ens[opt=lbfgs] | 2.42 | as_gp | True |
| smooth | 115 | gp_se | 1.01 | gp_se | 1.01 | catboost | 9.79 | mlp_ens[opt=lbfgs] | 3.53 | as_gp | False |
| steep_ridges_and_drops | 205 | catboost | 1.00 | smt_kpls | 1.36 | catboost | 1.00 | mlp_ens[opt=adam] | 1.61 | catboost | True |
| thesis | 156 | smt_krg | 1.00 | smt_krg | 1.00 | extratrees | 4.01 | mlp_ens[opt=lbfgs] | 2.72 | smt_krg | True |
| valley_shaped | 318 | pce_lars | 1.00 | smt_kpls | 9.31 | catboost | 23.88 | mlp_ens[opt=lbfgs] | 18.84 | pce_lars | True |

Overall pick confirmed as held-out best in 85% of categories.
