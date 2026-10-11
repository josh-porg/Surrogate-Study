# E3 inference — E2a-selected family champions (791734 records; excluded: smt_kpls, smt_krg)

Champions chosen on E2a (independent screening data), tested on E3. Unit of analysis: problem (per-problem mean of per-cell log NRMSE ratios). Ratios are row/column NRMSE: < 1 means the row champion is more accurate.

## Champions

|  | champion | E2a mean rank |
|---|---|---|
| GP family | hetgp | 6.1 |
| tree ensembles | catboost | 13.7 |
| networks | mlp_ens[opt=lbfgs] | 13.2 |
| kernel / RBF | rbf_tps | 13.2 |
| basis / regression | ppr | 17.6 |
| other | ens_press | 8.85 |

## Pairwise comparisons (Wilcoxon signed-rank, Holm; cluster-bootstrap CI; Bayesian signed-rank, ROPE ±10 %)

|  | problems | cells | ratio | ci_lo | ci_hi | P_row_better | P_equiv | P_col_better | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| ('GP family', 'tree ensembles') | 82 | 2255 | 0.453 | 0.359 | 0.565 | 1 | 0 | 0 | 5.3e-09 |
| ('GP family', 'networks') | 82 | 2258 | 0.577 | 0.497 | 0.671 | 1 | 0 | 0 | 6.68e-10 |
| ('GP family', 'kernel / RBF') | 82 | 2258 | 0.485 | 0.393 | 0.588 | 1 | 0 | 0 | 5.44e-11 |
| ('GP family', 'basis / regression') | 82 | 2258 | 0.482 | 0.399 | 0.572 | 1 | 0 | 0 | 4.11e-12 |
| ('GP family', 'other') | 82 | 2258 | 0.69 | 0.614 | 0.769 | 1 | 0 | 0 | 6.15e-09 |
| ('tree ensembles', 'networks') | 82 | 2255 | 1.27 | 1 | 1.63 | 0.002 | 0 | 0.998 | 0.0245 |
| ('tree ensembles', 'kernel / RBF') | 82 | 2255 | 1.07 | 0.901 | 1.27 | 0.0607 | 0 | 0.939 | 0.426 |
| ('tree ensembles', 'basis / regression') | 82 | 2481 | 1.03 | 0.892 | 1.19 | 0.515 | 0 | 0.484 | 0.991 |
| ('tree ensembles', 'other') | 82 | 2255 | 1.52 | 1.31 | 1.8 | 0 | 0 | 1 | 1.21e-07 |
| ('networks', 'kernel / RBF') | 82 | 2258 | 0.841 | 0.694 | 1.02 | 0.998 | 0 | 0.002 | 0.0316 |
| ('networks', 'basis / regression') | 82 | 2258 | 0.835 | 0.684 | 1.01 | 1 | 0 | 0 | 0.00471 |
| ('networks', 'other') | 82 | 2258 | 1.2 | 1.04 | 1.37 | 0.0005 | 0 | 1 | 0.00546 |
| ('kernel / RBF', 'basis / regression') | 82 | 2258 | 0.993 | 0.835 | 1.17 | 0.865 | 0 | 0.136 | 0.509 |
| ('kernel / RBF', 'other') | 82 | 2258 | 1.42 | 1.27 | 1.62 | 0 | 0 | 1 | 3.14e-08 |
| ('basis / regression', 'other') | 82 | 2258 | 1.43 | 1.26 | 1.64 | 0 | 0 | 1 | 9.8e-08 |

## Performance profiles (2255 cells with every champion)

ρ(τ) = share of cells within a factor τ of the best champion; ρ(1) = win share.

|  | ρ(1) | ρ(1.1) | ρ(1.5) | ρ(2) | ρ(5) |
|---|---|---|---|---|---|
| GP family | 0.44 | 0.58 | 0.83 | 0.92 | 0.99 |
| tree ensembles | 0.17 | 0.27 | 0.45 | 0.55 | 0.77 |
| networks | 0.16 | 0.22 | 0.43 | 0.58 | 0.88 |
| kernel / RBF | 0.08 | 0.22 | 0.40 | 0.56 | 0.80 |
| basis / regression | 0.07 | 0.15 | 0.38 | 0.58 | 0.82 |
| other | 0.09 | 0.25 | 0.55 | 0.72 | 0.92 |

## All methods: performance profiles (2041 complete cells, every config)

| cfg | family | ρ(1) | ρ(1.5) | ρ(2) | ρ(5) |
|---|---|---|---|---|---|
| hetgp | GP family | 0.07 | 0.64 | 0.81 | 0.93 |
| gp | GP family | 0.06 | 0.61 | 0.73 | 0.90 |
| as_gp | GP family | 0.14 | 0.54 | 0.69 | 0.87 |
| local_gp | GP family | 0.02 | 0.52 | 0.68 | 0.88 |
| gp_se | GP family | 0.09 | 0.50 | 0.67 | 0.86 |
| ens_press | other | 0.01 | 0.45 | 0.63 | 0.85 |
| pce_lars | basis / regression | 0.11 | 0.43 | 0.58 | 0.80 |
| treed_gp | GP family | 0.03 | 0.40 | 0.57 | 0.84 |
| dkl | networks | 0.03 | 0.32 | 0.54 | 0.83 |
| krr | kernel / RBF | 0.03 | 0.35 | 0.53 | 0.78 |
| mlp_ens[opt=adam] | networks | 0.00 | 0.33 | 0.52 | 0.80 |
| ppr | basis / regression | 0.03 | 0.33 | 0.52 | 0.78 |
| catboost | tree ensembles | 0.06 | 0.38 | 0.51 | 0.73 |
| rbf_tps | kernel / RBF | 0.02 | 0.33 | 0.50 | 0.77 |
| svr | kernel / RBF | 0.01 | 0.32 | 0.48 | 0.76 |
| rbf_mq | kernel / RBF | 0.03 | 0.32 | 0.48 | 0.78 |
| mlp_ens[opt=lbfgs] | networks | 0.06 | 0.29 | 0.47 | 0.80 |
| extratrees | tree ensembles | 0.04 | 0.36 | 0.46 | 0.70 |
| xgb | tree ensembles | 0.00 | 0.31 | 0.46 | 0.70 |
| lgbm | tree ensembles | 0.02 | 0.31 | 0.45 | 0.69 |
| rf | tree ensembles | 0.05 | 0.32 | 0.42 | 0.66 |
| gbt_committee | tree ensembles | 0.00 | 0.24 | 0.41 | 0.66 |
| cs_cosine | basis / regression | 0.01 | 0.25 | 0.40 | 0.70 |
| mls | kernel / RBF | 0.00 | 0.24 | 0.40 | 0.72 |
| kan[opt=lbfgs] | networks | 0.01 | 0.23 | 0.40 | 0.71 |
| ok_variogram | GP family | 0.01 | 0.27 | 0.40 | 0.69 |
| mars | basis / regression | 0.01 | 0.18 | 0.35 | 0.68 |
| rsm2 | basis / regression | 0.03 | 0.22 | 0.34 | 0.61 |
| cart | tree ensembles | 0.00 | 0.19 | 0.31 | 0.58 |
| lipschitz | other | 0.01 | 0.22 | 0.31 | 0.55 |
| delaunay_linear | other | 0.01 | 0.21 | 0.30 | 0.54 |
| cs_fourier | basis / regression | 0.03 | 0.12 | 0.23 | 0.52 |
| gmdh | basis / regression | 0.02 | 0.14 | 0.20 | 0.38 |

## Best method per problem category — selected on seeds 0-4, scored on held-out seeds 5-9

`×best` = geometric-mean NRMSE of the pick over that of the held-out best (1.00 = the pick is the best).

| category | cells | overall pick (seeds 0-4) | overall: ×best on 5-9 | GP pick (seeds 0-4) | GP: ×best on 5-9 | tree pick (seeds 0-4) | tree: ×best on 5-9 | network pick (seeds 0-4) | network: ×best on 5-9 | best on seeds 5-9 | overall pick still best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| additive | 96 | pce_lars | 1.00 | gp_se | 1.46 | catboost | 1.81 | kan[opt=lbfgs] | 1.28 | pce_lars | True |
| blind | 88 | hetgp | 1.05 | hetgp | 1.05 | catboost | 1.39 | mlp_ens[opt=adam] | 1.27 | as_gp | False |
| bowl_shaped | 36 | gmdh | 1.00 | gp_se | 324.64 | catboost | 4949.81 | mlp_ens[opt=lbfgs] | 775.64 | gmdh | True |
| discontinuity | 121 | rf | 1.00 | treed_gp | 1.16 | rf | 1.00 | mlp_ens[opt=adam] | 2.36 | rf | True |
| engineering | 133 | gp_se | 1.00 | gp_se | 1.00 | catboost | 5.68 | mlp_ens[opt=lbfgs] | 1.55 | hetgp | False |
| kink | 88 | gp | 1.00 | gp | 1.00 | catboost | 2.08 | mlp_ens[opt=lbfgs] | 1.77 | gp | True |
| localized | 33 | rbf_mq | 1.00 | hetgp | 1.42 | extratrees | 1.61 | dkl | 1.35 | rbf_mq | True |
| lowrank | 33 | gp_se | 1.00 | gp_se | 1.00 | catboost | 2.70 | kan[opt=lbfgs] | 1.53 | gp_se | True |
| many_local_minima | 114 | rbf_tps | 1.00 | hetgp | 1.16 | extratrees | 1.02 | dkl | 1.09 | rbf_tps | True |
| monotone | 33 | pce_lars | 1.00 | gp | 1.18 | catboost | 5.54 | mlp_ens[opt=lbfgs] | 1.73 | pce_lars | True |
| multiscale | 73 | as_gp | 1.00 | as_gp | 1.00 | catboost | 5.94 | mlp_ens[opt=adam] | 1.55 | as_gp | True |
| nonstationary | 73 | hetgp | 1.00 | hetgp | 1.00 | catboost | 2.14 | dkl | 2.23 | hetgp | True |
| oscillatory | 96 | cs_fourier | 1.00 | as_gp | 13.03 | extratrees | 14.90 | dkl | 16.22 | cs_fourier | True |
| other | 154 | pce_lars | 1.00 | hetgp | 1.23 | catboost | 1.90 | kan[opt=lbfgs] | 1.87 | pce_lars | True |
| plate_shaped | 106 | pce_lars | 1.00 | as_gp | 2.20 | catboost | 13.20 | mlp_ens[opt=lbfgs] | 1.62 | pce_lars | True |
| pole | 81 | hetgp | 1.00 | hetgp | 1.00 | extratrees | 2.25 | mlp_ens[opt=lbfgs] | 2.13 | hetgp | True |
| polynomial | 33 | pce_lars | 1.00 | gp_se | 388.43 | catboost | 7216.71 | mlp_ens[opt=lbfgs] | 1031.70 | pce_lars | True |
| prior_draw | 210 | hetgp | 1.00 | hetgp | 1.00 | extratrees | 1.16 | mlp_ens[opt=adam] | 1.36 | hetgp | True |
| regime | 81 | dkl | 1.02 | as_gp | 1.00 | extratrees | 1.07 | dkl | 1.02 | as_gp | False |
| ridge | 115 | as_gp | 1.00 | as_gp | 1.00 | catboost | 9.25 | mlp_ens[opt=lbfgs] | 2.42 | as_gp | True |
| smooth | 115 | gp_se | 1.01 | gp_se | 1.01 | catboost | 9.79 | mlp_ens[opt=lbfgs] | 3.53 | as_gp | False |
| steep_ridges_and_drops | 84 | catboost | 1.00 | hetgp | 1.48 | catboost | 1.00 | kan[opt=lbfgs] | 1.45 | catboost | True |
| thesis | 156 | gp | 1.00 | gp | 1.00 | extratrees | 2.74 | mlp_ens[opt=lbfgs] | 1.86 | gp | True |
| valley_shaped | 106 | pce_lars | 1.00 | hetgp | 13.27 | xgb | 28.73 | mlp_ens[opt=lbfgs] | 19.95 | pce_lars | True |

Overall pick confirmed as held-out best in 83% of categories.
