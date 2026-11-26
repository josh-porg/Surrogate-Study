# E4 — mechanisms and the R-AIR bias arm

## Mechanisms: no records yet

## R-AIR bias arm (6399 fits; 20 paired splits)

### Empty_Weight_Fraction

|  | winner | best | catboost | gp |
|---|---|---|---|---|
| ('nrmse', 'balanced') | ok_variogram | 0.909 | 0.951 | 0.969 |
| ('nrmse', 'biased') | ens_press | 0.745 | 0.804 | 0.812 |
| ('nrmse', 'weighted') | extratrees | 0.851 | 0.893 | 0.941 |
| ('nrmse_macro', 'balanced') | svr | 0.826 | 0.874 | 0.89 |
| ('nrmse_macro', 'biased') | ens_press | 0.729 | 0.793 | 0.791 |
| ('nrmse_macro', 'weighted') | extratrees | 0.816 | 0.864 | 0.911 |

Spearman correlation of learner rankings (macro NRMSE): biased vs weighted 0.80, biased vs balanced 0.79, weighted vs balanced 0.86

Top 8 by macro NRMSE per setting (rank):

| cfg | balanced | biased | weighted |
|---|---|---|---|
| ens_press | 26 | 1 | 19 |
| smt_kpls | 3 | 2 | 21 |
| extratrees | 4 | 3 | 1 |
| krr | 5 | 4 | 8 |
| rbf_mq | 13 | 5 | nan |
| ok_variogram | 2 | 6 | 4 |
| rbf_tps | 14 | 7 | 9 |
| svr | 1 | 8 | 3 |

### Fuel_Weight_Fraction

|  | winner | best | catboost | gp |
|---|---|---|---|---|
| ('nrmse', 'balanced') | extratrees | 0.377 | 0.411 | 0.427 |
| ('nrmse', 'biased') | extratrees | 0.307 | 0.339 | 0.327 |
| ('nrmse', 'weighted') | extratrees | 0.379 | 0.409 | 0.476 |
| ('nrmse_macro', 'balanced') | extratrees | 0.385 | 0.424 | 0.423 |
| ('nrmse_macro', 'biased') | extratrees | 0.328 | 0.369 | 0.356 |
| ('nrmse_macro', 'weighted') | extratrees | 0.401 | 0.438 | 0.504 |

Spearman correlation of learner rankings (macro NRMSE): biased vs weighted 0.73, biased vs balanced 0.84, weighted vs balanced 0.75

Top 8 by macro NRMSE per setting (rank):

| cfg | balanced | biased | weighted |
|---|---|---|---|
| extratrees | 1 | 1 | 1 |
| ens_press | 14 | 2 | 13 |
| local_gp | 2 | 3 | 12 |
| xgb | 13 | 4 | 7 |
| gp | 4 | 5 | 17 |
| smt_krg | 19 | 6 | 20 |
| hetgp | 6 | 7 | 15 |
| gp_se | 16 | 8 | 25 |

### Payload_Weight_Fraction

|  | winner | best | catboost | gp |
|---|---|---|---|---|
| ('nrmse', 'balanced') | extratrees | 0.537 | 0.549 | 0.582 |
| ('nrmse', 'biased') | extratrees | 0.434 | 0.468 | 0.495 |
| ('nrmse', 'weighted') | extratrees | 0.516 | 0.537 | 0.662 |
| ('nrmse_macro', 'balanced') | extratrees | 0.452 | 0.482 | 0.498 |
| ('nrmse_macro', 'biased') | extratrees | 0.395 | 0.44 | 0.473 |
| ('nrmse_macro', 'weighted') | extratrees | 0.464 | 0.488 | 0.662 |

Spearman correlation of learner rankings (macro NRMSE): biased vs weighted 0.74, biased vs balanced 0.78, weighted vs balanced 0.70

Top 8 by macro NRMSE per setting (rank):

| cfg | balanced | biased | weighted |
|---|---|---|---|
| extratrees | 1 | 1 | 1 |
| ens_press | 27 | 2 | 19 |
| rf | 4 | 3 | 7 |
| xgb | 16 | 4 | 4 |
| lgbm | 13 | 5 | 6 |
| gbt_committee | 3 | 6 | 3 |
| smt_kpls | 6 | 7 | 26 |
| qrf | 2 | 8 | 8 |
