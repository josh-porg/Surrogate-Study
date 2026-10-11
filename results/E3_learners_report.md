# E3 — learner study (786953 of 826 020 task records, 778780 ok fits; 2041 complete cells of 2484); excluded: smt_kpls, smt_krg

Clean-truth NRMSE; cells = problem × n × noise × level, mean over ≥ 5 seeds. Ranks are within cell among the configs present (O(n³)/network learners stop at n = 1000, `smt_krg` at 500, so large-n cells have fewer entrants). Interim numbers move as the remaining (mostly small-n) tasks arrive.

## Overall (mean rank within cell)

| cfg | family | mean rank | cells | median NRMSE | win share |
|---|---|---|---|---|---|
| hetgp | GP family | 7.85 | 1894 | 0.161 | 0.0723 |
| ens_press | other | 9.28 | 1894 | 0.208 | 0.0095 |
| gp | GP family | 9.32 | 1894 | 0.188 | 0.0512 |
| local_gp | GP family | 10.8 | 1894 | 0.211 | 0.0232 |
| as_gp | GP family | 11.3 | 1894 | 0.203 | 0.142 |
| gp_se | GP family | 12.7 | 1894 | 0.245 | 0.0935 |
| catboost | tree ensembles | 13.3 | 2041 | 0.248 | 0.0603 |
| rbf_tps | kernel / RBF | 13.3 | 1894 | 0.287 | 0.0164 |
| krr | kernel / RBF | 13.7 | 1894 | 0.285 | 0.0296 |
| treed_gp | GP family | 13.8 | 2041 | 0.238 | 0.0265 |
| extratrees | tree ensembles | 13.9 | 2041 | 0.285 | 0.0353 |
| pce_lars | basis / regression | 14.3 | 2041 | 0.335 | 0.11 |
| rbf_mq | kernel / RBF | 15.1 | 1894 | 0.278 | 0.0264 |
| ppr | basis / regression | 15.5 | 2041 | 0.263 | 0.0323 |
| dkl | networks | 15.7 | 1894 | 0.25 | 0.0285 |
| mlp_ens[opt=adam] | networks | 16.1 | 1894 | 0.284 | 0.00422 |
| ok_variogram | GP family | 16.5 | 1894 | 0.335 | 0.0132 |
| svr | kernel / RBF | 16.8 | 1894 | 0.339 | 0.0095 |
| mlp_ens[opt=lbfgs] | networks | 17 | 1894 | 0.276 | 0.057 |
| lgbm | tree ensembles | 17.2 | 2041 | 0.275 | 0.0206 |

## Best-of-family head-to-head (1894 cells where all three families have an entrant)

Share of cells each family's best config wins, by samples per dimension m = n/d:

| row_0 | GP family | tree ensembles | networks | cells |
|---|---|---|---|---|
| 5 | 0.51 | 0.43 | 0.06 | 340.00 |
| 10 | 0.69 | 0.26 | 0.05 | 340.00 |
| 20 | 0.78 | 0.19 | 0.03 | 340.00 |
| 50 | 0.69 | 0.16 | 0.15 | 340.00 |
| 100 | 0.57 | 0.15 | 0.28 | 340.00 |
| 300 | 0.56 | 0.10 | 0.34 | 194.00 |

By noise:

| row_0 | GP family | tree ensembles | networks |
|---|---|---|---|
| additive 0.01 | 0.66 | 0.18 | 0.15 |
| additive 0.1 | 0.60 | 0.19 | 0.21 |
| heavytail 0.01 | 0.66 | 0.26 | 0.09 |
| heavytail 0.1 | 0.57 | 0.31 | 0.11 |
| heteroscedastic 0.01 | 0.66 | 0.26 | 0.09 |
| heteroscedastic 0.1 | 0.61 | 0.26 | 0.13 |
| multiplicative 0.01 | 0.67 | 0.26 | 0.08 |
| multiplicative 0.1 | 0.56 | 0.31 | 0.12 |
| none 0 | 0.69 | 0.18 | 0.13 |
| numerical 0.01 | 0.67 | 0.26 | 0.08 |
| numerical 0.1 | 0.58 | 0.30 | 0.12 |

By problem category:

| row_0 | GP family | tree ensembles | networks |
|---|---|---|---|
| additive | 0.60 | 0.27 | 0.12 |
| blind | 0.63 | 0.08 | 0.29 |
| bowl_shaped | 0.83 | 0.00 | 0.17 |
| discontinuity | 0.11 | 0.74 | 0.15 |
| engineering | 0.89 | 0.00 | 0.11 |
| kink | 0.85 | 0.01 | 0.14 |
| localized | 0.58 | 0.24 | 0.18 |
| lowrank | 0.42 | 0.36 | 0.21 |
| many_local_minima | 0.43 | 0.26 | 0.30 |
| monotone | 0.82 | 0.00 | 0.18 |
| multiscale | 0.93 | 0.01 | 0.05 |
| nonstationary | 0.67 | 0.25 | 0.08 |
| oscillatory | 0.88 | 0.12 | 0.00 |
| other | 0.55 | 0.35 | 0.10 |
| plate_shaped | 0.65 | 0.00 | 0.35 |
| pole | 0.85 | 0.00 | 0.15 |
| polynomial | 0.88 | 0.00 | 0.12 |
| prior_draw | 0.52 | 0.47 | 0.01 |
| regime | 0.32 | 0.27 | 0.41 |
| ridge | 0.96 | 0.00 | 0.04 |
| smooth | 0.97 | 0.00 | 0.03 |
| steep_ridges_and_drops | 0.11 | 0.73 | 0.15 |
| thesis | 0.81 | 0.03 | 0.17 |
| valley_shaped | 0.76 | 0.22 | 0.02 |

## Thesis pair: GP vs XGBoost (ratio > 1 means GP more accurate)

|  | geo-mean NRMSE ratio xgb/gp | GP better (share) | cells |
|---|---|---|---|
| 5 | 1.67 | 0.80 | 340.00 |
| 10 | 2.04 | 0.81 | 340.00 |
| 20 | 2.51 | 0.83 | 340.00 |
| 50 | 2.64 | 0.70 | 340.00 |
| 100 | 2.63 | 0.66 | 340.00 |
| 300 | 3.08 | 0.61 | 194.00 |

By noise level (additive only):

| level | ratio | GP better |
|---|---|---|
| 0.0 | 5.70 | 0.83 |
| 0.01 | 3.27 | 0.82 |
| 0.1 | 1.46 | 0.74 |

## Gradient accuracy and calibration (medians over all fits; cover95 target 0.95)

| cfg | family | grad_nrmse | cover95 | nlpd | t_fit_s |
|---|---|---|---|---|---|
| hetgp | GP family | 0.502 | 0.915 | 0.303 | 1.11 |
| gp | GP family | 0.524 | 0.959 | -0.137 | 0.394 |
| as_gp | GP family | 0.57 | 0.958 | -0.211 | 0.851 |
| gp_se | GP family | 0.649 | 0.956 | 0.106 | 0.289 |
| treed_gp | GP family | 0.688 | 0.921 | 0.35 | 0.67 |
| rbf_mq | kernel / RBF | 0.711 | nan | nan | 0.0228 |
| rbf_tps | kernel / RBF | 0.713 | nan | nan | 0.00777 |
| krr | kernel / RBF | 0.764 | nan | nan | 0.468 |
| mlp_ens[opt=adam] | networks | 0.766 | 0.867 | 0.449 | 14.5 |
| mlp_ens[opt=lbfgs] | networks | 0.772 | 0.908 | 0.151 | 11.8 |
| ok_variogram | GP family | 0.795 | 0.994 | 0.2 | 0.0671 |
| svr | kernel / RBF | 0.8 | nan | nan | 0.649 |
| local_gp | GP family | 0.82 | 0.952 | -0.115 | 0.434 |
| pce_lars | basis / regression | 0.849 | nan | nan | 0.0593 |
| cs_cosine | basis / regression | 0.853 | nan | nan | 0.0545 |
| ppr | basis / regression | 0.866 | nan | nan | 2.77 |
| dkl | networks | 0.896 | 0.946 | 0.893 | 3.01 |
| rsm2 | basis / regression | 0.897 | nan | nan | 0.000903 |
| mars | basis / regression | 0.907 | nan | nan | 0.629 |
| mls | kernel / RBF | 1 | nan | nan | 0.000192 |
| gmdh | basis / regression | 1 | nan | nan | 0.0226 |
| kan[opt=lbfgs] | networks | 1.01 | nan | nan | 0.518 |
| cart | tree ensembles | 1.02 | nan | nan | 0.0943 |
| lipschitz | other | 1.09 | 1 | 1.12 | 0.000535 |
| delaunay_linear | other | 1.15 | nan | nan | 0.00313 |
| lgbm | tree ensembles | 1.17 | nan | nan | 1.98 |
| xgb | tree ensembles | 1.2 | nan | nan | 0.958 |
| ens_press | other | 1.4 | 0.916 | 0.38 | 2.62 |
| cs_fourier | basis / regression | 1.54 | nan | nan | 0.164 |
| gbt_committee | tree ensembles | 2.28 | 0.834 | 1.01 | 1.27 |
| extratrees | tree ensembles | 5.47 | 0.991 | 0.32 | 0.549 |
| catboost | tree ensembles | 5.69 | nan | nan | 0.481 |
| rf | tree ensembles | 5.78 | 0.986 | -0.103 | 0.86 |
