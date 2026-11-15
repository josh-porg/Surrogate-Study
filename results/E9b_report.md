# E9b — surrogate-managed optimization loops (4679 runs, 38 problems; 50 errors and 131 unconfirmed-f* runs dropped)

Regret normalised by std(f); success = regret < 0.01. Budgets in true evaluations ÷ d.

## 1. Success and geometric-mean regret

| method | success@10d | success@20d | success@50d | geo_regret@10d | geo_regret@20d | geo_regret@50d |
|---|---|---|---|---|---|---|
| tr_model_mgmt | 0.533 | 0.646 | 0.71 | 0.00147 | 0.000422 | 0.000146 |
| dycors | 0.582 | 0.623 | 0.672 | 0.00133 | 0.000447 | 0.000163 |
| turbo | 0.392 | 0.555 | 0.635 | 0.0185 | 0.00423 | 0.00137 |
| ego | 0.436 | 0.554 | 0.62 | 0.0057 | 0.00156 | 0.000524 |
| sobol | 0.168 | 0.218 | 0.285 | 0.177 | 0.131 | 0.0886 |
| random | 0.165 | 0.217 | 0.279 | 0.183 | 0.132 | 0.089 |

## 2. Paired comparisons (regret ratio < 1: the first method is better; Wilcoxon over problems)

|  | instances | regret_ratio | a_better | p |
|---|---|---|---|---|
| ('random', 'vs sobol @10d') | 734 | 1.04 | 0.435 | 0.126 |
| ('random', 'vs sobol @50d') | 734 | 1 | 0.525 | 0.667 |
| ('ego', 'vs sobol @10d') | 734 | 0.0322 | 0.752 | 5.27e-07 |
| ('ego', 'vs sobol @50d') | 734 | 0.00591 | 0.881 | 1.46e-07 |
| ('tr_model_mgmt', 'vs sobol @10d') | 734 | 0.0083 | 0.775 | 1.46e-07 |
| ('tr_model_mgmt', 'vs sobol @50d') | 734 | 0.00165 | 0.832 | 1.72e-07 |
| ('turbo', 'vs sobol @10d') | 717 | 0.112 | 0.821 | 1.14e-07 |
| ('turbo', 'vs sobol @50d') | 717 | 0.0167 | 0.888 | 1.14e-07 |
| ('dycors', 'vs sobol @10d') | 734 | 0.00752 | 0.86 | 1.14e-07 |
| ('dycors', 'vs sobol @50d') | 734 | 0.00184 | 0.835 | 1.14e-07 |
| ('ego', 'vs turbo @50d') | 717 | 0.326 | 0.476 | 0.42 |
| ('ego', 'vs dycors @50d') | 734 | 3.21 | 0.354 | 0.0691 |
| ('ego', 'vs tr_model_mgmt @50d') | 734 | 3.58 | 0.364 | 0.00394 |
| ('turbo', 'vs dycors @50d') | 717 | 9.17 | 0.335 | 0.000321 |
| ('turbo', 'vs tr_model_mgmt @50d') | 717 | 11 | 0.328 | 0.00563 |
| ('dycors', 'vs tr_model_mgmt @50d') | 734 | 1.12 | 0.36 | 0.316 |

## 3. Success at 50d by dimension

| d | dycors | ego | random | sobol | tr_model_mgmt | turbo |
|---|---|---|---|---|---|---|
| 2 | 0.62 | 0.76 | 0.48 | 0.46 | 0.73 | 0.72 |
| 3 | 1.00 | 1.00 | 0.20 | 0.20 | 1.00 | 0.95 |
| 5 | 0.67 | 0.59 | 0.26 | 0.27 | 0.68 | 0.61 |
| 7 | 0.60 | 0.65 | 0.00 | 0.00 | 1.00 | 0.55 |
| 8 | 0.90 | 0.90 | 0.00 | 0.00 | 1.00 | 0.60 |
| 10 | 0.68 | 0.45 | 0.16 | 0.19 | 0.67 | 0.56 |

## 3. Success at 50d by noise level

| noise | dycors | ego | random | sobol | tr_model_mgmt | turbo |
|---|---|---|---|---|---|---|
| 0.0 | 0.80 | 0.62 | 0.30 | 0.30 | 0.73 | 0.72 |
| 0.01 | 0.54 | 0.62 | 0.26 | 0.27 | 0.69 | 0.55 |

## 3. Success at 50d by problem category

| cat | dycors | ego | random | sobol | tr_model_mgmt | turbo |
|---|---|---|---|---|---|---|
| blind | 0.70 | 0.40 | 0.00 | 0.00 | 0.60 | 0.55 |
| bowl_shaped | 0.77 | 0.82 | 0.00 | 0.00 | 1.00 | 0.95 |
| discontinuity | 1.00 | 0.97 | 0.80 | 0.85 | 1.00 | 0.97 |
| engineering | 0.76 | 0.44 | 0.00 | 0.00 | 1.00 | 0.52 |
| kink | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| localized | 0.55 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 |
| many_local_minima | 0.28 | 0.05 | 0.00 | 0.00 | 0.11 | 0.13 |
| monotone | 0.95 | 1.00 | 0.00 | 0.00 | 1.00 | 0.10 |
| other | 0.80 | 0.98 | 0.65 | 0.65 | 0.98 | 0.95 |
| plate_shaped | 1.00 | 1.00 | 0.93 | 0.90 | 1.00 | 1.00 |
| pole | 0.50 | 0.56 | 0.11 | 0.11 | 0.67 | 0.33 |
| polynomial | 0.80 | 0.80 | 0.00 | 0.00 | 1.00 | 0.85 |
| prior_draw | 0.26 | 0.55 | 0.05 | 0.00 | 0.34 | 0.54 |
| regime | 0.50 | 0.55 | 0.20 | 0.20 | 0.65 | 0.65 |
| ridge | 1.00 | 1.00 | 0.80 | 0.90 | 1.00 | 1.00 |
| smooth | 0.90 | 0.85 | 0.05 | 0.15 | 1.00 | 0.88 |
| steep_ridges_and_drops | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 |
| thesis | 1.00 | 1.00 | 0.20 | 0.20 | 1.00 | 0.95 |
| valley_shaped | 0.88 | 0.97 | 0.78 | 0.77 | 1.00 | 0.95 |

## 4. Median loop wall time [s] (the method's own computation; true evaluations are cheap here)

| method | 2 | 3 | 5 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|
| dycors | 0.389 | 1.25 | 5.34 | 17.4 | 26.7 | 59.4 |
| ego | 16.7 | 26.7 | 87.3 | 122 | 186 | 618 |
| random | 0.000211 | 0.000231 | 0.00026 | 0.000326 | 0.000316 | 0.000447 |
| sobol | 0.00106 | 0.00124 | 0.00151 | 0.00177 | 0.002 | 0.00237 |
| tr_model_mgmt | 11.4 | 21.6 | 64.8 | 192 | 283 | 519 |
| turbo | 104 | 162 | 288 | 466 | 580 | 828 |

## 5. Bootstrapped-kriging EGO vs plain EGO (its subset: d ≤ 5, instances 0-4)

|  | instances | regret_ratio | a_better | p |
|---|---|---|---|---|
| ego_bootstrap vs ego @10d | 292 | 8.7 | 0.339 | 0.00275 |
| ego_bootstrap vs ego @20d | 292 | 18 | 0.253 | 0.000247 |
| ego_bootstrap vs ego @50d | 292 | 31.4 | 0.192 | 1.95e-05 |

## 6. Best method per problem at 50d (geometric-mean regret)

|  | problems won |
|---|---|
| tr_model_mgmt | 17 |
| dycors | 12 |
| ego | 5 |
| turbo | 4 |

| problem | dycors | ego | random | sobol | tr_model_mgmt | turbo | best |
|---|---|---|---|---|---|---|---|
| ackley_d10 | 0.0713 | 30.7 | 43.6 | 43.7 | 44.8 | 5.9 | dycors |
| ackley_d5 | 0.00468 | 8.16 | 17.3 | 19.5 | 14.8 | 1.76 | dycors |
| axis_step_d5 | 1.22e-07 | 1e-08 | 0.00221 | 0.00391 | 1e-08 | 1.09e-05 | ego |
| booth | 1.09e-05 | 8.13e-06 | 0.00406 | 0.00639 | 2.34e-06 | 1.31e-05 | tr_model_mgmt |
| borehole | 0.000114 | 8.39e-06 | 0.0902 | 0.0876 | 9.37e-07 | 0.0012 | tr_model_mgmt |
| branin | 5.23e-05 | 8.57e-06 | 0.00645 | 0.00487 | 3.18e-06 | 5e-05 | tr_model_mgmt |
| camel6 | 9.1e-06 | 5.19e-05 | 0.00158 | 0.00312 | 7.65e-06 | 0.000104 | tr_model_mgmt |
| composite_d5 | 4.33e-05 | 0.00175 | 1.03 | 1.06 | 0.00652 | 0.00578 | dycors |
| dixonpr_d5 | 6.18e-05 | 0.000474 | 0.00388 | 0.00477 | 0.000251 | 9.62e-05 | dycors |
| goldpr | 7.85e-06 | 0.000133 | 0.000413 | 0.000271 | 5.72e-05 | 1.58e-05 | dycors |
| gp_draw_d2 | 0.00531 | 6.26e-05 | 0.125 | 0.173 | 0.0004 | 0.00125 | ego |
| griewank_d5 | 0.00556 | 0.0265 | 0.158 | 0.268 | 0.00838 | 0.00887 | dycors |
| kink_floor_d5 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | 1e-08 | dycors |
| levy_d10 | 0.00241 | 0.0405 | 0.29 | 0.402 | 0.127 | 0.07 | dycors |
| levy_d5 | 0.000403 | 0.0288 | 0.105 | 0.11 | 0.0201 | 0.009 | dycors |
| michal_d5 | 1.6 | 1.48 | 3.61 | 4.02 | 0.575 | 0.586 | tr_model_mgmt |
| monotone_d5 | 1.19e-07 | 1e-08 | 2.42 | 2.04 | 6.26e-07 | 0.0165 | ego |
| oblique_step_d10 | 1.76e-08 | 2.01e-08 | 0.00409 | 0.0029 | 1e-08 | 1.01e-05 | tr_model_mgmt |
| piston | 6.99e-05 | 0.000908 | 0.199 | 0.219 | 1.18e-06 | 0.00662 | tr_model_mgmt |
| plateau_bump_d5 | 0.517 | 406 | 632 | 631 | 22.4 | 9.27 | dycors |
| pole_knee_d5 | 16.3 | 0.808 | 4.38e+03 | 4.33e+03 | 2.3 | 1.2e+03 | ego |
| quadratic_d5 | 1.04e-05 | 0.000536 | 0.329 | 0.29 | 1.45e-06 | 0.00236 | tr_model_mgmt |
| rastr | 0.0363 | 0.0354 | 0.274 | 0.25 | 0.0101 | 0.0128 | tr_model_mgmt |
| rastr_d5 | 0.405 | 0.536 | 1 | 1.1 | 0.476 | 0.288 | turbo |
| regime_d5 | 7.3e-05 | 2.4e-05 | 0.0842 | 0.0709 | 1.17e-05 | 0.000484 | tr_model_mgmt |
| rosen_d5 | 0.00016 | 0.000552 | 0.00507 | 0.00487 | 0.000169 | 0.000198 | dycors |
| rotated_ridge_d5 | 3.16e-06 | 2.64e-06 | 0.000685 | 0.00023 | 1.37e-06 | 1.27e-06 | turbo |
| rothyp_d10 | 5.32e-05 | 0.00368 | 0.463 | 0.496 | 2.93e-05 | 0.000565 | tr_model_mgmt |
| schaffer2 | 1.58 | 5.61 | 3.79 | 3.45 | 4.66 | 1.37 | turbo |
| smooth_trig_d10 | 5.79e-06 | 3.03e-05 | 0.0655 | 0.0275 | 2.26e-06 | 3.04e-05 | tr_model_mgmt |
| smooth_trig_d5 | 4.18e-06 | 2.62e-05 | 0.782 | 0.79 | 2.66e-06 | 0.00275 | tr_model_mgmt |
| stybtang | 0.00139 | 2.43e-05 | 0.0232 | 0.0142 | 5.5e-06 | 0.000109 | tr_model_mgmt |
| sumsqu_d5 | 3.78e-05 | 0.000206 | 0.116 | 0.179 | 2.73e-06 | 0.000209 | tr_model_mgmt |
| thesis_collapse_knee | 5.61e-07 | 1e-08 | 0.04 | 0.0361 | 1e-08 | 0.000298 | ego |
| tree_draw_d5 | 0.217 | 0.029 | 0.768 | 0.605 | 0.449 | 0.000231 | turbo |
| trid_d5 | 2.59e-05 | 0.000845 | 0.118 | 0.137 | 4.67e-06 | 0.000159 | tr_model_mgmt |
| wingweight | 1.41e-05 | 0.0433 | 0.732 | 0.814 | 1.65e-06 | 0.0109 | tr_model_mgmt |
| zakharov_d5 | 2.13e-05 | 4.37e-05 | 5.54e-05 | 7.22e-05 | 4.54e-05 | 2.65e-05 | dycors |
