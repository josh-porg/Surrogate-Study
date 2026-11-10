# E14 — self-designing surrogates vs learners on a Sobol' design (5040 fits, 28 problems)

Ratio < 1: the self-designing method has lower test NRMSE. 'best baseline' = the best of gp, pce_lars, catboost, rbf_tps in the cell (an oracle choice, so it favours the baselines). Wilcoxon over problems.

## 1. Overall

| method | vs_best | beats_best | p_best | vs_gp | beats_gp | p_gp | n_used_share |
|---|---|---|---|---|---|---|---|
| smolyak | 1.28 | 0.14 | 0.316 | 0.355 | 0.229 | 0.552 | 0.495 |
| tt_cross | 4.49 | 0.135 | 0.0118 | 1.24 | 0.193 | 0.412 | 0.8 |

## 2. By problem structure: ratio to the best baseline (share of cells won)

| struct | smolyak | tt_cross |
|---|---|---|
| additive | 0.16 (20%) | 0.37 (26%) |
| axis_step | 2.92 (0%) | 3.41 (0%) |
| benchmark | 1.25 (19%) | 22.68 (6%) |
| kink | 3.25 (0%) | 3.68 (0%) |
| lowrank | 1.93 (6%) | 0.01 (61%) |
| rotated_ridge | 2.60 (11%) | 8.80 (6%) |
| smooth_trig | 1.56 (12%) | 10.02 (6%) |

### By budget and noise

|  | ratio | wins |
|---|---|---|
| ('smolyak', 50, 0.0) | 4.4 | 0.0714 |
| ('smolyak', 50, 0.01) | 3.6 | 0.00714 |
| ('smolyak', 200, 0.0) | 0.178 | 0.307 |
| ('smolyak', 200, 0.01) | 3.93 | 0.0929 |
| ('smolyak', 1000, 0.0) | 0.0563 | 0.329 |
| ('smolyak', 1000, 0.01) | 7.17 | 0.0357 |
| ('tt_cross', 50, 0.0) | 32.2 | 0.0571 |
| ('tt_cross', 50, 0.01) | 5.91 | 0.05 |
| ('tt_cross', 200, 0.0) | 16.1 | 0.0786 |
| ('tt_cross', 200, 0.01) | 6.16 | 0.0714 |
| ('tt_cross', 1000, 0.0) | 0.0859 | 0.407 |
| ('tt_cross', 1000, 0.01) | 5.07 | 0.143 |

## 3. Best method per problem (geometric mean over budgets, noise, seeds)

|  | problems won |
|---|---|
| gp | 8 |
| smolyak | 7 |
| tt_cross | 6 |
| pce_lars | 4 |
| catboost | 2 |
| rbf_tps | 1 |

| problem | catboost | gp | pce_lars | rbf_tps | smolyak | tt_cross | best |
|---|---|---|---|---|---|---|---|
| ackley_d10 | 0.79 | 0.861 | 0.719 | 0.731 | 87.3 | 1.16 | pce_lars |
| ackley_d5 | 0.69 | 0.609 | 0.736 | 0.584 | 18.1 | 1.05 | rbf_tps |
| additive_d10 | 0.365 | 0.79 | 0.746 | 0.766 | 1.16 | 0.226 | tt_cross |
| additive_d2 | 0.075 | 0.00844 | 0.00599 | 0.0233 | 2.78e-05 | 0.00343 | smolyak |
| additive_d5 | 0.181 | 0.163 | 0.473 | 0.451 | 0.0257 | 0.0125 | tt_cross |
| axis_step_d2 | 0.0974 | 0.102 | 0.281 | 0.219 | 0.171 | 0.276 | catboost |
| axis_step_d5 | 0.0973 | 0.119 | 0.41 | 0.429 | 0.395 | 0.335 | catboost |
| borehole | 0.107 | 0.0061 | 0.114 | 0.0592 | 0.0381 | 0.0371 | gp |
| branin | 0.0925 | 0.00191 | 0.0187 | 0.0232 | 3.81e-05 | 0.00285 | smolyak |
| camel6 | 0.133 | 0.0141 | 5.42e-07 | 0.082 | 1.29e-06 | 7.8e-05 | pce_lars |
| goldpr | 0.13 | 0.0159 | 0.121 | 0.0414 | 7.62e-05 | 0.0274 | smolyak |
| kink_floor_d2 | 0.0616 | 0.0142 | 0.0588 | 0.0287 | 0.0461 | 0.0522 | gp |
| lowrank_product_d10 | 0.619 | 0.748 | 0.807 | 0.753 | 0.844 | 0.00842 | tt_cross |
| lowrank_product_d2 | 0.0936 | 0.00498 | 0.117 | 0.0166 | 0.0101 | 0.0022 | tt_cross |
| lowrank_product_d5 | 0.319 | 0.208 | 0.514 | 0.351 | 0.499 | 9.31e-05 | tt_cross |
| otlcircuit | 0.0752 | 0.00531 | 0.0349 | 0.0359 | 0.0185 | 0.075 | gp |
| piston | 0.137 | 0.0268 | 0.215 | 0.116 | 0.13 | 0.608 | gp |
| quadratic_d5 | 0.148 | 0.00335 | 4.59e-05 | 0.0384 | 0.000106 | 0.645 | pce_lars |
| rastr | 0.411 | 0.583 | 0.701 | 0.447 | 0.00649 | 0.57 | smolyak |
| rosen_d5 | 0.185 | 0.0583 | 0.0028 | 0.183 | 0.000134 | 0.919 | smolyak |
| rotated_ridge_d10 | 0.345 | 0.153 | 0.308 | 0.217 | 0.485 | 0.875 | gp |
| rotated_ridge_d2 | 0.0199 | 0.00053 | 0.00525 | 0.00392 | 0.000276 | 0.00125 | smolyak |
| rotated_ridge_d5 | 0.0797 | 0.0102 | 0.129 | 0.0711 | 0.108 | 0.511 | gp |
| smooth_trig_d10 | 0.202 | 0.0463 | 0.115 | 0.081 | 0.133 | 0.61 | gp |
| smooth_trig_d2 | 0.0434 | 0.00108 | 0.000181 | 0.00646 | 4.77e-05 | 0.000266 | smolyak |
| smooth_trig_d5 | 0.122 | 0.0106 | 0.102 | 0.0366 | 0.0515 | 0.535 | gp |
| sumsqu_d10 | 0.316 | 0.0889 | 5.42e-07 | 0.244 | 1.71e-06 | 0.0156 | pce_lars |
| wingweight | 0.138 | 0.00869 | 0.119 | 0.0428 | 0.0649 | 0.00185 | tt_cross |
