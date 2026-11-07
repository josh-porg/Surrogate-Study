# E13 — gradient-enhanced surrogates at matched cost (23658 fits, 24 problems)

Ratio < 1: the gradient-enhanced fit has lower test NRMSE. Wilcoxon over problems (cells averaged in log space). Value-only learners at cost 2 model an adjoint gradient (≈ one extra solve); at d + 1, finite differences.

## 1. The information in a gradient: the same method with vs without gradients on the same points

|  | cells | ratio | better | p |
|---|---|---|---|---|
| ge_pce | 1.08e+03 | 0.108 | 0.676 | 0.00252 |
| gek | 1.04e+03 | 0.295 | 0.834 | 4.42e-05 |
| hermite_rbf | 1.04e+03 | 0.488 | 0.726 | 0.00434 |
| sobolev_mlp | 1.08e+03 | 0.582 | 0.746 | 0.0526 |

## 2. Matched cost: gradient learners vs the best value-only learner per cell (oracle choice among six value-only learners, so this favours value-only)

|  | cells | ratio | better | p |
|---|---|---|---|---|
| ('ge_pce', 'value-only at 2x') | 1.08e+03 | 1.02 | 0.308 | 0.768 |
| ('gek', 'value-only at 2x') | 1.04e+03 | 4.04 | 0.459 | 0.509 |
| ('hermite_rbf', 'value-only at 2x') | 1.04e+03 | 8.84 | 0.186 | 0.00158 |
| ('sobolev_mlp', 'value-only at 2x') | 1.08e+03 | 9.29 | 0.378 | 0.00434 |
| ('best gradient learner', 'value-only at 2x') | 1.08e+03 | 0.39 | 0.561 | 0.0249 |
| ('ge_pce', 'value-only at d+1x') | 1.08e+03 | 4.22 | 0.0926 | 2.38e-07 |
| ('gek', 'value-only at d+1x') | 1.04e+03 | 17.3 | 0.157 | 2.38e-07 |
| ('hermite_rbf', 'value-only at d+1x') | 1.04e+03 | 37.8 | 0.0126 | 1.19e-07 |
| ('sobolev_mlp', 'value-only at d+1x') | 1.08e+03 | 38.5 | 0.132 | 5.96e-07 |
| ('best gradient learner', 'value-only at d+1x') | 1.08e+03 | 1.62 | 0.241 | 0.000742 |

## 2b. Matched cost, same method: with gradients on n points vs without on 2n or (d+1)n points (the median is shown because exactly recovered polynomial problems dominate geometric means)

|  | cells | ratio_geo | ratio_median | better | p |
|---|---|---|---|---|---|
| ('ge_pce', 'same method, value-only at 2x') | 1.08e+03 | 0.369 | 0.894 | 0.518 | 0.0526 |
| ('gek', 'same method, value-only at 2x') | 1.04e+03 | 0.581 | 0.605 | 0.717 | 0.00252 |
| ('hermite_rbf', 'same method, value-only at 2x') | 1.04e+03 | 0.867 | 0.781 | 0.579 | 0.034 |
| ('sobolev_mlp', 'same method, value-only at 2x') | 1.08e+03 | 0.956 | 0.531 | 0.698 | 0.241 |
| ('ge_pce', 'same method, value-only at d+1x') | 1.08e+03 | 2.26 | 1.56 | 0.146 | 5.33e-05 |
| ('gek', 'same method, value-only at d+1x') | 1.04e+03 | 1.43 | 1.12 | 0.369 | 0.0249 |
| ('hermite_rbf', 'same method, value-only at d+1x') | 1.04e+03 | 1.96 | 1.42 | 0.188 | 0.000175 |
| ('sobolev_mlp', 'same method, value-only at d+1x') | 1.08e+03 | 1.48 | 0.871 | 0.559 | 0.855 |

## 3. Best gradient learner vs best value-only learner at 2× cost, by noise level and n_base/d

|  | ratio | better | cells |
|---|---|---|---|
| (0.0, 5) | 0.382 | 0.542 | 120 |
| (0.0, 10) | 0.0715 | 0.458 | 120 |
| (0.0, 20) | 0.045 | 0.458 | 120 |
| (0.01, 5) | 0.698 | 0.617 | 120 |
| (0.01, 10) | 0.672 | 0.558 | 120 |
| (0.01, 20) | 0.733 | 0.592 | 120 |
| (0.1, 5) | 0.789 | 0.625 | 120 |
| (0.1, 10) | 0.821 | 0.575 | 120 |
| (0.1, 20) | 0.76 | 0.625 | 120 |

Per problem (ratio, all noise levels and sizes):

| problem | ratio |
|---|---|
| rosen_d5 | 0.00477 |
| dixonpr_d5 | 0.00506 |
| zakharov_d5 | 0.00714 |
| goldpr | 0.0901 |
| stybtang | 0.115 |
| smooth_trig_d10 | 0.17 |
| smooth_trig_d5 | 0.216 |
| rotated_ridge_d5 | 0.271 |
| wingweight | 0.32 |
| piston | 0.42 |
| borehole | 0.425 |
| quadratic_d5 | 0.581 |
| branin | 0.624 |
| kink_floor_d5 | 0.942 |
| trid_d5 | 0.956 |
| schaffer2 | 0.999 |
| michal_d5 | 1.03 |
| sumsqu_d5 | 1.22 |
| rastr_d5 | 1.34 |
| rothyp_d10 | 1.44 |
| levy_d5 | 1.48 |
| ackley_d5 | 1.57 |
| camel6 | 2.53 |
| griewank_d5 | 32.2 |

## 4. Median gradient NRMSE of the surrogate (lower is better)

| learner | value-only (same points) | with gradients |
|---|---|---|
| ge_pce | 0.648 | 0.215 |
| gek | 0.564 | 0.131 |
| gp | 0.458 | nan |
| hermite_rbf | 0.588 | 0.269 |
| pce_lars | 0.611 | nan |
| sobolev_mlp | 0.504 | 0.0954 |
