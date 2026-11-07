# E3s_tuning — a larger tuning budget for trees and networks (36090 paired cells, 82 problems)

Pairs: each tuned fit against its E3 twin on the same cell. Ratio < 1: tuning lowers the error. Wilcoxon over problems (cells averaged in log space).

## 1. Paired gain from tuning

|  | cells | ratio | improved | worse_by_10pct | fit_time_x | p_wilcoxon |
|---|---|---|---|---|---|---|
| catboost → catboost[tune=rs30] | 1.24e+04 | 0.892 | 0.752 | 0.0613 | 86.3 | 8.84e-15 |
| mlp_ens[opt=lbfgs] → mlp_ens[K=10,opt=lbfgs] | 1.13e+04 | 0.951 | 0.707 | 0.14 | 1.82 | 6.98e-12 |
| xgb → xgb[tune=rs30] | 1.24e+04 | 0.921 | 0.755 | 0.0806 | 15.6 | 3.6e-14 |

## 2. Distance to the per-cell best (all E3 methods plus the tuned fit; × best)

|  | x_best_before | x_best_after | best_in_cell_before | best_in_cell_after |
|---|---|---|---|---|
| catboost | 7.18 | 6.41 | 0.0149 | 0.0547 |
| mlp_ens[opt=lbfgs] | 5.25 | 4.99 | 0.0256 | 0.0553 |
| xgb | 8.5 | 7.82 | 0.0111 | 0.0256 |

## 3. Head-to-head with the best GP-family config in the same cell

|  | beats_best_GP_before | beats_best_GP_after | ratio_to_best_GP_before | ratio_to_best_GP_after |
|---|---|---|---|---|
| catboost | 0.128 | 0.147 | 3.73 | 3.33 |
| mlp_ens[opt=lbfgs] | 0.1 | 0.118 | 2.83 | 2.69 |
| xgb | 0.102 | 0.113 | 4.44 | 4.09 |

## 4. Tuned / E3 ratio by problem category

| cat | catboost | mlp_ens[opt=lbfgs] | xgb |
|---|---|---|---|
| additive | 0.928 | 0.945 | 0.85 |
| blind | 0.974 | 0.947 | 0.926 |
| bowl_shaped | 0.803 | 0.902 | 0.954 |
| discontinuity | 0.968 | 0.95 | 0.941 |
| engineering | 0.929 | 0.944 | 0.881 |
| kink | 0.933 | 0.925 | 0.91 |
| localized | 0.911 | 0.967 | 0.963 |
| lowrank | 0.883 | 0.98 | 0.931 |
| many_local_minima | 0.932 | 0.975 | 0.992 |
| monotone | 0.864 | 0.906 | 0.869 |
| multiscale | 0.859 | 0.942 | 0.865 |
| nonstationary | 0.891 | 1 | 1.02 |
| oscillatory | 0.907 | 1 | 0.93 |
| other | 0.866 | 0.974 | 0.829 |
| plate_shaped | 0.964 | 0.983 | 0.926 |
| pole | 0.732 | 1 | 1.03 |
| polynomial | 0.992 | 0.958 | 0.891 |
| prior_draw | 0.928 | 0.94 | 0.97 |
| regime | 0.971 | 0.973 | 0.977 |
| ridge | 0.872 | 0.923 | 0.898 |
| smooth | 0.915 | 0.945 | 0.887 |
| steep_ridges_and_drops | 0.935 | 0.962 | 0.919 |
| thesis | 0.796 | 0.891 | 0.94 |
| valley_shaped | 0.706 | 0.929 | 0.887 |
