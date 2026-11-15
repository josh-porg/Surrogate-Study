# E3s_gp_restarts — 5 ML-II restarts vs E3's 1 (44996 paired cells, 82 problems)

Ratio < 1: the restarts lower the error. Wilcoxon over problems (cells averaged in log space).

## 1. Paired gain from restarts

|  | cells | ratio | improved | rescued_2x | worse_10pct | fit_time_x | p_wilcoxon |
|---|---|---|---|---|---|---|---|
| as_gp | 1.13e+04 | 0.949 | 0.338 | 0.0314 | 0.019 | 2.48 | 1.07e-07 |
| gp | 1.13e+04 | 0.948 | 0.257 | 0.0337 | 0.0124 | 3.05 | 5.22e-07 |
| gp_se | 1.13e+04 | 0.892 | 0.26 | 0.0624 | 0.0174 | 3.01 | 1.01e-10 |
| hetgp | 1.11e+04 | 0.979 | 0.368 | 0.0165 | 0.026 | 2.74 | 9.46e-05 |

## 2. Distance to the per-cell best (× best)

|  | x_best_before | x_best_after | best_in_cell_before | best_in_cell_after |
|---|---|---|---|---|
| as_gp | 3.73 | 3.54 | 0.0824 | 0.0889 |
| gp | 3.26 | 3.09 | 0.0219 | 0.0249 |
| gp_se | 3.33 | 2.97 | 0.067 | 0.0691 |
| hetgp | 3.19 | 3.12 | 0.021 | 0.0266 |

## 3. Head-to-head with the best tree-ensemble config in the cell

|  | beats_best_tree_before | beats_best_tree_after | ratio_to_best_tree_before | ratio_to_best_tree_after |
|---|---|---|---|---|
| as_gp | 0.617 | 0.643 | 0.57 | 0.541 |
| gp | 0.687 | 0.715 | 0.499 | 0.473 |
| gp_se | 0.627 | 0.673 | 0.509 | 0.454 |
| hetgp | 0.679 | 0.691 | 0.487 | 0.477 |

## 4. Restart / E3 ratio by problem category

| cat | as_gp | gp | gp_se | hetgp |
|---|---|---|---|---|
| additive | 0.96 | 0.967 | 0.9 | 0.966 |
| blind | 0.907 | 0.957 | 0.945 | 0.979 |
| bowl_shaped | 0.977 | 0.991 | 0.804 | 1.08 |
| discontinuity | 0.966 | 0.946 | 0.898 | 0.989 |
| engineering | 0.993 | 0.992 | 0.961 | 0.999 |
| kink | 0.99 | 0.997 | 0.965 | 0.993 |
| localized | 0.815 | 0.844 | 0.819 | 0.992 |
| lowrank | 1.01 | 1.01 | 0.97 | 1.01 |
| many_local_minima | 0.92 | 0.889 | 0.839 | 0.913 |
| monotone | 1 | 0.996 | 0.942 | 0.992 |
| multiscale | 0.936 | 0.936 | 0.907 | 0.956 |
| nonstationary | 0.947 | 0.904 | 0.85 | 0.961 |
| oscillatory | 0.88 | 0.953 | 0.951 | 0.981 |
| other | 0.961 | 0.913 | 0.901 | 0.978 |
| plate_shaped | 0.997 | 0.977 | 0.935 | 0.994 |
| pole | 0.903 | 0.93 | 0.666 | 0.963 |
| polynomial | 0.977 | 0.972 | 0.907 | 1 |
| prior_draw | 0.911 | 0.898 | 0.858 | 0.964 |
| regime | 0.878 | 0.858 | 0.807 | 0.97 |
| ridge | 0.998 | 0.995 | 0.989 | 0.998 |
| smooth | 0.973 | 0.981 | 0.974 | 0.986 |
| steep_ridges_and_drops | 0.919 | 0.929 | 0.869 | 0.95 |
| thesis | 0.984 | 0.989 | 0.848 | 0.992 |
| valley_shaped | 0.978 | 0.991 | 0.918 | 0.995 |
