# E6 — multi-fidelity methods vs high-fidelity only (22371 fits, 27 pairs; 309 timeouts/errors dropped)

Ratio < 1: the MF method has lower test NRMSE. 'same HF' = HF-only on the same n_H points (the low-fidelity data is free); 'matched' = HF-only given the low-fidelity budget as extra HF points (n_H + n_L / κ).

## 1. Each MF method vs HF-only

| method | ('same HF', 'cells') | ('same HF', 'ratio') | ('same HF', 'wins') | ('same HF', 'p') | ('matched cost', 'cells') | ('matched cost', 'ratio') | ('matched cost', 'wins') | ('matched cost', 'p') |
|---|---|---|---|---|---|---|---|---|
| smt_mfk | 1.51e+03 | 0.0687 | 0.876 | 2.98e-08 | 1.51e+03 | 0.183 | 0.672 | 0.000427 |
| cv_select | 1.62e+03 | 0.305 | 0.847 | 1.49e-08 | 1.62e+03 | 0.813 | 0.576 | 0.248 |
| ar1_cokriging | 1.59e+03 | 0.342 | 0.845 | 2.09e-07 | 1.59e+03 | 0.913 | 0.524 | 0.562 |
| ar1_rhox | 1.62e+03 | 0.356 | 0.805 | 6.41e-07 | 1.62e+03 | 0.948 | 0.518 | 0.714 |
| nargp | 1.6e+03 | 0.371 | 0.833 | 1.31e-06 | 1.6e+03 | 0.99 | 0.527 | 0.934 |
| bridge_add | 1.59e+03 | 0.509 | 0.795 | 0.00388 | 1.59e+03 | 1.36 | 0.497 | 0.348 |
| hierarchical_kriging | 1.6e+03 | 0.702 | 0.657 | 0.117 | 1.6e+03 | 1.88 | 0.424 | 0.111 |
| mf_transfer_nn | 1.62e+03 | 0.997 | 0.59 | 0.822 | 1.62e+03 | 2.65 | 0.245 | 3.52e-05 |
| space_mapping | 1.59e+03 | 1.44 | 0.434 | 0.162 | 1.59e+03 | 3.83 | 0.228 | 5.49e-05 |
| mf_nn | 1.62e+03 | 1.88 | 0.378 | 0.00325 | 1.62e+03 | 5.01 | 0.159 | 2.83e-07 |
| bridge_mult | 1.6e+03 | 3.04 | 0.36 | 0.0121 | 1.6e+03 | 8.13 | 0.237 | 4.09e-05 |
| feature_aug | 1.59e+03 | 5.04 | 0.112 | 2.98e-08 | 1.59e+03 | 13.5 | 0.0276 | 1.49e-08 |

Oracle (best MF method per cell) vs matched HF-only: ratio 0.145, wins 0.812.

## 2. By pair family, matched cost (ratio; wins in brackets)

| fam | ar1_cokriging | ar1_rhox | bridge_add | bridge_mult | cv_select | feature_aug | hierarchical_kriging | mf_nn | mf_transfer_nn | nargp | smt_mfk | space_mapping |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.51 (84%) | 0.53 (82%) | 0.43 (88%) | 3.00 (38%) | 0.46 (86%) | 13.63 (0%) | 0.68 (67%) | 4.34 (15%) | 1.64 (34%) | 0.79 (67%) | 0.15 (81%) | 3.30 (19%) |
| mf2 | 0.86 (55%) | 0.88 (55%) | 1.25 (49%) | 11.89 (26%) | 0.85 (58%) | 14.87 (7%) | 3.38 (43%) | 5.13 (19%) | 3.45 (22%) | 1.00 (54%) | 0.08 (79%) | 4.42 (28%) |
| nonlinear | 2.36 (5%) | 2.51 (5%) | 12.34 (0%) | 18.01 (0%) | 2.33 (9%) | 12.30 (0%) | 3.29 (3%) | 7.78 (4%) | 3.05 (18%) | 2.65 (9%) | 1.08 (25%) | 5.06 (6%) |
| shifted | 0.69 (67%) | 0.71 (65%) | 0.92 (64%) | 1.89 (47%) | 0.40 (86%) | 9.43 (0%) | 1.05 (53%) | 1.86 (35%) | 1.14 (47%) | 0.38 (87%) | 0.25 (72%) | 1.31 (49%) |
| weak | 0.98 (47%) | 1.07 (47%) | 0.87 (49%) | 15.30 (4%) | 0.89 (49%) | 15.88 (0%) | 1.06 (44%) | 9.45 (2%) | 4.19 (5%) | 1.17 (44%) | 0.27 (57%) | 6.68 (3%) |

### By n_H / d and n_L / n_H, matched cost (best MF method per cell)

|  | ratio | wins | cells |
|---|---|---|---|
| (3, 4) | 0.361 | 0.807 | 270 |
| (3, 16) | 0.577 | 0.481 | 270 |
| (10, 4) | 0.118 | 0.926 | 270 |
| (10, 16) | 0.126 | 0.878 | 270 |
| (30, 4) | 0.0326 | 0.963 | 270 |
| (30, 16) | 0.0901 | 0.815 | 270 |

## 3. Does the low/high correlation r predict the gain? (Spearman between r and log ratio at matched cost, over pairs; negative = higher r, bigger gain)

|  | spearman |
|---|---|
| ar1_cokriging | -0.332 |
| ar1_rhox | -0.361 |
| bridge_add | -0.219 |
| bridge_mult | -0.371 |
| cv_select | -0.413 |
| feature_aug | -0.1 |
| hierarchical_kriging | -0.416 |
| mf_nn | -0.277 |
| mf_transfer_nn | -0.393 |
| nargp | -0.454 |
| smt_mfk | 0.0501 |
| space_mapping | -0.556 |

## 4. Best method per pair (geometric-mean NRMSE over all cells)

|  | pairs won |
|---|---|
| smt_mfk | 14 |
| hf_only@matched | 6 |
| bridge_add | 2 |
| nargp | 2 |
| bridge_mult | 1 |
| ar1_rhox | 1 |
| cv_select | 1 |

| pair | best | r | best/matched-HF |
|---|---|---|---|
| mf2_bohachevsky | hf_only@matched | 0.851 | 1 |
| mf2_booth | smt_mfk | 0.922 | 0.0529 |
| mf2_borehole | smt_mfk | 1 | 0.162 |
| mf2_branin | smt_mfk | 0.179 | 0.175 |
| mf2_currin | bridge_mult | 0.997 | 0.468 |
| mf2_forrester | smt_mfk | 0.723 | 0.000141 |
| mf2_hartmann6 | ar1_rhox | 0.913 | 0.71 |
| mf2_himmelblau | smt_mfk | 0.757 | 0.0907 |
| mf2_park91a | hf_only@matched | 0.993 | 1 |
| mf2_park91b | smt_mfk | 1 | 0.00395 |
| mf2_six_hump_camelback | smt_mfk | 0.729 | 0.0659 |
| mf_linear__kink_floor_d2 | bridge_add | 0.95 | 0.375 |
| mf_linear__rotated_ridge_d5 | cv_select | 0.95 | 0.303 |
| mf_linear__smooth_trig_d2 | smt_mfk | 0.95 | 0.0369 |
| mf_linear__smooth_trig_d5 | smt_mfk | 0.95 | 0.0639 |
| mf_nonlinear__kink_floor_d2 | hf_only@matched | 0.968 | 1 |
| mf_nonlinear__rotated_ridge_d5 | hf_only@matched | 0.986 | 1 |
| mf_nonlinear__smooth_trig_d2 | smt_mfk | 0.954 | 0.201 |
| mf_nonlinear__smooth_trig_d5 | hf_only@matched | 0.952 | 1 |
| mf_shifted__kink_floor_d2 | nargp | 0.992 | 0.422 |
| mf_shifted__rotated_ridge_d5 | nargp | 1 | 0.205 |
| mf_shifted__smooth_trig_d2 | smt_mfk | 0.998 | 0.0355 |
| mf_shifted__smooth_trig_d5 | smt_mfk | 0.999 | 0.269 |
| mf_weak__kink_floor_d2 | bridge_add | 0.5 | 0.625 |
| mf_weak__rotated_ridge_d5 | hf_only@matched | 0.5 | 1 |
| mf_weak__smooth_trig_d2 | smt_mfk | 0.5 | 0.0913 |
| mf_weak__smooth_trig_d5 | smt_mfk | 0.5 | 0.0432 |

## 5. Decision rules at matched cost (vs spending the whole budget on HF points)

|  | uses_MF | ratio | wins | worse_10pct |
|---|---|---|---|---|
| r² ≥ 0.5: co-kriging, else HF-only | 0.8 | 0.878 | 0.442 | 0.319 |
| r² ≥ 0.8: co-kriging, else HF-only | 0.672 | 0.869 | 0.393 | 0.258 |
| r² ≥ 0.9: co-kriging, else HF-only | 0.552 | 0.889 | 0.307 | 0.226 |
| r² ≥ 0.95: co-kriging, else HF-only | 0.378 | 0.88 | 0.227 | 0.137 |
| r² ≥ 0.99: co-kriging, else HF-only | 0.234 | 0.825 | 0.19 | 0.0383 |
| always cv_select | nan | 0.816 | 0.574 | 0.393 |
| always ar1_cokriging | nan | 0.913 | 0.524 | 0.432 |
| oracle best MF method | 1 | 0.141 | 0.813 | 0.168 |

### What cv_select chose, by pair family (share of cells)

| fam | ar1_cokriging | bridge_add | hf_only | hierarchical_kriging | nargp |
|---|---|---|---|---|---|
| linear | 0.25 | 0.57 | 0.00 | 0.07 | 0.10 |
| mf2 | 0.40 | 0.19 | 0.05 | 0.17 | 0.18 |
| nonlinear | 0.22 | 0.02 | 0.30 | 0.07 | 0.39 |
| shifted | 0.14 | 0.05 | 0.00 | 0.02 | 0.79 |
| weak | 0.11 | 0.59 | 0.05 | 0.12 | 0.14 |
