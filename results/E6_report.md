# E6 — multi-fidelity methods vs high-fidelity only (14416 fits, 27 pairs; 164 timeouts/errors dropped)

Ratio < 1: the MF method has lower test NRMSE. 'same HF' = HF-only on the same n_H points (the low-fidelity data is free); 'matched' = HF-only given the low-fidelity budget as extra HF points (n_H + n_L / κ).

## 1. Each MF method vs HF-only

| method | ('same HF', 'cells') | ('same HF', 'ratio') | ('same HF', 'wins') | ('same HF', 'p') | ('matched cost', 'cells') | ('matched cost', 'ratio') | ('matched cost', 'wins') | ('matched cost', 'p') |
|---|---|---|---|---|---|---|---|---|
| ar1_cokriging | 1.59e+03 | 0.342 | 0.845 | 2.09e-07 | 1.59e+03 | 0.913 | 0.524 | 0.562 |
| nargp | 1.6e+03 | 0.371 | 0.833 | 1.31e-06 | 1.6e+03 | 0.99 | 0.527 | 0.934 |
| bridge_add | 1.59e+03 | 0.509 | 0.795 | 0.00388 | 1.59e+03 | 1.36 | 0.497 | 0.348 |
| hierarchical_kriging | 1.6e+03 | 0.702 | 0.657 | 0.117 | 1.6e+03 | 1.88 | 0.424 | 0.111 |
| space_mapping | 1.59e+03 | 1.44 | 0.434 | 0.162 | 1.59e+03 | 3.83 | 0.228 | 5.49e-05 |
| mf_nn | 1.62e+03 | 1.88 | 0.378 | 0.00325 | 1.62e+03 | 5.01 | 0.159 | 2.83e-07 |
| bridge_mult | 1.6e+03 | 3.04 | 0.36 | 0.0121 | 1.6e+03 | 8.13 | 0.237 | 4.09e-05 |

Oracle (best MF method per cell) vs matched HF-only: ratio 0.713, wins 0.626.

## 2. By pair family, matched cost (ratio; wins in brackets)

| fam | ar1_cokriging | bridge_add | bridge_mult | hierarchical_kriging | mf_nn | nargp | space_mapping |
|---|---|---|---|---|---|---|---|
| linear | 0.51 (84%) | 0.43 (88%) | 3.00 (38%) | 0.68 (67%) | 4.34 (15%) | 0.79 (67%) | 3.30 (19%) |
| mf2 | 0.86 (55%) | 1.25 (49%) | 11.89 (26%) | 3.38 (43%) | 5.13 (19%) | 1.00 (54%) | 4.42 (28%) |
| nonlinear | 2.36 (5%) | 12.34 (0%) | 18.01 (0%) | 3.29 (3%) | 7.78 (4%) | 2.65 (9%) | 5.06 (6%) |
| shifted | 0.69 (67%) | 0.92 (64%) | 1.89 (47%) | 1.05 (53%) | 1.86 (35%) | 0.38 (87%) | 1.31 (49%) |
| weak | 0.98 (47%) | 0.87 (49%) | 15.30 (4%) | 1.06 (44%) | 9.45 (2%) | 1.17 (44%) | 6.68 (3%) |

### By n_H / d and n_L / n_H, matched cost (best MF method per cell)

|  | ratio | wins | cells |
|---|---|---|---|
| (3, 4) | 0.511 | 0.763 | 270 |
| (3, 16) | 1.16 | 0.459 | 270 |
| (10, 4) | 0.545 | 0.696 | 270 |
| (10, 16) | 0.761 | 0.563 | 270 |
| (30, 4) | 0.548 | 0.77 | 270 |
| (30, 16) | 0.97 | 0.504 | 270 |

## 3. Does the low/high correlation r predict the gain? (Spearman between r and log ratio at matched cost, over pairs; negative = higher r, bigger gain)

|  | spearman |
|---|---|
| ar1_cokriging | -0.332 |
| bridge_add | -0.219 |
| bridge_mult | -0.371 |
| hierarchical_kriging | -0.416 |
| mf_nn | -0.277 |
| nargp | -0.454 |
| space_mapping | -0.556 |

## 4. Best method per pair (geometric-mean NRMSE over all cells)

|  | pairs won |
|---|---|
| hf_only@matched | 12 |
| bridge_add | 6 |
| nargp | 4 |
| bridge_mult | 3 |
| ar1_cokriging | 2 |

| pair | best | r | best/matched-HF |
|---|---|---|---|
| mf2_bohachevsky | hf_only@matched | 0.851 | 1 |
| mf2_booth | hf_only@matched | 0.922 | 1 |
| mf2_borehole | bridge_mult | 1 | 0.35 |
| mf2_branin | hf_only@matched | 0.179 | 1 |
| mf2_currin | bridge_mult | 0.997 | 0.468 |
| mf2_forrester | ar1_cokriging | 0.723 | 0.229 |
| mf2_hartmann6 | bridge_mult | 0.913 | 0.789 |
| mf2_himmelblau | hf_only@matched | 0.757 | 1 |
| mf2_park91a | hf_only@matched | 0.993 | 1 |
| mf2_park91b | ar1_cokriging | 1 | 0.158 |
| mf2_six_hump_camelback | hf_only@matched | 0.729 | 1 |
| mf_linear__kink_floor_d2 | bridge_add | 0.95 | 0.375 |
| mf_linear__rotated_ridge_d5 | bridge_add | 0.95 | 0.322 |
| mf_linear__smooth_trig_d2 | bridge_add | 0.95 | 0.944 |
| mf_linear__smooth_trig_d5 | bridge_add | 0.95 | 0.327 |
| mf_nonlinear__kink_floor_d2 | hf_only@matched | 0.968 | 1 |
| mf_nonlinear__rotated_ridge_d5 | hf_only@matched | 0.986 | 1 |
| mf_nonlinear__smooth_trig_d2 | hf_only@matched | 0.954 | 1 |
| mf_nonlinear__smooth_trig_d5 | hf_only@matched | 0.952 | 1 |
| mf_shifted__kink_floor_d2 | nargp | 0.992 | 0.422 |
| mf_shifted__rotated_ridge_d5 | nargp | 1 | 0.205 |
| mf_shifted__smooth_trig_d2 | nargp | 0.998 | 0.786 |
| mf_shifted__smooth_trig_d5 | nargp | 0.999 | 0.315 |
| mf_weak__kink_floor_d2 | bridge_add | 0.5 | 0.625 |
| mf_weak__rotated_ridge_d5 | hf_only@matched | 0.5 | 1 |
| mf_weak__smooth_trig_d2 | hf_only@matched | 0.5 | 1 |
| mf_weak__smooth_trig_d5 | bridge_add | 0.5 | 0.253 |
