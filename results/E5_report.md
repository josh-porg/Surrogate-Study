# E5 — adaptive sampling vs a Sobol' design (5720 designs, 102953 deliverable fits, 22 problems)

Ratio < 1: the scout's design gives a lower test NRMSE than the Sobol' design of the same size, for the same deliverable. Wilcoxon over problems.

## 1. Each scout vs Sobol', all deliverables and sizes

| scout | ratio | wins | worse_10pct | p | t_design_s |
|---|---|---|---|---|---|
| lola_voronoi | 0.906 | 0.516 | 0.241 | 0.0391 | 9.2 |
| gp:imse_alc | 0.953 | 0.493 | 0.231 | 0.0854 | 20.3 |
| gp:mepe | 0.968 | 0.428 | 0.347 | 0.656 | 128 |
| seq_maximin | 0.97 | 0.444 | 0.3 | 0.503 | 3.6 |
| gp:max_variance | 1.08 | 0.283 | 0.508 | 0.000593 | 22.6 |
| mlp_ens:max_variance | 1.15 | 0.25 | 0.592 | 0.000121 | 730 |
| random | 1.17 | 0.235 | 0.499 | 4.77e-07 | 0.1 |
| gp:eigf | 1.17 | 0.309 | 0.514 | 0.00126 | 27.9 |
| gbt_committee:max_variance | 1.56 | 0.136 | 0.74 | 4.77e-07 | 105 |
| rf:max_variance | 1.61 | 0.142 | 0.747 | 1.43e-06 | 71.1 |
| qrf:max_variance | 1.7 | 0.136 | 0.771 | 4.77e-07 | 135 |
| ngboost:max_variance | 1.73 | 0.0982 | 0.776 | 4.77e-07 | 358 |

## 2. Scout × deliverable (ratio to Sobol')

| scout | catboost | gp | mlp_ens[lbfgs] | pce_lars | rbf_tps | rf |
|---|---|---|---|---|---|---|
| lola_voronoi | 0.91 | 0.884 | 0.877 | 0.972 | 0.818 | 0.985 |
| gp:imse_alc | 1.01 | 0.84 | 0.929 | 0.976 | 0.947 | 1.03 |
| gp:mepe | 0.917 | 0.939 | 0.984 | 1.04 | 0.915 | 1.02 |
| seq_maximin | 0.99 | 0.94 | 0.914 | 0.979 | 0.929 | 1.07 |
| gp:max_variance | 1.1 | 1.03 | 0.981 | 1.13 | 1.09 | 1.2 |
| mlp_ens:max_variance | 1.1 | 1.28 | 0.932 | 1.21 | 1.18 | 1.22 |
| random | 1.23 | 1.17 | 1.17 | 1.14 | 1.16 | 1.15 |
| gp:eigf | 0.996 | 1.28 | 1.33 | 1.28 | 1.09 | 1.08 |
| gbt_committee:max_variance | 1.45 | 1.81 | 1.64 | 1.63 | 1.56 | 1.32 |
| rf:max_variance | 1.45 | 1.92 | 1.85 | 1.7 | 1.62 | 1.25 |
| qrf:max_variance | 1.49 | 1.98 | 2.01 | 1.89 | 1.65 | 1.3 |
| ngboost:max_variance | 1.66 | 1.93 | 1.91 | 1.87 | 1.65 | 1.41 |

## 3. Does the deliverable need to match the scout? (variance-rule scouts; ratio to Sobol')

|  | own_deliverable | other_deliverables |
|---|---|---|
| gp | 1.01 | 1.05 |
| rf | 1.25 | 1.7 |
| mlp_ens | 0.932 | 1.2 |

## 4. By budget (ratio to Sobol')

| scout | 10d | 20d | 50d |
|---|---|---|---|
| lola_voronoi | 0.935 | 0.957 | 0.83 |
| gp:imse_alc | 0.994 | 0.972 | 0.898 |
| gp:mepe | 0.994 | 0.984 | 0.929 |
| seq_maximin | 0.977 | 0.996 | 0.937 |
| gp:max_variance | 1.1 | 1.14 | 1.02 |
| mlp_ens:max_variance | 1.16 | 1.22 | 1.07 |
| random | 1.15 | 1.19 | 1.16 |
| gp:eigf | 1.1 | 1.2 | 1.21 |
| gbt_committee:max_variance | 1.39 | 1.59 | 1.71 |
| rf:max_variance | 1.35 | 1.66 | 1.87 |
| qrf:max_variance | 1.38 | 1.77 | 2 |
| ngboost:max_variance | 1.39 | 1.81 | 2.05 |

### By noise level

| scout | 0.0 | 0.01 |
|---|---|---|
| lola_voronoi | 0.895 | 0.917 |
| gp:imse_alc | 0.946 | 0.96 |
| gp:mepe | 0.953 | 0.984 |
| seq_maximin | 0.963 | 0.977 |
| gp:max_variance | 1.07 | 1.1 |
| mlp_ens:max_variance | 1.14 | 1.16 |
| random | 1.17 | 1.16 |
| gp:eigf | 1.17 | 1.17 |
| gbt_committee:max_variance | 1.57 | 1.55 |
| rf:max_variance | 1.61 | 1.61 |
| qrf:max_variance | 1.74 | 1.66 |
| ngboost:max_variance | 1.74 | 1.72 |

### Best scout per problem (ratio to Sobol'; > 1 means Sobol' was best)

| problem | best | ratio |
|---|---|---|
| ackley_d5 | lola_voronoi | 0.991 |
| axis_step_d5 | rf:max_variance | 0.899 |
| branin | gp:imse_alc | 0.747 |
| camel6 | gp:mepe | 0.614 |
| goldpr | lola_voronoi | 0.64 |
| kink_floor_d2 | seq_maximin | 0.9 |
| levy_d5 | lola_voronoi | 0.988 |
| michal_d5 | lola_voronoi | 1.01 |
| multiscale_d2 | seq_maximin | 0.968 |
| nonstationary_d2 | lola_voronoi | 0.792 |
| plateau_bump_d2 | lola_voronoi | 0.697 |
| regime_d5 | lola_voronoi | 1.01 |
| rosen_d5 | lola_voronoi | 0.939 |
| rotated_ridge_d5 | gp:imse_alc | 1.02 |
| smooth_trig_d5 | seq_maximin | 0.935 |
| stybtang_d5 | random | 1.06 |
| thesis_anisotropic | lola_voronoi | 0.773 |
| thesis_clamp_floor | gp:imse_alc | 0.954 |
| thesis_cliff | lola_voronoi | 1.02 |
| thesis_collapse_knee | lola_voronoi | 0.828 |
| thesis_oscillatory | gp:imse_alc | 0.983 |
| thesis_smooth | seq_maximin | 0.836 |
