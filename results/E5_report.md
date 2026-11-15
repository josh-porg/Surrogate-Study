# E5 — adaptive sampling vs a Sobol' design (8580 designs, 154429 deliverable fits, 22 problems)

Ratio < 1: the scout's design gives a lower test NRMSE than the Sobol' design of the same size, for the same deliverable. Wilcoxon over problems.

## 1. Each scout vs Sobol', all deliverables and sizes

| scout | ratio | wins | worse_10pct | p | t_design_s |
|---|---|---|---|---|---|
| lola_voronoi | 0.906 | 0.516 | 0.241 | 0.0391 | 9.2 |
| gp:tead [kb] | 0.93 | 0.494 | 0.282 | 0.353 | 21.3 |
| gp:imse_alc [kb] | 0.93 | 0.538 | 0.184 | 0.0251 | 17 |
| mlp_ens:masa [kb] | 0.945 | 0.444 | 0.367 | 0.353 | 1.95e+03 |
| gp:imse_alc | 0.953 | 0.493 | 0.231 | 0.0854 | 20.3 |
| gp:mepe [kb] | 0.96 | 0.446 | 0.332 | 0.424 | 107 |
| gp:mepe | 0.968 | 0.428 | 0.347 | 0.656 | 128 |
| seq_maximin | 0.97 | 0.444 | 0.3 | 0.503 | 3.6 |
| gp:max_variance [kb] | 0.977 | 0.406 | 0.378 | 1 | 18.8 |
| mipt | 0.989 | 0.453 | 0.232 | 0.633 | 4.33 |
| gp:sfcvt [kb] | 1.04 | 0.339 | 0.439 | 0.0738 | 39.9 |
| mlp_ens:max_variance [kb] | 1.07 | 0.334 | 0.502 | 0.0391 | 1.76e+03 |
| gp:max_variance | 1.08 | 0.283 | 0.508 | 0.000593 | 22.6 |
| mlp_ens:max_variance | 1.15 | 0.25 | 0.592 | 0.000121 | 730 |
| random | 1.17 | 0.235 | 0.499 | 4.77e-07 | 0.1 |
| gp:eigf | 1.17 | 0.309 | 0.514 | 0.00126 | 27.9 |
| gp:eigf [kb] | 1.18 | 0.308 | 0.516 | 0.00109 | 25.7 |
| gbt_committee:max_variance [kb] | 1.45 | 0.169 | 0.687 | 4.77e-07 | 212 |
| rf:max_variance [kb] | 1.54 | 0.166 | 0.717 | 6.68e-06 | 136 |
| gbt_committee:max_variance | 1.56 | 0.136 | 0.74 | 4.77e-07 | 105 |
| rf:max_variance | 1.61 | 0.142 | 0.747 | 1.43e-06 | 71.1 |
| ngboost:max_variance [kb] | 1.65 | 0.11 | 0.749 | 4.77e-07 | 852 |
| qrf:max_variance [kb] | 1.67 | 0.144 | 0.753 | 4.77e-07 | 167 |
| qrf:max_variance | 1.7 | 0.136 | 0.771 | 4.77e-07 | 135 |
| ngboost:max_variance | 1.73 | 0.0982 | 0.776 | 4.77e-07 | 358 |

## 2. Scout × deliverable (ratio to Sobol')

| scout | catboost | gp | mlp_ens[lbfgs] | pce_lars | rbf_tps | rf |
|---|---|---|---|---|---|---|
| lola_voronoi | 0.91 | 0.884 | 0.877 | 0.972 | 0.818 | 0.985 |
| gp:tead [kb] | 0.932 | 0.97 | 0.851 | 1.01 | 0.815 | 1.01 |
| gp:imse_alc [kb] | 0.963 | 0.818 | 0.9 | 0.989 | 0.925 | 0.998 |
| mlp_ens:masa [kb] | 0.921 | 1.03 | 0.763 | 1.04 | 0.906 | 1.04 |
| gp:imse_alc | 1.01 | 0.84 | 0.929 | 0.976 | 0.947 | 1.03 |
| gp:mepe [kb] | 0.899 | 0.958 | 0.954 | 1.05 | 0.896 | 1.01 |
| gp:mepe | 0.917 | 0.939 | 0.984 | 1.04 | 0.915 | 1.02 |
| seq_maximin | 0.99 | 0.94 | 0.914 | 0.979 | 0.929 | 1.07 |
| gp:max_variance [kb] | 0.993 | 0.879 | 0.897 | 1.05 | 0.981 | 1.07 |
| mipt | 1.01 | 0.99 | 0.97 | 0.948 | 0.996 | 1.02 |
| gp:sfcvt [kb] | 1.09 | 1.04 | 0.953 | 0.985 | 1.07 | 1.1 |
| mlp_ens:max_variance [kb] | 0.983 | 1.21 | 0.831 | 1.25 | 1.07 | 1.12 |
| gp:max_variance | 1.1 | 1.03 | 0.981 | 1.13 | 1.09 | 1.2 |
| mlp_ens:max_variance | 1.1 | 1.28 | 0.932 | 1.21 | 1.18 | 1.22 |
| random | 1.23 | 1.17 | 1.17 | 1.14 | 1.16 | 1.15 |
| gp:eigf | 0.996 | 1.28 | 1.33 | 1.28 | 1.09 | 1.08 |
| gp:eigf [kb] | 0.985 | 1.29 | 1.33 | 1.33 | 1.1 | 1.08 |
| gbt_committee:max_variance [kb] | 1.32 | 1.68 | 1.51 | 1.52 | 1.45 | 1.24 |
| rf:max_variance [kb] | 1.36 | 1.87 | 1.75 | 1.64 | 1.56 | 1.19 |
| gbt_committee:max_variance | 1.45 | 1.81 | 1.64 | 1.63 | 1.56 | 1.32 |
| rf:max_variance | 1.45 | 1.92 | 1.85 | 1.7 | 1.62 | 1.25 |
| ngboost:max_variance [kb] | 1.57 | 1.83 | 1.78 | 1.87 | 1.58 | 1.36 |
| qrf:max_variance [kb] | 1.44 | 2.07 | 2 | 1.71 | 1.67 | 1.27 |
| qrf:max_variance | 1.49 | 1.98 | 2.01 | 1.89 | 1.65 | 1.3 |
| ngboost:max_variance | 1.66 | 1.93 | 1.91 | 1.87 | 1.65 | 1.41 |

## 3. Does the deliverable need to match the scout? (variance-rule scouts; ratio to Sobol')

|  | own_deliverable | other_deliverables |
|---|---|---|
| gp | 0.999 | 1.03 |
| rf | 1.23 | 1.67 |
| mlp_ens | 0.862 | 1.12 |

## 3b. The same scout with E5's batch vs the kriging-believer batch (seeds 0-4, ratio to Sobol')

|  | E5_batch | believer_batch |
|---|---|---|
| gbt_committee:max_variance | 1.57 | 1.45 |
| gp:eigf | 1.17 | 1.18 |
| gp:imse_alc | 0.955 | 0.93 |
| gp:max_variance | 1.09 | 0.977 |
| gp:mepe | 0.965 | 0.96 |
| mlp_ens:max_variance | 1.15 | 1.07 |
| ngboost:max_variance | 1.73 | 1.65 |
| qrf:max_variance | 1.72 | 1.67 |
| rf:max_variance | 1.63 | 1.54 |

## 4. By budget (ratio to Sobol')

| scout | 10d | 20d | 50d |
|---|---|---|---|
| lola_voronoi | 0.935 | 0.957 | 0.83 |
| gp:tead [kb] | 0.955 | 0.968 | 0.869 |
| gp:imse_alc [kb] | 0.952 | 0.941 | 0.898 |
| mlp_ens:masa [kb] | 0.987 | 0.992 | 0.863 |
| gp:imse_alc | 0.994 | 0.972 | 0.898 |
| gp:mepe [kb] | 0.98 | 0.977 | 0.924 |
| gp:mepe | 0.994 | 0.984 | 0.929 |
| seq_maximin | 0.977 | 0.996 | 0.937 |
| gp:max_variance [kb] | 0.987 | 1.01 | 0.934 |
| mipt | 0.933 | 1.02 | 1.02 |
| gp:sfcvt [kb] | 1.03 | 1.11 | 0.982 |
| mlp_ens:max_variance [kb] | 1.11 | 1.1 | 0.994 |
| gp:max_variance | 1.1 | 1.14 | 1.02 |
| mlp_ens:max_variance | 1.16 | 1.22 | 1.07 |
| random | 1.15 | 1.19 | 1.16 |
| gp:eigf | 1.1 | 1.2 | 1.21 |
| gp:eigf [kb] | 1.13 | 1.19 | 1.22 |
| gbt_committee:max_variance [kb] | 1.29 | 1.47 | 1.6 |
| rf:max_variance [kb] | 1.27 | 1.57 | 1.84 |
| gbt_committee:max_variance | 1.39 | 1.59 | 1.71 |
| rf:max_variance | 1.35 | 1.66 | 1.87 |
| ngboost:max_variance [kb] | 1.31 | 1.72 | 2 |
| qrf:max_variance [kb] | 1.36 | 1.66 | 2.07 |
| qrf:max_variance | 1.38 | 1.77 | 2 |
| ngboost:max_variance | 1.39 | 1.81 | 2.05 |

### By noise level

| scout | 0.0 | 0.01 |
|---|---|---|
| lola_voronoi | 0.895 | 0.917 |
| gp:tead [kb] | 0.92 | 0.939 |
| gp:imse_alc [kb] | 0.935 | 0.925 |
| mlp_ens:masa [kb] | 0.935 | 0.955 |
| gp:imse_alc | 0.946 | 0.96 |
| gp:mepe [kb] | 0.951 | 0.969 |
| gp:mepe | 0.953 | 0.984 |
| seq_maximin | 0.963 | 0.977 |
| gp:max_variance [kb] | 0.966 | 0.988 |
| mipt | 0.988 | 0.99 |
| gp:sfcvt [kb] | 1.02 | 1.06 |
| mlp_ens:max_variance [kb] | 1.07 | 1.07 |
| gp:max_variance | 1.07 | 1.1 |
| mlp_ens:max_variance | 1.14 | 1.16 |
| random | 1.17 | 1.16 |
| gp:eigf | 1.17 | 1.17 |
| gp:eigf [kb] | 1.18 | 1.18 |
| gbt_committee:max_variance [kb] | 1.46 | 1.44 |
| rf:max_variance [kb] | 1.55 | 1.54 |
| gbt_committee:max_variance | 1.57 | 1.55 |
| rf:max_variance | 1.61 | 1.61 |
| ngboost:max_variance [kb] | 1.71 | 1.6 |
| qrf:max_variance [kb] | 1.67 | 1.67 |
| qrf:max_variance | 1.74 | 1.66 |
| ngboost:max_variance | 1.74 | 1.72 |

### Best scout per problem (ratio to Sobol'; > 1 means Sobol' was best)

| problem | best | ratio |
|---|---|---|
| ackley_d5 | lola_voronoi | 0.991 |
| axis_step_d5 | rf:max_variance [kb] | 0.828 |
| branin | gp:max_variance [kb] | 0.73 |
| camel6 | gp:tead [kb] | 0.582 |
| goldpr | mlp_ens:masa [kb] | 0.612 |
| kink_floor_d2 | gp:imse_alc [kb] | 0.843 |
| levy_d5 | lola_voronoi | 0.988 |
| michal_d5 | gp:tead [kb] | 1 |
| multiscale_d2 | seq_maximin | 0.968 |
| nonstationary_d2 | mlp_ens:masa [kb] | 0.784 |
| plateau_bump_d2 | gp:tead [kb] | 0.691 |
| regime_d5 | gp:max_variance [kb] | 0.977 |
| rosen_d5 | lola_voronoi | 0.939 |
| rotated_ridge_d5 | gp:imse_alc [kb] | 0.976 |
| smooth_trig_d5 | seq_maximin | 0.935 |
| stybtang_d5 | mipt | 0.983 |
| thesis_anisotropic | lola_voronoi | 0.773 |
| thesis_clamp_floor | gp:imse_alc [kb] | 0.909 |
| thesis_cliff | gp:imse_alc [kb] | 1.02 |
| thesis_collapse_knee | mlp_ens:masa [kb] | 0.821 |
| thesis_oscillatory | gp:imse_alc [kb] | 0.974 |
| thesis_smooth | gp:imse_alc [kb] | 0.833 |
