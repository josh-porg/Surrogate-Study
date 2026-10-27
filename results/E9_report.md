# E9 — optimization on the surrogate (2315 tasks, 38 problems)

Regret normalised by std(f); success = regret < 0.01. Instances with an unconfirmed f* are excluded.

## 1. Surrogates at the full budget (500d)

| surr | success_avg | geo_regret_avg | best optimizer | success_best | median NRMSE |
|---|---|---|---|---|---|
| truth | 0.808 | 7.75e-07 | adam|fd | 1 | 0 |
| ens_press | 0.46 | 0.0102 | lbfgsb|model | 0.521 | 0.137 |
| mlp_ens[opt=lbfgs] | 0.382 | 0.0171 | cmaes|none | 0.482 | 0.18 |
| gp | 0 | 31.3 | adam|fd | 0 | 0.679 |
| catboost | 0 | 44.8 | trust_constr|model | 0 | 0.699 |
| as_gp | 0 | 49 | adam|fd | 0 | 0.951 |
| idw | 0 | 45.9 | adam|fd | 0 | 0.963 |
| hetgp | 0 | 33.2 | adam|fd | 0 | 0.674 |
| lgbm | 0 | 45.5 | adam|fd | 0 | 0.696 |
| krr | 0 | 23.1 | adam|fd | 0 | 0.672 |
| mars | 0 | 42.5 | adam|fd | 0 | 0.756 |
| pce_lars | 0 | 20.6 | adam|fd | 0 | 0.733 |
| rbf_tps | 0 | 33.9 | adam|fd | 0 | 0.671 |
| rsm2 | 0 | 47.7 | slsqp|fd | 0 | 0.741 |

## 2. H6: does surrogate accuracy predict optimization regret? (Spearman ρ of NRMSE vs regret rankings across surrogates, per problem × instance × n × noise × optimizer)

| cls | count | mean | 25% | 50% | 75% |
|---|---|---|---|---|---|
| first-order SA | 50.00 | 0.46 | 0.37 | 0.41 | 0.64 |
| global | 20.00 | 0.53 | 0.36 | 0.50 | 0.77 |
| local derivative-free | 10.00 | 0.45 | 0.30 | 0.46 | 0.62 |
| local gradient | 40.00 | 0.45 | 0.26 | 0.43 | 0.55 |

Overall median ρ = 0.43; share of settings with ρ > 0.5: 36%.

## 3. Surrogate gradient vs finite differences (< 1: the surrogate's gradient gives lower regret)

| method | model / fd regret (geo) | model better (share) | pairs |
|---|---|---|---|
| adam | 0.96 | 0.34 | 2271 |
| basin_hopping | 0.22 | 0.52 | 2271 |
| bb_gd | 1.26 | 0.33 | 2271 |
| ipopt_multistart | 1.00 | 0.00 | 2 |
| lbfgsb | 0.13 | 0.54 | 2271 |
| robbins_monro | 0.69 | 0.48 | 2271 |
| sgd_momentum | 0.93 | 0.36 | 2271 |
| slsqp | 0.16 | 0.53 | 2271 |
| trust_constr | 0.22 | 0.61 | 2271 |

By surrogate:

| surr | model / fd (geo) |
|---|---|
| as_gp | 0.99 |
| catboost | 1.00 |
| ens_press | 0.91 |
| gp | 1.00 |
| hetgp | 1.00 |
| idw | 0.99 |
| krr | 1.00 |
| lgbm | 1.00 |
| mars | 1.00 |
| mlp_ens[opt=lbfgs] | 0.18 |
| pce_lars | 1.01 |
| rbf_tps | 1.00 |
| rsm2 | 1.00 |
| truth | 0.14 |

## 4. Optimizing the surrogate vs taking the best training point

Best optimizer on the surrogate beats the best training point in 76% of tasks; geometric-mean regret 0.00146 vs 0.117.

## 5. Success rate vs evaluations (×d), averaged over optimizers

| surr | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| as_gp | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ens_press | 0.30 | 0.35 | 0.40 | 0.43 | 0.44 | 0.46 |
| mlp_ens[opt=lbfgs] | 0.23 | 0.28 | 0.32 | 0.36 | 0.37 | 0.38 |
| truth | 0.40 | 0.54 | 0.60 | 0.73 | 0.81 | 0.81 |

## 6. Best surrogate + optimizer per problem (26 distinct pairs over 38 problems)

| problem | best pair (geo-mean regret) |
|---|---|
| ackley_d10 | pce_lars + robbins_monro|fd |
| ackley_d5 | ens_press + trust_constr|fd |
| axis_step_d5 | mlp_ens[opt=lbfgs] + pso|none |
| booth | ens_press + bb_gd|fd |
| borehole | mlp_ens[opt=lbfgs] + pso|none |
| branin | ens_press + robbins_monro|model |
| camel6 | ens_press + slsqp|model |
| composite_d5 | ens_press + cobyla|none |
| dixonpr_d5 | ens_press + spsa|none |
| goldpr | ens_press + kiefer_wolfowitz|none |
| gp_draw_d2 | ens_press + cmaes|none |
| griewank_d5 | ens_press + bb_gd|model |
| kink_floor_d5 | ens_press + adam|fd |
| levy_d10 | ens_press + basin_hopping|fd |
| levy_d5 | ens_press + trust_constr|model |
| michal_d5 | ens_press + differential_evolution|none |
| monotone_d5 | ens_press + adam|model |
| oblique_step_d10 | mlp_ens[opt=lbfgs] + pso|none |
| piston | ens_press + adam|model |
| plateau_bump_d5 | mlp_ens[opt=lbfgs] + bb_gd|model |
| pole_knee_d5 | mlp_ens[opt=lbfgs] + pso|none |
| quadratic_d5 | mlp_ens[opt=lbfgs] + slsqp|model |
| rastr | ens_press + cobyla|none |
| rastr_d5 | ens_press + spsa|none |
| regime_d5 | mlp_ens[opt=lbfgs] + pso|none |
| rosen_d5 | ens_press + basin_hopping|fd |
| rotated_ridge_d5 | mlp_ens[opt=lbfgs] + spsa|none |
| rothyp_d10 | ens_press + trust_constr|fd |
| schaffer2 | ens_press + trust_constr|model |
| smooth_trig_d10 | mlp_ens[opt=lbfgs] + robbins_monro|fd |
| smooth_trig_d5 | mlp_ens[opt=lbfgs] + cobyla|none |
| stybtang | mlp_ens[opt=lbfgs] + differential_evolution|none |
| sumsqu_d5 | ens_press + lbfgsb|model |
| thesis_collapse_knee | mlp_ens[opt=lbfgs] + sgd_momentum|fd |
| tree_draw_d5 | ens_press + nelder_mead|none |
| trid_d5 | ens_press + trust_constr|fd |
| wingweight | ens_press + basin_hopping|model |
| zakharov_d5 | mlp_ens[opt=lbfgs] + sgd_momentum|fd |

## 7. H7 cost crossover: no E9b records yet
