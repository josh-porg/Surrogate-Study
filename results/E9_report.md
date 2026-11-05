# E9 — optimization on the surrogate (22673 tasks, 38 problems; records: E9_optim, E9_ipopt)

Regret normalised by std(f); success = regret < 0.01. Instances with an unconfirmed f* are excluded.

## 1. Surrogates at the full budget (500d)

| surr | success_avg | geo_regret_avg | best optimizer | success_best | median NRMSE |
|---|---|---|---|---|---|
| truth | 0.644 | 3.89e-05 | pso|none | 0.872 | 0 |
| ens_press | 0.458 | 0.00891 | lbfgsb|model | 0.514 | 0.189 |
| smt_krg | 0.426 | 0.00657 | slsqp|model | 0.481 | 0.13 |
| mlp_ens[opt=lbfgs] | 0.366 | 0.0199 | cmaes|none | 0.462 | 0.228 |
| hetgp | 0.41 | 0.011 | cmaes|none | 0.461 | 0.135 |
| gp | 0.408 | 0.0116 | cmaes|none | 0.459 | 0.144 |
| smt_kpls | 0.392 | 0.0102 | pso|none | 0.448 | 0.179 |
| rsm2 | 0.378 | 0.00965 | cobyla|none | 0.413 | 0.447 |
| pce_lars | 0.372 | 0.00926 | trust_constr|fd | 0.411 | 0.348 |
| as_gp | 0.323 | 0.0299 | ipopt_multistart|model | 0.375 | 0.215 |
| rbf_tps | 0.318 | 0.0171 | ipopt_multistart|model | 0.361 | 0.292 |
| krr | 0.3 | 0.0341 | pso|none | 0.332 | 0.292 |
| rf | 0.183 | 0.116 | direct|none | 0.325 | 0.46 |
| catboost | 0.121 | 0.305 | cmaes|none | 0.201 | 0.341 |
| mars | 0.172 | 0.0972 | slsqp|model | 0.2 | 0.337 |
| lgbm | 0.0959 | 0.336 | pso|none | 0.192 | 0.35 |
| idw | 0.0982 | 0.372 | direct|none | 0.15 | 0.836 |

## 2. H6: does surrogate accuracy predict optimization regret? (Spearman ρ of NRMSE vs regret rankings across surrogates, per problem × instance × n × noise × optimizer)

| cls | count | mean | 25% | 50% | 75% |
|---|---|---|---|---|---|
| first-order SA | 14324.00 | 0.43 | 0.17 | 0.47 | 0.74 |
| global | 5694.00 | 0.45 | 0.19 | 0.51 | 0.74 |
| local derivative-free | 2849.00 | 0.43 | 0.17 | 0.47 | 0.74 |
| local gradient | 14252.00 | 0.39 | 0.11 | 0.43 | 0.69 |

Overall median ρ = 0.46; share of settings with ρ > 0.5: 46%.

## 3. Surrogate gradient vs finite differences (< 1: the surrogate's gradient gives lower regret)

| method | model / fd regret (geo) | model better (share) | pairs |
|---|---|---|---|
| adam | 0.96 | 0.26 | 22091 |
| basin_hopping | 0.72 | 0.32 | 22091 |
| bb_gd | 0.97 | 0.24 | 22091 |
| ipopt_multistart | 0.46 | 0.45 | 22091 |
| lbfgsb | 0.67 | 0.30 | 22091 |
| robbins_monro | 0.74 | 0.37 | 22091 |
| sgd_momentum | 0.92 | 0.27 | 22091 |
| slsqp | 0.70 | 0.31 | 22091 |
| trust_constr | 0.67 | 0.41 | 22091 |

By surrogate:

| surr | model / fd (geo) |
|---|---|
| as_gp | 0.83 |
| catboost | 0.96 |
| ens_press | 0.89 |
| gp | 0.86 |
| hetgp | 0.82 |
| idw | 0.94 |
| krr | 0.90 |
| lgbm | 0.96 |
| mars | 0.96 |
| mlp_ens[opt=lbfgs] | 0.17 |
| pce_lars | 0.90 |
| rbf_tps | 0.90 |
| rf | 0.99 |
| rsm2 | 0.90 |
| smt_kpls | 0.38 |
| smt_krg | 0.49 |
| truth | 0.57 |

## 4. Optimizing the surrogate vs taking the best training point

Best optimizer on the surrogate beats the best training point in 69% of tasks; geometric-mean regret 0.00422 vs 0.107.

## 5. Success rate vs evaluations (×d), averaged over optimizers

| surr | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| ens_press | 0.29 | 0.35 | 0.40 | 0.43 | 0.44 | 0.46 |
| mlp_ens[opt=lbfgs] | 0.23 | 0.27 | 0.31 | 0.34 | 0.35 | 0.37 |
| smt_krg | 0.27 | 0.33 | 0.38 | 0.40 | 0.41 | 0.43 |
| truth | 0.37 | 0.45 | 0.53 | 0.58 | 0.62 | 0.64 |

## 6. Best surrogate + optimizer per problem (37 distinct pairs over 38 problems)

| problem | best pair (geo-mean regret) |
|---|---|
| ackley_d10 | pce_lars + robbins_monro|fd |
| ackley_d5 | smt_kpls + direct|none |
| axis_step_d5 | rbf_tps + cobyla|none |
| booth | rsm2 + adam|fd |
| borehole | rbf_tps + basin_hopping|fd |
| branin | smt_kpls + ipopt_multistart|model |
| camel6 | pce_lars + cmaes|none |
| composite_d5 | ens_press + ipopt_multistart|model |
| dixonpr_d5 | rsm2 + robbins_monro|fd |
| goldpr | idw + direct|none |
| gp_draw_d2 | catboost + pso|none |
| griewank_d5 | pce_lars + nelder_mead|none |
| kink_floor_d5 | as_gp + adam|fd |
| levy_d10 | krr + trust_constr|model |
| levy_d5 | krr + robbins_monro|model |
| michal_d5 | catboost + differential_evolution|none |
| monotone_d5 | ens_press + adam|model |
| oblique_step_d10 | mlp_ens[opt=lbfgs] + pso|none |
| piston | gp + cmaes|none |
| plateau_bump_d5 | smt_krg + differential_evolution|none |
| pole_knee_d5 | smt_krg + pso|none |
| quadratic_d5 | rsm2 + slsqp|fd |
| rastr | idw + direct|none |
| rastr_d5 | ens_press + spsa|none |
| regime_d5 | rsm2 + cobyla|none |
| rosen_d5 | ens_press + basin_hopping|fd |
| rotated_ridge_d5 | as_gp + trust_constr|fd |
| rothyp_d10 | pce_lars + basin_hopping|model |
| schaffer2 | idw + pso|none |
| smooth_trig_d10 | krr + kiefer_wolfowitz|none |
| smooth_trig_d5 | smt_kpls + lbfgsb|fd |
| stybtang | smt_kpls + trust_constr|model |
| sumsqu_d5 | pce_lars + pso|none |
| thesis_collapse_knee | smt_krg + lbfgsb|model |
| tree_draw_d5 | catboost + direct|none |
| trid_d5 | rsm2 + trust_constr|fd |
| wingweight | ens_press + basin_hopping|model |
| zakharov_d5 | as_gp + ipopt_multistart|model |

## 8. Cost: optimizers at matched evaluation budgets

Success at 10d / 50d / 500d evaluations (scored when the counter reaches the checkpoint, so overshooting the cap gives no advantage); budget used = evaluations ÷ (500·d), median and share of runs past the cap; wall = median seconds per run. Model-gradient calls are free here (not counted).

### on the surrogates

| opt | success_10 | success_50 | success_500 | used_median | past_cap | wall_s | rank_10 | rank_50 | rank_500 |
|---|---|---|---|---|---|---|---|---|---|
| pso|none | 0.175 | 0.314 | 0.351 | 1 | 0 | 1.03 | 15 | 7 | 1 |
| cobyla|none | 0.303 | 0.347 | 0.35 | 0.235 | 0 | 2.15 | 6 | 1 | 2 |
| cmaes|none | 0.16 | 0.308 | 0.347 | 0.502 | 0.264 | 1.17 | 18 | 8 | 3 |
| trust_constr|model | 0.315 | 0.327 | 0.335 | 0.145 | 0 | 3.49 | 3 | 4 | 4 |
| nelder_mead|none | 0.136 | 0.279 | 0.334 | 0.628 | 0 | 0.819 | 20 | 16 | 5 |
| ipopt_multistart|model | 0.304 | 0.325 | 0.333 | 0.724 | 0.451 | 2.26 | 5 | 5 | 6 |
| lbfgsb|model | 0.329 | 0.333 | 0.332 | 0.0394 | 9.21e-05 | 0.429 | 2 | 2 | 7 |
| slsqp|model | 0.33 | 0.331 | 0.331 | 0.0344 | 9.21e-05 | 0.259 | 1 | 3 | 8 |
| basin_hopping|model | 0.313 | 0.319 | 0.32 | 0.0532 | 0.00341 | 0.553 | 4 | 6 | 9 |
| direct|none | 0.163 | 0.269 | 0.317 | 0.349 | 0.427 | 0.448 | 17 | 18 | 10 |
| adam|model | 0.205 | 0.305 | 0.306 | 1 | 0 | 11.8 | 12 | 9 | 11 |
| adam|fd | 0.0976 | 0.142 | 0.305 | 1 | 1 | 0.654 | 24 | 26 | 12 |
| slsqp|fd | 0.272 | 0.301 | 0.303 | 0.145 | 0.00299 | 0.166 | 8 | 10 | 13 |
| trust_constr|fd | 0.193 | 0.272 | 0.302 | 0.978 | 0.465 | 2.09 | 13 | 17 | 14 |
| lbfgsb|fd | 0.259 | 0.295 | 0.302 | 0.233 | 0.108 | 0.248 | 9 | 12 | 15 |
| sgd_momentum|model | 0.21 | 0.295 | 0.302 | 1 | 0 | 11.8 | 11 | 12 | 16 |
| bb_gd|model | 0.29 | 0.297 | 0.3 | 1 | 0 | 11.8 | 7 | 11 | 17 |
| bb_gd|fd | 0.136 | 0.292 | 0.299 | 1 | 1 | 0.389 | 19 | 14 | 18 |
| sgd_momentum|fd | 0.0921 | 0.165 | 0.297 | 1 | 1 | 0.65 | 25 | 24 | 19 |
| basin_hopping|fd | 0.251 | 0.281 | 0.296 | 0.324 | 0.144 | 0.385 | 10 | 15 | 20 |
| spsa|none | 0.125 | 0.205 | 0.293 | 1 | 1 | 1.36 | 21 | 21 | 21 |
| ipopt_multistart|fd | 0.039 | 0.26 | 0.289 | 2.62 | 0.919 | 3.16 | 26 | 19 | 22 |
| differential_evolution|none | 0.118 | 0.169 | 0.287 | 0.99 | 0 | 1.54 | 23 | 23 | 23 |
| kiefer_wolfowitz|none | 0.184 | 0.23 | 0.252 | 1 | 0 | 1.21 | 14 | 20 | 24 |
| robbins_monro|model | 0.166 | 0.199 | 0.225 | 1 | 0 | 11.8 | 16 | 22 | 25 |
| robbins_monro|fd | 0.124 | 0.161 | 0.197 | 1 | 1 | 0.732 | 22 | 25 | 26 |

### on the true function

| opt | success_10 | success_50 | success_500 | used_median | past_cap | wall_s | rank_10 | rank_50 | rank_500 |
|---|---|---|---|---|---|---|---|---|---|
| pso|none | 0.313 | 0.599 | 0.872 | 1 | 0 | 0.157 | 14 | 11 | 1 |
| cmaes|none | 0.3 | 0.627 | 0.815 | 0.516 | 0.0436 | 0.461 | 15 | 9 | 2 |
| ipopt_multistart|model | 0.586 | 0.662 | 0.747 | 0.261 | 0.104 | 0.407 | 6 | 6 | 3 |
| cobyla|none | 0.602 | 0.722 | 0.747 | 0.228 | 0 | 1.8 | 4 | 1 | 3 |
| differential_evolution|none | 0.18 | 0.33 | 0.728 | 0.99 | 0 | 0.479 | 21 | 22 | 5 |
| lbfgsb|model | 0.651 | 0.719 | 0.725 | 0.0431 | 0 | 0.0656 | 1 | 2 | 6 |
| trust_constr|model | 0.594 | 0.676 | 0.725 | 0.147 | 0 | 1.7 | 5 | 4 | 6 |
| slsqp|model | 0.646 | 0.714 | 0.719 | 0.0656 | 0 | 0.0573 | 2 | 3 | 8 |
| lbfgsb|fd | 0.578 | 0.662 | 0.719 | 0.275 | 0.0409 | 0.084 | 8 | 6 | 8 |
| nelder_mead|none | 0.243 | 0.559 | 0.719 | 0.62 | 0 | 0.205 | 19 | 14 | 8 |
| slsqp|fd | 0.583 | 0.659 | 0.708 | 0.202 | 0.00545 | 0.0676 | 7 | 8 | 11 |
| direct|none | 0.253 | 0.512 | 0.7 | 0.262 | 0.305 | 0.0537 | 17 | 18 | 12 |
| basin_hopping|model | 0.61 | 0.67 | 0.698 | 0.0618 | 0 | 0.0959 | 3 | 5 | 13 |
| basin_hopping|fd | 0.55 | 0.61 | 0.695 | 0.381 | 0.0763 | 0.116 | 9 | 10 | 14 |
| trust_constr|fd | 0.39 | 0.583 | 0.692 | 0.993 | 0.496 | 1.33 | 11 | 12 | 15 |
| ipopt_multistart|fd | 0.0463 | 0.556 | 0.651 | 1.36 | 0.935 | 0.905 | 26 | 15 | 16 |
| adam|model | 0.373 | 0.583 | 0.597 | 1 | 0 | 1.31 | 12 | 12 | 17 |
| adam|fd | 0.128 | 0.226 | 0.583 | 1 | 1 | 0.179 | 24 | 25 | 18 |
| sgd_momentum|model | 0.368 | 0.534 | 0.583 | 1 | 0 | 1.3 | 13 | 17 | 18 |
| bb_gd|model | 0.529 | 0.548 | 0.548 | 1 | 0 | 1.33 | 10 | 16 | 20 |
| bb_gd|fd | 0.185 | 0.507 | 0.542 | 1 | 1 | 0.14 | 20 | 19 | 21 |
| sgd_momentum|fd | 0.114 | 0.275 | 0.542 | 1 | 1 | 0.178 | 25 | 24 | 21 |
| spsa|none | 0.166 | 0.332 | 0.531 | 1 | 1 | 0.302 | 22 | 21 | 23 |
| kiefer_wolfowitz|none | 0.289 | 0.401 | 0.45 | 1 | 0 | 0.205 | 16 | 20 | 24 |
| robbins_monro|model | 0.245 | 0.33 | 0.39 | 1 | 0 | 1.25 | 18 | 22 | 25 |
| robbins_monro|fd | 0.155 | 0.221 | 0.324 | 1 | 1 | 0.177 | 23 | 26 | 26 |

## 7. H7 cost crossover: no E9b records yet
