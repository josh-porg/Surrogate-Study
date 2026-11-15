# all — optimization on the surrogate (5787 tasks, 38 problems; records: E9_optim, E9_ipopt, E9x_optim)

Regret normalised by std(f); success = regret < 0.01. Instances with an unconfirmed f* are excluded.

## 1. Surrogates at the full budget (500d)

| surr | success_avg | geo_regret_avg | best optimizer | success_best | median NRMSE |
|---|---|---|---|---|---|
| truth | 0.641 | 4.56e-05 | pso|none | 0.886 | 0 |
| ens_press | 0.459 | 0.00939 | lbfgsb|fd | 0.522 | 0.174 |
| smt_krg | 0.411 | 0.00563 | slsqp|model | 0.486 | 0.117 |
| mlp_ens[opt=lbfgs] | 0.368 | 0.0211 | cmaes|none | 0.482 | 0.206 |
| hetgp | 0.401 | 0.0107 | pso|none | 0.467 | 0.128 |
| gp | 0.398 | 0.0119 | cmaes|none | 0.457 | 0.13 |
| smt_kpls | 0.375 | 0.00837 | pso|none | 0.445 | 0.17 |
| pce_lars | 0.364 | 0.00797 | trust_constr|fd | 0.405 | 0.331 |
| rsm2 | 0.365 | 0.00924 | lbfgsb|fd | 0.402 | 0.425 |
| as_gp | 0.321 | 0.0276 | pso|none | 0.37 | 0.207 |
| rbf_tps | 0.308 | 0.0206 | ipopt_multistart|fd | 0.355 | 0.279 |
| krr | 0.297 | 0.038 | trust_constr|model | 0.337 | 0.278 |
| rf | 0.177 | 0.118 | differential_evolution|none | 0.307 | 0.452 |
| mars | 0.181 | 0.108 | slsqp|fd | 0.217 | 0.322 |
| catboost | 0.119 | 0.326 | cmaes|none | 0.196 | 0.34 |
| lgbm | 0.0943 | 0.367 | pso|none | 0.185 | 0.342 |
| idw | 0.0938 | 0.414 | basin_hopping|fd | 0.147 | 0.831 |

## 2. H6: does surrogate accuracy predict optimization regret? (Spearman ρ of NRMSE vs regret rankings across surrogates, per problem × instance × n × noise × optimizer)

| cls | count | mean | 25% | 50% | 75% |
|---|---|---|---|---|---|
| first-order SA | 3622.00 | 0.45 | 0.19 | 0.50 | 0.75 |
| global | 1432.00 | 0.47 | 0.22 | 0.53 | 0.76 |
| local derivative-free | 721.00 | 0.45 | 0.21 | 0.47 | 0.75 |
| local gradient | 3309.00 | 0.41 | 0.15 | 0.46 | 0.70 |

Overall median ρ = 0.48; share of settings with ρ > 0.5: 48%.

## 3. Surrogate gradient vs finite differences (< 1: the surrogate's gradient gives lower regret)

| method | model / fd regret (geo) | model better (share) | pairs |
|---|---|---|---|
| adam | 0.95 | 0.26 | 5635 |
| basin_hopping | 0.65 | 0.32 | 5635 |
| bb_gd | 0.95 | 0.25 | 5635 |
| ipopt_multistart | 0.48 | 0.43 | 3063 |
| lbfgsb | 0.62 | 0.30 | 5635 |
| robbins_monro | 0.76 | 0.38 | 5635 |
| sgd_momentum | 0.87 | 0.27 | 5635 |
| slsqp | 0.63 | 0.31 | 5635 |
| trust_constr | 0.59 | 0.41 | 5635 |

By surrogate:

| surr | model / fd (geo) |
|---|---|
| as_gp | 0.83 |
| catboost | 0.98 |
| ens_press | 0.86 |
| gp | 0.86 |
| hetgp | 0.85 |
| idw | 0.96 |
| krr | 0.92 |
| lgbm | 0.99 |
| mars | 0.97 |
| mlp_ens[opt=lbfgs] | 0.19 |
| pce_lars | 0.92 |
| rbf_tps | 0.94 |
| rf | 0.99 |
| rsm2 | 0.93 |
| smt_kpls | 0.22 |
| smt_krg | 0.32 |
| truth | 0.58 |

## 4. Optimizing the surrogate vs taking the best training point

Best optimizer on the surrogate beats the best training point in 69% of tasks; geometric-mean regret 0.00411 vs 0.12.

## 5. Success rate vs evaluations (×d), averaged over optimizers

| surr | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| ens_press | 0.29 | 0.34 | 0.39 | 0.42 | 0.44 | 0.46 |
| mlp_ens[opt=lbfgs] | 0.22 | 0.27 | 0.31 | 0.34 | 0.36 | 0.37 |
| smt_krg | 0.26 | 0.30 | 0.35 | 0.38 | 0.40 | 0.41 |
| truth | 0.36 | 0.44 | 0.52 | 0.58 | 0.61 | 0.64 |

## 6. Best surrogate + optimizer per problem (34 distinct pairs over 38 problems)

| problem | best pair (geo-mean regret) |
|---|---|
| ackley_d10 | pce_lars + robbins_monro|fd |
| ackley_d5 | smt_kpls + direct|none |
| axis_step_d5 | ens_press + lbfgsb|fd |
| booth | rsm2 + adam|fd |
| borehole | mlp_ens[opt=lbfgs] + pso|none |
| branin | smt_kpls + ipopt_multistart|model |
| camel6 | pce_lars + cmaes|none |
| composite_d5 | gp + differential_evolution|none |
| dixonpr_d5 | pce_lars + ipopt_multistart|model |
| goldpr | smt_kpls + pso|none |
| gp_draw_d2 | ens_press + cmaes|none |
| griewank_d5 | pce_lars + nelder_mead|none |
| kink_floor_d5 | as_gp + adam|fd |
| levy_d10 | pce_lars + ipopt_multistart|model |
| levy_d5 | krr + direct|none |
| michal_d5 | catboost + differential_evolution|none |
| monotone_d5 | ens_press + adam|fd |
| oblique_step_d10 | mlp_ens[opt=lbfgs] + lbfgsb|model |
| piston | hetgp + cobyla|none |
| plateau_bump_d5 | idw + spsa|none |
| pole_knee_d5 | smt_krg + pso|none |
| quadratic_d5 | rsm2 + adam|fd |
| rastr | idw + basin_hopping|fd |
| rastr_d5 | ens_press + spsa|none |
| regime_d5 | as_gp + slsqp|model |
| rosen_d5 | smt_krg + slsqp|model |
| rotated_ridge_d5 | as_gp + trust_constr|fd |
| rothyp_d10 | pce_lars + adam|fd |
| schaffer2 | idw + pso|none |
| smooth_trig_d10 | as_gp + slsqp|fd |
| smooth_trig_d5 | smt_kpls + lbfgsb|fd |
| stybtang | smt_kpls + trust_constr|model |
| sumsqu_d5 | pce_lars + adam|fd |
| thesis_collapse_knee | smt_krg + cobyla|none |
| tree_draw_d5 | catboost + cmaes|none |
| trid_d5 | rsm2 + adam|model |
| wingweight | ens_press + basin_hopping|model |
| zakharov_d5 | as_gp + trust_constr|fd |

## 7. H7 cost crossover: no E9b records yet
