# E9 — optimization on the surrogate (11936 tasks, 38 problems)

Regret normalised by std(f); success = regret < 0.01. Instances with an unconfirmed f* are excluded.

## 1. Surrogates at the full budget (500d)

| surr | success_avg | geo_regret_avg | best optimizer | success_best | median NRMSE |
|---|---|---|---|---|---|
| truth | 0.808 | 7.75e-07 | trust_constr|model | 1 | 0 |
| smt_krg | 0.535 | 0.00135 | ipopt_multistart|model | 0.672 | 0.0233 |
| smt_kpls | 0.457 | 0.00645 | ipopt_multistart|model | 0.539 | 0.0994 |
| pce_lars | 0.411 | 0.00472 | ipopt_multistart|model | 0.532 | 0.227 |
| ens_press | 0.456 | 0.0092 | lbfgsb|model | 0.514 | 0.189 |
| hetgp | 0.453 | 0.0071 | cmaes|none | 0.513 | 0.0545 |
| rbf_tps | 0.343 | 0.0133 | ipopt_multistart|model | 0.513 | 0.199 |
| gp | 0.443 | 0.00872 | cmaes|none | 0.499 | 0.0633 |
| mlp_ens[opt=lbfgs] | 0.37 | 0.0191 | cmaes|none | 0.462 | 0.228 |
| as_gp | 0.362 | 0.0216 | ipopt_multistart|model | 0.454 | 0.181 |
| rsm2 | 0.401 | 0.00669 | slsqp|model | 0.447 | 0.371 |
| krr | 0.33 | 0.0231 | ipopt_multistart|model | 0.442 | 0.162 |
| rf | 0.179 | 0.149 | nelder_mead|none | 0.312 | 0.307 |
| lgbm | 0.106 | 0.344 | pso|none | 0.232 | 0.243 |
| mars | 0.189 | 0.0739 | cobyla|none | 0.229 | 0.259 |
| catboost | 0.114 | 0.344 | cobyla|none | 0.203 | 0.177 |
| idw | 0.0891 | 0.44 | cobyla|none | 0.136 | 0.846 |

## 2. H6: does surrogate accuracy predict optimization regret? (Spearman ρ of NRMSE vs regret rankings across surrogates, per problem × instance × n × noise × optimizer)

| cls | count | mean | 25% | 50% | 75% |
|---|---|---|---|---|---|
| first-order SA | 7984.00 | 0.45 | 0.20 | 0.50 | 0.76 |
| global | 3190.00 | 0.49 | 0.28 | 0.55 | 0.77 |
| local derivative-free | 1596.00 | 0.46 | 0.22 | 0.51 | 0.77 |
| local gradient | 6680.00 | 0.39 | 0.12 | 0.43 | 0.69 |

Overall median ρ = 0.49; share of settings with ρ > 0.5: 49%.

## 3. Surrogate gradient vs finite differences (< 1: the surrogate's gradient gives lower regret)

| method | model / fd regret (geo) | model better (share) | pairs |
|---|---|---|---|
| adam | 0.96 | 0.27 | 11614 |
| basin_hopping | 0.60 | 0.35 | 11614 |
| bb_gd | 0.99 | 0.26 | 11614 |
| ipopt_multistart | 0.56 | 0.40 | 1470 |
| lbfgsb | 0.53 | 0.33 | 11614 |
| robbins_monro | 0.70 | 0.40 | 11614 |
| sgd_momentum | 0.91 | 0.28 | 11614 |
| slsqp | 0.56 | 0.34 | 11614 |
| trust_constr | 0.55 | 0.45 | 11614 |

By surrogate:

| surr | model / fd (geo) |
|---|---|
| as_gp | 0.81 |
| catboost | 0.99 |
| ens_press | 0.90 |
| gp | 0.86 |
| hetgp | 0.82 |
| idw | 0.98 |
| krr | 0.94 |
| lgbm | 0.99 |
| mars | 0.96 |
| mlp_ens[opt=lbfgs] | 0.20 |
| pce_lars | 0.92 |
| rbf_tps | 0.93 |
| rf | 0.99 |
| rsm2 | 0.93 |
| smt_kpls | 0.34 |
| smt_krg | 0.26 |
| truth | 0.14 |

## 4. Optimizing the surrogate vs taking the best training point

Best optimizer on the surrogate beats the best training point in 72% of tasks; geometric-mean regret 0.00269 vs 0.108.

## 5. Success rate vs evaluations (×d), averaged over optimizers

| surr | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| pce_lars | 0.25 | 0.30 | 0.34 | 0.37 | 0.39 | 0.40 |
| smt_kpls | 0.29 | 0.34 | 0.39 | 0.42 | 0.44 | 0.45 |
| smt_krg | 0.32 | 0.39 | 0.45 | 0.49 | 0.51 | 0.53 |
| truth | 0.40 | 0.54 | 0.60 | 0.73 | 0.81 | 0.81 |

## 6. Best surrogate + optimizer per problem (32 distinct pairs over 38 problems)

| problem | best pair (geo-mean regret) |
|---|---|
| ackley_d10 | pce_lars + robbins_monro|fd |
| ackley_d5 | krr + differential_evolution|none |
| axis_step_d5 | hetgp + pso|none |
| booth | smt_kpls + differential_evolution|none |
| borehole | rbf_tps + basin_hopping|fd |
| branin | smt_kpls + trust_constr|model |
| camel6 | pce_lars + slsqp|model |
| composite_d5 | smt_krg + nelder_mead|none |
| dixonpr_d5 | rsm2 + robbins_monro|fd |
| goldpr | smt_kpls + differential_evolution|none |
| gp_draw_d2 | lgbm + cmaes|none |
| griewank_d5 | rsm2 + nelder_mead|none |
| kink_floor_d5 | as_gp + adam|fd |
| levy_d10 | krr + kiefer_wolfowitz|none |
| levy_d5 | krr + pso|none |
| michal_d5 | catboost + differential_evolution|none |
| monotone_d5 | as_gp + adam|fd |
| oblique_step_d10 | mlp_ens[opt=lbfgs] + pso|none |
| piston | as_gp + cobyla|none |
| plateau_bump_d5 | smt_krg + differential_evolution|none |
| pole_knee_d5 | smt_krg + lbfgsb|model |
| quadratic_d5 | rsm2 + slsqp|fd |
| rastr | smt_kpls + spsa|none |
| rastr_d5 | pce_lars + direct|none |
| regime_d5 | as_gp + pso|none |
| rosen_d5 | smt_krg + lbfgsb|model |
| rotated_ridge_d5 | as_gp + trust_constr|fd |
| rothyp_d10 | pce_lars + ipopt_multistart|model |
| schaffer2 | idw + cmaes|none |
| smooth_trig_d10 | krr + kiefer_wolfowitz|none |
| smooth_trig_d5 | smt_krg + pso|none |
| stybtang | pce_lars + differential_evolution|none |
| sumsqu_d5 | smt_krg + spsa|none |
| thesis_collapse_knee | gp + ipopt_multistart|model |
| tree_draw_d5 | catboost + direct|none |
| trid_d5 | smt_krg + lbfgsb|model |
| wingweight | hetgp + lbfgsb|model |
| zakharov_d5 | as_gp + trust_constr|fd |

## 7. H7 cost crossover: no E9b records yet
