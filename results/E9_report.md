# E9 — optimization on the surrogate (2 tasks, 1 problems)

Regret normalised by std(f); success = regret < 0.01. Instances with an unconfirmed f* are excluded.

## 1. Surrogates at the full budget (500d)

| surr | success_avg | geo_regret_avg | best optimizer | success_best | median NRMSE |
|---|---|---|---|---|---|
| truth | 0.808 | 7.75e-07 | adam|fd | 1 | 0 |

## 3. Surrogate gradient vs finite differences (< 1: the surrogate's gradient gives lower regret)

| method | model / fd regret (geo) | model better (share) | pairs |
|---|---|---|---|
| adam | 1.00 | 0.00 | 2 |
| basin_hopping | 1.00 | 0.00 | 2 |
| bb_gd | 1.00 | 0.00 | 2 |
| ipopt_multistart | 1.00 | 0.00 | 2 |
| lbfgsb | 1.00 | 0.00 | 2 |
| robbins_monro | 0.00 | 1.00 | 2 |
| sgd_momentum | 0.00 | 0.50 | 2 |
| slsqp | 1.00 | 0.00 | 2 |
| trust_constr | 1.00 | 0.50 | 2 |

By surrogate:

| surr | model / fd (geo) |
|---|---|
| truth | 0.14 |

## 4. Optimizing the surrogate vs taking the best training point

Best optimizer on the surrogate beats the best training point in nan% of tasks; geometric-mean regret nan vs nan.

## 5. Success rate vs evaluations (×d), averaged over optimizers

| surr | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| truth | 0.40 | 0.54 | 0.60 | 0.73 | 0.81 | 0.81 |

## 6. Best surrogate + optimizer per problem (0 distinct pairs over 0 problems)

| problem | best pair (geo-mean regret) |
|---|---|

## 7. H7 cost crossover: no E9b records yet
