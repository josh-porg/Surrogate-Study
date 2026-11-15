# E2b — survey-extension learners on the E2a screening grid

Tasks: {('E2a_learners', 'ok'): 32300, ('E2a_learners', 'infeasible'): 60, ('E2a_learners', 'timeout'): 40, ('E2b_learners', 'ok'): 13539, ('E2b_learners', 'timeout'): 21, ('E2b_learners', 'infeasible'): 840}. Ranks are among all E2a and E2b configs in each problem × n × noise cell (mean NRMSE over ≥ 3 seeds). Screening only, never used for inference.

## 1. Each E2b config

| cfg | cells | mean_rank | of_configs | cells_won | ratio_to_best_E2a | beats_best_E2a | failed_share |
|---|---|---|---|---|---|---|---|
| warped_gp | 72 | 14.4 | 63.7 | 16 | 5.52 | 0.111 | 0.00139 |
| sparse_gp | 72 | 17.1 | 63.7 | 3 | 7.55 | 0 | 0 |
| kernel_flow_gp | 72 | 18.4 | 63.7 | 6 | 8.03 | 0 | 0.00417 |
| deep_gp | 72 | 19.1 | 63.7 | 10 | 7.73 | 0.0972 | 0.00278 |
| bnn_laplace | 72 | 21.2 | 63.7 | 6 | 6.32 | 0.0556 | 0 |
| bnn_vi[kl_weight=0.1] | 72 | 24.4 | 63.7 | 0 | 9.55 | 0 | 0 |
| bnn_vi[kl_weight=0.01] | 72 | 26.2 | 63.7 | 1 | 8.71 | 0 | 0.00833 |
| sindy_ae_static | 71 | 29.6 | 63.7 | 0 | 11 | 0 | 0.0125 |
| bart | 72 | 33.1 | 63.7 | 7 | 15.9 | 0.0278 | 0 |
| natural_neighbor | 30 | 33.1 | 63.7 | 2 | 66.6 | 0 | 0.583 |
| pspline_te | 30 | 34.3 | 63.7 | 3 | 65.5 | 0.0667 | 0.583 |
| rat_ls[reg=tik] | 72 | 40.1 | 63.7 | 3 | 21.6 | 0.0278 | 0 |
| moe | 72 | 40.3 | 63.7 | 1 | 16.7 | 0 | 0 |
| ft_als | 72 | 42.1 | 63.7 | 0 | 18.2 | 0 | 0 |
| gam | 72 | 44.1 | 63.7 | 0 | 21.6 | 0.0139 | 0 |
| lowrank_cp | 72 | 44.9 | 63.7 | 0 | 21.6 | 0 | 0 |
| rat_ls | 72 | 48 | 63.7 | 3 | 46.3 | 0.0278 | 0 |
| bnn_vi | 72 | 49.5 | 63.7 | 1 | 24.6 | 0 | 0 |
| symreg | 72 | 51.8 | 63.7 | 0 | 31.5 | 0 | 0 |
| paaa | 72 | 62.8 | 63.7 | 0 | 752 | 0 | 0 |

## 2. Best E2b config per problem (ratio to the best E2a config, averaged over that problem's cells)

| problem | best E2b config | ratio |
|---|---|---|
| axis_step_d5 | warped_gp | 1.22 |
| borehole | rat_ls[reg=tik] | 1.82 |
| composite_d5 | deep_gp | 0.956 |
| fourier_sparse_d2 | kernel_flow_gp | 2.95e+05 |
| gp_draw_d5 | kernel_flow_gp | 1.14 |
| kink_floor_d2 | bnn_laplace | 1.2 |
| nonstationary_d2 | warped_gp | 1.39 |
| plateau_bump_d2 | warped_gp | 1.08 |
| rotated_ridge_d5 | bnn_laplace | 1.64 |
| smooth_trig_d5 | rat_ls[reg=tik] | 4.02 |
| thesis_collapse_knee | warped_gp | 1.68 |
| tree_draw_d5 | bart | 1 |
