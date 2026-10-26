# E12 — rational surrogates (5275 task records, 5144 ok fits, 252 cells)

× best = geometric-mean NRMSE over the per-cell best config (1 = best everywhere).

## Each config, × best by problem group and noise level

| cfg | rational 0 | rational 0.001 | rational 0.01 | rational 0.1 | control 0 | control 0.001 | control 0.01 | control 0.1 | cells |
|---|---|---|---|---|---|---|---|---|---|
| rat_ls | 1.00 | nan | nan | nan | nan | nan | nan | nan | 1 |
| mlp_ens[opt=lbfgs] | 2.76 | 1.42 | 1.42 | 1.69 | 1.03 | 1.05 | 1.04 | 1.12 | 250 |
| ppr | 4.94 | 2.46 | 2.30 | 1.32 | 4.37 | 4.80 | 3.38 | 1.35 | 251 |
| paaa | 7.19 | nan | nan | nan | nan | nan | nan | nan | 1 |
| rat_ls[reg=pos] | 13.51 | nan | nan | nan | nan | nan | nan | nan | 1 |
| rat_ls[reg=tik] | 2555.71 | nan | nan | nan | nan | nan | nan | nan | 1 |
| gp_ratmean[profile=aaa] | 37240956830666.21 | nan | nan | nan | nan | nan | nan | nan | 1 |
| ridge_aaa[profile=aaa] | 57953965555978.49 | nan | nan | nan | nan | nan | nan | nan | 1 |
| ridge_aaa[profile=cleanup] | 61916759367819.73 | nan | nan | nan | nan | nan | nan | nan | 1 |
| gp_ratmean[profile=cleanup] | 70047551411534.95 | nan | nan | nan | nan | nan | nan | nan | 1 |
| gp_ratmean[profile=fh] | 76466700703804.89 | nan | nan | nan | nan | nan | nan | nan | 1 |
| ridge_aaa[profile=fh] | 80266608694717.53 | nan | nan | nan | nan | nan | nan | nan | 1 |
| gp_ratmean[profile=caaa] | 147410247000232.53 | nan | nan | nan | nan | nan | nan | nan | 1 |
| ridge_aaa[profile=caaa] | 194288735037224.16 | nan | nan | nan | nan | nan | nan | nan | 1 |
| additive_aaa[profile=cleanup] | 325707799945430.25 | nan | nan | nan | nan | nan | nan | nan | 1 |
| additive_aaa[profile=svd] | 325799358070912.38 | nan | nan | nan | nan | nan | nan | nan | 1 |
| additive_aaa[profile=aaa] | 325820121628475.25 | nan | nan | nan | nan | nan | nan | nan | 1 |
| additive_aaa[profile=caaa] | 325991706140239.38 | nan | nan | nan | nan | nan | nan | nan | 1 |
| additive_aaa[profile=fh] | 355241776023413.06 | nan | nan | nan | nan | nan | nan | nan | 1 |
| ridge_aaa[profile=svd] | 385582778873698.69 | nan | nan | nan | nan | nan | nan | nan | 1 |

## Wins (cells where the config is best)

|  | cells |
|---|---|
| mlp_ens[opt=lbfgs] | 174 |
| ppr | 77 |
| rat_ls | 1 |

## Best rational config vs best reference learner, per cell (< 1: rational more accurate)

|  | rational best wins (share) | best rational / best reference (geo) | cells |
|---|---|---|---|
| ('control', 0.0) | 0.00 | nan | 21 |
| ('control', 0.001) | 0.00 | nan | 21 |
| ('control', 0.01) | 0.00 | nan | 21 |
| ('control', 0.1) | 0.00 | nan | 21 |
| ('rational structure', 0.0) | 0.02 | 0.00 | 42 |
| ('rational structure', 0.001) | 0.00 | nan | 42 |
| ('rational structure', 0.01) | 0.00 | nan | 42 |
| ('rational structure', 0.1) | 0.00 | nan | 42 |
