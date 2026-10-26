# E12 — rational surrogates (80640 task records, 69049 ok fits, 252 cells)

× best = geometric-mean NRMSE over the per-cell best config (1 = best everywhere); each cell capped at 1000×.

## Each config, × best by problem group and noise level

| cfg | rational 0 | rational 0.001 | rational 0.01 | rational 0.1 | control 0 | control 0.001 | control 0.01 | control 0.1 | cells |
|---|---|---|---|---|---|---|---|---|---|
| rat_ls | 9.19 | 4.64 | 5.54 | 5.22 | 18.36 | 8.83 | 9.14 | 18.78 | 252 |
| aaa_rational | 18.26 | 102.80 | 106.00 | 164.19 | nan | nan | nan | nan | 60 |
| paaa | 26.67 | 126.93 | 149.56 | 186.60 | 148.60 | 384.96 | 431.48 | 176.00 | 252 |
| rat_ls[reg=pos] | 81.01 | 28.28 | 13.46 | 4.88 | 27.03 | 12.03 | 7.76 | 3.87 | 252 |
| aaa_rational[profile=aaa] | 117.92 | 11.61 | 11.93 | 12.33 | nan | nan | nan | nan | 60 |
| aaa_rational[profile=fh] | 130.50 | 6.38 | 4.03 | 3.17 | nan | nan | nan | nan | 60 |
| aaa_rational[profile=caaa] | 182.59 | 10.94 | 3.31 | 1.51 | nan | nan | nan | nan | 60 |
| gp | 182.69 | 10.77 | 4.02 | 2.06 | 12.11 | 2.83 | 2.14 | 1.75 | 252 |
| aaa_rational[profile=cleanup] | 203.65 | 10.28 | 10.05 | 14.87 | nan | nan | nan | nan | 60 |
| as_gp | 208.68 | 19.58 | 5.34 | 1.77 | 18.79 | 5.69 | 3.48 | 2.01 | 252 |
| ppr | 237.05 | 42.69 | 8.02 | 1.96 | 37.89 | 19.66 | 8.04 | 2.38 | 251 |
| gp_ratmean[profile=fh] | 243.45 | 23.14 | 8.13 | 3.46 | 69.48 | 38.28 | 16.51 | 5.62 | 252 |
| aaa_rational[profile=svd] | 252.84 | 43.84 | 95.52 | 130.85 | nan | nan | nan | nan | 60 |
| gam | 254.65 | 75.71 | 15.93 | 3.34 | 51.64 | 35.48 | 15.42 | 4.59 | 252 |
| additive_aaa[profile=aaa] | 259.75 | 59.71 | 15.16 | 3.51 | 55.19 | 34.94 | 15.30 | 4.59 | 252 |
| additive_aaa[profile=fh] | 263.91 | 69.36 | 16.98 | 5.42 | 56.46 | 40.36 | 18.80 | 5.98 | 252 |
| mlp_ens[opt=lbfgs] | 265.29 | 22.91 | 5.13 | 2.50 | 15.76 | 4.48 | 2.47 | 1.97 | 250 |
| gp_ratmean[profile=caaa] | 268.33 | 25.11 | 11.67 | 3.08 | 29.06 | 8.43 | 5.53 | 3.10 | 252 |
| additive_aaa[profile=cleanup] | 270.35 | 61.72 | 18.42 | 3.58 | 54.24 | 35.09 | 14.94 | 4.53 | 252 |
| catboost | 273.85 | 84.99 | 16.29 | 3.29 | 52.10 | 23.56 | 9.36 | 2.56 | 252 |
| additive_aaa[profile=svd] | 275.35 | 72.69 | 17.17 | 3.38 | 57.57 | 39.77 | 15.50 | 4.58 | 252 |
| additive_aaa[profile=caaa] | 281.04 | 90.04 | 19.75 | 3.89 | 62.08 | 35.91 | 17.23 | 4.72 | 252 |
| ridge_aaa[profile=fh] | 285.98 | 66.01 | 14.64 | 3.94 | 98.20 | 65.71 | 28.80 | 8.69 | 252 |
| xgb | 287.82 | 100.54 | 19.59 | 3.72 | 68.44 | 36.52 | 14.42 | 3.54 | 252 |
| gp_ratmean[profile=aaa] | 291.45 | 17.36 | 6.05 | 2.50 | 30.64 | 7.24 | 4.18 | 2.57 | 252 |
| pce_lars | 301.88 | 108.95 | 21.29 | 3.91 | 61.80 | 18.57 | 9.14 | 2.93 | 252 |
| rat_ls[reg=tik] | 308.62 | 3.22 | 3.37 | 4.78 | 21.04 | 7.48 | 6.85 | 4.50 | 252 |
| ridge_aaa[profile=caaa] | 309.60 | 66.21 | 19.97 | 3.41 | 60.42 | 40.92 | 19.60 | 5.34 | 252 |
| gp_ratmean[profile=cleanup] | 342.05 | 28.58 | 14.69 | 7.84 | 72.82 | 35.22 | 24.86 | 16.35 | 252 |
| ridge_aaa[profile=aaa] | 417.00 | 92.29 | 23.44 | 5.47 | 104.66 | 54.14 | 32.30 | 9.43 | 252 |
| ridge_aaa[profile=cleanup] | 458.36 | 96.73 | 31.65 | 11.11 | 123.23 | 83.95 | 55.46 | 25.83 | 252 |
| ridge_aaa[profile=svd] | 612.78 | 271.64 | 146.41 | 67.10 | 295.73 | 314.94 | 230.73 | 136.49 | 252 |

## Wins (cells where the config is best)

|  | cells |
|---|---|
| rat_ls | 53 |
| gp | 31 |
| rat_ls[reg=tik] | 28 |
| as_gp | 26 |
| rat_ls[reg=pos] | 19 |
| mlp_ens[opt=lbfgs] | 11 |
| catboost | 11 |
| additive_aaa[profile=aaa] | 10 |
| paaa | 10 |
| ppr | 7 |
| gp_ratmean[profile=aaa] | 6 |
| aaa_rational | 5 |
| aaa_rational[profile=caaa] | 4 |
| aaa_rational[profile=cleanup] | 4 |
| additive_aaa[profile=fh] | 4 |
| gam | 4 |
| pce_lars | 3 |
| additive_aaa[profile=cleanup] | 3 |
| ridge_aaa[profile=svd] | 2 |
| gp_ratmean[profile=cleanup] | 2 |
| additive_aaa[profile=caaa] | 2 |
| gp_ratmean[profile=caaa] | 2 |
| aaa_rational[profile=fh] | 2 |
| aaa_rational[profile=aaa] | 1 |
| xgb | 1 |
| additive_aaa[profile=svd] | 1 |

## Best rational config vs best reference learner, per cell (< 1: rational more accurate)

|  | rational best wins (share) | best rational / best reference (geo) | cells |
|---|---|---|---|
| ('control', 0.0) | 0.52 | 0.05 | 21 |
| ('control', 0.001) | 0.38 | 1.21 | 21 |
| ('control', 0.01) | 0.19 | 1.28 | 21 |
| ('control', 0.1) | 0.05 | 1.25 | 21 |
| ('rational structure', 0.0) | 0.88 | 0.00 | 42 |
| ('rational structure', 0.001) | 0.81 | 0.16 | 42 |
| ('rational structure', 0.01) | 0.83 | 0.43 | 42 |
| ('rational structure', 0.1) | 0.67 | 0.85 | 42 |
