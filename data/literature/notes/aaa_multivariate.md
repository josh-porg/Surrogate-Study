# Multivariate AAA / rational surrogates — novelty check (2026-10-11)

Searches (web, extended): AAA × ridge/active subspace; low-rank/tensor/TT × AAA; p-AAA scattered data; additive /
backfitting × AAA; Smolyak × rational; GP with rational mean; AAA in surrogate-model benchmarks. Plus a full read of
the 2025 AAA review (Nakatsukasa, Sète & Trefethen, arXiv 2510.16237), whose §15 states that "no true multivariate AAA
algorithm has yet been discovered" and lists p-AAA (tensor grid), slice-wise univariate AAA (Trefethen 2023) and
Austin et al. (2021) as the multivariate options.

## Established (implemented as baselines, `learners/rational.py`)
| Method | Source | Data | Notes |
|---|---|---|---|
| p-AAA | Carracedo Rodriguez, Balicki & Gugercin, SIAM J. Sci. Comput. 2023 (arXiv 2003.06536) | tensor grid | parametric dynamical systems |
| p-AAA, scattered data | Balicki & Gugercin, arXiv 2510.22861 (Oct 2025), code on Zenodo 17885790 | scattered | implemented as `paaa` (d-variate generalisation of their 2-D formulas) |
| low-rank p-AAA | Balicki & Gugercin, arXiv 2502.03204 (2025) | full tensor grid | low-rank tensor barycentric coefficients; not implemented (needs grid data; E12 grid arm if added) |
| LS multivariate rational p/q | Austin et al. 2021 (Apprentice) | scattered | implemented as `rat_ls` (+ Sanathanan–Koerner) |
| rational approx. with curves of singularities | Boullé, Herremans & Huybrechs, arXiv 2312.13202 (2024) | — | cited; specialised to known singular curves |

## Not found in the literature (E12 contributions, to re-check before submission)
- **AAA along an active-subspace / ridge direction** (`ridge_aaa`). Ridge/active-subspace surrogates use polynomials or
  GPs as the profile (Constantine; Hokanson & Constantine); no rational/AAA profile found.
- **Additive model with AAA terms** (`additive_aaa`). Backfitting literature uses splines / local polynomials.
- **GP with a rational (AAA ridge) mean function** (`gp_ratmean`). GP trends in the surrogate literature are constant,
  linear or polynomial.
- **Sparse-grid (Smolyak) rational interpolation** — no source found (planned, needs Smolyak designs).
- **A benchmark of rational surrogates against GP / trees / networks** in the surrogate-modelling sense, including a
  noise sweep — none found. The AAA literature is about approximation of exact functions.

## Smoke-test findings worth reporting (E12 design rationale)
- `rat_ls` and `paaa` reach machine precision on pole/knee functions (pole_knee d2/d5) where the GP is at 1e-2..1e-3.
- Plain AAA on noisy 1-D data (partial residuals at ≈ 4 % noise) places real poles inside the interval at every degree
  ≥ 4 (Froissart doublets): a pure-AAA additive model collapses. Binning + pole/spike cleanup + spline start fixes it
  (hybrid). Noise robustness is therefore E12's main factor.
- The search was by one session on 2026-10-11; re-run (and check Google Scholar / MathSciNet) before claiming novelty
  in the paper.
