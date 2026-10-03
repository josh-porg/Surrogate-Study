# NSF ACCESS Explore allocation — draft request

Draft for the author to submit at https://allocations.access-ci.org (Explore ACCESS). Explore requests need a short
summary and a CV; a graduate-student PI also uploads a signed advisor letter on letterhead. Check every field before
submitting — Claude has not submitted anything.

## Title
Accuracy Is Not the Objective: A Global Benchmark of Surrogate-Model Pipelines for Expensive Engineering Analyses

## Public overview (one paragraph)
Surrogate models replace expensive simulations, such as finite-element and CFD analyses, inside design optimization
loops. Engineering practice favours Gaussian-process surrogates, while tabular machine learning favours tree
ensembles, and recent evidence shows that the most accurate surrogate is not always the one that leads to the best
design. This project runs a controlled factorial study of complete surrogate pipelines — sampling, uncertainty-driven
adaptive sampling, the surrogate model itself (about fifty methods, from 1950s response surfaces and 1960s
polynomial networks to modern kriging, gradient boosting, neural ensembles and tabular foundation models),
multi-fidelity fusion, conversion to optimizer-ready splines, and eleven optimizers — on more than 150 analytic test
problems with known optima, varying dimension, sample size, noise type and fidelity structure. Pipelines are scored by
the true regret of the optimizer's answer at matched simulation cost. The work is embarrassingly parallel (hundreds of
thousands of independent single-core model fits) and is packaged as a resumable job-array workflow in Python.

## Resources requested
- **Purdue Anvil CPU** (or another ACCESS CPU resource with Python 3.11+): request **60 000 ACCESS credits**
  (≈ 50 000 core-hours). Basis: measured per-fit costs from the project's timing experiment (E0, results in
  `results/raw/E0_timing/`) × the reduced experimental design in `docs/STUDY_PLAN.md` §4b, with a 2× margin for
  re-runs. Update this number from the E0 measurements before submitting.
- Storage: < 50 GB (results are compact JSONL / Parquet).
- No GPU required (an optional small GPU share for neural-network learners could be added later).

## Software
Python 3.11–3.13, NumPy, SciPy, scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch (CPU), GPyTorch, SMT, CasADi
(IPOPT), CMA; all pip-installable into a user virtual environment. Job arrays via SLURM using the project's
`--shard i/k` runner option.

## People
- PI: Joshua D. Poznański (graduate student, University of Kansas, Aerospace Engineering) — confirm current status.
- Advisor letter: Dr Ronald M. Barrett-Gonzalez (thesis advisor) — signed letter on KU letterhead stating awareness
  of and engagement in guiding the computational work.

## Advisor letter — template (for the advisor to adapt and sign)
> To the ACCESS Allocations Team: I am aware of and support the computational work proposed by my graduate student,
> Joshua D. Poznański, in the Explore request "Accuracy Is Not the Objective: A Global Benchmark of Surrogate-Model
> Pipelines for Expensive Engineering Analyses". The project extends the surrogate-modelling study of his M.S.
> thesis (University of Kansas, 2026), and I will guide the computational activity. — [Name, title, signature, date]
