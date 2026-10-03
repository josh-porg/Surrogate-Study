# Bug & Methodology Tracker

Defects in the **thesis surrogate study** (code: `COMPASS/compass/surrogate/`, write-up:
`Maters-Project-V3/chapters/Appendix_Spline_Study.tex` + `chap_12_modeling_framework.tex` §"Surrogate Modeling Strategy")
and, later, in this paper's own code.

The thesis repo is **read-only** — nothing here is fixed there. Each item is either fixed in the new study code
(`SurrogatePaper/code/`) or turned into a stated limitation in the paper.

Severity: **S1** could change a headline conclusion · **S2** biases a number or weakens a claim · **S3** cosmetic/documentation.
Status: `open` · `fixed-in-new-code` · `paper-limitation` · `wontfix`.

| ID | Sev | Status | Short title |
|----|-----|--------|-------------|
| B-01 | S1 | open | Canonical CV scored against **noisy** targets → every model floors at the 3 % noise |
| B-02 | S1 | open | Pure max-variance GP scout ≈ a space-filling design; the control is i.i.d. random, not LHS/Sobol |
| B-03 | S1 | open | Scouts only tested with a GP deliverable (cross-combination confound) |
| B-04 | S1 | open | Optimizer performance on the surrogate never measured |
| B-05 | S2 | open | lin/log target picked per cell by **test-fold** error (selection on test data) |
| B-06 | S2 | open | Noise-floor estimator (2nd differences) overstates noise by ≈1.65× plus curvature |
| B-07 | S2 | open | `nn_committee` in code ≠ `nn_committee` in thesis text |
| B-08 | S2 | open | 8 seeds → low statistical power; many "n.s." results are under-powered, not null |
| B-09 | S2 | open | Canonical surfaces: 3-D only, mild features, one noise level, one n |
| B-10 | S2 | open | Upscaling: only one grid size (12³), one interpolant; "B-spline" is `RegularGridInterpolator(cubic)` |
| B-11 | S2 | open | GBT-vs-GP real-cache ranking has no CI/test, yet drives the "GBT is interesting" claim |
| B-12 | S2 | open | Scout GP and deliverable GP use different hyperparameter-restart settings |
| B-13 | S3 | open | Thesis says canonical inputs are "LHS-like"; code draws i.i.d. uniform |
| B-14 | S3 | open | Speed table measured at one training size (n = 200); GP cost scales with n |
| B-15 | S3 | open | Production held-out comparison (Table `real_heldout_final`) is a single run |
| B-16 | S2 | open | Committee scouts: 4 bootstrap members; GBT spread is piecewise-constant → many ties |

---

## B-01 · Canonical CV scored against noisy targets (S1)

**Where:** `COMPASS/compass/surrogate/study.py:117-136` (`_canon_cell`) → `benchmark.py:74-91` (`cv_folds`).
`C.sample(s, n, noise=0.03)` returns noisy `y`; `cv_folds` scores held-out folds against that same noisy `y`.

**Effect:** The expected CV MSE is `E‖f − m‖² + σ²` (see MATHEMATICS §2.3). With σ = 3 % relative noise, no model can
score below ≈3 %. The thesis table shows GP at 3.0–3.2 % on 5 of 6 surfaces — i.e. **GP sits at the noise floor
and the ranking is compressed into the gap above it**. "GP was best everywhere, even where expected to lose"
(Appendix, Tables canonical_a/b) is at least partly this artifact. The scout study (`adaptive.py:137-139`) does score
against clean truth, so it is not affected.

**Fix (new code):** score every canonical experiment against the **noise-free** truth on a large independent test set
(≥ 10⁴ points, or quadrature for integrated error); report noisy-CV only as the "what a practitioner sees" column.

## B-02 · Max-variance scout is close to a space-filling design; control is weak (S1)

**Where:** `adaptive.py:46-53` (GPVarianceScout), `adaptive.py:91-96` (RandomScout).

For a GP with fixed hyperparameters the posterior variance does **not depend on observed y** — only on X. Greedy
max-variance is then a sequential space-filling (maximin-like) design. y-dependence enters only through the
hyperparameter refit each round (ARD length-scales). So "GP scout beats random by 44 %" may largely be
"space-filling beats i.i.d. random", which is well known (Koksma–Hlawka; LHS literature).

**Fix:** add non-adaptive space-filling controls of equal budget: LHS, maximin-LHS, Sobol', Halton, and sequential
maximin (no model). Add a "frozen-hyperparameter GP variance" ablation to separate the space-filling effect from the
y-adaptive effect. Only adaptivity that beats Sobol'/maximin is a real scouting benefit.

## B-03 · Scout × deliverable cross-combinations never tested (S1)

**Where:** `adaptive.py:20-22` — "the deliverable is a GP, the same for every scout".
The user's point: the GBT committee scout was judged on how well its points served a GP, not a GBT. A scout chooses
points that reduce *its own* uncertainty; the natural pairing is scout-family = deliverable-family.

**Fix:** full factorial scout ∈ {…} × deliverable ∈ {…}, reporting the diagonal (native) and off-diagonal (cross)
separately. Also the thesis future-work item (native UQ per family: RF/GBT/NN ensembles, NGBoost, quantile forests,
conformal).

## B-04 · Optimizer performance never measured (S1)

The thesis selected the surrogate by global accuracy and assumed the most accurate surrogate gives the best optimum.
Known counter-evidence: Eggensperger et al. 2015 (RF surrogates preserve optimizer rankings better than GP);
Williams & Cremaschi 2021 (best-for-approximation ≠ best-for-optimization); Jin 2011 (rank preservation matters more
than accuracy for EAs). **Fix:** measure the true-function regret of the optimizer's answer, for several optimizer
types (see PLAN Phase 4).

## B-05 · Target transform chosen on the test folds (S2)

`study.py:122-134` evaluates lin and log, then keeps the one with the better mean **held-out** error. This is model
selection on test data → optimistic by the max of two noisy estimates. **Fix:** nested CV, or fix the transform a priori,
or report both.

## B-06 · Noise-floor estimator biased high (S2)

`benchmark.py:230-242`: median |Δ²y| / median(y) along R at fixed (p, T). For i.i.d. noise of std σ,
Δ²ε has std √6·σ and median |Δ²ε| ≈ 0.674·√6·σ ≈ 1.65σ. Curvature of the true surface adds further.
Grouping by `round(p,1)` and `round(T,1)` assumes a structured grid. The "~12 % noise floor" quoted in the thesis is
therefore an over-estimate (true σ perhaps ~7 %). **Fix:** use replicate-free estimators (Rice/difference-based with the
√6 correction, or GP nugget MLE, or nearest-neighbour Gasser–Sroka–Jennen), and state it as an estimate with CI.

## B-07 · `nn_committee` implementation does not match the thesis text (S2)

Thesis Appendix: `nn_committee` uses "`nn`'s five-member bootstrap-trained deep ensemble" (torch).
Code `adaptive.py:85-88`: `rom = "mlp"` (sklearn MLPRegressor), `n_members = 4` inherited from `_CommitteeScout`.
Also the thesis notes `nn` silently disappears without torch (Python 3.13). Documentation bug; the reported numbers
are for 4 sklearn MLPs.

## B-08 · Low statistical power (S2)

8 seeds per cell. Wilcoxon signed-rank with n = 8 has a minimum two-sided p of 0.0078; the Benavoli posteriors in the
table take only values in {0.1, 0.2, 0.5, 0.8, 0.9}. Many "not significant" results are under-powered.
**Fix:** ≥ 30 seeds (cheap on analytic surfaces), common random numbers across methods, Friedman + Nemenyi /
critical-difference diagrams, hierarchical Bayesian comparison across surfaces, effect sizes.

## B-09 · Canonical surface set too narrow (S2)

Six hand-built 3-D surfaces at one n = 140 and one 3 % noise. Feature amplitudes are mild (e.g. knee active only in
R < 45 mm). No dimension scaling, no multimodal/valley/plateau classes, no discontinuity in value
(the "cliff" is a smooth logistic with slope 14). **Fix:** standard suites (Surjanovic–Bingham VLSE, BBOB/COCO, Jamil–Yang)
classified by landscape features, swept over d, n, noise.

## B-10 · Single upscaling configuration (S2)

`strategies.py:359-397`: 12 points per axis, `RegularGridInterpolator(method="cubic")`, grid bounded by training
min/max with clipping. Not the CasADi B-spline that the optimizer actually uses. Grid resolution was not swept, so
the claim "gp_bspline matches gp" holds for this grid only. **Fix:** sweep grid size, interpolant (linear, cubic
tensor, not-a-knot B-spline, smoothing P-spline, MBA, THB, Smolyak sparse grid, Chebyshev), and measure derivative
error too.

## B-11 · GBT ranking on real cache unsupported by statistics (S2)

Table `cv_full`: GBT 11.3 % vs GP 12.7 %, single 5-fold CV, no CI; footnote says "qualitatively robust". This is the
result Alonso/Needels found interesting, so the paper must re-establish it with repeated CV, CIs, and a clean
explanation (see STUDY §H3).

## B-12 · Scout vs deliverable hyper-parameter settings differ (S2)

Scout GP: `make("gp")` → 4 restarts. Deliverable GP in the scout study: `_FAST_GP = dict(restarts=1)`
(`study.py:161`). The scout and the model it is meant to serve are fit differently.

## B-13 · "LHS-like" claim vs i.i.d. uniform sampling (S3)

`canonical.py:122-134` docstring says "LHS-like"; code is `rng.uniform`. Same for `grid_pool`.

## B-14 · Speed table at one n (S3)

`study.py:395-` fits at n = 200. GP prediction is O(n) per point (mean) and O(n²) (variance); spline is O(1).
The 19–32× speed-up is n-dependent. **Fix:** report cost curves vs n, and include fit time and simulation cost.

## B-15 · Single-run production comparison (S3)

Appendix Table `real_heldout_final` is one trajectory (stated honestly in the thesis). Re-run with seeds.

## B-16 · Committee scout details (S2)

4 bootstrap members is a very small committee (std estimate from 4 samples has ~35 % relative error). Tree-ensemble
predictions are piecewise constant, so member spread is constant over cells and `np.argmax` resolves ties by pool
order. **Fix:** ≥ 10–20 members, or native estimators (infinitesimal jackknife for RF, NGBoost, quantile regression
forests), tie-breaking by distance to existing points.

---

## Math 796 HW1 optimizer comparison (prior work reused as baseline)

Source: `data/reference/math796_hw1/HW1_script.m`, report `MATH_796_HW_1.pdf`. Not thesis defects, but they limit
reuse of those numbers in the paper; the paper re-runs the optimizer study in Python.

| ID | Sev | Status | Issue |
|----|-----|--------|-------|
| H-01 | S2 | open | Single start point x0 = (20, 30) for every function and optimizer (`evaluate_optimizers`), outside the standard VLSE domain of most functions (e.g. Rastrigin [−5.12, 5.12]²; Ackley [−32.8, 32.8]² is fine) → results mix "find the basin from far away" with "optimize". |
| H-02 | S2 | open | Unbounded problems for all solvers except `surrogateopt` ([−70, 130]²) → unequal problem definitions across optimizers. |
| H-03 | S2 | open | Normalised objective = f / min f found; with optima at 0 (or negative) this explodes (table shows 10³⁰) — use regret f − f* with the known f*, or log-regret. |
| H-04 | S3 | open | `rng default` → one seed; stochastic optimizers need ≥ 30 seeds with CIs. GA initial population is drawn but other stochastic solvers use defaults. |

---

## New-code bugs and design issues (`code/surr/`)

| ID | Sev | Status | Issue |
|----|-----|--------|-------|
| N-01 | S1 | fixed (wrapper) — must be used | **Centre-of-box optima.** Many VLSE optima sit at the box centre (Rastrigin, Ackley, sphere-likes). DIRECT samples the centre first and "solved" Rastrigin with regret 0 in the smoke test; space-filling designs also hit the centre. Fix: `problems.base.shifted(p, seed)` gives BBOB-style randomly shifted instances (±20 % of range); all optimizer and sampling experiments use shifted instances, one per seed. |
| N-02 | S2 | fixed | `mf_weak` pairs had r = 0.78/0.85 instead of 0.5: the discrepancy φ is itself correlated with f_H, so with b ≥ 0 the correlation cannot fall below corr(f_H, φ). Allowing b < 0 hits the target exactly (all pairs now r = 0.950 / 0.500). |
| N-03 | S3 | blocked (author) | TabPFN v2 weights require accepting the Prior Labs licence and an API key (`TABPFN_TOKEN`). The learner raises a clear error until then; it never prompts interactively. |
| N-04 | S3 | documented | `nargp` is the augmented-input approximation (one ARD Matérn GP on (x, f̂_L)) — scikit-learn has no active-dimension product kernels for the exact NARGP kernel k_x·k_f + k_δ. Exact version planned with gpytorch. |
| N-05 | S3 | documented | `mepe` uses exact leave-one-out refits only for n ≤ 40; above that a 3-NN residual proxy (cost). |
| N-06 | S2 | open | `gmdh` NRMSE 7.5 on rotated_ridge_d5 (n = 80) — extrapolation blow-up of composed quadratics. Check whether external-criterion split or missing regularisation is the cause before reporting it as a property of GMDH. |
| N-07 | S3 | open | `kan` under-performs on smooth 2-D (NRMSE 0.16 vs GP 0.002) — tune grid size / epochs / early stopping before E2 so KAN gets a fair configuration (STUDY_PLAN principle 3). |
| N-08 | S3 | open | `bridge_mult` explodes (NRMSE 39–48) when f̂_L crosses or nears zero — genuine weakness of the ratio target (same singularity as the thesis TMF with t_scalar → 0), but the shift heuristic should be documented in the paper. |
| N-09 | S3 | noted | Optimizer budgets are enforced between multistarts, so IPOPT and trust-constr can overshoot by one local run (e.g. 662 / 600). Regret is reported against *actual* evaluations. |
