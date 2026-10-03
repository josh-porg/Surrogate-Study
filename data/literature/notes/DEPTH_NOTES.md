# Depth-read notes

Full-text reads (PDF → text in `data/local/txt/`, gitignored) unless marked *abstract-level*. Each note: what they
did · key quantitative findings · what it means for us (RQ/H from STUDY.md) · what we reuse. Keys = `paper/refs.bib`.

---

## williams2021novel — Williams & Cremaschi, ESCAPE-31 (CACE vol. 50), 2021
- **Did:** 8 surrogates (MARS, RF, 1-hidden-layer ANN, ELM, GP, SVM, ALAMO, RBFN) on **99 VLSE optimization test
  functions** (the same library and *the same shape categories* as the author's Math 796 HW1), d ∈ {2,4,6,8,10},
  n ∈ {50…1600}, Halton/Sobol/LHS → 693 datasets, 16 632 models; hyper-parameters by 10-fold CV; metric adjusted-R²
  (on training data, penalised by parameter count — not a held-out metric).
- **Found:** sampling scheme (Halton/Sobol/LHS) made no significant difference. GP most often best on valley, bowl,
  "other"; ALAMO frequent winner on bowl; **MARS most often best on many-local-minima**. Many ties not significantly
  different. Strongest single predictor of model accuracy: **minimum Mahalanobis distance between training points**
  (ρ = −0.71 with RF adj-R²) — i.e. a spacing/fill feature, not a landscape feature. RF recommender: 87 % accuracy,
  86 % precision.
- **For us:** closest precedent to RQ1/RQ7 on our exact suite. Gaps we fill: no noise, no held-out clean-truth error,
  no optimizer regret in this conference version (the CERD journal version adds SBO — *williams2021selection*, read
  abstract-level: RF, SVR, GP most robust for SBO), no MF, no upscaling, no scouts. Reuse: their 40 data attributes as
  candidate ELA covariates; the "fill-distance predicts accuracy" result is consistent with kernel theory (M.7–M.8
  power function ∝ fill distance) and should be tested explicitly.

## eggensperger2015efficient — AAAI 2015
- **Did:** 9 HPO datasets; regression models GB, RF, GP (Spearmint, MCMC hyper-parameters), SVR, NuSVR, kNN, linear,
  ridge; metrics RMSE and Spearman CC in 5-fold CV, and whether optimizers (SMAC, TPE, Spearmint) behave the same on the
  surrogate as on the real benchmark.
- **Found:** "**RF achieved the highest CC on all 9 datasets, while GB tended to yield the lowest RMSE**"; GP best of the
  non-tree models. Tree surrogates reproduced optimizer trajectories; GP surrogates could give qualitatively different
  optimizer behaviour.
- **For us:** direct evidence for **H6** — the RMSE winner and the rank-preservation winner differ, and rank
  preservation is what matters for optimizer behaviour. Reuse: report RMSE *and* Spearman/Kendall on sublevel sets.

## eggensperger2018efficient — Mach. Learn. 2018
- Extends to algorithm configuration; uses **quantile regression forests** for randomized targets; surrogate
  benchmarks with 10²–10⁴× speed-ups and small ranking error. Warns: training data should be collected where
  optimizers actually go (high-performance regions) — a sampling × task coupling (RQ2/RQ5).

## grinsztajn2022tree — NeurIPS D&B 2022
- **Did:** 45 datasets (~10 k samples, "typical tabular"), 20 000 CPU-h random HP search per learner.
- **Found (mechanisms):** (1) **Smoothing the training target with a Gaussian kernel strongly hurts trees, barely
  affects NNs** → targets are irregular and NNs are biased to smooth solutions (spectral bias, Rahaman 2019). (2) MLPs
  are hurt far more by uninformative features. (3) Rotation-invariant learners (MLP) have worst-case sample complexity
  linear in the number of irrelevant features (Ng 2004); random rotations of the inputs hurt trees and help the
  NN-vs-tree gap close → **data orientation carries information**.
- **For us:** gives a **ready diagnostic for H3** on the FEA cache: (a) smooth the target at increasing length-scales
  and watch GBT vs GP error; (b) randomly rotate inputs; (c) add uninformative inputs. A stationary-kernel GP is, like an
  MLP, biased to smooth functions (and isotropic RBF is rotation-invariant). Also motivates F10: if orientation matters,
  *learned* rotations (active subspaces) cut both ways.

## moustapha2022active — Struct. Saf. 2022
- **Did:** modular active-learning reliability framework (surrogate × reliability estimator × learning function ×
  stopping criterion); 39 strategies × 20 problems, > 12 000 runs, replications; codes in UQLab.
- **Found:** "**no strategy consistently outperforms the others**"; clear feature-dependent patterns (dimension,
  failure-probability magnitude); "essentially no drawbacks in using surrogate models". Won 24/27 problems of an
  external blind challenge (TNO).
- **For us:** the closest *methodological* template for our stage-swap ablation (F0's S1–S6). Copy the presentation:
  per-module recommendations conditioned on problem features, plus a blind real-problem check (our FEA + aircraft data).

## luthen2021sparse — SIAM/ASA JUQ 2021
- **Did:** sparse-PCE framework (basis × sampling × solver), 11 engineering-like functions, 30–50 replications.
- **Found:** sampler + solver choice → **orders-of-magnitude** relative-MSE differences; rankings of solvers and
  samplers are largely independent; low-d small-n: BCS best, SPLOO robust; high-d: BCS + LHS, no advanced sampling beats
  LHS.
- **For us:** (i) evidence that within-family choices can dominate between-family choices — so each family must be
  given its best configuration (fairness of the comparison); (ii) "no advanced sampling beats LHS in high-d" parallels
  B-02 — test adaptive vs LHS/Sobol as the real control. Replications 30–50 = our seed target.

## vanrijn2021finding — van Rijn & Schmitt 2021 (arXiv 2103.03280)
- **Did:** "error grids" of hierarchical-surrogate MSE over (n_h, n_l), many MF benchmark functions (their `mf2`
  package), subsampling estimate of the grid from one DoE, budget-split rule from the error-grid gradient.
- **Found:** correlation between fidelities is a poor guide alone; the error grid's gradient tells how to split the next
  budget; fails where the dominant behaviour changes with n.
- **For us:** RQ3 factor design (n_h, n_l, κ) should be presented as error grids per learner family; the subsampling
  trick lets us estimate grids on the **real FEA cache** (t_scalar = LF at every point, so any n_l is available).

## ober2021promises — UAI 2021
- **Found:** DKL's good results come mostly from **implicit regularisation by minibatch SGD**, not from the marginal
  likelihood; with many hyper-parameters the marginal likelihood "tries to correlate all the datapoints" → worse
  over-fitting than a plain NN; a fully Bayesian treatment of the NN weights fixes it.
- **For us:** H8's prediction that DKL over-fits at n ≲ 20d is grounded; implement DKL with (a) ML-II full-batch,
  (b) minibatch, (c) kernel-flow objective, (d) small φ; expect (a) to fail.

## lutjens2024climatebench — Lütjens et al. (arXiv 2408.05288)
- **Found:** on ClimateBench a **linear pattern-scaling baseline beat a 100 M-parameter foundation model (ClimaX) on
  3 of 4 variables**; the winner flipped with the number of realisations (internal-variability noise) — a
  bias–variance/noise effect; "a cautionary tale" about not benchmarking against simple methods.
- **For us:** (i) always include simple baselines (linear/quadratic RSM, IDW, kNN); (ii) noise level can flip rankings —
  exactly B-01's mechanism and our noise factor.

## lamperti2018agent — J. Econ. Dyn. Control 2018
- **Did:** XGBoost surrogate + iterative ("intelligent") sampling to calibrate agent-based models; tested on 10⁴–10⁶
  out-of-sample points.
- **Found / argued:** kriging limited beyond ~20 parameters and "suffers from smoothness assumptions that collapse
  interesting patterns" on ragged ABM surfaces; boosted trees handle ragged surfaces.
- **For us:** an independent field (computational economics) reaching the same GBT-over-GP conclusion for *rough*
  simulator responses — supports H3's "irregularity" explanation.

## neufang2024surrogate — Neufang, …, del Rio Chanona 2024 (arXiv 2412.13948)
- Model-based DFO benchmark (CUATRO, SRBF, DYCORS, COBYLA, TuRBO, SNOBFIT, BO) on Ackley/Levy/Rosenbrock/ill-conditioned
  quadratic and chemical-process cases. Rankings **changed between test functions and engineering cases** (CUATRO,
  SNOBFIT): "assessing … in engineering applications must be done in addition to … traditional mathematical test
  functions".
- **For us:** justifies the real-data anchors (RQ8) and the optimizer factor (RQ5); DYCORS/SRBF are strong
  RBF-based baselines to include.

## mainini2022analytical — AIAA (NATO AVT-354), arXiv 2204.07867
- **Suite:** MF1 Forrester (continuous + discontinuous; 4 fidelities), MF2 Rosenbrock (scalable), MF3 shifted-rotated
  Rastrigin (fidelity-parametric), MF4 heterogeneous, MF5 spring–mass system, MF6 Paciorek (noisy). Metrics: global
  accuracy (vs highest fidelity) and goal-oriented optimization accuracy, cost-normalised.
- **For us:** adopt as the MF part of the canonical suite (RQ3); reuse their metric definitions so results are comparable
  with the AVT-354 community (Alonso/Needels' audience). Follow-up framework paper: *mainini2025framework* (ACME 2025).

## Abstract-level (full text not openly available or not yet read)
- **fuhg2021state** — adaptive sampling for kriging; success depends on problem features and analysis goal; open toolbox.
- **peherstorfer2018survey** — adaptation / fusion / filtering taxonomy; keep HF in the loop for guarantees.
- **toal2015some** — MF kriging helps only if r² high (≳0.9) and LF share of budget in ~10–80 %.
- **hollmann2025tabpfn** — beats tuned GBT ≤ 10 k samples in one forward pass; regression supported.
- **needels2024trajectory / needels2024efficient** — trajectory-informed sampling for MF GP surrogates of hypersonic
  aero/aerothermal loads; adjoint-guided. Request PDFs (AIAA paywall) — or ask Dr Needels directly.
- **williams2021selection** (journal) — adds SBO: RF, SVR, GP most robust for optimization; MARS, GP best for
  approximation; PRESTO tool.

## Cross-cutting synthesis (feeds paper §2–§6)
1. **No universal winner, but predictable winners.** Every large benchmark (Jin 2001; Williams 2021; Fuhg 2021;
   Moustapha 2022; Lüthen 2021; Neufang 2024) concludes "no single best", then finds feature-dependent patterns. Our
   contribution must therefore be a *conditional* map, with the conditioning features measurable from the initial design.
2. **Two cultures, one mechanism.** Engineering (GP-dominant) evaluates on smooth analytic functions, small n, often
   noise-free; tabular ML/AutoML/ABM (tree-dominant) evaluates on irregular, noisy, larger-n, partially irrelevant-input
   data. Grinsztajn's smoothing experiment and Lamperti's ragged-surface argument point to **target irregularity +
   noise + irrelevant/misoriented inputs** as the switch. The thesis FEA cache sits between the two cultures.
3. **Accuracy ≠ decision quality.** RMSE winner ≠ rank-correlation winner (Eggensperger 2015); approximation winner ≠
   SBO winner (Williams 2021 journal); test-function winner ≠ engineering-case winner (Neufang 2024); noise level
   flips winners (Lütjens 2024). Our regret-at-matched-cost end-point follows.
4. **Within-family configuration matters as much as family** (Lüthen 2021: orders of magnitude). Fair comparison
   requires per-family tuning with equal budget — and we must report tuning cost.
5. **Simple baselines are mandatory** (Lütjens 2024).
