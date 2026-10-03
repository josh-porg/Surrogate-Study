# Study Document — content for the paper

Holds everything the paper needs: framing, research questions, hypotheses, frameworks, the experimental design,
results as they arrive, and the outline. Math lives in MATHEMATICS.md (cited as M.x); references in SOURCES.md;
defects in BUGS.md (B-xx); schedule in PLAN.md.

---

## 1. Working title and pitch

**Working title:** *Accuracy Is Not the Objective: A Cross-Disciplinary Review and Global Study of Surrogate Pipelines —
Sampling, Learning, Multi-Fidelity Fusion, Upscaling and Optimization — for Expensive Engineering Analyses*

Alternative: *When Do Gradient-Boosted Trees Beat Gaussian Processes? A Factorial Study of Surrogate Pipelines from
Approximation Theory to Optimizer Regret.*

**Pitch (3 sentences).** Engineering surrogate practice is dominated by Gaussian processes, tabular machine learning is
dominated by tree ensembles, and in a thesis FEA study gradient-boosted trees out-predicted every GP variant on the
real data while losing on synthetic surfaces. We review surrogate, uncertainty, sampling, multi-fidelity, upscaling and
optimization methods across disciplines, then run a factorial study that varies surface class, dimension, budget,
noise, fidelity structure and cost, scout × deliverable pairing, upscaling, and optimizer type, scoring pipelines by
**true-function regret per unit simulation cost**. We give rules — backed by approximation theory and by
landscape features measurable from the initial design — for choosing a pipeline, and explain when and why the
"most accurate" surrogate is not the one that yields the best design.

**Audience:** aerospace MDO (Alonso, Needels — multi-fidelity GP / EGO people), computational statistics (computer
experiments), AutoML/tabular ML. Venue candidates: *AIAA Journal*, *Structural and Multidisciplinary Optimization*,
*Archives of Computational Methods in Engineering* (review + study format fits), *JCP*, *SIAM/ASA JUQ*.

## 2. What the thesis established (baseline, with caveats)

Source: `Maters-Project-V3/chapters/Appendix_Spline_Study.tex`, `chap_12_modeling_framework.tex` §12 "Surrogate
Modeling Strategy"; code `COMPASS/compass/surrogate/`.

| Thesis claim | Evidence | Caveat (BUGS) |
|---|---|---|
| GP lowest/tied-lowest error on all 6 canonical surfaces | Tables canonical_a/b, n = 140, 3 % noise, 8 repeats | B-01 noise floor compresses ranking; B-05; B-09 |
| GP→grid→cubic interpolant keeps GP accuracy, 19–32× faster | canonical_b, eval_speed | B-10 one grid; B-14 one n |
| GP-variance scout saves ~44 % FEA on `collapse_knee` (p<0.05) | Table scout_knee | B-02 control is i.i.d. random; B-03 GP deliverable only; B-08 power |
| Scouting n.s. on real cache | Table scout_realcache | B-08; B-06 noise floor overstated |
| Random beats scouts on `oscillatory` | Table scout_oscillatory | consistent with B-02 (space-filling vs localisation) |
| GBT 11.3 % vs GP 12.7 % CV on real cache; GBT not used (non-smooth) | Table cv_full | B-11 no CI |
| Target/space choice minor; floor-bound exclusion is the big lever | target_selection | B-05 |
| (Thermochemistry) direct uniform sampling beat surrogates when cached evaluations are cheap | §12 text | cost crossover — §M.8 |

**Gaps named by the user:** (G1) no multi-fidelity study for any surrogate; (G2) scouts tested only against a GP
deliverable, no native or cross combinations; (G3) one upscaling technique; (G4) too few canonical surfaces;
(G5) optimizer performance on the surrogate never measured; (G6) no comparison of optimizer types; (G7) assumed
closest surrogate ⇒ best optimum; (G8) no scaling laws vs data amount, acquisition cost, MF quality, noise quantity and
quality, surface type; (G9) coupling between algorithm, surrogate and design space not studied.

## 3. Data assets

| Asset | Contents | Status |
|---|---|---|
| `COMPASS/results/tmf_cache/tmf_bbd96c4b56.csv` | 901 FEA runs (60×12 mesh), 623 non-floor-bound; inputs R, p, T; outputs t_fem, t_scalar, TMF, damage; ~961 CPU-h total, mean 1.07 h/run | **Available** — the dataset the thesis used. Pre-Bauschinger. |
| `tmf_dca5926004.csv` | 203 runs (103 usable) — later physics version | Available |
| `tmf_5f77ee7aa5.csv` | 15 runs on coarse 40×10 mesh | Available — a tiny **mesh-fidelity** level |
| `t_scalar` column | closed-form cruise sizing at every point | **Free low-fidelity model** for MF experiments on real data |
| Analytic suites | VLSE, BBOB/COCO, Jamil–Yang, `mf2`, Mainini MF set | public |
| Math 796 HW1 test suite | 32 two-dimensional VLSE functions in 6 categories (many local minima 9, bowl 5, plate 5, valley 4, steep ridges/drops 3, other 6) + MATLAB optimizer comparison | **Found**, copied to `data/reference/math796_hw1/`; seeds the canonical suite and the optimizer baselines |
| Aircraft performance dataset (Math 796 final report; repo `aircraftPerformancePrediction`) | historical aircraft, 5 outputs (weight fractions, T/W, shaft power), heterogeneous engine types | **In scope (author, 2026-10-03).** Available. Second **real tabular engineering** dataset where trees did well: test R² GBT 0.549, deep ANN 0.313, transformer 0.585, linear −0.853 (with jets). Candidate for RQ8 transfer — different regime (n in the hundreds, d larger, strongly stratified) |

So, contrary to the worry, the real FEA data still exist and no FEM re-runs are needed. The real data serve as the
**engineering anchor** (one realistic, noisy, regime-partitioned 3-D problem with a natural LF model); the canonical
suites carry the factorial science.

## 4. Research questions

- **RQ1 (learners).** How does each surrogate family's accuracy — value, derivative, ranking, calibration — scale with
  budget n, dimension d, noise level/type, and landscape class? Where are the crossovers (e.g. GP→GBT)?
- **RQ2 (sampling).** Does adaptive sampling beat the best *non-adaptive space-filling* design (not just i.i.d. random),
  and does the benefit depend on whether scout and deliverable share a model family (native vs cross)?
- **RQ3 (multi-fidelity).** For each learner family, when does adding a low-fidelity source pay, as a function of
  LF–HF correlation r, relationship type (linear/nonlinear), cost ratio κ and LF budget fraction φ? Do simple
  feature-augmented MF trees/NNs compete with co-kriging/NARGP?
- **RQ4 (upscaling).** Which upscaling route (none; grid→interpolant at various resolutions; smoothing; P-spline/MBA/THB
  directly; sparse grids; distillation) preserves or improves accuracy and derivative quality — and can upscaling
  regularise a rough learner (GBT) into a good optimizer-facing surrogate?
- **RQ5 (optimization).** Given a surrogate, which optimizer class yields the lowest true regret, and how weakly does
  global accuracy predict regret (M.16–M.17)? Is there an algorithm × surrogate × landscape interaction?
- **RQ6 (cost).** At what simulation cost does each level of pipeline sophistication pay for itself?
- **RQ7 (selection).** Can landscape features computed on the initial design predict the best pipeline, out of sample?
- **RQ8 (real data).** Do the canonical findings transfer to the real FEA problem, and what explains GBT's advantage there?

## 5. Hypotheses (pre-registered before running)

- **H1** On smooth, stationary, low-d surfaces at small n, GPs (and PCE/RBF) beat trees in clean-truth error; the gap
  shrinks with noise and closes/reverses with discontinuities, plateaus/floors, regime changes, uninformative inputs
  and larger n.
- **H2** Most of the thesis "GP-variance scout" gain is reproduced by a maximin/Sobol' design with no model (B-02).
  Genuinely y-adaptive acquisition helps only on localised-feature surfaces, and helps most when scout = deliverable family.
- **H3** GBT's real-cache win stems from one or more of: (a) heavy-tailed / heteroscedastic FEA noise, (b) regime
  structure inside the "smooth" region (residual floor/kink effects), (c) the target being nearly piecewise in one input
  (`R` collapse knee), (d) GP hyper-parameter pathologies (nugget at bound — documented in strategies.py:197-203).
  Each is testable by reproducing it on a canonical surface with the matching feature.
  *Diagnostics from the literature (DEPTH_NOTES):* (i) Grinsztajn's target-smoothing test — smooth the FEA target with a
  Gaussian kernel at increasing length-scales; if GBT degrades and GP does not, the data are irregular in a way GP
  cannot represent; (ii) random input rotations (trees are orientation-sensitive, isotropic GP/MLP are not);
  (iii) added uninformative inputs. Run the same three on canonical surfaces with known regularity.
- **H4** MF helps only above a correlation threshold (r ≳ 0.9, Toal) and within an LF budget-fraction window; nonlinear
  MF (NARGP, feature-augmented NN/GBT) widens the window when the relationship is nonlinear. On the real data, the
  `t_scalar` multiplicative bridge (= thesis TMF) is already most of the MF benefit.
- **H5** Upscaling a smooth learner costs ≈ nothing at adequate grid size; upscaling a rough learner with a *smoothing*
  spline improves gradient-optimizer regret relative to the raw learner, but tensor grids become infeasible beyond d≈6.
- **H6** Global RMSE ranks pipelines differently from optimizer regret; Kendall-τ on the near-optimal sublevel set
  predicts regret better than RMSE for comparison-based optimizers; sup-norm + gradient error near x* predicts it for
  gradient-based optimizers.
- **H8 (F10)** A change of basis wins exactly when the surface is simple in that basis, and the gain is predictable
  from the initial design: Fourier/CS wins on oscillatory surfaces (where the thesis GP-variance scout lost to random),
  wavelets on localised knees/cliffs, active-subspace/PPR on rotated-ridge and anisotropic surfaces, tensor-train on
  additive/separable ones, and physics-dictionary latents where such features exist. Learned nonlinear encoders (DKL,
  SINDy-AE) over-fit at n ≲ 20d and only pay at larger n or with LF inputs; the kernel-flow objective reduces that
  over-fitting relative to ML-II. Compression (randomised SVD / TT) speeds up upscaling and scout refits by orders of
  magnitude at negligible error up to a rank threshold, and makes upscaling feasible for d > 6.
- **H7** There is a cost crossover: below some c_H (relative to fit cost), uniform/space-filling sampling with a simple
  learner wins (as for the thesis thermochemistry ROM).

## 6. Frameworks (pipelines) and stages

A **framework** is a dataflow; a **pipeline** is a framework with one algorithm chosen per stage.

| ID | Framework | Stages |
|---|---|---|
| F0 | **Thesis two-stage**: space-filling init → scout loop → intermediate model → dense grid → spline deliverable → optimizer | S1 S2 S3 S4 S5 S6 |
| F1 | One-shot: DoE → learner → optimizer | S1 S4 S6 |
| F2 | Adaptive direct-smooth: adaptive sampling → smooth learner used directly (GP mean, P-spline, RBF, NN) → optimizer | S1 S2 S3 S4 S6 |
| F3 | Optimization-directed (EGO/BO, SMAC, TPE, TuRBO, DYCORS): infill aims at the optimum, not global accuracy | S1 S2(opt) S3 S6 |
| F4 | Trust-region model management (provably convergent; corrections enforce first-order consistency) | S1 S4 S6 |
| F5 | Multi-fidelity: LF sampled densely → HF correction (bridge, co-kriging, recursive, hierarchical, NARGP, MF-NN, feature-augmented trees, space mapping) → any downstream | S0 + any |
| F6 | Partitioned: classifier (feasible / floor-bound / infeasible) + per-regime learners (TGP, MoE, gp_regime) | S4' |
| F7 | Ensemble: multiple learners combined by CV weights or stacking | S4'' |
| F8 | Physics-structured: physics dictionary / physics mean / monotone constraints | S4 |
| F9 | Meta-learned prior: TabPFN / neural process as learner and scout | S3 S4 |
| F10 | **Change-of-basis / latent** (from the author's DMD / CS-DMD / SINDy-autoencoder ROM work): transform x and/or represent f in a basis where it is sparse, low-rank or low-dimensional; fit there; optionally co-learn transform + fit; optionally compress intermediate stages for speed (M §3.10) | S4 (+ S2 when the method picks its own samples, e.g. TT-cross; + S5 compressed upscaling) |

**Stage menus** (the "swap in a bunch of algorithms" ablation):

- **S0 fidelity allocation:** HF only · fixed LF fraction φ · cost-aware MF acquisition.
- **S1 initial design:** i.i.d. random · LHS · maximin-LHS · Sobol' · Halton · CVT · tensor grid.
- **S2 acquisition:** none · max-variance · IMSE/ALC · EIGF · MEPE · LOLA-Voronoi · TEAD · QBC · CV-error · sequential
  maximin (model-free) · EI/LCB (optimization-directed) · U-function (boundary-directed) · cost-weighted variants.
- **S3 scout / uncertainty model:** GP (frozen θ / refit θ) · RF-IJ · QRF · bootstrap GBT (K=4/10/20) · NGBoost · deep
  ensemble · MC dropout · BART · TabPFN · conformal wrappers.
- **S4 learner (deliverable or intermediate):** linear/poly · sparse PCE · sparse physics · RBF (TPS, MQ, Gaussian) · GP
  (Matérn 3/2, 5/2, RBF; ARD; nugget policy) · heteroscedastic GP · TGP · MARS · SVR · KRR · RF · GBT (XGB/LightGBM/
  CatBoost, ± monotone) · BART · MLP · deep ensemble · KAN · TabPFN · P-spline/GAM · IDW · natural neighbour · MLS ·
  GMDH · AAA-rational · ensembles.
  **F10 additions:** PPR · active-subspace GP / polynomial ridge · input-warped GP · deep kernel learning (ML-II vs
  kernel-flow objective) · static SINDy-autoencoder (encoder + sparse dictionary) · compressed-sensing fits in Fourier /
  Chebyshev / Legendre / wavelet bases · functional SVD (2-D) / tensor-train (TT-cross) · combinations (AS rotation → TT;
  log/warp → CS) · AutoML co-search over {transform × basis × learner}.
- **S5 upscaling:** none · grid m ∈ {6, 12, 24, 48}/axis × {linear, cubic tensor, not-a-knot B-spline, smoothing
  spline} · MBA · THB · P-spline on scattered data · Smolyak sparse grid · Chebyshev · NN/KAN distillation ·
  **compressed upscaling** (randomised HOSVD / TT-SVD of the baked grid, or TT-cross of the intermediate model directly).
- **S3/S2 speed variants:** Nyström / random-feature GP scouts (compressed refits inside the AL loop).
- **S6 optimizer:** IPOPT/SLSQP multistart · L-BFGS-B · trust-constr · Nelder–Mead · COBYLA · DIRECT · DE · CMA-ES ·
  PSO · basin hopping · (F3/F4: optimizer may query f).

The full cross product is far too large; see the design in §7.

## 7. Experimental design

### 7.1 Factors
| Factor | Levels |
|---|---|
| Problem | ~40 analytic functions spanning ELA classes (smooth/separable, ill-conditioned valley, multimodal regular, multimodal irregular, plateau/floor, kink, discontinuity, localised knee, oscillatory, additive vs interacting, uninformative inputs) from VLSE/BBOB/Jamil–Yang + the 6 thesis surfaces + engineering functions (Borehole, OTL, Piston, Wing weight) + real FEA |
| Dimension d | 2, 3, 5, 8, 10 (20 for scalable learners) |
| Budget n (HF-equivalent) | 5d, 10d, 20d, 50d, 100d, 300d |
| Noise level | 0, 0.5 %, 1 %, 3 %, 10 % of sd(f) |
| Noise type | additive, multiplicative, heteroscedastic, heavy-tailed, deterministic-numerical |
| MF: correlation r | 0.5, 0.8, 0.9, 0.95, 0.99; linear vs nonlinear relationship |
| MF: cost ratio κ | 3, 10, 30, 100, 1000 |
| MF: LF fraction φ | 0, 0.1, 0.25, 0.5, 0.8 |
| HF unit cost c_H (for wall-clock regret) | 1 ms, 0.1 s, 10 s, 1 h |
| Seeds | ≥ 30 (common random numbers) |

### 7.2 Phased design (to keep it tractable)
1. **Screen (Phase 1):** each stage's menu varied one-at-a-time around a reference pipeline, on a 12-problem screening
   subset, 3 budgets, 2 noise levels → keep the top ~5 per stage plus every thesis method and every "surprise".
2. **Learner study (RQ1, RQ8):** survivors × all problems × d × n × noise, clean-truth scoring, derivative and ranking
   metrics. Fixes B-01, B-05, B-09.
3. **Sampling study (RQ2):** scout × deliverable full factorial (native + cross), with space-filling controls. Fixes B-02,
   B-03, B-12, B-16.
4. **MF study (RQ3):** MF methods × learner families × (r, relationship, κ, φ), plus real-data `t_scalar`/coarse mesh.
5. **Upscaling study (RQ4):** S5 menu × smooth and rough learners, with derivative metrics. Fixes B-10.
6. **Optimization study (RQ5, RQ6):** surrogate × optimizer × problem; regret, success rate, cost. Fixes B-04.
7. **Selection model (RQ7):** ELA features → best pipeline; leave-problem-out validation.
8. **Real-data transfer (RQ8):** repeat key comparisons on the FEA cache with 30-seed repeated CV and retrospective active
   learning; test H3's explanations.

### 7.3 Scoring
Primary: clean-truth NRMSE (M.3), true regret (M.1) at matched cost (M.18). Secondary: derivative error (M.4), Kendall τ
on sublevel sets, calibration (NLPD, coverage), fit/predict time. Statistics per M §9.

### 7.4 Compute estimate (to refine after Phase 1)
Analytic evaluations are ~free; cost is model fitting. Rough: 40 problems × 5 d × 6 n × 5 noise × 30 seeds ≈ 180 k
datasets per learner; at ~0.5 s average fit (GP at n≤3000 dominates) and ~25 learners ≈ 625 CPU-h for Phase 2 alone →
feasible on the "more powerful computer" or a cluster; prune n×d corners where GP is O(n³)-infeasible.

## 8. Paper outline (part review, part study)

1. **Introduction** — expensive analyses in design; the accuracy-vs-decision gap; GP-vs-trees tension across fields;
   contributions.
2. **A cross-disciplinary map of surrogate modelling** (review; SOURCES §1–2)
   2.1 Lineages: geostatistics → DACE/kriging; approximation theory → splines/RBF/sparse grids/rational; statistics →
   regression/MARS/GAM/BART; ML → trees/NN/foundation models; physics → PCE/ROM/space mapping. 2.2 Forgotten and
   under-used methods (GMDH, Shepard/Sibson/MLS, Lipschitz optimal recovery, AAA rational, MBA/THB, Smolyak).
   2.3 What each field optimises for and why their rankings disagree.
3. **Uncertainty, sampling and active learning** (SOURCES §3–4).
4. **Multi-fidelity modelling** (SOURCES §5; M §5) — including the "un-named MF" in many engineering ROMs (bridge targets).
4b. **Changing the domain: latent, sparse-basis and low-rank surrogates** (SOURCES §2.7; M §3.10) — the ROM view
   (POD/DMD/SINDy) carried over from dynamics to static response surfaces; co-learning; compression for speed.
5. **From learner to optimizer-ready surrogate: upscaling** (SOURCES §6; M §6).
6. **Surrogate-based optimization and the accuracy fallacy** (SOURCES §7; M §7).
7. **Theory** — approximation rates, noise floors, no-free-lunch, regret bounds; what theory predicts per landscape class.
8. **Study design** (§7 here).
9. **Results** — RQ1–RQ8.
10. **Practical guidance** — decision chart keyed to measurable features, budget and cost.
11. **Limitations, threats to validity.**
12. **Conclusions.** Appendices: full tables, reproducibility (code + data DOI), thesis errata (BUGS S1/S2 items).

## 9. Results log

_Empty — no new experiments run yet._ Entries go here as `YYYY-MM-DD · experiment ID · finding · file`.

### Smoke-test signals (1 seed, small n — direction only, not results)
- 2026-10-03 · Thesis code, GP on `smooth`, seed 0: noisy-fold CV 3.3 % (published 3.0 ± 0.1) vs clean-truth
  ≈ 1.1 % — the published canonical numbers are mostly the 3 % noise floor (B-01). Full test: E1.
- 2026-10-03 · Fourier-sparse probe, n = 120: CS in a periodic Fourier basis recovers it exactly (NRMSE 0.000), GP
  0.70; wrong basis (cosine) 0.89 — basis match decides (H8).
- 2026-10-03 · Plateau-bump probe: GP max-variance scout ≈ random ≈ model-free maximin (1.04–1.06); LOLA-Voronoi
  0.67, EI 0.77 (B-02 / H2 direction).
- 2026-10-03 · Upscaling XGBoost: gradient NRMSE 122 raw → 1.03 tensor spline → 0.50 smoothing spline, value error
  also improves (H5 direction).
- 2026-10-03 · Multiplicative bridge (= thesis TMF target) blows up (NRMSE 39–48) when the LF passes near zero.
- 2026-10-03 · **NN training optimizer:** deep ensemble trained full-batch with L-BFGS vs Adam: NRMSE 0.004 vs 0.067
  (smooth 2-D, n = 40) and 0.006 vs 0.029 (5-D ridge, n = 80); Barzilai–Borwein 0.131 / 0.016; SGD 0.141 / 0.084.
  At engineering sample sizes the training optimizer can matter more than the network — now an E2 factor.
- 2026-10-03 · SGD-family *design* optimizers (Adam, momentum, BB, Robbins–Monro, Kiefer–Wolfowitz, SPSA) trail
  L-BFGS/CMA-ES on smooth deterministic problems once objective scaling is fixed; their test is noisy gradients (E9).

## 10. Preliminary observations (no new runs; from reading code and data)

- 2026-10-03 · The thesis real FEA cache is intact: 901 runs, 623 usable, ~961 CPU-h. FEM re-runs are unnecessary.
- 2026-10-03 · The cache carries a free LF model (`t_scalar`) at every point, and 15 coarse-mesh runs → MF on real data
  is possible.
- 2026-10-03 · Canonical CV in the thesis is noise-floored (B-01); GP's "win everywhere" must be re-tested before it is
  repeated in the paper.
- 2026-10-03 · The GP-variance scout is mathematically a space-filling rule when θ is fixed (M.8; B-02).
- 2026-10-03 · The thesis `oscillatory` surface (random sampling beat all scouts) is sparse in a Fourier basis — the
  natural first test of the F10 compressed-sensing branch (H8).
- 2026-10-03 · Trees also out-predicted the deep ANN on the author's aircraft-performance data (R² 0.55 vs 0.31), the
  second engineering dataset with this pattern; there, a transformer edged GBT (0.585).
