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
- **H9 (E12, added 2026-10-11, before E12 runs)** Rational surrogates beat every other class by orders of magnitude on
  clean data with pole/knee/resonance structure, and lose that edge as noise rises (≥ 1 %), because AAA-type methods
  interpolate and noise creates spurious pole–zero pairs. Pairing AAA with a dimension-reducing structure (ridge
  direction, additive terms) carries the gain past d ≈ 3 only when the structure is present. A GP with a rational
  mean keeps most of the gain and stays robust to noise, because the GP absorbs what the rational trend misses.
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

Entries: `YYYY-MM-DD · experiment ID · finding · file`.

- 2026-10-03 · **E1** · Reproduction gate passed: 71/72 published App. G canonical cells reproduced within CIs with
  the thesis' own COMPASS code (miss: `gp_physics_mean` on `clamp_floor`, 3.8 vs 4.4 — likely a later edit of the
  sparse-physics dictionary). · `results/E1_canon_report.md`
- 2026-10-03 · **E1** · Clean-truth scoring (30 seeds): true relative errors are 2–4× below the published noisy-CV
  numbers (GP ≈ 0.8–2.4 % vs 3.0–3.8 %); per-surface rankings almost unchanged (Spearman 0.91–0.99). The 3 % noise
  floor *compressed* the gaps — GP's lead over GBT on `smooth` is 2.4× on clean truth vs 1.2× as published. The
  thesis conclusion (GP best on the canonical surfaces) stands and is stronger; the published magnitudes are not.
- 2026-10-04 · **E2a** · Learner screening, 32 300 fits (45 configs × 12 problems × 3 n × 2 noise × 10 seeds;
  Kaggle + laptop). GP family leads overall (mean rank: SMT kriging 7.4, hetGP 7.8, SMT-KPLS 9.0, GP 9.4); designed
  strengths confirmed (trees win `axis_step` and `tree_draw`; CS-Fourier wins `fourier_sparse`; AS-GP wins
  `rotated_ridge` and `smooth_trig`). **Expectations contradicted:** MARS does not win `kink_floor` (AS-GP, GP,
  local GP do); KRR/SVR beat ML-II GPs on the GP prior draw (check the draw's construction and GP fitting, BUGS
  N-14). Training optimizer at scale: L-BFGS best for MLP ensembles (mean rank 17.0 vs AdamW 18.7, Adam 19.0, BB
  23.7, SGD 25.3) and KAN. Shortlist for E3: 35 configs (top 6 ∪ top 2 per problem ∪ thesis counterparts ∪ best
  per method family). · `results/E2a_learners_report.md`, `results/E2a_shortlist.json`
- 2026-10-03 · **E1** · `gp_regime` is identical to `gp` on every canonical surface (no regimes to switch on);
  grid/B-spline variants of GP match raw GP within CI (thesis claim confirmed on clean truth). · same file
- 2026-10-10 · **E3 (interim: all non-SMT learners complete, 783 560 fits; SMT kriging still running)** ·
  2 041 complete cells (problem × n × noise × level, 10 seeds). Best-of-family head-to-head: the GP family wins
  51-78 % of cells at every samples-per-dimension m = n/d. Tree ensembles are strongest at m = 5 (43 %) and fade
  with data (10 % at m = 300). Networks rise with data (3-6 % at m ≤ 20, 34 % at m = 300). Trees win
  discontinuities (74 %) and steep ridges/drops (73 %). GP leads on smooth (97 %), ridge (96 %), multiscale (93 %),
  engineering (89 %) and the thesis problems (81 %). Thesis pair GP vs XGBoost: geometric-mean NRMSE ratio
  xgb/gp 1.7 at m = 5 rising to 3.1 at m = 300, but noise closes the gap (5.7× noise-free, 1.5× at 10 % additive).
  `prior_draw` is near even (GP 52 %, trees 47 %), which is unexpected for GP prior draws (BUGS N-14 still open).
  Gradients: GP family best (grad NRMSE ≈ 0.5); boosted trees 1.2-5.8. The wave-1 preview (L-BFGS MLP ensembles
  leading among the slow learners at n = 600-1000) does not hold over all n: overall mean rank 17. · `results/E3_learners_report.md`
- 2026-10-11 · **E3 inference (non-SMT, E2a-selected champions)** · Champions picked on E2a: hetgp, catboost,
  mlp_ens[lbfgs], rbf_tps, ppr, ens_press. On 82 problems, Wilcoxon signed-rank with Holm: the GP champion beats
  every other champion. Geometric-mean NRMSE ratio vs catboost 0.45 (cluster-bootstrap CI 0.36–0.57,
  p_Holm = 5e-9), vs networks 0.58 (0.50–0.67), vs rbf_tps 0.49. The Bayesian signed-rank test (ROPE ±10 %) puts
  P(GP better) = 1 for all five pairs. Trees vs RBF and trees vs basis are undecided (p_Holm 0.43 / 0.99). The
  per-problem effects are wide, so P(equivalent) = 0 everywhere: the family matters a lot per problem even where
  the average is even. Performance profiles: GP champion within 1.5× of the best in 83 % of cells and within 2× in
  92 % (trees 45 % / 55 %). Networks are worst at m = 5–10 and second at m = 100–300. GP fits used E3's 1 restart
  (N-14), so these margins are conservative. · `results/E3_stats_report.md`,
  `results/figures/E3_performance_profiles.png`

- 2026-10-25 · **E3 + E3x complete (D-9; 169 problems, 4 647 complete cells at n ≤ 1000, 38 methods)** · No
  universal winner: 37 of 38 methods are best in some cell. Held-out selection (picked on seeds 0-4, scored on 5-9),
  geometric mean × per-cell oracle / within 2× / worst decile: single best (pce_lars) 2.79 / 65 % / 12.1×; best per
  nominal category (26) 1.65 / 86 % / 2.8×; best per data-driven type (26) 1.42 / 91 % / 1.9×; best per problem
  1.19 / 95 % / 1.5×. Three coarse data-driven types (ARI vs categories 0.08): rugged/discontinuous/multimodal (71
  problems, boosted trees: catboost 1.25×), polynomial-like (24, pce_lars 1.67×, rsm2 2.09×), smooth/kinked/ridge (74,
  smt_kpls 1.91×, hetgp/gp 1.97×); every type spans d = 2-10. Method groups (11, ARI vs families 0.47) cut across
  families: ok_variogram with idw/knn/lipschitz, smt_kpls with the RBF/MLS interpolators, pce_lars with krr/svr.
  At n ≤ 500 with smt_krg (3 956 cells): wins pce_lars 705, smt_krg 512, as_gp 293, rsm2 276, catboost 224; smt_krg
  has the best worst decile (4.5× vs pce_lars 11.1×). The top of the table depends on the suite (E3 core: GP/kriging;
  E3x classic functions: PCE/RSM), but see the next entry: the overall ordering is stable (ρ = 0.89).
  The selection numbers here are in-sample over problems; the LOPO versions are in the next entry.
  · `results/E3_taxonomy.md`
- 2026-10-25 · **E3 + E3x selection inference** (`code/analysis/e3_selection_stats.py`; unit = problem, 169;
  leave-one-problem-out picks on seeds 0-4, scored on seeds 5-9). (1) Kendall's W of method rankings across
  problems 0.37 (CI 0.34–0.42): rankings agree only weakly. Within data-driven types: polynomial-like 0.82,
  smooth/kinked 0.64, rugged 0.39 (no more concordant than overall). (2) Geometric mean × oracle (cell-weighted) /
  worst problem decile: single best 2.79 / 15.5×; nominal category 2.07 / 9.6×; 3 data-driven types 2.36 / 33×;
  26 data-driven types 1.61 / 3.0×; per problem 1.19 / 1.5×. With Holm, every policy beats the single best
  (p ≤ 4e-6). 26 types beat nominal categories (×1.41, CI 1.11–1.88, p_Holm 0.004). 3 types do not beat nominal
  categories (p 0.40), and their misassigned problems are costly. (3) Permutation test, 3 types vs 500 equal-size
  random partitions: gain ×1.06 vs 95th percentile ×1.03, p = 0.012: the structure is real but coarse types give
  little. (4) E3 vs E3x rankings: Spearman 0.89 (0.81–0.95), so the ordering is stable but the winner moves.
  pce_lars goes 10th → 1st (p_Holm 6e-4); hetgp and gp do worse on E3x (p_Holm 0.005, 0.025). At n ≤ 500, smt_krg
  is 1st on E3 and 2nd on E3x. E2a-champion tests on all problems: the GP champion (now smt_krg) beats every other
  champion (ratio 0.37–0.61, p_Holm ≤ 2e-19). · `results/E3_selection_stats.md`, `results/E3_stats_report.md`
### Deviations from the pre-registered design (STUDY_PLAN principle 5)
- **D-1 · 2026-10-09 · E3 · `smt_krg` capped at n ≤ 500** (the other O(n³) learners keep n ≤ 1000, CAP1000). On
  Kaggle every `smt_krg` fit at n = 1000 hit the 900 s per-task cap (131/131, SMT track wave 1), so the 2 700
  n > 500 tasks would have cost ≈ 675 CPU-h and recorded only timeouts. `smt_kpls` (same model family, PLS-reduced
  hyper-parameters) keeps n ≤ 1000 (≈ 210 s per fit at n = 1000). The 131 timeout records stay in the data and are
  reported as "exceeds 900 s at n = 1000". Effect on conclusions: the large-n comparison of the GP family rests on
  `gp`, `gp_se`, `hetgp` and `smt_kpls`. Approved by the author. · `code/surr/runner.py` (CAP500)
- **D-2 · 2026-10-11 · E3 · six E2a-cut methods reinstated:** `idw`, `knn`, `qrf`, `ngboost`, `mc_dropout`,
  `poly_ridge` (E2a configs unchanged; 146 780 tasks, ≈ 80 CPU-h). The study is a review and survey of method
  classes, not only a horse race, and these were cut at E2a because their family was already represented, not on
  accuracy. QRF and NGBoost each won 7–8 E2a cells. `mc_dropout` gradients fixed first (BUGS N-19); its earlier
  E2a gradient metric is invalid. Approved by the author.
- **D-3 · 2026-10-11 · E3 analysis · method-level results lead.** Family claims use champions selected on E2a
  (independent data). Per-category "best method" picks are selected on seeds 0–4 and reported on held-out seeds
  5–9. Pairwise tests: Wilcoxon signed-rank with Holm (as in the thesis); effect sizes are geometric-mean ratios
  with a cluster bootstrap over problems; Bayesian signed-rank with ±10 % ROPE; Dolan–Moré performance profiles.
- **D-4 · 2026-10-11 · survey extension and symmetric tuning arm.** Twelve method classes added
  (`learners/survey.py`): GAM, tensor P-spline, symbolic regression (gplearn), sparse GP (SGPR), warped GP, 2-layer
  deep GP, last-layer-Laplace BNN, mean-field VI BNN (textbook ELBO and tempered KL weights 0.1 / 0.01, because the
  untempered ELBO underfits at these n), BART (stochtree), Gaussian-gated mixture of experts, natural-neighbour
  (Sibson) interpolation, and AAA rational (1-D, studied in E6). They are screened in E2b on the exact E2a grid
  (9 360 fits, laptop), then go to E3. E3s_tuning gives the leading trees a 30-trial random search and the MLP
  ensemble 10 members on seeds 0–4, the counterpart of E3s_gp_restarts, so no family is compared at a budget only
  it received. TabPFN stays blocked by its licence (BUGS N-03). Approved by the author.
- **D-5 · 2026-10-11 · new study E12 (rational surrogates, H9).** Established: `paaa` (scattered-data p-AAA),
  `rat_ls` (LS rational + Sanathanan–Koerner). New pairings, with no prior use found (novelty note:
  `data/literature/notes/aaa_multivariate.md`): `ridge_aaa`, `additive_aaa` (spline/rational hybrid), `gp_ratmean`.
  Univariate `aaa_rational` on new 1-D problems. New block P-RAT (10 problems; outside core_set, so E3 is
  unchanged). 35 280 fits, local after E2b. Run walls are set by cost, not by expected failure: cheap runs where a
  method should fail are kept as evidence (author's rule).
- **D-6 · 2026-10-11 · E12 robust-rational variants (before E12 runs).** Five 1-D profile methods shared by all
  pairings: hold-out AAA with pole/spike rejection; residue-based Froissart cleanup (detect and remove, LS refit;
  Nakatsukasa et al. 2018); SVD numerical-rank stopping (robust-Padé idea, Gonnet et al. 2013); continuum AAA on a
  1-D GP mean (Driscoll et al. 2024); Floater–Hormann (provably pole-free). `rat_ls` gains a Tikhonov denominator
  (q = 1 + Σ b_k φ_k, degree-weighted penalty, λ on hold-out) and a positive-Bernstein denominator (provably
  pole-free on the box). E12 = 32 configs, 80 640 fits. Approved by the author.
- **D-7 · 2026-10-11 · planned methods implemented for fixed designs** (`learners/latent.py`): Kernel-Flows GP,
  low-rank CP (the scattered-data functional SVD), functional tensor train by ALS, static SINDy autoencoder. ALS needs
  an explicit intercept and a warm-started final refit (non-convex). Smoke test: CP beats the GP 8× (d = 5) and 28×
  (d = 10) on low-rank products; Kernel Flows recover nonstationary_d2 where ML-II fails (0.065 vs 1.0). Added to E2b
  (now 14 400 fits). E2a/E2b screening is a fairness and smoke check only: its 12 problems favour GP and tree
  families and have no polynomial, additive, low-rank or network-native case. No method is cut on E2a/E2b; every
  method goes to E3's 82-problem core set. Still planned, needing their own designs or data: Smolyak, TT-cross
  (sampling/upscaling studies), Sobolev MLP (gradient data; with gradient-enhanced kriging).
- **D-8 · 2026-10-13 · headline analysis = selection, with data-driven groupings (author's direction).** The paper's
  claim is that no surrogate wins everywhere but the winner is predictable: a guide to which method when and the
  precursor to a feature-based selector (E10), uniting methods from different fields. Primary E3 results are now:
  (1) the selection gap: single best method vs best per problem type vs best per problem vs the per-cell oracle,
  with picks learned on seeds 0–4 and scored on seeds 5–9; (2) method groups and problem types learned from the
  results (`code/analysis/e3_taxonomy.py`), not the a-priori families and categories. The E2a-champion family
  comparison is demoted to a secondary result, and E2a/E2b are used only as screening, never for inference.
  First numbers (n ≤ 1000, 2 229 cells): 38 of 39 methods win somewhere. Single best is 2.56× the oracle (geometric
  mean), best per type 1.49×, best per problem 1.18×. Data-driven method groups agree only partly with the nominal
  families (adjusted Rand 0.39): KAN, NGBoost and QRF behave like trees; SMT-KPLS and ordinary kriging behave like
  the RBF/kernel interpolators, not the GP core; MARS pairs with PCE. Data-driven problem types barely follow the
  nominal categories (adjusted Rand 0.08), yet at equal granularity (24) they select as well (1.49×) with
  lighter tails (worst decile 1.9× vs 2.4×). E10's selector should therefore target the data-driven types or the
  winners directly, from features measured on early samples.
- **D-9 · 2026-10-13 · E3x: every problem E3's core set left out, at the full E3 design (author: "add them all
  back in").** 89 problems: the 11 designed P-FEAT probes at d = 10 (nonstationary, regime, tree/nn draws, pole-knee,
  plateau-bump, oblique-step, multiscale, low-rank product, monotone, quadratic), the 68 remaining VLSE benchmark
  functions (20 2-D plus the scalable families at d = 3/5/8/10) and the 10 P-RAT problems. Same learners as E3;
  1 057 520 fits (~2 000 CPU-h, ~5 days on Kaggle). A separate experiment, so E3's finished tasks and the sensitivity
  runs derived from them are unchanged. The analysis reads E3 and E3x together. Queue: GP restarts → E3x → tuning.

### Smoke-test signals (1 seed, small n — direction only, not results)
- 2026-10-11 · **Rational surrogates (E12 smoke, seed 0).** Clean data: `rat_ls`/`paaa` reach 1e-15 on pole/knee
  and ratio problems (GP 1e-2..1e-3); 1-D raw AAA 1e-11..1e-16. At 1 % noise raw AAA collapses (0.09–0.28) while
  `rat_ls` stays 5–17× ahead of the GP on rational structure (2–4e-3 vs 2–4e-2; resonance 2e-3 vs 0.79). Hold-out
  AAA beats the GP 7× on near_pole at 1 % noise. GP + continuum AAA ≈ GP (a rational compression of the GP, not a gain).
  The Tikhonov denominator stops `rat_ls` blow-ups (additive_d5: 24 → 0.68). The positive-Bernstein denominator fails
  on resonance2d (6.2): positive Bernstein coefficients are sufficient, not necessary, for q > 0, and exclude
  denominators with a near-zero minimum. The GP with a rational mean beats the GP 6× on ridge_pole_d5 at 1 % noise.
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
