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
- 2026-10-26 · **E12 complete (H9; 80 640 tasks, 69 049 ok fits, 252 cells)** · Non-ok: 11 520 by design
  (univariate AAA on d > 1) + 71 timeouts/crashes (0.1 %). Best rational config vs best reference learner per cell
  (share won / geometric-mean ratio): rational-structure problems at noise 0, 0.1 %, 1 %, 10 %: 88 % / ≈0
  (machine precision), 81 % / 0.16, 83 % / 0.43, 67 % / 0.85; controls without rational structure: 52 % / 0.05
  (clean only), then 38 % / 1.21, 19 % / 1.28, 5 % / 1.25. So rational surrogates win where the structure exists
  and keep part of the margin up to 10 % noise; on other functions they help only on clean data. Workhorse:
  least-squares rational `rat_ls` (most wins, 53 cells); Tikhonov-regularised `rat_ls[reg=tik]` is the most
  noise-robust on rational problems (3.2–3.4× the per-cell best at 0.1–1 % noise vs gp 10.8 / 4.0). Univariate
  AAA collapses with noise (18× → 164× the best), but the robust profiles recover most of it (continuum AAA 1.5×,
  Floater–Hormann 3.2× at 10 %). The novel pairings track their parents: additive-AAA ≈ GAM, GP with an AAA mean
  ≈ GP (2.5× vs 2.1× at 10 %), ridge-AAA worse. H9 supported for the established methods, not for the pairings.
  · `results/E12_report.md`
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
- **D-10 · 2026-10-25 · later studies coded; Kaggle stays serial (author's call).** E9 (`E9_optim`, 24 700 tasks):
  16 surrogates spanning the E3 behaviour groups + the true function × 17 optimizers (9 gradient methods twice:
  finite differences inside the budget, and the surrogate's own gradient) × 38 problems spanning the three
  data-driven types × shifted instances 0-9 (the plan said 30; 10 for cost, ~400 CPU-h) × n ∈ {10d, 50d} × noise
  {0, 1 %}. One budget of 500·d surrogate evaluations, with the incumbent scored on the true function at 10d … 500d
  (anytime regret; a run that overshoots its budget is scored at the budget). 13 of the 380 instances have an
  unconfirmed f* (michal_d5, griewank_d5, borehole, tree_draw_d5, pole_knee_d5 seeds); flagged per record. E4:
  `E4_mechanisms` (30 400 fits; Grinsztajn smoothing / rotation / uninformative inputs + GP bound checks) and
  `E4_rair` (7 380 fits; biased / inverse-frequency weighted bootstrap / engine-balanced, paired splits) run on the
  laptop (R-AIR is real data and never leaves it). E10 prototype: `code/analysis/e10_selector.py`. Kaggle order:
  E12 → E9 → tuning, one track at a time. Laptop order (`code/cluster/local_queue.py`): R-AIR → E10 features → E4.
- **D-11 · 2026-10-25 · E13, E6, E8, E5 coded and queued (Kaggle, after the tuning check, serially).**
  E13 (`E13_gradient`, 23 760 tasks): new learners `gek`, `hermite_rbf`, `ge_pce`, `sobolev_mlp`
  (`code/surr/learners/gradient.py`; analytic gradients and derivative covariances checked against finite
  differences in `code/tests/test_gradient_learners.py`) vs the same methods without derivatives + gp + pce_lars at
  cost_mult ∈ {1, 2, d + 1}; 24 problems, 5 seeds (plan: 10; cost). Gradient noise is independent of value noise at
  the same relative level. E6 (`E6_mf`, 14 580): 8 MF methods + HF-only at matched cost on all 27 P-MF pairs, nested
  Sobol' designs; the plan's r × κ × φ grid is represented by the constructed linear (r 0.95) / weak (0.5) /
  nonlinear / shifted pairs rather than a continuous sweep. E8 (`E8_upscale`, 7 560 tasks; one intermediate fit +
  5 upscalers + 6 optimizers each) on d = 2 and 5 (tensor grids are infeasible above; recorded). E5 (`E5_sampling`,
  5 720): 13 scouts × 22 problems × noise {0, 1 %} × 10 seeds (plan: 30), batch of d points per scout refit
  (kriging-believer fill), deliverables fitted at 10d / 20d / 50d. Estimated ~300 CPU-h in total (~1.5 waves).
  Analyses ready: `code/analysis/e9_analysis.py`, `e12_analysis.py`.
- **D-12 · 2026-10-26 · E11 on R-FEA coded (`E11_fea`, 11 070 local fits).** The thesis cache's 623 physical runs
  (converged, not floor-bound; read in place from COMPASS, read-only). Every E3 learner config × use of the free
  scalar model {direct: log t_fem; tmf: log TMF, rebuilt as TMF·t_scalar (the thesis bridge); aug: t_scalar as a
  4th input} × n_train {60, 140, all ≈ 498} × 30 paired 80/20 splits; metric = relative RMSE of t_fem (thesis
  metric). Replaces the plan's nested CV (the learners tune internally). Not yet coded for E11: retrospective active
  learning on the cache and the optimizer proxy (rank of the chosen design among held-out runs). Laptop order: R-AIR
  → E11 → E10 features → E4 mechanisms. The local queue now has a 70-min watchdog: a runner stalled at start under
  memory pressure for 21 h on 2026-10-25 (it checks --max-hours only when a task finishes).
  Added the same day: the optimizer proxy for real data (Kendall τ on held-out runs, true percentile of the run the
  surrogate would pick, top-10 % overlap) in every E11 record, and `E11_al` (180 local tasks): retrospective active
  learning with the cache's training part as the pool, 9 scouts × 20 splits, 5 deliverables at 30 / 60 / 140 runs.
- **D-13 · 2026-10-26 · E14 added: self-designing surrogates (the coverage matrix's last planned methods).**
  Smolyak (combination technique, nested Clenshaw–Curtis, barycentric interpolation) and TT-cross (alternating
  maxvol cross on a Chebyshev grid, rank and grid sized to the budget) in `code/surr/selfdesign.py`, tested in
  `code/tests/test_selfdesign.py` (Smolyak exact on low-degree polynomials and convergent; TT-cross recovers a rank-3
  function to 1e-15). They choose their own points, so `E14_selfdesign` (4 704 tasks) compares them with gp,
  pce_lars, catboost and rbf_tps on a Sobol' design of the same budget {50, 200, 1000}. Smolyak uses only the
  largest level that fits (200 → 61 points at d = 5); records keep the points used. Kaggle, after E5.
- **D-14 · 2026-10-26 · the coverage matrix's remaining planned methods.** (a) Upscalers `smolyak_grid` and
  `tt_grid` (the self-designing interpolants applied to the intermediate model, ≤ 20 000 model evaluations): the
  first upscalers feasible above d = 5, so E8 gains 4 problems at d = 10 (n = 20d, 5 instances: one d = 10 task
  took 45 min on the laptop; 7 840 tasks in all, 2 h per-task cap). (b) E9b
  (`E9b_managed`, 4 560 tasks, `code/surr/managed.py`, tests `code/tests/test_managed.py`): EGO, TuRBO-1, DYCORS and
  trust-region model management query the TRUE function, budget 50·d, plus random and Sobol' baselines, on the E9
  problems × 10 instances × noise {0, 1 %}; the incumbent is the best OBSERVED point. GP hyper-parameters are
  re-optimised every 10 evaluations (cost control; the acquisition rules are unchanged). The E9 report's new section
  7 compares managed loops, fit-once-then-optimize and optimizers on f at matched true-evaluation cost (H7).
  (c) `mf_nn` (Meng & Karniadakis 2020) joins E6 (now 16 200 tasks); its L2 weight on the nonlinear map is chosen by
  5-fold CV on the HF points (a fixed 1e-2 forced a linear map and underfit the training data, 29 % NRMSE).
  Still planned, not coded: upscalers MBA and THB (multilevel / truncated hierarchical B-splines).
  Kaggle order: E9 → tuning → E13 → E6 → E8 → E5 → E14 → E9b.
- **D-15 · 2026-10-31 · E8 extended before it runs (author's suggestions).** (a) IPOPT with the surrogate gradient
  joins E8's optimizers, so E8 reproduces the thesis pipeline end to end (intermediate model → upscaled surrogate →
  IPOPT). (b) Two stochastic upscalers (`code/surr/upscalers.py`; tests `code/tests/test_stochastic_upscalers.py`):
  `pou_quadratics`, a partition of unity of K local quadratics fitted to the intermediate's predictions (smooth,
  exact gradients; a stochastic gradient draws one local quadratic by its weight at x), and `subsample_ensemble`,
  the mean of 8 cubic-RBF members each fitted to a random half of the training data (a stochastic gradient uses one
  member: unbiased for the mean gradient, its noise = model uncertainty — the plan's E9 arm (b)). (c) A stochastic
  arm: Adam, SGD-momentum and Robbins–Monro run with the stochastic gradient ("stoch") next to the same methods
  with the exact surrogate gradient ("model"), so the effect of gradient noise on finding the global optimum is
  measured directly. `optimizers.run` gained a `stochastic_grad` pass-through (default path unchanged; tests
  re-run).
- **D-16 · 2026-10-31 · E15 added: the surrogate–optimizer interface (closes a class of gaps, author's
  direction).** The literature review found the ensemble/stochastic gap to be one case of a wider blind spot:
  surrogates are compared by accuracy, rarely by how the optimizer consumes the trained model. E15
  (`E15_coupling`, 8 360 tasks; `code/surr/coupling.py`, tests `code/tests/test_coupling.py`) fits one surrogate
  and applies every coupling strategy at 500·d surrogate evaluations: presentation (mean + L-BFGS-B / CMA-ES,
  Thompson sampling over members, member-sampled SGD, Gaussian-smoothing Adam, Langevin control, consensus-based
  optimization), exact structure-exploiting optimization (MILP over RF / XGBoost / LightGBM, HiGHS; exact against a
  dense grid in the tests), and exploitation/validity (distance-guarded objective, pessimistic μ + σ), each scored by
  true regret and the optimizer's curse (surrogate − true value at the pick). New intermediate models:
  `ens_optw` (CV-optimised weights, Acar & Rais-Rohani 2009) and `mlp_rpf` (randomised prior functions, Osband et
  al. 2018) in `code/surr/learners/ensembles.py`. E9b gains bootstrapped-kriging EGO (Kleijnen et al. 2012) at
  d ≤ 5, instances 0-4 (B = 10 GP refits per evaluation). Not covered (no solver here, recorded as planned): MILP
  for ReLU networks and deterministic global optimization of GPs (MAiNGO). Kaggle: after E9b. *Superseded by D-17.*
- **D-17 · 2026-11-01 · E15 exact optimization extended to ReLU networks and GPs (author's question "can we write
  them?").** Both now run with open-source tools only. (a) `mlp_relu`: a ReLU deep ensemble (d→32→32→1, K = 5,
  bootstrap as `mlp_ens`; narrower than `mlp_ens` so the program stays solvable; the activation and width differ
  and are reported). `coupling.milp_relu` writes the big-M encoding of Fischetti & Jo (2018), with interval-arithmetic
  pre-activation bounds and stable units fixed or linear, and solves it with HiGHS (120 s limit). The tests check that
  the program's objective equals the network's prediction at the recovered point, and that the point is no worse
  than a 401² grid. (b) `coupling.gp_bnb`: spatial branch-and-bound on the GP mean (Matérn-3/2, Matérn-5/2, SE; the
  sklearn `gp`/`gp_se` learners). Each box takes the larger of two lower bounds:
  - an interval bound on each kernel term (Franey, Ranjan & Chipman 2011);
  - a centred RKHS bound, μ(c) − Σ|∂_jμ(c)| h_j − ‖μ‖_H √N(ρ), with N(r) = c[2 − 2κ(r) + 2rκ′(r) − κ″(0)r²].

  The interval bound alone failed to certify Branin at n = 40 (gap 75 after 2·10⁵ nodes); with the centred bound it
  certified in 354 nodes. The tests check that N is increasing, the analytic gradient, and the centred bound against
  sampled minima on 300 random boxes. Limits are 120 s and 2·10⁵ nodes, and the remaining certified gap is recorded.
  Timings (laptop):

  | Cell | GP branch-and-bound | ReLU MILP |
  |---|---|---|
  | levy_d5, n = 50 | certified, 28 s | gap 9·10⁻⁵, 55 s |
  | levy_d5, n = 250 | gap 1.35 at the node cap | gap 0.22 at the time limit |
  | smooth_trig_d10, n = 100 | certified, 30 s | 6.5 s |
  | ackley_d10, n = 500 | gap 21 at the node cap | no incumbent in 120 s, recorded as `no_point` |

  Exact strategies now always record solver status, even when they return no point. Before this change a stale
  status from an earlier call could carry over. E15 is now 12 surrogates, 9 120 tasks. `e15_analysis.py` §4 reports
  the certified share and compares certified with uncertified optima. The paper's §6b no longer lists these methods
  as omissions. Still not reimplemented: Anderson et al. (2020) ideal formulations, and McCormick bounds on the GP
  variance and acquisition functions (Schweidtmann et al. 2021).

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
- **D-18 · 2026-11-01 · E8 cost: Smolyak predict cache (results unchanged).** An E8 smoke test of levy_d5 + CatBoost
  took 7 695 s on the laptop. `code/analysis/e8_profile.py` traced 4 328 s of it to the Smolyak upscaler. Its
  predict rebuilt the same 1-D barycentric matrices for each of the 771 combination-technique terms, and each
  finite-difference gradient costs 2d predicts. The matrices are now cached per (axis, level) within each call.
  `tests/test_selfdesign.py` checks the result is bit-identical to the old code (kept inline) on 15 cases at
  d = 2, 5, 10. A single-point predict at d = 5 fell from 94 to 13 ms, and the Smolyak part of the cell from
  4 328 s to 567 s, so the cell now takes about 11 min. E8 is now about 5 Kaggle waves instead of about 24. E14
  results are unaffected. E8 records now include per-optimizer wall time (`t_opt`), which measures time only.
- **D-19 · 2026-11-05 · E9x: E9 widened to all 171 E3/E3x problems (reduced design queued; full design not
  queued).**
  - **Why.** E9 ran 38 of the 171 accuracy problems: 25 E3 core and 13 E3x. It used none of the 10 rational
    (P-RAT) problems and none of the rational learners. That is too few problems to train the selector (E10) on
    optimization outcomes. It also leaves open whether E12's accuracy advantage for rational structure carries over
    to optimization.
  - **Design.** `E9x_optim` (`code/surr/runner.py: exp_E9x_optim`, `e9x_problems`) covers the 133 problems E9 left
    out. These are 47 P-FEAT, 43 P-VLSEd, 25 P-VLSE2, 10 P-RAT, 5 P-THESIS and 3 P-ENG, with d = 1–13. Learners are
    E9's 16 plus `rat_ls` and `rat_ls[reg=tik]`, the two most robust rational learners in E12.
  - **Reduced design (queued).** Shifted instances 0–4, n ∈ {10d, 50d}, noise-free only, with the E9 optimizers,
    budget and checkpoints. That is 24 595 tasks, Kaggle track `e9x`, after E15. All 133 problems have f* for
    instances 0–4; 122 have all five confirmed, and the analysis excludes unconfirmed instances as in E9.
  - **CMA-ES at d = 1.** The `cma` package does not run 1-D problems, so on the 5 one-dimensional problems CMA-ES
    is recorded as an error. Every other optimizer runs (smoke-tested).
  - **Analysis.** `e9_analysis.py E9x` reports E9x alone. `e9_analysis.py all` reports all 171 problems on the
    matched design (E9 restricted to instances 0–4, noise-free). The default E9 report now merges the E9_ipopt
    re-runs; its numbers are otherwise unchanged (diff-checked).
  - **Full design (NOT queued; possible later run).** Instances 0–9 × n ∈ {10d, 50d} × noise {none, additive 1 %},
    as E9. That is about 1 330 × (1 + 4 × 18) ≈ 97 000 tasks, roughly four times E9 on Kaggle. Run it only if
    E9x's noise-free results leave the noise question open for the new problems. To queue it, widen
    `exp_E9x_optim`'s loops into a separate experiment name, so the reduced task keys stay finished.
- **D-20 · 2026-11-05 · E9 cost accounting made explicit (analysis only; no re-runs).** Prompted by the author's
  question: is PSO's first place at 500·d only an effect of the budget?
  - **Scoring is fair at every checkpoint.** `optimizers._Counter` records the incumbent at the moment the
    evaluation count reaches 10d … 500d. Several optimizers expose only iteration limits and run past the cap: on
    the surrogates, IPOPT with finite differences has a median 2.6× and a maximum 10.8× the budget, and IPOPT with
    the model gradient 45 % of runs past the cap. Trust-constr, DIRECT and CMA-ES do the same. This wastes wall time
    but cannot improve a score. Methods that stop early keep their final incumbent.
  - **Model gradients are not counted.** Gradient calls in the `|model` arm are free. L-BFGS-B, SLSQP and basin
    hopping with model gradients use a median 3–6 % of the evaluation budget, so their counts understate their
    cost. E13 covers gradient cost.
  - **Ranking depends on budget.** `e9_analysis.py` §8 reports success at 10d / 50d / 500d with budget used and
    wall time.

    | Setting | Top at 10d (low budget) | Top at 500d |
    |---|---|---|
    | On the true function | L-BFGS-B and SLSQP with model gradients, 0.65 (PSO 0.31) | PSO 0.87, CMA-ES 0.82 |
    | On a surrogate | SLSQP and L-BFGS-B with model gradients, 0.33 at 10d, ≈ their 500d value (PSO 0.18) | PSO 0.35 |

    A single full-budget ranking is therefore not a statement of efficiency. The paper reports the curves.
- **R-E3s/E13 · 2026-11-07 · Results: the tuning sensitivity run and E13 are complete** (`code/analysis/e3s_tuning.py`,
  `e13_analysis.py`; reports `results/E3s_tuning_report.md`, `results/E13_report.md`).
  - **Tuning (36 090 cells paired with their E3 twins).**

    | Learner | Error ratio, tuned / E3 | Cells improved | Fit time |
    |---|---|---|---|
    | CatBoost, 30-trial random search | 0.89 | 75 % | ×86 |
    | XGBoost, 30-trial random search | 0.92 | 76 % | ×16 |
    | MLP ensemble, 10 members | 0.95 | 71 % | ×1.8 |

    All three gains are significant over problems (p < 1e-11). Distance to the per-cell best moves only from
    7.2 to 6.4 (CatBoost), 8.5 to 7.8 (XGBoost) and 5.3 to 5.0 (MLP). The share of cells won against the best
    GP-family config rises by only 1–2 points, to 15 %, 11 % and 12 %. E3's ranking is therefore not an artefact of
    under-tuning the trees and networks. This is the symmetric counterpart of the GP-restart sensitivity run.
  - **E13 (gradients at matched cost).**
    - *Same points:* gradients cut error for every method; the geometric-mean ratio is 0.11 (GE-PCE) to 0.58
      (Sobolev MLP).
    - *At adjoint cost:* with the same method given 2n value-only points, GEK wins 72 % of cells (median ratio
      0.61), Hermite RBF 58 % and Sobolev MLP 70 %; GE-PCE is a wash (52 %).
    - *At finite-difference cost ((d+1)n points):* the gradient version loses for GE-PCE, GEK and Hermite RBF
      (wins in 15–37 % of cells). Sobolev MLP is a wash.
    - *Across learners at 2× cost:* the best gradient learner beats the best value-only one in 56 % of cells. The
      gain is largest noise-free with larger designs (geometric-mean ratio 0.05–0.07, dominated by exactly
      recovered polynomial problems), and small with 1–10 % noise (0.67–0.82).
    - *Where gradients hurt:* rugged problems (Ackley, Levy, Rastrigin, Griewank).
    - *Conclusion:* gradients are worth having when they are adjoint-cheap and the response is smooth. Finite-
      difference gradients are better spent as extra samples.
- **R-E5/E6/E14 · 2026-11-09 · Results: E5, E6 and E14 are complete; E8 is re-queued** (`code/analysis/e5_analysis.py`,
  `e6_analysis.py`, `e14_analysis.py`; reports `results/E5_report.md`, `E6_report.md`, `E14_report.md`).
  - **E8 ran nothing:** all 7 840 tasks errored because the f\* table was not shipped (BUGS N-24). It is re-queued
    after E9b and the E6 re-run.
  - **E6 (multi-fidelity, 16 011 fits).**
    - *Low-fidelity data for free:* with the same HF points, the best MF methods cut error to 0.34–0.37× (AR(1)
      co-kriging, NARGP) and win 83–85 % of cells.
    - *At matched cost:* against the low-fidelity budget spent on extra HF points, no single method wins
      significantly. Co-kriging gives 0.91×, winning 52 %. The best MF method per cell (an oracle) gives 0.71×,
      winning 63 %.
    - *Which pairs pay:* MF wins on linear and shifted pairs (0.4–0.5× for the best method) and loses on every
      nonlinear pair (2.4× or worse). Weak pairs (r = 0.5) are a wash. Matched HF-only is the best choice on 12 of
      27 pairs.
    - *Correlation and data:* a higher low/high correlation r predicts a larger gain, but only moderately
      (Spearman −0.2 to −0.56). MF pays more with n_L / n_H = 4 than with 16 at κ = 10.
    - *Not reliable:* bridge_mult, mf_nn and space_mapping are worse than HF-only at matched cost (3.8–8×).
    - *Bug:* `feature_aug` ran as NARGP (BUGS N-27); its 1 620 tasks are re-queued as track e6fa.
  - **E5 (adaptive sampling, 5 720 designs × 6 deliverables × 3 sizes).**
    - *Only space-filling-like rules beat Sobol', and only slightly:* LOLA-Voronoi 0.91× (p = 0.04), GP IMSE
      0.95×, MEPE 0.97×, sequential maximin 0.97×.
    - *Pure variance scouts lose:* GP max-variance 1.08×, and the tree/boosting committees 1.56–1.73× (they cluster
      points). Random sampling 1.17×.
    - *Matching the deliverable to the scout:* it matters for non-GP scouts. An MLP-ensemble scout gives 0.93× for
      an MLP deliverable but 1.20× for the others; an RF scout gives 1.25× vs 1.70×.
    - *Budget:* the gains of the good scouts grow with budget (LOLA 0.83× at 50d).
    - *Thesis bugs:* this answers B-02 (a GP variance scout is no better than a space-filling design; here it is
      worse) and B-03 (scouts tested beyond the GP deliverable).
    - *Caveat (2026-11-10, BUGS N-28):* E5 scored each batch of d points with one unrefitted scout, so the
      model-based scouts' batches bunched. Their E5 numbers are biased against them; E5x re-runs them with a
      kriging-believer batch. The model-free results (LOLA, maximin, MIPT, random, Sobol') stand.
  - **E14 (self-designing surrogates, 5 040 fits).**
    - *Smolyak sparse grid:* dominant on smooth, low-dimensional, noise-free problems (Branin, Goldstein–Price,
      Rosenbrock, additive_d2: errors of 1e-4 to 1e-5). It loses badly with 1 % noise (3.6–7× the best baseline)
      and on steps and kinks.
    - *TT-cross:* wins on low-rank products (0.01× the best baseline, 61 % of cells) and additive functions, and
      fails elsewhere at small budgets.
    - *Neither replaces a Sobol' design plus a learner:* best per problem is gp 8, smolyak 7, tt_cross 6, pce_lars 4,
      catboost 2, rbf_tps 1.
    - *Selection-guide reading:* both are structure-specific tools. Use them when the structure is known and the
      data are noise-free.
- **D-21 · 2026-11-10 · Literature pass for E5/E6, and the E5x/E6x extensions** (author's request).
  - **What the literature recommends for adaptive global-accuracy sampling.**
    - Fuhg, Fau & Nackenhorst (2021, *Arch. Comput. Methods Eng.* 28, review and benchmark): EIGF and the
      model-free MIPT are reliable and need little tuning. MEPE, SFCVT, MASA, SSA, TEAD and AME reach similar,
      promising results. LOLA and CV-Voronoi converge slowly.
    - Mo et al. (2017, *Water Resour. Res.* 53, TEAD): implemented from the authors' MATLAB.
    - Crombecq et al. (2011, *EJOR* 214, MIPT): sequential space-filling designs as good as the best one-shot
      designs.
    - Toolboxes: BoTorch (`qNegIntegratedPosteriorVariance`) and Emukit (`IntegratedVarianceReduction`,
      `ModelVariance`) ship integrated variance. BoTorch's docs state that a standard GP's posterior variance does not
      depend on y, which is the mechanism behind B-02.
    - Sauer, Gramacy & Higdon (2023, *Technometrics*): with a stationary GP, sequential design can only be
      space-filling; adaptivity needs a nonstationary model (deep GP).
  - **What the literature recommends for multi-fidelity.**
    - Recursive co-kriging (Le Gratiet & Garnier 2014) and NARGP (Perdikaris et al. 2017) are the practical
      defaults. In an independent comparison (Brevault et al. 2020, arXiv 2006.16728), NARGP was more accurate than
      MF-DGP, which has many more hyperparameters. Cutajar et al. (2019) report MF-DGP gains mainly when HF data are
      very scarce.
    - Toal (2015, *SMO* 51): co-kriging needs strongly correlated fidelities; r² > 0.9 is the commonly cited
      threshold, not yet checked in the paper itself.
    - Engineering practice: SMT's MFK; transfer learning of networks.
  - **E5x (2 860 tasks, seeds 0–4):**
    - every model-based E5 scout re-run with the kriging-believer batch (N-28);
    - four methods rated well by Fuhg et al. that E5 lacked: MIPT, TEAD (GP), SFCVT (GP, closed-form LOO), MASA
      (MLP-ensemble committee).
    - The Sobol' control comes from E5's records.
  - **E6x (6 480 tasks, E6's cells):**
    - ar1_rhox (ρ(x) linear, Le Gratiet & Garnier 2014);
    - smt_mfk (SMT's MFK as shipped);
    - mf_transfer_nn (pre-train on LF, fine-tune on HF);
    - cv_select (K-fold CV on the HF points among hf_only, AR1, NARGP, bridge_add, hierarchical kriging).
  - **Toal-style budget rule (from existing records, no new runs):** use co-kriging if the pilot r² ≥ t, else spend
    the LF budget on HF points. Against matched HF-only:
    - t = 0.99: 0.83×, more than 10 % worse in only 4 % of cells;
    - t = 0.9: 0.89×;
    - always co-kriging: 0.91×, but more than 10 % worse in 43 % of cells;
    - the oracle: 0.69×.
    - So a pilot correlation check gives most of the safe gain (`e6_analysis.py` §5).
  - **Not implemented:**
    - deep-GP active learning (Sauer et al. 2023) and MF-DGP (Cutajar et al. 2019): they need the deepgp (R) and
      Emukit/GPy stacks, which are not in the offline Kaggle wheels;
    - hetGP replication designs (Binois et al. 2019): a noisy-sampling question outside E5's design.
  - **Queue:** E9b → E6 feature_aug re-run → E6x → E5x → E8 → E15 → E9x.
