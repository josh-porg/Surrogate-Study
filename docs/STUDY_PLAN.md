# Study Plan — experiments E0–E11

Operational plan for the study part of the paper. Research questions (RQ), hypotheses (H) and frameworks (F) are
defined in STUDY.md; maths in MATHEMATICS.md (M.x); thesis defects in BUGS.md (B-xx). The original study is
Poznański (2026), M.S. thesis, Univ. of Kansas — Sec. 14.3 and App. G (`poznanski2026towards`).

## 0. Principles (apply to every experiment)
1. **Score against clean truth** on synthetic problems (≥10⁴ Sobol' test points); noisy-CV reported only as the
   "practitioner's view" column (fixes B-01).
2. **Common random numbers:** for a given seed every pipeline sees the same initial design, noise draw, test set and
   optimizer starts. **30 seeds** default (20 where a cost cap forces it) (fixes B-08).
3. **Fair tuning:** every learner gets the same hyper-parameter budget (nested CV, fixed number of trials); tuning
   cost is recorded and reported (Lüthen 2021 lesson).
4. **Simple baselines always present:** linear/quadratic RSM, IDW, kNN, plus i.i.d. random *and* Sobol'/maximin designs
   (Lütjens 2024 lesson; fixes B-02).
5. **Pre-registered:** hypotheses H1–H8 are frozen before E3 runs; deviations are logged in STUDY §9.
6. **Primary end-point = true regret at matched simulation cost (M.1, M.18).** RMSE, derivative error, Kendall τ on
   sublevel sets, calibration are secondary.
7. **Reproducible:** every run is a config row → content-hash → result file; resumable, shardable, seeded.

## 0b. Coverage and cost rules (added 2026-10-03)
- **Strength / weakness / blind coverage** (`code/surr/coverage.yaml` → `docs/PROBLEM_MATRIX.md`, enforced by
  `code/surr/check_coverage.py`): every learner, sampler, acquisition rule, multi-fidelity method, upscaler and
  optimizer has at least one problem designed to favour it and one designed to hurt it, stating the mechanism; a
  **blind set** (random composite, oblique step, borehole, OTL, piston, wing weight, VLSE "other", both real datasets)
  has no designed winner, so the results also show what wins where nobody predicted it. Native prior draws (a GP
  draw, a random-tree draw, a random-network draw) give each family a problem from its own prior.
- **Complexity** (MATHEMATICS §11, paper §8b): every method's fit/predict time and memory are tabulated; memory walls
  (exact GP/RBF O(n²), TabPFN O(n²), tensor grids O(mᵈ), Delaunay O(n^⌈d/2⌉)) mark factor cells "infeasible" rather
  than dropping them; measured times accompany asymptotic ones.
- **Shifted instances**: all optimisation and sampling experiments use BBOB-style shifted instances (BUGS N-01).

## 1. Problem suite
| Block | Contents | d | Purpose |
|---|---|---|---|
| P-VLSE2 | the 32 Math 796 HW1 functions (6 VLSE categories) | 2 | continuity with author's prior work; visualisable |
| P-VLSEd | scalable VLSE / BBOB functions (Rosenbrock, Rastrigin, Ackley, Levy, Griewank, Zakharov, Styblinski–Tang, Dixon–Price, sum-squares, Michalewicz, …) | 3, 5, 8, 10 | dimension scaling |
| P-ENG | Borehole (8), OTL circuit (6), Piston (7), Wing weight (10), Currin, Park | 2–10 | engineering-like emulation functions |
| P-THESIS | the 6 thesis canonical surfaces (smooth, collapse_knee, clamp_floor, anisotropic, oscillatory, cliff) | 3 | direct comparison with App. G |
| P-FEAT | constructed feature probes: true discontinuity, plateau/floor, ridge (rotated), additive, low-rank product, Fourier-sparse, wavelet-local, uninformative inputs (+k dummy dims) | 2–10 | isolate one mechanism each (H1, H3, H8) |
| P-MF | Mainini AVT-354 set (Forrester cont./discont., Rosenbrock, shifted-rotated Rastrigin, heterogeneous, spring–mass, Paciorek-noisy) + `mf2` (Branin, Currin, Park, Borehole, Hartmann6, Bohachevsky, Booth, Himmelblau, Six-hump camel) + tunable-r construction (M.2) | 1–10 | RQ3 |
| R-FEA | thesis cache `tmf_bbd96c4b56` (901 runs, 623 usable, 3 inputs) + `t_scalar` LF at every point + 15 coarse-mesh runs; `tmf_dca5926004` (post-Bauschinger) as a second version | 3 | engineering anchor (RQ8) |
| R-AIR | aircraft performance data (Math 796), 5 outputs, stratified by engine type | ~10–20 | second real tabular anchor (RQ8) |
| D-LARGE | **large-n datasets for co-learning (author's note):** analytic problems sampled at n = 10³–10⁵; public regression sets (to verify and select in E7: e.g. NASA airfoil self-noise, combined-cycle power plant, building energy efficiency, concrete strength, OpenML-CTR23 suite) | 5–20 | let F10 learned encoders (DKL, SINDy-AE, KAN, TabPFN) be tested where they *should* work, not only at n≈100 |

Noise models (M §1.1): none · additive · multiplicative · heteroscedastic · heavy-tailed · deterministic-numerical.
Levels: 0, 0.5, 1, 3, 10 % of sd(f).

## 2. Experiments

### E0 — Infrastructure and port validation (prerequisite)
- Build `code/surr/`: problem library (with gradients, known optima, MF variants, noise wrappers), learner registry
  (`fit / predict / predict_std / grad`), samplers & acquisitions, MF wrappers, upscalers, optimizer harness, runner,
  stats/plots. Port COMPASS strategies (read-only source).
- **Gate:** reproduce thesis App. G Tables canonical_a/b and scout_knee with the *original* settings (noisy CV, 8 seeds,
  n = 140, 3 % noise) to within their CIs. If not reproduced, stop and debug before any new result.
- Output: unit tests, timing table per learner × n (feeds the compute plan).

### E1 — Thesis errata (corrected App. G)
- Re-run App. G with fixes: clean-truth scoring (B-01), nested lin/log choice (B-05), 30 seeds (B-08), corrected noise
  estimator on R-FEA (B-06), repeated-CV CIs on the real-cache table (B-11), Sobol'/maximin controls (B-02),
  matched scout/deliverable hyper-parameters (B-12).
- Output: side-by-side "as published / corrected" tables → paper appendix "Revisiting Poznański (2026)".
- **Decision point:** does GP still win every canonical surface once the noise floor is removed?

### E2 — Screening (one-factor-at-a-time around a reference pipeline)
- Reference: Sobol' init → GP-variance scout → GP intermediate → 24-pt/axis cubic grid → IPOPT multistart.
- Vary one stage at a time over its full menu (STUDY §6 S0–S6 + F10) on a 12-problem screening subset × 3 budgets ×
  2 noise levels × 10 seeds.
- Learner-internal factor screened here: the **training optimizer** of network learners (Adam, AdamW, SGD+momentum,
  L-BFGS, Barzilai–Borwein) — smoke test showed L-BFGS 4–17× better than Adam at n ≤ 80.
- **Keep** per stage: top ~5 by primary metric, every thesis method, anything that wins on ≥1 problem class ("surprise
  rule"). Output: shortlist + measured fit/predict times (refines compute budget).

### E3 — Learner study (RQ1, H1)
- Survivors (~12 learners) × P-VLSE2, P-VLSEd, P-ENG, P-THESIS, P-FEAT × d × n ∈ {5d,10d,20d,50d,100d,300d} × noise
  (5 levels, additive; 2 levels × other 4 types) × 30 seeds; one-shot Sobol' designs.
- Metrics: clean NRMSE, sup error, H¹ error, Kendall τ (global and on 10 % sublevel set), NLPD/coverage (UQ learners),
  fit/predict time.
- Analysis: learning curves (log-error vs log-n), **crossover maps** (where GBT overtakes GP in (n, noise, feature)
  space), mixed-effects model (M §9), CD diagrams per problem class.

### E4 — Mechanism tests for the GBT result (H3)
- Grinsztajn diagnostics on P-FEAT, R-FEA, R-AIR: (a) Gaussian-kernel smoothing of the training target at increasing
  length-scales; (b) random input rotations; (c) k added uninformative inputs; plus (d) GP nugget/length-scale
  pathology check (bounds hit) and (e) heavy-tail noise probe (residual kurtosis on R-FEA).
- Output: which mechanism(s) reproduce "GBT > GP" on synthetic data with the same signature as R-FEA.

### E5 — Sampling study: scout × deliverable (RQ2, H2)
- Full factorial scout ∈ {GP-var (refit θ), GP-var (frozen θ), RF-IJ, QRF, GBT committee K∈{4,20}, NGBoost, deep
  ensemble, BART, TabPFN, conformal-wrapped, LOLA-Voronoi, EIGF, MEPE, sequential maximin (model-free), random, Sobol'}
  × deliverable ∈ {GP, GBT, RF, NN ensemble, RBF, P-spline, best F10} on P-THESIS + P-FEAT + 8 VLSE functions, n0 = 2d+2,
  budget to 50d, 30 seeds.
- Report native (diagonal) vs cross (off-diagonal) separately; samples-to-target vs Sobol' control (not only random).
- Retrospective active learning on R-FEA (pool = cache) with the same grid.

### E6 — Multi-fidelity (RQ3, H4)
- MF method ∈ {HF-only, additive/multiplicative/comprehensive bridge, AR1 co-kriging, recursive, hierarchical kriging,
  NARGP, MF deep GP, MF-NN (Meng–Karniadakis), feature-augmented GBT/RF/NN/TabPFN, space mapping} × P-MF ×
  r ∈ {0.5, 0.8, 0.9, 0.95, 0.99} × relation {linear, nonlinear} × κ ∈ {3, 10, 30, 100, 1000} × φ ∈ {0, .1, .25, .5, .8}
  × 30 seeds, at matched cost.
- Output: **error grids** (van Rijn) per learner family; "MF pays" region maps; real R-FEA: HF = t_fem, LF = t_scalar
  (free) and coarse mesh (15 pts) — test whether the thesis' TMF target already captures the MF gain.

### E7 — Change-of-basis / co-learning family (H8)
- F10 methods (PPR, active-subspace GP, polynomial ridge, warped GP, DKL [ML-II full-batch / minibatch / kernel-flow /
  fully Bayesian], static SINDy-AE, CS in Fourier/Chebyshev/Legendre/wavelet, functional SVD, TT-cross, AutoML
  co-search) on P-FEAT (each probe matched to the basis it favours *and* to bases it does not), P-THESIS, R-FEA, and
  **D-LARGE at n = 10³–10⁵** so learned encoders are tested in their intended regime.
- Compression sub-study: randomised SVD / HOSVD / TT on the upscaling grid and Nyström/RFF scouts → speed-vs-error
  curves, d up to 10.

### E8 — Upscaling (RQ4, H5)
- S5 menu (none; grid m ∈ {6,12,24,48}/axis × {linear, cubic tensor, not-a-knot B-spline, smoothing spline}; MBA; THB;
  P-spline direct; Smolyak; Chebyshev; NN/KAN distillation; TT-compressed grid) × intermediate ∈ {GP, GBT, RF, NN,
  best F10} × P-THESIS + 10 P-VLSEd/P-ENG problems at d ∈ {2,3,5,8} × 20 seeds.
- Metrics: value and H¹/H² error added by upscaling (M.15), evaluation cost, memory; and the E9 regret of the result.
- Key question: does smoothing upscaling turn GBT into a good optimizer-facing surrogate?

### E9 — Optimization on the surrogate (RQ5, RQ6, H6, H7)
- Surrogate (E3/E8 survivors incl. upscaled) × optimizer ∈ {IPOPT multistart, SLSQP, L-BFGS-B, trust-constr,
  Nelder–Mead, COBYLA, DIRECT, DE, CMA-ES, PSO, basin hopping} × problems with known optima × budgets × 30 seeds.
- Stochastic-approximation family: Adam, SGD+momentum, Barzilai–Borwein GD, Robbins–Monro (1951),
  Kiefer–Wolfowitz (1952), SPSA (Spall 1992) — run (a) on deterministic surrogates, (b) with *stochastic surrogate
  gradients* (each step uses one randomly drawn ensemble member / posterior sample, so surrogate uncertainty becomes
  gradient noise), and (c) on noisy true functions. All first-order methods normalise f by its spread over 2d+2
  counted samples (automatic gain setting).
- Surrogate-managed loops that may query f: EGO/EI, LCB, TuRBO, SMAC-style RF-EI, DYCORS, trust-region model management.
- Metrics: regret f(x̂) − f*, success rate (regret < ε), distance to x*, evaluations of f and of s, wall time with
  c_H ∈ {1 ms, 0.1 s, 10 s, 1 h}.
- Analysis: rank correlation between RMSE ranking and regret ranking per optimizer class (**H6**); cost crossover map
  (**H7**); surrogate × optimizer × landscape interaction (the coupling the author flagged).

### E10 — Selection model (RQ7)
- Features from the *initial design only*: ELA set (flacco-style), fill distance, estimated smoothness/anisotropy,
  noise-to-signal estimate, LF correlation (if MF), d, n, c_H.
- Learn a recommender (best pipeline / top-k) with leave-problem-out and leave-class-out validation; report regret vs
  single-best pipeline and vs oracle; validate blind on R-FEA and R-AIR.

### E11 — Real-data transfer (RQ8)
- R-FEA and R-AIR: 30× repeated nested CV for all survivors; retrospective AL (E5 grid); MF on R-FEA (E6); E4
  diagnostics; optimizer proxy on R-FEA = rank of the chosen design among held-out runs (no analytic truth).
- Output: does the canonical crossover map predict what happens on real data?

## 3. Dependencies and order
```
E0 ──► E1 ──► E2 ──┬──► E3 ──► E4
                   ├──► E5
                   ├──► E6
                   ├──► E7
                   └──► E8 ──► E9 ──► E10 ──► E11
```
E3–E8 can run in parallel once E2 has fixed the shortlist; E9 needs E3/E8 surrogates; E10 needs E3–E9 results.

## 4. Compute plan
| Exp. | Approx. fits/runs | Est. CPU-h | Where |
|---|---|---|---|
| E0 | timing + reproduction | 2–5 | laptop |
| E1 | ~1×10⁴ | 5–15 | laptop |
| E2 | ~2×10⁵ | 40–80 | laptop or cluster |
| E3 | ~1.5×10⁶ | 500–800 | cluster |
| E4 | ~5×10⁴ | 20–40 | laptop |
| E5 | ~3×10⁵ AL loops (each ~50 refits) | 600–1200 | cluster |
| E6 | ~5×10⁵ | 300–600 | cluster |
| E7 | ~1×10⁵ (DKL/NN heavy; D-LARGE) | 300–800 | cluster (GPU optional) |
| E8 | ~1×10⁵ | 100–200 | cluster |
| E9 | ~5×10⁵ optimizer runs | 300–600 | cluster |
| E10–E11 | small | 10–30 | laptop |
| **Total** | | **~2 200–4 400 CPU-h** | |

*Estimates — replaced by measured timings after E0/E2.* Laptop (12 threads) ≈ 250 CPU-h/day if left running →
E0–E2, E4, E10–E11 locally; the rest on **KU CRC**, `sixhour` partition for many short array tasks (each shard
< 5.5 h, resumable), `math` partition for longer jobs. Runner writes one Parquet file per shard (content-hashed
configs), so jobs can be killed/resubmitted without loss, and results are synced back with `rsync`.

### 4b. Compute strategy without the KU cluster (decided 2026-10-03)
KU CRC access is unlikely for now. Plan, in order of use:

1. **Cut the load first (target: 3–5× fewer CPU-hours).**
   - *Measure, then budget:* E0 timing (all learners × n ∈ 20…2000 × d ∈ {2,5,10}) replaces the estimates above;
     cells past a measured wall are marked infeasible and not run.
   - *Sequential seeds:* 10 seeds first; add seeds (to 30) only for comparisons whose paired CI still straddles
     zero — a standard sequential-testing saving of roughly half the seeds.
   - *Successive halving over learners* within each problem class: after the first two budgets, drop learners that are
     worse than the class leader by > 2× NRMSE on every problem of the class (recorded, so the dropping is reported).
   - *Fractional design for nuisance factors:* full factorial only over learner × problem × n; noise type × level is
     run on a balanced fraction (each noise type at all levels for a Latin subset of problems); the mixed-effects
     model (M §9) estimates the effects from the fraction.
   - *Cheaper inner loops, stated in the paper:* active-learning GPs refit hyper-parameters every 5th step with rank-1
     Cholesky updates between (O(n²) instead of O(n³) per step); 1 optimiser restart inside loops.
   - *One fit, all metrics:* every task returns value, rank, sublevel-set, calibration and gradient metrics at once.
2. **Laptop (default executor):** 10 of 12 threads via `python -m surr.runner <exp> --workers 10`, resumable, so it can
   run overnight and be interrupted. ≈ 240 CPU-h per full day.
3. **Kaggle notebooks (burst capacity, free):** 4 CPU cores / 29 GB RAM per session, 12-h sessions; the runner's
   `--shard i/k` option splits an experiment into independent shards. `code/cluster/kaggle/` packages the code and
   pushes one notebook per shard with the Kaggle API; results come back as JSONL. Needs the author's Kaggle API token
   in `~/.kaggle/kaggle.json` (author action — Claude does not create accounts or handle the token's value).
4. **NSF ACCESS Explore allocation (the real fix):** up to 400 000 ACCESS credits (≈ 334 000 core-hours on Purdue
   Anvil) for a one-paragraph request + CV; a graduate-student PI needs a signed advisor letter. Draft request:
   `docs/ACCESS_EXPLORE_REQUEST.md`. The runner's sharding maps directly onto SLURM job arrays.
5. **Not used:** GitHub Actions (free and large for public repos, but GitHub's terms restrict Actions to building,
   testing and deploying the repository's software — using it as a compute farm risks the account); Colab free tier
   (2 cores, session limits).

**Data rule for any off-machine compute:** synthetic-problem experiments only. The FEA cache (R-FEA) stays on this
machine until the author clears distribution of the missile-duct data (PLAN open question 5).

**Cluster access prerequisites (author, if KU CRC becomes available later):** KU VPN connected (login node `hpc.crc.ku.edu` is not reachable off-campus —
probe timed out 2026-10-03), CRC username, and key-based SSH (`ssh-copy-id`); Claude will not enter passwords or
handle Duo prompts. Then: Python env on the cluster (`module load` + venv from `requirements-lock.txt`), SLURM array
template in `code/cluster/`.

## 5. Outputs → paper
| Paper element | From |
|---|---|
| App.: Revisiting Poznański (2026) — errata tables | E1 |
| Fig.: learning curves & GP/GBT crossover maps | E3 |
| Fig.: mechanism diagnostics (smoothing/rotation/dummy) | E4 |
| Fig.: scout × deliverable heat-map (native vs cross) | E5 |
| Fig.: MF error grids & "MF pays" regions per learner family | E6 |
| Fig.: basis-match matrix (F10 method × feature probe); compression speed–error curves | E7 |
| Fig.: upscaling error decomposition and regret | E8 |
| Fig.: RMSE-rank vs regret-rank scatter; cost crossover | E9 |
| Table/flowchart: selection rules from initial-design features | E10 |
| Section: real-data transfer | E11 |

## 6. Milestones
| # | Milestone | Exit criterion |
|---|---|---|
| M1 | E0 done | thesis tables reproduced within CI; timing table |
| M2 | E1 done | errata tables; decision on GP claim |
| M3 | E2 done | frozen shortlist; frozen hypotheses; measured compute budget |
| M4 | E3–E8 done | all result Parquets; first figures |
| M5 | E9–E11 done | regret analysis; selection model; real-data transfer |
| M6 | Draft v1 | study sections written; review sections tied to results; send to Alonso & Needels |
