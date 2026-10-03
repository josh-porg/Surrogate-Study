# Plan

Project: a journal paper built from the thesis surrogate study (Maters-Project-V3 Appendix "Surrogate Strategy
Comparison" + §12 "Surrogate Modeling Strategy"), extended into a cross-disciplinary review + factorial study.
Interested readers: Dr J. J. Alonso (Stanford), Dr J. Needels (Sandia).

**Rules:** the thesis repo `Maters-Project-V3` is read-only. The thesis code `COMPASS` is read-only too
(we import or copy from it; we never edit it). All new code, data and results live in this repo.

Status keys: `todo` · `doing` · `done` · `blocked`.

## Documents
| Doc | Purpose |
|---|---|
| [PLAN.md](PLAN.md) | phases, tasks, decisions, open questions (this file) |
| [STUDY.md](STUDY.md) | paper content: RQs, hypotheses, frameworks, design, results log, outline |
| [SOURCES.md](SOURCES.md) | literature review and source list (✔ verified / ○ to verify) |
| [BUGS.md](BUGS.md) | thesis methodology defects and new-code bugs |
| [MATHEMATICS.md](MATHEMATICS.md) | formal definitions and derivations (M.x) |

## Phase 0 — Set-up and audit (2026-10-03)
| # | Task | Status |
|---|---|---|
| 0.1 | Locate thesis doc, thesis code, real FEA data | done — data intact in `COMPASS/results/tmf_cache/` |
| 0.2 | Audit thesis surrogate study code vs write-up → BUGS.md | done (first pass, 16 items) |
| 0.3 | Create repo + five documents | done |
| 0.4 | Locate the optimization test surfaces | done — Math 796 HW1 suite (32 VLSE functions) copied to `data/reference/math796_hw1/` |
| 0.5 | Python environment | done — `.venv` (Python 3.13): numpy, scipy, pandas, scikit-learn 1.9, xgboost, lightgbm, torch 2.14 (CPU), gpytorch, smt, emukit, mf2, ngboost, pygam, csaps, tabpfn, pydmd, pysindy, casadi, cma; pinned in `requirements-lock.txt` |
| 0.7 | Add change-of-basis / latent family F10 (user request) to SOURCES §2.7, MATH §3.10, STUDY F10/H8 | done |
| 0.8 | Do NOT open or copy MATLAB DMD / flow-field data in `OneDrive/research`, `OneDrive/MATLAB/DMD` (large; user request) | standing rule |
| 0.6 | Decide compute: laptop for screening; "powerful computer" for Phases 2–6 | todo (ask) |

## Phase 1 — Literature (breadth then depth)
| # | Task | Status |
|---|---|---|
| 1.1 | Breadth pass 1: reviews across disciplines; method taxonomy | done (36 searches) |
| 1.2 | Breadth pass 2 (other fields) | done — SOURCES §1.7b |
| 1.3 | Verify all references, build `paper/refs.bib` | done — 254/254 via `code/lit/verify_refs.py` + audit + `manual_refs.bib`; 0 BibTeX warnings |
| 1.4 | Depth reads | done for 15 open-access papers (`data/literature/notes/DEPTH_NOTES.md`); abstract-level only: Fuhg 2021, Peherstorfer 2018, Toal 2015, TabPFN, Needels (AIAA paywall), Jin 2001 |
| 1.5 | Write review sections | done — `paper/main.tex` + `paper/sections/01–09`, compiles with MiKTeX (29 pp.) |
| 1.6 | Get paywalled PDFs (Needels SciTech 2024 + dissertation, Toal 2015, Jin 2001, Fuhg 2021, Liu–Ong–Cai 2018) via KU library; or ask Dr Needels | todo (author) |
| 1.7 | Second review pass on the draft once study results exist (tie review claims to our findings) | todo |

## Phase 2 — Study code (`code/`)
| # | Task | Status |
|---|---|---|
| 2.1 | Problem library | **done** — `code/surr/problems/`: 32 HW1 VLSE (optima verified numerically, all match), 56 scalable VLSE, 6 engineering, 6 thesis surfaces, 19 probe families × d∈{2,5,10} + dummy-input variants, 27 MF pairs (16 constructed, 11 `mf2`), BBOB-style shifted instances; 161 single-fidelity problems |
| 2.1b | Strength / weakness / blind coverage matrix + checker | **done** — `code/surr/coverage.yaml`, `check_coverage.py` → `docs/PROBLEM_MATRIX.md`; every method has ≥1 designed win and loss, blind set neutral, every probe discriminates |
| 2.2 | Learner registry | **done (41 implemented, 11 planned)** — classical/discontinued (RSM 1951, OK-variogram 1963, GMDH 1968, Shepard 1968, Hardy MQ 1971, Duchon TPS 1977, PPR 1981, MLS 1981, kNN, CART, MARS 1991, Lipschitz, Delaunay), GP family (GP, GP-SE, SMT KRG, SMT KPLS, hetGP, local GP, treed GP, KRR, SVR), trees (RF, ExtraTrees, QRF, XGBoost, LightGBM, CatBoost, NGBoost, GBT committee), nets (deep ensemble, MC dropout, KAN, DKL, TabPFN*), basis (sparse PCE, CS-Fourier, CS-cosine, AS-GP, polynomial ridge), PRESS ensemble. *TabPFN needs licence (N-03) |
| 2.3 | Samplers & acquisitions | **done** — 7 samplers, 11 acquisition rules |
| 2.4 | MF methods | **done (8/9)** — HF-only, multiplicative and additive bridge, recursive AR1 co-kriging, hierarchical kriging, NARGP (augmented), feature-augmented, space mapping; MF-NN planned |
| 2.4c | Remaining planned methods | todo — natural neighbour, Smolyak, BART, sparse GP, TT-cross, functional SVD, static SINDy-AE, kernel-flow GP, warped GP, Sobolev MLP, AAA; MF-NN; upscalers MBA, THB, Smolyak grid, TT grid; optimizer loops EGO, TuRBO, DYCORS, trust-region model management |
| 2.5 | Upscalers (S5) | todo |
| 2.4b | F10 learners: PPR, active-subspace GP, polynomial ridge, warped GP, DKL (ML-II + kernel-flow), static SINDy-AE, CS in Fourier/Chebyshev/Legendre/wavelet, functional SVD, TT-cross; transform × learner co-search | todo |
| 2.5b | Compressed upscaling (randomised HOSVD / TT-SVD) and Nyström/RFF scouts, with speed-vs-error curves | todo |
| 2.5a | Upscalers | **done (4/8)** — none, tensor spline (thesis path), separable smoothing spline, direct P-spline/GAM |
| 2.6 | Optimizer harness (S6) with regret/success/cost logging | **done (11/15)** — IPOPT (CasADi), SLSQP, L-BFGS-B, trust-constr, Nelder–Mead, COBYLA, DIRECT, DE, CMA-ES, PSO, basin hopping; uniform budgeted interface |
| 2.6b | Smoke tests | done — `code/tests/smoke_learners.py`, `smoke_algorithms.py`; outputs in `results/smoke_*.txt` |
| 2.7 | Experiment runner: config-driven, seeded (CRN), resumable, result store (parquet), parallel | todo |
| 2.8 | Statistics & plots: CD diagrams, performance/data profiles, mixed-effects fits | todo |
| 2.9 | Unit tests incl. regression test reproducing thesis canonical numbers with thesis settings (proves port fidelity), then the corrected numbers | todo |

## Phase 3 — Reproduce, then correct, the thesis
| # | Task | Status |
|---|---|---|
| 3.1 | Reproduce thesis Tables canonical_a/b, scout_knee, cv_full with the original settings | todo |
| 3.2 | Re-run with clean-truth scoring (B-01), nested target choice (B-05), 30 seeds (B-08) → errata table | todo |
| 3.3 | Space-filling controls for the scout study (B-02) | todo |
| 3.4 | Real-cache GBT vs GP with repeated CV + CIs (B-11); corrected noise estimate (B-06) | todo |

## Phase 4 — Factorial study (STUDY §7.2 steps 1–8)
| # | Task | Status |
|---|---|---|
| 4.1 | Screening | todo |
| 4.2 | Learner study (RQ1) | todo |
| 4.3 | Sampling: scout × deliverable (RQ2) | todo |
| 4.4 | Multi-fidelity (RQ3) incl. real `t_scalar` and coarse-mesh LF | todo |
| 4.5 | Upscaling (RQ4) | todo |
| 4.6 | Optimizer study (RQ5, RQ6) | todo |
| 4.7 | Selection model from ELA features (RQ7) | todo |
| 4.8 | Real-data transfer and GBT explanation (RQ8, H3) | todo |

## Phase 5 — Writing and submission
| # | Task | Status |
|---|---|---|
| 5.1 | Paper skeleton (LaTeX, AIAA or Elsevier template — decide venue) | todo |
| 5.2 | Figures: pipeline diagram, taxonomy map, CD diagrams, scaling-law plots, decision chart | todo |
| 5.3 | Internal review; send draft to Alonso / Needels | todo |
| 5.4 | Code + data release (Zenodo DOI) | todo |

## Decisions
| Date | Decision | Why |
|---|---|---|
| 2026-10-03 | Keep the real FEA cache as the engineering anchor; canonical suites carry the factorial | Data still exist; FEM re-runs not needed |
| 2026-10-03 | Score canonical experiments against clean truth | B-01 |
| 2026-10-03 | Primary end-point = true regret at matched cost, not RMSE | G5–G7; M.16–M.17 |
| 2026-10-03 | New repo `SurrogatePaper`, thesis and COMPASS read-only | user instruction |
| 2026-10-03 | Cite the thesis formally everywhere: `poznanski2026towards` (M.S. thesis, Univ. of Kansas, 2026; Sec. 14.3 p. 203, App. G p. 319); never "the thesis" anecdotally. Repository handle/DOI still to add | user instruction; archival document |
| 2026-10-03 | Co-learning methods (DKL, SINDy-AE, KAN, TabPFN) also get large-n datasets (D-LARGE: generated analytic + public regression sets) | user note — test them in their intended regime |
| 2026-10-03 | Cluster: KU CRC, partitions `sixhour` (short array shards < 5.5 h) and `math`; Claude may submit via terminal once VPN + username + key-SSH exist | user instruction |
| 2026-10-03 | KU CRC unlikely → compute = load reduction + laptop (10 workers, resumable runner) + Kaggle shards (author token) + NSF ACCESS Explore request (draft `docs/ACCESS_EXPLORE_REQUEST.md`, needs advisor letter); GitHub Actions rejected (ToS); real FEA data never leaves this machine | STUDY_PLAN §4b |
| 2026-10-03 | E1 uses the thesis' own COMPASS code (imported read-only) for the reproduction gate, then re-scores the same fits against clean truth | isolates the methodology fix from any porting difference |
| 2026-10-03 | Study plan E0–E11 written: `docs/STUDY_PLAN.md` (+ page `docs/study_plan.html`) | — |

## Open questions for the author
1. ~~Test-surface repo~~ — resolved: Math 796 HW1 suite.
2. Compute — answered: this laptop (12 threads, CPU-only) unless free compute is better. **Recommendation: KU CRC
   cluster** — the author has used it (SLURM scripts in `OneDrive/research`, `--partition=capl`, 48 cores/node); free
   for KU users if the account is still active. Fallbacks: Kaggle notebooks (free CPU/GPU, weekly quota), Google Colab
   (session limits). Design the runner to be resumable and shardable (task 2.7) so either works.
6. ~~Aircraft dataset~~ — answered yes: second real-data case (RQ8).
7. ~~Field outputs~~ — answered no: F10 limited to scalar response surfaces.
3. Target venue and length; co-authorship with Alonso / Needels (affects framing toward MF-GP)?
4. Is the post-Bauschinger cache (`tmf_dca5926004`, 103 usable) to be used as a second real dataset, or only the thesis one?
5. Any export-control / distribution review needed for publishing the FEA data (missile duct application)? Default plan:
   publish non-dimensionalised data, or only results.
