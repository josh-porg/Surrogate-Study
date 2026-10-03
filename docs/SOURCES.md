# Sources & Literature Review

Working literature base for the paper. Organised **breadth-first**: first the reviews across disciplines, then a
taxonomy of every method family we could find (classical, abandoned/under-explored, non-aerospace, current
state-of-the-art), then uncertainty, sampling, multi-fidelity, upscaling, optimizers, benchmarks, theory and statistics.

**Verification status (2026-10-03): all 254 references are verified.** `code/lit/verify_refs.py` checks every entry in
`data/literature/refs_seed.csv` against Crossref / arXiv, an audit pass caught ~30 wrong auto-matches (fixed by pinned
DOIs, arXiv preference, or hand entries in `data/literature/manual_refs.bib`), and the result is
**`paper/refs.bib` — the authoritative metadata**. The ✔/○ flags in the tables below are historical (✔ = found during the
search, ○ = entered from memory); every ○ item has since been verified. Corrections found by verification:
- Mainini et al. follow-up framework paper is *Arch. Comput. Methods Eng.* **2025** (key `mainini2025framework`), not 2026.
- Williams & Cremaschi have two 2021 papers: the ESCAPE-31 conference paper (`williams2021novel`, 99 VLSE functions —
  the one read in full) and the CERD journal paper (`williams2021selection`, adds SBO).
- Needels: AIAA SciTech 2024 paper `needels2024trajectory` (DOI 10.2514/6.2024-1013) in addition to the dissertation.
- Process-systems SBO benchmark arXiv 2412.13948 is Neufang et al. (`neufang2024surrogate`).
- Option-pricing NN review is Ruf & Wang (`ruf2020neural`).

Depth-read notes: `data/literature/notes/DEPTH_NOTES.md`. Review draft: `paper/main.tex` (compiles; 29 pp. incl. refs).

**Status of the method in our study** — `T` tested in thesis · `P` planned for the paper · `C` candidate (decide in
Phase 1 screening) · `R` reviewed only (discussed, not run).

---

## 0. People and context

- **J. J. Alonso** (Stanford, Aerospace Design Lab; SU2) and **J. Needels** (Stanford PhD 2024 under Alonso, now Sandia).
  Needels' dissertation: *Efficient Multidisciplinary Analysis and Optimization of Hypersonic Vehicles Using
  Multi-Fidelity Surrogate Models* (Stanford, 2024) ✔; 2023 Stanford affiliates talk *Trajectory Informed
  Multi-fidelity Surrogates for Hypersonic Vehicle Optimization* (multi-fidelity GPs of aero/aerothermal loads,
  adjoint-guided sampling) ✔. Thesis cites `needels2023efficient` (EGO scout in constrained aerospace conceptual
  design). **Implication:** multi-fidelity GP is the audience's home method; the paper must treat MF rigorously and
  must explain the GBT result in terms they will accept (inductive bias, noise, partitioning, data size).
  - https://aa.sites.stanford.edu/sites/g/files/sbiybj17081/files/media/file/affiliates_program_meeting_2023_slides_am_0.pdf

---

## 1. Reviews and surveys, across disciplines

### 1.1 Engineering design / MDO (aerospace-dominant)
| Ref | Note | Flag |
|---|---|---|
| Simpson, Poplinski, Koch, Allen 2001, *Eng. with Computers* 17 — "Metamodels for computer-based engineering design: survey and recommendations" | Classic RSM/kriging/NN survey | ○ |
| Jin, Chen, Simpson 2001, *SMO* 23:1–13 — "Comparative studies of metamodelling techniques under multiple modelling criteria" | Poly, MARS, RBF, kriging on 14 problems; closest ancestor of our study | ✔ https://doi.org/10.1007/s00158-001-0160-4 |
| Queipo et al. 2005, *Prog. Aerosp. Sci.* 41 — "Surrogate-based analysis and optimization" | Thesis-cited | ○ |
| Wang & Shan 2007, *J. Mech. Des.* 129 — "Review of metamodeling techniques in support of engineering design optimization" | | ○ |
| Forrester, Sóbester, Keane 2008 — *Engineering Design via Surrogate Modelling* (Wiley) | Textbook; thesis-cited | ✔ |
| Forrester & Keane 2009, *Prog. Aerosp. Sci.* 45 — "Recent advances in surrogate-based optimization" | | ✔ (eprints.soton.ac.uk/65935) |
| Viana, Simpson, Balabanov, Toropov 2014, *AIAA J.* 52 — "Metamodeling in MDO: where have we been, where are we going?" | | ○ |
| Alizadeh, Allen, Mistree 2020, *Res. Eng. Des.* 31:275–298 — "Managing computational complexity using surrogate models: a critical review" | 200+ papers; size–accuracy–time trade; MARS/RSM/kriging categories | ✔ https://doi.org/10.1007/s00163-020-00336-7 |
| Kůdela & Matoušek 2022, *Arch. Comput. Methods Eng.* — "Recent advances and applications of surrogate models for FEM computations: a review" | 180 papers; FEM-specific (our real data are FEM) | ✔ |
| Yondo, Andrés, Valero 2018, *Prog. Aerosp. Sci.* 96 — "A review on design of experiments and surrogate models in aircraft real-time and many-query aerodynamic analyses" | | ○ |
| Bouhlel, Hwang, Bartoli, Lafage, Morlier, Martins 2019, *Adv. Eng. Softw.* — SMT: Surrogate Modeling Toolbox | Software; KPLS, GEK, MFK; Martins group | ○ |
| Saves et al. 2024 — SMT 2.0 | mixed-variable, hierarchical GPs | ○ |
| Marine engineering scoping review, arXiv 2404.18654 | GP/kriging = 34 % of SBO applications | ✔ https://arxiv.org/pdf/2404.18654 |

### 1.2 Process systems / chemical engineering
| Ref | Note | Flag |
|---|---|---|
| Bhosekar & Ierapetritou 2018, *Comput. Chem. Eng.* 108 — "Advances in surrogate based modeling, feasibility analysis, and optimization: a review" | thesis-cited | ○ |
| Williams & Cremaschi 2021, *Chem. Eng. Res. Des.* 170:76–89 — "Selection of surrogate modeling techniques for surface approximation and surrogate-based optimization" | **Key**: MARS/GP best for approximation; **RF, SVR, GP most robust for optimization** — directly supports B-04; PRESTO recommender | ✔ https://par.nsf.gov/servlets/purl/10297247 |
| "Surrogate-based optimization techniques for process systems engineering", arXiv 2412.13948 | DYCORS best, SRBF second on unconstrained synthetic benchmark | ✔ https://arxiv.org/pdf/2412.13948 |
| Cozad, Sahinidis, Miller 2014, *AIChE J.* — ALAMO: learning surrogate models with automated algebraic modeling | sparse algebraic regression + adaptive sampling; a "sparse_physics" ancestor | ○ |
| McBride & Sundmacher 2019, *Chem. Ing. Tech.* — overview of surrogate modeling in chemical process engineering | | ○ |

### 1.3 Water resources / hydrology / environment
| Ref | Note | Flag |
|---|---|---|
| Razavi, Tolson, Burn 2012, *Water Resour. Res.* 48, W07401 — "Review of surrogate modeling in water resources" | Distinguishes response-surface vs lower-fidelity-physics surrogates; warns on SBO pitfalls | ✔ doi:10.1029/2011WR011527 |
| Asher, Croke, Jakeman, Peeters 2015, *Water Resour. Res.* — "A review of surrogate models and their application to groundwater modeling" | | ○ |

### 1.4 Building energy
| Westermann & Evins 2019, *Energy and Buildings* 198 — "Surrogate modelling for sustainable building design – a review" | ✔ |

### 1.5 Structural reliability / UQ
| Ref | Note | Flag |
|---|---|---|
| Moustapha, Marelli, Sudret 2022, *Struct. Saf.* — "Active learning for structural reliability: survey, general framework and benchmark" | >12 000 problems; modular framework (surrogate × learning function × stopping) = exactly the "swap the parts" idea | ✔ https://arxiv.org/pdf/2106.01713 |
| Moustapha & Sudret 2019, *SMO* — "Surrogate-assisted reliability-based design optimization: a survey and a new general framework" | | ✔ https://arxiv.org/pdf/1901.03311 |
| Lüthen, Marelli, Sudret 2021, *SIAM/ASA JUQ* 9 — "Sparse polynomial chaos expansions: literature survey and benchmark" | solver & sampling choice → orders-of-magnitude MSE differences | ✔ https://arxiv.org/abs/2002.01290 |
| Teixeira, Nogal, O'Connor 2021, *Struct. Saf.* — adaptive approaches in metamodel-based reliability analysis: a review | | ○ |

### 1.6 Electromagnetics / microwave
| Bandler et al. 1994, *IEEE T-MTT* 42 — Space mapping technique for electromagnetic optimization | ○ (1993/1994 — check) |
| Koziel, Cheng, Bandler 2008, *IEEE Microwave Mag.* — "Space mapping" review | ✔ (lecture notes desi.iteso.mx) |
| Koziel & Leifsson (eds.) 2013 — *Surrogate-Based Modeling and Optimization: Applications in Engineering* (Springer) | ✔ |

### 1.7 Cosmology / physics emulators
| Heitmann et al. 2009/2014 — Coyote Universe / "The Cosmic Emu" (process convolution + PCA + GP) | ✔ (alcf.anl.gov) |
| Neural emulators superseding GPs at scale (e.g. GokuNEmu, MF-Box 2306.03144) | ✔ |

### 1.7b Breadth pass 2 (2026-10-03): other simulation-heavy fields
| Field | Ref (bib key) | Take-away |
|---|---|---|
| Climate | Watson-Parris et al. 2022 ClimateBench (`watsonparris2022climatebench`); Lütjens et al. 2024 (`lutjens2024climatebench`) | **linear pattern scaling beat a 100 M-parameter foundation model on 3/4 variables**; winner flips with internal-variability noise → noise level is a ranking factor; always run simple baselines |
| Finance | Ludkovski 2023 *Annu. Rev. Stat.* (`ludkovski2023statistical`); Ruf & Wang 2019/20 (`ruf2020neural`) | surrogates for option pricing/calibration: DNN, GP, **GBM, smoothing splines, Chebyshev** all in use |
| Computational economics (ABM) | Lamperti, Roventini, Sani 2018 (`lamperti2018agent`) | **XGBoost** surrogate + iterative sampling; kriging rejected for ragged surfaces and >20 parameters |
| Epidemiology / systems biology | Vernon, Liu, Goldstein 2018; Andrianakis et al. 2015; hmer package (`vernon2018bayesian`, `andrianakis2015bayesian`, `iskauskas2024hmer`) | **history matching**: emulators used to *rule out* implausible space in waves — analogue of the thesis floor-bound/infeasible screening |
| Petroleum | Energies 2022 proxy-model review (`bahrami2022proxy`) | "proxy models"; lack of validation standards noted |
| Turbomachinery | Li & Zheng 2017 (`li2017review`) | RSM, kriging, ANN dominate |
| Additive manufacturing | Hu & Mahadevan 2017 (`hu2017uncertainty`) | GP / SVM process maps; multi-physics UQ |
| Nuclear, crashworthiness | (search only) | moving to graph/operator networks on field outputs — out of our scalar scope |
| Statistics of computer experiments | Sacks et al. 1989; Santner et al. 2003; Kleijnen 2015; Gramacy 2020; Cressie 1990 (origins of kriging) | lineage for §2.1 of the paper |

### 1.8 Machine learning / AutoML / tabular learning (non-aerospace, highly relevant to the GBT result)
| Ref | Note | Flag |
|---|---|---|
| Grinsztajn, Oyallon, Varoquaux 2022, NeurIPS D&B — "Why do tree-based models still outperform deep learning on typical tabular data?" | Trees win at ~10k samples; NN must (1) be robust to uninformative features, (2) preserve data orientation, (3) learn irregular functions | ✔ https://arxiv.org/abs/2207.08815 |
| Shwartz-Ziv & Armon 2022, *Information Fusion* — "Tabular data: deep learning is not all you need" | | ○ |
| McElfresh et al. 2023, NeurIPS — "When do neural nets outperform boosted trees on tabular data?" | | ○ |
| Hollmann et al. 2025, *Nature* 637 — TabPFN: "Accurate predictions on small data with a tabular foundation model" | Beats tuned GBT ≤10k samples; in-context Bayesian regression; **candidate surrogate with native UQ** | ✔ PMC11711098 |
| Eggensperger, Hutter, Hoos, Leyton-Brown 2015, AAAI — "Efficient benchmarking of hyperparameter optimizers via surrogates" | **Key for B-04**: RF surrogates preserve optimizer rankings; GP surrogates gave qualitatively different results | ✔ https://ojs.aaai.org/index.php/AAAI/article/view/9375 |
| Eggensperger et al. 2018, *Mach. Learn.* — surrogate benchmarks for algorithm configuration | | ✔ arXiv 1703.10342 |
| Hutter, Xu, Hoos, Leyton-Brown 2014, *AIJ* 206 — "Algorithm runtime prediction: methods & evaluation" | RF vs GP vs ridge vs NN on performance surfaces | ○ |

### 1.9 Evolutionary computation
| Jin 2005, *Soft Comput.* — "A comprehensive survey of fitness approximation in evolutionary computation" | ○ |
| Jin 2011, *Swarm Evol. Comput.* 1(2):61–70 — "Surrogate-assisted evolutionary computation: recent advances and future challenges" | ✔ |
| Loshchilov, Schoenauer, Sebag 2012 — saACM-ES (rank-SVM surrogate inside CMA-ES): **only the ranking matters** | ✔ arXiv 1206.5780 |

### 1.10 Multi-fidelity reviews
| Peherstorfer, Willcox, Gunzburger 2018, *SIAM Review* 60 — "Survey of multifidelity methods in uncertainty propagation, inference, and optimization" | adaptation / fusion / filtering taxonomy | ✔ https://cims.nyu.edu/~pehersto/preprints/multi-fidelity-survey-peherstorfer-willcox-gunzburger.pdf |
| Fernández-Godino, Park, Kim, Haftka 2016/2019 — "Review of multi-fidelity models" | | ✔ arXiv 1609.07196 |
| Brevault, Balesdent, Hebbal 2020, *Aerosp. Sci. Technol.* — "Overview of GP based multi-fidelity techniques with variable relationship between fidelities" | | ✔ arXiv 2006.16728 |
| Giselle Fernández-Godino 2023 — review update | ○ |
| Do & Zhang? "Multi-fidelity Bayesian optimization: a review" arXiv 2311.13050 | | ✔ (authors to confirm) |
| Mainini et al. 2022/2024, *AIAA J.* — "Analytical benchmark problems for multifidelity optimization methods" (NATO AVT-354) | **benchmark suite + metrics for MF** | ✔ https://arxiv.org/pdf/2204.07867 |
| 2026, *Arch. Comput. Methods Eng.* — "Analytical benchmark problems and methodological framework for the assessment and comparison of multifidelity optimization methods" | follow-up | ✔ (pureportal.strath.ac.uk) |

### 1.11 Adaptive sampling reviews
| Liu, Ong, Cai 2018, *SMO* 57 — "A survey of adaptive sampling for global metamodeling in support of simulation-based complex engineering design" | thesis-cited | ✔ |
| Fuhg, Fau, Nackenhorst 2021, *Arch. Comput. Methods Eng.* 28 — "State-of-the-art and comparative review of adaptive sampling methods for kriging" | **success depends on problem features and goal**; open toolbox | ✔ https://doi.org/10.1007/s11831-020-09474-6 |
| Garud, Karimi, Kraft 2017, *Ind. Eng. Chem. Res.* — "Design of computer experiments: a review" | | ○ |
| Settles 2009 — Active learning literature survey (UW-Madison TR 1648) | thesis-cited | ○ |

### 1.12 Uncertainty quantification in ML
| Gillis & Trappenberg 2026 — "Uncertainty quantification for trustworthy deep learning: methods and measures" | ✔ arXiv 2607.28248 |
| Ulmer, Hardmeier, Frellsen 2023, *TMLR* — "Prior and posterior networks: a survey on evidential deep learning" | ✔ arXiv 2110.03051 |
| Abdar et al. 2021, *Information Fusion* 76 — "A review of uncertainty quantification in deep learning" | ○ |
| Angelopoulos & Bates 2023 — "Conformal prediction: a gentle introduction" | ○ |

---

## 2. Taxonomy of surrogate / approximation methods (breadth-first)

Columns: **Family · method · origin (field, year) · inductive bias / why it might win or lose here · UQ native? · smooth C²? · status · key refs.**

### 2.1 Global parametric
| Method | Origin | Bias / relevance | UQ | C² | St | Refs |
|---|---|---|---|---|---|---|
| Polynomial response surface (RSM) | Box & Wilson 1951, chem. eng. | low-order global; fails on local features | via regression | ✓ | C | Box & Wilson 1951 ○; Myers & Montgomery ○ |
| Sparse dictionary regression (LASSO/OMP/SINDy/ALAMO) | statistics 1996 / dyn. sys. 2016 | physics features in basis | bootstrap | ✓ | T (`sparse_physics`) | Tibshirani 1996 ○; Brunton et al. 2016 ○; Cozad 2014 ○ |
| Polynomial chaos (PCE), sparse PCE (LARS, OMP, SP, BCS) | Wiener 1938; Ghanem & Spanos 1991; Xiu & Karniadakis 2002 | orthogonal polynomials; spectral for analytic f | via CV/bootstrap | ✓ | P | Lüthen 2021 ✔ |
| Chebyshev tensor / Chebfun-style approximation | approximation theory | spectral convergence for analytic f on boxes | ✗ | ✓ | C | Trefethen *ATAP* 2013 ○ |
| Rational approximation (AAA, Padé, Loewner, vector fitting) | Padé 1892; Antoulas–Anderson 1986; Nakatsukasa–Sète–Trefethen 2018 | captures poles / near-singular knees (the 1/R knee!) | ✗ | ✓ | C | AAA ✔ arXiv 1612.00337; multivariate p-AAA ○ |
| Symbolic regression (GP-SR, PySR, AI Feynman) | Koza 1992; Schmidt & Lipson 2009 | interpretable closed form | ✗ | ✓ | C | ○ |
| GMDH / polynomial neural networks | **Ivakhnenko 1968**, Ukraine (fish population) | self-organising polynomial nets with external (validation) criterion; "first deep learning"; largely abandoned in the West | ✗ | ✓ | C | ✔ Wikipedia/Schmidhuber |

### 2.2 Kernel / distance-based
| Method | Origin | Bias / relevance | UQ | C² | St | Refs |
|---|---|---|---|---|---|---|
| Kriging / GP (stationary, ARD, Matérn) | **Krige 1951, Matheron 1963 (mining geostatistics)**; Sacks, Welch, Mitchell, Wynn 1989 (DACE) | smoothness prior set by kernel; O(n³) | ✓ | ✓ (ν>2) | T | Rasmussen & Williams 2006 ○; Sacks 1989 ○ |
| Universal kriging / GP with mean function | geostatistics | trend + residual | ✓ | ✓ | T (`gp_physics_mean`) | |
| Blind kriging | Joseph, Hung, Sudjianto 2008 | variable-selected trend | ✓ | ✓ | C | ○ |
| Gradient-enhanced kriging (GEK) | Morris, Mitchell, Ylvisaker 1993 | uses adjoints (Needels uses adjoint guidance) | ✓ | ✓ | C | ○ |
| KPLS kriging (high-d) | Bouhlel et al. 2016 | PLS-reduced length-scales | ✓ | ✓ | C | ○ |
| Heteroscedastic GP / stochastic kriging | Goldberg 1998; Ankenman, Nelson, Staum 2010 | input-dependent noise (FEA noise!) | ✓ | ✓ | P | ✔ (Lancaster eprints 65040) |
| Student-t GP / robust GP | Jylänki 2011 | outlier robustness | ✓ | ✓ | C | ○ |
| Sparse / inducing-point GP (FITC, SVGP), SKI/KISS-GP | Snelson 2006; Titsias 2009; Wilson & Nickisch 2015 | scales to large n | ✓ | ✓ | R | ○ |
| Local GP (laGP) | Gramacy & Apley 2015 | local nonstationarity, big n | ✓ | ~ | C | ○ |
| Treed GP (TGP) | **Gramacy & Lee 2008 — motivated by a rocket-booster CFD experiment** | partitions + local GPs; regime switching | ✓ | ✗ at splits | P | ✔ arXiv 0710.4536 |
| Deep GP | Damianou & Lawrence 2013 | warped nonstationary | ✓ | ✓ | C | ○ |
| Warped GP / input warping | Snelson 2004; Snoek 2014 | monotone transforms (log target generalised) | ✓ | ✓ | C | ○ |
| RBF interpolation / regression (multiquadric, TPS, Gaussian, polyharmonic) | **Hardy 1971 (geodesy, multiquadric)**; Duchon 1977 (TPS) | interpolant with native-space error bounds | ✗ (power function only) | ✓ | T (`rbf`) | Wendland 2004 ✔; Fasshauer 2007 ○ |
| Compactly supported RBF (Wendland) | Wendland 1995 | sparse matrices | ✗ | ✓ | C | ✔ |
| RBF partition of unity | Wendland 2002 | local + scalable | ✗ | ✓ | C | ○ |
| Support vector regression; LS-SVM | Vapnik 1995; Suykens 1999 | ε-insensitive, robust | ✗ | ✓ | P | ○ |
| Kernel ridge regression / random Fourier features | Rahimi & Recht 2007 | GP mean without variance; scalable | ✗ | ✓ | C | ○ |
| Inverse distance weighting (Shepard) | **Shepard 1968** | simplest scattered interpolant; flat spots at data | ✗ | ✗ | C | ○ |
| Natural-neighbour (Sibson) interpolation | **Sibson 1981** | Voronoi-based, C¹ off data | ✗ | C¹ | C | ○ |
| Moving least squares | **Lancaster & Šalkauskas 1981** | local polynomial; meshfree methods | ✗ | ✓ | C | ○ |
| Kernel smoothing (Nadaraya–Watson), LOESS | 1964; Cleveland 1979 | local averaging | bootstrap | ✓ | C | ○ |
| k-NN regression | Fix & Hodges 1951 | piecewise constant baseline | ✗ | ✗ | C | ○ |
| Lipschitz interpolation / optimal recovery | Sukharev 1978; Beliakov 2006 | worst-case optimal bounds; monotone/Lipschitz info | bounds | ✗ | C | ○ |

### 2.3 Spline / piecewise-polynomial
| Method | Origin | Bias / relevance | UQ | C² | St | Refs |
|---|---|---|---|---|---|---|
| Tensor-product B-spline interpolation (structured grid) | Schoenberg 1946; de Boor 1978 | production deliverable; curse of dimensionality n_axis^d | ✗ | ✓ | T | de Boor 2001 ○ (thesis-cited) |
| Smoothing splines | Reinsch 1967; **Wahba 1990 (equivalent to GP posterior mean)** | noise handling | Bayesian CI | ✓ | P | ○ |
| P-splines (B-spline + difference penalty) | Eilers & Marx 1996 | fit splines **directly to scattered noisy data** — skips the GP→grid step | ✓ (mixed-model) | ✓ | P | ✔ (*Stat. Sci.* 11) |
| Thin-plate regression splines / GAMs (mgcv) | Wood 2003; Hastie & Tibshirani 1990 | additive + interactions | ✓ | ✓ | P | ○ |
| MARS | Friedman 1991 | adaptive hinge basis; kinks (clamp_floor!) | ✗ | ✗ (C⁰; cubic variant C¹) | P | ○ |
| Multilevel B-spline approximation (MBA) | Lee, Wolberg, Shin 1997 (*IEEE TVCG*) | coarse-to-fine lattices from scattered data, C² | ✗ | ✓ | P (upscaling) | ✔ |
| Hierarchical / THB-splines, T-splines, LR-splines | Forsey & Bartels 1988; Giannelli, Jüttler, Speleers 2012 | **local refinement without full tensor grid** | ✗ | ✓ | P (upscaling) | ✔ |
| Sparse-grid (Smolyak) interpolation | **Smolyak 1963**; Bungartz & Griebel 2004 | breaks n^d for mixed smoothness; adaptive variants | ✗ | ✓ (poly) | P | ○ |
| Monotone / shape-preserving splines (Fritsch–Carlson, SCAM) | 1980; Pya & Wood 2015 | physics: damage monotone in thickness | ✗ | C¹ | C | ○ |
| Simplex / Delaunay piecewise-linear | Thesis `linear` baseline | | ✗ | ✗ | T | |
| Bernstein / Bézier patches; NURBS | Bernstein 1912; Bézier 1960s | geometry community | ✗ | ✓ | R | |
| Wavelet / multiresolution regression | Donoho & Johnstone 1994 | local features at multiple scales | ✗ | depends | C | ○ |

### 2.4 Trees and ensembles
| Method | Origin | Bias / relevance | UQ | C² | St | Refs |
|---|---|---|---|---|---|---|
| CART | Breiman et al. 1984 | axis-aligned piecewise constant | ✗ | ✗ | R | ○ |
| Random forest | Breiman 2001 | bagging + feature subsampling | IJ / jackknife | ✗ | P | Wager, Hastie, Efron 2014 ✔ (JMLR 15) |
| Quantile regression forest | Meinshausen 2006 | conditional distribution | ✓ | ✗ | P | ✔ |
| Gradient-boosted trees (XGBoost, LightGBM, CatBoost) | Friedman 2001; Chen & Guestrin 2016 | **thesis real-cache winner**; handles regimes/kinks, robust to uninformative inputs | ✗ native | ✗ | T | ○ |
| NGBoost (natural-gradient boosting) | Duan et al. 2020 | distributional boosting | ✓ | ✗ | P | ✔ (arXiv 1910.03225) |
| Boosting inference / CLT intervals | arXiv 2509.23127 | | ✓ | ✗ | C | ✔ |
| BART | Chipman, George, McCulloch 2010 | Bayesian sum-of-trees; used as computer-experiment emulator with sequential design | ✓ | ✗ | P | ✔ arXiv 1203.1078 |
| GP-BART | 2022 | trees with GP leaves | ✓ | partial | C | ✔ arXiv 2204.02112 |
| Mondrian forests | Lakshminarayanan, Roy, Teh 2014/2016 | online, calibrated UQ | ✓ | ✗ | C | ○ |
| Model trees (M5, linear leaves), soft/smooth trees | Quinlan 1992; Irsoy 2012 | **smooth or piecewise-linear trees → differentiable GBT** | ✗ | partial | C | ○ |
| Monotone-constrained GBT | XGBoost option | physics monotonicity | ✗ | ✗ | C | |
| Tree → smooth: GBT baked to grid then spline | thesis `gbt_grid` | upscaling makes trees optimizer-compatible | ✗ | ✓ | T | |

### 2.5 Neural networks
| Method | Origin | Bias | UQ | C² | St | Refs |
|---|---|---|---|---|---|---|
| MLP (point estimate) | Rumelhart 1986 | | ✗ | ✓ (smooth act.) | T (`mlp`) | |
| Deep ensembles | Lakshminarayanan, Pritzel, Blundell 2017 | | ✓ | ✓ | T (`nn`) | ○ |
| MC dropout | Gal & Ghahramani 2016 | | ✓ | ✓ | C | ○ |
| Bayesian NN (HMC, VI, Laplace) | MacKay 1992; Neal 1996; Daxberger 2021 (Laplace redux) | | ✓ | ✓ | C | ○ |
| Evidential regression | Amini et al. 2020 | single-pass UQ | ✓ | ✓ | C | Ulmer ✔ |
| Sobolev training (fit values + derivatives) | Czarnecki et al. 2017 | derivative accuracy for optimizers | ✗ | ✓ | C | ✔ arXiv 1706.04859 |
| Kolmogorov–Arnold networks (spline edges) | Liu et al. 2024 (ICLR 2025); KA theorem 1957 | spline-native NN; small-data science fits | ✗ | ✓ | P | ✔ arXiv 2404.19756 |
| RBF networks / ELM | Broomhead & Lowe 1988; Huang 2006 | | ✗ | ✓ | C | ○ |
| Physics-informed NN | Raissi, Perdikaris, Karniadakis 2019 | needs PDE residual — out of scope (black-box) | | | R | ○ |
| Neural operators (DeepONet, FNO) | Lu 2021; Li 2021 | field-to-field; out of scope (scalar outputs) | | | R | ○ |
| TabPFN v2 (prior-fitted transformer) | Hollmann et al. 2025 | in-context Bayesian inference, ≤10k samples | ✓ | ~ | P | ✔ |
| Neural processes | Garnelo 2018 | meta-learned GP-like | ✓ | ✓ | C | ○ |

### 2.6 Hybrid, fusion, ensembles, partitioning
| Method | Notes | St | Refs |
|---|---|---|---|
| Ensemble of surrogates (best-PRESS, weighted average) | Goel, Haftka, Shyy, Queipo 2007 *SMO* 33:199–216; Viana, Haftka, Steffen 2009 *SMO* 39:439–457 | P | ✔ |
| Stacking / super learner | Wolpert 1992; van der Laan 2007 | P | ○ |
| Mixture of experts / gated GPs | Jacobs et al. 1991; Rasmussen & Ghahramani 2002 | C | ○ |
| Regime partition + per-regime model (COMPASS `gp_regime`) | thesis production | T | |
| Classification + regression (feasible/floor-bound/infeasible classifier, then regress) | thesis screening is a hand-made version | P | |
| Residual (physics/LF) + data correction | `gp_physics_mean`; additive/multiplicative bridge functions | T/P | |
| GBT + GP residual / GP on GBT leaves | | C | |
| Space mapping (input mapping LF→HF) | Bandler; Koziel | P (MF) | ✔ |

### 2.7 Change-of-basis / latent-space surrogates (data-driven-ROM family) — added 2026-10-03

Idea (author's, from DMD / compressed-sensing DMD / SINDy-autoencoder ROM work): find a domain or basis in which
the surface is simple — sparse, low-rank, low-dimensional or separable — fit there, and map back. Optionally
**co-learn** the transform and the fit (AutoML-style), and use randomised **compression** to speed up intermediate
stages the way compressed DMD speeds up DMD without removing it.

Four sub-families, by *where* the change of basis acts:

**(a) Input-space linear reduction (rotate/project x, then fit):**
| Method | Origin | Notes | St | Flag |
|---|---|---|---|---|
| Projection pursuit regression f≈Σg_j(w_jᵀx) | **Friedman & Stuetzle 1981, JASA** | old, under-used in engineering; ancestor of single-layer NN and ridge approximation | P | ✔ |
| Active subspaces (C=E[∇f∇fᵀ] eigenvectors) | Constantine 2015 (SIAM book) | needs gradients — available from a GP/adjoint (Needels uses adjoints) | P | ✔ |
| Polynomial ridge approximation (Grassmann manifold opt.) | Hokanson & Constantine 2018, *SISC* | no gradients needed | P | ✔ arXiv 1702.05859 |
| GP on active subspace / probabilistic ridge | Tripathy, Bilionis, Gonzalez 2016; Seshadri et al. 2019 | joint subspace + GP | C | ✔ arXiv 1809.06581 |
| Sliced inverse regression, PLS (KPLS kriging) | Li 1991; Bouhlel 2016 | supervised linear reduction | C | ○ |
| Random embeddings (REMBO) | Wang et al. 2016, *JAIR* | optimization in random low-d subspace | C | ✔ arXiv 1301.1942 |

**(b) Input-space nonlinear co-learning (learn φ(x) and the fit jointly):**
| Method | Origin | Notes | St | Flag |
|---|---|---|---|---|
| Deep kernel learning k(φ_w(x),φ_w(x')) | Wilson, Hu, Salakhutdinov, Xing 2016 (AISTATS); SV-DKL (NeurIPS 2016) | **exactly "live co-learning of domain + fit"**, by GP marginal likelihood | P | ✔ |
| DKL over-fitting pitfalls | Ober, Rasmussen, van der Wilk 2021 (UAI) "The promises and pitfalls of deep kernel learning" | marginal likelihood can over-correlate at small n — critical for our n≈50–150 | R | ○ |
| Kernel flows (learn kernel by "halve the data, keep accuracy") | Owhadi & Yoo 2019, *JCP* | CV-like kernel learning; less over-fitting than ML-II | P | ✔ arXiv 1808.04475 |
| Input warping (Kumaraswamy/Beta CDF per axis) | Snoek et al. 2014 (ICML) | monotone per-axis transform; generalises the thesis' log(p) | P | ○ |
| Manifold GP | Calandra et al. 2016 | NN feature map + GP | C | ○ |
| SINDy autoencoder (latent coords + sparse dictionary) | **Champion, Lusch, Kutz, Brunton 2019, *PNAS*** | dynamics version; static analogue = encoder + sparse dictionary regression on y | P (adapted) | ✔ |
| Latent-space BO with VAEs | Gómez-Bombarelli et al. 2018, *ACS Cent. Sci.* | for structured inputs; unsupervised AE on uniformly sampled x carries no information about f | R | ✔ |
| KAN (learnable spline edges) | Liu et al. 2024 | already §2.5; is itself a learned univariate-basis decomposition | P | ✔ |
| Fourier-feature / SIREN networks | Tancik et al. 2020; Sitzmann et al. 2020 | learned spectral basis; spectral-bias fix for oscillatory f | C | ○ |

**(c) Output/function-space sparse bases + compressed sensing (fit few coefficients in a fixed basis):**
| Method | Origin | Notes | St | Flag |
|---|---|---|---|---|
| Compressed sensing (ℓ1 recovery, RIP) | Candès, Romberg, Tao 2006; Donoho 2006 | n ∝ sparsity·polylog, not ∝ basis size | P | ○ |
| CS for bounded orthonormal systems (Fourier, Legendre, Chebyshev) | Rauhut 2010; Rauhut & Ward 2012 | sampling density matters (Chebyshev measure) | P | ○ |
| Compressive-sampling PCE | Doostan & Owhadi 2011, *JCP*; Hampton & Doostan 2015 | sparse PCE from few runs | P | ✔ |
| CS-PCE + D-optimal design | Diaz, Doostan, Hampton 2018 | design + sparse recovery | C | ✔ arXiv 1712.10131 |
| Infinite-dimensional CS / function interpolation | Adcock 2015 | | R | ✔ arXiv 1509.06073 |
| Sparse Fourier / spectral-mixture kernels | Wilson & Adams 2013 | GP whose spectrum is learned | C | ○ |
| Wavelet-sparse regression | Donoho & Johnstone 1994 | local features (knee, cliff) sparse in wavelets | C | ○ |

**(d) Low-rank function decompositions (the SVD of a surface):**
| Method | Origin | Notes | St | Flag |
|---|---|---|---|---|
| Functional SVD / Schmidt decomposition f(x,y)≈Σσ_r u_r(x)v_r(y) | Schmidt 1907; Chebfun2 (Townsend & Trefethen 2013) | 2-D surfaces are often numerically low-rank | P | ○ |
| Tensor-train / TT-cross approximation | Oseledets & Tyrtyshnikov 2010, *Lin. Alg. Appl.* | learns a d-dim function from O(d r² m) adaptively chosen samples | P | ○ |
| Canonical (CP) tensor regression / separated representations | Beylkin & Mohlenkamp 2005; Doostan, Validi, Iaccarino 2013 | | C | ○ |
| Tensor-train GP / functional TT surrogates | Gorodetsky & Jakeman 2018 (C3) | | C | ○ |

**(e) Compression to speed the intermediate process (compressed/randomised-DMD analogue):**
| Method | Origin | Use in our pipeline | St | Flag |
|---|---|---|---|---|
| Compressed DMD / randomised DMD | Erichson, Brunton, Kutz 2016 (*J. Real-Time Image Proc.*); Erichson et al. 2019 (arXiv 1702.02912) | template: sketch, decompose small matrix, lift | R | ✔ |
| Randomised SVD (range finder) | Halko, Martinsson, Tropp 2011, *SIAM Rev.* | compress the dense upscaling grid (stage S5) | P | ○ |
| Randomised HOSVD / TT-SVD of the baked grid | Minster, Saibaba, Kilmer 2020 | grid m^d → d·m·r² storage and O(1)-ish evaluation; breaks the d≈6 wall of §M.6 | P | ○ |
| Nyström / random Fourier features / inducing points for GP | Williams & Seeger 2001; Rahimi & Recht 2007; Titsias 2009 | speed scout refits inside the AL loop | P | ○ |
| Sketched least squares (CountSketch/SRHT) | Woodruff 2014 | P-spline / PCE fits on big grids | C | ○ |
| POD + regression of modal coefficients (non-intrusive ROM) | Guo & Hesthaven 2018 (*CMAME*); Audouze et al. 2009 | only if FEA **field** outputs exist (they are not in the thesis cache) | R | ○ |

**(f) AutoML / pipeline co-search** — choose (transform, basis, learner, hyper-parameters) jointly:
CASH problem (Thornton et al. 2013, Auto-WEKA ○); Auto-sklearn (Feurer et al. 2015 ○); TPOT (Olson 2016 ○);
for physics, sparse-regression model selection (ALAMO) and SINDy library search. Our version: a transform library
{identity, log, warp, PCA/AS rotation, PPR, Fourier, Chebyshev, wavelet, TT, learned encoder} × learner library, searched
by nested CV or BO under a fit-time budget.

---

## 3. Uncertainty estimators (scouts need one)

| Estimator | Applies to | Notes | Flag |
|---|---|---|---|
| GP posterior variance | GP | y-independent for fixed θ (see BUGS B-02) | |
| Bootstrap / query-by-committee | any | Seung, Opper, Sompolinsky 1992 (thesis-cited) | ○ |
| Infinitesimal jackknife / jackknife-after-bootstrap | RF | Wager et al. 2014 | ✔ |
| Quantile regression forest | RF | Meinshausen 2006 | ✔ |
| NGBoost / distributional GBT | GBT | | ✔ |
| Virtual ensembles (CatBoost), boosting dropout (DART) | GBT | Malinin, Prokhorenkova, Ustimenko 2021 | ○ |
| Deep ensembles / MC dropout / Laplace / evidential | NN | | ✔ (surveys) |
| Conformal prediction (split, jackknife+, CV+, locally adaptive) | any | distribution-free coverage; Vovk 2005; Barber et al. 2021 | ○ |
| Cross-validation error (LOO) as uncertainty | any | Kleijnen & van Beers 2004 jackknife variance; Le Gratiet 2013 fast CV for kriging | ✔ arXiv 1210.6187 |
| Local gradient/nonlinearity (LOLA) | any | Crombecq et al. 2011 | ○ |

---

## 4. Sampling: initial designs and acquisition (adaptive) rules

**Initial / space-filling designs:** random; LHS (McKay, Beckman, Conover 1979 ○); maximin LHS (Morris & Mitchell 1995 ○);
orthogonal arrays; Sobol' (1967 ○) / Halton (1960 ○) low-discrepancy (Koksma–Hlawka bound); CVT (Du, Faber, Gunzburger
1999 ○); minimax designs; full factorial / tensor grid (thesis original); sparse grids.

**Global-accuracy (exploration) acquisition:** max variance (MacKay 1992 ALM ○); ALC / integrated variance reduction
(Cohn 1996 ○; Sacks 1989 IMSE); mutual information (Krause, Singh, Guestrin 2008 ○); EIGF (Lam 2008 ○); MEPE
(Liu, Xu, Wang 2017 ○); LOLA-Voronoi (Crombecq 2011 ○); CV-Voronoi (Xu 2014 ○); TEAD (Mo et al. 2017 ○); query-by-committee;
gradient/curvature-weighted; sequential maximin (model-free). Benchmarked in Fuhg 2021 ✔.

**Optimization-directed acquisition:** EI / EGO (Jones, Schonlau, Welch 1998 ○, thesis-cited); PI (Kushner 1964 ○);
GP-UCB (Srinivas et al. 2010 ○, thesis-cited); knowledge gradient (Frazier 2009 ○); entropy search / PES / MES
(Hennig & Schuler 2012; Hernández-Lobato 2014; Wang & Jegelka 2017 ○); Thompson sampling; constrained EI
(Schonlau 1998; Gardner 2014 ○); TuRBO trust regions (Eriksson et al. 2019 NeurIPS ✔); SMAC RF-EI (Hutter, Hoos,
Leyton-Brown 2011 ✔); TPE (Bergstra et al. 2011 ✔); stochastic RBF / DYCORS (Regis & Shoemaker 2007, 2013 ○).

**Reliability/contour-directed:** EGRA (Bichon 2008 ○), AK-MCS U-function (Echard 2011 ○) — relevant for the
floor-bound / infeasible boundary of the real problem.

**Cost-aware / multi-fidelity acquisition:** cost-weighted EI, MF-KG, MF-MES, MF-GP-UCB (Kandasamy 2016 ○);
"Finding efficient trade-offs in multi-fidelity response surface modelling" arXiv 2103.03280 ✔.

---

## 5. Multi-fidelity methods

| Method | Relation assumed | Refs | Flag |
|---|---|---|---|
| Additive / multiplicative / hybrid bridge (correction) functions | f_H = ρ f_L + δ | Chang 1993; Alexandrov 2001; Gano 2005 | ○ |
| Co-kriging, autoregressive (AR1) | f_H = ρ f_L + δ, δ ⟂ f_L | **Kennedy & O'Hagan 2000, *Biometrika* 87:1–13** | ✔ |
| Co-kriging for optimization, exchange algorithm | | Forrester, Sóbester, Keane 2007, *Proc. R. Soc. A* 463:3251 | ✔ |
| Recursive co-kriging | identical predictor, nested independent fits | Le Gratiet & Garnier 2014 (arXiv 1210.0686) | ✔ |
| Hierarchical kriging (LF model as trend) | | Han & Görtz 2012, *AIAA J.* 50(9):1885–1896 | ✔ |
| NARGP (nonlinear AR GP) | f_H = z(f_L(x), x) | Perdikaris et al. 2017, *Proc. R. Soc. A* | ✔ |
| Deep GP multi-fidelity | | Cutajar et al. 2019 (arXiv 1903.07320) | ✔ |
| MF neural networks (composite linear + nonlinear) | | Meng & Karniadakis 2020, *JCP* (arXiv 1903.00104) | ✔ |
| Space mapping (input-space alignment) | x_L = P(x_H) | Bandler 1994; Koziel 2008 | ✔ |
| MF Monte Carlo / control variates | estimation, not surrogate | Peherstorfer 2016 | ✔ (survey) |
| MF trees / GBT with LF prediction as a feature ("feature-augmented") | simple, rarely studied | — | gap |
| Transfer / warm-start learning | | | ○ |
| When does MF help? correlation r > 0.9, ≥10 % and ≤80 % budget to LF | Toal 2015, *SMO* 51 "Some considerations regarding the use of multi-fidelity kriging" | ✔ (eprints.soton 373482) |
| MF benchmark functions: Forrester, Currin, Park, Borehole, Branin, Hartmann6, Bohachevsky, Booth, Himmelblau… | `mf2` Python package (van Rijn & Schmitt 2020, JOSS) | ✔ github.com/sjvrijn/mf2 |
| MF analytical benchmark + metrics (NATO AVT-354) | Mainini et al. | ✔ |

**Real-data MF opportunity (our own):** the thesis FEA cache stores `t_scalar` (closed-form cruise sizing,
~free) and `t_fem` (FEA, ~1 CPU-h each) at the same 901 points, plus 15 coarse-mesh (40×10) points. The thesis TMF
target *is* a multiplicative bridge `t_fem = TMF · t_scalar` — i.e. an un-named multi-fidelity model.

---

## 6. Upscaling / "baking": intermediate model → structured representation

Thesis tested one: GP → 12³ grid → cubic tensor interpolant. Alternatives to evaluate:
1. Grid resolution sweep and adaptive (non-uniform) knot placement (de Boor optimal knots ○).
2. Tensor B-spline interpolation vs **smoothing** spline on the grid (removes intermediate-model noise).
3. **Skip the grid**: fit P-splines / MBA / THB-splines / smoothing splines / RBF directly to scattered data.
4. Sparse grids (Smolyak; Clenshaw–Curtis/Leja nodes) — n grows ~n log^{d−1} n instead of n^d.
5. Chebyshev tensor interpolation of the intermediate model (spectral).
6. Function-space projection (L² projection of m onto spline space; quasi-interpolants, Lyche–Schumaker ○).
7. Distillation: train a smooth NN/KAN on dense samples of the intermediate (knowledge distillation; Hinton 2015 ○).
8. Moment-matching / derivative-matching (Hermite) grids if intermediate provides gradients (GP does).

---

## 7. Optimizers (to test on the surrogate)

| Class | Methods | Refs |
|---|---|---|
| Gradient, interior point | IPOPT (thesis production via CasADi) | Wächter & Biegler 2006 ○ |
| Gradient, SQP / quasi-Newton | SNOPT, SLSQP (Kraft 1988), L-BFGS-B, trust-constr | Nocedal & Wright 2006 ○ (thesis-cited) |
| Derivative-free local | Nelder–Mead 1965, COBYLA (Powell 1994), BOBYQA (Powell 2009), MADS/NOMAD (Audet & Dennis 2006) | ○ |
| Deterministic global | DIRECT (Jones, Perttunen, Stuckman 1993), branch-and-bound (convex relaxations) | ○ |
| Stochastic global | DE (Storn & Price 1997), CMA-ES (Hansen 2001/2016), PSO (Kennedy & Eberhart 1995), GA, SA, basin hopping (Wales & Doye 1997, thesis-cited) | ○ |
| Multistart / coverage confidence | thesis Eq. coverage bound | thesis |
| Surrogate-managed | trust-region model management (Alexandrov, Dennis, Lewis, Torczon 1998 ○), EGO, SMAC, TuRBO, DYCORS, SNOBFIT (Huyer & Neumaier 2008 ○) | ✔/○ |

**Accuracy vs optimizer interaction (the thesis' untested assumption):** Williams & Cremaschi 2021 ✔; Eggensperger 2015 ✔;
Jin 2011 ✔; Loshchilov 2012 ✔ (rank-only surrogate). Gradient-based optimizers additionally need **derivative**
accuracy (Sobolev norms) and suffer from spurious local minima introduced by interpolants (Gibbs/overshoot; GBT
staircases are fatal for gradient methods, harmless for DE/CMA-ES).

---

## 8. Test-function suites and landscape characterisation

| Suite | Notes | Flag |
|---|---|---|
| Surjanovic & Bingham — Virtual Library of Simulation Experiments | emulation + optimization + MF functions (Borehole, OTL, Piston, Wing weight, Currin, Park, Forrester…) | ✔ https://www.sfu.ca/~ssurjano/ |
| Jamil & Yang 2013, *IJMMNO* 4(2) — 175 benchmark functions | modality, separability, valley | ✔ arXiv 1308.4008 |
| BBOB / COCO (Hansen et al. 2009/2021) | 24 noiseless + 30 noisy functions in 5 groups | ✔ (via 1206.5780) |
| CEC competition suites | | ○ |
| `mf2` multi-fidelity functions | | ✔ |
| Mainini MF analytical benchmarks | | ✔ |
| User's Math 796 HW1 suite (32 VLSE 2-D functions in 6 categories, MATLAB) | found in `OneDrive/MATLAB/Autumn 2025/Math 796/HW_1`; copied to `data/reference/math796_hw1/` | ✔ |
| Exploratory landscape analysis (ELA) — Mersmann et al. 2011; Kerschke & Trautmann 2019 (flacco) | features (modality, separability, curvature, ruggedness, scaling) → **predict which surrogate/optimizer wins** | ✔ |
| Saves/Kerschke: surrogate selection from ELA features | | ✔ (snippet) |

---

## 9. Theory: computation and approximation

| Topic | Refs | Flag |
|---|---|---|
| No free lunch for optimization / learning | Wolpert & Macready 1997; Wolpert 1996 | ○ |
| Information-based complexity, curse of dimensionality for approximation | Traub, Wasilkowski, Woźniakowski 1988; Novak & Woźniakowski 2008 | ○ |
| Minimax nonparametric rates n^{−s/(2s+d)} | Stone 1982 | ○ |
| Kernel interpolation error: power function, fill distance h^{ν}, native spaces | Wendland 2004 *Scattered Data Approximation*; Schaback 1995 (uncertainty/trade-off principle) | ✔ |
| GP posterior contraction | van der Vaart & van Zanten 2008/2011 | ○ |
| EI convergence rates | Bull 2011, JMLR 12:2879–2904 | ✔ |
| GP-UCB regret | Srinivas et al. 2010 | ○ |
| Spline approximation O(h^{k}) and derivative orders | de Boor 2001 | ○ |
| Sparse-grid complexity (mixed smoothness) | Bungartz & Griebel 2004 *Acta Numerica* | ○ |
| Tree/ensemble approximation & consistency | Biau & Scornet 2016 *TEST*; Wager & Athey 2018 | ○ |
| NN approximation rates | Barron 1993; Yarotsky 2017 | ○ |
| Gibbs phenomenon / Runge | | ○ |
| Low-discrepancy, Koksma–Hlawka | Niederreiter 1992 | ○ |
| Equivalence: smoothing spline = GP posterior mean; kriging = RBF interpolant + polynomial | Kimeldorf & Wahba 1970; Wahba 1990 | ○ |

---

## 10. Benchmarking methodology and statistics

| Ref | Use | Flag |
|---|---|---|
| Benavoli, Corani, Demšar, Zaffalon 2017, JMLR 18 — "Time for a change: Bayesian analysis" | thesis-cited; Bayesian signed-rank / hierarchical | ○ |
| Demšar 2006, JMLR 7 — statistical comparisons of classifiers over multiple data sets (Friedman + Nemenyi, CD diagrams) | | ○ |
| Dolan & Moré 2002 — performance profiles; Moré & Wild 2009 — data profiles | optimizer benchmarking | ○ |
| Bartz-Beielstein et al. 2020 — "Benchmarking in optimization: best practice and open issues" (arXiv 2007.03488) | | ○ |
| Hansen et al. 2021, COCO platform (*Optim. Methods Softw.*) | anytime ECDFs | ○ |

---

## 11. Gap statement (draft)

Across disciplines the surveys agree that (i) no surrogate dominates (Jin 2001; Fuhg 2021; Williams & Cremaschi 2021;
Lüthen 2021), (ii) the right choice depends on landscape features and the downstream task, and (iii) GP/kriging
dominates *engineering* practice (≈34 % of SBO applications) while tree ensembles dominate *tabular ML*. What we did not
find: a study that **jointly** varies (a) surface class, (b) data budget, (c) noise level and type, (d) fidelity
structure and cost ratio, (e) scout/acquisition × deliverable family, (f) upscaling step, and (g) downstream optimizer
type, and scores the end-to-end result by **true-function regret per unit of simulation cost**, while explaining
the tabular-ML vs engineering disagreement. That is the paper.

## 12. Search log

| Date | Query theme | Notes |
|---|---|---|
| 2026-10-03 | engineering surrogate reviews; MF surveys; tabular trees vs NN; adaptive sampling surveys; SBO benchmarks; water-resources review; HPO surrogate benchmarks; KOH/recursive co-kriging; NARGP/MF-NN; tree UQ; SMAC/TPE/TuRBO; DL-UQ surveys; Needels/Alonso; Williams–Cremaschi; MF benchmarks; Fuhg; TabPFN; Alizadeh; GMDH; AAA; MBA/P-splines/THB; BART/TGP; cosmology & building emulators; sparse PCE; Jin 2011; test suites; ensembles; hierarchical kriging; Jin–Chen–Simpson; ELA; space mapping; mf2/Toal; stochastic kriging; KAN/Sobolev; Bull/Wendland; reliability AL; FEM surrogates | 36 searches, first pass |
| 2026-10-03 | change-of-basis family: deep kernel learning; active subspaces / ridge; compressed & randomised DMD; CS-PCE; SINDy autoencoder; REMBO / latent BO; kernel flows; PPR | 6 searches |
| 2026-10-03 | breadth pass 2: reservoir proxies, ClimateBench, finance, ABM calibration, history matching, nuclear, crash, turbomachinery, AM; GP-vs-GBT small-data; YAHPO; kriging origins; DACE; DKL pitfalls; Gramacy/Kleijnen books | 12 searches |
| 2026-10-03 | verification: 254 refs via Crossref/arXiv + audit; 17 open-access full texts read (DEPTH_NOTES.md) | done |
| later (optional) | geostatistics history; petroleum proxy models; finance/option-pricing surrogates; pharmacometrics/systems biology emulators; agent-based-model calibration; nuclear engineering; climate emulators (e.g. ClimateBench); automotive crash; turbomachinery; additive manufacturing; Russian/Soviet approximation literature; information-geometric / optimal-recovery views | breadth pass 2 |
