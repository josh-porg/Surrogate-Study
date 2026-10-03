# Mathematics

Formal statement of every object in the study. Notation is fixed here and reused in STUDY.md and the paper.
Equations are numbered `(M.x)` so other documents can cite them.

---

## 1. Problem setting

- Input domain $\mathcal X=[0,1]^d$ (inputs mapped affinely or log-affinely to the unit cube).
- Expensive truth $f:\mathcal X\to\mathbb R$. Observations $y_i=f(x_i)+\varepsilon_i$.
- Fidelity levels $f_1,\dots,f_L$ with $f_L=f$, unit costs $c_1<\dots<c_L$.
- Budget $B$; a design $D=\{(x_i,\ell_i,y_i)\}$ is admissible if $\sum_i c_{\ell_i}\le B$.
- A **pipeline** $\Pi$ maps a design to a deliverable surrogate $s=\Pi(D)$, which an optimizer $\mathcal O$ uses to return
  $\hat x=\mathcal O(s)$.

The engineering question is not "is $s$ close to $f$" but

$$
\text{regret}(\Pi,\mathcal O;B)=f(\hat x)-\min_{x\in\mathcal X}f(x),\qquad \hat x=\mathcal O(\Pi(D_B)). \tag{M.1}
$$

For constrained problems $\min f$ s.t. $g(x)\le 0$, regret is measured on the true $f$, with a feasibility indicator on the
true $g$.

### 1.1 Noise models (factor N in the study)
| Type | Model | Where it arises |
|---|---|---|
| none | $\varepsilon=0$ | analytic codes |
| homoscedastic additive | $\varepsilon\sim\mathcal N(0,\sigma^2)$ | |
| multiplicative (thesis) | $y=f(1+\eta)$, $\eta\sim\mathcal N(0,\sigma_r^2)$ | relative FEA scatter |
| heteroscedastic | $\varepsilon\sim\mathcal N(0,\sigma^2(x))$ | solver difficulty varies |
| heavy-tailed / outliers | Student-$t_\nu$ or mixture $(1-p)\mathcal N+p\,\mathcal N(0,k^2\sigma^2)$ | non-converged runs |
| deterministic "numerical" noise | $y=f+\sigma\,h(\omega x)$, $h$ high-frequency, **repeatable** | mesh/tolerance noise; replicates give zero variance |

Deterministic noise matters because replicate-based noise estimates return 0, GP nuggets still fit it, and
gradient-based optimizers see spurious minima at scale $\sim 1/\omega$.

### 1.2 Multi-fidelity test construction
For an analytic $f$, low-fidelity versions with controllable correlation $r$ and structure:
$$
f_1(x)=a\,f(x)+b\,\phi(x)+c,\qquad \phi\ \text{a smooth or rough discrepancy}, \tag{M.2}
$$
with $(a,b)$ solved to hit a target Pearson $r=\operatorname{corr}_{x\sim U}(f,f_1)$ and also a nonlinear variant
$f_1=\psi(f)+b\phi$ (monotone $\psi$, e.g. $\tanh$) to separate AR1-compatible from nonlinear relationships. Cost ratio
$\kappa=c_L/c_1\in\{3,10,30,100,1000\}$.

---

## 2. Error metrics

Let $s$ be a deliverable, $T=\{x_j\}_{j=1}^{M}$ an independent test set ($M\ge10^4$, Sobol') or a quadrature rule.

$$
\text{RMSE}=\Big(\tfrac1M\sum_j (s(x_j)-f(x_j))^2\Big)^{1/2},\quad
\text{NRMSE}=\text{RMSE}/\operatorname{sd}_T(f),\quad
\text{rel-RMSE}=\Big(\tfrac1M\sum_j\big(\tfrac{s-f}{f}\big)^2\Big)^{1/2}. \tag{M.3}
$$
Also $L^\infty$ (max) error, $R^2$, and **derivative error** in a Sobolev seminorm
$$
|s-f|_{H^1}^2\approx\tfrac1M\sum_j\|\nabla s(x_j)-\nabla f(x_j)\|^2,\qquad |s-f|_{H^2}\ \text{via Hessians}. \tag{M.4}
$$
Ranking fidelity: Kendall $\tau$ and Spearman $\rho$ between $s(T)$ and $f(T)$, and top-$q$ overlap
(fraction of the true best $q$-quantile retrieved).
Calibration (if $s$ gives $\mu,\sigma$): NLPD, CRPS, empirical coverage of the $1-\alpha$ interval, and
$z$-score variance.

### 2.1 The thesis metric and why it changes rankings (B-01)
Thesis canonical CV used noisy targets. For a deliverable $s$ trained independently of a test point,
$$
\mathbb E\big[(s(x)-y)^2\big]=\mathbb E\big[(s(x)-f(x))^2\big]+\sigma^2(x). \tag{M.5}
$$
So noisy CV-MSE = clean MSE + noise variance. With multiplicative noise and $f\approx1.3$, relative noisy RMSE
$\ge\sigma_r$. Two models with clean RMSE $e_1<e_2$ appear as $\sqrt{e_1^2+\sigma^2}$ vs $\sqrt{e_2^2+\sigma^2}$;
the ratio shrinks toward 1 as $\sigma$ grows — e.g. clean 0.5 % vs 2 % (4×) becomes 3.04 % vs 3.61 % (1.19×) at
$\sigma=3\%$. This is the compression seen in thesis Tables canonical_a/b.

### 2.2 Noise-level estimation (B-06)
Thesis estimator: $\hat\sigma=\operatorname{med}|\Delta^2 y|/\operatorname{med}(y)$. For i.i.d. $\varepsilon$,
$\Delta^2\varepsilon_i=\varepsilon_{i+1}-2\varepsilon_i+\varepsilon_{i-1}$ has variance $6\sigma^2$, and for Gaussian
$\operatorname{med}|Z|=0.6745\,\mathrm{sd}(Z)$, so
$$
\operatorname{med}|\Delta^2\varepsilon|\approx0.6745\sqrt6\,\sigma\approx1.65\,\sigma, \tag{M.6}
$$
plus a curvature bias $\approx|f''|h^2$. Corrected (Gasser–Sroka–Jennen-type) estimator:
$\hat\sigma=\operatorname{med}|\Delta^2y|/(0.6745\sqrt6)$ on locally sorted neighbours, cross-checked against the GP
nugget MLE.

---

## 3. Surrogate families

### 3.1 Gaussian process / kriging
Prior $f\sim\mathcal{GP}(m(x),k_\theta(x,x'))$, Matérn-$\nu$ ARD kernel
$$
k(x,x')=\sigma_f^2\frac{2^{1-\nu}}{\Gamma(\nu)}\big(\sqrt{2\nu}\,r\big)^\nu K_\nu\big(\sqrt{2\nu}\,r\big),\quad
r^2=\sum_{k=1}^d\frac{(x_k-x'_k)^2}{\ell_k^2}. \tag{M.7}
$$
With $K=k(X,X)+\sigma_n^2 I$:
$$
\mu(x)=m(x)+k_x^\top K^{-1}(y-m),\qquad \sigma^2(x)=k(x,x)-k_x^\top K^{-1}k_x. \tag{M.8}
$$
Hyperparameters by maximising $\log p(y\mid\theta)=-\tfrac12 (y-m)^\top K^{-1}(y-m)-\tfrac12\log|K|-\tfrac n2\log2\pi$.
Cost: fit $O(n^3)$ per likelihood evaluation, predict mean $O(n)$, variance $O(n^2)$ per point.

**Key fact for B-02:** $\sigma^2(x)$ in (M.8) does not contain $y$. With $\theta$ fixed, greedy
$x_{n+1}=\arg\max\sigma^2(x)$ is a deterministic function of $X$ alone — a space-filling rule. $y$ enters only via
$\hat\theta(y)$.

Mean function variants: constant (ordinary kriging), polynomial (universal), physics dictionary (thesis
`gp_physics_mean`), low-fidelity model (hierarchical kriging, §5).

Heteroscedastic / stochastic kriging: $K=k(X,X)+\operatorname{diag}(\sigma_n^2(x_i))$ with $\log\sigma_n^2$ itself a GP.

### 3.2 RBF interpolation and regression
$$
s(x)=\sum_{i=1}^n w_i\,\varphi(\|x-x_i\|)+\sum_{j}\beta_j p_j(x),\qquad
\begin{pmatrix}\Phi+\lambda I&P\\P^\top&0\end{pmatrix}\begin{pmatrix}w\\\beta\end{pmatrix}=\begin{pmatrix}y\\0\end{pmatrix}. \tag{M.9}
$$
$\lambda=0$ interpolates; $\lambda>0$ is the RBF analogue of a nugget. With conditionally positive definite $\varphi$
the RBF interpolant equals the kriging mean with a generalised covariance (Matheron); thin-plate $\varphi=r^2\log r$
(2-D) minimises the bending energy.

### 3.3 Polynomial chaos / sparse regression
$s(x)=\sum_{\alpha\in\mathcal A}c_\alpha\Psi_\alpha(x)$ with $\Psi_\alpha$ orthonormal (Legendre on uniform inputs).
Sparse: $\min_c\|y-\Psi c\|_2^2+\lambda\|c\|_1$ (LASSO/LARS) with $\mathcal A$ by total degree $p$ and hyperbolic
truncation $\|\alpha\|_q\le p$, $q<1$. The thesis `sparse_physics` is the same with a non-orthogonal physics dictionary.

### 3.4 Splines
Univariate B-splines of order $k$ (degree $k-1$) on knots $t$, Cox–de Boor recursion
$$
B_{i,1}(u)=\mathbf 1[t_i\le u<t_{i+1}],\quad
B_{i,k}(u)=\frac{u-t_i}{t_{i+k-1}-t_i}B_{i,k-1}(u)+\frac{t_{i+k}-u}{t_{i+k}-t_{i+1}}B_{i+1,k-1}(u). \tag{M.10}
$$
Tensor product $s(x)=\sum_{i_1\cdots i_d}c_{i_1\cdots i_d}\prod_k B_{i_k,k}(x_k)$; with $m$ knots per axis the number of
coefficients is $\sim m^d$.

**P-spline (direct on scattered data):** $\min_c\|y-Bc\|^2+\sum_k\lambda_k\|D_k^{(q)}c\|^2$ with $D^{(q)}$ a $q$-th order
difference matrix along axis $k$ — fits splines to scattered noisy data without an intermediate model.

**Interpolation error (cubic, $k=4$, mesh $h$, $f\in C^4$):**
$$
\|f-I_hf\|_\infty\le C\,h^4\|f^{(4)}\|_\infty,\qquad \|(f-I_hf)^{(r)}\|_\infty\le C_r h^{4-r}\|f^{(4)}\|_\infty. \tag{M.11}
$$
Derivatives lose one order each — relevant because IPOPT consumes gradients and Hessians.

**Smolyak sparse grid:** $A(q,d)=\sum_{q-d+1\le|\mathbf i|\le q}(-1)^{q-|\mathbf i|}\binom{d-1}{q-|\mathbf i|}
(U^{i_1}\otimes\dots\otimes U^{i_d})$; point count $O(N\log^{d-1}N)$ vs $N^d$ for full tensor grids at the same
1-D resolution, under mixed (dominating-derivative) smoothness.

### 3.5 MARS
$s(x)=\beta_0+\sum_m\beta_m\prod_{k\in K_m}[\pm(x_k-t_{km})]_+$ — forward stepwise hinge addition, backward GCV pruning.
C⁰ (kinks are representable exactly: suited to `clamp_floor`).

### 3.6 Trees and ensembles
A regression tree is $T(x)=\sum_{\ell}v_\ell\mathbf 1[x\in R_\ell]$ (axis-aligned boxes).
Random forest: $s(x)=\frac1B\sum_bT_b(x)$, trees on bootstrap samples with feature subsampling.
Gradient boosting: $s_M(x)=s_0+\nu\sum_{m=1}^MT_m(x)$, each $T_m$ fit to the negative gradient of the loss at
$s_{m-1}$; XGBoost adds $\Omega(T)=\gamma|T|+\tfrac12\lambda\|v\|^2$.

Properties that matter here:
- $\nabla s=0$ almost everywhere; $s$ is discontinuous on split hyperplanes → unusable by gradient optimizers
  without upscaling; harmless for DE/CMA-ES/Nelder–Mead.
- Invariant to monotone transforms of each input (orientation-preserving, axis-aligned) — robust to uninformative
  inputs and badly scaled inputs (Grinsztajn et al. 2022).
- Can represent regime changes, plateaus and floors exactly; smooth kernels cannot (Gibbs-like overshoot).
- Approximation rate for Lipschitz $f$ with $n$ cells in $d$ dims: piecewise constant error $O(n^{-1/d})$, slower than
  kernels' $O(h^{\nu})$ for smooth $f$ — so trees should lose on smooth surfaces at small n, and the GBT real-cache win
  points to **non-smoothness or noise structure** in the FEA data (hypothesis H3).

### 3.7 Neural networks
MLP $s(x)=W_L\sigma(\cdots\sigma(W_1x+b_1)\cdots)+b_L$. Deep ensemble: $\mu=\frac1K\sum s_k$,
$\sigma^2=\frac1K\sum(s_k-\mu)^2(+\text{aleatoric heads})$.
Sobolev training: loss $\sum_i(s(x_i)-y_i)^2+\gamma\|\nabla s(x_i)-g_i\|^2$.
KAN: $s(x)=\sum_{q}\Phi_q\big(\sum_p\phi_{q,p}(x_p)\big)$ with $\phi,\Phi$ learned B-splines (Kolmogorov–Arnold form).

### 3.8 Ensembles of surrogates
Weighted average $s=\sum_m w_ms_m$, $\sum w_m=1$. PRESS-based weights (Goel 2007):
$w_m\propto(E_m+\alpha\bar E)^\beta$, $\beta<0$, $E_m$ the LOO-CV error. Optimal (Viana 2009): minimise
$w^\top Cw$ s.t. $\mathbf 1^\top w=1$, $C_{ij}=\frac1n e_i^\top e_j$ from CV residual vectors.

### 3.9 Partitioned (regime) models
$s(x)=\sum_r\mathbf 1[x\in\mathcal R_r]s_r(x)$ (hard) or $\sum_r\pi_r(x)s_r(x)$ with gating $\pi$ (soft, mixture of
experts). Treed GP learns $\{\mathcal R_r\}$ by Bayesian CART. For the real FEA problem $\mathcal R$ is partly known
(floor-bound / feasible / infeasible) — a classifier $c(x)$ plus a regressor on the feasible region.

---

### 3.10 Change-of-basis / latent surrogates (family F10)

General form: an input map $\phi:\mathcal X\to\mathcal Z$, a basis $\{\psi_\alpha\}$ on $\mathcal Z$ (or a learner $g$), and
$$
s(x)=g\big(\phi(x)\big)=\sum_{\alpha\in\mathcal A}c_\alpha\,\psi_\alpha\big(\phi(x)\big). \tag{M.19}
$$
The bet is that $f\circ\phi^{-1}$ is *simpler* than $f$: sparse ($|\mathcal A|$ small), low-rank, low-dimensional or
separable. Each sub-family fixes or learns a different part.

**(a) Linear input reduction.** $\phi(x)=W^\top x$, $W\in\mathbb R^{d\times k}$, $W^\top W=I$.
Active subspace: $C=\mathbb E[\nabla f\,\nabla f^\top]=V\Lambda V^\top$, $W=V_{:,1:k}$; if $\lambda_{k+1:d}$ are small then
$\mathbb E\big[(f(x)-\mathbb E[f\mid W^\top x])^2\big]\le c\,(\lambda_{k+1}+\dots+\lambda_d)$ (Poincaré-type bound). $\nabla f$ is
estimated from a GP mean, a local-linear fit, or adjoints. Ridge approximation fits $W$ and $g$ jointly:
$\min_{W\in\mathrm{Gr}(k,d),\,g\in\mathcal P_p}\sum_i(y_i-g(W^\top x_i))^2$ (variable projection on the Grassmannian).
PPR: $s(x)=\sum_{j=1}^J g_j(w_j^\top x)$, fitted stage-wise with 1-D smoothers.

**(b) Nonlinear co-learning.** Deep kernel learning: $k_{\theta,w}(x,x')=k_\theta\big(\phi_w(x),\phi_w(x')\big)$,
$$
(\hat\theta,\hat w)=\arg\max_{\theta,w}\ \log p(y\mid X,\theta,w). \tag{M.20}
$$
With $\dim w\gg n$ the marginal likelihood can be driven up by making $\phi_w$ collapse inputs to fit $y$ (over-correlation);
mitigations: small $\phi$, weight priors (fully Bayesian over $w$), kernel-flow objective
$\rho=\|u_{\text{half}}-u_{\text{full}}\|^2_{k}/\|u_{\text{full}}\|^2_{k}$ (Owhadi & Yoo) minimised over random half-subsets instead of ML-II.
Static SINDy-autoencoder analogue:
$$
\min_{w,\xi}\ \tfrac1n\sum_i\big(y_i-\Theta(\phi_w(x_i))\,\xi\big)^2+\lambda\|\xi\|_1\ \ (+\ \mu\|x_i-\psi(\phi_w(x_i))\|^2), \tag{M.21}
$$
$\Theta$ a dictionary (polynomials, physics terms). The reconstruction term is optional: for a static map with
uniformly sampled $x$, an unsupervised autoencoder learns the sampling density, which carries no information about $f$;
only the supervised term finds a useful $\phi$. For multi-fidelity, feed $(x,f_L(x))$ to $\phi$.

**(c) Sparse basis + compressed sensing.** Orthonormal basis (Fourier, Legendre, Chebyshev, wavelet), measurement
matrix $\Psi_{i\alpha}=\psi_\alpha(x_i)$, $|\mathcal A|=N\gg n$:
$$
\hat c=\arg\min\|c\|_1\ \text{s.t.}\ \|\Psi c-y\|_2\le\eta. \tag{M.22}
$$
If $f$ is (approximately) $s$-sparse and $\Psi/\sqrt n$ satisfies RIP of order $2s$ — which holds with high probability for
bounded orthonormal systems when $n\gtrsim K^2 s\log^3 s\log N$ samples are drawn from the orthogonality measure ($K$ =
sup-norm bound of the basis) — then
$\|f-s\|_{L^2}\lesssim\sigma_s(c)_1/\sqrt s+\eta$, with $\sigma_s(c)_1$ the best $s$-term $\ell_1$ error. Two consequences for
the study: (i) the sample count scales with the sparsity $s$, so oscillatory surfaces (sparse in Fourier; the thesis
`oscillatory` case where GP-variance scouting failed) should be cheap in this family; (ii) RIP needs *random* samples from
the right measure (Chebyshev density for polynomial bases) — adaptive or maximin designs can break the guarantee. This is a
sampling × learner coupling that will show up in the scout × deliverable factorial.

**(d) Low-rank (the SVD of a surface).** In 2-D, $f(x_1,x_2)=\sum_r\sigma_r u_r(x_1)v_r(x_2)$ (Schmidt/functional SVD);
truncation at rank $R$ has $L^2$ error $(\sum_{r>R}\sigma_r^2)^{1/2}$. In $d$ dims, tensor train:
$$
f(x)\approx G_1(x_1)G_2(x_2)\cdots G_d(x_d),\qquad G_k(x_k)\in\mathbb R^{r_{k-1}\times r_k},\ r_0=r_d=1, \tag{M.23}
$$
storage $O(d\,m\,r^2)$ instead of $m^d$; TT-cross builds it from $O(d\,m\,r^2)$ adaptively chosen evaluations (the
sample-selection rule is part of the method — another built-in "scout"). Additive functions have TT-rank 2; smooth
products have rank 1; rotated ridges have high rank in the original axes but rank 1 after an active-subspace rotation —
hence (a)∘(d) combinations.

**(e) Compression for speed (compressed-DMD analogue).** Randomised range finder: $\Omega\in\mathbb R^{N\times(k+p)}$
Gaussian, $Y=A\Omega$, $Q=\mathrm{qr}(Y)$, $B=Q^\top A$, SVD of the small $B$. Expected error
$\mathbb E\|A-QQ^\top A\|_F\le\big(1+\tfrac{k}{p-1}\big)^{1/2}\big(\sum_{j>k}\sigma_j^2\big)^{1/2}$ (Halko–Martinsson–Tropp).
Compressed DMD does the same on $A$ sketched by a measurement matrix, with cost scaling with intrinsic rank, not ambient size.
Our analogues, each a speed/accuracy trade measured explicitly:
1. **Compressed upscaling:** compress the baked grid tensor $M\in\mathbb R^{m^d}$ by randomised HOSVD / TT-SVD; spline the
   factor matrices instead of the full grid. Error = (M.15) learning + upscaling + truncation $\varepsilon_{\text{TT}}$.
   Or skip the grid and TT-cross the intermediate model directly.
2. **Compressed scout refits:** GP with Nyström rank-$q$ or random Fourier features, $O(nq^2)$ instead of $O(n^3)$ per AL step.
3. **Sketched regression** for P-spline / PCE / CS fits on large dictionaries.
None removes a stage; each makes a stage cheaper, which matters only when fit cost is non-negligible vs $c_H$ (M.18) —
i.e. for cheap simulators, large $n$, or high $d$.

**(f) Pipeline co-search.** Choose $(\phi,\{\psi\},g,\text{hyper-params})$ by
$\min_{\text{pipeline}}\widehat{\mathrm{err}}_{\text{nested-CV}}$ s.t. fit time $\le\tau$ — the CASH problem. Selection
optimism (B-05) applies: report the outer-fold error only.

## 4. Uncertainty estimators and acquisition functions

**Committee / bootstrap:** $\hat\sigma^2(x)=\frac1{K-1}\sum_k(s_k(x)-\bar s(x))^2$. With $K=4$ the relative standard error
of $\hat\sigma$ is $\approx1/\sqrt{2(K-1)}\approx0.41$ (B-16).

**Infinitesimal jackknife (RF):** $\hat V_{IJ}(x)=\sum_{i=1}^n\operatorname{Cov}_b(N_{bi},T_b(x))^2$, $N_{bi}$ = times
point $i$ appears in bootstrap $b$ (Wager et al. 2014), with finite-$B$ bias correction.

**Split conformal:** calibration residuals $R_i=|y_i-s(x_i)|$, $\hat q=$ the $\lceil(n_c+1)(1-\alpha)\rceil$-th smallest;
interval $s(x)\pm\hat q$ — marginal coverage $\ge1-\alpha$ regardless of model. Locally adaptive: normalise by
$\hat\sigma(x)$.

**Acquisitions (global accuracy):**
$$
\text{MaxVar: }\arg\max\sigma^2(x);\quad
\text{IMSE/ALC: }\arg\min_{\tilde x}\int\sigma^2_{D\cup\tilde x}(x)\,dx;\quad
\text{EIGF: }(\mu(x)-y_{nn(x)})^2+\sigma^2(x). \tag{M.12}
$$
LOLA-Voronoi: $H(x_i)=\text{Voronoi volume}_i\times\text{local-linear-fit residual}_i$, sample in the largest cell of the
top-ranked point.

**Acquisitions (optimization):** with incumbent $f^\*$, $z=(f^\*-\mu)/\sigma$,
$$
\text{EI}(x)=\sigma(x)\,[z\Phi(z)+\phi(z)],\qquad \text{UCB/LCB}(x)=\mu(x)-\beta^{1/2}\sigma(x). \tag{M.13}
$$
Constrained EI multiplies by $P(g(x)\le0)$. Cost-aware: divide acquisition by $c_\ell$.

---

## 5. Multi-fidelity formulations

**AR1 co-kriging (Kennedy–O'Hagan):**
$$
f_\ell(x)=\rho_{\ell-1}f_{\ell-1}(x)+\delta_\ell(x),\qquad \delta_\ell\sim\mathcal{GP}\ \perp\ f_{\ell-1}. \tag{M.14}
$$
Recursive (Le Gratiet): fit $f_1$ GP, then for nested designs $X_\ell\subset X_{\ell-1}$ fit $\rho_{\ell-1}$ and $\delta_\ell$ by GLS
on $y_\ell-\rho\,\mu_{\ell-1}(X_\ell)$; same predictor as KOH with $L$ independent $O(n_\ell^3)$ fits.

**Hierarchical kriging (Han–Görtz):** $f_H(x)=\beta_0\,\hat f_L(x)+Z(x)$ — the LF predictor is the trend.

**NARGP (Perdikaris):** $f_\ell(x)=z_\ell\big(f_{\ell-1}(x),x\big)$, $z_\ell\sim\mathcal{GP}$ with kernel
$k=k_\rho(x,x')\,k_f(f_{\ell-1}(x),f_{\ell-1}(x'))+k_\delta(x,x')$; $f_{\ell-1}$ replaced by its posterior mean (or MC).

**Bridge (correction) functions:** additive $f_H\approx f_L+\delta$, multiplicative $f_H\approx\beta(x)f_L$,
comprehensive $f_H\approx\beta(x)f_L+\delta(x)$. **The thesis TMF target is the multiplicative bridge**
$t_{\text{fem}}=\text{TMF}(x)\cdot t_{\text{scalar}}(x)$, with $t_{\text{scalar}}$ the zero-cost LF model.

**Feature-augmented MF (for any learner, incl. GBT/NN):** train $s_H$ on inputs $(x,\hat f_L(x))$. AR1 is the linear-GP
special case; NARGP the GP-kernel special case; for trees it lets splits act on the LF prediction directly.

**Space mapping:** $f_H(x)\approx f_L(P(x))$, $P$ affine, fit by parameter extraction
$P=\arg\min\sum_i\|f_H(x_i)-f_L(P(x_i))\|^2$.

**Cost model and when MF pays.** Equal-cost comparison: HF-only uses $n_H^{(0)}=B/c_H$; MF uses
$n_Hc_H+n_Lc_L=B$. With LF fraction $\phi=n_Lc_L/B$, Toal's (2015) empirical guidance: benefit requires $r^2\gtrsim0.9^2$
and $0.1\lesssim\phi\lesssim0.8$. Our factor grid sweeps $(r,\kappa,\phi)$ to map this region for every learner, not only
co-kriging. A simple analytic heuristic for AR1 with known $\rho$: residual variance
$\operatorname{Var}(\delta)=(1-r^2)\operatorname{Var}(f_H)$, so the MF error at $n_H$ points behaves like an HF-only fit of a
function whose variance is reduced by $(1-r^2)$ — plus the LF model error at $n_L$ points.

---

## 6. Upscaling (two-stage) error decomposition

Pipeline: intermediate $m=\mathcal M(D)$, grid $G_h$ with spacing $h$, interpolant $I_h$. Deliverable
$s=I_h[m|_{G_h}]$. By the triangle inequality,
$$
\|f-s\|\le\underbrace{\|f-m\|}_{\text{learning error}}+\underbrace{\|m-I_hm\|}_{\text{upscaling error}}. \tag{M.15}
$$
For cubic splines and smooth $m$, the second term is $O(h^4\|m^{(4)}\|)$ (M.11). Implications:
1. Upscaling can only *add* error unless $I_h$ also smooths (e.g. smoothing spline acts as a low-pass filter on an
   over-fit $m$; then $\|f-s\|<\|f-m\|$ is possible — a testable claim: **does upscaling regularise a rough $m$ such as
   GBT?**).
2. For a non-smooth $m$ (trees), $\|m^{(4)}\|$ is unbounded: the grid samples a staircase; the spline interpolant
   rings between steps. Error then scales like $O(\text{jump}\times\text{Gibbs fraction})$ and does not vanish with
   $h\to0$ — finer grids reproduce the staircase more faithfully, which may be *worse* for gradient optimizers.
3. Grid cost $N=m^d$ points; at $m=12$, $d=3$: 1 728; $d=6$: 3·10⁶; $d=10$: 6·10¹⁰ — tensor upscaling stops being
   viable around $d\approx6$. Sparse grids, MBA, THB and P-splines on scattered data are the alternatives.

Derivative version: $|f-s|_{H^r}\le|f-m|_{H^r}+|m-I_hm|_{H^r}$, the second term $O(h^{4-r})$.

---

## 7. Optimizer on the surrogate

Let $\hat x\in\arg\min s$ and $x^\*\in\arg\min f$. Then
$$
f(\hat x)-f(x^\*)\le\big[f(\hat x)-s(\hat x)\big]+\big[s(x^\*)-f(x^\*)\big]\le2\|f-s\|_\infty. \tag{M.16}
$$
This is the only link between accuracy and regret, and it is (a) a sup-norm bound — RMSE does not control it — and
(b) loose: an $s$ with large error away from the optimum but correct near $x^\*$ has small regret. Sharper:
$$
f(\hat x)-f(x^\*)\le2\sup_{x\in\mathcal L_\epsilon}|f-s|,\qquad \mathcal L_\epsilon=\{x:f(x)\le f^\*+\epsilon\}\cup\{x:s(x)\le s^\*+\epsilon\}, \tag{M.17}
$$
so only accuracy on the (true and surrogate) near-optimal sublevel sets matters. **Hence "most globally accurate ≠
best optimum" is expected, not paradoxical** — a key message of the paper. Ranking preservation (Kendall τ on
$\mathcal L_\epsilon$) is the right proxy for comparison-based optimizers (CMA-ES, DE, Nelder–Mead); sup-norm accuracy
plus gradient accuracy near $x^\*$ is the right proxy for gradient-based optimizers.

For gradient-based local solvers, a KKT point $\hat x$ of $s$ satisfies $\|\nabla f(\hat x)\|\le|f-s|_{W^{1,\infty}}$;
spurious stationary points arise wherever $\nabla s$ vanishes but $\nabla f$ does not (trees everywhere; interpolants
with ringing).

Optimizer factor in the study: IPOPT/SLSQP (gradient, local) from multistart; L-BFGS-B; Nelder–Mead; COBYLA; DE;
CMA-ES; DIRECT; plus surrogate-managed loops (EGO, TuRBO, trust-region model management) where the optimizer can call
$f$ again. Scoring: regret (M.1), distance $\|\hat x-x^\*\|$, success rate (regret $<\epsilon$), function-evaluation
cost on $s$ and on $f$.

---

## 8. Cost accounting

Total cost of a pipeline run:
$$
C_{\text{tot}}=\sum_\ell n_\ell c_\ell+C_{\text{fit}}(n)+N_{\text{opt}}\,c_{\text{pred}}+C_{\text{upscale}}. \tag{M.18}
$$
For FEA ($c_H\approx1$ CPU-h measured: mean wall 1.07 h over 901 runs in `tmf_bbd96c4b56`) every surrogate cost is
negligible and the comparison reduces to $n_\ell$. For cheap simulators (thermochemistry: thesis found direct uniform
sampling beat surrogates) the fit/predict terms dominate — the **crossover cost ratio** at which surrogate
sophistication pays is a reported quantity. We sweep a hypothetical $c_H\in\{10^{-3},10^{-1},10^{1},10^{3}\}$ s.

---

## 9. Statistics for the comparison

- **Paired design / common random numbers:** for seed $s$, every pipeline sees the same initial design, noise draws,
  test set and optimizer starts.
- **Per-problem:** Wilcoxon signed-rank on paired log-errors; Holm correction. Minimum attainable two-sided $p$ with
  $n$ pairs is $2^{1-n}$: $n=8\Rightarrow0.0078$ (thesis); $n=30\Rightarrow1.9\cdot10^{-9}$.
- **Across problems:** Friedman test on mean ranks, Nemenyi post-hoc, critical-difference diagrams (Demšar 2006);
  Bayesian hierarchical signed-rank (Benavoli et al. 2017) with ROPE.
- **Profiles:** performance profiles $\rho_\Pi(\tau)=\frac1{|P|}|\{p:r_{p,\Pi}\le\tau\}|$ (Dolan–Moré) with
  $r_{p,\Pi}=\text{err}_{p,\Pi}/\min_\Pi\text{err}_{p,\Pi}$; data profiles vs budget (Moré–Wild).
- **Explaining variation:** mixed-effects model
  $\log\text{err}=\beta_0+\text{pipeline}\times(\log n+\sigma+r+\kappa+\text{ELA features}+d)+(1\mid\text{problem})+\epsilon$
  — answers "how does favourability scale with data, noise, fidelity, cost, surface type".
- **Power:** pilot effect sizes → choose seeds so a 10 % relative error difference is detected at 80 % power.

---

## 10. Landscape features (surface-type covariates)

From ELA (Mersmann 2011; Kerschke & Trautmann 2019): meta-model fit quality (linear/quadratic $R^2$, condition
number of quadratic Hessian), y-distribution skewness/kurtosis, level-set separability, dispersion, nearest-better
clustering (multimodality), information content (ruggedness), plus our own: estimated smoothness (Matérn-$\nu$ MLE),
anisotropy ratio $\max\ell_k/\min\ell_k$, fraction of domain with $|\nabla^2 f|$ above a threshold (localisation),
discontinuity indicator, noise-to-signal ratio. These are computed on the initial design so that a **selection rule**
(which pipeline to use) can be learned and validated out-of-sample.

---

## 11. Computational complexity (time and memory)

Notation: $n$ training points, $d$ inputs, $M$ query points, $P$ basis/dictionary size, $q$ inducing points or random
features, $B$ trees, $T$ boosting rounds, $D$ tree depth, $W$ network weights, $E$ epochs, $K$ ensemble members,
$m$ grid points per axis, $k$ spline order ($k=4$ cubic), $r$ tensor rank, $s$ sparsity, $L$ fidelity levels with $n_\ell$
points, $H$ hyper-parameter optimiser iterations, $R$ restarts. "Fit" includes hyper-parameter learning where the method
needs it (factor $HR$); "predict" is per query point. Entries are standard asymptotic costs of the usual algorithm.
Constants matter at our sizes, so costs are also **measured** in E0/E2 (timing table): the table says how costs scale,
the measurements say which is cheaper at a given $n, d$.

### 11.1 Learners
| Method | Fit time | Fit memory | Predict (mean) | Predict (uncertainty) | Notes |
|---|---|---|---|---|---|
| Polynomial RSM, degree $p$ | $O(nP^2+P^3)$, $P=\binom{d+p}{p}$ | $O(nP)$ | $O(P)$ | $O(P^2)$ (OLS variance) | $P\sim d^p/p!$ |
| Sparse regression / sparse PCE (LARS, OMP, LASSO-CD) | LARS $O(nP\min(n,P))$; OMP $O(snP)$; CD $O(nP)$ per sweep | $O(nP)$ | $O(s)$ | bootstrap $\times K$ | total-degree $P$ grows combinatorially in $d$; hyperbolic truncation tames it |
| Exact GP / kriging | $O(HR\,n^3)$ (Cholesky per likelihood evaluation) | $O(n^2)$ | $O(nd)$ | $O(n^2)$ | 8 B·$n^2$: $n=10^4$ → 0.8 GB; rank-1 updates make an AL step $O(n^2)$ |
| Sparse / inducing GP (FITC, SVGP), Nyström | $O(nq^2)$ | $O(nq)$ | $O(q)$ | $O(q^2)$ | $q\ll n$ |
| Random-feature KRR / GP | $O(nq^2+q^3)$ | $O(nq)$ | $O(qd)$ | $O(q^2)$ | |
| Local GP (laGP), neighbourhood $k_n$ | none (lazy) | $O(n)$ | $O(k_n^3)$ per point | included | parallel over queries |
| Heteroscedastic GP / stochastic kriging | $\approx2\times$ exact GP | $O(n^2)$ | $O(nd)$ | $O(n^2)$ | |
| Treed GP (Bayesian), $S$ MCMC sweeps | $O(S\sum_j n_j^3)$ over leaves $j$ | $O(\sum_j n_j^2)$ | $O(S\,n_j)$ | sample-based | partitioning *reduces* the cubic cost |
| Deep kernel learning (exact GP) | $O(E\,(n^3+nW))$ | $O(n^2+W)$ | $O(nd+W)$ | $O(n^2)$ | with structured interpolation (SKI) $\approx O(En)$ |
| RBF interpolation / regression (dense) | $O(n^3)$ | $O(n^2)$ | $O(nd)$ | power function $O(n^2)$ | compact support: sparse; fast multipole: $O(n\log n)$ |
| SVR / LS-SVM | QP $O(n^2)$–$O(n^3)$; LS-SVM $O(n^3)$ | $O(n^2)$ kernel | $O(n_{sv}d)$ | none | |
| IDW (Shepard), $k$-NN | $O(1)$ store; kd-tree build $O(dn\log n)$ | $O(nd)$ | brute $O(nd)$; kd-tree $\approx O(\log n)$ for $d\lesssim10$ | none | kd-trees degrade to brute force as $d$ grows |
| Natural neighbour / Delaunay-linear | Delaunay $O(n\log n)$ for $d\le3$, $O(n^{\lceil d/2\rceil})$ worst case | number of simplices | $O(\log n)$ point location | none | **infeasible beyond $d\approx6$** (the thesis `linear` baseline) |
| Moving least squares | none | $O(nd)$ | $O(k_nP^2+P^3)$ per point | none | |
| MARS, $M$ basis functions | $O(ndM^3)$ naive, $O(ndM^2)$ with fast updates | $O(nM)$ | $O(M)$ | none | |
| CART | $O(dn\log n)$ presorted | $O(n)$ nodes | $O(D)$ | none | |
| Random forest | $O(B\,d'\,n\log n)$ | $O(Bn)$ | $O(BD)$ | IJ variance $O(Bn)$; QRF $O(BD+n)$ | parallel over trees |
| Gradient boosting (histogram, $b$ bins) | $O(T(nd+2^Ddb))$ | $O(nd+T2^D)$ | $O(TD)$ | committee $\times K$; NGBoost $\approx2\times$ | XGBoost, LightGBM |
| BART, $m_t$ trees, $S$ sweeps | $\approx O(S\,m_t\,n)$ | $O(m_t n)$ | $O(S\,m_t D)$ | from posterior samples | |
| MLP / deep ensemble | $O(KEnW)$ | $O(KW)$ + batch | $O(KW)$ | spread over $K$; MC dropout $\times$ passes | |
| KAN, $G$ spline intervals per edge | $\approx O(EnWG)$ | $O(WG)$ | $O(WG)$ | none | |
| TabPFN ($\ell$ transformer layers) | no gradient training; $O(\ell n^2)$ attention over the training set, $O(\ell nM)$ test-to-train | $O(n^2)$ activations | amortised over the batch | native | practical limit $n\lesssim10^4$ |
| GMDH polynomial network | $O(\text{layers}\cdot\binom{d_\ell}{2}n)$ | $O(nd_\ell^2)$ | $O(\text{units})$ | none | |
| AAA rational (1-D), $m_s$ support points | $O(nm_s^3)$ | $O(nm_s)$ | $O(m_s)$ | none | multivariate variants cost more |
| Ensemble of surrogates | sum over members + CV $\times k_{cv}$ | sum | sum | member spread | |

### 11.2 Change-of-basis family (F10)
| Method | Fit time | Memory | Predict | Notes |
|---|---|---|---|---|
| Active subspace from $N_g$ gradients | $O(N_gd^2+d^3)$ + gradient cost | $O(d^2)$ | $O(dk)$ + learner in $k$ dims | GP-mean gradients cost $O(nd)$ each |
| Polynomial ridge (variable projection) | $O(I\,n\,(dk+P_k^2))$ per Grassmann iteration | $O(nP_k)$ | $O(dk+P_k)$ | $P_k$ polynomial terms in $k$ dims |
| Projection pursuit, $J$ ridges | $O(JIn\log n)$ | $O(n)$ | $O(Jd)$ | |
| Compressed sensing ($\ell_1$), basis size $N$ | OMP $O(snN)$; ADMM/IPM $O(InN)$ | $O(nN)$ | $O(s)$ | samples needed scale with $s\log^{c}N$, not $N$ |
| Functional SVD (2-D), $m\times m$ samples | full $O(m^3)$; randomised $O(m^2\log r+mr^2)$ | $O(mr)$ | $O(r)$ | |
| Tensor train, cross approximation | $O(dmr^3)$ work, $O(dmr^2)$ **function evaluations** | $O(dmr^2)$ | $O(dr^2)$ | linear in $d$ |
| Randomised SVD, $N_1\times N_2$ matrix | $O(N_1N_2\log k+(N_1+N_2)k^2)$ vs full $O(N_1N_2\min(N_1,N_2))$ | $O((N_1+N_2)k)$ | none | the compressed-DMD trick |
| Static SINDy autoencoder | $O(En(W+P))$ | $O(W+P)$ | $O(W+s)$ | |

### 11.3 Upscaling representations
| Representation | Build | Storage | Evaluate (value, gradient, Hessian) | Notes |
|---|---|---|---|---|
| Dense grid of the intermediate model | $m^d$ intermediate predictions (GP: $O(m^dn)$) | $O(m^d)$ | none | $m=24$: $d=3$ → $1.4\cdot10^4$ points; $d=6$ → $1.9\cdot10^8$ (1.5 GB at 8 B); $d=10$ → $6\cdot10^{13}$, impossible |
| Tensor B-spline interpolant | $O(dm^d)$ (separable banded 1-D solves) | $O(m^d)$ coefficients | $O(k^d)=4^d$ per point, derivatives same order | independent of $n$: the source of the thesis 19–32× speed-up |
| Tensor smoothing / P-spline, $K$ bases per axis | dense $O(nK^{2d}+K^{3d})$; array arithmetic (GLAM) much less | $O(K^{2d})$ | $O(k^d)$ | additive P-spline/GAM: $O(ndK)$ |
| Multilevel B-spline (MBA), $h$ levels | $O(h(n+m_h^d))$ | $O(m_h^d)$ | $O(k^d)$ | |
| THB-splines | $O(\#\text{active basis}\cdot n)$ | adaptive | $O(k^d)$ per level | refined only where needed |
| Smolyak sparse grid, level $\lambda$ | $O(2^\lambda\lambda^{d-1})$ intermediate evaluations | same | $O(2^\lambda\lambda^{d-1})$ | breaks $m^d$ under mixed smoothness |
| Chebyshev tensor | $O(m^d\log m)$ (DCT) | $O(m^d)$ | $O(dm^d)$ naive | |
| TT-compressed grid | TT-SVD $O(dmr^3)$ after the grid, or TT-cross without it | $O(dmr^2)$ | $O(dr^2)$ (+ spline factors) | the route past $d\approx6$ |

### 11.4 Loops and pipelines
- **Active-learning loop**, budget $N$, candidate pool $M_c$: each step refits and scores the pool. GP with full refit:
  $\sum_{j=n_0}^{N}O(j^3+M_cj^2)=O(N^4+M_cN^3)$; with rank-1 Cholesky updates at fixed θ: $O(j^2)$ per step, so
  $O(N^3+M_cN^3)$ in total; a $K$-tree committee: $O(K\,T\,d\,N^2)$; IMSE/ALC add $O(M_c\,n_{int}\,j^2)$ per step.
- **Multi-fidelity GP:** joint co-kriging $O((\sum_\ell n_\ell)^3)$; recursive (Le Gratiet) $O(\sum_\ell n_\ell^3)$, usually dominated
  by the cheapest level, which has the most points; NARGP adds Monte-Carlo propagation $O(S_{mc}n_\ell^2)$ per level;
  feature-augmented trees and networks cost one extra input.
- **Optimisers on a surrogate** (overhead per iteration, excluding surrogate calls): IPOPT/SQP $O(d^3)$ linear algebra;
  L-BFGS $O(dm_{\text{hist}})$; Nelder–Mead $O(d)$; CMA-ES $O(\lambda d^2)$ per generation plus an $O(d^3)$ eigendecomposition
  every $\sim d$ generations; DE/PSO $O(\lambda d)$; DIRECT memory grows with the number of rectangles. Total = iterations ×
  (overhead + surrogate predict cost), so an $O(k^d)$ spline versus an $O(nd)$ GP changes wall time, not the answer.
- **End-to-end (M.18):** $C_{tot}=\sum_\ell n_\ell c_\ell+C_{fit}+C_{AL}+C_{upscale}+N_{opt}c_{pred}$. For FEA ($c_H\approx1$ CPU-h)
  every surrogate term is negligible up to $n\sim10^4$; for cheap simulators the $O(n^3)$, $O(N^4)$ and $m^d$ terms decide
  the winner. This is the H7 cost crossover.

**Memory walls in this study:** exact GP and RBF $O(n^2)$ (0.8 GB at $n=10^4$, 80 GB at $10^5$); TabPFN attention $O(n^2)$;
full tensor grids $O(m^d)$; Delaunay $O(n^{\lceil d/2\rceil})$. They set hard caps in the E3/E7/E8 factor grids. Cells
beyond a wall are recorded as "infeasible", never silently dropped.
