# Methods lookup notes

Concise reference for writing the Methods section; not final paper prose. Checked
against the current fitting code and figure READMEs on 2026-08-31.

## Figure and fitting map

- **Direct-data figure (Figure 1):** descriptive data analyses; no model-fitting
  method to add here yet.
- **IPL figure (Figure 2):** direct IPL/vanilla TIED fit to RT and choice data
  with stochastic variational inference (SVI).
- **NPL+alpha figure (Figure 4):** condition-wise Gamma/Omega SVI followed by a
  per-animal NPL+alpha fit using the Gamma/Omega MSE objective.
- **Lapse supplement:** direct IPL+lapse and NPL+alpha+lapse SVI fits, compared
  with their no-lapse versions.

## 1. Aborts fitting

### Fit and data selection

- **Algorithm:** per-animal Variational Bayesian Monte Carlo (VBMC).
- **Truncation:** exclude trials below \(T_{\mathrm{trunc}}=0.300\,\mathrm{s}\)
  (300 ms); use \(T_{\mathrm{trunc}}=0.150\,\mathrm{s}\) (150 ms) for
  `LED34_even`.
- **Fitted parameters:** proactive drift \(V_A\), bound \(\theta_A\), and
  afferent delay \(t_{A,\mathrm{aff}}\). The code name `t_A_aff` corresponds to
  the delay that may be written as \(\delta_A\).
- **Prior:** independent trapezoidal priors over the implemented hard and
  plausible parameter ranges.

### Proactive first-passage distribution

For \(\tau>0\), the proactive first-hitting-time density is

\[
\rho_A(\tau)=
\frac{\theta_A}{\sqrt{2\pi\tau^3}}
\exp\!\left[-\frac{V_A^2(\tau-\theta_A/V_A)^2}{2\tau}\right].
\]

Its cumulative distribution is

\[
C_A(\tau)=
\Phi\!\left(\frac{V_A(\tau-\theta_A/V_A)}{\sqrt{\tau}}\right)
+e^{2V_A\theta_A}
\Phi\!\left(-\frac{V_A(\tau+\theta_A/V_A)}{\sqrt{\tau}}\right),
\qquad S_A(\tau)=1-C_A(\tau),
\]

where \(\Phi\) is the standard-normal CDF.

### Abort/censoring likelihood

For retained trials \(t\geq T_{\mathrm{trunc}}\), with total fixation time
\(t\) and stimulus onset \(t_{\mathrm{stim}}\),

\[
\mathcal L_A(t,t_{\mathrm{stim}})=
\begin{cases}
\dfrac{\rho_A(t-t_{A,\mathrm{aff}})}
{S_A(T_{\mathrm{trunc}}-t_{A,\mathrm{aff}})},
& t<t_{\mathrm{stim}} \quad\text{(abort)},\\[8pt]
\dfrac{S_A(t_{\mathrm{stim}}-t_{A,\mathrm{aff}})}
{S_A(T_{\mathrm{trunc}}-t_{A,\mathrm{aff}})},
& t>t_{\mathrm{stim}},\ t_{\mathrm{stim}}>T_{\mathrm{trunc}}
\quad\text{(valid/right-censored)},\\[8pt]
1,
& t>t_{\mathrm{stim}},\ t_{\mathrm{stim}}\leq T_{\mathrm{trunc}}.
\end{cases}
\]

- Abort trials contribute a hitting-time density.
- Valid trials contribute survival of the proactive process until stimulus
  onset, i.e. right censoring rather than a proactive hitting time.
- Posterior means of \(V_A,\theta_A,t_{A,\mathrm{aff}}\) are held fixed in the
  later valid-trial fits.

### Proactive-evidence race likelihood used later

For response bound \(b\), define

\[
\tau_A=t-t_{A,\mathrm{aff}},
\qquad
\tau_E=t-t_{\mathrm{stim}}-t_{E,\mathrm{aff}},
\qquad
C_E=C_E^R+C_E^L.
\]

The bound-specific race density is

\[
\begin{aligned}
p_b(t)={}&\rho_A(\tau_A)
\left\{
\frac{1}{2}\left[1-C_E(\tau_E+\Delta_{\mathrm{go}})\right]
+C_E^b(\tau_E+\Delta_{\mathrm{go}})-C_E^b(\tau_E)
\right\}\\
&+\rho_E^b(\tau_E)\left[1-C_A(\tau_A)\right].
\end{aligned}
\]

- First term: the proactive process hits first; choice is read randomly if
  evidence survives, or from the evidence bound reached during
  \(\Delta_{\mathrm{go}}\).
- Second term: the evidence process reaches bound \(b\) while the proactive
  process survives.

## 2. IPL fitting

### Fit

- **Paper-facing name:** IPL; current code also calls this model vanilla TIED.
- **Algorithm:** NumPyro SVI, ELBO objective, full-rank Gaussian variational
  posterior.
- **Data:** valid left/right choice trials in the retained \(0\) to \(1\) s
  post-stimulus RT window.
- **Fitted global parameters:**
  \(\lambda,T_0,\theta_E,w,\Delta_{\mathrm{go}}\).
- **Fitted condition parameters:** one \(t_{E,\mathrm{aff}}\) for each observed
  ABL-by-signed-ILD condition.
- **Fixed parameters:** \(V_A,\theta_A,t_{A,\mathrm{aff}}\), set to the
  per-animal abort-fit posterior means.
- Evidence starting-point bias:
  \(Z_E=(w-0.5)\,2\theta_E\).

### IPL expressions

With \(\chi=17.37\),

\[
\Gamma=
\theta_E\tanh\!\left(\frac{\lambda\,\mathrm{ILD}}{\chi}\right),
\]

\[
\Omega=
\frac{10^{\lambda\,\mathrm{ABL}/20}
\cosh\!\left(\lambda\,\mathrm{ILD}/\chi\right)}
{T_0\theta_E^2}.
\]

- \(\Gamma\): signed evidence/discriminability term.
- \(\Omega\): condition-dependent time-scale term.
- The bound-specific race density \(p_b(t)\) is conditioned on the retained RT
  window:

\[
\widetilde p_b(t)=
\frac{p_b(t)}
{C_{A\cup E}(t_{\mathrm{stim}}+1)-C_{A\cup E}(t_{\mathrm{stim}})},
\qquad
C_{A\cup E}=C_A+C_E-C_AC_E,
\]

  using the truncation-conditioned proactive CDF.

## 3. Direct NPL+alpha fitting

### Fit

- **Algorithm/data/likelihood:** direct RT-and-choice SVI using the same valid
  trials, proactive-evidence race, and RT-window conditioning as IPL.
- **Fitted parameters:**
  \(\lambda,T_0,\theta_E,w,\Delta_{\mathrm{go}},l,\alpha\), plus condition-wise
  \(t_{E,\mathrm{aff}}\).
- **Fixed parameters:** \(V_A,\theta_A,t_{A,\mathrm{aff}}\) from the abort fit.
- \(l\) is `rate_norm_l`; \(\alpha\) controls the asymmetric contribution in
  the divisive normalization denominator.

### NPL+alpha firing-rate form

Following the power-law normalization notation of Eq. 3.2.14 in
`Deconstruction_Reaction_Time.pdf`, define

\[
f(P_\pm)=\left(\frac{P_\pm}{P_0}\right)^\lambda,
\]

where \(P_R\) and \(P_L\) are the sound pressures at the right and left ears,
and \(P_0\) is the reference pressure. The alpha-extended normalized firing
rates are

\[
r_R=r_0\frac{f(P_R)}
{[f(P_R)]^l+\alpha[f(P_L)]^l},
\qquad
r_L=r_0\frac{f(P_L)}
{[f(P_L)]^l+\alpha[f(P_R)]^l}.
\]

- When \(\alpha=1\), this reduces to the normalization form in Eq. 3.2.14.
- The common firing-rate scale \(r_0\) is absorbed into \(T_0\) in the current
  implementation.

These give

\[
\Gamma=\theta_E\frac{r_R-r_L}{r_R+r_L},
\qquad
\Omega=\frac{r_R+r_L}{T_0\theta_E^2}.
\]

- \(\alpha\) was introduced to improve the tuning-curve match and test whether
  this can correct the Omega mismatch.
- **Important:** this is a direct RT-and-choice SVI route. Its NPL+alpha
  parameters are not the parameter source used in the current main Figure 4;
  that figure uses the MSE route in Section 5.

## 4. Lapse fitting

### Models and parameters

- Direct SVI fits were run for both **IPL+lapse** and
  **NPL+alpha+lapse**, alongside their no-lapse versions.
- Core model parameters and lapse parameters are fitted jointly.
- Added global parameters:
  - \(\pi_L\) (`lapse_prob`): probability of a lapse.
  - \(q_R\) (`lapse_prob_right`): probability of a right choice during a
    lapse.

### Lapse mixture

For bound/choice \(b\),

\[
p_{\mathrm{mix}}(t,b)=
(1-\pi_L)p_{\mathrm{model}}(t,b)
+\pi_L
\frac{q_R\mathbf{1}_{b=R}+(1-q_R)\mathbf{1}_{b=L}}
{1\,\mathrm{s}}.
\]

- \(p_{\mathrm{model}}\) is the normalized IPL or NPL+alpha race density.
- Lapse RT is uniform over the retained 1 s window.
- Lapse choices need not be unbiased; \(q_R\) allows a right/left lapse bias.
- The lapse supplement compares IPL, NPL+alpha, IPL+lapse, and
  NPL+alpha+lapse using a common posterior-mean RT-and-choice log likelihood on
  the same valid \(\mathrm{RT}<1\) s fitting trials.
- The Gamma panel additionally uses condition-wise Gamma/Omega SVI fits with
  and without lapses.

## 5. MSE fitting

### Stage 1: condition-wise Gamma/Omega SVI

- Direct RT-and-choice SVI fit for each animal.
- **Condition-wise fitted parameters:**
  \(\Gamma_c,\Omega_c,t_{E,\mathrm{aff},c}\).
- **Global fitted parameters:** \(w,\Delta_{\mathrm{go}}\).
- **Fixed parameters:** \(V_A,\theta_A,t_{A,\mathrm{aff}}\) from the abort fit.
- Retain the posterior mean \(\widehat\Gamma_c,\widehat\Omega_c\) for every
  observed condition \(c\).

### Stage 2: NPL+alpha MSE mapping

- **Algorithm:** bounded nonlinear least squares, run separately for each
  animal.
- **Fitted point-estimate parameters:**

\[
\phi=(\lambda,T_0,\theta_E,l,\alpha).
\]

- Fit the NPL+alpha predictions to the Stage-1 posterior means using the
  standardized objective

\[
\min_{\phi}\sum_c
\left[
\frac{\Gamma_\phi(c)-\widehat\Gamma_c}{s_\Gamma}
\right]^2
+
\left[
\frac{\Omega_\phi(c)-\widehat\Omega_c}{s_\Omega}
\right]^2,
\]

  where \(s_\Gamma\) and \(s_\Omega\) are the across-condition standard
  deviations of the Stage-1 Gamma and Omega posterior means.

- This is called the Gamma+Omega **MSE fit** in the figure pipeline; the
  implementation is standardized nonlinear least squares.
- It fits Gamma/Omega summaries, not the individual RT-and-choice trials.
- It returns deterministic per-animal point estimates, not a Bayesian
  posterior for \(\phi\).
- Current Figure 4 uses these MSE-fitted NPL+alpha parameters together with
  \(w,\Delta_{\mathrm{go}}\), and condition-wise \(t_{E,\mathrm{aff}}\) from
  the Stage-1 SVI fit.
