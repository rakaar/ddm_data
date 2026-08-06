# NPL+alpha uniform-delay feasibility

`validate_integrated_reactive_cdf.py` tests whether a uniform distribution for
condition-wise `t_E_aff` can be marginalized analytically without changing the
rest of the NPL+alpha likelihood.

For a delay `D ~ Uniform(d_low, d_high)`, the bound-specific reactive terms are

```text
convolved CDF = [H(t - d_low) - H(t - d_high)] / (d_high - d_low)
convolved PDF = [F(t - d_low) - F(t - d_high)] / (d_high - d_low)
```

where `H(t) = integral_0^t F(u) du`. The script compares a finite spectral
series for `H` at `K = 5, 10, 20` against numerical integration of the current
small-time image-series CDF. It also checks `dH/dt = F`, direct numerical
delay convolution, the zero-width limit, and parameter cases drawn from all 30
patience-12 NPL+alpha condition-delay SVI fits.

Run from the repository root:

```bash
.venv/bin/python fit_animal_by_animal/npl_alpha_uniform_delay/validate_integrated_reactive_cdf.py
```

Outputs are written to `validation_outputs/`. This is a feasibility experiment;
it does not alter the production JAX likelihood.

`integrated_cdf_representative_curve_comparison.png` gives a direct visual
comparison for three posterior-derived parameter sets. Its top row shows the
numerically integrated and finite-series analytic `H(t)` curves. Its bottom row
shows the corresponding CDF after convolution with a 5 ms uniform delay.

The current validation result is that none of `K = 5, 10, 20` is accurate
enough for likelihood use. At `K = 20`, the worst integrated-CDF error is about
`5.4e-4`, and differencing across a 5 ms uniform delay can amplify the worst CDF
error to about `0.11` near delay onset. A later implementation should therefore
use a small-time analytic representation, a hybrid series, or controlled
quadrature rather than this finite spectral antiderivative alone.

`validate_uniform_delay_simulator_vs_likelihood.py` tests the resulting `K=100`
uniform-delay likelihood against direct simulation and independent 64-node
Gauss-Legendre delay mixtures. It produces both a full-support comparison and a
data-like retained-window comparison. The latter first removes absolute RTs
below the batch abort cutoff (`300 ms` normally, `150 ms` for `LED34_even`),
then retains `0 <= RTwrtStim < 1 s` and renormalizes the two choice-joint
densities with one shared window mass. The proactive PDF/CDF is conditioned on
survival past the left cutoff; the evidence process is not truncated.

```bash
.venv/bin/python fit_animal_by_animal/npl_alpha_uniform_delay/validate_uniform_delay_simulator_vs_likelihood.py
```

The retained-window validation reuses the 100,000 full-support simulations per
cell and stores deterministic supplemental caches until at least 100,000 trials
remain in every conditioned cell. For the three current representative cases,
the fixed stimulus onset is `500 ms`, later than the `300 ms` abort cutoff, so
the subsequent nonnegative stimulus-relative window is already a subset of the
left-truncated responses. The script asserts this explicitly while retaining
the correct ordering and likelihood conditioning.

`plot_npl_alpha_uniform_delay_led7_all_animal_diagnostics.py` summarizes the six
completed LED7 NPL+alpha RT+choice fits with condition-wise Uniform evidence
delays. It creates a 1 x 6 convergence grid, a 1 x 6 animal-wise delay-support
grid, and an across-animal delay plot. In the across-animal plot, broad
translucent bars show the equally averaged fitted Uniform support and narrow
capped bars show the SEM of animal posterior-mean delay centers.

## SVI fitting pipeline

`uniform_delay_likelihood_utils.py` contains the JAX implementation used by the
fit. It evaluates the proactive/reactive race likelihood after analytically
marginalizing the evidence density and CDF over a Uniform delay distribution.
The fixed-delay implementation and Gauss-Legendre comparisons remain available
for validation.

`uniform_delay_svi_utils.py` defines the NumPyro model and constrained parameter
transforms. The fitted parameter vector contains seven global parameters
(`rate_lambda`, `T_0`, `theta_E`, `w`, `del_go`, `rate_norm_l`, and `alpha`), 30
condition-wise delay centers, and 30 condition-wise delay widths: 67 parameters
for an animal with the full LED7/LED8 stimulus grid. Delay support is constrained
to `[0, 1] s`, with widths between 1 and 100 ms.

`validate_uniform_delay_svi_led7_92.py` checks all 30 LED7/92 conditions and both
response bounds on a 1 ms grid. It compares the analytic delay-marginalized
terms at 200 integrated-CDF terms with an independent 64-node Gauss-Legendre
mixture.

`numpyro_svi_npl_alpha_uniform_delay_single_animal.py` fits one animal to the
successful RT+choice rows with `0 <= RTwrtStim < 1 s`. It initializes the global
parameters, delay centers, and covariance from the corresponding point-delay
NPL+alpha SVI fit, then adds one delay-width latent per condition. The default
run uses a full-rank guide and patience-12 restore-best stopping after at least
50,000 iterations. It saves posterior samples, the variational state, condition
tables, windowed negative-ELBO history, run metadata, and convergence plots.

For example, run LED8/105 from the repository root with:

```bash
NUMPYRO_SVI_BATCH=LED8 NUMPYRO_SVI_ANIMAL=105 \
  .venv/bin/python -u \
  fit_animal_by_animal/npl_alpha_uniform_delay/numpyro_svi_npl_alpha_uniform_delay_single_animal.py
```

`run_numpyro_svi_npl_alpha_uniform_delay_led7.py` is the resumable sequential
runner for the LED7 animals. It verifies the reference fits and raw condition
grids, runs each missing fit, creates per-animal diagnostics, and maintains a
CSV/JSON status ledger under the selected output root.

## Fit diagnostics

`plot_npl_alpha_uniform_delay_single_animal_rtds.py` recreates the exact retained
fit rows, evaluates the model at 1 ms resolution, and plots signed-choice RTDs
after equal averaging across signs. The empirical histogram width is controlled
by `NUMPYRO_SVI_DIAG_DATA_BIN_S`; the main LED7 and LED8/105 diagnostics use
10 ms bins and display 0--600 ms without renormalizing to that display range.

`plot_npl_alpha_uniform_delay_single_animal_parameter_diagnostics.py` creates the
seven-global-parameter corner plot, the 2 x 5 condition-delay density grid, and
the delay-center/support-versus-ILD figure for one animal.

`plot_npl_alpha_uniform_delay_led7_all_animal_diagnostics.py` reads the six LED7
fit folders and creates the across-animal loss and delay summaries in addition
to regenerating the individual-animal outputs.
