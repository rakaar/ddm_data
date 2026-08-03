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
