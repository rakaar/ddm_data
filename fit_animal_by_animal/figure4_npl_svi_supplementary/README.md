# NPL SVI Supplementary Figure 4

This folder redraws the final Figure 4 v2 layout using the direct patience-12
NPL+alpha condition-delay SVI fit rather than the animal-wise Gamma/Omega MSE
parameter mapping.

## Inputs

- Direct NPL+alpha fit root:
  `../numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs/`
- Existing paper-panel payload:
  `three_npl_param_source_comparison/three_npl_param_sources_patience12_3x5.pkl`
- Gamma/Omega scatter targets: condition posterior means from the patience-12
  92-parameter Gamma/Omega/delay SVI fit.
- Posterior corner inputs: each animal's
  `main_fullrank_posterior_samples.npz`, with 10,000 samples for
  `rate_lambda`, `T_0`, `theta_E`, `rate_norm_l`, and `alpha`.

The psychometric and slope panels omit nonexistent SD model points above
`|ILD|=8`. The RT-quantile panel uses q10/q30/q50/q70/q90 and holds SD
`t_E_aff` flat beyond `|ILD|=8` for the continuous model curves.

## Corner Plot

The corner block keeps the upper-triangular layout of Figure 4 v2. Each blue
dot is one animal's posterior mean. Unfilled blue outlines are 95% covariance
ellipses calculated from that animal's posterior samples. Diagonal panels rank
animals by posterior mean and show marginal 2.5--97.5% credible intervals.

## Reproduce

From the repository root:

```bash
.venv/bin/python -u fit_animal_by_animal/figure4_npl_svi_supplementary/build_npl_svi_fig4_supplementary_data.py
.venv/bin/python -u fit_animal_by_animal/figure4_npl_svi_supplementary/plot_npl_svi_fig4_supplementary.py
```

Outputs:

- `npl_svi_patience12_fig4_supplementary_bundle.pkl`
- `figure4_supplementary_npl_svi_patience12.png`
- `figure4_supplementary_npl_svi_patience12.pdf`

The builder checks 30 animals, 864 rows from each source fit, 10,000 finite
posterior samples per animal, the SD high-ILD mask, the SD flat-delay quantile
rule, and agreement between rebuilt posterior means and the existing direct-SVI
diagnostic payload.
