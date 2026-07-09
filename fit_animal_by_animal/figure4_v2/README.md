# Figure 4 v2 NPL+alpha Gamma+Omega MSE Reproduction

This folder contains the compact files needed to redraw the current Fig. 4-style
NPL+alpha diagnostic using animal-wise parameters fit by MSE to condition
Gamma/Omega means.

## What This Figure Shows

- Model family: NPL+alpha.
- NPL+alpha parameter source: per-animal MSE fit to Gamma and Omega from the
  patience12 92-parameter big Gamma/Omega/delay SVI fit.
- Source method key: `mse_gamma_omega`, matching the first row of
  `fit_animal_by_animal/compare_three_npl_param_sources_plus_ipl_patience12_4x5.py`.
- `w`, `del_go`, and condition-wise `t_E_aff` come from the same 92-parameter
  big SVI fit.
- Psychometric and psychometric-slope model points omit SD model entries for
  `|ILD| > 8`.
- RT quantiles use q10/q30/q50/q70/q90 and the SD flat-delay rule beyond
  `|ILD| = 8`.
- Gamma and Omega condition points come from the 92-parameter condition
  posterior means; solid model curves come from the NPL+alpha functional
  predictions using the Gamma+Omega MSE parameters.
- The combined corner version adds an upper-triangular parameter matrix using
  the individual animal-wise MSE point estimates for `lambda_prime`, `T_0`,
  `theta_E`, `rate_norm_l`, and `alpha`; it has no posterior ellipses or error
  bars because the MSE fit is deterministic. Diagonal panels include light
  dashed vertical guides at the parameter medians.

## Files

- `build_mse_gamma_omega_fig4_v2_data.py`: rebuilds the compact panel pickles
  from the saved 3x5 diagnostic payload.
- `plot_mse_gamma_omega_fig4_v2.py`: redraws the paper Fig. 4-style figure from
  the panel pickles in this folder, adding an Omega panel beside Gamma and using
  bottom-row order Gamma, Omega, psychometric-slope scatter.
- `plot_mse_gamma_omega_fig4_v2_with_upper_corner.py`: redraws the same
  Fig. 4-style panels and adds an upper-triangular MSE point-estimate parameter
  matrix in the style of `figure_4_with_corner_using_template.py`.
- `mse_gamma_omega_npl_alpha_*_fig4_v2_data.pkl`: compact panel-data pickles.
- `mse_gamma_omega_npl_alpha_fig4_v2_bundle.pkl`: bundled copy of the compact
  panel data and validation checks.
- `figure4_v2_mse_gamma_omega_npl_alpha.png` / `.pdf`: current rendered figure.
- `figure4_v2_mse_gamma_omega_npl_alpha_with_upper_corner.png` / `.pdf`: current
  rendered combined figure with the upper-triangular MSE parameter matrix.

## Reproduce The Figure From This Folder

From the repository root:

```bash
.venv/bin/python -u fit_animal_by_animal/figure4_v2/plot_mse_gamma_omega_fig4_v2.py
```

This is the normal reproducible path because it uses only the compact pickles
stored here.

To reproduce the combined Figure 4 + MSE corner version:

```bash
.venv/bin/python -u fit_animal_by_animal/figure4_v2/plot_mse_gamma_omega_fig4_v2_with_upper_corner.py
```

## Rebuild The Panel Pickles

Only needed if the upstream 3x5 diagnostic payload is available locally and the
compact panel pickles should be regenerated:

```bash
.venv/bin/python -u fit_animal_by_animal/figure4_v2/build_mse_gamma_omega_fig4_v2_data.py
.venv/bin/python -u fit_animal_by_animal/figure4_v2/plot_mse_gamma_omega_fig4_v2.py
```

The builder validates:

- 30 animals.
- 864 condition rows from the patience12 92-parameter big SVI fit.
- 30 Gamma+Omega MSE parameter rows.
- the MSE CSV path is the `objective_variants/gamma_omega` file used by the
  first row of the 3x5/4x5 diagnostics.
- zero SD psychometric/slope model entries at `|ILD| > 8`.
- 30 SD-flat model entries at `|ILD| = 16` for each ABL in the continuous
  quantile curves.
- paper quantiles exactly `[0.1, 0.3, 0.5, 0.7, 0.9]`.
