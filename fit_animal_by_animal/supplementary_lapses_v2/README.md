# Supplementary Lapses Figure v2

This folder regenerates the old 2 x 4 lapses supplementary figure using the
current SVI fit families.

## Fit sources

- `IPL`: `../numpyro_svi_vanilla_condition_delay_patience12_min50k_restore_best_outputs`
- `NPL`: `../numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs`
- `IPL_L`: `../numpyro_svi_vanilla_lapse_condition_delay_patience12_min50k_restore_best_outputs`
- `NPL_L`: `../numpyro_svi_npl_alpha_lapse_condition_delay_patience12_min50k_restore_best_outputs`
- Gamma panel, no lapse: `../../fit_each_condn/svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs`
- Gamma panel, lapse: `../../fit_each_condn/svi_big_gamma_omega_delay_lapse_patience12_restore_best_all_animals_outputs`

In figure text, `NPL` and `NPL_L` are compact paper-facing labels. They refer to
the current NPL+alpha and NPL+alpha+lapse SVI fits.

## Scripts

- `build_svi_lapses_supp_v2_data.py` validates the roots, computes common
  posterior-mean RT+choice log likelihoods on the valid RT<1 SVI fitting trials,
  extracts posterior summaries, and saves compact CSV/PKL products.
- `plot_svi_lapses_supp_v2.py` loads the compact data product and assembles the
  paper-style 2 x 4 PNG/PDF with the old `figure_template.py` layout.

Run from the repository root:

```bash
.venv/bin/python fit_animal_by_animal/supplementary_lapses_v2/build_svi_lapses_supp_v2_data.py
.venv/bin/python fit_animal_by_animal/supplementary_lapses_v2/plot_svi_lapses_supp_v2.py
```

Outputs are written under `outputs/`.
