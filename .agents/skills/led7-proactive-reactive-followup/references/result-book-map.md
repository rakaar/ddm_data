# LED7 Result-Book Map

Use this as a navigation aid. Captions in the dated Markdown pages remain the final figure-level record.

## 2026-08-13

File: `docs/results/2026-08-13.md`

Contains the earlier truncated step-jump SVI diagnostics and the first no-truncation+lapse animal-wise convergence and RT-relative-to-LED figures. Treat truncated and no-truncation entries as separate model branches.

## 2026-08-14

File: `docs/results/2026-08-14.md`

Main current abort diagnostic:

- `docs/assets/results/2026-08-14/led7_no_trunc_exp_lapse_rtwrtled_six_fits_vs_aggregate_2x8.png`

This compares six animal-wise fits, their pooled/trial-weighted visualization, and the independently fitted aggregate model. It is the clearest visual distinction between mixing animal fits and using one aggregate parameter vector.

## 2026-08-25

File: `docs/results/2026-08-25.md`

Main proactive parameter comparison:

- `docs/assets/results/2026-08-25/led7_old_vbmc_vs_no_trunc_lapse_svi_proactive_params.png`

This compares paper-era truncated VBMC parameters with current no-truncation+lapse step-jump SVI counterparts. The likelihoods differ, so describe it as a corresponding-parameter comparison, not a refit under identical assumptions.

The page also contains the animal-wise, equal-animal, and aggregate lapse summary table. Keep “equal-animal average” distinct from the pooled aggregate posterior.

## 2026-08-29

File: `docs/results/2026-08-29.md`

Likelihood validation and valid-trial figures include:

- `docs/assets/results/2026-08-29/led7_super_animal_valid_npl_alpha_convergence.png`
- `docs/assets/results/2026-08-29/led7_super_animal_valid_npl_alpha_convergence_displaced_only.png`
- `docs/assets/results/2026-08-29/led7_super_animal_off_valid_npl_alpha_rtds.png`
- `docs/assets/results/2026-08-29/led7_super_animal_on_bilateral_valid_npl_alpha_rtds.png`
- `docs/assets/results/2026-08-29/led7_super_animal_off_vs_on_npl_alpha_parameters.png`
- `docs/assets/results/2026-08-29/led7_98_valid_npl_alpha_pilot_parameter_comparison.png`

The paper decision is to use displaced branches for both OFF and bilateral ON at the next regeneration. Existing main RTD and parameter figures may still use the original automatic warm-ON selection, so check their captions and regenerate them before treating them as canonical. Preserve the original automatic warm-ON selection as audit provenance.

## Publishing A New Figure

Use the `update-result-book` skill. Every entry should state:

- model and likelihood branch;
- animals and pooling rule;
- LED scope, especially all-ON versus bilateral-ON;
- RT and abort filters;
- exact fit root and source script;
- posterior summary used for theory;
- bin width, theory resolution, x-range, and normalization/denominator;
- whether uncertainty is posterior uncertainty, across-animal SEM, or absent.

Do not overwrite a historical figure when adding a substantively different fit. Replace an existing asset only when the user explicitly asks to correct or restyle that same result.

After publication, run `.venv/bin/python -m mkdocs build --strict`. If the user asks for Drive backup, use the fit-backup skill and update the Drive ledger; result-book publication alone is not a fit backup.
