# DDM Data Repository

## Root Level Scripts

- `check_led8_session_training_repeat_rows.py`: Quick LED8 CSV audit for session/training/repeat filters. Counts filtered rows by repeat value, animal, and LED ON/OFF status from `outMatrix_LED8_converted.csv`.
- `convert_outMatrix_LED8_mat_to_csv.py`: Converts numeric fields from `outMatrix_LED8.mat` into a long-format CSV (`outMatrix_LED8_converted.csv`) and reports skipped nonnumeric/object fields.
- `schematics_for_fct.py`: Generates schematic plots for the proactive LED model including RT wrt LED theory/data comparisons, drift switch visualizations, corner plots, and RTD wrt fixation plots. Includes `save_plot_payload()` calls for data persistence and comprehensive plot metadata storage.
- `fct_march_5_panel_v2.py`: Panel plotting script for FCT analysis created on March 5th, version 2.
- `fct_panel_march_26.py`: Panel plotting script for FCT analysis created on March 26th.
- `juan_ddm_math.py`: JAX/NumPyro analytical two-bound DDM likelihood helpers for the simple 2AFC teaching fit.
- `juan_fit_teaching_2afc_ddm.py`: Small teaching SVI script for fitting a standard two-bound DDM to 2AFC workshop data, using `juan_ddm_math.py`.

## Root Level Docs

- `CODEX_GIT_WRAPUP.md`: Reusable shorthand request for Codex. Referring to this file means: check `git status`, document new or undocumented files in the relevant `README.md` files, commit the intended changes, and push to `origin/main`.
- `FIT_BACKUP_LEDGER.md`: Running ledger for large fit result locations and Google Drive backups, including which animals ran on lavos/ganon, local and remote result paths, Drive folders, counts, and backup status.
- `led8_led_off_rtds_task.md`: Task note specifying the LED8 session-type-8 LED-off RTD plotting requirements and suggested script structure for `fitting_aborts/led8_session8_led_off_rtds.py`.
- `paper_notes/model_taxonomy_and_paper_notes.md`: Repo-owned notes on IPL/vanilla TIED, NPL, NPL+alpha, lapse variants, Gamma/Omega fits, and comparison rules. The repo-local Codex skill in `.agents/skills/ddm-model-comparison/` points future agents to this note before model-comparison work.

## Utility Scripts

- `scripts/backup_ganon_fit_results_to_drive.sh`: Idempotent `rclone copy` wrapper for backing up the copied ganon ABL-specific NPL+alpha+ILD2 upstream results and condition-by-condition outputs to Google Drive under `raga:ddm_fit_backups_20260604_ganon/`.

## `fit_animal_by_animal/` Scripts

## `fitting_aborts/` Scripts

- `compare_vbmc_model_fits.py`: Compares VBMC model fits between drift jump and bound drop models. Loads pickle files from both models, extracts ELBO and log-likelihood values, and displays side-by-side comparison with clear print statements. Uses cell structure (# %%) for easy execution.
- `plot_vbmc_param_summary_per_animal_NO_TRUNC_with_lapse.py`: Plots VBMC parameter summaries across animals for the no-truncation model with an exponential lapse component.
- `sim_proactive_LED_bound_drop_fixed_params.py`: Simulates the proactive LED model with a dropping bound (both constant step and dynamic linear drop) and fixed parameters.
- `pyddm_play.py`: Experimental script for testing PyDDM library functionality and comparing single-bound vs two-bound models.
- `pyddm_play_bound_drop.py`: PyDDM script testing time-dependent bound drop (linear decrease from theta_A0 to theta_A_min).
- `pyddm_play_drift_up_bound_drop.py`: PyDDM script testing combined drift increase and bound drop dynamics over time.
