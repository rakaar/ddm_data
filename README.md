# DDM Data Repository

## Root Level Scripts

- `check_led8_session_training_repeat_rows.py`: Quick LED8 CSV audit for session/training/repeat filters. Counts filtered rows by repeat value, animal, and LED ON/OFF status from `outMatrix_LED8_converted.csv`.
- `convert_outMatrix_LED8_mat_to_csv.py`: Converts numeric fields from `outMatrix_LED8.mat` into a long-format CSV (`outMatrix_LED8_converted.csv`) and reports skipped nonnumeric/object fields.
- `schematics_for_fct.py`: Generates schematic plots for the proactive LED model including RT wrt LED theory/data comparisons, drift switch visualizations, corner plots, and RTD wrt fixation plots. Includes `save_plot_payload()` calls for data persistence and comprehensive plot metadata storage.
- `fct_march_5_panel_v2.py`: Panel plotting script for FCT analysis created on March 5th, version 2.
- `fct_panel_march_26.py`: Panel plotting script for FCT analysis created on March 26th.

## Root Level Docs

- `CODEX_GIT_WRAPUP.md`: Reusable shorthand request for Codex. Referring to this file means: check `git status`, document new or undocumented files in the relevant `README.md` files, commit the intended changes, and push to `origin/main`.
- `led8_led_off_rtds_task.md`: Task note specifying the LED8 session-type-8 LED-off RTD plotting requirements and suggested script structure for `fitting_aborts/led8_session8_led_off_rtds.py`.

## `fit_animal_by_animal/` Scripts

## `fitting_aborts/` Scripts

- `compare_vbmc_model_fits.py`: Compares VBMC model fits between drift jump and bound drop models. Loads pickle files from both models, extracts ELBO and log-likelihood values, and displays side-by-side comparison with clear print statements. Uses cell structure (# %%) for easy execution.
- `plot_vbmc_param_summary_per_animal_NO_TRUNC_with_lapse.py`: Plots VBMC parameter summaries across animals for the no-truncation model with an exponential lapse component.
- `sim_proactive_LED_bound_drop_fixed_params.py`: Simulates the proactive LED model with a dropping bound (both constant step and dynamic linear drop) and fixed parameters.
- `pyddm_play.py`: Experimental script for testing PyDDM library functionality and comparing single-bound vs two-bound models.
- `pyddm_play_bound_drop.py`: PyDDM script testing time-dependent bound drop (linear decrease from theta_A0 to theta_A_min).
- `pyddm_play_drift_up_bound_drop.py`: PyDDM script testing combined drift increase and bound drop dynamics over time.
