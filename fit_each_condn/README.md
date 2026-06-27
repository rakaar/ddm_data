# Fit Each Condition

This folder contains scripts for fitting condition-by-condition gamma and omega parameters for each animal.

## Condition-by-Condition Fitting

### Fitting Scripts
- `fit_single_rat_condn_by_condn_fix_t_E_w_del_go_all_animals_loop_for_paper.py` - Fit condition-by-condition gamma/omega for each animal with t_E_aff, w, del_go fixed
- `fit_single_rat_condn_by_condn_4_params_fix_w_mean_all_animals_loop.py` - Fit condition-by-condition Gamma/Omega/`t_E_aff`/`del_go` for all animals while fixing each animal's `w` to the mean from `five_param_w_mean_median_by_animal.csv`; saves pickles and per-condition corner plots in separate 4-param fixed-`w` output folders.
- `fit_single_rat_condn_by_condn_3_params_fix_w_del_go_from_abl_specific_ild2_all_animals_loop.py` - Fit condition-by-condition Gamma/Omega/`t_E_aff` for each animal after fixing `w` and `del_go` to posterior means from the completed ABL-specific NPL+alpha+ILD2-delay animal fit; skips animals whose upstream fit result is not ready and saves pickles/corner plots in dedicated 3-param output folders.
- `benchmark_svi_3param_gamma_omega_t_E_aff.py` - Benchmarks a NumPyro SVI version of the same 3-param condition fit (`gamma`, `omega`, `t_E_aff`) for a selected animal/condition set, fixing `w` and `del_go` from the current animal-wise NPL+alpha+condition-delay SVI posterior and comparing SVI posterior means/intervals against the existing 3-param VBMC condition-fit cache.
- `fit_svi_cond_by_cond_gamma_omega_fixed_from_animal_svi.py` - Fits condition-by-condition Gamma/Omega with NumPyro SVI while fixing `w`, `del_go`, and each condition's `t_E_aff` to posterior means from the current animal-wise NPL+alpha+condition-delay SVI fit; writes per-animal posterior samples and an aggregate Gamma/Omega cache.
- `svi_big_gamma_omega_delay_single_animal.py` - Prototype one-animal NumPyro SVI fit that jointly estimates condition-wise Gamma/Omega/`t_E_aff` plus global `w` and `del_go` using the direct Gamma/Omega likelihood; defaults to a smoke test on `LED8/105`.
- `run_svi_big_gamma_omega_delay_all_animals.py` - Tmux-friendly all-animal runner for `svi_big_gamma_omega_delay_single_animal.py`; discovers the 30 animals from the completed animal-wise NPL+alpha condition-delay SVI outputs and saves per-animal posterior-sample bundles, loss/convergence tables, and loss/condition-parameter PNGs in a separate all-animal output folder. Supports explicit output roots and stopping-rule controls such as patience restore-best with a minimum step count.
- `plot_svi_big_gamma_omega_delay_all_animals_condition_params.py` - Aggregates the completed big Gamma/Omega/delay SVI posterior means across animals and plots mean +/- SEM condition-wise Gamma/Omega/`t_E_aff` against ILD for ABL 20/40/60.
- `svi_gamma_omega_likelihood_utils.py` - Shared JAX likelihood helpers for direct Gamma/Omega SVI scripts.
- `run_big_gamma_omega_delay_convergence_audit.py` - Runs selected big Gamma/Omega/delay SVI animals with early stopping disabled in a separate audit output folder to test whether the original `stable_3_windows` stop criterion changed posterior means.
- `compare_big_gamma_omega_delay_convergence_audit.py` - Compares original early-stopped big SVI fits against no-early-stop audit reruns, reporting post-stop ELBO gain per trial and standardized posterior-mean shifts for Gamma/Omega/`t_E_aff`, `w`, and `del_go`.
- `diagnose_big_gamma_omega_delay_loss_minima_methods.py` - Tests loss-minimum selection rules on the six 50k no-early-stop audit curves, comparing raw, rolling-smoothed, and rebound-confirmed minima for future long-run stopping.
- `compare_old_stop_vs_patience8_restore_best_params.py` - Compares the original old-stop big SVI outputs against the patience8 restore-best checkpoint on the six audit animals, plotting loss curves and Gamma/Omega condition means.
- `plot_big_gamma_omega_delay_patience12_loss_grid.py` - Plots a 5x6 grid of all 30 patience12 restore-best big SVI loss curves, marking the restored-best checkpoint and the final checked step.
- `plot_svi_big_gamma_omega_delay_patience12_all_animals_condition_params.py` - Aggregates the completed patience12 restore-best big SVI posterior means across animals and plots mean +/- SEM condition-wise Gamma/Omega/`t_E_aff`.
- `plot_svi_big_gamma_omega_delay_patience12_vs_old_condition_params.py` - Overlays old-rule and patience12 restore-best all-animal condition means and plots signed paired deltas across animals.
- `consolidate_svi_gamma_omega_fixed_from_animal_svi_reruns.py` - Merges the full `all_observed` condition-by-condition SVI run with selected rerun folders, replacing only rerun conditions and writing a clean consolidated result folder with regenerated aggregate CSVs, per-animal NPZ files, and a replacement manifest.
- `compare_svi_cond_gamma_omega_with_npl_alpha_svi.py` - Compares consolidated condition-by-condition Gamma/Omega SVI fits against Gamma/Omega curves implied by the matching animal-wise NPL+alpha condition-delay SVI posterior means, and overlays per-animal MSE alpha-model fits to the condition means.
- `compare_big_svi_gamma_omega_with_mse_alpha_model.py` - Fits the Gamma/Omega alpha model by MSE to each animal's completed big Gamma/Omega/delay SVI condition means, then plots averaged condition Gamma/Omega against averaged per-animal MSE functional curves with SEM shading.
- `compare_npl_svi_vs_big_svi_mse_alpha_params.py` - Compares animal-wise NPL+alpha condition-delay SVI posterior parameters against per-animal MSE alpha-model parameters fit to the completed big SVI condition Gamma/Omega means.
- `compare_npl_svi_vs_mse_gamma_omega_alpha_params.py` - Plots animal-wise NPL+alpha SVI posterior means and 95% intervals against the per-animal MSE alpha-model parameters to diagnose why the MSE fit better matches the condition Gamma/Omega curves.
- `compare_npl_svi_vs_mse_params_rt_choice_loglike.py` - Evaluates the original NPL+alpha condition-delay RT+choice likelihood on the same SVI fitting trials for each animal, comparing NPL SVI posterior-mean parameters against the per-animal MSE Gamma/Omega parameter substitution and plotting total, per-trial, and delta log-likelihood diagnostics.
- `fit_mean_omega_alpha_model.py` - Loads animal condition-fit Gamma/Omega posteriors from selectable fit families, averages across animals with SEM error bars, fits the shared alpha interaction model to mean Gamma/Omega, saves fit diagnostic figures, exports 5-parameter `w` summaries, includes an analytical-vs-firing-rate Gamma/Omega formula check, and plots the ABL 40 / ILD +16 omega numerator, denominator, and omega values over an alpha sweep.
- `fit_mean_omega_alpha_model_abl20_40.py` - Runs the same mean Gamma/Omega alpha-model fit using only ABL 20 and 40 condition-fit points, while also plotting ABL 60 means and extrapolated curves; saves a dedicated fit figure and MSE summary artifact.
- `diagnostics_cond_by_cond_fit_fix_t_E_aff_w_del_go_all_animals_for_paper.py` - Diagnostics for condition-by-condition fits
- `compare_cond_gamma_omega_with_npl_alpha.py` - Compares condition-fit Gamma/Omega means against curves implied by animal-wise NPL+alpha fits, and also contrasts direct mean-fit versus per-animal MSE alpha-model refits.
- `compare_cond_gamma_omega_with_npl_alpha_ild2_delay.py` - Compares condition-fit Gamma/Omega means against curves implied by the copied NPL+alpha+ILD2-delay animal-wise fits, and overlays per-animal MSE-fitted Gamma/Omega alpha-model curves averaged across animals; supports the same selectable condition-fit source families as `fit_mean_omega_alpha_model.py` and saves source-tagged figure/CSV summaries under `NPL_alpha_ILD2_fit_results/gamma_omega_comparison/`.
- `compare_cond_gamma_omega_with_npl_alpha_abl_specific_ild2_delay.py` - Compares Gamma/Omega posterior means from the 3-param condition fits against Gamma/Omega curves implied by the all-30 NPL+alpha+ABL-specific ILD2-delay animal fits.
- `compare_cond_gamma_omega_with_npl_alpha_condition_t_E_aff_fixed_delay.py` - Compares Gamma/Omega posterior means from the 3-param condition fits against curves implied by the all-30 NPL+alpha fits with condition-specific `t_E_aff` fixed, the previous ABL-specific ILD2-delay fits, and per-animal MSE-fitted Gamma/Omega alpha-model curves.
- `explore_gamma_omega_mse_fit_objectives.py` - Fits the Gamma/Omega alpha model per animal under three MSE objectives, using both Gamma/Omega, Gamma only, or Omega only, then plots the fitted and held-out Gamma/Omega predictions in a `3 x 2` grid.
- `compare_4_param_t_E_aff_with_npl_alpha_ild2_delay.py` - Compares 4-parameter fixed animal-mean-`w` condition-fit `t_E_aff` means against continuous NPL+alpha+ILD2 animal-wise delay curves averaged across animals.
- `compare_3_param_t_E_aff_with_abl_specific_ild2_delay.py` - Compares 3-param condition-fit `t_E_aff` values with NPL+alpha+ABL-specific ILD2 delay curves under full-range and SD-observed-range averaging policies; writes the cache used by downstream fixed-delay diagnostics.
- `compare_t_E_aff_with_npl_alpha_abl_specific_ild2_and_mse_delay.py` - Compares condition `t_E_aff` with both the NPL+alpha+ABL-specific ILD2 delay curve and unconstrained per-animal MSE delay fits.
- `compare_4_param_fixed_w_with_npl_alpha_ild2_w.py` - Compares the fixed animal-wise `mean_w` used by 4-parameter fixed-`w` condition fits against animal-wise NPL+alpha+ILD2 posterior mean `w`.
- `watch_abl_specific_ild2_results_and_run_3_param_cond_fit.py` - Tmux-friendly watcher that waits for each animal's ABL-specific NPL+alpha+ILD2-delay result, checks whether all expected 3-param condition-fit pickles and corner plots already exist, and launches the one-animal condition fit only when needed.

### Watcher Usage
Run the watcher from the repo root with the virtual environment interpreter:

```bash
COND_FIT_N_JOBS=4 WATCHER_POLL_SECONDS=300 .venv/bin/python -u fit_each_condn/watch_abl_specific_ild2_results_and_run_3_param_cond_fit.py
```

Useful overrides are `DESIRED_BATCHES_OVERRIDE`, `BATCH_ANIMAL_PAIRS_OVERRIDE`, `BATCH_CSV_DIR_OVERRIDE`, `ABORT_PARAMS_DIR_OVERRIDE`, `ABL_SPECIFIC_RESULT_DIR_OVERRIDE`, `WATCHER_DRY_RUN=1`, `WATCHER_ONCE=1`, and `COND_FIT_MAX_FAILED_ATTEMPTS=3`. Child logs are written under `fit_each_condn/watcher_logs/`, per-condition fit failures are recorded under `fit_each_condn/each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_failed_conditions/`, and a same-host duplicate watcher is prevented by `fit_each_condn/watcher_locks/watcher.lock`.

### Quantile Slope vs Omega Ratio
- `compare_quantile_slopes_with_cond_omega.py` - Recomputes Fig. 1 RT-quantile scaling slopes, compares `1 + slope` against condition-fit `omega_60 / omega_ABL` for ABL 20 and 40 using selectable condition-fit families, and saves averaged and animal-wise comparison plots plus an animal-wise CSV.
- `omega_slope_compare_utils.py` - Shared helpers for loading batch CSVs, recomputing quantile slopes, aggregating signed condition-fit omega values into absolute-ILD values, and computing averaged or animal-wise omega ratios.

### Posterior Visualization
- `plot_gamma_omega_posteriors_all_animals.py` - Creates PDF files with gamma and omega posterior distributions for each animal (10 ILDs × 3 ABLs grid)
- `plot_gamma_omega_posteriors_all_animals_by_stimulus.py` - Plots gamma and omega posterior distributions aggregated across all animals for each stimulus condition
- `plot_5_param_delay_by_abl_ild.py` - Aggregates 5-parameter condition fits across animals and plots `t_E_aff`, `del_go`, and `w` by ILD, colored by ABL, with SEM error bars.
- `plot_4_param_fix_w_mean_t_E_aff_by_abl_ild.py` - Aggregates 4-parameter fixed animal-mean-`w` condition fits across animals and plots `t_E_aff` by ILD, colored by ABL, with SEM error bars.
- `plot_5_param_omega_delay_posteriors_abl60_ild_pm16.py` - Overlays animal-wise 5-parameter posterior histograms for omega and `t_E_aff` at ABL 60 and ILD ±16.
- `plot_5_param_delay_distribution_abl60_ild_pm16.py` - Loads 5-parameter condition fits for all animals at ABL 60 and ILD ±16, computes VP posterior means, and plots the pooled across-animal `t_E_aff` distribution with a 72.5 ms reference line.
- `plot_5_param_corner_omega_t_E_aff_abl60_ild16.py` - Creates per-animal mini-corner posterior panels for omega and `t_E_aff` at a selected ABL/ILD, including 2.5% and 97.5% intervals.
- `plot_5_param_corner_ellipse_omega_t_E_aff_abl60_ild16.py` - Creates a combined corner-style summary of animal posterior means and covariance ellipses for omega, `t_E_aff`, and `w`.
- `plot_5_param_animal_param_lines_omega_t_E_aff.py` - Plots animal-wise posterior mean parameter values for omega, `t_E_aff`, and `w`, with across-animal mean and median reference lines.

### Utility Scripts
- `led_off_gamma_omega_pdf_utils.py` - PDF/CDF functions for gamma/omega models, including vectorized implementations.
- `gamma_omega_alpha_utils.py` - Shared helpers for loading condition-fit posterior means from selectable filename suffixes, aggregating Gamma/Omega across animals, and evaluating the alpha interaction model.

### Data Files
- `each_animal_cond_fit_gama_omega_pkl_files/` - VBMC fit results for each condition (ABL, ILD) per animal
- `each_animal_cond_fit_5_params_pkl_files/` - VBMC fit results for condition-by-condition Gamma/Omega/`t_E_aff`/`w`/`del_go` fits.
- `each_animal_cond_fit_4_params_fix_w_mean_pkl_files/` - VBMC fit results for condition-by-condition Gamma/Omega/`t_E_aff`/`del_go` fits with animal-wise mean `w` fixed.
- `each_animal_cond_fit_4_params_fix_w_mean_corner_plots/` - Per-condition corner plots for the 4-param fixed-mean-`w` fits.
- `each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_pkl_files/` - VBMC fit results for condition-by-condition Gamma/Omega/`t_E_aff` fits with `w` and `del_go` fixed from the ABL-specific NPL+alpha+ILD2-delay result key `vbmc_norm_alpha_abl_specific_ild2_delay_tied_results`; filenames use suffix `_FIX_w_del_go_FROM_ABL_SPECIFIC_ILD2_3_params`.
- `each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_corner_plots/` - Per-condition corner plots for the 3-param fixed-`w`/`del_go` ABL-specific ILD2-source fits.
- `svi_gamma_omega_fixed_from_animal_svi_condition_delay_results/all_observed_with_30k_reruns/` - Consolidated condition-by-condition Gamma/Omega NumPyro SVI outputs: all 864 observed conditions across 30 animals, with the 20 extra-step rerun conditions replacing the original non-stable `all_observed` rows; the tracked `consolidation_manifest.json` records the base/rerun source folders and replacement counts while the large generated outputs remain ignored and backed up externally.
- `svi_condition_gamma_omega_vs_npl_alpha_svi_comparison/` - Gamma/Omega comparison outputs for consolidated condition SVI fits versus the matching animal-wise NPL+alpha SVI expression curves and per-animal MSE alpha-model fits.
- `each_animal_cond_fit_gama_omega_pkl_files_LAPSES/` - Condition fits with lapse parameters
- `quantile_slope_vs_cond_omega_ratio.png/.pkl` - Outputs from `compare_quantile_slopes_with_cond_omega.py` for the averaged quantile-slope vs omega-ratio comparison.
- `animalwise_quantile_slope_vs_cond_omega_ratio.png/.csv` - Animal-wise `2 x 5` comparison grid and table from `compare_quantile_slopes_with_cond_omega.py`.
