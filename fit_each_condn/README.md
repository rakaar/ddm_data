# Fit Each Condition

This folder contains scripts for fitting condition-by-condition gamma and omega parameters for each animal.

## Condition-by-Condition Fitting

### Fitting Scripts
- `fit_single_rat_condn_by_condn_fix_t_E_w_del_go_all_animals_loop_for_paper.py` - Fit condition-by-condition gamma/omega for each animal with t_E_aff, w, del_go fixed
- `fit_single_rat_condn_by_condn_4_params_fix_w_mean_all_animals_loop.py` - Fit condition-by-condition Gamma/Omega/`t_E_aff`/`del_go` for all animals while fixing each animal's `w` to the mean from `five_param_w_mean_median_by_animal.csv`; saves pickles and per-condition corner plots in separate 4-param fixed-`w` output folders.
- `fit_single_rat_condn_by_condn_3_params_fix_w_del_go_from_abl_specific_ild2_all_animals_loop.py` - Fit condition-by-condition Gamma/Omega/`t_E_aff` for each animal after fixing `w` and `del_go` to posterior means from the completed ABL-specific NPL+alpha+ILD2-delay animal fit; skips animals whose upstream fit result is not ready and saves pickles/corner plots in dedicated 3-param output folders.
- `fit_mean_omega_alpha_model.py` - Loads animal condition-fit Gamma/Omega posteriors from selectable fit families, averages across animals with SEM error bars, fits the shared alpha interaction model to mean Gamma/Omega, saves fit diagnostic figures, exports 5-parameter `w` summaries, includes an analytical-vs-firing-rate Gamma/Omega formula check, and plots the ABL 40 / ILD +16 omega numerator, denominator, and omega values over an alpha sweep.
- `fit_mean_omega_alpha_model_abl20_40.py` - Runs the same mean Gamma/Omega alpha-model fit using only ABL 20 and 40 condition-fit points, while also plotting ABL 60 means and extrapolated curves; saves a dedicated fit figure and MSE summary artifact.
- `diagnostics_cond_by_cond_fit_fix_t_E_aff_w_del_go_all_animals_for_paper.py` - Diagnostics for condition-by-condition fits
- `compare_cond_gamma_omega_with_npl_alpha.py` - Compares condition-fit Gamma/Omega means against curves implied by animal-wise NPL+alpha fits, and also contrasts direct mean-fit versus per-animal MSE alpha-model refits.
- `compare_cond_gamma_omega_with_npl_alpha_ild2_delay.py` - Compares condition-fit Gamma/Omega means against curves implied by the copied NPL+alpha+ILD2-delay animal-wise fits, and overlays per-animal MSE-fitted Gamma/Omega alpha-model curves averaged across animals; supports the same selectable condition-fit source families as `fit_mean_omega_alpha_model.py` and saves source-tagged figure/CSV summaries under `NPL_alpha_ILD2_fit_results/gamma_omega_comparison/`.
- `compare_4_param_t_E_aff_with_npl_alpha_ild2_delay.py` - Compares 4-parameter fixed animal-mean-`w` condition-fit `t_E_aff` means against continuous NPL+alpha+ILD2 animal-wise delay curves averaged across animals.
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
- `each_animal_cond_fit_gama_omega_pkl_files_LAPSES/` - Condition fits with lapse parameters
- `quantile_slope_vs_cond_omega_ratio.png/.pkl` - Outputs from `compare_quantile_slopes_with_cond_omega.py` for the averaged quantile-slope vs omega-ratio comparison.
- `animalwise_quantile_slope_vs_cond_omega_ratio.png/.csv` - Animal-wise `2 x 5` comparison grid and table from `compare_quantile_slopes_with_cond_omega.py`.
