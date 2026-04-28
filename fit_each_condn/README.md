# Fit Each Condition

This folder contains scripts for fitting condition-by-condition gamma and omega parameters for each animal.

## Condition-by-Condition Fitting

### Fitting Scripts
- `fit_single_rat_condn_by_condn_fix_t_E_w_del_go_all_animals_loop_for_paper.py` - Fit condition-by-condition gamma/omega for each animal with t_E_aff, w, del_go fixed
- `fit_mean_omega_alpha_model.py` - Loads animal condition-fit Gamma/Omega posteriors from selectable fit families, averages across animals with SEM error bars, fits the shared alpha interaction model to mean Gamma/Omega, saves fit diagnostic figures, exports 5-parameter `w` summaries, includes an analytical-vs-firing-rate Gamma/Omega formula check, and plots the ABL 40 / ILD +16 omega numerator, denominator, and omega values over an alpha sweep.
- `diagnostics_cond_by_cond_fit_fix_t_E_aff_w_del_go_all_animals_for_paper.py` - Diagnostics for condition-by-condition fits

### Quantile Slope vs Omega Ratio
- `compare_quantile_slopes_with_cond_omega.py` - Recomputes Fig. 1 RT-quantile scaling slopes, compares `1 + slope` against condition-fit `omega_60 / omega_ABL` for ABL 20 and 40 using selectable condition-fit families, and saves averaged and animal-wise comparison plots plus an animal-wise CSV.
- `omega_slope_compare_utils.py` - Shared helpers for loading batch CSVs, recomputing quantile slopes, aggregating signed condition-fit omega values into absolute-ILD values, and computing averaged or animal-wise omega ratios.

### Posterior Visualization
- `plot_gamma_omega_posteriors_all_animals.py` - Creates PDF files with gamma and omega posterior distributions for each animal (10 ILDs × 3 ABLs grid)
- `plot_gamma_omega_posteriors_all_animals_by_stimulus.py` - Plots gamma and omega posterior distributions aggregated across all animals for each stimulus condition
- `plot_5_param_delay_by_abl_ild.py` - Aggregates 5-parameter condition fits across animals and plots `t_E_aff`, `del_go`, and `w` by ILD, colored by ABL, with SEM error bars.
- `plot_5_param_omega_delay_posteriors_abl60_ild_pm16.py` - Overlays animal-wise 5-parameter posterior histograms for omega and `t_E_aff` at ABL 60 and ILD ±16.
- `plot_5_param_corner_omega_t_E_aff_abl60_ild16.py` - Creates per-animal mini-corner posterior panels for omega and `t_E_aff` at a selected ABL/ILD, including 2.5% and 97.5% intervals.
- `plot_5_param_corner_ellipse_omega_t_E_aff_abl60_ild16.py` - Creates a combined corner-style summary of animal posterior means and covariance ellipses for omega, `t_E_aff`, and `w`.
- `plot_5_param_animal_param_lines_omega_t_E_aff.py` - Plots animal-wise posterior mean parameter values for omega, `t_E_aff`, and `w`, with across-animal mean and median reference lines.

### Utility Scripts
- `led_off_gamma_omega_pdf_utils.py` - PDF/CDF functions for gamma/omega models, including vectorized implementations.
- `gamma_omega_alpha_utils.py` - Shared helpers for loading condition-fit posterior means from selectable filename suffixes, aggregating Gamma/Omega across animals, and evaluating the alpha interaction model.

### Data Files
- `each_animal_cond_fit_gama_omega_pkl_files/` - VBMC fit results for each condition (ABL, ILD) per animal
- `each_animal_cond_fit_5_params_pkl_files/` - VBMC fit results for condition-by-condition Gamma/Omega/`t_E_aff`/`w`/`del_go` fits.
- `each_animal_cond_fit_gama_omega_pkl_files_LAPSES/` - Condition fits with lapse parameters
- `quantile_slope_vs_cond_omega_ratio.png/.pkl` - Outputs from `compare_quantile_slopes_with_cond_omega.py` for the averaged quantile-slope vs omega-ratio comparison.
- `animalwise_quantile_slope_vs_cond_omega_ratio.png/.csv` - Animal-wise `2 x 5` comparison grid and table from `compare_quantile_slopes_with_cond_omega.py`.
