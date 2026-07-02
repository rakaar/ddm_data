# Files
## Data
- batch_csvs: contains the experimental data for each animal in format of batch_{batch_name}_valid_and_aborts.csv

## Fit animal wise
- `animal_wise_fit_3_models_script_refactor.py`: Specify batch name and conditions, fit animal data - aborts, TIED + 3 variants
- `animal_wise_plotting_utils.py`:  Diagnostics Plotting utils for animal wise fit
- `time_vary_norm_utils.py`  - Likelihood funcs for animal wise fitting
- `proactive_plus_lapse_plus_reactive_uitls.py` - Shared likelihood helpers for proactive + lapse + reactive fitting, including right-truncated valid-trial logpdfs and the newer choice-collapsed RT-density helper used by the LED-OFF no-choice ABL-delay fits.
- `check_fig_8G.py`: Exploratory firing-rate/gain calculation for a Figure 8G-style binaural interaction panel across ABL and ILD grids.
- `plot_intended_fix_distributions.py`: Plots LED7 and LED8 `intended_fix` histograms from `batch_csvs` with shifted-exponential fits and saves one PNG per batch.
- `print_LED8_session_types_and_LED_trial_values.py`: Prints LED8 `session_type` groups and their unique `LED_trial` values from `outLED8.csv`.
- `save_LED8_filtered_valid_and_aborts_rtwrtstim_le_1.py`: Scratch/export helper for filtering LED8 rows by session/training/repeat/LED/trial outcome and checking `RTwrtStim <= 1` criteria.
- `plot_gamma_omega_alpha_sweep.py`: Exploratory plot of Gamma/Omega curves over ILD and ABL while sweeping the alpha interaction parameter in the binaural firing-rate expression.

##  Animal wise exploration of TIED

### Psychometrics
- `make_all_animals_psycho_single_figure.py` - psychometric of each animal
- `aggregate_psychometric_by_abl.py` - aggregate psychometric of all animals by ABL 20,40,60
- `plot_slope_ratios_histograms.py` - slope fit psychometric and variability within ABL vs variability within animals

### Chronometrics
- `mean_chrono_plot.py` - Chronometric of each animal
- `aggregate_chrono.py` - Chronometric averaged across animals for ABL 20,40,60
- `aggregate_chrono_fit_tanh_cfixed.py` - Fit shadlen expression on chronometric of each animal
- `aggregate_chrono_fit_tanh_cfixed.py` - Fit shadlen expression on mean chronometric of all animals
- `fit_chrono_tanh_curve_full_formula.py` -  Fit TIED expression on chronometric of each animal

### Quantilies
- `qq_per_animal_per_batch.py` - Q_ABL - Q_highest vs Q_highest for each animal
- `quantile_per_ABL_of_each_animal.py`- Quantile of each animal for each ABL
- `aggregate_qq_diff_across_animals.py` : aggregate Q - Q_60 vs Q_60 and line fit
- `aggregate_qq_diff_xleq_03.py`  - aggregate Q - Q_60 vs Q_60 and line fit for Q_60 <= 0.3



# decoding conf figs
- `decoding_conf_rtd_scales.py` - RTD scale
- `see_only_psycho_all_ILD_tied.py` - psychometric plots, vanila, norm
- `see_only_rtd_avg_vs_avg_norm_and_vanila.py` - RTD plots, vanila, norm

- `decoding_conf_NEW_psychometric_fit_vbmc_all_animals.py` - psychometric fit
- `decoding_conf_psy_fit_see_psycho.py`, `decoding_conf_psy_fit_see_rtds.py` - psychometric fit diagnostics

# paper
- see_only_rtd_avg_vs_avg_norm_and_vanila_for_paper.py: RTD plots, vanila, norm
- see_only_psycho_all_ILD_tied_for_paper.py: Psychometric plots, vanila, norm

## gamma cond by cond fit vs theoretical obtained from model fit
- `fit_single_rat_condn_by_condn_fix_t_E_w_del_go_all_animals_loop_for_paper.py` - Fit cond by cond for each animal (2 params: gamma, omega; w, t_E_aff, del_go fixed from parametric fit)
- `fit_single_rat_condn_by_condn_5_params_all_animals_loop.py` - Fit cond by cond for each animal (5 params: gamma, omega, t_E_aff, w, del_go all fitted per condition). Saves to `each_animal_cond_fit_5_params_pkl_files/`
- `diagnostics_cond_by_cond_fit_fix_t_E_aff_w_del_go_all_animals_for_paper.py` - Diagnostics for cond by cond fit done above
- `compare_gamma_from_cond_fit_and_norm_model_fit_for_paper.py` - Compare gamma from cond by cond fit and gamma from vanilla,norm model fit

# Fig 1 
- `animal_specific_rtd_plots_for_paper.py` - RTD plots, Q-Q plots, rescaled RTD
- `rtd_scaling_for_paper.py` - average RTD, averaged Q-Q plots, averaged rescaled RTD
- `animal_specific_rtd_plots_for_paper.py` - RTDs original, rescaled on raw KDE data section
- `animal_specific_chronometric_for_paper.py` - Chronometric plots - animal wise and average

- `aggregate_psychometric_by_abl_for_paper_fig1.py` : Psychometric 1 x 4
- `animal_specific_chronometric_for_paper_fig1.py`: Chronometric 1 x 4

- `scaling_quantiles_not_RTD_for_paper_fig1.py` : Scaling quantiles not RTD
- `animal_specific_rtd_plots_for_paper_for_fig1.py` : RTD plots, Q-Q plots, rescaled RTD for each animal, average, KDE

- `qq_animals_and_average.py` - Q vs Q with min RT cut off
- `fit_qq_animals_and_avg.py` - Q - Q_60 vs Q_60
- `all_animals_rtd_cdf_plots.py` -CDF for cut off
- `plot_slope_ratios_histograms.py` - psychometric slopes for weber law for fig 1

# fig 1
- 1x4 psychometric: `aggregate_psychometric_by_abl_for_paper_fig1.py`
- 1x4 chronometric: `animal_specific_chronometric_for_paper_fig1.py`
- 1 x 3 unscaled quantiles: `scaling_quantiles_not_RTD_for_paper_fig1.py`
- 1 x 5 line fit: `qq_animals_and_average.py`
- 1 x 1 scaled quantile: `scaling_quantiles_not_RTD_for_paper_fig1.py`
- slope vs animal, 2 histograms: `plot_slope_ratios_histograms.py`
- mean RT vs ABL, ILD: `animal_specific_chronometric_for_paper_fig1.py`
- **Ultimate**: `make_fig1.py`

# fig template
- `figure_template.py` - figure template class
- `test_fig_template.py` - example of using the template

# proving that vanilla works for small ILDs
- `prove_vanila_works_for_small_ILD.py` - Vanilla TIED fit and diagnose on abs ILD 1,2,4
- `see_only_quantile_avg_vs_avg_norm_and_vanila_for_fig2_small_ILDs_only.py` - animal avg quantiles, vanilla for small ILDs
- `see_only_psycho_all_ILD_tied_for_paper_for_fig2_small_ILDs_only.py`- animal avg psycho, vanilla for small ILDs
- `see_only_psycho_all_ILD_tied_only_small_ILDs.py` - psychometric slopes, vanilla for small ILDs
- `see_only_quantile_avg_vs_avg_norm_and_vanila_for_fig2_cont_ILD_small_ILDs.py` - animal avg quantiles, vanilla but on continous ILD for small ILDs

# fig 2
- `fig2_all_using_template.py` - figure 2 - all figures in one panel
- `fig2_all_using_template_size_match_fig4.py` - figure 2 with panel sizes matching figure 4
- `see_only_quantile_avg_vs_avg_norm_and_vanila_for_fig2.py` - animal avg quantiles, vanilla, norm
- `see_only_quantile_avg_vs_avg_norm_and_vanila_for_fig2_cont_ILD.py` - animal avg quantiles, vanilla, norm but on continous ILD
- `plot_fig2_quantile_sd_ild_mismatch_check.py` - Diagnostic side-by-side Figure 2 quantile comparison using all 30 animals: saved vanilla theory with SD contributing through `|ILD|=16` vs matched-grid theory with SD theory included only through `|ILD|=8`.
- `see_only_psycho_all_ILD_tied_for_paper_for_fig2.py` - animal avg psycho, vanilla, norm
- `compare_gamma_from_cond_fit_and_norm_model_fit_for_paper.py` - gamma: cond by cond fit vs model fit
(but cond by cond fit is done here - `fit_single_rat_condn_by_condn_fix_t_E_w_del_go_all_animals_loop_for_paper.py`)
- `see_only_psycho_all_ILD_tied.py` - psychometric slope model vs data - vanilla, norm

### psycho fit, del E,go fixed
- `decoding_conf_NEW_psychometric_fit_vbmc_all_animals_pot_supp_for_paper.py` - psychometric ONLY fits
- `decoding_conf_psy_fit_see_rtds_per_animal.py`: RTDs of above fit, per animal
- `decoding_conf_psy_fit_see_rtds_supp_for_paper.py` - RTDs of above fit, average of all animals
- `fit_animal_by_animal/decoding_conf_psy_fit_see_psycho_for_supp_for_paper.py` - psycho only fit diagnostics by plotting only psychometrics
- `decoding_conf_psy_fit_see_quantiles_supp_for_paper.py` - psycho only diagnostics by plotting only quantiles

### psycho fit , T_0 ALSO fixed
-`decoding_conf_NEW_psychometric_fit_vbmc_all_animals_pot_supp_for_paper_T0_also_fixed.py` - psychometric fits, T_0 also fixed
- `decoding_conf_psy_fit_see_psycho_for_supp_for_paper_fixed_T_0.py` - psy fits diagnostics, T_0 fixed, diagnostics-psy
- `decoding_conf_psy_fit_see_rtds_supp_for_paper_T_0_fixed.py` - psy fits diagnose, T_0 fixed, diagnostics-rtds

- `decoding_conf_psy_fit_see_quantiles_supp_for_paper_T_0_fixed.py` - psycho diagnostics, T_0 fixed, plotting only quantiles
- `decoding_conf_psy_fit_see_rtds_per_animal_T_0_also_fixed.py` - psycho diagnostics, T_0 fixed, plotting RTDs per animal

# Potential supplementary
- `all_animals_rtd_cdf_plots.py` - CDF for min RT Cut off
- `fit_qq_animals_and_avg.py` - Q - Q_60 vs Q_60

## to show not non-identifiability of params in vanilla 
- `show_unidentifiability.py` - gamma linear, sigmoid overlapping

## supp
- `corner_cum_animal_params_for_paper.py` - corner style, mean and sample covariance ellipse
- `compare_animal_params_for_paper.py` - for each model, for each animal, param analysis - tables, descending order plots, ELBO & loglike comparison
- `generate_latex_tables.py` - Generates LaTeX tables from CSV files for paper parameter tables (aborts, vanilla, norm models)

## Supplementary Lapses Figure
See [`lapses_figures_list.md`](lapses_figures_list.md) for full documentation of the 2×4 lapses supplementary figure (layout, pkl files, producer scripts).

### Additional lapse-rate diagnostics
- `rate_lambda_vs_lapse_prob_save_data.py` - Samples `rate_lambda` vs lapse rate from norm+lapse fits and saves `rate_lambda_vs_lapse_prob_data.pkl` plus scatter plot.
- `theta_vs_lapse_prob_save_data.py` - Samples `theta_E` vs lapse rate from norm+lapse fits and saves `theta_vs_lapse_prob_data.pkl` plus scatter plot.
- `lapse_rate_and_gamma_effect_on_psychometric.py` - Toy psychometric curves showing lapse-rate/gamma effects.

## Lapse model comparisons
- `compare_vanilla_norm_lapse_elbos.py` - Compare ELBOs for vanilla/norm vs lapse variants (CSV + plots).
- `compare_vanilla_norm_lapse_loglike_v2.py` - Compare log-likelihoods for vanilla/norm vs lapse variants (manual calc, CSV + plots).


# animal wise, model wise
- animal_wise_vanila_fit.py
- animal_wise_norm_fit.py

## Alpha-normalized fit path
- `save_valid_and_aborts_batches.py` - Builds per-batch valid-and-abort CSVs from `raw_data/` inputs into `raw_data/batch_csvs/`; current default excludes `abort_event == 4` so generated filenames match the watcher/fitting defaults.
- `animal_wise_norm_tied_fit_from_abort_params.py` - standalone baseline normalized TIED fit that reuses previously fitted proactive parameters from the older animal-wise result pickles.
- `animal_wise_norm_alpha_tied_fit_from_abort_params.py` - alpha-normalized TIED fit that adds `alpha` to the normalized model and saves outputs under `NPL_alpha_animal_fits/`.
- `animal_wise_norm_alpha_tied_fit_from_abort_params_ild2_delay.py` - alpha-normalized TIED fit variant that loads the same abort parameters but replaces scalar `t_E_aff` with a fitted delay surface `bias + c1*ABL + c2*|ILD| + c3*ILD^2`; saves distinct ILD2-delay outputs under `NPL_alpha_animal_fits/`.
- `animal_wise_norm_alpha_tied_fit_from_abort_params_abl_specific_ild2_delay.py` - alpha-normalized TIED fit variant with one ILD2 delay curve for each hardcoded ABL 20/40/60 (`bias[ABL] + c1[ABL]*|ILD| + c2[ABL]*ILD^2`); filters valid fit trials to those ABLs, loads abort-source pickles from `aborts_ipl_npl_time_fit_results/`, and saves under `NPL_alpha_ABL_specific_ILD2_delay_fit_results/` with result key `vbmc_norm_alpha_abl_specific_ild2_delay_tied_results` and file suffix `NORM_ALPHA_ABL_SPECIFIC_ILD2_DELAY_FROM_ABORTS`. The script supports run-control environment variables for long fits: `VBMC_MAX_FUN_EVALS`, `VBMC_FUN_EVAL_SCALE`, `N_SIM_OVERRIDE`, `DIAGNOSTIC_N_JOBS`, and `FIT_RANDOM_SEED`. It writes the posterior pkl immediately after VBMC/posterior sampling before simulation diagnostics, then overwrites it after diagnostics complete.
- `animal_wise_norm_alpha_tied_fit_from_abort_params_condition_t_E_aff_fixed_delay.py` - alpha-normalized TIED refit that fixes each condition's `t_E_aff` to the cached condition-by-condition posterior mean while fitting the shared animal-wise NPL+alpha parameters, including `w` and `del_go`; saves under `NPL_alpha_condition_t_E_aff_fixed_delay_fit_results/` with result key `vbmc_norm_alpha_condition_t_E_aff_fixed_delay_tied_results`. Use `DRY_RUN_ONLY=1` for cache/data preflight and `RUN_DIAGNOSTICS_AFTER_FIT=0` for fit-only runs.
- `NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30/CONSOLIDATION_SUMMARY.md` - tracked provenance summary for the all-30 fixed-condition-`t_E_aff` VBMC result consolidation; the large pickles/PDFs/logs in the same generated output folder remain ignored and backed up externally.
- `numpyro_svi_npl_alpha_condition_delay_single_animal.py` - exploratory `# %%` NumPyro/JAX SVI prototype for one animal, fitting shared NPL+alpha parameters plus per-condition `t_E_aff`; defaults to a full-rank Gaussian guide initialized from the upstream VBMC global posterior covariance with chunked convergence checks, and the JAX likelihood/guide helpers live in `numpyro_npl_alpha_svi_utils.py`.
- `diagnose_numpyro_svi_npl_alpha_condition_delay_single_animal.py` - single-animal SVI diagnostics for RTDs by ABL and `|ILD|` plus psychometric curves, using 1 ms model RTDs, 20 ms data bins, and the same abort-event/truncation convention as the all-animal RTD comparison.
- `run_numpyro_svi_npl_alpha_condition_delay_all_animals.py` - tmux-friendly runner for the 30 single-animal NumPyro SVI fits; runs each fit and diagnostic sequentially, saves per-animal logs/ledger rows, and bundles sampled posterior arrays plus saved summaries into `main_fullrank_variational_posterior_bundle.pkl`.
- `diagnose_numpyro_svi_npl_alpha_condition_delay_all_animals.py` - aggregate NumPyro SVI diagnostics across the 30 completed animals, plotting average RTDs by ABL, RTDs by ABL and `|ILD|` over full/zoomed windows, and psychometric curves; data RTDs include valid trials plus abort events 3/4 after batch-specific truncation, while model RTDs use 1 ms posterior-mean curves.
- `plot_numpyro_svi_npl_alpha_condition_delay_params_all_animals.py` - aggregate posterior-parameter plots for the NumPyro SVI fits: shared non-delay parameters by animal with 95% posterior intervals, plus across-animal mean `t_E_aff` versus ILD for each ABL with SEM bars.
- `plot_numpyro_svi_npl_alpha_condition_delay_patience12_loss_grid.py` - Plots a 5 x 6 grid of all 30 patience12 restore-best NPL+alpha condition-delay SVI loss curves, marking the restored-best checkpoint used for posterior sampling and the final checked step.
- `numpyro_vanilla_condition_delay_svi_utils.py` - JAX/NumPyro port of the old vanilla/IPL TIED likelihood branch (`is_norm=False`, `is_time_vary=False`) for fitting `rate_lambda`, `T_0`, `theta_E`, `w`, `del_go`, and condition-wise `t_E_aff`.
- `validate_jax_vanilla_likelihood_port.py` - Compares the vanilla/IPL JAX likelihood port against the old NumPy scalar likelihood on real LED8 and LED34_even rows before running SVI.
- `numpyro_svi_vanilla_condition_delay_single_animal.py` - Single-animal vanilla/IPL condition-delay SVI fit that loads fixed abort parameters, initializes delays from the condition `t_E_aff` cache, uses batch-specific abort truncation, and saves loss CSVs plus variational-posterior bundles.
- `run_numpyro_svi_vanilla_condition_delay_all_animals.py` - Tmux-friendly all-30 runner for the vanilla/IPL condition-delay SVI fits, discovering animals from `aborts_ipl_npl_time_fit_results/` and writing a batch ledger under the output root.
- `plot_numpyro_svi_vanilla_condition_delay_patience12_loss_grid.py` - Plots a 5 x 6 grid of all 30 patience12 restore-best vanilla/IPL condition-delay SVI loss curves.
- `compare_npl_condition_delay_vs_big_gamma_omega_delay_params.py` - Compares patience12 37-parameter NPL+alpha condition-delay SVI against the patience12 92-parameter Gamma/Omega/`t_E_aff` SVI for `w`, `del_go`, and across-animal condition delay curves.
- `compare_ipl_npl_big_svi_w_delgo_teaff.py` - Compares patience12 IPL/vanilla condition-delay SVI, 37-parameter NPL+alpha condition-delay SVI, and 92-parameter Gamma/Omega/`t_E_aff` SVI for `w`, `del_go`, and across-animal condition delay scatter/error bars.
- `plot_ipl_svi_50k_fig2_diagnostics.py` - One-row Fig 2-style diagnostic for the direct patience12 50k IPL/vanilla condition-delay SVI fit, comparing IPL functional Gamma/Omega, psychometric, slope, and paper RT-quantile predictions against the 92-parameter Gamma/Omega/delay SVI targets and behavioral data.
- `compare_three_npl_param_sources_patience12_3x5.py` - Three-row diagnostic comparing Gamma+Omega MSE, Omega-only MSE, and direct 37-parameter SVI NPL parameter sources across Gamma/Omega curves, psychometrics, slopes, and paper RT quantiles.
- `compare_three_npl_param_sources_patience12_params_by_animal.py` - 2 x 3 animal-wise parameter comparison for the same three NPL+alpha parameter sources, plotting Gamma+Omega MSE, Omega-only MSE, and direct patience12 37-parameter SVI posterior means with 95% intervals.
- `compare_three_npl_param_sources_plus_ipl_patience12_4x5.py` - Extends the three-source diagnostic with an IPL Gamma+Omega MSE row; the IPL row fits `rate_lambda`, `T_0`, and `theta_E` to patience12 92-parameter Gamma/Omega means and uses vanilla Fig 2 RTD equations with big-SVI `w`, `del_go`, and condition `t_E_aff` for psychometric, slope, and RT-quantile panels.
- `figure4_mse_npl_params_patience12_big_svi.py` - Figure 4-style psychometric, slope, and RT-quantile diagnostics using per-animal MSE-fit NPL+alpha parameters with patience12 big-SVI `w`, `del_go`, and condition `t_E_aff`; also writes a paper-quantile comparison of no SD extrapolation vs SD flat-held delay beyond `|ILD|=8`; accepts `MSE_FIG4_*` environment overrides so Gamma+Omega, Gamma-only, and Omega-only MSE variants can reuse the same script.
- `run_figure4_mse_npl_objective_variants_patience12.py` - Runs the Figure 4-style diagnostic for the three patience12 MSE objective variants, consuming the per-objective parameter CSVs from `fit_each_condn/run_patience12_big_svi_gamma_omega_mse_objective_variants.py`.
- `time_vary_norm_alpha_utils.py` - alpha-aware likelihood, CDF, and RT-density helpers used by the alpha-normalized fit.
- `time_vary_and_norm_alpha_simulators.py` - alpha-aware simulation wrappers for the normalized and time-varying normalized models.
- `test_norm_alpha_likelihood_vectorized.py` - local scalar-vs-vectorized likelihood comparison for the alpha-normalized model.

# Fitting related plots
- `compare_animal_params_for_paper.py` - Plots parameters (mean + 95% CI) for each animal across multiple models (aborts, vanilla tied, norm tied, time-varying norm tied). Outputs PDFs like `compare_animals_all_batches_vbmc_norm_tied_results.pdf`.
- `compare_npl_vs_npl_alpha_params.py` - Compares posterior mean parameters and 95% percentile intervals for baseline NPL versus NPL+alpha fits, one PDF page per parameter.
- `compare_npl_alpha_vs_ild2_delay_params.py` - Compares posterior mean parameters and 95% percentile intervals for NPL+alpha versus NPL+alpha+ILD2-delay fits, excluding delay terms because the ILD2 delay is stimulus-dependent.
- `compare_npl_alpha_ild2_elbo_loglike.py` - Matches animals across NPL, NPL+alpha, and NPL+alpha+ILD2 fits, then saves a two-page raw loglike/ELBO comparison PDF plus a per-animal CSV summary under `NPL_alpha_ILD2_fit_results/elbo_loglike_comparison/`.
- `plot_ild2_delay_heatmaps_from_results.py` - Reads copied ILD2 result pickles, reconstructs the fitted delay function on the observed ABL by `|ILD|` grid for each animal, and saves a multipage PDF with per-animal, mean, and median delay heatmaps plus printed coefficient summaries.
- `compare_rtd_psychometric_abl_specific_delay_vs_condition_delay.py` - Diagnostic RTD/psychometric comparison that keeps NPL+alpha+ABL-specific ILD2 parameters fixed and substitutes only condition-fit `t_E_aff`, including ABL-by-|ILD| RTD panels with 1 ms model curves and batch-specific abort truncation.
- `compare_fixed_condition_t_E_aff_vs_abl_specific_ild2_params_3x3.py` - 3 x 3 animal-wise comparison of the seven shared non-delay NPL+alpha parameters for the fixed condition `t_E_aff` refit versus the earlier ABL-specific ILD2-delay fit, with 95% posterior intervals.
- `compare_fixed_condition_t_E_aff_vs_abl_specific_ild2_w_del_go.py` - Compares `w` and `del_go` posterior means with 95% intervals for the refit that fixes condition `t_E_aff` versus the earlier ABL-specific ILD2-delay fit.
- `plot_abl_specific_ild2_delay_posteriors_against_bounds.py` - Overlays animal-wise ABL-specific ILD2 delay coefficient posteriors against hard/plausible VBMC bounds.
- `plot_abl_specific_ild2_posteriors_with_mse_coefficients.py` - Compares the same ABL-specific ILD2 delay posteriors against unconstrained per-animal MSE delay coefficients fit from condition `t_E_aff` values.
- `calculate_lambda_times_ell.py` - Computes per-animal mean of `rate_lambda * (1 - rate_norm_l)` from norm tied results and prints the average across animals (with optional histogram).
- `corner_cum_animal_params_for_paper_norm.py` - Creates corner-style scatter plots specifically for the normalized model (`vbmc_norm_tied_results`), where each point represents an animal's mean parameter values.
- `animal_wise_plotting_utils.py` - shared plotting helpers for the animal-wise fit PDFs; tied-model summary tables now include `alpha` when that parameter is present.

# Quantile Goodness-of-Fit: Theory vs Data at |ILD|=16

These scripts compare theoretical RT quantiles (from model fits) against empirical RT quantiles at |ILD|=16 for ABL=20,40,60. Both scripts properly handle the ILD asymmetry by averaging theoretical CDFs for ILD=+16 and ILD=-16 (since empirical data pools both).

- **`show_cond_fit_goodness_cond_vs_data.py`** - Uses **condition-by-condition gamma/omega fits**
  - Loads gamma, omega from `fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files/`
  - Loads w, t_E_aff, del_go (avg of vanilla+norm) and abort params from animal pkl files
  - Outputs: `ILD_16_cond_fit_quantiles.pkl`, `cond_fit_quantiles_theory_vs_data_ild16_v2.png`
  - See `show_cond_fit_goodness_cond_vs_data_README.md` for detailed documentation

- **`show_cond_fit_goodness_vanilla_vs_data.py`** - Uses **vanilla tied model parameters directly**
  - Computes gamma/omega from vanilla tied params using `cum_pro_and_reactive_time_vary_fn`
  - No condition-fit pkl files needed - uses only `results_{batch}_animal_{id}.pkl`
  - Outputs: `ILD_16_vanila_model_quantiles.pkl`, `vanila_fit_quantiles_theory_vs_data_ild16.png`

- **`ILD_16_quantiles_fit_compare_cond_vs_vanila.py`** - Comparison plot of both methods
  - Loads pkl files from both scripts above
  - Plots cond-fit (solid) vs vanilla (dotted) theoretical quantiles with empirical data

- **`quantile_gof_all_ILDs.py`** - Compute R² goodness-of-fit for all ILDs (1, 2, 4, 8, 16)
  - Parallel processing across ILDs for efficiency
  - Saves `quantiles_gof_ILD_{ild}.pkl` for each ILD with theory quantiles and R² metrics
  - Compares cond-fit (2-param: gamma, omega) vs vanilla model at each ILD

- **`quantile_gof_all_ILDs_more_params.py`** - Compute R² goodness-of-fit for 5-param condition fits
  - Uses condition fits where gamma, omega, t_E_aff, w, del_go are all fitted per condition
  - Reads pkl files from `each_animal_cond_fit_5_params_pkl_files/`
  - Saves `quantiles_gof_ILD_{ild}_more_params.pkl` for integration with `plot_R2_vs_ILD.py`

- **`ILD_quantiles_fit_compare_cond_vs_vanila.py`** - Generalized comparison plot for any ILD
  - Change `ILD_TARGET` (1, 2, 4, 8, or 16) at top of script
  - Loads from `quantiles_gof_ILD_{ild}.pkl` files
  - Plots cond-fit (solid) vs vanilla (dotted) with empirical data

- **`plot_R2_vs_ILD.py`** - Plot R² vs ILD summary
  - Plot 1: R² per ABL vs ILD (colored by ABL: blue=20, orange=40, green=60; dot=cond, cross=vanilla)
  - Plot 2: R² averaged across ABL vs ILD (dot=cond, cross=vanilla)
  - Plot for paper: Clean R² comparison of Cond-5p, Norm, Vanilla models
  - Includes comparison table: Cond-5p vs Norm
  - Outputs: `R2_per_ABL_vs_ILD.png`, `R2_mean_vs_ILD.png`, `R2_vs_ILD_for_paper.png/.pdf`

- **`plot_delay_vs_ILD_from_cond_fit.py`** - Plot t_E_aff (sensory delay) vs ILD
  - Extracts t_E_aff from 5-param condition fits for all animals
  - One figure per |ILD|, animals on y-axis, t_E_aff on x-axis
  - Averages +ILD and -ILD values, colors by ABL
  - Outputs: `t_E_aff_vs_animal_ILD_{1,2,4,8,16}.png`

- **`plot_w_vs_ILD_from_cond_fit.py`** - Plot w (urgency weight) vs ILD
  - Extracts w from 5-param condition fits for all animals
  - One figure per |ILD|, animals on y-axis, w on x-axis
  - Averages +ILD and -ILD values, colors by ABL
  - Outputs: `w_vs_animal_ILD_{1,2,4,8,16}.png`

- **`ABL_averaged_cond_vs_vanila_fit.py`** - ABL-averaged quantiles vs ILD plot
  - Averages RT quantiles across ABLs (20, 40, 60) for each ILD
  - Plots 5 representative quantiles (10%, 30%, 50%, 70%, 90%)
  - Compares empirical data, cond-fit, and vanilla model predictions

# Fig 4
- `corner_cum_animal_params_for_paper_norm.py` - Corner plot of norm model parameters with per-animal posterior ellipses and ranked diagonal panels
- `fig4_all_using_template.py` - Figure 4 standalone: 2x2 grid with psychometric, quantiles, gamma, and slopes plots
- `figure_4_with_corner_using_template.py` - Combined Figure 4 + corner plot: 2x2 fig4 on left, 4x4 corner plot on right
- `generate_psy_slopes_npl_alpha_ild2_delay_for_fig4.py` - Builds psychometric and slope data pickles for the NPL+alpha+ILD2-delay model using the fitted delay surface.
- `generate_psy_npl_alpha_ild2_with_npl_delay_for_fig4.py` - Builds a hybrid psychometric/slope diagnostic where NPL+alpha+ILD2 parameters are used but evidence delay is replaced by each animal's constant NPL `t_E_aff`.
- `generate_psy_npl_alpha_ild2_with_npl_delay_delgo_for_fig4.py` - Builds a second hybrid psychometric/slope diagnostic where NPL+alpha+ILD2 parameters are used but both `t_E_aff` and `del_go` are replaced by each animal's NPL values.
- `generate_quantiles_npl_alpha_ild2_delay_for_fig4.py` - Builds RT quantile data pickles for the NPL+alpha+ILD2-delay model on discrete and continuous ILD grids; current top-of-file quantile settings can write the q10 pickle used for 0.1:0.1:0.9 diagnostics.
- `plot_rtds_npl_alpha_ild2_delay_per_abl_for_fig4.py` - Builds ABL 20/60, `|ILD|` 1/2/4/8/16 RTD diagnostics for the NPL+alpha+ILD2-delay model versus data using 20 ms bins, saved in `figure_4_diagnostics_part2`.
- `plot_quantiles_npl_alpha_ild2_delay_per_abl_for_fig4.py` - Plots per-ABL model-vs-data RT quantiles from saved quantile pickles, including the q10 NPL+alpha+ILD2-delay ABL 20/60 diagnostic.
- `figure_4_npl_alpha_ild2_delay_using_template.py` - Recreates the Figure 4 psychometric, quantile, and slope panels for the NPL+alpha+ILD2-delay model from the generated pickles; masks SD model psychometric entries at `|ILD| > 8` and recomputes slopes from that corrected psychometric grid.
- `figure4_mse_npl_params_patience12_big_svi.py` - Builds a 2 x 3 Figure 4-style diagnostic using per-animal MSE-fit NPL+alpha parameters with `w`, `del_go`, and condition-wise `t_E_aff` from the all-30 patience12 big Gamma/Omega/delay SVI fit; includes psychometric, slopes, all-decile continuous/discrete RT quantiles, paper-style 10/30/50/70/90 quantile versions, and a side-by-side paper-quantile comparison with SD delays flat-held beyond `|ILD|=8`.
- `compare_npl_vs_npl_alpha_ild2_quantiles_slopes.py` - Compares baseline NPL, NPL+alpha+ILD2-delay, and vanilla TIED diagnostics, including quantiles, psychometric slopes, empirical-grid accuracy, slope-vs-accuracy panels, and selected animal psychometric/P(correct) checks.
- `permutation_test_ild2_minus_npl_slopes.py` - Paired label-shuffle/sign-flip test of ILD2-minus-NPL psychometric slope differences for each ABL.
- `plot_led7_93_npl_vs_ild2_psychometric_formula.py` - LED7-93 diagnostic comparing saved model psychometric fits, direct gamma-formula psychometrics, and reactive-only area psychometrics for NPL and NPL+alpha+ILD2-delay.
- `plot_sd_slope_ild16_inclusion_check.py` - SD-batch diagnostic measuring how model psychometric slopes change when model-only `|ILD|=16` points are excluded and slopes are refit only on empirical ILDs.
- `plot_npl_vs_ild2_slope_accuracy_empirical_grid.py` - 2x5 IPL/NPL/NPL+alpha+ILD2/hybrid-delay/hybrid-delay-go model-vs-data scatter plot for psychometric slopes and accuracy, with model values evaluated only on each animal's empirical stimulus grid.
- `plot_selected_model_slope_accuracy_empirical_grid.py` - 6x3 selected-model empirical-grid slope/accuracy diagnostic for NPL, NPL+alpha+ILD2, and IPL / vanilla TIED, including paired label-swap tests for signed model-data bias and bootstrap CIs for mean absolute error, saved in `figure_4_diagnostics_part2`.
- `plot_npl_vs_ild2_accuracy_change.py` - Per-animal NPL vs NPL+alpha+ILD2 accuracy comparison on each animal's empirical ABL/ILD grid, with sorted percent-change diagnostics.
- `plot_ild2_accuracy_change_vs_params.py` - Compares NPL+alpha+ILD2 accuracy changes against fitted parameter means and parameter deltas, saving scatter diagnostics and a CSV summary.
- `permutation_test_model_vs_data_accuracy.py` - Paired label-swap test of model-vs-data accuracy differences for IPL, NPL, and NPL+alpha+ILD2, with paired t-test and Wilcoxon checks.
- `plot_fig4_quantile_sd_ild_mismatch_check.py` - Diagnostic side-by-side Figure 4 quantile comparison using all 30 animals: saved NPL theory with SD contributing through `|ILD|=16` vs matched-grid theory with SD theory included only through `|ILD|=8`.
