# Files
## Data
- batch_csvs: contains the experimental data for each animal in format of batch_{batch_name}_valid_and_aborts.csv

## Fit animal wise
- `animal_wise_fit_3_models_script_refactor.py`: Specify batch name and conditions, fit animal data - aborts, TIED + 3 variants
- `animal_wise_plotting_utils.py`:  Diagnostics Plotting utils for animal wise fit
- `time_vary_norm_utils.py`  - Likelihood funcs for animal wise fitting

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

This section documents the workflow for creating the supplementary figure showing lapse rate analysis across animals.

### Data Generation
- **`lapses_supp_figure_save_data.py`** - Generates and saves data for all lapses supplementary plots
  - **Panel a**: Lapse rate distribution across animals (histogram with median line)
    - Loads lapse parameters from both vanilla+lapse and norm+lapse models
    - Averages lapse probability across both models per animal
    - Saves to: `supp_lapses_distr_plot.pkl`
  
  - **Panel b**: Gamma separated by median lapse rate (3 panels, one per ABL)
    - Loads condition-by-condition gamma fits with lapses
    - Splits animals into two groups: below median vs. at/above median lapse rate
    - Computes mean gamma and SEM for each ILD and ABL
    - Saves to: `gamma_sep_by_median_lapse_rate_data.pkl`
  
  - **Panel c**: Rate_norm_l vs lapse rate scatter plot
    - Loads norm+lapse model VBMC results
    - Samples from posterior for each animal (5000 samples/animal)
    - Computes covariance ellipses (95% confidence) for each animal
    - Fits linear regression across all samples
    - Saves to: `rate_norm_l_vs_lapse_prob_data.pkl`

### Figure Assembly
- **`supp_lapses_using_template.py`** - Assembles final 1×5 panel figure using figure template
  - Uses `figure_template.py` for consistent styling
  - Layout: 1 row × 7 columns (with gaps between groups)
    - Column 0: Panel **a** (lapse distribution histogram)
    - Column 1: Gap (0.15 width ratio)
    - Columns 2-4: Panel **b** (3 gamma plots for ABL 20, 40, 60)
    - Column 5: Gap (0.15 width ratio)
    - Column 6: Panel **c** (rate_norm_l vs lapse scatter with ellipses and sample points)
  - Loads all data from pickle files generated by `lapses_supp_figure_save_data.py`
  - Outputs: `supp_lapses_figure_1x5.png` and `supp_lapses_figure_1x5.pdf`

- **`supp_lapses_using_template_no_elipse_pts.py`** - Same as above but with cleaner Panel c
  - Panel c shows only covariance ellipses and linear fit line (no scatter points)
  - Outputs: `supp_lapses_figure_1x5_no_ellipse_pts.png` and `supp_lapses_figure_1x5_no_ellipse_pts.pdf`


### Data Files
- `lapse_parameters_all_animals.pkl` - Lapse parameters from vanilla+lapse and norm+lapse fits
- `supp_lapses_distr_plot.pkl` - Lapse rate histogram data
- `gamma_sep_by_median_lapse_rate_data.pkl` - Gamma data split by median lapse rate
- `rate_norm_l_vs_lapse_prob_data.pkl` - Scatter plot data with ellipses
- `lapse_rate_loglike_diff_data.pkl` - Lapse-rate vs log-likelihood difference data (for lapses likelihood plots)

### Additional lapse-rate diagnostics
- `rate_lambda_vs_lapse_prob_save_data.py` - Samples `rate_lambda` vs lapse rate from norm+lapse fits and saves `rate_lambda_vs_lapse_prob_data.pkl` plus scatter plot.
- `theta_vs_lapse_prob_save_data.py` - Samples `theta_E` vs lapse rate from norm+lapse fits and saves `theta_vs_lapse_prob_data.pkl` plus scatter plot.
- `lapse_rate_and_gamma_effect_on_psychometric.py` - Toy psychometric curves showing lapse-rate/gamma effects.
- `lapses_likelihood_plot_for_paper.py` - Plots lapse rate vs log-likelihood differences (NPL - (IPL + lapses) and NPL + lapses - (IPL + lapses)). Saves `lapse_rate_loglike_diff_data.pkl`.
- `plot_gamma_for_norm_lapse_for_paper.py` - Plots norm+lapse gamma across ABL=20/40/60 in a single panel using `gamma_sep_by_median_lapse_rate_data.pkl`. Outputs `gamma_norm_lapse_all_ABL.png/.pdf`.
- `params_npl_npl_plus_lapse_ordered_for_paper.py` - Plots norm vs norm+lapse params ordered by NPL lapse rate (rate_norm_l, lambda, theta_E, T_0). Outputs `param_<param>_ordered_by_npl_lapse.png/.pdf`.

## Lapse model comparisons
- `compare_vanilla_norm_lapse_elbos.py` - Compare ELBOs for vanilla/norm vs lapse variants (CSV + plots).
- `compare_vanilla_norm_lapse_loglike_v2.py` - Compare log-likelihoods for vanilla/norm vs lapse variants (manual calc, CSV + plots).


# animal wise, model wise
- animal_wise_vanila_fit.py
- animal_wise_norm_fit.py

# Fitting related plots
- `compare_animal_params_for_paper.py` - Plots parameters (mean + 95% CI) for each animal across multiple models (aborts, vanilla tied, norm tied, time-varying norm tied). Outputs PDFs like `compare_animals_all_batches_vbmc_norm_tied_results.pdf`.
- `calculate_lambda_times_ell.py` - Computes per-animal mean of `rate_lambda * (1 - rate_norm_l)` from norm tied results and prints the average across animals (with optional histogram).
- `corner_cum_animal_params_for_paper_norm.py` - Creates corner-style scatter plots specifically for the normalized model (`vbmc_norm_tied_results`), where each point represents an animal's mean parameter values.

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