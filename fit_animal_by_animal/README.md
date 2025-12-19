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
- `fit_single_rat_condn_by_condn_fix_t_E_w_del_go_all_animals_loop_for_paper.py` - Fit cond by cond for each animal
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
    - Column 6: Panel **c** (rate_norm_l vs lapse scatter with ellipses)
  - Loads all data from pickle files generated by `lapses_supp_figure_save_data.py`
  - Outputs: `supp_lapses_figure_1x5.png` and `supp_lapses_figure_1x5.pdf`


### Data Files
- `lapse_parameters_all_animals.pkl` - Lapse parameters from vanilla+lapse and norm+lapse fits
- `supp_lapses_distr_plot.pkl` - Lapse rate histogram data
- `gamma_sep_by_median_lapse_rate_data.pkl` - Gamma data split by median lapse rate
- `rate_norm_l_vs_lapse_prob_data.pkl` - Scatter plot data with ellipses


# animal wise, model wise
- animal_wise_vanila_fit.py
- animal_wise_norm_fit.py

# Fitting related plots
- `compare_animal_params_for_paper.py` - Plots parameters (mean + 95% CI) for each animal across multiple models (aborts, vanilla tied, norm tied, time-varying norm tied). Outputs PDFs like `compare_animals_all_batches_vbmc_norm_tied_results.pdf`.
- `corner_cum_animal_params_for_paper_norm.py` - Creates corner-style scatter plots specifically for the normalized model (`vbmc_norm_tied_results`), where each point represents an animal's mean parameter values.

# Fig 4
- `corner_cum_animal_params_for_paper_norm.py` - Corner plot of norm model parameters with per-animal posterior ellipses and ranked diagonal panels
- `fig4_all_using_template.py` - Figure 4 standalone: 2x2 grid with psychometric, quantiles, gamma, and slopes plots
- `figure_4_with_corner_using_template.py` - Combined Figure 4 + corner plot: 2x2 fig4 on left, 4x4 corner plot on right