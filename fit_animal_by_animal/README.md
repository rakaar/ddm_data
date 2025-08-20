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


- `decoding_conf_NEW_psychometric_fit_vbmc_all_animals_pot_supp_for_paper.py` - psychometric ONLY fits
- `decoding_conf_psy_fit_see_rtds_per_animal.py`: RTDs of above fit, per animal
- `decoding_conf_psy_fit_see_rtds_supp_for_paper.py` - RTDs of above fit, average of all animals

# Potential supplementary
- `all_animals_rtd_cdf_plots.py` - CDF for min RT Cut off
- `fit_qq_animals_and_avg.py` - Q - Q_60 vs Q_60