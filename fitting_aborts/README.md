# VBMC Fitting aborts after realizing right LED time
- `LED_off_on_data_fit_aborts_with_trunc_and_censor.ipynb` - LED off, data, Left trunc, right censor
- `vbmc_V_A_step_jump_fit_LED_ON_data.ipynb` - LED on, V_A step jump, data, Left trunc, right censor
- `compare_V_A_step_LED_on_vanila_LED_off_with_data.ipynb` - compare LED off and LED on aborts fit with Left trunc and right censor
- `vbmc_ONLY_step_jump_fit_LED_ON_data.ipynb` - fit only post LED drift, other pararms are same as that of LED off fit
- `compare_ONLY_V_A_step_LED_on_vanila_LED_off_with_data.ipynb` - Compared LED off and LED on post-LED drift fit
- `V_A_step_jump_fit_censor_post_LED_sim_data.ipynb` - Post LED censoring on simulated data
- `V_A_step_jump_fit_censor_post_LED_real_data.ipynb` - Post LED censoring on real data, Able to fit Post LED aborts with step jump change in drift
- `V_A_step_jump_fit_censor_post_LED_real_data.py` - python file, same as above
- `post_LED_drift_and_motor_fit_VBMC.ipynb` - LED on data, fits post-LED drift and motor delay (2 params), V_A and theta_A fixed
- `aborts_animal_wise_explore.py` - plots abort rate vs LED onset time per animal for LED ON/OFF, with peak time and area calculations. **Includes permutation test for abort-fraction differences across LED ON conditions (left/right/bilateral inhibition)**
- `less_than_300_LED7.py` - analyzes aborts with timed_fix < 300ms for LED7 session across animals
- `explore_animal_variability_LED_ON.py` - **NEW: explores animal-by-animal variability in LED ON abort patterns and statistics**
- `plot_vbmc_param_summary_per_animal.py` - **NEW: generates summary plots of VBMC parameter estimates per animal for LED ON fits**
- `simulate_and_fit_proactive_all_at_once.py` - Simulates proactive process with single bound accumulator, drift change after LED onset, and compares theoretical vs simulated RT distributions for LED ON/OFF trials
- `testing_v1_proactive_LED_theory.py` - Same as above but with left truncation (T_trunc=0.6s) and censoring verification. Includes theoretical PDF calculations with truncation, Monte Carlo averaging over (t_stim, t_LED) pairs, and survival probability calculations for censoring validation
- `test_vbmc_proactive_led_likelihood.py` - Simulates proactive LED ON/OFF data, defines truncation/censoring log-likelihood, and validates it via histogram vs theory plus censoring checks
- `vbmc_simulated_data_proactive_LED_fit.py` - VBMC parameter recovery on simulated proactive LED data. Simulates 3000 trials (V_A_base=1.8, V_A_post_LED=2.4, theta_A=1.5, t_aff=40ms, t_effect=35ms, motor_delay=50ms, T_trunc=0.3s), fits all 6 params with trapezoidal priors, outputs corner plot, RTD comparison, and RT-wrt-LED histograms
- `vbmc_real_data_proactive_LED_fit.py` - VBMC fitting of real animal data with proactive LED model. Set ANIMAL_IDX at top to select animal. Fits all 6 params, outputs corner plot, RTD comparison (data vs theory vs sim), and RT-wrt-LED histograms (data vs sim). Uses trial index sampling to preserve (t_LED, t_stim) correlation
- `vbmc_real_data_proactive_LED_fit_all_animals_agg.py` - Same as above but aggregates data from ALL animals instead of fitting a single animal. Outputs vbmc_real_all_animals_*.pdf/pkl files
- `vbmc_real_data_proactive_LED_fit_all_animals_agg_CORR_ID_no_trunc_exp_lapse.py` - Aggregates data from ALL animals, uses CORR_ID logic, no truncation, exponential lapse.
- `plot_data_rt_wrt_led_all_animals.py` - generates data-only RT wrt LED plots for all animals in one row, uses same filtering and abort/truncation logic as vbmc_real_data_proactive_LED_fit_CORR_ID.py, plots LED ON/OFF area-weighted histograms (density * abort fraction), x-range -0.2 to 0.2, bin width 0.05, saves PNG data_rt_wrt_led_all_animals_1x6.png
- `plot_data_rt_wrt_led_trunc_vs_no_trunc_per_animal.py` - **NEW: compares RT wrt LED distributions with vs. without truncation (removing aborts < T_trunc) for each animal in a grid layout.**
- `plot_vbmc_param_summary_per_animal_NO_TRUNC_with_lapse.py` - **NEW: generates summary error-bar plots (mean ± 95% CI) of VBMC parameters per animal + aggregate for the NO_TRUNC_with_lapse model.**
- `tachometric_and_abort_fraction_analysis.py` - **DELETED: functionality merged into other analysis scripts**
- `vbmc_compare_LED_fit_average_animals.py` - UPDATED: now supports both mean and median aggregation methods via AGG_METHOD parameter, changed animal list to [93, 98, 99, 100], updated bin width to 0.005, outputs files with aggregation method in filename
- `vbmc_real_data_proactive_LED_fit_CORR_ID.py` - UPDATED: added LED ON trial upweighting (weight=10) in compute_trial_loglike function to emphasize LED ON trials during fitting
- `check_proactive_LED_with_identifiable_params.py` - Sanity-check script comparing equivalent simulator parameterizations for identifiable delays
- `bads_real_data_proactive_LED_fit_CORR_ID.py` - Same as vbmc_real_data_proactive_LED_fit_CORR_ID.py but uses BADS (Bayesian Adaptive Direct Search) for point-estimate optimization instead of VBMC. Minimizes negative log-likelihood directly (no priors). Outputs best-fit params, RTD comparison, and RT-wrt-LED diagnostics
- `vbmc_compare_LED_fit_average_animals.py` - Averages RT-wrt-LED abort-rate scaled histograms across all animals. Loads per-animal VBMC posteriors (CORR_ID), computes data and simulated scaled histograms per animal, then averages and plots
- `sum_invariant_delays_proactive_rtds.py` - Utilities/experiments for invariances in proactive RTDs with combined delay parameters
- `test_stim_timing_distributions.py` - Debug script to verify if t_stim and t_LED distributions differ between LED ON/OFF trials (they do not)
- `post_LED_censor_utils.py` - Utility functions for post-LED censoring (cum_A_t_fn etc.)


- `fit_added_noise/psiam_tied_dv_map_utils_with_PDFs.py` - has post LED effect funcs - `stupid_f_integral` and `PA_with_LEDON_2`

## Proactive Process Model Documentation

See [PROACTIVE_MODEL_DOCUMENTATION.md](PROACTIVE_MODEL_DOCUMENTATION.md) for detailed explanation of the proactive process model, theoretical functions, truncation, censoring, and likelihood calculations used for model fitting.