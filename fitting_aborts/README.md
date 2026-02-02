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
- `aborts_animal_wise_explore.py` - plots abort rate vs LED onset time per animal for LED ON/OFF, with peak time and area calculations
- `less_than_300_LED7.py` - analyzes aborts with timed_fix < 300ms for LED7 session across animals
- `simulate_and_fit_proactive_all_at_once.py` - Simulates proactive process with single bound accumulator, drift change after LED onset, and compares theoretical vs simulated RT distributions for LED ON/OFF trials
- `testing_v1_proactive_LED_theory.py` - Same as above but with left truncation (T_trunc=0.6s) and censoring verification. Includes theoretical PDF calculations with truncation, Monte Carlo averaging over (t_stim, t_LED) pairs, and survival probability calculations for censoring validation
- `test_vbmc_proactive_led_likelihood.py` - Simulates proactive LED ON/OFF data, defines truncation/censoring log-likelihood, and validates it via histogram vs theory plus censoring checks
- `vbmc_simulated_data_proactive_LED_fit.py` - VBMC parameter recovery on simulated proactive LED data. Simulates 3000 trials (V_A_base=1.8, V_A_post_LED=2.4, theta_A=1.5, t_aff=40ms, t_effect=35ms, motor_delay=50ms, T_trunc=0.3s), fits all 6 params with trapezoidal priors, outputs corner plot, RTD comparison, and RT-wrt-LED histograms
- `vbmc_real_data_proactive_LED_fit.py` - VBMC fitting of real animal data with proactive LED model. Set ANIMAL_IDX at top to select animal. Fits all 6 params, outputs corner plot, RTD comparison (data vs theory vs sim), and RT-wrt-LED histograms (data vs sim). Uses trial index sampling to preserve (t_LED, t_stim) correlation
- `vbmc_real_data_proactive_LED_fit_all_animals_agg.py` - Same as above but aggregates data from ALL animals instead of fitting a single animal. Outputs vbmc_real_all_animals_*.pdf/pkl files
- `vbmc_real_data_proactive_LED_fit_all_animals_agg_CORR_ID.py` - All-animals aggregate fit using identifiable delay parameterization (del_a_minus_del_LED, del_m_plus_del_LED); includes updated likelihood, posterior plots, and diagnostics
- `check_proactive_LED_with_identifiable_params.py` - Sanity-check script comparing equivalent simulator parameterizations for identifiable delays
- `sum_invariant_delays_proactive_rtds.py` - Utilities/experiments for invariances in proactive RTDs with combined delay parameters
- `test_stim_timing_distributions.py` - Debug script to verify if t_stim and t_LED distributions differ between LED ON/OFF trials (they do not)


- `fit_added_noise/psiam_tied_dv_map_utils_with_PDFs.py` - has post LED effect funcs - `stupid_f_integral` and `PA_with_LEDON_2`

## Proactive Process Model Documentation

See [PROACTIVE_MODEL_DOCUMENTATION.md](PROACTIVE_MODEL_DOCUMENTATION.md) for detailed explanation of the proactive process model, theoretical functions, truncation, censoring, and likelihood calculations used for model fitting.