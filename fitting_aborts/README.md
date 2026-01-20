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
- `simulate_and_fit_proactive_all_at_once_truncate_300.py` - Same as above but with left truncation (T_trunc=0.6s) and censoring verification. Includes theoretical PDF calculations with truncation, Monte Carlo averaging over (t_stim, t_LED) pairs, and survival probability calculations for censoring validation


- `fit_added_noise/psiam_tied_dv_map_utils_with_PDFs.py` - has post LED effect funcs - `stupid_f_integral` and `PA_with_LEDON_2`

## Proactive Process Model Documentation

See [PROACTIVE_MODEL_DOCUMENTATION.md](PROACTIVE_MODEL_DOCUMENTATION.md) for detailed explanation of the proactive process model, theoretical functions, truncation, censoring, and likelihood calculations used for model fitting.