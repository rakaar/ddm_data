# lapse in PSIAM-TIED model.

**`simple_lapse_test.py`**
- Testing likelihood with simulated data - lapse model

**`test_lapses.py`**
- VBMC on simulated data - lapses, pro + TIED

**`test_lapse_sim_data_fix_rt_lapse.py`**
- VBMC on simulated data with fixed lapse RT window (0.9s) - only fits lapse_prob
- Similar to test_lapses.py but with T_lapse_max fixed to 0.9 and not fitted as a parameter

**`lapses_fit_single_animal.py`** (located in `../fit_animal_by_animal/`)
- VBMC fit on exp data single animal: lapses + vanilla model

---

## New Lapse Model Files (in `../fit_animal_by_animal/`)

**`test_lapse_model_fit_on_accuracy.py`**
- Scipy curve_fit for lapse models (log-odds & psychometric) with ILD_bias parameter
- Fits both unbiased lapse (a/2) and biased lapse (a*lapse_pR) models

**`test_lapse_model_fit_on_accuracy_no_ILD_bias.py`**
- Same as above but **without ILD_bias** parameter (ILD_bias = 0)
- Fits unbiased and biased lapse models with parameters: [a, d, th, lapse_pR]

**`test_lapse_model_fit_on_accuracy_no_lapses.py`**
- Model with **no lapses (a=0 fixed)** and **ILD_bias restored**
- Only one function per fit type since lapse_pR is not needed
- Fits parameters: [d, th, ILD_bias]

**`vbmc_fit_logodds_lapse_biased.py`**
- VBMC fit of biased lapse model in log-odds space using posterior mode

**`vbmc_fit_psychometric_lapse_biased.py`**
- VBMC fit of biased lapse model in probability space using posterior mode

**`calculate_sigma_for_vbmc.py`**
- Calculate empirical standard deviations for Gaussian likelihoods in VBMC



