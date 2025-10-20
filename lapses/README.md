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

**`lapses_fit_single_animal_norm_model.py`** (located in `../fit_animal_by_animal/`)
- VBMC fit on exp data single animal: lapses + **normalized model**
- Fits 9 parameters: `rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right`
- CLI args: `--batch`, `--animal`, `--init-type` (vanilla|norm), `--output-dir` (default: `oct_6_7_large_bounds_diff_init_lapse_fit`)
- Supports two initialization types: vanilla (small T_0, large theta_E) or norm (large T_0, small theta_E)
- Supports optional right truncation at 1s via `DO_RIGHT_TRUNCATE` flag
- Compares Norm vs Norm+Lapse models with parameter distributions and simulated RTDs
- All output files include `{init_type}` suffix for tracking initialization method

**`compare_all_four_models.py`** (located in `../fit_animal_by_animal/`)
- **Comprehensive comparison of all 4 models**: Vanilla, Vanilla+Lapse, Norm, Norm+Lapse
- Loads parameters from: baseline models (`results_{batch}_animal_{id}.pkl`) and lapse models (`vbmc_*_lapses_truncate_1s.pkl`)
- Uses `vp.sample()` to extract lapse model parameters from saved VBMC posteriors
- Simulates 1M trials per model using `simulate_psiam_tied_rate_norm` from `lapses_utils`
- Generates 3 comprehensive plots:
  - **RT Distributions**: Grid layout (ABL rows × |ILD| columns) with step histograms for all 4 models + data
  - **Psychometric Curves**: P(right) vs ILD for each ABL, all models overlaid
  - **Log Odds**: Log(P(R)/P(L)) vs ILD for each ABL
- Configuration: batch LED8, animal 105, `DO_RIGHT_TRUNCATE=True`

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

**`lapse_log_odd_psy_fit_all_animals.py`** (located in `../fit_animal_by_animal/`)
- **Comprehensive analysis of lapse models across all animals**
- Processes animals from batches: SD, LED34, LED6, LED8, LED7, LED34_even
- Batch-specific T_trunc: 0.15s for LED34_even, 0.3s for others; right truncation at 1.0s
- Performs **two independent fits** per animal:
  - **Psychometric fit**: Fits `psyc_lapse_biased` to P(right) data
- Parameters fitted: [a, d, th, lapse_pR] with R² evaluation
- **Outputs**:
  - `lapse_model_params_all_animals.pkl`: Dictionary with all fit parameters and R² values
  - `lapse_model_fits_all_animals.pdf`: Multi-page PDF with psychometric + log-odds plots per animal
  - **`lapse_model_parameter_comparison.csv`**: Sortable table of all parameters as percentages
    - `lapse_model_r2_comparison.png`: R² comparison (2 panels) sorted by average R²
    - `lapse_model_a_param_comparison.png`: Lapse rate parameter 'a' side-by-side bars with R² annotations
    - `lapse_model_lapse_pR_comparison.png`: Lapse bias parameter with reference line at 50%
- CSV table columns: Batch, Animal, R²_LogOdds (%), R²_Psychometric (%), a_LogOdds (%), a_Psychometric (%), lapse_pR_LogOdds (%), lapse_pR_Psychometric (%)
- PDF titles include all fitted parameters: a (%), d (4 decimals), th (2 decimals), lapse_pR (%), R² (%)

**`run_norm_lapse_fits_batch_animal_pairs.py`** (located in `../fit_animal_by_animal/`)
- Batch runner for norm+lapse single-animal fits over multiple (batch, animal) pairs with both init types
- Runs 24 animals × 2 init types (vanilla, norm) = 48 total fits by default
- CLI args: `--pairs` (batch:animal format), `--init-types` (vanilla and/or norm), `--output-dir` (default: `oct_6_7_large_bounds_diff_init_lapse_fit`)
- Default animals from LED34_even, LED7, LED6, LED34, LED8 batches
- Saves VBMC pkl, parameter-comparison text, and diagnostic figures for each (batch, animal, init_type) combination

**`lapses_fit_single_animal_norm_model_fix_t_E_aff_del_go.py`** (located in `../fit_animal_by_animal/`)
- Same as `lapses_fit_single_animal_norm_model.py`, but fixes `t_E_aff` and `del_go` from the average of vanilla and norm tied fits loaded from `results_{batch}_animal_{id}.pkl`.
- Fits 7 parameters: `rate_lambda, T_0, theta_E, w, rate_norm_l, lapse_prob, lapse_prob_right`.
- CLI args: `--batch`, `--animal`, `--init-type` (nor|norm), `--output-dir` (default: `oct_6_7_large_bounds_diff_init_lapse_fit_t_E_aff_del_go_fixed`)
- Output filenames include `t_E_aff_del_go_fixed`.

**`run_vanilla_lapse_fits_all_animals.py`** (located in `../fit_animal_by_animal/`)
- Batch runner for vanilla+lapse single-animal fits discovered automatically from CSVs.
- CLI args: `--batches` (defaults to SD LED34 LED6 LED8 LED7 LED34_even), `--output-dir`, `--python`, `--dry-run`, `--start-from`.
- Invokes `lapses_fit_single_animal.py` for each (batch, animal) pair.

**`lapse_model_large_bounds_elbo_analysis.py`** (located in `../fit_animal_by_animal/`)
- Analyzes VBMC convergence by extracting ELBO and stable flags from lapse model fit results across all animals and init types

**`compare_vanilla_norm_lapse_elbos.py`** (located in `../fit_animal_by_animal/`)
- **ELBO comparison notebook** for vanilla+lapse vs norm+lapse models (Oct 9-10 fits)
- Reads VBMC pkl files from `oct_9_10_vanila_lapse_model_fit_files/` and `oct_9_10_norm_lapse_model_fit_files/`
- Compares 18 common (batch, animal) pairs with original vanilla/norm model ELBOs from `results_{batch}_animal_{id}.pkl`
- Extracts convergence info: ELBO, stability flag, elbo_sd, n_iterations
- **Outputs**:
  - Formatted table showing all 8 metrics per animal (V+L stable, N+L stable, V+L ELBO, N+L ELBO, OG V ELBO, OG N ELBO)
  - Summary statistics: stability counts, mean/median/min/max ELBO improvements
  - `vanilla_norm_lapse_elbo_comparison.csv`: Full comparison table
  - `elbo_comparisons_bar_plots.png`: **3 bar plot panels** with y-axis limited to [-100, 100]:
    - Panel 1: Vanilla+Lapse ELBO - Original Vanilla ELBO
    - Panel 2: Vanilla+Lapse ELBO - Original Norm ELBO
    - Panel 3: Norm+Lapse ELBO - Original Norm ELBO
  - Green bars = positive ELBO difference (improvement), Red bars = negative (worse)
- **Key findings**: All 18/18 animals converged stably; median ELBO improvements ~+8 (vanilla+lapse) and ~+3 (norm+lapse), but mean improvements negative due to a few outliers with large ELBO drops (LED34_59: -1428, LED34_63: -752)

---

## ELBO Comparison Scripts

**`run_lapse_fit_led34_filtered.py`** (located in `../fit_animal_by_animal/`)
- Batch runner for LED34 filtered lapse fits
- Processes 4 animals: 45, 57, 59, 61
- Invokes `lapses_fit_single_animal.py` with `--is-stim-filtered` flag
- Output directory: `led34_vanila_lapse_led34_filered`
- Runs vanilla+lapse VBMC fits on stimulus-filtered data

**`compare_led34_elbos.py`** (located in `../fit_animal_by_animal/`)
- **Comprehensive comparison notebook for LED34 animals** with three comparison sections:
- **Section 1: Vanilla Filtered vs Vanilla+Lapse Filtered**
  - Compares baseline vanilla model with lapse model (both with stimulus filtering)
  - ELBO bar plot showing improvement (Lapse - Vanilla)
  - Parameter comparison tables for all 6 parameters + 2 lapse parameters
- **Section 2: Vanilla Filtered vs Vanilla Unfiltered**
  - Compares filtered vs unfiltered vanilla baseline models
  - ELBO bar plot (Filtered - Unfiltered)
  - Parameter comparison tables
- **Section 3: Vanilla+Lapse Filtered vs Vanilla+Lapse Unfiltered**
  - Compares filtered vs unfiltered lapse models
  - Parameter comparison tables for all 8 parameters (including lapse_prob, lapse_prob_right)
- **Data sources**:
  - Filtered vanilla: `led34_filter_files/vanila/vbmc_PKL_file_vanilla_tied_results_batch_LED34_animal_{animal}_FILTERED.pkl`
  - Filtered lapse: `led34_filter_files/vanila_lapse/vbmc_vanilla_tied_results_batch_LED34_animal_{animal}_lapses_truncate_1s_stim_filtered.pkl`
  - Unfiltered vanilla: `led34_filter_files/vanila/results_LED34_animal_{animal}.pkl`
  - Unfiltered lapse: `oct_9_10_vanila_lapse_model_fit_files/vbmc_vanilla_tied_results_batch_LED34_animal_{animal}_lapses_truncate_1s.pkl`
- Parameters displayed with 3 decimal places; ELBO differences shown on bar plots with value labels

**`compare_vanilla_norm_lapse_elbos.py`** (located in `../fit_animal_by_animal/`)
- **ELBO comparison notebook** for vanilla+lapse vs norm+lapse models (Oct 9-10 fits)
- Reads VBMC pkl files from `oct_9_10_vanila_lapse_model_fit_files/` and `oct_9_10_norm_lapse_model_fit_files/`
- Compares 18 common (batch, animal) pairs with original vanilla/norm model ELBOs from `results_{batch}_animal_{id}.pkl`
- Extracts convergence info: ELBO, stability flag, elbo_sd, n_iterations
- **Outputs**:
  - Formatted table showing all 8 metrics per animal (V+L stable, N+L stable, V+L ELBO, N+L ELBO, OG V ELBO, OG N ELBO)
  - Summary statistics: stability counts, mean/median/min/max ELBO improvements
  - `vanilla_norm_lapse_elbo_comparison.csv`: Full comparison table
  - `elbo_comparisons_bar_plots.png`: **3 bar plot panels** with y-axis limited to [-100, 100]:
    - Panel 1: Vanilla+Lapse ELBO - Original Vanilla ELBO
    - Panel 2: Vanilla+Lapse ELBO - Original Norm ELBO
    - Panel 3: Norm+Lapse ELBO - Original Norm ELBO
  - Green bars = positive ELBO difference (improvement), Red bars = negative (worse)
- **Key findings**: All 18/18 animals converged stably; median ELBO improvements ~+8 (vanilla+lapse) and ~+3 (norm+lapse), but mean improvements negative due to a few outliers with large ELBO drops (LED34_59: -1428, LED34_63: -752)

---

## Log-Likelihood Comparison Scripts

**`compare_vanilla_norm_lapse_loglike_v2.py`** (located in `../fit_animal_by_animal/`)
- Manually calculates log-likelihoods from VBMC parameters and data (instead of using pre-computed ELBO values) for vanilla+lapse and norm+lapse models
- Computes log-likelihoods using parallel processing for all valid trials with model parameters sampled from VBMC posteriors
- Includes parameter comparison figures showing means and 95% CIs for Vanilla vs Vanilla+Lapse and Norm vs Norm+Lapse models with lapse parameters displayed only for lapse models

**`compare_vanilla_norm_lapse_loglike_v2_div_by_N.py`** (located in `../fit_animal_by_animal/`)
- Extended version that normalizes all log-likelihood comparisons by dividing by N_trials for per-trial metrics
- Generates 4 bar plots comparing model pairs and scatter plot of lapse probability vs log-likelihood improvement over Norm model

**`compare_vanilla_norm_lapse_loglike_v2_div_by_N_with_aborts.py`** (located in `../fit_animal_by_animal/`)
- Includes BOTH valid and abort trials (abort_event==3, TotalFixTime>T_trunc) in log-likelihood calculation for all 4 models (Vanilla, Norm, Vanilla+Lapse, Norm+Lapse)
- Abort log-likelihood is identical across all models (depends only on V_A, theta_A, t_A_aff), ensuring fair apples-to-apples comparisons normalized by (N_valid + N_aborts)
- Generates `lapse_parameters_all_animals.pkl` with lapse parameters and per-trial log-likelihoods for all 4 models
- Produces 4 bar plot comparisons and scatter plots of lapse probability vs log-likelihood improvement

**`kl_distance_and_elbo.py`** (located in `../fit_each_condn/`)
- Computes KL divergence KL(q||p) between Gaussian and uniform distributions using numerical integration
- Primary use case: Understanding how prior bounds affect KL divergence in VBMC
- Example: q = N(0.9, 0.05) (Gaussian approximation of posterior), p = Uniform(0, width) (prior)
- Function `compute_kl_divergence_gaussian_uniform()` with parameters:
  - `q_mean`, `q_std`: Gaussian distribution parameters
  - `p_lower`, `p_upper`: Uniform distribution bounds
  - Returns KL(q||p) via numerical integration over q_mean ± 6*q_std
- Test cases included: Comparing KL divergence for different uniform prior widths
- **Key insight**: Wider uniform priors → lower KL divergence (less information needed to transform from q to p)
- Useful for understanding ELBO = log p(data) - KL(q||p) tradeoffs when adjusting prior bounds in VBMC fits

---

## Gamma Analysis by Lapse Groups

**`compare_gamma_by_lapse_prob_groups_for_paper.py`** (located in `../fit_each_condn/`)
- Two grouping strategies to compare average gamma from condition-by-condition fits across ABLs:
  1. **By vanilla lapse probability**: Low (<1.5%) vs High (≥1.5%) lapse groups
  2. **By log-likelihood comparison**: Animals where Vanilla+Lapse > Norm vs Vanilla+Lapse < Norm
- Generates two plots showing gamma curves for each group across all ABLs and ILDs

---

# reading pkl files
- READING_VBMC_PICKLE_FILES.md
- **READING_LAPSE_PARAMETERS_PKL.md** - Documentation for `lapse_parameters_all_animals.pkl` containing lapse parameters and per-trial log-likelihoods for all 4 models
 