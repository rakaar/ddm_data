# show_cond_fit_goodness_cond_vs_data.py - Handover Summary

## Purpose
Validates condition-by-condition gamma/omega fits by comparing **theoretical RT quantiles** against **empirical RT quantiles** at |ILD| = 16 dB for ABL = 20, 40, 60 dB.

## Key Finding
**V2 works correctly.** An earlier V1 approach using PDF integration was incorrect and has been deleted.

---

## Script Structure (4 cells)

### Cell 1: Load Empirical Quantile Data
- Loads `fig1_quantiles_plot_data.pkl` containing pre-computed empirical RT quantiles
- Extracts mean and SEM for |ILD| = 16 across ABL = 20, 40, 60

### Cell 2: Load Parameters for All Animals
- Loads **gamma, omega** from condition-fit pickle files:
  - Path: `fit_each_condn/each_animal_cond_fit_gama_omega_pkl_files/`
  - File: `vbmc_cond_by_cond_{batch}_{animal}_{ABL}_ILD_16_FIX_t_E_w_del_go_same_as_parametric.pkl`
- Loads **w, t_E_aff, del_go** (averaged from vanilla + norm tied results) and **abort params** (V_A, theta_A, t_A_aff) from:
  - Path: `fit_animal_by_animal/results_{batch}_animal_{id}.pkl`

### Cell 3: Compute Theoretical CDF (V2 Method)
**Key formula:**
```python
CDF = c_A + c_E - c_A * c_E
```
Where:
- `c_A = cum_A_t_fn(t_wrt_fix - t_A_aff, V_A, theta_A)` — abort CDF
- `c_E = CDF_E_with_w(t_wrt_stim - t_E_aff, gamma, omega, +1, w) + CDF_E_with_w(..., -1, w)` — evidence CDF (both bounds)

**Time handling:**
- `t_pts_wrt_stim` = time relative to stimulus onset
- `t_pts_wrt_fix = t_pts_wrt_stim + t_stim` — time relative to fixation
- Sample 1000 `t_stim` values from each animal's `intended_fix` distribution
- Average CDF across samples

**Normalization (important!):**
```python
cdf_normalized = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
```
This removes pre-stimulus abort probability so CDF starts at 0.

### Cell 4: Invert CDF → Quantiles & Plot
- Inverts each animal's CDF to get RT at 50 quantile levels (9% to 91%)
- Averages quantiles across animals with SEM
- Overlays theoretical curves on empirical data points

---

## Key Functions Used (from `led_off_gamma_omega_pdf_utils.py`)
- `cum_A_t_fn(t, V_A, theta_A)` — cumulative abort probability
- `CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t, gamma, omega, bound, w, K_max)` — evidence CDF with starting point `w`

---

## Output Files
- `cond_fit_theoretical_cdf_ild16_v2.png` — CDF plot
- `cond_fit_quantiles_theory_vs_data_ild16_v2.png` — Quantile comparison plot

---

## Why V1 Was Wrong
V1 used `up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn` to compute PDFs, then integrated to get CDF. This method produced incorrect results. The simpler direct CDF formula in V2 is correct.

---

## Parameters Summary
| Parameter | Source |
|-----------|--------|
| gamma, omega | Condition-by-condition VBMC fit (per ABL, ILD) |
| w, t_E_aff, del_go | Average of vanilla + norm tied results |
| V_A, theta_A, t_A_aff | Abort fit results |
| t_stim | Sampled from animal's `intended_fix` distribution |
