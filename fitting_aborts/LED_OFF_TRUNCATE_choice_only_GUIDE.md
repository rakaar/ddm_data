# Diagnostics Pipeline Guide

## Key Files

| File | Purpose |
|------|---------|
| `fit_only_LED_off_with_LED_ON_fits_truncate_NOT_censor_ABL_delay_no_choice.py` | **Fitting script** — runs VBMC to produce pkl results |
| `fit_only_LED_off_with_LED_ON_fit_diagnostics_truncate_not_censor_ABL_delay_no_choice_FAST.py` | **Diagnostics script** (fast, vectorized) — loads pkl, plots theory vs data |
| `fit_only_LED_off_with_LED_ON_fit_diagnostics_truncate_not_censor_ABL_delay_no_choice.py` | Original diagnostics (slow scalar version, kept for reference) |

## Model

Proactive + lapse + reactive DDM, **choice-collapsed** (no-choice), with ABL-specific evidence-afferent delays (`t_E_aff_20`, `t_E_aff_40`, `t_E_aff_60`). Normalization is ON (`is_norm=True`), time-varying is OFF (`is_time_vary=False`).

**9 fitted parameters** (via VBMC): `rate_lambda`, `T_0`, `theta_E`, `w`, `t_E_aff_20`, `t_E_aff_40`, `t_E_aff_60`, `del_go`, `rate_norm_l`.

**6 loaded proactive parameters** (fixed from a prior fit): `V_A`, `theta_A`, `del_a_minus_del_LED`, `del_m_plus_del_LED`, `lapse_prob`, `beta_lapse`.

## The 4 Fit Cases

Two flags control which pkl file is selected:

1. **`truncate_rt_wrt_stim_s`** — RT truncation threshold: `0.115` (115 ms) or `0.130` (130 ms)
2. **`fix_trial_count_by_abl`** — Whether to subsample trials per ABL: `True` (fixN) or `False` (allvalid)

This gives 4 combinations, each with its own pkl:

| truncation | fixN | run_tag | pkl suffix |
|------------|------|---------|------------|
| 115 ms | OFF | `trunc115ms_allvalid` | `..._trunc115ms_allvalid.pkl` |
| 115 ms | ON  | `trunc115ms_fixN_20-1300_40-2300_60-3400` | `..._trunc115ms_fixN_20-1300_40-2300_60-3400.pkl` |
| 130 ms | OFF | `trunc130ms_allvalid` | `..._trunc130ms_allvalid.pkl` |
| 130 ms | ON  | `trunc130ms_fixN_20-1300_40-2300_60-3400` | `..._trunc130ms_fixN_20-1300_40-2300_60-3400.pkl` |

Pkl files live in: `fitting_aborts/norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay_no_choice/`

When `fixN=ON`, the **fitting script** subsamples per-ABL trial counts (`{20: 1300, 40: 2300, 60: 3400}`) before running VBMC. The **diagnostics script** always plots ALL available data trials (not the subsampled set), but uses the parameters from the corresponding fixN pkl.

## How to Switch Cases in Diagnostics

Edit lines ~75-78 of the FAST diagnostics file:

```python
truncate_rt_wrt_stim_s = 0.115   # or 0.130
fix_trial_count_by_abl = True    # or False
fixed_trial_counts_by_abl = {20: 1300, 40: 2300, 60: 3400}
```

The script auto-builds the `run_tag` and loads the matching pkl. If the pkl doesn't exist, it raises `FileNotFoundError`.

## Data Normalization

- `normalize_per_abl = False` — data histograms use `counts / (n_total_abl × bin_width)` so the area under each ABL's histogram reflects that ABL's proportion of total trials (not normalized to 1).
- Theory curves are plotted un-normalized (raw density), matching this scale.

## Vectorization (FAST vs original)

The FAST file replaced a scalar for-loop (`4001 time points × N_mc samples × 3 ABLs`) with vectorized calls to `_vec` functions from `time_vary_norm_utils.py`. Speedup: ~260x. Results match to ~1e-13.

Key vectorized functions (from `fit_animal_by_animal/time_vary_norm_utils.py`):
- `CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn_vec`
- `rho_E_minus_small_t_NORM_rate_norm_time_varying_fn_vec`
- `rho_A_t_VEC_fn`

## Running

```bash
.venv/bin/python fitting_aborts/fit_only_LED_off_with_LED_ON_fit_diagnostics_truncate_not_censor_ABL_delay_no_choice_FAST.py
```

Output plots go to: `fitting_aborts/norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay_no_choice/diagnostics/`
