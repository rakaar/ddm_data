# Proactive Process Model Documentation

## Overview

This document explains the proactive process model used for fitting abort data with LED ON/OFF trials, including theoretical functions, truncation, censoring, and likelihood calculations.

## Model Description

The proactive process model is a single-bound accumulator model where:

1. **Proactive accumulation starts at `t_aff`** (proactive afferent delay)
2. **Drift rate changes after LED onset** (for LED ON trials only)
3. **Response occurs when accumulator hits threshold `theta_A`**
4. **RT includes motor delay**

### Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `V_A_base` | Drift rate before LED onset | 1.8 |
| `V_A_post_LED` | Drift rate after LED onset | 2.4 |
| `theta_A` | Decision threshold | 1.5 |
| `t_aff` | Proactive afferent delay | 40 ms |
| `t_effect` | Time after LED when drift changes | 35 ms |
| `motor_delay` | Motor response delay | 50 ms |
| `T_trunc` | Left truncation time | 0.6 s |

### Time Definitions

- **`t`**: Time from fixation start (RT)
- **`t_stim`**: Stimulus onset time relative to fixation start
- **`t_LED`**: LED onset time relative to fixation start
- **`t_effect`**: Time after LED when drift changes (i.e., drift changes at `t_LED + t_effect`)
- **`T_trunc`**: Left truncation time (trials with RT ≤ T_trunc are excluded)

## Theoretical Functions

### 1. Base PA PDF: `d_A_RT(a, t)`

Standard first passage time PDF for a drift diffusion process with single absorbing boundary.

```python
d_A_RT(a, t) = (1 / sqrt(2*pi*t^3)) * exp(-(1 - a*t)^2 / (2*t))
```

**Parameters:**
- `a`: Scaled drift rate (V_A * theta_A)
- `t`: Time (adjusted for delays)

**Usage:** Used for LED OFF trials and for LED ON trials before drift change.

### 2. Post-LED PDF: `stupid_f_integral(v, vON, theta, t, tp)`

PDF for proactive process after drift rate change, computed via integral expression.

**Parameters:**
- `v`: Pre-LED drift rate (V_A_base)
- `vON`: Post-LED drift rate (V_A_post_LED)
- `theta`: Decision threshold (theta_A)
- `t`: Time since drift change
- `tp`: Time from start to drift change

**Usage:** Used for LED ON trials after drift change.

### 3. Combined LED ON PDF: `PA_with_LEDON_2(t, v, vON, a, tfix, tled, delta_A)`

Combines pre-LED and post-LED contributions.

```python
if (t + tfix) <= tled + 1e-6:
    # Before LED drift change
    return d_A_RT(v * a, (t - delta_A + tfix) / (a^2)) / (a^2)
else:
    # After LED drift change
    return stupid_f_integral(v, vON, a, t + tfix - tled, tled - delta_A + tfix)
```

**Parameters:**
- `t`: Time from fixation start (RT - motor_delay)
- `v`: Pre-LED drift (V_A_base)
- `vON`: Post-LED drift (V_A_post_LED)
- `a`: Threshold (theta_A)
- `tfix`: Fixation start time (0 for RT wrt stim)
- `tled`: LED onset time (t_LED)
- `delta_A`: Combined delay (t_aff + motor_delay)

## Adapted Function with Separate Parameters

### `PA_with_LEDON_2_adapted(t, v, vON, a, t_aff, motor_delay, tled, t_effect, T_trunc=None)`

This function separates `delta_A` into `t_aff` and `motor_delay`, and adds `t_effect` for delayed drift change.

**Key logic:**

```python
# Adjust time for motor delay
t_adj = t - motor_delay

if T_trunc is not None and t <= T_trunc:
    return 0  # Left truncation

if t_adj <= (tled + t_effect):
    # Before LED drift change
    pdf = d_A_RT(v * a, (t_adj - t_aff) / (a^2)) / (a^2)
else:
    # After LED drift change
    t_post_led = t_adj - tled - t_effect
    tp = tled + t_effect - t_aff
    
    if tp <= 0:
        # Edge case: drift change before proactive start
        pdf = d_A_RT(vON * a, (t_adj - t_aff) / (a^2)) / (a^2)
    else:
        pdf = stupid_f_integral(v, vON, a, t_post_led, tp)

# Apply truncation normalization
if T_trunc is not None:
    cdf_trunc = numerical_integration(PDF, 0, T_trunc)
    pdf = pdf / (1 - cdf_trunc)

return pdf
```

## Truncation

### Left Truncation

Trials with RT ≤ T_trunc are excluded from analysis. The PDF must be renormalized.

**Truncated PDF:**
```
f_truncated(t) = 0                          if t ≤ T_trunc
f_truncated(t) = f(t) / (1 - CDF(T_trunc))  if t > T_trunc
```

**Implementation:**
```python
if t <= T_trunc:
    return 0
else:
    cdf_trunc = trapz(PDF, 0, T_trunc)
    return PDF / (1 - cdf_trunc)
```

**Note:** No analytical CDF exists for `PA_with_LEDON_2`, so we compute CDF numerically using `trapz`.

## Censoring

### Right Censoring

Trials are censored after stimulus onset (t_stim). We need to compute the survival probability.

**Survival probability (probability RT > t_stim):**
```
S(t_stim) = 1 - CDF(t_stim)
```

**With truncation (conditional survival):**
```
S(t_stim | RT > T_trunc) = P(RT > t_stim | RT > T_trunc)
                         = (1 - CDF(t_stim)) / (1 - CDF(T_trunc))
```

**Edge case:** If `t_stim ≤ T_trunc`, then `S(t_stim | RT > T_trunc) = 1` (any trial surviving past T_trunc automatically survived past t_stim).

### Implementation

```python
# CDF at t_stim
cdf_t_stim = trapz(PDF, 0, t_stim)

# CDF at T_trunc
cdf_T_trunc = trapz(PDF, 0, T_trunc)

# Survival probability
if t_stim <= T_trunc:
    survival = 1.0
else:
    survival = (1 - cdf_t_stim) / (1 - cdf_T_trunc)
```

## Monte Carlo Averaging

Since `t_stim` and `t_LED` vary across trials, we average over sampled pairs.

**Procedure:**
1. Sample N_mc pairs of (t_stim, t_LED) from data
2. For each pair, compute theoretical PDF
3. Average PDFs across all samples

**Implementation:**
```python
def compute_theoretical_RT_distribution(V_A_base, V_A_post_LED, theta_A, 
                                        t_aff, t_effect, motor_delay,
                                        N_mc=1000, t_max=5.0, dt=0.01, 
                                        T_trunc=None):
    t_pts = np.arange(0, t_max, dt)
    pdf_samples = np.zeros((N_mc, len(t_pts)))
    
    for i in range(N_mc):
        # Sample (t_stim, t_LED) pair
        is_led_trial = np.random.random() < 1/3
        if is_led_trial:
            t_LED = np.random.choice(LED_times)
            t_stim = np.random.choice(stim_times)
        else:
            t_LED = None
            t_stim = np.random.choice(stim_times_off)
        
        # Compute PDF for this pair
        for j, t in enumerate(t_pts):
            if is_led_trial:
                pdf_samples[i, j] = PA_with_LEDON_2_adapted(
                    t, V_A_base, V_A_post_LED, theta_A,
                    t_aff, motor_delay, t_LED, t_effect, T_trunc)
            else:
                # LED OFF: use base drift
                pdf = d_A_RT(V_A_base * theta_A, 
                            (t - motor_delay - t_aff) / (theta_A**2)) / (theta_A**2)
                if T_trunc is not None:
                    # Apply truncation
                    cdf_trunc = trapz(PDF, 0, T_trunc)
                    pdf = pdf / (1 - cdf_trunc)
                pdf_samples[i, j] = pdf
    
    # Average across samples
    pdf_mean = np.mean(pdf_samples, axis=0)
    return t_pts, pdf_mean
```

## Verification

### 1. PDF Matching

Compare theoretical PDF (from Monte Carlo averaging) with simulated histogram.

**Procedure:**
1. Simulate N_sim trials with known parameters
2. Compute theoretical PDF via Monte Carlo
3. Plot both on same axes
4. They should match if likelihood function is correct

### 2. Censoring Verification

Compare fraction of trials after t_stim from simulation vs theory.

**Simulated fraction:**
```python
sim_after_t_stim = sum(1 for rt, t_stim in zip(sim_rts, sim_t_stims) 
                       if rt > t_stim and rt > T_trunc)
frac_sim = sim_after_t_stim / len(sim_rts)
```

**Theoretical fraction:**
```python
# Monte Carlo over (t_stim, t_LED) pairs
survival_samples = []
for _ in range(N_mc):
    t_stim = np.random.choice(stim_times)
    cdf_t_stim = trapz(PDF, 0, t_stim)
    cdf_T_trunc = trapz(PDF, 0, T_trunc)
    
    if t_stim <= T_trunc:
        survival = 1.0
    else:
        survival = (1 - cdf_t_stim) / (1 - cdf_T_trunc)
    
    survival_samples.append(survival)

frac_theory = np.mean(survival_samples)
```

**Verification:** `frac_sim` should match `frac_theory`.

## Likelihood Function for Fitting

For fitting with VBMC, the likelihood function should:

1. **Handle LED ON/OFF separately:**
   - LED ON: Use `PA_with_LEDON_2_adapted`
   - LED OFF: Use `d_A_RT`

2. **Apply truncation:**
   - Filter data: only trials with RT > T_trunc
   - Normalize PDF by `(1 - CDF(T_trunc))`

3. **Average over (t_stim, t_LED):**
   - For each trial, sample N_mc pairs of (t_stim, t_LED)
   - Compute PDF for each pair
   - Average PDFs

4. **Compute log-likelihood:**
   ```python
   log_likelihood = sum(log(PDF_i)) for all trials i
   ```

### Example Likelihood Function

```python
def log_likelihood(params, data):
    V_A_base, V_A_post_LED, theta_A, t_aff, t_effect, motor_delay = params
    T_trunc = 0.6
    
    # Filter data for truncation
    data_filtered = data[data['RT'] > T_trunc]
    
    log_lik = 0
    for _, trial in data_filtered.iterrows():
        t = trial['RT']
        t_stim = trial['t_stim']
        is_led_trial = trial['is_led_trial']
        
        # Monte Carlo averaging
        pdf_vals = []
        for _ in range(N_mc):
            if is_led_trial:
                t_LED = np.random.choice(LED_times)
                pdf = PA_with_LEDON_2_adapted(
                    t, V_A_base, V_A_post_LED, theta_A,
                    t_aff, motor_delay, t_LED, t_effect, T_trunc)
            else:
                pdf = d_A_RT(V_A_base * theta_A, 
                            (t - motor_delay - t_aff) / (theta_A**2)) / (theta_A**2)
                # Apply truncation
                cdf_trunc = trapz(PDF, 0, T_trunc)
                pdf = pdf / (1 - cdf_trunc)
            pdf_vals.append(pdf)
        
        pdf_mean = np.mean(pdf_vals)
        log_lik += np.log(pdf_mean)
    
    return log_lik
```

## Files

### Main Scripts

- **`simulate_and_fit_proactive_all_at_once.py`**: Simulates proactive process without truncation, compares theoretical vs simulated RT distributions
- **`simulate_and_fit_proactive_all_at_once_truncate_300.py`**: Same as above with left truncation (T_trunc=0.6s) and censoring verification

### Utility Functions

- **`../fit_each_condn/psiam_tied_dv_map_utils_with_PDFs.py`**: Contains theoretical functions:
  - `d_A_RT(a, t)`: Base PA PDF
  - `stupid_f_integral(v, vON, theta, t, tp)`: Post-LED PDF
  - `PA_with_LEDON_2(t, v, vON, a, tfix, tled, delta_A)`: Combined LED ON PDF

## References

- Drift diffusion model with single absorbing boundary
- First passage time distributions
- Left truncation and right censoring in survival analysis
- VBMC (Variational Bayesian Monte Carlo) for model fitting
