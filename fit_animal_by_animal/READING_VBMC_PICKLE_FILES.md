# Guide to Reading VBMC Pickle Files

This document provides a comprehensive guide for reading parameters, ELBO values, and log-likelihoods from VBMC (Variational Bayesian Monte Carlo) pickle files in this codebase.

---

## Table of Contents
1. [Reading from Dictionary-based Pickle Files (`results_*.pkl`)](#1-reading-from-dictionary-based-pickle-files)
2. [Reading from VP Object-based Pickle Files (Lapse Model Fits)](#2-reading-from-vp-object-based-pickle-files)
3. [Structure of `results_*.pkl` Files](#3-structure-of-results_pkl-files)
4. [Common Pitfalls and Solutions](#4-common-pitfalls-and-solutions)

---

## 1. Reading from Dictionary-based Pickle Files

### File Pattern
- **Vanilla/Norm fits**: `results_{batch}_animal_{animal_id}.pkl`
- **LED34 filtered**: `led34_filter_files/vanilla/results_LED34_animal_{id}_filtered.pkl`
- **LED34 norm filtered**: `led34_filter_files/norm/results_LED34_animal_{id}_NORM_filtered.pkl`

### Structure
These files contain nested dictionaries with keys like:
- `vbmc_vanilla_tied_results` (for vanilla model)
- `vbmc_norm_tied_results` (for norm model)

Each nested dictionary contains:
- **Parameters**: Mean values and sample arrays
- **ELBO**: Evidence Lower Bound value
- **Log-likelihood**: Model fit quality metric
- **Convergence info**: Stability flags, iterations, etc.

### Code Example: Reading Parameters, ELBO, and Log-Likelihood

```python
import pickle
import numpy as np

# Load the pickle file
pkl_path = 'results_LED7_animal_103.pkl'
with open(pkl_path, 'rb') as f:
    results = pickle.load(f)

# ===== VANILLA MODEL =====
vanilla_results = results['vbmc_vanilla_tied_results']

# Read mean parameters (direct values)
rate_lambda = vanilla_results['rate_lambda']
T_0 = vanilla_results['T_0']
theta_E = vanilla_results['theta_E']
w = vanilla_results['w']
t_E_aff = vanilla_results['t_E_aff']
del_go = vanilla_results['del_go']

print(f"Vanilla parameters:")
print(f"  rate_lambda: {rate_lambda:.4f}")
print(f"  T_0: {T_0:.4f}")
print(f"  theta_E: {theta_E:.4f}")
print(f"  w: {w:.4f}")

# Read ELBO (if available)
elbo = vanilla_results.get('elbo', None)
print(f"  ELBO: {elbo}")

# Read log-likelihood (if available)
loglike = vanilla_results.get('loglike', None)
print(f"  Log-likelihood: {loglike}")

# Read parameter samples (for uncertainty quantification)
if 'rate_lambda_samples' in vanilla_results:
    rate_lambda_samples = vanilla_results['rate_lambda_samples']
    print(f"  rate_lambda std: {np.std(rate_lambda_samples):.4f}")

# ===== NORM MODEL =====
norm_results = results['vbmc_norm_tied_results']

# Norm model has an additional parameter: rate_norm_l
rate_norm_l = norm_results['rate_norm_l']
print(f"\nNorm rate_norm_l: {rate_norm_l:.4f}")

# Read ELBO and log-likelihood
norm_elbo = norm_results.get('elbo', None)
norm_loglike = norm_results.get('loglike', None)
print(f"Norm ELBO: {norm_elbo}")
print(f"Norm Log-likelihood: {norm_loglike}")
```

### Reading Abort Parameters

Abort parameters are typically stored in the same `results_*.pkl` files:

```python
# Read abort parameters (used for computing truncated likelihoods)
abort_results = results.get('abort_tied_results', {})

V_A = abort_results['V_A']
theta_A = abort_results['theta_A']
t_A_aff = abort_results['t_A_aff']

print(f"Abort parameters:")
print(f"  V_A: {V_A:.4f}")
print(f"  theta_A: {theta_A:.4f}")
print(f"  t_A_aff: {t_A_aff:.4f}")
```

---

## 2. Reading from VP Object-based Pickle Files

### File Pattern
- **Vanilla+Lapse**: `oct_9_10_vanila_lapse_model_fit_files/vbmc_vanilla_tied_results_batch_{batch}_animal_{animal}_lapses_truncate_1s.pkl`
- **Norm+Lapse**: `oct_9_10_norm_lapse_model_fit_files/vbmc_norm_tied_results_batch_{batch}_animal_{animal}_lapses_truncate_1s_NORM.pkl`

### Structure
These files contain a **VBMC object** directly (not a dictionary). The object has:
- `iteration_history`: Dictionary with arrays tracking optimization progress
- `iteration_history['vp']`: Array of Variational Posterior (VP) objects
- Last VP object contains the final fitted distribution

### Code Example: Reading from VP Objects

```python
import pickle
import numpy as np

# Load the VBMC object
pkl_path = 'oct_9_10_vanila_lapse_model_fit_files/vbmc_vanilla_tied_results_batch_LED7_animal_103_lapses_truncate_1s.pkl'
with open(pkl_path, 'rb') as f:
    vbmc = pickle.load(f)

# ===== ACCESS ITERATION HISTORY =====
iter_hist = vbmc.iteration_history

# Read ELBO standard deviation (convergence metric)
elbo_sd_arr = iter_hist['elbo_sd']
final_elbo_sd = float(elbo_sd_arr[-1])
print(f"Final ELBO SD: {final_elbo_sd:.6f}")

# Read stability flag
stable_arr = iter_hist['stable']
is_stable = bool(stable_arr[-1])
print(f"Converged: {is_stable}")

# Read number of iterations
iter_arr = iter_hist['iter']
n_iterations = int(iter_arr[-1])
print(f"Number of iterations: {n_iterations}")

# ===== ACCESS VP OBJECT =====
vp_arr = iter_hist['vp']
last_vp = vp_arr[-1]  # Get final VP object

# Sample parameters from VP
# IMPORTANT: sample() returns a TUPLE: (samples, log_weights)
vp_samples, log_weights = last_vp.sample(int(1e6))  # Sample 1 million points

# For vanilla+lapse model (8 parameters):
# [rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, lapse_prob_right]
rate_lambda = np.mean(vp_samples[:, 0])
T_0 = np.mean(vp_samples[:, 1])
theta_E = np.mean(vp_samples[:, 2])
w = np.mean(vp_samples[:, 3])
t_E_aff = np.mean(vp_samples[:, 4])
del_go = np.mean(vp_samples[:, 5])
lapse_prob = np.mean(vp_samples[:, 6])
lapse_prob_right = np.mean(vp_samples[:, 7])

print(f"\nVanilla+Lapse parameters (from VP):")
print(f"  rate_lambda: {rate_lambda:.4f}")
print(f"  lapse_prob: {lapse_prob:.4f}")
print(f"  lapse_prob_right: {lapse_prob_right:.4f}")

# For norm+lapse model (9 parameters):
# [rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right]
# rate_norm_l = np.mean(vp_samples[:, 6])
# lapse_prob = np.mean(vp_samples[:, 7])
# lapse_prob_right = np.mean(vp_samples[:, 8])

# ===== CALCULATE PARAMETER UNCERTAINTIES =====
rate_lambda_std = np.std(vp_samples[:, 0])
lapse_prob_std = np.std(vp_samples[:, 6])
print(f"\nParameter uncertainties:")
print(f"  rate_lambda std: {rate_lambda_std:.4f}")
print(f"  lapse_prob std: {lapse_prob_std:.4f}")

# ===== COMPUTE LOG-LIKELIHOOD MANUALLY =====
# VP objects don't directly store log-likelihood
# You need to compute it from parameters and data
# See compare_vanilla_norm_lapse_loglike_v2.py for implementation
```

### Reading ELBO from VP Statistics

```python
# Alternative: Try reading from VP stats (may not always be available)
if hasattr(last_vp, 'stats') and 'e_log_joint' in last_vp.stats:
    e_log_joint = float(last_vp.stats['e_log_joint'])
    print(f"Expected log joint (from VP stats): {e_log_joint:.2f}")
else:
    print("Log-likelihood not directly available in VP stats")
    print("Must be computed manually from parameters and data")
```

---

## 3. Structure of `results_*.pkl` Files

### Complete Structure Diagram

```
results_{batch}_animal_{animal_id}.pkl
│
├── vbmc_vanilla_tied_results/           # Vanilla model fit
│   ├── rate_lambda: float               # Mean evidence accumulation rate
│   ├── rate_lambda_samples: ndarray     # VBMC samples for rate_lambda
│   ├── T_0: float                       # Mean non-decision time (s)
│   ├── T_0_samples: ndarray             # VBMC samples for T_0
│   ├── theta_E: float                   # Mean decision threshold (half-width)
│   ├── theta_E_samples: ndarray         # VBMC samples for theta_E
│   ├── w: float                         # Mean bias parameter (0.5 = unbiased)
│   ├── w_samples: ndarray               # VBMC samples for w
│   ├── t_E_aff: float                   # Mean sensory encoding time (s)
│   ├── t_E_aff_samples: ndarray         # VBMC samples for t_E_aff
│   ├── del_go: float                    # Mean motor execution time (s)
│   ├── del_go_samples: ndarray          # VBMC samples for del_go
│   ├── elbo: float                      # Evidence Lower Bound (may be missing)
│   ├── loglike: float                   # Log-likelihood (may be missing)
│   └── ... (other fit metadata)
│
├── vbmc_norm_tied_results/              # Norm model fit
│   ├── rate_lambda: float               # Mean baseline rate
│   ├── rate_lambda_samples: ndarray
│   ├── T_0: float
│   ├── T_0_samples: ndarray
│   ├── theta_E: float
│   ├── theta_E_samples: ndarray
│   ├── w: float
│   ├── w_samples: ndarray
│   ├── t_E_aff: float
│   ├── t_E_aff_samples: ndarray
│   ├── del_go: float
│   ├── del_go_samples: ndarray
│   ├── rate_norm_l: float               # ADDITIONAL: Normalization parameter
│   ├── rate_norm_l_samples: ndarray
│   ├── elbo: float
│   ├── loglike: float
│   └── ...
│
└── abort_tied_results/                  # Abort behavior parameters
    ├── V_A: float                       # Abort drift rate
    ├── V_A_samples: ndarray
    ├── theta_A: float                   # Abort threshold
    ├── theta_A_samples: ndarray
    ├── t_A_aff: float                   # Abort encoding time
    ├── t_A_aff_samples: ndarray
    └── ...
```

### Parameter Descriptions

| Parameter | Model | Description | Units |
|-----------|-------|-------------|-------|
| `rate_lambda` | Both | Evidence accumulation rate (baseline) | Hz or 1/s |
| `T_0` | Both | Non-decision time (motor + sensory delays) | seconds |
| `theta_E` | Both | Decision threshold (half-width) | evidence units |
| `w` | Both | Bias parameter (0.5 = unbiased, >0.5 = right bias) | proportion |
| `t_E_aff` | Both | Sensory encoding time | seconds |
| `del_go` | Both | Motor execution time | seconds |
| `rate_norm_l` | Norm only | Normalization parameter for rate | dimensionless |
| `lapse_prob` | Lapse models | Overall probability of lapse trial | probability [0,1] |
| `lapse_prob_right` | Lapse models | Probability of choosing right given lapse | probability [0,1] |

---

## 4. Common Pitfalls and Solutions

### ❌ Pitfall 1: VP sample() returns tuple, not array

```python
# WRONG:
vp_samples = last_vp.sample(1000)
rate_lambda = np.mean(vp_samples[:, 0])  # TypeError: tuple indices must be integers

# CORRECT:
vp_samples, log_weights = last_vp.sample(1000)  # Unpack tuple
rate_lambda = np.mean(vp_samples[:, 0])  # Now works!
```

### ❌ Pitfall 2: Log-likelihood not in VP objects

```python
# WRONG:
loglike = last_vp.loglike  # AttributeError

# CORRECT:
# Must compute manually from parameters and data
# See calculate_loglike_from_vbmc() in compare_vanilla_norm_lapse_loglike_v2.py
```

### ❌ Pitfall 3: Different parameter orders for vanilla vs norm

```python
# Vanilla+Lapse (8 params):
# [rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, lapse_prob_right]
lapse_prob = np.mean(vp_samples[:, 6])  # Index 6

# Norm+Lapse (9 params):
# [rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right]
lapse_prob = np.mean(vp_samples[:, 7])  # Index 7 (not 6!)
```

### ❌ Pitfall 4: ELBO/loglike may be missing in old files

```python
# WRONG:
loglike = vanilla_results['loglike']  # KeyError if not present

# CORRECT:
loglike = vanilla_results.get('loglike', None)  # Returns None if missing
if loglike is None:
    print("Log-likelihood not available, must compute manually")
```

### ❌ Pitfall 5: LED34 batch has different file structure

```python
# For LED34 batch, files are in subdirectories:
# Vanilla: led34_filter_files/vanilla/results_LED34_animal_{id}_filtered.pkl
# Norm: led34_filter_files/norm/results_LED34_animal_{id}_NORM_filtered.pkl

def get_results_path(batch, animal_id, model_type='vanilla'):
    if batch == 'LED34_even':
        if model_type == 'vanilla':
            return f'led34_filter_files/vanilla/results_LED34_animal_{animal_id}_filtered.pkl'
        else:
            return f'led34_filter_files/norm/results_LED34_animal_{animal_id}_NORM_filtered.pkl'
    else:
        return f'results_{batch}_animal_{animal_id}.pkl'
```

---

## 5. Complete Working Example

Here's a complete script that safely reads all available data:

```python
import pickle
import numpy as np
import os

def read_vbmc_results(pkl_path, model_type='vanilla'):
    """
    Safely read VBMC results from either dict-based or VP-based pickle files.
    
    Args:
        pkl_path: Path to pickle file
        model_type: 'vanilla' or 'norm'
    
    Returns:
        dict with keys: params, elbo, loglike, stable, n_iterations
    """
    result = {
        'params': {},
        'elbo': None,
        'loglike': None,
        'stable': None,
        'n_iterations': None
    }
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check if it's a dict-based file (results_*.pkl)
    if isinstance(data, dict):
        key = f'vbmc_{model_type}_tied_results'
        if key in data:
            model_data = data[key]
            
            # Read parameters
            param_names = ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go']
            if model_type == 'norm':
                param_names.append('rate_norm_l')
            
            for param in param_names:
                result['params'][param] = model_data.get(param, None)
            
            # Read ELBO and loglike
            result['elbo'] = model_data.get('elbo', None)
            result['loglike'] = model_data.get('loglike', None)
    
    # Check if it's a VP-based file (lapse model fits)
    elif hasattr(data, 'iteration_history'):
        iter_hist = data.iteration_history
        
        # Read convergence info
        if 'elbo_sd' in iter_hist:
            result['elbo'] = float(iter_hist['elbo_sd'][-1])
        if 'stable' in iter_hist:
            result['stable'] = bool(iter_hist['stable'][-1])
        if 'iter' in iter_hist:
            result['n_iterations'] = int(iter_hist['iter'][-1])
        
        # Sample parameters from VP
        if 'vp' in iter_hist:
            vp_arr = iter_hist['vp']
            last_vp = vp_arr[-1]
            vp_samples, _ = last_vp.sample(int(1e6))
            
            # Determine parameter order based on shape
            n_params = vp_samples.shape[1]
            
            if n_params == 8:  # Vanilla+Lapse
                param_names = ['rate_lambda', 'T_0', 'theta_E', 'w', 
                               't_E_aff', 'del_go', 'lapse_prob', 'lapse_prob_right']
            elif n_params == 9:  # Norm+Lapse
                param_names = ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 
                               'del_go', 'rate_norm_l', 'lapse_prob', 'lapse_prob_right']
            else:
                param_names = [f'param_{i}' for i in range(n_params)]
            
            for i, param in enumerate(param_names):
                result['params'][param] = float(np.mean(vp_samples[:, i]))
    
    return result

# Example usage
pkl_path = 'results_LED7_animal_103.pkl'
if os.path.exists(pkl_path):
    vanilla_data = read_vbmc_results(pkl_path, model_type='vanilla')
    print("Vanilla parameters:", vanilla_data['params'])
    print("Vanilla loglike:", vanilla_data['loglike'])
```

---

## 6. Quick Reference

### Read from results_*.pkl (dict-based)
```python
with open(pkl_path, 'rb') as f:
    results = pickle.load(f)
vanilla = results['vbmc_vanilla_tied_results']
rate_lambda = vanilla['rate_lambda']
loglike = vanilla.get('loglike', None)
```

### Read from lapse fit (VP-based)
```python
with open(pkl_path, 'rb') as f:
    vbmc = pickle.load(f)
last_vp = vbmc.iteration_history['vp'][-1]
vp_samples, _ = last_vp.sample(int(1e6))
rate_lambda = np.mean(vp_samples[:, 0])
elbo_sd = float(vbmc.iteration_history['elbo_sd'][-1])
```

---

**Last Updated**: October 2025  
**Author**: Based on codebase analysis of animal sound localization DDM fits
