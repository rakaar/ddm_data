# %%
"""
Compare ELBO between:
1. Vanilla model (from animal_wise_vanilla_fit.py)
2. Lapse model with stimulus filtering (from lapses_fit_single_animal.py)
"""

import pickle
import os
import sys

# Try to import pyvbmc if available
try:
    from pyvbmc import VBMC
    PYVBMC_AVAILABLE = True
except ImportError:
    PYVBMC_AVAILABLE = False
    print("WARNING: pyvbmc not available, will try alternative unpickling methods")

def extract_convergence_info_from_vbmc(pkl_path):
    """
    Extract convergence information from a VBMC pickle file.
    
    Returns:
        dict with keys: elbo, elbo_sd, stable, n_iterations
    """
    try:
        if PYVBMC_AVAILABLE:
            # Use standard pickle if pyvbmc is available
            with open(pkl_path, 'rb') as f:
                vbmc = pickle.load(f)
        else:
            # Try dill if available
            try:
                import dill
                with open(pkl_path, 'rb') as f:
                    vbmc = dill.load(f)
            except ImportError:
                # Last resort: try standard pickle anyway
                with open(pkl_path, 'rb') as f:
                    vbmc = pickle.load(f)
        
        # Extract from iteration_history
        if hasattr(vbmc, 'iteration_history'):
            iter_hist = vbmc.iteration_history
            
            result = {}
            
            if 'elbo' in iter_hist:
                elbo_arr = iter_hist['elbo']
                result['elbo'] = float(elbo_arr[-1])
            else:
                result['elbo'] = None
            
            if 'elbo_sd' in iter_hist:
                elbo_sd_arr = iter_hist['elbo_sd']
                result['elbo_sd'] = float(elbo_sd_arr[-1])
            else:
                result['elbo_sd'] = None
            
            if 'stable' in iter_hist:
                stable_arr = iter_hist['stable']
                result['stable'] = bool(stable_arr[-1])
            else:
                result['stable'] = None
            
            if 'iter' in iter_hist:
                iter_arr = iter_hist['iter']
                result['n_iterations'] = int(iter_arr[-1])
            else:
                result['n_iterations'] = len(iter_hist)
            
            return result
        else:
            return {'elbo': None, 'elbo_sd': None, 'stable': None, 'n_iterations': None}
    
    except Exception as e:
        print(f"Error reading {pkl_path}: {e}")
        import traceback
        traceback.print_exc()
        return {'elbo': None, 'elbo_sd': None, 'stable': None, 'n_iterations': None, 'error': str(e)}


def get_vanilla_elbo_from_results(pkl_path):
    """
    Load original vanilla ELBO from results pickle (animal_wise_vanilla_fit.py output).
    
    Returns:
        dict with keys: elbo, elbo_sd
    """
    result = {'elbo': None, 'elbo_sd': None, 'stable': None}
    
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return result
    
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract vanilla ELBO - it's stored directly in the results dict
        if 'vbmc_vanilla_tied_results' in results:
            vanilla_results = results['vbmc_vanilla_tied_results']
            result['elbo'] = vanilla_results.get('elbo', None)
            result['elbo_sd'] = vanilla_results.get('elbo_sd', None)
            # Note: stable info might not be in the results dictprint("- Vanilla model: Fit on FILTERED stimuli (ABL=[20,40,60], ILD=[-16,-8,-4,-2,-1,1,2,4,8,16])")
        
        return result
    
    except Exception as e:
        print(f"Error reading {pkl_path}: {e}")
        return result


# Configuration
batch = 'LED34'
animal_id = 45

# File paths
lapse_dir = 'test_led34_low_trials'
lapse_pkl = os.path.join(lapse_dir, f'vbmc_vanilla_tied_results_batch_{batch}_animal_{animal_id}_lapses_truncate_1s_stim_filtered.pkl')

vanilla_pkl = f'results_{batch}_animal_{animal_id}.pkl'

print("="*80)
print(f"ELBO Comparison: Batch {batch}, Animal {animal_id}")
print("="*80)

# Load lapse model results (from VBMC pickle)
print(f"\nLoading lapse model (stimulus filtered) from:")
print(f"  {lapse_pkl}")
lapse_info = extract_convergence_info_from_vbmc(lapse_pkl)

# Load vanilla model results (from results pickle)
print(f"\nLoading vanilla model (original) from:")
print(f"  {vanilla_pkl}")
vanilla_info = get_vanilla_elbo_from_results(vanilla_pkl)

# Display results
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nVanilla Model (Original - No Filtering):")
print(f"  ELBO:        {vanilla_info['elbo']:.2f}" if vanilla_info['elbo'] is not None else "  ELBO:        N/A")
print(f"  ELBO SD:     {vanilla_info['elbo_sd']:.2f}" if vanilla_info['elbo_sd'] is not None else "  ELBO SD:     N/A")
print(f"  Stable:      {vanilla_info['stable']}" if vanilla_info['stable'] is not None else "  Stable:      N/A")

print(f"\nLapse Model (With Stimulus Filtering):")
if lapse_info['elbo'] is not None:
    print(f"  ELBO:        {lapse_info['elbo']:.2f}")
    print(f"  ELBO SD:     {lapse_info['elbo_sd']:.2f}" if lapse_info['elbo_sd'] is not None else "  ELBO SD:     N/A")
    print(f"  Stable:      {lapse_info['stable']}")
    print(f"  Iterations:  {lapse_info['n_iterations']}")
else:
    print(f"  ELBO:        N/A (Cannot load without pyvbmc module)")
    print(f"  NOTE: Install pyvbmc to extract ELBO from VBMC pickle files")

# Calculate difference
if vanilla_info['elbo'] is not None and lapse_info['elbo'] is not None:
    elbo_diff = lapse_info['elbo'] - vanilla_info['elbo']
    print(f"\n" + "-"*80)
    print(f"ELBO Improvement (Lapse - Vanilla): {elbo_diff:+.2f}")
    if elbo_diff > 0:
        print(f"  → Lapse model is BETTER (higher ELBO)")
    else:
        print(f"  → Vanilla model is BETTER (higher ELBO)")
    print("-"*80)
elif vanilla_info['elbo'] is not None:
    print(f"\n" + "-"*80)
    print(f"Cannot compare ELBOs without lapse model ELBO")
    print(f"To fix: Install pyvbmc or run this script in the environment where the fit was performed")
    print("-"*80)

print("\n" + "="*80)
print("NOTES:")
print("- Vanilla model: Fit on FILTERED stimuli (ABL=[20,40,60], ILD=[-16,-8,-4,-2,-1,1,2,4,8,16])")
print("- Lapse model: Fit on SAME FILTERED stimuli")
print("- These models ARE directly comparable - both fit on identical data!")
print("- The ELBO difference shows whether adding lapse parameters improves model fit")
print("- Parameter comparison available in: test_led34_low_trials/param_comparison_batch_LED34_animal_45_vanilla_lapse_stim_filtered.txt")
print("="*80)
