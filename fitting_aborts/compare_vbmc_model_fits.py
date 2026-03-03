"""
Compare VBMC model fits between drift jump and bound drop models
================================================================

This script loads the VBMC results from two different model fits:
1. Drift jump model (proactive process with drift jumps)
2. Bound drop model (proactive process with linear bound decrease)

It extracts and compares the ELBO and log-likelihood values from both models.

SCRIPT STRUCTURE (cell-by-cell):
--------------------------------
1. **Imports** - pickle, numpy, pandas
2. **Load Model Results** - Load pickle files from both models
3. **Extract Metrics** - Extract ELBO and log-likelihood from each model
4. **Compare Results** - Display side-by-side comparison
5. **Diagnostic Info** - Show structure of VBMC objects for debugging

"""

# %%
import pickle
import numpy as np
import pandas as pd
import os

# %%
# =============================================================================
# DEFINE FILE PATHS
# =============================================================================
# File paths for the two models (when ANIMAL_ID = None, i.e., all animals aggregated)
DRIFT_JUMP_MODEL_PATH = "vbmc_real_all_animals_fit_NO_TRUNC_with_lapse.pkl"
BOUND_DROP_MODEL_PATH = "vbmc_real_all_animals_fit_bound_drop_no_saturate_PYDDM.pkl"

# Check if files exist
print("Checking file existence:")
for path, name in [(DRIFT_JUMP_MODEL_PATH, "Drift Jump"), 
                   (BOUND_DROP_MODEL_PATH, "Bound Drop")]:
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"  {name}: {path} - {'EXISTS' if exists else 'MISSING'} ({size:,} bytes)")

# %%
# =============================================================================
# LOAD MODEL RESULTS
# =============================================================================
def load_vbmc_results(file_path):
    """Load VBMC results from pickle file"""
    print(f"\nLoading: {file_path}")
    try:
        with open(file_path, "rb") as f:
            vp = pickle.load(f)
        print(f"  Successfully loaded object of type: {type(vp)}")
        return vp
    except Exception as e:
        print(f"  ERROR loading file: {e}")
        return None

# Load both models
drift_jump_vp = load_vbmc_results(DRIFT_JUMP_MODEL_PATH)
bound_drop_vp = load_vbmc_results(BOUND_DROP_MODEL_PATH)

# %%
# =============================================================================
# EXTRACT METRICS FROM VBMC OBJECTS
# =============================================================================
def extract_vbmc_metrics(vp, model_name):
    """Extract ELBO and other metrics from VBMC variational posterior"""
    print(f"\n=== {model_name} Model Metrics ===")
    
    if vp is None:
        print("  ERROR: VP object is None")
        return {}
    
    # Print basic info about the VP object
    print(f"  VP Type: {type(vp)}")
    
    metrics = {}
    
    # Method 1: Extract ELBO from stats (this worked)
    if hasattr(vp, 'stats') and isinstance(vp.stats, dict):
        if 'elbo' in vp.stats:
            metrics["elbo_final"] = float(vp.stats['elbo'])
            print(f"  ✓ ELBO: {metrics['elbo_final']:.6f}")
        
        if 'e_log_joint' in vp.stats:
            metrics["expected_log_joint"] = float(vp.stats['e_log_joint'])
            print(f"  ✓ Expected Log Joint: {metrics['expected_log_joint']:.6f}")
        
        if 'entropy' in vp.stats:
            metrics["entropy"] = float(vp.stats['entropy'])
            print(f"  ✓ Entropy: {metrics['entropy']:.6f}")
    
    # Method 2: Get parameter moments (this worked)
    if hasattr(vp, 'moments'):
        try:
            moments = vp.moments()
            metrics["parameter_moments"] = moments.flatten()
            print(f"  ✓ Parameter Moments: {metrics['parameter_moments']}")
        except Exception as e:
            print(f"  Error getting parameter moments: {e}")
    
    # Method 3: Get parameter mode (this worked)
    if hasattr(vp, 'mode'):
        try:
            mode_params = vp.mode()
            metrics["parameter_mode"] = mode_params
            print(f"  ✓ Parameter Mode: {metrics['parameter_mode']}")
        except Exception as e:
            print(f"  Error getting parameter mode: {e}")
    
    # Method 4: Get sample info (this worked)
    if hasattr(vp, 'sample'):
        try:
            samples = vp.sample(100)[0]  # Get 100 samples
            metrics["sample_shape"] = samples.shape
            print(f"  ✓ Sample Shape: {samples.shape}")
        except Exception as e:
            print(f"  Error getting samples: {e}")
    
    return metrics

# Extract metrics from both models
drift_jump_metrics = extract_vbmc_metrics(drift_jump_vp, "Drift Jump")
bound_drop_metrics = extract_vbmc_metrics(bound_drop_vp, "Bound Drop")

# %%
# =============================================================================
# COMPARE RESULTS SIDE BY SIDE
# =============================================================================
def compare_models(drift_metrics, bound_metrics):
    """Create side-by-side comparison of model metrics"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # ELBO comparison
    if "elbo_final" in drift_metrics and "elbo_final" in bound_metrics:
        drift_elbo = drift_metrics["elbo_final"]
        bound_elbo = bound_metrics["elbo_final"]
        diff = drift_elbo - bound_elbo
        
        print(f"\n📊 ELBO COMPARISON:")
        print(f"  Drift Jump Model:  {drift_elbo:.6f}")
        print(f"  Bound Drop Model:  {bound_elbo:.6f}")
        print(f"  Difference:        {diff:+.6f}")
        print(f"  Better Model:       {'Drift Jump' if diff > 0 else 'Bound Drop'}")
    
    # Expected log joint comparison
    if "expected_log_joint" in drift_metrics and "expected_log_joint" in bound_metrics:
        drift_elj = drift_metrics["expected_log_joint"]
        bound_elj = bound_metrics["expected_log_joint"]
        diff = drift_elj - bound_elj
        
        print(f"\n📊 EXPECTED LOG JOINT COMPARISON:")
        print(f"  Drift Jump Model:  {drift_elj:.6f}")
        print(f"  Bound Drop Model:  {bound_elj:.6f}")
        print(f"  Difference:        {diff:+.6f}")
        print(f"  Better Model:       {'Drift Jump' if diff > 0 else 'Bound Drop'}")
    
    # Entropy comparison
    if "entropy" in drift_metrics and "entropy" in bound_metrics:
        drift_ent = drift_metrics["entropy"]
        bound_ent = bound_metrics["entropy"]
        diff = drift_ent - bound_ent
        
        print(f"\n📊 ENTROPY COMPARISON:")
        print(f"  Drift Jump Model:  {drift_ent:.6f}")
        print(f"  Bound Drop Model:  {bound_ent:.6f}")
        print(f"  Difference:        {diff:+.6f}")
        print(f"  Interpretation:    {'Higher uncertainty' if diff > 0 else 'Lower uncertainty'}")
    
    # Parameter comparison
    drift_params = drift_metrics.get("parameter_moments", np.array([]))
    bound_params = bound_metrics.get("parameter_moments", np.array([]))
    
    if len(drift_params) > 0 and len(bound_params) > 0:
        print(f"\n📊 PARAMETER COMPARISON:")
        print(f"  Number of Parameters: {len(drift_params)}")
        print(f"  Drift Jump Parameters: [{', '.join([f'{p:.6f}' for p in drift_params])}]")
        print(f"  Bound Drop Parameters: [{', '.join([f'{p:.6f}' for p in bound_params])}]")
        
        if len(drift_params) == len(bound_params):
            print(f"\n  Parameter-wise Differences:")
            for i, (d, b) in enumerate(zip(drift_params, bound_params)):
                diff = d - b
                print(f"    Param {i+1}: {diff:+.6f}")
    
    # Sample info
    drift_samples = drift_metrics.get("sample_shape", (0,))
    bound_samples = bound_metrics.get("sample_shape", (0,))
    
    if len(drift_samples) > 1 and len(bound_samples) > 1:
        print(f"\n📊 SAMPLE INFO:")
        print(f"  Drift Jump: {drift_samples[1]} params × {drift_samples[0]} samples")
        print(f"  Bound Drop: {bound_samples[1]} params × {bound_samples[0]} samples")
    
    # Summary interpretation
    print(f"\n🎯 SUMMARY:")
    if "elbo_final" in drift_metrics and "elbo_final" in bound_metrics:
        drift_elbo = drift_metrics["elbo_final"]
        bound_elbo = bound_metrics["elbo_final"]
        diff = abs(drift_elbo - bound_elbo)
        
        if diff < 1e-6:
            print("  • ELBO values are nearly identical (difference < 1e-6)")
        elif diff < 1e-3:
            print("  • ELBO values are very similar (difference < 1e-3)")
        else:
            print(f"  • ELBO values differ by {diff:.6f}")
        
        print(f"  • Higher ELBO indicates better model fit: {'Drift Jump' if drift_elbo > bound_elbo else 'Bound Drop'}")
    
    # Create DataFrame for saving but don't display it
    comparison_data = []
    
    if "elbo_final" in drift_metrics and "elbo_final" in bound_metrics:
        drift_elbo = drift_metrics["elbo_final"]
        bound_elbo = bound_metrics["elbo_final"]
        diff = drift_elbo - bound_elbo
        
        comparison_data.append({
            "Metric": "ELBO (Final)",
            "Drift Jump Model": f"{drift_elbo:.6f}",
            "Bound Drop Model": f"{bound_elbo:.6f}",
            "Difference": f"{diff:+.6f}",
            "Better Model": "Drift Jump" if diff > 0 else "Bound Drop"
        })
    
    if "expected_log_joint" in drift_metrics and "expected_log_joint" in bound_metrics:
        drift_elj = drift_metrics["expected_log_joint"]
        bound_elj = bound_metrics["expected_log_joint"]
        diff = drift_elj - bound_elj
        
        comparison_data.append({
            "Metric": "Expected Log Joint",
            "Drift Jump Model": f"{drift_elj:.6f}",
            "Bound Drop Model": f"{bound_elj:.6f}",
            "Difference": f"{diff:+.6f}",
            "Better Model": "Drift Jump" if diff > 0 else "Bound Drop"
        })
    
    if "entropy" in drift_metrics and "entropy" in bound_metrics:
        drift_ent = drift_metrics["entropy"]
        bound_ent = bound_metrics["entropy"]
        diff = drift_ent - bound_ent
        
        comparison_data.append({
            "Metric": "Entropy",
            "Drift Jump Model": f"{drift_ent:.6f}",
            "Bound Drop Model": f"{bound_ent:.6f}",
            "Difference": f"{diff:+.6f}",
            "Better Model": "Higher (more uncertain)" if diff > 0 else "Lower (more certain)"
        })
    
    return pd.DataFrame(comparison_data)

# Run comparison
comparison_df = compare_models(drift_jump_metrics, bound_drop_metrics)

# %%
# =============================================================================
# SAVE COMPARISON RESULTS
# =============================================================================
# Save comparison to CSV for record-keeping
comparison_csv_path = "model_comparison_results.csv"
comparison_df.to_csv(comparison_csv_path, index=False)
print(f"\n📁 Comparison results saved to: {comparison_csv_path}")

# Also save raw metrics as pickle
metrics_dict = {
    "drift_jump": drift_jump_metrics,
    "bound_drop": bound_drop_metrics,
    "comparison_dataframe": comparison_df
}

with open("model_comparison_raw_metrics.pkl", "wb") as f:
    pickle.dump(metrics_dict, f)

print(f"📁 Raw metrics saved to: model_comparison_raw_metrics.pkl")
print("\n✅ Script complete!")
