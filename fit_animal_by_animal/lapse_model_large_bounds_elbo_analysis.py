# %%
#!/usr/bin/env python3
"""
Analyze convergence of lapse model fits from VBMC pickle files.
Extracts ELBO and stable flag from all fitted animals.
"""
# %%
import pickle
import os
import glob
import pandas as pd
import argparse
import numpy as np

# %%
def extract_convergence_info(pkl_path):
    """
    Extract convergence information from a VBMC pickle file.
    
    Returns:
        dict with keys: elbo, elbo_sd, stable, n_iterations, rate_norm_l
    """
    try:
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
            
            # Extract rate_norm_l from vp samples
            # Parameters: rate_lambda, T_0, theta_E, w, t_E_aff, del_go, rate_norm_l, lapse_prob, lapse_prob_right
            # rate_norm_l is at index 6
            try:
                if hasattr(vbmc, 'vp'):
                    vp = vbmc.vp
                    vp_samples = vp.sample(int(1e5))[0]  # Use 100k samples for efficiency
                    rate_norm_l_samples = vp_samples[:, 6]
                    result['rate_norm_l'] = float(np.mean(rate_norm_l_samples))
                else:
                    result['rate_norm_l'] = None
            except Exception as e:
                print(f"Warning: Could not extract rate_norm_l from {pkl_path}: {e}")
                result['rate_norm_l'] = None
            
            return result
        else:
            return {'elbo': None, 'elbo_sd': None, 'stable': None, 'n_iterations': None, 'rate_norm_l': None}
    
    except Exception as e:
        print(f"Error reading {pkl_path}: {e}")
        return {'elbo': None, 'elbo_sd': None, 'stable': None, 'n_iterations': None, 'rate_norm_l': None, 'error': str(e)}

# %%
def parse_filename(filename):
    """
    Parse the pickle filename to extract batch, animal, and init_type.
    Expected format: vbmc_norm_tied_results_batch_{batch}_animal_{animal}_lapses_truncate_1s_{init_type}.pkl
    """
    # Remove .pkl extension
    name = filename.replace('.pkl', '')
    
    # Split by underscores
    parts = name.split('_')
    
    # Find batch name (after 'batch_')
    batch_idx = parts.index('batch') + 1
    batch_parts = []
    animal_idx = None
    
    for i in range(batch_idx, len(parts)):
        if parts[i] == 'animal':
            animal_idx = i + 1
            break
        batch_parts.append(parts[i])
    
    batch = '_'.join(batch_parts)
    
    # Get animal ID
    animal = int(parts[animal_idx])
    
    # Get init_type (last part)
    init_type = parts[-1]
    
    return batch, animal, init_type

# %%
def get_original_elbos(batch, animal_id, results_dir):
    """
    Load original vanilla and norm ELBO values from results pickle.
    
    Returns:
        dict with keys: og_vanilla_elbo, og_norm_elbo
    """
    pkl_fname = f'results_{batch}_animal_{animal_id}.pkl'
    pkl_path = os.path.join(results_dir, pkl_fname)
    
    result = {'og_vanilla_elbo': None, 'og_norm_elbo': None}
    
    if not os.path.exists(pkl_path):
        return result
    
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract vanilla ELBO
        if 'vbmc_vanilla_tied_results' in results:
            result['og_vanilla_elbo'] = results['vbmc_vanilla_tied_results'].get('elbo', None)
        
        # Extract norm ELBO
        if 'vbmc_norm_tied_results' in results:
            result['og_norm_elbo'] = results['vbmc_norm_tied_results'].get('elbo', None)
        
        return result
    except Exception as e:
        print(f"Warning: Could not load original ELBOs from {pkl_path}: {e}")
        return result

# %%
# Configuration
input_dir = 'oct_6_7_large_bounds_diff_init_lapse_fit'
output_csv = 'large_bounds_lapse_convergence_results.csv'  # Set to None if you don't want to save CSV
output_csv_subset = 'large_bounds_lapse_elbo_comparison.csv'  # Subset CSV with key columns
RESULTS_DIR = os.path.dirname(__file__)  # Directory containing original results

# %%
# Find all pickle files
pkl_pattern = os.path.join(input_dir, 'vbmc_norm_tied_results_batch_*_animal_*_lapses_truncate_1s_*.pkl')
pkl_files = glob.glob(pkl_pattern)

print(f"Found {len(pkl_files)} pickle files in {input_dir}\n")

# %%
# Extract info from each file
results = []
for pkl_file in sorted(pkl_files):
    filename = os.path.basename(pkl_file)
    batch, animal, init_type = parse_filename(filename)
    
    conv_info = extract_convergence_info(pkl_file)
    
    results.append({
        'batch': batch,
        'animal': animal,
        'init_type': init_type,
        'stable': conv_info['stable'],
        'elbo': conv_info['elbo'],
        'elbo_sd': conv_info['elbo_sd'],
        'n_iterations': conv_info['n_iterations'],
        'rate_norm_l': conv_info['rate_norm_l']
    })

# %%
# Create DataFrame
df = pd.DataFrame(results)

# %%
# Pivot to get the desired format: batch, animal, vanilla_stable, norm_stable, vanilla_elbo, norm_elbo
pivot_data = []

for (batch, animal), group in df.groupby(['batch', 'animal']):
    row = {'batch': batch, 'animal': animal}
    
    for init_type in ['vanilla', 'norm']:
        init_data = group[group['init_type'] == init_type]
        if len(init_data) > 0:
            row[f'init_type_{init_type}_stable'] = init_data.iloc[0]['stable']
            row[f'init_type_{init_type}_elbo'] = init_data.iloc[0]['elbo']
            row[f'init_type_{init_type}_elbo_sd'] = init_data.iloc[0]['elbo_sd']
            row[f'init_type_{init_type}_n_iterations'] = init_data.iloc[0]['n_iterations']
            row[f'init_type_{init_type}_rate_norm_l'] = init_data.iloc[0]['rate_norm_l']
        else:
            row[f'init_type_{init_type}_stable'] = None
            row[f'init_type_{init_type}_elbo'] = None
            row[f'init_type_{init_type}_elbo_sd'] = None
            row[f'init_type_{init_type}_n_iterations'] = None
            row[f'init_type_{init_type}_rate_norm_l'] = None
    
    # Load original ELBOs
    og_elbos = get_original_elbos(batch, animal, RESULTS_DIR)
    row['og_vanilla_elbo'] = og_elbos['og_vanilla_elbo']
    row['og_norm_elbo'] = og_elbos['og_norm_elbo']
    
    pivot_data.append(row)

result_df = pd.DataFrame(pivot_data)

# Sort by batch and animal
result_df = result_df.sort_values(['batch', 'animal'])

# %%
# Print as dataframe
print("\n" + "=" * 120)
print("LAPSE MODEL CONVERGENCE ANALYSIS")
print("=" * 120)

# Set pandas display options for better formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)

print(result_df.to_string(index=False))
print("=" * 120)

# %%
# Summary statistics
print("\nSUMMARY:")
print(f"Total animals: {len(result_df)}")

vanilla_stable_count = result_df['init_type_vanilla_stable'].sum()
norm_stable_count = result_df['init_type_norm_stable'].sum()
vanilla_total = result_df['init_type_vanilla_stable'].notna().sum()
norm_total = result_df['init_type_norm_stable'].notna().sum()

print(f"Vanilla stable: {vanilla_stable_count}/{vanilla_total} ({100*vanilla_stable_count/vanilla_total:.1f}%)")
print(f"Norm stable: {norm_stable_count}/{norm_total} ({100*norm_stable_count/norm_total:.1f}%)")

# %%
# Save to CSV if requested
if output_csv:
    # Round numeric columns to 3 decimal places for readability
    df_to_save = result_df.copy()
    numeric_cols = ['init_type_vanilla_elbo', 'init_type_norm_elbo', 
                    'init_type_vanilla_elbo_sd', 'init_type_norm_elbo_sd',
                    'init_type_vanilla_rate_norm_l', 'init_type_norm_rate_norm_l',
                    'og_vanilla_elbo', 'og_norm_elbo']
    for col in numeric_cols:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].round(3)
    
    df_to_save.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

# %%
# Save subset CSV with key comparison columns
if output_csv_subset:
    subset_cols = ['batch', 'animal', 
                   'init_type_vanilla_elbo', 'init_type_norm_elbo',
                   'og_vanilla_elbo', 'og_norm_elbo',
                   'init_type_vanilla_rate_norm_l', 'init_type_norm_rate_norm_l']
    
    # Check which columns exist
    available_cols = [col for col in subset_cols if col in result_df.columns]
    
    df_subset = result_df[available_cols].copy()
    
    # Round numeric columns
    numeric_subset_cols = ['init_type_vanilla_elbo', 'init_type_norm_elbo',
                           'og_vanilla_elbo', 'og_norm_elbo',
                           'init_type_vanilla_rate_norm_l', 'init_type_norm_rate_norm_l']
    for col in numeric_subset_cols:
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].round(3)
    
    df_subset.to_csv(output_csv_subset, index=False)
    print(f"Subset CSV saved to: {output_csv_subset}")

# %%
# Optional: CLI wrapper function for command-line usage
def main():
    parser = argparse.ArgumentParser(description='Analyze convergence of lapse model VBMC fits')
    parser.add_argument('--input-dir', default='oct_6_7_large_bounds_diff_init_lapse_fit',
                        help='Directory containing VBMC pickle files')
    parser.add_argument('--output-csv', default=None,
                        help='Optional: save results to CSV file')
    parser.add_argument('--output-csv-subset', default=None,
                        help='Optional: save subset CSV with key comparison columns')
    args = parser.parse_args()
    
    results_dir = os.path.dirname(__file__)
    
    # Find all pickle files
    pkl_pattern = os.path.join(args.input_dir, 'vbmc_norm_tied_results_batch_*_animal_*_lapses_truncate_1s_*.pkl')
    pkl_files = glob.glob(pkl_pattern)
    
    print(f"Found {len(pkl_files)} pickle files in {args.input_dir}\n")
    
    # Extract info from each file
    results = []
    for pkl_file in sorted(pkl_files):
        filename = os.path.basename(pkl_file)
        batch, animal, init_type = parse_filename(filename)
        
        conv_info = extract_convergence_info(pkl_file)
        
        results.append({
            'batch': batch,
            'animal': animal,
            'init_type': init_type,
            'stable': conv_info['stable'],
            'elbo': conv_info['elbo'],
            'elbo_sd': conv_info['elbo_sd'],
            'n_iterations': conv_info['n_iterations'],
            'rate_norm_l': conv_info['rate_norm_l']
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Pivot to get the desired format: batch, animal, vanilla_stable, norm_stable, vanilla_elbo, norm_elbo
    pivot_data = []
    
    for (batch, animal), group in df.groupby(['batch', 'animal']):
        row = {'batch': batch, 'animal': animal}
        
        for init_type in ['vanilla', 'norm']:
            init_data = group[group['init_type'] == init_type]
            if len(init_data) > 0:
                row[f'init_type_{init_type}_stable'] = init_data.iloc[0]['stable']
                row[f'init_type_{init_type}_elbo'] = init_data.iloc[0]['elbo']
                row[f'init_type_{init_type}_elbo_sd'] = init_data.iloc[0]['elbo_sd']
                row[f'init_type_{init_type}_n_iterations'] = init_data.iloc[0]['n_iterations']
                row[f'init_type_{init_type}_rate_norm_l'] = init_data.iloc[0]['rate_norm_l']
            else:
                row[f'init_type_{init_type}_stable'] = None
                row[f'init_type_{init_type}_elbo'] = None
                row[f'init_type_{init_type}_elbo_sd'] = None
                row[f'init_type_{init_type}_n_iterations'] = None
                row[f'init_type_{init_type}_rate_norm_l'] = None
        
        # Load original ELBOs
        og_elbos = get_original_elbos(batch, animal, results_dir)
        row['og_vanilla_elbo'] = og_elbos['og_vanilla_elbo']
        row['og_norm_elbo'] = og_elbos['og_norm_elbo']
        
        pivot_data.append(row)
    
    result_df = pd.DataFrame(pivot_data)
    
    # Sort by batch and animal
    result_df = result_df.sort_values(['batch', 'animal'])
    
    # Print as dataframe
    print("\n" + "=" * 120)
    print("LAPSE MODEL CONVERGENCE ANALYSIS")
    print("=" * 120)
    
    # Set pandas display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(result_df.to_string(index=False))
    print("=" * 120)
    
    # Summary statistics
    print("\nSUMMARY:")
    print(f"Total animals: {len(result_df)}")
    
    vanilla_stable_count = result_df['init_type_vanilla_stable'].sum()
    norm_stable_count = result_df['init_type_norm_stable'].sum()
    vanilla_total = result_df['init_type_vanilla_stable'].notna().sum()
    norm_total = result_df['init_type_norm_stable'].notna().sum()
    
    print(f"Vanilla stable: {vanilla_stable_count}/{vanilla_total} ({100*vanilla_stable_count/vanilla_total:.1f}%)")
    print(f"Norm stable: {norm_stable_count}/{norm_total} ({100*norm_stable_count/norm_total:.1f}%)")
    
    # Save to CSV if requested
    if args.output_csv:
        # Round numeric columns to 3 decimal places for readability
        df_to_save = result_df.copy()
        numeric_cols = ['init_type_vanilla_elbo', 'init_type_norm_elbo', 
                        'init_type_vanilla_elbo_sd', 'init_type_norm_elbo_sd',
                        'init_type_vanilla_rate_norm_l', 'init_type_norm_rate_norm_l',
                        'og_vanilla_elbo', 'og_norm_elbo']
        for col in numeric_cols:
            if col in df_to_save.columns:
                df_to_save[col] = df_to_save[col].round(3)
        
        df_to_save.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to: {args.output_csv}")
    
    # Save subset CSV with key comparison columns
    if args.output_csv_subset:
        subset_cols = ['batch', 'animal', 
                       'init_type_vanilla_elbo', 'init_type_norm_elbo',
                       'og_vanilla_elbo', 'og_norm_elbo',
                       'init_type_vanilla_rate_norm_l', 'init_type_norm_rate_norm_l']
        
        # Check which columns exist
        available_cols = [col for col in subset_cols if col in result_df.columns]
        
        df_subset = result_df[available_cols].copy()
        
        # Round numeric columns
        numeric_subset_cols = ['init_type_vanilla_elbo', 'init_type_norm_elbo',
                               'og_vanilla_elbo', 'og_norm_elbo',
                               'init_type_vanilla_rate_norm_l', 'init_type_norm_rate_norm_l']
        for col in numeric_subset_cols:
            if col in df_subset.columns:
                df_subset[col] = df_subset[col].round(3)
        
        df_subset.to_csv(args.output_csv_subset, index=False)
        print(f"Subset CSV saved to: {args.output_csv_subset}")

# %%
# Uncomment to run as CLI script
# if __name__ == '__main__':
#     main()

# %%
