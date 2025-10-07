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

# %%
def extract_convergence_info(pkl_path):
    """
    Extract convergence information from a VBMC pickle file.
    
    Returns:
        dict with keys: elbo, elbo_sd, stable, n_iterations
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
            
            return result
        else:
            return {'elbo': None, 'elbo_sd': None, 'stable': None, 'n_iterations': None}
    
    except Exception as e:
        print(f"Error reading {pkl_path}: {e}")
        return {'elbo': None, 'elbo_sd': None, 'stable': None, 'n_iterations': None, 'error': str(e)}

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
# Configuration
input_dir = 'oct_6_7_large_bounds_diff_init_lapse_fit'
output_csv = None  # Set to filename if you want to save CSV, e.g., 'lapse_convergence_results.csv'

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
        'n_iterations': conv_info['n_iterations']
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
        else:
            row[f'init_type_{init_type}_stable'] = None
            row[f'init_type_{init_type}_elbo'] = None
            row[f'init_type_{init_type}_elbo_sd'] = None
            row[f'init_type_{init_type}_n_iterations'] = None
    
    pivot_data.append(row)

result_df = pd.DataFrame(pivot_data)

# Sort by batch and animal
result_df = result_df.sort_values(['batch', 'animal'])

# %%
# Print as table
print("=" * 150)
print("LAPSE MODEL CONVERGENCE ANALYSIS")
print("=" * 150)
print()

# Print header
print(f"{'Batch':<15} {'Animal':<8} {'init_Vanilla_Stable':<16} {'init_Norm_Stable':<16} {'init_Vanilla_ELBO':<18} {'init_Norm_ELBO':<18}")
print("-" * 150)

# Print rows
for _, row in result_df.iterrows():
    vanilla_stable = str(row['init_type_vanilla_stable']) if pd.notna(row['init_type_vanilla_stable']) else 'N/A'
    norm_stable = str(row['init_type_norm_stable']) if pd.notna(row['init_type_norm_stable']) else 'N/A'
    vanilla_elbo = f"{row['init_type_vanilla_elbo']:.2f}" if pd.notna(row['init_type_vanilla_elbo']) else 'N/A'
    norm_elbo = f"{row['init_type_norm_elbo']:.2f}" if pd.notna(row['init_type_norm_elbo']) else 'N/A'
    
    print(f"{row['batch']:<15} {row['animal']:<8} {vanilla_stable:<16} {norm_stable:<16} {vanilla_elbo:<18} {norm_elbo:<18}")

print("=" * 150)

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
    result_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

# %%
# Optional: CLI wrapper function for command-line usage
def main():
    parser = argparse.ArgumentParser(description='Analyze convergence of lapse model VBMC fits')
    parser.add_argument('--input-dir', default='oct_6_7_large_bounds_diff_init_lapse_fit',
                        help='Directory containing VBMC pickle files')
    parser.add_argument('--output-csv', default=None,
                        help='Optional: save results to CSV file')
    args = parser.parse_args()
    
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
            'n_iterations': conv_info['n_iterations']
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
            else:
                row[f'init_type_{init_type}_stable'] = None
                row[f'init_type_{init_type}_elbo'] = None
                row[f'init_type_{init_type}_elbo_sd'] = None
                row[f'init_type_{init_type}_n_iterations'] = None
        
        pivot_data.append(row)
    
    result_df = pd.DataFrame(pivot_data)
    
    # Sort by batch and animal
    result_df = result_df.sort_values(['batch', 'animal'])
    
    # Print as table
    print("=" * 150)
    print("LAPSE MODEL CONVERGENCE ANALYSIS")
    print("=" * 150)
    print()
    
    # Print header
    print(f"{'Batch':<15} {'Animal':<8} {'init_Vanilla_Stable':<16} {'init_Norm_Stable':<16} {'init_Vanilla_ELBO':<18} {'init_Norm_ELBO':<18}")
    print("-" * 150)
    
    # Print rows
    for _, row in result_df.iterrows():
        vanilla_stable = str(row['init_type_vanilla_stable']) if pd.notna(row['init_type_vanilla_stable']) else 'N/A'
        norm_stable = str(row['init_type_norm_stable']) if pd.notna(row['init_type_norm_stable']) else 'N/A'
        vanilla_elbo = f"{row['init_type_vanilla_elbo']:.2f}" if pd.notna(row['init_type_vanilla_elbo']) else 'N/A'
        norm_elbo = f"{row['init_type_norm_elbo']:.2f}" if pd.notna(row['init_type_norm_elbo']) else 'N/A'
        
        print(f"{row['batch']:<15} {row['animal']:<8} {vanilla_stable:<16} {norm_stable:<16} {vanilla_elbo:<18} {norm_elbo:<18}")
    
    print("=" * 150)
    
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
        result_df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to: {args.output_csv}")

# %%
# Uncomment to run as CLI script
# if __name__ == '__main__':
#     main()
