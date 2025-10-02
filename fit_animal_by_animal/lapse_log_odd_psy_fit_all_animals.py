"""
Lapse model fitting using log odds and psychometric approaches across all animals.
Generates comparison plots and saves fitted parameters for each animal.
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from collections import defaultdict
import os
import pickle
from tqdm import tqdm

# %%
# Model functions
def log_sigmoid_v3_lapse_biased(x, a, d, th, lapse_pR):
    """Log odds model with biased lapse parameter"""
    f = th * np.tanh(d * x)
    p0 = 1.0 / (1.0 + (np.exp(-2*f)))  # sigma(f)
    p_plus = a*lapse_pR + (1.0 - a) * p0
    p_minus = a*(1-lapse_pR) + (1.0 - a) * (1.0 - p0)
    eps = 0
    return np.log((p_plus + eps) / (p_minus + eps))


def psyc_lapse_biased(x, a, d, th, lapse_pR):
    """Psychometric model with biased lapse parameter"""
    f = th * np.tanh(d * x)
    p0 = 1.0 / (1.0 + (np.exp(-2*f)))  # sigma(f)
    p_plus = a*lapse_pR + (1.0 - a) * p0
    return p_plus


# %%
# Configuration
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
DO_RIGHT_TRUNCATE = True

# T_trunc configuration per batch
def get_T_trunc(batch_name):
    """Get T_trunc value based on batch name"""
    if batch_name == 'LED34_even':
        return 0.15
    else:
        return 0.3

# %%
# Load and merge data from all batches
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files 
    if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)

merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

# Get batch-animal pairs
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))

print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")

if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch, animal in batch_animal_pairs:
        animal_str = str(animal)
        if animal_str not in batch_to_animals[batch]:
            batch_to_animals[batch].append(animal_str)

    # Print summary table
    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0

    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")

    for batch in sorted(animal_strings.keys()):
        animals_str = animal_strings[batch]
        print(f"{batch:<{max_batch_len}}  {animals_str}")

# %%
def process_animal(batch_name, animal_id):
    """
    Process a single animal: load data, fit models, return parameters and data for plotting.
    """
    T_trunc = get_T_trunc(batch_name)
    
    # Load data
    csv_filename = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    exp_df = pd.read_csv(csv_filename)
    df_valid_and_aborts = exp_df[
        (exp_df['success'].isin([1,-1])) |
        (exp_df['abort_event'] == 3)
    ].copy()
    
    df_valid_animal = df_valid_and_aborts[
        (df_valid_and_aborts['animal'] == animal_id) & 
        (df_valid_and_aborts['success'].isin([1,-1]))
    ]
    
    # Apply right truncation filter if enabled
    if DO_RIGHT_TRUNCATE:
        df_valid_animal_filtered = df_valid_animal[
            (df_valid_animal['RTwrtStim'] > 0) & 
            (df_valid_animal['RTwrtStim'] <= 1.0)
        ].copy()
    else:
        df_valid_animal_filtered = df_valid_animal[df_valid_animal['RTwrtStim'] > 0].copy()
    
    df_valid_animal_filtered['abs_ILD'] = np.abs(df_valid_animal_filtered['ILD'])
    
    # Get unique ABL and ILD values
    ABL_vals = sorted(df_valid_animal_filtered['ABL'].unique())
    ILD_vals = sorted(df_valid_animal_filtered['ILD'].unique())
    
    # Limit to 3 ABLs
    ABL_vals = ABL_vals[:3]
    
    # Collect all log odds data across all ABLs for fitting
    all_x_logodds = []
    all_y_logodds = []
    
    for abl in ABL_vals:
        for ild in ILD_vals:
            empirical_subset = df_valid_animal_filtered[
                (df_valid_animal_filtered['ABL'] == abl) & 
                (df_valid_animal_filtered['ILD'] == ild)
            ]
            if len(empirical_subset) > 0:
                p_right = np.mean(empirical_subset['choice'] == 1)
                p_left = np.mean(empirical_subset['choice'] == -1)
                if p_left > 0 and p_right > 0:
                    log_odds_empirical = np.log(p_right / p_left)
                    all_x_logodds.append(ild)
                    all_y_logodds.append(log_odds_empirical)
    
    all_x_logodds = np.array(all_x_logodds)
    all_y_logodds = np.array(all_y_logodds)
    
    # Remove any NaN values
    valid_mask = ~np.isnan(all_y_logodds)
    all_x_logodds = all_x_logodds[valid_mask]
    all_y_logodds = all_y_logodds[valid_mask]
    
    # Fit log-odds model
    bounds_logodds = ([0.0, 0.01, 0.01, 0.0], [0.1, 10, 50, 1.0])
    p0_logodds = [0.02, 0.09, 3.0, 0.5]  # [a, d, th, lapse_pR]
    
    logodds_params = None
    logodds_r2 = None
    
    try:
        popt_logodds, _ = curve_fit(log_sigmoid_v3_lapse_biased, all_x_logodds, all_y_logodds, 
                                     p0=p0_logodds, bounds=bounds_logodds)
        y_pred_logodds = log_sigmoid_v3_lapse_biased(all_x_logodds, *popt_logodds)
        logodds_r2 = r2_score(all_y_logodds, y_pred_logodds)
        logodds_params = {
            'a': popt_logodds[0],
            'd': popt_logodds[1],
            'th': popt_logodds[2],
            'lapse_pR': popt_logodds[3],
            'r2': logodds_r2
        }
    except Exception as e:
        print(f"  Log-odds fitting failed for {batch_name}, animal {animal_id}: {e}")
    
    # Collect all psychometric data across all ABLs for fitting
    all_x_psyc = []
    all_y_psyc = []
    
    for abl in ABL_vals:
        for ild in ILD_vals:
            empirical_subset = df_valid_animal_filtered[
                (df_valid_animal_filtered['ABL'] == abl) & 
                (df_valid_animal_filtered['ILD'] == ild)
            ]
            if len(empirical_subset) > 0:
                p_right_empirical = np.mean(empirical_subset['choice'] == 1)
                all_x_psyc.append(ild)
                all_y_psyc.append(p_right_empirical)
    
    all_x_psyc = np.array(all_x_psyc)
    all_y_psyc = np.array(all_y_psyc)
    
    # Remove any NaN values
    valid_mask_psyc = ~np.isnan(all_y_psyc)
    all_x_psyc = all_x_psyc[valid_mask_psyc]
    all_y_psyc = all_y_psyc[valid_mask_psyc]
    
    # Fit psychometric model
    bounds_psyc = ([0.0, 0.01, 0.01, 0.0], [0.1, 10, 50, 1.0])
    p0_psyc = [0.02, 0.09, 3.0, 0.5]  # [a, d, th, lapse_pR]
    
    psyc_params = None
    psyc_r2 = None
    
    try:
        popt_psyc, _ = curve_fit(psyc_lapse_biased, all_x_psyc, all_y_psyc, 
                                  p0=p0_psyc, bounds=bounds_psyc)
        y_pred_psyc = psyc_lapse_biased(all_x_psyc, *popt_psyc)
        psyc_r2 = r2_score(all_y_psyc, y_pred_psyc)
        psyc_params = {
            'a': popt_psyc[0],
            'd': popt_psyc[1],
            'th': popt_psyc[2],
            'lapse_pR': popt_psyc[3],
            'r2': psyc_r2
        }
    except Exception as e:
        print(f"  Psychometric fitting failed for {batch_name}, animal {animal_id}: {e}")
    
    # Return all necessary data for plotting
    return {
        'batch_name': batch_name,
        'animal_id': animal_id,
        'T_trunc': T_trunc,
        'ABL_vals': ABL_vals,
        'ILD_vals': ILD_vals,
        'df_filtered': df_valid_animal_filtered,
        'logodds_params': logodds_params,
        'psyc_params': psyc_params
    }


# %%
def plot_animal_data(animal_data, pdf):
    """
    Create two 1x3 plots for an animal:
    1. Psychometric curves for each ABL
    2. Log odds for each ABL
    """
    batch_name = animal_data['batch_name']
    animal_id = animal_data['animal_id']
    ABL_vals = animal_data['ABL_vals']
    ILD_vals = animal_data['ILD_vals']
    df_filtered = animal_data['df_filtered']
    logodds_params = animal_data['logodds_params']
    psyc_params = animal_data['psyc_params']
    
    # Create figure with 2 rows of plots
    fig = plt.figure(figsize=(15, 10))
    
    # Row 1: Psychometric curves
    axes_psyc = [plt.subplot(2, 3, i+1) for i in range(3)]
    
    x_model = np.linspace(-16, 16, 300)
    
    for idx, abl in enumerate(ABL_vals):
        ax = axes_psyc[idx]
        
        # Collect empirical data
        empirical_p_right = []
        empirical_ild_vals = []
        
        for ild in ILD_vals:
            empirical_subset = df_filtered[
                (df_filtered['ABL'] == abl) & 
                (df_filtered['ILD'] == ild)
            ]
            if len(empirical_subset) > 0:
                p_right_empirical = np.mean(empirical_subset['choice'] == 1)
                empirical_p_right.append(p_right_empirical)
                empirical_ild_vals.append(ild)
        
        # Plot empirical data
        ax.plot(empirical_ild_vals, empirical_p_right, 's', color='black', markersize=8, 
                label='Data', markerfacecolor='none', markeredgecolor='black', alpha=0.7, markeredgewidth=2)
        
        # Plot psychometric fit
        if psyc_params is not None:
            y_psyc = psyc_lapse_biased(x_model, psyc_params['a'], psyc_params['d'], 
                                       psyc_params['th'], psyc_params['lapse_pR'])
            ax.plot(x_model, y_psyc, '-', color='blue', linewidth=2, 
                    label=f'Psyc fit (pR={psyc_params["lapse_pR"]:.3f})', alpha=0.8)
        
        # Plot log-odds fit converted to P(right)
        if logodds_params is not None:
            y_logodds = psyc_lapse_biased(x_model, logodds_params['a'], logodds_params['d'], 
                                          logodds_params['th'], logodds_params['lapse_pR'])
            ax.plot(x_model, y_logodds, '--', color='red', linewidth=2, 
                    label=f'Log-odds fit (pR={logodds_params["lapse_pR"]:.3f})', alpha=0.8)
        
        # Formatting
        ax.set_title(f'ABL = {abl} dB', fontsize=12)
        ax.set_xlabel('ILD (dB)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('P(choice = right)', fontsize=11)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.set_ylim(0, 1)
    
    # Row 2: Log odds
    axes_logodds = [plt.subplot(2, 3, i+4) for i in range(3)]
    
    for idx, abl in enumerate(ABL_vals):
        ax = axes_logodds[idx]
        
        # Collect empirical log odds
        empirical_log_odds = []
        empirical_ild_vals = []
        
        for ild in ILD_vals:
            empirical_subset = df_filtered[
                (df_filtered['ABL'] == abl) & 
                (df_filtered['ILD'] == ild)
            ]
            if len(empirical_subset) > 0:
                p_right = np.mean(empirical_subset['choice'] == 1)
                p_left = np.mean(empirical_subset['choice'] == -1)
                if p_left > 0 and p_right > 0:
                    log_odds_empirical = np.log(p_right / p_left)
                    empirical_log_odds.append(log_odds_empirical)
                    empirical_ild_vals.append(ild)
        
        # Plot empirical data
        ax.plot(empirical_ild_vals, empirical_log_odds, 's', color='black', markersize=8, 
                label='Data', markerfacecolor='none', markeredgecolor='black', alpha=0.7, markeredgewidth=2)
        
        # Plot log-odds fit
        if logodds_params is not None:
            y_logodds = log_sigmoid_v3_lapse_biased(x_model, logodds_params['a'], logodds_params['d'], 
                                                     logodds_params['th'], logodds_params['lapse_pR'])
            ax.plot(x_model, y_logodds, '-', color='red', linewidth=2, 
                    label=f'Log-odds fit (pR={logodds_params["lapse_pR"]:.3f})', alpha=0.8)
        
        # Plot psychometric fit converted to log-odds
        if psyc_params is not None:
            y_psyc_prob = psyc_lapse_biased(x_model, psyc_params['a'], psyc_params['d'], 
                                            psyc_params['th'], psyc_params['lapse_pR'])
            # Convert to log-odds
            y_psyc_logodds = np.log(y_psyc_prob / (1 - y_psyc_prob + 1e-6))
            ax.plot(x_model, y_psyc_logodds, '--', color='blue', linewidth=2, 
                    label=f'Psyc fit (pR={psyc_params["lapse_pR"]:.3f})', alpha=0.8)
        
        # Formatting
        ax.set_title(f'ABL = {abl} dB', fontsize=12)
        ax.set_xlabel('ILD (dB)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('log(P(right) / P(left))', fontsize=11)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.set_ylim(-5, 5)
    
    # Overall title with parameters
    title_lines = [f'Batch {batch_name}, Animal {animal_id} - Lapse Model Fits']
    
    if psyc_params is not None:
        psyc_line = (f'Psyc: a={psyc_params["a"]*100:.2f}%, d={psyc_params["d"]:.4f}, '
                     f'th={psyc_params["th"]:.2f}, lapse_pR={psyc_params["lapse_pR"]*100:.2f}%, '
                     f'R²={psyc_params["r2"]*100:.2f}%')
        title_lines.append(psyc_line)
    
    if logodds_params is not None:
        logodds_line = (f'LogOdds: a={logodds_params["a"]*100:.2f}%, d={logodds_params["d"]:.4f}, '
                        f'th={logodds_params["th"]:.2f}, lapse_pR={logodds_params["lapse_pR"]*100:.2f}%, '
                        f'R²={logodds_params["r2"]*100:.2f}%')
        title_lines.append(logodds_line)
    
    title_text = '\n'.join(title_lines)
    fig.suptitle(title_text, fontsize=10, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# %%
# Main processing loop
print("\n" + "="*60)
print("Processing all animals...")
print("="*60)

all_params = {}
pdf_filename = 'lapse_model_fits_all_animals.pdf'

with PdfPages(pdf_filename) as pdf:
    for batch_name, animal_id in tqdm(batch_animal_pairs, desc="Processing animals"):
        try:
            print(f"\nProcessing {batch_name}, animal {animal_id}...")
            
            # Process animal
            animal_data = process_animal(batch_name, int(animal_id))
            
            # Store parameters
            all_params[(batch_name, animal_id)] = {
                'logodds_fit': animal_data['logodds_params'],
                'psychometric_fit': animal_data['psyc_params'],
                'T_trunc': animal_data['T_trunc'],
                'n_trials': len(animal_data['df_filtered'])
            }
            
            # Create plots
            plot_animal_data(animal_data, pdf)
            
            # Print summary
            if animal_data['logodds_params'] is not None:
                print(f"  Log-odds fit: R² = {animal_data['logodds_params']['r2']:.4f}")
            if animal_data['psyc_params'] is not None:
                print(f"  Psychometric fit: R² = {animal_data['psyc_params']['r2']:.4f}")
                
        except Exception as e:
            print(f"  ERROR processing {batch_name}, animal {animal_id}: {e}")
            all_params[(batch_name, animal_id)] = {
                'logodds_fit': None,
                'psychometric_fit': None,
                'T_trunc': get_T_trunc(batch_name),
                'n_trials': 0,
                'error': str(e)
            }

print(f"\n{'='*60}")
print(f"Saved plots to: {pdf_filename}")
print(f"{'='*60}")

# %%
# Save parameters dictionary
params_filename = 'lapse_model_params_all_animals.pkl'
with open(params_filename, 'wb') as f:
    pickle.dump(all_params, f)

print(f"Saved parameters to: {params_filename}")

# %%
# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

successful_fits = 0
failed_fits = 0

for (batch, animal), params in all_params.items():
    if params['logodds_fit'] is not None and params['psychometric_fit'] is not None:
        successful_fits += 1
    else:
        failed_fits += 1

print(f"Total animals processed: {len(all_params)}")
print(f"Successful fits: {successful_fits}")
print(f"Failed fits: {failed_fits}")

# Print parameter comparison for successfully fitted animals
print("\n" + "="*60)
print("PARAMETER COMPARISON (first 5 animals)")
print("="*60)

count = 0
for (batch, animal), params in all_params.items():
    if count >= 5:
        break
    if params['logodds_fit'] is not None and params['psychometric_fit'] is not None:
        print(f"\n{batch}, Animal {animal}:")
        print(f"  Log-odds:     a={params['logodds_fit']['a']:.4f}, "
              f"d={params['logodds_fit']['d']:.4f}, "
              f"th={params['logodds_fit']['th']:.4f}, "
              f"lapse_pR={params['logodds_fit']['lapse_pR']:.4f}, "
              f"R²={params['logodds_fit']['r2']:.4f}")
        print(f"  Psychometric: a={params['psychometric_fit']['a']:.4f}, "
              f"d={params['psychometric_fit']['d']:.4f}, "
              f"th={params['psychometric_fit']['th']:.4f}, "
              f"lapse_pR={params['psychometric_fit']['lapse_pR']:.4f}, "
              f"R²={params['psychometric_fit']['r2']:.4f}")
        count += 1

print("\n" + "="*60)
print("PROCESSING COMPLETE")
print("="*60)

# %%
# Create comparison plots

# Filter successful fits
successful_animals = []
for (batch, animal), params in all_params.items():
    if params['logodds_fit'] is not None and params['psychometric_fit'] is not None:
        successful_animals.append((batch, animal))

print(f"\nCreating comparison plots for {len(successful_animals)} animals with successful fits...")

# %%
# Figure 1: R² comparison plot in ascending order
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(successful_animals)*0.5), 10))

# Sort animals by average R² (average of log-odds and psychometric R²)
animals_with_r2 = []
for batch, animal in successful_animals:
    params = all_params[(batch, animal)]
    avg_r2 = (params['logodds_fit']['r2'] + params['psychometric_fit']['r2']) / 2
    animals_with_r2.append((batch, animal, avg_r2))

animals_sorted = sorted(animals_with_r2, key=lambda x: x[2])

# Extract data for plotting
labels = [f"{batch}\n{animal}" for batch, animal, _ in animals_sorted]
x_pos = np.arange(len(labels))
logodds_r2 = [all_params[(batch, animal)]['logodds_fit']['r2'] for batch, animal, _ in animals_sorted]
psyc_r2 = [all_params[(batch, animal)]['psychometric_fit']['r2'] for batch, animal, _ in animals_sorted]

# Top panel: Log-odds R²
ax1.bar(x_pos, logodds_r2, color='red', alpha=0.7, label='Log-odds fit')
ax1.set_ylabel('R² (Log-odds fit)', fontsize=12, fontweight='bold')
ax1.set_title('Model Fit Quality: R² by Animal (Sorted by Average R²)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0.85, 1)
ax1.legend(fontsize=10)

# Bottom panel: Psychometric R²
ax2.bar(x_pos, psyc_r2, color='blue', alpha=0.7, label='Psychometric fit')
ax2.set_ylabel('R² (Psychometric fit)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Batch - Animal', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0.85, 1)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('lapse_model_r2_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: lapse_model_r2_comparison.png")

# %%
# Figure 2: Lapse rate "a" parameter comparison
fig2, ax = plt.subplots(1, 1, figsize=(max(20, len(successful_animals)*1), 6))

# Use same animal ordering as Figure 1
labels = [f"{batch}\n{animal}" for batch, animal, _ in animals_sorted]
x_pos = np.arange(len(labels))
logodds_a = [all_params[(batch, animal)]['logodds_fit']['a'] for batch, animal, _ in animals_sorted]
psyc_a = [all_params[(batch, animal)]['psychometric_fit']['a'] for batch, animal, _ in animals_sorted]

# Width of bars
width = 0.35

# Create bars
bars1 = ax.bar(x_pos - width/2, logodds_a, width, color='red', alpha=0.7, label='Log-odds fit')
bars2 = ax.bar(x_pos + width/2, psyc_a, width, color='blue', alpha=0.7, label='Psychometric fit')

# Add R² values on top of bars
for i, (batch, animal, _) in enumerate(animals_sorted):
    logodds_r2_val = all_params[(batch, animal)]['logodds_fit']['r2']
    psyc_r2_val = all_params[(batch, animal)]['psychometric_fit']['r2']
    
    # R² for log-odds bar
    ax.text(x_pos[i] - width/2, logodds_a[i], f'{logodds_r2_val:.2f}', 
            ha='center', va='bottom', fontsize=7, color='darkred', fontweight='bold')
    
    # R² for psychometric bar
    ax.text(x_pos[i] + width/2, psyc_a[i], f'{psyc_r2_val:.2f}', 
            ha='center', va='bottom', fontsize=7, color='darkblue', fontweight='bold')

ax.set_ylabel('Lapse rate parameter "a"', fontsize=12, fontweight='bold')
ax.set_xlabel('Batch - Animal', fontsize=12, fontweight='bold')
ax.set_title('Lapse Rate Parameter "a" Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('lapse_model_a_param_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: lapse_model_a_param_comparison.png")

# %%
# Figure 3: Lapse bias "lapse_pR" parameter comparison
fig3, ax = plt.subplots(1, 1, figsize=(max(12, len(successful_animals)*0.5), 6))

# Use same animal ordering as Figure 1
logodds_lapse_pR = [all_params[(batch, animal)]['logodds_fit']['lapse_pR'] for batch, animal, _ in animals_sorted]
psyc_lapse_pR = [all_params[(batch, animal)]['psychometric_fit']['lapse_pR'] for batch, animal, _ in animals_sorted]

# Create bars
bars1 = ax.bar(x_pos - width/2, logodds_lapse_pR, width, color='red', alpha=0.7, label='Log-odds fit')
bars2 = ax.bar(x_pos + width/2, psyc_lapse_pR, width, color='blue', alpha=0.7, label='Psychometric fit')

# Add R² values on top of bars
for i, (batch, animal, _) in enumerate(animals_sorted):
    logodds_r2_val = all_params[(batch, animal)]['logodds_fit']['r2']
    psyc_r2_val = all_params[(batch, animal)]['psychometric_fit']['r2']
    
    # R² for log-odds bar
    ax.text(x_pos[i] - width/2, logodds_lapse_pR[i], f'{logodds_r2_val:.2f}', 
            ha='center', va='bottom', fontsize=7, color='darkred', fontweight='bold')
    
    # R² for psychometric bar
    ax.text(x_pos[i] + width/2, psyc_lapse_pR[i], f'{psyc_r2_val:.2f}', 
            ha='center', va='bottom', fontsize=7, color='darkblue', fontweight='bold')

ax.set_ylabel('Lapse bias parameter "lapse_pR"', fontsize=12, fontweight='bold')
ax.set_xlabel('Batch - Animal', fontsize=12, fontweight='bold')
ax.set_title('Lapse Bias Parameter "lapse_pR" Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)  # Reference line at 0.5 (unbiased)

plt.tight_layout()
plt.savefig('lapse_model_lapse_pR_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: lapse_model_lapse_pR_comparison.png")

print("\n" + "="*60)
print("ALL COMPARISON PLOTS CREATED")
print("="*60)

# %%
# Print comprehensive parameter table
print("\n" + "="*150)
print("PARAMETER COMPARISON TABLE (sorted by Psychometric R² ascending)")
print("="*150)

# Sort by psychometric R²
animals_sorted_by_psyc_r2 = []
for batch, animal in successful_animals:
    params = all_params[(batch, animal)]
    psyc_r2 = params['psychometric_fit']['r2']
    animals_sorted_by_psyc_r2.append((batch, animal, psyc_r2))

animals_sorted_by_psyc_r2 = sorted(animals_sorted_by_psyc_r2, key=lambda x: x[2])

# Print header
header = (f"{'Batch-Animal':<15} | "
          f"{'R² (LO)':<10} {'R² (Psy)':<10} | "
          f"{'a (LO)':<10} {'a (Psy)':<10} | "
          f"{'lapse_pR (LO)':<12} {'lapse_pR (Psy)':<12}")
print(header)
print("-" * 150)

# Print rows
for batch, animal, _ in animals_sorted_by_psyc_r2:
    params = all_params[(batch, animal)]
    
    logodds_r2 = params['logodds_fit']['r2'] * 100
    psyc_r2 = params['psychometric_fit']['r2'] * 100
    
    logodds_a = params['logodds_fit']['a'] * 100
    psyc_a = params['psychometric_fit']['a'] * 100
    
    logodds_lapse_pR = params['logodds_fit']['lapse_pR'] * 100
    psyc_lapse_pR = params['psychometric_fit']['lapse_pR'] * 100
    
    row = (f"{batch}-{animal:<8} | "
           f"{logodds_r2:>6.2f} %  {psyc_r2:>6.2f} %  | "
           f"{logodds_a:>6.2f} %  {psyc_a:>6.2f} %  | "
           f"{logodds_lapse_pR:>6.2f} %    {psyc_lapse_pR:>6.2f} %")
    print(row)

print("="*150)
print(f"Total animals: {len(animals_sorted_by_psyc_r2)}")
print("Legend: LO = Log-odds fit, Psy = Psychometric fit")
print("All values shown as percentages (multiplied by 100)")
print("="*150)

# %%
# Save table as CSV
csv_filename = 'lapse_model_parameter_comparison.csv'

# Create DataFrame for CSV export
import csv

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write headeri
    csv_writer.writerow([
        'Batch', 'Animal',
        'R²_LogOdds (%)', 'R²_Psychometric (%)',
        'a_LogOdds (%)', 'a_Psychometric (%)',
        'lapse_pR_LogOdds (%)', 'lapse_pR_Psychometric (%)'
    ])
    
    # Write data rows
    for batch, animal, _ in animals_sorted_by_psyc_r2:
        params = all_params[(batch, animal)]
        
        csv_writer.writerow([
            batch,
            animal,
            f"{params['logodds_fit']['r2'] * 100:.2f}",
            f"{params['psychometric_fit']['r2'] * 100:.2f}",
            f"{params['logodds_fit']['a'] * 100:.2f}",
            f"{params['psychometric_fit']['a'] * 100:.2f}",
            f"{params['logodds_fit']['lapse_pR'] * 100:.2f}",
            f"{params['psychometric_fit']['lapse_pR'] * 100:.2f}"
        ])

print(f"\n✓ Parameter comparison table saved to: {csv_filename}")
print(f"  Rows: {len(animals_sorted_by_psyc_r2)}, sorted by Psychometric R² (ascending)")
# %%
