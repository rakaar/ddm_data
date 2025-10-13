# Compare Vanilla+lapse model vs Norm model for LED6 single animal
# Vanilla+lapse results from oct_9_10_vanila_lapse_model_fit_files/
# Norm results from results_LED6_animal_*.pkl
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import sys
import os

# Add parent directory's lapses folder to path
sys.path.append('../lapses')
from lapses_utils import simulate_psiam_tied_rate_norm

# %%
# =============================================================================
# STEP 1: LOAD PICKLE FILES FOR VANILLA+LAPSE AND NORM MODELS
# =============================================================================

# Configuration
batch_name = 'LED7'
animal_id = 103 # Change this to 81, 82, 84, or 86 for different LED6 animals
output_dir = 'LED7_vanilla_lapse_vs_norm_analysis'
os.makedirs(output_dir, exist_ok=True)

# Paths to results
vanilla_lapse_pkl_path = f'oct_9_10_vanila_lapse_model_fit_files/vbmc_vanilla_tied_results_batch_{batch_name}_animal_{animal_id}_lapses_truncate_1s.pkl'
norm_pkl_path = f'results_{batch_name}_animal_{animal_id}.pkl'

DO_RIGHT_TRUNCATE = True
max_rt = 1.0  # Truncate at 1s

print(f"\n{'='*70}")
print(f"ANALYZING: Batch {batch_name}, Animal {animal_id}")
print(f"{'='*70}\n")

# %%
# Load CSV data for the animal
csv_filename = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
exp_df = pd.read_csv(csv_filename)

# Create df_valid_and_aborts
df_valid_and_aborts = exp_df[
    (exp_df['success'].isin([1,-1])) |
    (exp_df['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]

# Filter for the specific animal
df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal_id]
df_aborts_animal = df_aborts[df_aborts['animal'] == animal_id]
df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]

# Apply right truncation
if DO_RIGHT_TRUNCATE:
    df_valid_animal = df_valid_animal[df_valid_animal['RTwrtStim'] < max_rt]

print(f"Animal {animal_id}: {len(df_valid_animal)} valid trials (after truncation)")

# %%
# Load Vanilla+lapse model parameters
print(f"\n{'='*70}")
print(f"LOADING VANILLA+LAPSE MODEL PARAMETERS")
print(f"{'='*70}")
print(f"Path: {vanilla_lapse_pkl_path}\n")

with open(vanilla_lapse_pkl_path, 'rb') as f:
    vp_vanilla_lapse = pickle.load(f).vp

# Sample from VBMC VP
vp_samples = vp_vanilla_lapse.sample(int(1e6))[0]
# Params: rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, lapse_prob_right
vanilla_lapse_rate_lambda = np.mean(vp_samples[:, 0])
vanilla_lapse_T_0 = np.mean(vp_samples[:, 1])
vanilla_lapse_theta_E = np.mean(vp_samples[:, 2])
vanilla_lapse_w = np.mean(vp_samples[:, 3])
vanilla_lapse_t_E_aff = np.mean(vp_samples[:, 4])
vanilla_lapse_del_go = np.mean(vp_samples[:, 5])
vanilla_lapse_lapse_prob = np.mean(vp_samples[:, 6])
vanilla_lapse_lapse_prob_right = np.mean(vp_samples[:, 7])
vanilla_lapse_Z_E = (vanilla_lapse_w - 0.5) * 2 * vanilla_lapse_theta_E

print(f"Vanilla+Lapse Model Parameters:")
print(f"  rate_lambda      : {vanilla_lapse_rate_lambda:.6f}")
print(f"  T_0 (ms)         : {vanilla_lapse_T_0*1000:.6f}")
print(f"  theta_E          : {vanilla_lapse_theta_E:.6f}")
print(f"  w                : {vanilla_lapse_w:.6f}")
print(f"  Z_E              : {vanilla_lapse_Z_E:.6f}")
print(f"  t_E_aff (ms)     : {vanilla_lapse_t_E_aff*1000:.6f}")
print(f"  del_go (ms)      : {vanilla_lapse_del_go*1000:.6f}")
print(f"  lapse_prob       : {vanilla_lapse_lapse_prob:.6f}")
print(f"  lapse_prob_right : {vanilla_lapse_lapse_prob_right:.6f}")

# %%
# Load Norm model parameters
print(f"\n{'='*70}")
print(f"LOADING NORM MODEL PARAMETERS")
print(f"{'='*70}")
print(f"Path: {norm_pkl_path}\n")

with open(norm_pkl_path, 'rb') as f:
    fit_results_data = pickle.load(f)

# Extract norm tied samples
norm_tied_samples = fit_results_data['vbmc_norm_tied_results']
norm_rate_lambda = np.mean(norm_tied_samples['rate_lambda_samples'])
norm_T_0 = np.mean(norm_tied_samples['T_0_samples'])
norm_theta_E = np.mean(norm_tied_samples['theta_E_samples'])
norm_w = np.mean(norm_tied_samples['w_samples'])
norm_t_E_aff = np.mean(norm_tied_samples['t_E_aff_samples'])
norm_del_go = np.mean(norm_tied_samples['del_go_samples'])
norm_rate_norm_l = np.mean(norm_tied_samples['rate_norm_l_samples'])
norm_Z_E = (norm_w - 0.5) * 2 * norm_theta_E

# Extract abort params (needed for simulation)
abort_samples = fit_results_data['vbmc_aborts_results']
V_A = np.mean(abort_samples['V_A_samples'])
theta_A = np.mean(abort_samples['theta_A_samples'])
t_A_aff = np.mean(abort_samples['t_A_aff_samp'])

print(f"Norm Model Parameters:")
print(f"  rate_lambda      : {norm_rate_lambda:.6f}")
print(f"  T_0 (ms)         : {norm_T_0*1000:.6f}")
print(f"  theta_E          : {norm_theta_E:.6f}")
print(f"  w                : {norm_w:.6f}")
print(f"  Z_E              : {norm_Z_E:.6f}")
print(f"  t_E_aff (ms)     : {norm_t_E_aff*1000:.6f}")
print(f"  del_go (ms)      : {norm_del_go*1000:.6f}")
print(f"  rate_norm_l      : {norm_rate_norm_l:.6f}")
print(f"\nAbort Parameters (used for simulation):")
print(f"  V_A              : {V_A:.6f}")
print(f"  theta_A          : {theta_A:.6f}")
print(f"  t_A_aff (ms)     : {t_A_aff*1000:.6f}")

# %%
# Print parameter comparison table
print("\n" + "="*100)
print(f"PARAMETER COMPARISON: Vanilla+Lapse vs Norm (Batch {batch_name}, Animal {animal_id})")
print("="*100)
print(f"{'Parameter':<20} {'Vanilla+Lapse':<20} {'Norm':<20} {'|Diff|':<20} {'% Diff':<20}")
print("-"*100)

params_to_compare = [
    ('rate_lambda', vanilla_lapse_rate_lambda, norm_rate_lambda),
    ('T_0 (ms)', vanilla_lapse_T_0*1000, norm_T_0*1000),
    ('theta_E', vanilla_lapse_theta_E, norm_theta_E),
    ('w', vanilla_lapse_w, norm_w),
    ('Z_E', vanilla_lapse_Z_E, norm_Z_E),
    ('t_E_aff (ms)', vanilla_lapse_t_E_aff*1000, norm_t_E_aff*1000),
    ('del_go (ms)', vanilla_lapse_del_go*1000, norm_del_go*1000),
]

for param_name, val_lapse, val_norm in params_to_compare:
    diff = abs(val_lapse - val_norm)
    pct_diff = 100 * (val_lapse - val_norm) / val_norm if val_norm != 0 else np.nan
    print(f"{param_name:<20} {val_lapse:<20.6f} {val_norm:<20.6f} {diff:<20.6f} {pct_diff:<+20.2f}")

print(f"{'lapse_prob':<20} {vanilla_lapse_lapse_prob:<20.6f} {'N/A':<20} {'N/A':<20} {'N/A':<20}")
print(f"{'lapse_prob_right':<20} {vanilla_lapse_lapse_prob_right:<20.6f} {'N/A':<20} {'N/A':<20} {'N/A':<20}")
print(f"{'rate_norm_l':<20} {'N/A (=0)':<20} {norm_rate_norm_l:<20.6f} {'N/A':<20} {'N/A':<20}")
print("="*100)

# %%
# =============================================================================
# STEP 2: GENERATE SIMULATED DATA
# =============================================================================

# Simulation parameters
N_sim = int(1e6)
dt = 1e-4
T_lapse_max = max_rt

print(f"\n{'='*70}")
print(f"SIMULATING RTDs: Vanilla+Lapse vs Norm")
print(f"Batch: {batch_name}, Animal: {animal_id}")
print(f"N_sim: {N_sim:,}, dt: {dt}")
print(f"{'='*70}\n")

# Sample t_stim, ABL, ILD from the animal's valid trials
t_stim_samples = df_valid_animal['intended_fix'].sample(N_sim, replace=True).values
ABL_unique = df_valid_animal['ABL'].unique()
ILD_unique = df_valid_animal['ILD'].unique()

ABL_samples = np.random.choice(ABL_unique, size=N_sim, replace=True)
ILD_samples = np.random.choice(ILD_unique, size=N_sim, replace=True)

# Define simulation functions
def simulate_single_trial_vanilla_lapse(i):
    choice, rt, is_act = simulate_psiam_tied_rate_norm(
        V_A, theta_A, ABL_samples[i], ILD_samples[i],
        vanilla_lapse_rate_lambda, vanilla_lapse_T_0, vanilla_lapse_theta_E, vanilla_lapse_Z_E,
        t_stim_samples[i], t_A_aff, vanilla_lapse_t_E_aff, vanilla_lapse_del_go,
        0.0, dt, lapse_prob=vanilla_lapse_lapse_prob, T_lapse_max=T_lapse_max,
        lapse_prob_right=vanilla_lapse_lapse_prob_right
    )
    return {
        'choice': choice,
        'rt': rt,
        'is_act': is_act,
        'ABL': ABL_samples[i],
        'ILD': ILD_samples[i],
        't_stim': t_stim_samples[i]
    }

def simulate_single_trial_norm(i):
    choice, rt, is_act = simulate_psiam_tied_rate_norm(
        V_A, theta_A, ABL_samples[i], ILD_samples[i],
        norm_rate_lambda, norm_T_0, norm_theta_E, norm_Z_E,
        t_stim_samples[i], t_A_aff, norm_t_E_aff, norm_del_go,
        norm_rate_norm_l, dt, lapse_prob=0.0, T_lapse_max=T_lapse_max
    )
    return {
        'choice': choice,
        'rt': rt,
        'is_act': is_act,
        'ABL': ABL_samples[i],
        'ILD': ILD_samples[i],
        't_stim': t_stim_samples[i]
    }

print("Simulating with VANILLA+LAPSE model...")
vanilla_lapse_sim_results = Parallel(n_jobs=-2, verbose=5)(
    delayed(simulate_single_trial_vanilla_lapse)(i) for i in tqdm(range(N_sim))
)

print("\nSimulating with NORM model...")
norm_sim_results = Parallel(n_jobs=-2, verbose=5)(
    delayed(simulate_single_trial_norm)(i) for i in tqdm(range(N_sim))
)

# Convert to DataFrames
vanilla_lapse_sim_df = pd.DataFrame(vanilla_lapse_sim_results)
norm_sim_df = pd.DataFrame(norm_sim_results)

# Compute RT relative to stimulus onset
vanilla_lapse_sim_df['rt_minus_t_stim'] = vanilla_lapse_sim_df['rt'] - vanilla_lapse_sim_df['t_stim']
norm_sim_df['rt_minus_t_stim'] = norm_sim_df['rt'] - norm_sim_df['t_stim']

print(f"\nVanilla+Lapse simulation: {len(vanilla_lapse_sim_df):,} trials")
print(f"Norm simulation: {len(norm_sim_df):,} trials")
print(f"{'='*70}\n")

# %%
# Filter trials where rt_minus_t_stim > 0 and apply right truncation
vanilla_lapse_sim_df_filtered = vanilla_lapse_sim_df[vanilla_lapse_sim_df['rt_minus_t_stim'] > 0].copy()
norm_sim_df_filtered = norm_sim_df[norm_sim_df['rt_minus_t_stim'] > 0].copy()

if DO_RIGHT_TRUNCATE:
    vanilla_lapse_sim_df_filtered = vanilla_lapse_sim_df_filtered[
        vanilla_lapse_sim_df_filtered['rt_minus_t_stim'] < max_rt
    ].copy()
    norm_sim_df_filtered = norm_sim_df_filtered[
        norm_sim_df_filtered['rt_minus_t_stim'] < max_rt
    ].copy()

# Create abs_ILD column
vanilla_lapse_sim_df_filtered['abs_ILD'] = np.abs(vanilla_lapse_sim_df_filtered['ILD'])
norm_sim_df_filtered['abs_ILD'] = np.abs(norm_sim_df_filtered['ILD'])

# Prepare empirical data
df_valid_animal_filtered = df_valid_animal[df_valid_animal['RTwrtStim'] > 0].copy()
if DO_RIGHT_TRUNCATE:
    df_valid_animal_filtered = df_valid_animal_filtered[
        df_valid_animal_filtered['RTwrtStim'] < max_rt
    ].copy()
df_valid_animal_filtered['abs_ILD'] = np.abs(df_valid_animal_filtered['ILD'])

print(f"Filtered Vanilla+Lapse trials (rt > 0, < {max_rt}s): {len(vanilla_lapse_sim_df_filtered):,}")
print(f"Filtered Norm trials (rt > 0, < {max_rt}s): {len(norm_sim_df_filtered):,}")
print(f"Filtered empirical trials (rt > 0, < {max_rt}s): {len(df_valid_animal_filtered):,}")

# Get unique ABLs and abs_ILDs
ABL_vals = sorted(vanilla_lapse_sim_df_filtered['ABL'].unique())
abs_ILD_vals = sorted(vanilla_lapse_sim_df_filtered['abs_ILD'].unique())

print(f"\nABL values: {ABL_vals}")
print(f"Absolute ILD values: {abs_ILD_vals}")

# %%
# =============================================================================
# STEP 3: PLOT REACTION TIME DISTRIBUTIONS
# =============================================================================

# Plot RT distributions: 3 rows (ABL) x 5 columns (abs_ILD)
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle(f'RT Distributions: Vanilla+Lapse (blue) vs Norm (red) vs Data (green)\nBatch {batch_name}, Animal {animal_id}', 
             fontsize=16, fontweight='bold')

bins = np.arange(0, 1.3, 0.01)

for row_idx, abl in enumerate(ABL_vals[:3]):  # Limit to 3 ABLs
    for col_idx, abs_ild in enumerate(abs_ILD_vals[:5]):  # Limit to 5 abs_ILDs
        ax = axes[row_idx, col_idx]
        
        # Filter data for this ABL and abs_ILD
        vanilla_lapse_data = vanilla_lapse_sim_df_filtered[
            (vanilla_lapse_sim_df_filtered['ABL'] == abl) & 
            (vanilla_lapse_sim_df_filtered['abs_ILD'] == abs_ild)
        ]['rt_minus_t_stim']
        
        norm_data = norm_sim_df_filtered[
            (norm_sim_df_filtered['ABL'] == abl) & 
            (norm_sim_df_filtered['abs_ILD'] == abs_ild)
        ]['rt_minus_t_stim']
        
        empirical_data = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['abs_ILD'] == abs_ild)
        ]['RTwrtStim']
        
        # Plot histograms
        if len(vanilla_lapse_data) > 0:
            ax.hist(vanilla_lapse_data, bins=bins, density=True, histtype='step', 
                   color='blue', linewidth=2, label='Vanilla+Lapse')
        
        if len(norm_data) > 0:
            ax.hist(norm_data, bins=bins, density=True, histtype='step', 
                   color='red', linewidth=2, label='Norm')
        
        if len(empirical_data) > 0:
            ax.hist(empirical_data, bins=bins, density=True, histtype='step', 
                   color='green', linewidth=2, label='Data')
        
        # Set labels and title
        if row_idx == 0:
            ax.set_title(f'|ILD|={abs_ild}', fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(f'ABL={abl}\nDensity', fontsize=10)
        if row_idx == 2:
            ax.set_xlabel('RT (s)', fontsize=10)
        
        # Add legend only to top-left subplot
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=9)
        
        # Add grid and format
        ax.set_xlim(0, 0.7)
        ax.grid(alpha=0.3)

plt.tight_layout()
rt_dists_png = os.path.join(output_dir, f'rtds_vanilla_lapse_vs_norm_vs_data_{batch_name}_animal_{animal_id}.png')
fig.savefig(rt_dists_png, dpi=300, bbox_inches='tight')
print(f"\nSaved: {rt_dists_png}")
plt.show()

# %%
# =============================================================================
# STEP 4: PLOT PSYCHOMETRIC CURVES (P(Choose Right) vs ILD)
# =============================================================================

# Get unique ILD values (not absolute)
ILD_vals = sorted(vanilla_lapse_sim_df_filtered['ILD'].unique())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Psychometric Curves: P(Choose Right) vs ILD\nVanilla+Lapse (blue) vs Norm (red) vs Data (green) - Batch {batch_name}, Animal {animal_id}', 
             fontsize=14, fontweight='bold')

for idx, abl in enumerate(ABL_vals[:3]):  # 3 ABLs
    ax = axes[idx]
    
    vanilla_lapse_p_right = []
    norm_p_right = []
    empirical_p_right = []
    empirical_sem = []  # Standard error of mean for data
    
    for ild in ILD_vals:
        # Vanilla+Lapse model
        vanilla_lapse_subset = vanilla_lapse_sim_df_filtered[
            (vanilla_lapse_sim_df_filtered['ABL'] == abl) & 
            (vanilla_lapse_sim_df_filtered['ILD'] == ild)
        ]
        if len(vanilla_lapse_subset) > 0:
            p_right = np.mean(vanilla_lapse_subset['choice'] == 1)
        else:
            p_right = np.nan
        vanilla_lapse_p_right.append(p_right)
        
        # Norm model
        norm_subset = norm_sim_df_filtered[
            (norm_sim_df_filtered['ABL'] == abl) & 
            (norm_sim_df_filtered['ILD'] == ild)
        ]
        if len(norm_subset) > 0:
            p_right = np.mean(norm_subset['choice'] == 1)
        else:
            p_right = np.nan
        norm_p_right.append(p_right)
        
        # Empirical data
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            choices = (empirical_subset['choice'] == 1).astype(int)
            p_right = np.mean(choices)
            sem = np.std(choices) / np.sqrt(len(choices))
        else:
            p_right = np.nan
            sem = 0
        empirical_p_right.append(p_right)
        empirical_sem.append(sem)
    
    # Plot
    ax.plot(ILD_vals, vanilla_lapse_p_right, 'o-', color='blue', markersize=8, 
            label='Vanilla+Lapse', linewidth=2, alpha=0.7)
    ax.plot(ILD_vals, norm_p_right, 'x-', color='red', markersize=10, 
            markeredgewidth=2, label='Norm', linewidth=2, alpha=0.7)
    ax.errorbar(ILD_vals, empirical_p_right, yerr=empirical_sem, fmt='s', color='green', 
                markersize=8, label='Data', capsize=0, linewidth=2, elinewidth=1.5)
    
    # Formatting
    ax.set_title(f'ABL = {abl} dB', fontsize=12)
    ax.set_xlabel('ILD (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('P(Choose Right)', fontsize=11)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)

plt.tight_layout()
p_right_png = os.path.join(output_dir, f'psychometric_vanilla_lapse_vs_norm_vs_data_{batch_name}_animal_{animal_id}.png')
fig.savefig(p_right_png, dpi=300, bbox_inches='tight')
print(f"Saved: {p_right_png}")
plt.show()

# %%
# =============================================================================
# STEP 5: PLOT LOG-ODDS (log(P(right) / P(left)) vs ILD)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Log-Odds: log(P(right) / P(left)) vs ILD\nVanilla+Lapse (blue) vs Norm (red) vs Data (green) - Batch {batch_name}, Animal {animal_id}', 
             fontsize=14, fontweight='bold')

for idx, abl in enumerate(ABL_vals[:3]):  # 3 ABLs
    ax = axes[idx]
    
    vanilla_lapse_log_odds = []
    norm_log_odds = []
    empirical_log_odds = []
    
    for ild in ILD_vals:
        # Vanilla+Lapse model
        vanilla_lapse_subset = vanilla_lapse_sim_df_filtered[
            (vanilla_lapse_sim_df_filtered['ABL'] == abl) & 
            (vanilla_lapse_sim_df_filtered['ILD'] == ild)
        ]
        if len(vanilla_lapse_subset) > 0:
            p_right = np.mean(vanilla_lapse_subset['choice'] == 1)
            p_left = np.mean(vanilla_lapse_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_vanilla_lapse = np.log(p_right / p_left)
            else:
                log_odds_vanilla_lapse = np.nan
        else:
            log_odds_vanilla_lapse = np.nan
        vanilla_lapse_log_odds.append(log_odds_vanilla_lapse)
        
        # Norm model
        norm_subset = norm_sim_df_filtered[
            (norm_sim_df_filtered['ABL'] == abl) & 
            (norm_sim_df_filtered['ILD'] == ild)
        ]
        if len(norm_subset) > 0:
            p_right = np.mean(norm_subset['choice'] == 1)
            p_left = np.mean(norm_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_norm = np.log(p_right / p_left)
            else:
                log_odds_norm = np.nan
        else:
            log_odds_norm = np.nan
        norm_log_odds.append(log_odds_norm)
        
        # Empirical data
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right = np.mean(empirical_subset['choice'] == 1)
            p_left = np.mean(empirical_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_empirical = np.log(p_right / p_left)
            else:
                log_odds_empirical = np.nan
        else:
            log_odds_empirical = np.nan
        empirical_log_odds.append(log_odds_empirical)
    
    # Plot
    ax.plot(ILD_vals, vanilla_lapse_log_odds, 'o', color='blue', markersize=8, 
            label='Vanilla+Lapse', markerfacecolor='blue', markeredgecolor='blue')
    ax.plot(ILD_vals, norm_log_odds, 'x', color='red', markersize=10, 
            markeredgewidth=2, label='Norm')
    ax.plot(ILD_vals, empirical_log_odds, 's', color='green', markersize=8, 
            label='Data', markerfacecolor='green', markeredgecolor='green')
    
    # Formatting
    ax.set_title(f'ABL = {abl} dB', fontsize=12)
    ax.set_xlabel('ILD (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('log(P(right) / P(left))', fontsize=11)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 5)

plt.tight_layout()
log_odds_png = os.path.join(output_dir, f'log_odds_vanilla_lapse_vs_norm_vs_data_{batch_name}_animal_{animal_id}.png')
fig.savefig(log_odds_png, dpi=300, bbox_inches='tight')
print(f"Saved: {log_odds_png}")
plt.show()

# %%
print(f"\n{'='*70}")
print(f"ANALYSIS COMPLETE!")
print(f"All plots saved to: {output_dir}/")
print(f"{'='*70}")

# %%
# =============================================================================
# HYPOTHESIS: ELBO difference is high if lapse probability is high
# =============================================================================

print(f"\n{'='*70}")
print(f"TESTING HYPOTHESIS: ELBO Difference vs Lapse Probability")
print(f"{'='*70}\n")

# Analyze all LED6 animals
led6_animals = [81, 82, 84, 86]
lapse_probs = []
elbo_differences = []
animal_labels = []

for animal in led6_animals:
    print(f"Processing Animal {animal}...")
    
    # Load Vanilla+Lapse VBMC object
    vanilla_lapse_pkl = f'oct_9_10_vanila_lapse_model_fit_files/vbmc_vanilla_tied_results_batch_LED6_animal_{animal}_lapses_truncate_1s.pkl'
    try:
        with open(vanilla_lapse_pkl, 'rb') as f:
            vbmc_vanilla_lapse = pickle.load(f)
            vp_obj = vbmc_vanilla_lapse.vp
            
            # Extract ELBO from iteration history
            iter_hist = vbmc_vanilla_lapse.iteration_history
            if 'elbo' in iter_hist:
                elbo_vanilla_lapse = float(iter_hist['elbo'][-1])  # Take last ELBO value
            else:
                print(f"  Warning: Could not find ELBO in iteration history for animal {animal}")
                continue
            
            # Sample to get lapse_prob
            vp_samples_temp = vp_obj.sample(int(1e6))[0]
            lapse_prob = np.mean(vp_samples_temp[:, 6])  # 7th parameter is lapse_prob
            
        # Load vanilla model ELBO from results pkl
        results_pkl = f'results_LED6_animal_{animal}.pkl'
        with open(results_pkl, 'rb') as f:
            results_data = pickle.load(f)
            # Get vanilla tied ELBO
            if 'vbmc_vanilla_tied_results' in results_data:
                vanilla_tied_results = results_data['vbmc_vanilla_tied_results']
                if 'elbo' in vanilla_tied_results:
                    elbo_vanilla = vanilla_tied_results['elbo']
                else:
                    print(f"  Warning: Could not find vanilla ELBO for animal {animal}")
                    continue
            else:
                print(f"  Warning: Could not find vanilla tied results for animal {animal}")
                continue
        
        # Compute ELBO difference (Vanilla+Lapse - Vanilla)
        # Positive means vanilla+lapse is better
        elbo_diff = elbo_vanilla_lapse - elbo_vanilla
        
        lapse_probs.append(lapse_prob)
        elbo_differences.append(elbo_diff)
        animal_labels.append(str(animal))
        
        print(f"  Lapse prob: {lapse_prob:.6f}")
        print(f"  ELBO (Vanilla+Lapse): {elbo_vanilla_lapse:.2f}")
        print(f"  ELBO (Vanilla): {elbo_vanilla:.2f}")
        print(f"  ELBO Difference: {elbo_diff:.2f}\n")
        
    except FileNotFoundError as e:
        print(f"  Skipping animal {animal}: File not found ({e})")
        continue
    except Exception as e:
        print(f"  Error processing animal {animal}: {e}")
        continue

# %%
# Plot lapse_prob vs ELBO difference
if len(lapse_probs) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(lapse_probs, elbo_differences, s=200, alpha=0.7, color='purple', edgecolor='black', linewidth=2)
    
    # Annotate each point with animal ID
    for i, label in enumerate(animal_labels):
        ax.annotate(f'Animal {label}', 
                   (lapse_probs[i], elbo_differences[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Add a trend line if we have enough points
    if len(lapse_probs) >= 2:
        z = np.polyfit(lapse_probs, elbo_differences, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(lapse_probs), max(lapse_probs), 100)
        ax.plot(x_line, p(x_line), "--", color='red', linewidth=2, alpha=0.5, 
               label=f'Linear fit: y={z[0]:.1f}x + {z[1]:.1f}')
    
    # Formatting
    ax.set_xlabel('Lapse Probability (Vanilla+Lapse Model)', fontsize=13, fontweight='bold')
    ax.set_ylabel('ELBO Difference\n(Vanilla+Lapse - Vanilla)', fontsize=13, fontweight='bold')
    ax.set_title('Hypothesis: Higher Lapse Probability → Larger ELBO Improvement\nBatch LED6', 
                fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    if len(lapse_probs) >= 2:
        ax.legend(fontsize=11)
    
    # Add text box with correlation
    if len(lapse_probs) >= 2:
        correlation = np.corrcoef(lapse_probs, elbo_differences)[0, 1]
        textstr = f'Correlation: {correlation:.3f}\nn = {len(lapse_probs)} animals'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    hypothesis_png = os.path.join(output_dir, f'hypothesis_lapse_prob_vs_elbo_diff_LED6.png')
    fig.savefig(hypothesis_png, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {hypothesis_png}")
    plt.show()
else:
    print("No data available to plot!")

# %%
# =============================================================================
# HYPOTHESIS EXTENDED: ALL ANIMALS - ELBO Difference vs Lapse Probability
# =============================================================================

print(f"\n{'='*70}")
print(f"TESTING HYPOTHESIS: ELBO Difference vs Lapse Probability")
print(f"ANALYZING ALL ANIMALS ACROSS ALL BATCHES")
print(f"{'='*70}\n")

import glob

# Directories
vanilla_lapse_dir = 'oct_9_10_vanila_lapse_model_fit_files'
results_dir = '.'

# Find all vanilla+lapse pickle files
vanilla_lapse_files = glob.glob(os.path.join(vanilla_lapse_dir, 'vbmc_vanilla_tied_results_batch_*_animal_*_lapses_truncate_1s.pkl'))
print(f"Found {len(vanilla_lapse_files)} vanilla+lapse pickle files\n")

# Storage for all animals
all_lapse_probs = []
all_elbo_differences = []
all_batches = []
all_animals = []

for pkl_path in vanilla_lapse_files:
    filename = os.path.basename(pkl_path)
    
    # Parse filename to extract batch and animal
    # Format: vbmc_vanilla_tied_results_batch_{batch}_animal_{animal}_lapses_truncate_1s.pkl
    try:
        name_parts = filename.replace('.pkl', '').split('_')
        batch_idx = name_parts.index('batch') + 1
        animal_idx = name_parts.index('animal') + 1
        
        # Extract batch (handle multi-word batches like LED34_even)
        batch_parts = []
        for i in range(batch_idx, len(name_parts)):
            if name_parts[i] == 'animal':
                break
            batch_parts.append(name_parts[i])
        batch = '_'.join(batch_parts)
        
        animal = int(name_parts[animal_idx])
        
        print(f"Processing {batch} - Animal {animal}...")
        
        # Load Vanilla+Lapse VBMC object
        with open(pkl_path, 'rb') as f:
            vbmc_vanilla_lapse = pickle.load(f)
            vp_obj = vbmc_vanilla_lapse.vp
            
            # Extract ELBO from iteration history
            iter_hist = vbmc_vanilla_lapse.iteration_history
            if 'elbo' in iter_hist:
                elbo_vanilla_lapse = float(iter_hist['elbo'][-1])
            else:
                print(f"  Warning: No ELBO found, skipping...")
                continue
            
            # Sample to get lapse_prob
            vp_samples_temp = vp_obj.sample(int(1e6))[0]
            lapse_prob = np.mean(vp_samples_temp[:, 6])  # 7th parameter is lapse_prob
        
        # Load vanilla model ELBO from results pkl
        results_pkl = f'results_{batch}_animal_{animal}.pkl'
        if not os.path.exists(results_pkl):
            print(f"  Warning: Results file not found, skipping...")
            continue
            
        with open(results_pkl, 'rb') as f:
            results_data = pickle.load(f)
            
            if 'vbmc_vanilla_tied_results' in results_data:
                vanilla_tied_results = results_data['vbmc_vanilla_tied_results']
                if 'elbo' in vanilla_tied_results:
                    elbo_vanilla = vanilla_tied_results['elbo']
                else:
                    print(f"  Warning: No vanilla ELBO found, skipping...")
                    continue
            else:
                print(f"  Warning: No vanilla tied results found, skipping...")
                continue
        
        # Compute ELBO difference
        elbo_diff = elbo_vanilla_lapse - elbo_vanilla
        
        # Store results
        all_lapse_probs.append(lapse_prob)
        all_elbo_differences.append(elbo_diff)
        all_batches.append(batch)
        all_animals.append(animal)
        
        print(f"  Lapse prob: {lapse_prob:.6f}")
        print(f"  ELBO difference: {elbo_diff:.2f}\n")
        
    except Exception as e:
        print(f"  Error processing {filename}: {e}\n")
        continue

print(f"\nSuccessfully processed {len(all_lapse_probs)} animals")

# %%
# Plot 1: UNNORMALIZED ELBO difference (Vanilla+Lapse - Vanilla) for ALL animals
batches_to_exclude = ['LED34', 'LED7']  # List of batches to exclude from analysis

if len(all_lapse_probs) > 0:
    # Filter data based on batches_to_exclude
    if len(batches_to_exclude) > 0:
        plot_indices = [i for i, b in enumerate(all_batches) if b not in batches_to_exclude]
        plot_lapse_probs = [all_lapse_probs[i] for i in plot_indices]
        plot_elbo_diffs = [all_elbo_differences[i] for i in plot_indices]
        plot_batches = [all_batches[i] for i in plot_indices]
        excluded_str = ', '.join(batches_to_exclude)
    else:
        plot_lapse_probs = all_lapse_probs
        plot_elbo_diffs = all_elbo_differences
        plot_batches = all_batches
        excluded_str = 'None'
    
    # Get unique batches for color mapping (use all batches for consistent colors)
    unique_batches_all = sorted(set(all_batches))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_batches_all)))
    batch_to_color = {batch: colors[i] for i, batch in enumerate(unique_batches_all)}
    
    # Get unique batches in plot data
    unique_batches = sorted(set(plot_batches))
    
    # UNNORMALIZED PLOT
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Scatter plot with different colors for each batch
    for batch in unique_batches:
        batch_mask = [b == batch for b in plot_batches]
        batch_lapse_probs = [lp for lp, m in zip(plot_lapse_probs, batch_mask) if m]
        batch_elbo_diffs = [ed for ed, m in zip(plot_elbo_diffs, batch_mask) if m]
        
        ax.scatter(batch_lapse_probs, batch_elbo_diffs, 
                  s=100, alpha=0.6, color=batch_to_color[batch], 
                  edgecolor='black', linewidth=1, label=batch)
    
    # Add overall trend line
    if len(plot_lapse_probs) >= 2:
        z = np.polyfit(plot_lapse_probs, plot_elbo_diffs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(plot_lapse_probs), max(plot_lapse_probs), 100)
        ax.plot(x_line, p(x_line), "--", color='red', linewidth=3, alpha=0.7, 
               label=f'Linear fit: y={z[0]:.1f}x + {z[1]:.1f}')
    
    # Formatting
    ax.set_xlabel('Lapse Probability (Vanilla+Lapse Model)', fontsize=14, fontweight='bold')
    ax.set_ylabel('ELBO Difference\n(Vanilla+Lapse - Vanilla)', fontsize=14, fontweight='bold')
    title_str = 'Hypothesis: Higher Lapse Probability → Larger ELBO Improvement\n'
    if len(batches_to_exclude) > 0:
        title_str += f'All Animals (Excluded: {excluded_str})'
    else:
        title_str += 'All Animals Across All Batches'
    ax.set_title(title_str, fontsize=15, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    
    # Add text box with correlation
    if len(plot_lapse_probs) >= 2:
        correlation = np.corrcoef(plot_lapse_probs, plot_elbo_diffs)[0, 1]
        textstr = f'Correlation: {correlation:.3f}\nn = {len(plot_lapse_probs)} animals\n{len(unique_batches)} batches'
        if len(batches_to_exclude) > 0:
            textstr += f'\nExcluded: {excluded_str}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    filename_suffix = '_excluded_' + '_'.join(batches_to_exclude) if len(batches_to_exclude) > 0 else ''
    hypothesis_all_png = os.path.join(output_dir, f'hypothesis_lapse_prob_vs_elbo_diff_ALL_ANIMALS_UNNORMALIZED{filename_suffix}.png')
    fig.savefig(hypothesis_all_png, dpi=300, bbox_inches='tight')
    print(f"\nSaved unnormalized: {hypothesis_all_png}")
    plt.show()
    
else:
    print("No data available to plot!")

# %%
# Plot 2: NORMALIZED ELBO difference (Vanilla+Lapse - Vanilla) for ALL animals
if len(all_lapse_probs) > 0:
    # Filter data based on batches_to_exclude (same as unnormalized plot)
    if len(batches_to_exclude) > 0:
        plot_indices = [i for i, b in enumerate(all_batches) if b not in batches_to_exclude]
        plot_lapse_probs = [all_lapse_probs[i] for i in plot_indices]
        plot_elbo_diffs = [all_elbo_differences[i] for i in plot_indices]
        plot_batches = [all_batches[i] for i in plot_indices]
        excluded_str = ', '.join(batches_to_exclude)
    else:
        plot_lapse_probs = all_lapse_probs
        plot_elbo_diffs = all_elbo_differences
        plot_batches = all_batches
        excluded_str = 'None'
    
    # Get unique batches for color mapping (use all batches for consistent colors)
    unique_batches_all = sorted(set(all_batches))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_batches_all)))
    batch_to_color = {batch: colors[i] for i, batch in enumerate(unique_batches_all)}
    
    # Get unique batches in plot data
    unique_batches = sorted(set(plot_batches))
    
    # Normalize ELBO differences to [-1, 1] range
    elbo_min = min(plot_elbo_diffs)
    elbo_max = max(plot_elbo_diffs)
    elbo_range = elbo_max - elbo_min
    
    if elbo_range > 0:
        # Map to [-1, 1]: normalized = 2 * (value - min) / (max - min) - 1
        plot_elbo_diffs_normalized = [2 * (ed - elbo_min) / elbo_range - 1 for ed in plot_elbo_diffs]
    else:
        plot_elbo_diffs_normalized = [0] * len(plot_elbo_diffs)
    
    print(f"\nELBO difference normalization:")
    print(f"  Original range: [{elbo_min:.2f}, {elbo_max:.2f}]")
    print(f"  Normalized range: [{min(plot_elbo_diffs_normalized):.3f}, {max(plot_elbo_diffs_normalized):.3f}]")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Scatter plot with different colors for each batch
    for batch in unique_batches:
        batch_mask = [b == batch for b in plot_batches]
        batch_lapse_probs = [lp for lp, m in zip(plot_lapse_probs, batch_mask) if m]
        batch_elbo_diffs_norm = [ed for ed, m in zip(plot_elbo_diffs_normalized, batch_mask) if m]
        
        ax.scatter(batch_lapse_probs, batch_elbo_diffs_norm, 
                  s=100, alpha=0.6, color=batch_to_color[batch], 
                  edgecolor='black', linewidth=1, label=batch)
    
    # Add overall trend line
    if len(plot_lapse_probs) >= 2:
        z = np.polyfit(plot_lapse_probs, plot_elbo_diffs_normalized, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(plot_lapse_probs), max(plot_lapse_probs), 100)
        ax.plot(x_line, p(x_line), "--", color='red', linewidth=3, alpha=0.7, 
               label=f'Linear fit: y={z[0]:.2f}x + {z[1]:.2f}')
    
    # Formatting
    ax.set_xlabel('Lapse Probability (Vanilla+Lapse Model)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized ELBO Difference\n(Vanilla+Lapse - Vanilla)', fontsize=14, fontweight='bold')
    title_str = 'Hypothesis: Higher Lapse Probability → Larger ELBO Improvement\n'
    if len(batches_to_exclude) > 0:
        title_str += f'All Animals (Excluded: {excluded_str})'
    else:
        title_str += 'All Animals Across All Batches'
    ax.set_title(title_str, fontsize=15, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.set_ylim(-1.1, 1.1)  # Set y-axis limits for normalized data
    
    # Add text box with correlation
    if len(plot_lapse_probs) >= 2:
        correlation = np.corrcoef(plot_lapse_probs, plot_elbo_diffs_normalized)[0, 1]
        textstr = f'Correlation: {correlation:.3f}\nn = {len(plot_lapse_probs)} animals\n{len(unique_batches)} batches'
        if len(batches_to_exclude) > 0:
            textstr += f'\nExcluded: {excluded_str}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    filename_suffix = '_excluded_' + '_'.join(batches_to_exclude) if len(batches_to_exclude) > 0 else ''
    hypothesis_all_png = os.path.join(output_dir, f'hypothesis_lapse_prob_vs_elbo_diff_ALL_ANIMALS_NORMALIZED{filename_suffix}.png')
    fig.savefig(hypothesis_all_png, dpi=300, bbox_inches='tight')
    print(f"\nSaved normalized: {hypothesis_all_png}")
    plt.show()
    
    # Print summary statistics by batch
    print(f"\n{'='*70}")
    excluded_info = f' (Excluded: {excluded_str})' if len(batches_to_exclude) > 0 else ''
    print(f"SUMMARY BY BATCH{excluded_info}")
    print(f"{'='*70}")
    for batch in unique_batches:
        batch_mask = [b == batch for b in plot_batches]
        batch_lapse_probs = [lp for lp, m in zip(plot_lapse_probs, batch_mask) if m]
        batch_elbo_diffs_norm = [ed for ed, m in zip(plot_elbo_diffs_normalized, batch_mask) if m]
        
        if len(batch_lapse_probs) >= 2:
            batch_corr = np.corrcoef(batch_lapse_probs, batch_elbo_diffs_norm)[0, 1]
            print(f"{batch:<15} n={len(batch_lapse_probs):<3} Correlation: {batch_corr:+.3f}")
        else:
            print(f"{batch:<15} n={len(batch_lapse_probs):<3} (insufficient data for correlation)")
    print(f"{'='*70}")
    
else:
    print("No data available to plot!")

# %%
# =============================================================================
# HYPOTHESIS: Vanilla+Lapse - Norm ELBO Difference vs Lapse Probability
# =============================================================================

print(f"\n{'='*70}")
print(f"TESTING HYPOTHESIS: (Vanilla+Lapse - Norm) ELBO vs Lapse Prob")
print(f"{'='*70}\n")

# Storage for Vanilla+Lapse vs Norm comparison
all_lapse_probs_norm = []
all_elbo_diff_vs_norm = []
all_batches_norm = []
all_animals_norm = []

for pkl_path in vanilla_lapse_files:
    filename = os.path.basename(pkl_path)
    
    try:
        name_parts = filename.replace('.pkl', '').split('_')
        batch_idx = name_parts.index('batch') + 1
        animal_idx = name_parts.index('animal') + 1
        
        # Extract batch
        batch_parts = []
        for i in range(batch_idx, len(name_parts)):
            if name_parts[i] == 'animal':
                break
            batch_parts.append(name_parts[i])
        batch = '_'.join(batch_parts)
        animal = int(name_parts[animal_idx])
        
        print(f"Processing {batch} - Animal {animal}...")
        
        # Load Vanilla+Lapse VBMC object
        with open(pkl_path, 'rb') as f:
            vbmc_vanilla_lapse = pickle.load(f)
            vp_obj = vbmc_vanilla_lapse.vp
            
            # Extract ELBO from iteration history
            iter_hist = vbmc_vanilla_lapse.iteration_history
            if 'elbo' in iter_hist:
                elbo_vanilla_lapse = float(iter_hist['elbo'][-1])
            else:
                print(f"  Warning: No ELBO found, skipping...")
                continue
            
            # Sample to get lapse_prob
            vp_samples_temp = vp_obj.sample(int(1e6))[0]
            lapse_prob = np.mean(vp_samples_temp[:, 6])
        
        # Load Norm model ELBO from results pkl
        results_pkl = f'results_{batch}_animal_{animal}.pkl'
        if not os.path.exists(results_pkl):
            print(f"  Warning: Results file not found, skipping...")
            continue
            
        with open(results_pkl, 'rb') as f:
            results_data = pickle.load(f)
            
            if 'vbmc_norm_tied_results' in results_data:
                norm_tied_results = results_data['vbmc_norm_tied_results']
                if 'elbo' in norm_tied_results:
                    elbo_norm = norm_tied_results['elbo']
                else:
                    print(f"  Warning: No norm ELBO found, skipping...")
                    continue
            else:
                print(f"  Warning: No norm tied results found, skipping...")
                continue
        
        # Compute ELBO difference (Vanilla+Lapse - Norm)
        elbo_diff = elbo_vanilla_lapse - elbo_norm
        
        # Store results
        all_lapse_probs_norm.append(lapse_prob)
        all_elbo_diff_vs_norm.append(elbo_diff)
        all_batches_norm.append(batch)
        all_animals_norm.append(animal)
        
        print(f"  Lapse prob: {lapse_prob:.6f}")
        print(f"  ELBO diff (V+L - Norm): {elbo_diff:.2f}\n")
        
    except Exception as e:
        print(f"  Error processing {filename}: {e}\n")
        continue

print(f"\nSuccessfully processed {len(all_lapse_probs_norm)} animals")

# %%
# Plot: UNNORMALIZED ELBO difference (Vanilla+Lapse - Norm) vs Lapse Prob
batches_to_exclude_norm = ['LED34', 'LED7']  # List of batches to exclude

if len(all_lapse_probs_norm) > 0:
    # Filter data based on batches_to_exclude_norm
    if len(batches_to_exclude_norm) > 0:
        plot_indices = [i for i, b in enumerate(all_batches_norm) if b not in batches_to_exclude_norm]
        plot_lapse_probs = [all_lapse_probs_norm[i] for i in plot_indices]
        plot_elbo_diffs = [all_elbo_diff_vs_norm[i] for i in plot_indices]
        plot_batches = [all_batches_norm[i] for i in plot_indices]
        excluded_str = ', '.join(batches_to_exclude_norm)
    else:
        plot_lapse_probs = all_lapse_probs_norm
        plot_elbo_diffs = all_elbo_diff_vs_norm
        plot_batches = all_batches_norm
        excluded_str = 'None'
    
    # Get unique batches for color mapping
    unique_batches_all = sorted(set(all_batches_norm))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_batches_all)))
    batch_to_color = {batch: colors[i] for i, batch in enumerate(unique_batches_all)}
    
    unique_batches = sorted(set(plot_batches))
    
    # UNNORMALIZED PLOT
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for batch in unique_batches:
        batch_mask = [b == batch for b in plot_batches]
        batch_lapse_probs = [lp for lp, m in zip(plot_lapse_probs, batch_mask) if m]
        batch_elbo_diffs = [ed for ed, m in zip(plot_elbo_diffs, batch_mask) if m]
        
        ax.scatter(batch_lapse_probs, batch_elbo_diffs, 
                  s=100, alpha=0.6, color=batch_to_color[batch], 
                  edgecolor='black', linewidth=1, label=batch)
    
    # Add trend line
    if len(plot_lapse_probs) >= 2:
        z = np.polyfit(plot_lapse_probs, plot_elbo_diffs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(plot_lapse_probs), max(plot_lapse_probs), 100)
        ax.plot(x_line, p(x_line), "--", color='red', linewidth=3, alpha=0.7, 
               label=f'Linear fit: y={z[0]:.1f}x + {z[1]:.1f}')
    
    # Formatting
    ax.set_xlabel('Lapse Probability (Vanilla+Lapse Model)', fontsize=14, fontweight='bold')
    ax.set_ylabel('ELBO Difference\n(Vanilla+Lapse - Norm)', fontsize=14, fontweight='bold')
    title_str = 'Hypothesis: Higher Lapse Probability → Larger ELBO Improvement\n'
    if len(batches_to_exclude_norm) > 0:
        title_str += f'Vanilla+Lapse vs Norm (Excluded: {excluded_str})'
    else:
        title_str += 'Vanilla+Lapse vs Norm (All Animals)'
    ax.set_title(title_str, fontsize=15, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    
    # Add text box with correlation
    if len(plot_lapse_probs) >= 2:
        correlation = np.corrcoef(plot_lapse_probs, plot_elbo_diffs)[0, 1]
        textstr = f'Correlation: {correlation:.3f}\nn = {len(plot_lapse_probs)} animals\n{len(unique_batches)} batches'
        if len(batches_to_exclude_norm) > 0:
            textstr += f'\nExcluded: {excluded_str}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    filename_suffix = '_excluded_' + '_'.join(batches_to_exclude_norm) if len(batches_to_exclude_norm) > 0 else ''
    fig.savefig(os.path.join(output_dir, f'hypothesis_vanilla_lapse_vs_norm_UNNORMALIZED{filename_suffix}.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved unnormalized plot: hypothesis_vanilla_lapse_vs_norm_UNNORMALIZED{filename_suffix}.png")
    plt.show()
    
    # Print summary
    print(f"\n{'='*70}")
    excluded_info = f' (Excluded: {excluded_str})' if len(batches_to_exclude_norm) > 0 else ''
    print(f"SUMMARY: Vanilla+Lapse vs Norm{excluded_info}")
    print(f"{'='*70}")
    for batch in unique_batches:
        batch_mask = [b == batch for b in plot_batches]
        batch_lapse_probs = [lp for lp, m in zip(plot_lapse_probs, batch_mask) if m]
        batch_elbo_diffs = [ed for ed, m in zip(plot_elbo_diffs, batch_mask) if m]
        
        if len(batch_lapse_probs) >= 2:
            batch_corr = np.corrcoef(batch_lapse_probs, batch_elbo_diffs)[0, 1]
            print(f"{batch:<15} n={len(batch_lapse_probs):<3} Correlation: {batch_corr:+.3f}")
        else:
            print(f"{batch:<15} n={len(batch_lapse_probs):<3} (insufficient data)")
    print(f"{'='*70}")
    
else:
    print("No data available!")
# %%
