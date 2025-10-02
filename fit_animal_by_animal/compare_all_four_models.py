# Compare all four models: Vanilla, Vanilla+Lapse, Norm, Norm+Lapse
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import sys
sys.path.append('../lapses')
from lapses_utils import simulate_psiam_tied_rate_norm

# %%
# Configuration
batch_name = 'LED8'
animal = 105
DO_RIGHT_TRUNCATE = True
T_trunc = 0.3

print(f'Batch: {batch_name}, Animal: {animal}')
print(f'DO_RIGHT_TRUNCATE: {DO_RIGHT_TRUNCATE}')
print(f'T_trunc: {T_trunc}')

# %%
# Load experimental data
csv_filename = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
exp_df = pd.read_csv(csv_filename)

df_valid_and_aborts = exp_df[
    (exp_df['success'].isin([1,-1])) |
    (exp_df['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]
df_valid = df_valid_and_aborts[df_valid_and_aborts['success'].isin([1, -1])]

# Filter for specific animal
df_valid_animal = df_valid[df_valid['animal'] == animal].copy()
df_aborts_animal = df_aborts[df_aborts['animal'] == animal].copy()

# Apply right truncation if needed
if DO_RIGHT_TRUNCATE:
    df_valid_animal_filtered = df_valid_animal[df_valid_animal['RTwrtStim'] < 1].copy()
    max_rt = 1
    print(f'Applied right truncation at 1s')
else:
    df_valid_animal_filtered = df_valid_animal.copy()
    max_rt = df_valid_animal['RTwrtStim'].max()

print(f'Valid trials for animal {animal}: {len(df_valid_animal_filtered)}')

# %%
# Load baseline models (Vanilla and Norm) from main results file
pkl_file = f'results_{batch_name}_animal_{animal}.pkl'
with open(pkl_file, 'rb') as f:
    fit_results_data = pickle.load(f)

# Load abort parameters
abort_samples = fit_results_data['vbmc_aborts_results']
V_A = np.mean(abort_samples['V_A_samples'])
theta_A = np.mean(abort_samples['theta_A_samples'])
t_A_aff = np.mean(abort_samples['t_A_aff_samp'])

print(f'\\nAbort parameters loaded:')
print(f'V_A: {V_A:.4f}, theta_A: {theta_A:.4f}, t_A_aff: {t_A_aff:.4f}')

# %%
# Extract VANILLA model parameters
vanilla_tied_samples = fit_results_data['vbmc_vanilla_tied_results']
vanilla_rate_lambda = np.mean(vanilla_tied_samples['rate_lambda_samples'])
vanilla_T_0 = np.mean(vanilla_tied_samples['T_0_samples'])
vanilla_theta_E = np.mean(vanilla_tied_samples['theta_E_samples'])
vanilla_w = np.mean(vanilla_tied_samples['w_samples'])
vanilla_t_E_aff = np.mean(vanilla_tied_samples['t_E_aff_samples'])
vanilla_del_go = np.mean(vanilla_tied_samples['del_go_samples'])
vanilla_Z_E = (vanilla_w - 0.5) * 2 * vanilla_theta_E

print(f'\\n=== VANILLA MODEL ===')
print(f'rate_lambda: {vanilla_rate_lambda:.6f}')
print(f'T_0 (ms): {vanilla_T_0*1000:.6f}')
print(f'theta_E: {vanilla_theta_E:.6f}')
print(f'w: {vanilla_w:.6f}')
print(f'Z_E: {vanilla_Z_E:.6f}')
print(f't_E_aff (ms): {vanilla_t_E_aff*1000:.6f}')
print(f'del_go (ms): {vanilla_del_go*1000:.6f}')

# %%
# Extract NORM model parameters
norm_tied_samples = fit_results_data['vbmc_norm_tied_results']
norm_rate_lambda = np.mean(norm_tied_samples['rate_lambda_samples'])
norm_T_0 = np.mean(norm_tied_samples['T_0_samples'])
norm_theta_E = np.mean(norm_tied_samples['theta_E_samples'])
norm_w = np.mean(norm_tied_samples['w_samples'])
norm_t_E_aff = np.mean(norm_tied_samples['t_E_aff_samples'])
norm_del_go = np.mean(norm_tied_samples['del_go_samples'])
norm_rate_norm_l = np.mean(norm_tied_samples['rate_norm_l_samples'])
norm_Z_E = (norm_w - 0.5) * 2 * norm_theta_E

print(f'\\n=== NORM MODEL ===')
print(f'rate_lambda: {norm_rate_lambda:.6f}')
print(f'T_0 (ms): {norm_T_0*1000:.6f}')
print(f'theta_E: {norm_theta_E:.6f}')
print(f'w: {norm_w:.6f}')
print(f'Z_E: {norm_Z_E:.6f}')
print(f't_E_aff (ms): {norm_t_E_aff*1000:.6f}')
print(f'del_go (ms): {norm_del_go*1000:.6f}')
print(f'rate_norm_l: {norm_rate_norm_l:.6f}')

# %%
# Load VANILLA + LAPSE model
vanilla_lapse_pkl = f'vbmc_vanilla_tied_results_batch_{batch_name}_animal_{animal}_lapses_truncate_1s.pkl'
with open(vanilla_lapse_pkl, 'rb') as f:
    vanilla_lapse_vbmc = pickle.load(f)

vanilla_lapse_vp = vanilla_lapse_vbmc.vp
vanilla_lapse_samples_full = vanilla_lapse_vp.sample(int(1e6))[0]

vanilla_lapse_rate_lambda = np.mean(vanilla_lapse_samples_full[:, 0])
vanilla_lapse_T_0 = np.mean(vanilla_lapse_samples_full[:, 1])
vanilla_lapse_theta_E = np.mean(vanilla_lapse_samples_full[:, 2])
vanilla_lapse_w = np.mean(vanilla_lapse_samples_full[:, 3])
vanilla_lapse_t_E_aff = np.mean(vanilla_lapse_samples_full[:, 4])
vanilla_lapse_del_go = np.mean(vanilla_lapse_samples_full[:, 5])
vanilla_lapse_lapse_prob = np.mean(vanilla_lapse_samples_full[:, 6])
vanilla_lapse_lapse_prob_right = np.mean(vanilla_lapse_samples_full[:, 7])
vanilla_lapse_Z_E = (vanilla_lapse_w - 0.5) * 2 * vanilla_lapse_theta_E

print(f'\\n=== VANILLA + LAPSE MODEL ===')
print(f'rate_lambda: {vanilla_lapse_rate_lambda:.6f}')
print(f'T_0 (ms): {vanilla_lapse_T_0*1000:.6f}')
print(f'theta_E: {vanilla_lapse_theta_E:.6f}')
print(f'w: {vanilla_lapse_w:.6f}')
print(f'Z_E: {vanilla_lapse_Z_E:.6f}')
print(f't_E_aff (ms): {vanilla_lapse_t_E_aff*1000:.6f}')
print(f'del_go (ms): {vanilla_lapse_del_go*1000:.6f}')
print(f'lapse_prob: {vanilla_lapse_lapse_prob:.6f}')
print(f'lapse_prob_right: {vanilla_lapse_lapse_prob_right:.6f}')

# %%
# Load NORM + LAPSE model
norm_lapse_pkl = f'vbmc_norm_tied_results_batch_{batch_name}_animal_{animal}_lapses_truncate_1s.pkl'
with open(norm_lapse_pkl, 'rb') as f:
    norm_lapse_vbmc = pickle.load(f)

norm_lapse_vp = norm_lapse_vbmc.vp
norm_lapse_samples_full = norm_lapse_vp.sample(int(1e6))[0]

norm_lapse_rate_lambda = np.mean(norm_lapse_samples_full[:, 0])
norm_lapse_T_0 = np.mean(norm_lapse_samples_full[:, 1])
norm_lapse_theta_E = np.mean(norm_lapse_samples_full[:, 2])
norm_lapse_w = np.mean(norm_lapse_samples_full[:, 3])
norm_lapse_t_E_aff = np.mean(norm_lapse_samples_full[:, 4])
norm_lapse_del_go = np.mean(norm_lapse_samples_full[:, 5])
norm_lapse_rate_norm_l = np.mean(norm_lapse_samples_full[:, 6])
norm_lapse_lapse_prob = np.mean(norm_lapse_samples_full[:, 7])
norm_lapse_lapse_prob_right = np.mean(norm_lapse_samples_full[:, 8])
norm_lapse_Z_E = (norm_lapse_w - 0.5) * 2 * norm_lapse_theta_E

print(f'\\n=== NORM + LAPSE MODEL ===')
print(f'rate_lambda: {norm_lapse_rate_lambda:.6f}')
print(f'T_0 (ms): {norm_lapse_T_0*1000:.6f}')
print(f'theta_E: {norm_lapse_theta_E:.6f}')
print(f'w: {norm_lapse_w:.6f}')
print(f'Z_E: {norm_lapse_Z_E:.6f}')
print(f't_E_aff (ms): {norm_lapse_t_E_aff*1000:.6f}')
print(f'del_go (ms): {norm_lapse_del_go*1000:.6f}')
print(f'rate_norm_l: {norm_lapse_rate_norm_l:.6f}')
print(f'lapse_prob: {norm_lapse_lapse_prob:.6f}')
print(f'lapse_prob_right: {norm_lapse_lapse_prob_right:.6f}')

# %%
# Comprehensive Parameter Comparison Table
print(f'\\n\\n{"="*120}')
print(f'COMPREHENSIVE PARAMETER COMPARISON - All Four Models')
print(f'Batch: {batch_name}, Animal: {animal}')
print(f'{"="*120}')
print(f'{"Parameter":<20} {"Vanilla":<18} {"Vanilla+Lapse":<18} {"Norm":<18} {"Norm+Lapse":<18} {"Units"}')
print(f'{"-"*120}')
print(f'{"rate_lambda":<20} {vanilla_rate_lambda:<18.6f} {vanilla_lapse_rate_lambda:<18.6f} {norm_rate_lambda:<18.6f} {norm_lapse_rate_lambda:<18.6f} {""}')
print(f'{"T_0":<20} {vanilla_T_0*1000:<18.6f} {vanilla_lapse_T_0*1000:<18.6f} {norm_T_0*1000:<18.6f} {norm_lapse_T_0*1000:<18.6f} {"(ms)"}')
print(f'{"theta_E":<20} {vanilla_theta_E:<18.6f} {vanilla_lapse_theta_E:<18.6f} {norm_theta_E:<18.6f} {norm_lapse_theta_E:<18.6f} {""}')
print(f'{"w":<20} {vanilla_w:<18.6f} {vanilla_lapse_w:<18.6f} {norm_w:<18.6f} {norm_lapse_w:<18.6f} {""}')
print(f'{"Z_E":<20} {vanilla_Z_E:<18.6f} {vanilla_lapse_Z_E:<18.6f} {norm_Z_E:<18.6f} {norm_lapse_Z_E:<18.6f} {""}')
print(f'{"t_E_aff":<20} {vanilla_t_E_aff*1000:<18.6f} {vanilla_lapse_t_E_aff*1000:<18.6f} {norm_t_E_aff*1000:<18.6f} {norm_lapse_t_E_aff*1000:<18.6f} {"(ms)"}')
print(f'{"del_go":<20} {vanilla_del_go*1000:<18.6f} {vanilla_lapse_del_go*1000:<18.6f} {norm_del_go*1000:<18.6f} {norm_lapse_del_go*1000:<18.6f} {"(ms)"}')
print(f'{"rate_norm_l":<20} {"N/A":<18} {"N/A":<18} {norm_rate_norm_l:<18.6f} {norm_lapse_rate_norm_l:<18.6f} {""}')
print(f'{"lapse_prob":<20} {"N/A":<18} {vanilla_lapse_lapse_prob:<18.6f} {"N/A":<18} {norm_lapse_lapse_prob:<18.6f} {""}')
print(f'{"lapse_prob_right":<20} {"N/A":<18} {vanilla_lapse_lapse_prob_right:<18.6f} {"N/A":<18} {norm_lapse_lapse_prob_right:<18.6f} {""}')
print(f'{"="*120}')

# %%
# Simulation parameters
N_sim = int(1e6)
dt = 1e-4
T_lapse_max = max_rt

print(f'\\n\\nStarting simulations with N_sim = {N_sim:,}')

# Sample trial conditions from empirical data
np.random.seed(42)
trial_indices = np.random.choice(len(df_valid_animal_filtered), N_sim, replace=True)
ABL_samples = df_valid_animal_filtered['ABL'].values[trial_indices]
ILD_samples = df_valid_animal_filtered['ILD'].values[trial_indices]
t_stim_samples = df_valid_animal_filtered['intended_fix'].values[trial_indices]

# %%
# Simulation functions for all four models
def simulate_vanilla(i):
    choice, rt, is_act = simulate_psiam_tied_rate_norm(
        V_A, theta_A, ABL_samples[i], ILD_samples[i],
        vanilla_rate_lambda, vanilla_T_0, vanilla_theta_E, vanilla_Z_E,
        t_stim_samples[i], t_A_aff, vanilla_t_E_aff, vanilla_del_go,
        0.0, dt, lapse_prob=0.0, T_lapse_max=T_lapse_max
    )
    return {
        'choice': choice, 'rt': rt, 'is_act': is_act,
        'ABL': ABL_samples[i], 'ILD': ILD_samples[i], 't_stim': t_stim_samples[i]
    }

def simulate_vanilla_lapse(i):
    choice, rt, is_act = simulate_psiam_tied_rate_norm(
        V_A, theta_A, ABL_samples[i], ILD_samples[i],
        vanilla_lapse_rate_lambda, vanilla_lapse_T_0, vanilla_lapse_theta_E, vanilla_lapse_Z_E,
        t_stim_samples[i], t_A_aff, vanilla_lapse_t_E_aff, vanilla_lapse_del_go,
        0.0, dt, lapse_prob=vanilla_lapse_lapse_prob, T_lapse_max=T_lapse_max,
        lapse_prob_right=vanilla_lapse_lapse_prob_right
    )
    return {
        'choice': choice, 'rt': rt, 'is_act': is_act,
        'ABL': ABL_samples[i], 'ILD': ILD_samples[i], 't_stim': t_stim_samples[i]
    }

def simulate_norm(i):
    choice, rt, is_act = simulate_psiam_tied_rate_norm(
        V_A, theta_A, ABL_samples[i], ILD_samples[i],
        norm_rate_lambda, norm_T_0, norm_theta_E, norm_Z_E,
        t_stim_samples[i], t_A_aff, norm_t_E_aff, norm_del_go,
        norm_rate_norm_l, dt, lapse_prob=0.0, T_lapse_max=T_lapse_max
    )
    return {
        'choice': choice, 'rt': rt, 'is_act': is_act,
        'ABL': ABL_samples[i], 'ILD': ILD_samples[i], 't_stim': t_stim_samples[i]
    }

def simulate_norm_lapse(i):
    choice, rt, is_act = simulate_psiam_tied_rate_norm(
        V_A, theta_A, ABL_samples[i], ILD_samples[i],
        norm_lapse_rate_lambda, norm_lapse_T_0, norm_lapse_theta_E, norm_lapse_Z_E,
        t_stim_samples[i], t_A_aff, norm_lapse_t_E_aff, norm_lapse_del_go,
        norm_lapse_rate_norm_l, dt, lapse_prob=norm_lapse_lapse_prob, T_lapse_max=T_lapse_max,
        lapse_prob_right=norm_lapse_lapse_prob_right
    )
    return {
        'choice': choice, 'rt': rt, 'is_act': is_act,
        'ABL': ABL_samples[i], 'ILD': ILD_samples[i], 't_stim': t_stim_samples[i]
    }

# %%
# Run simulations
print('Simulating VANILLA model...')
vanilla_sim_results = Parallel(n_jobs=-2, verbose=5)(
    delayed(simulate_vanilla)(i) for i in tqdm(range(N_sim))
)

print('\\nSimulating VANILLA+LAPSE model...')
vanilla_lapse_sim_results = Parallel(n_jobs=-2, verbose=5)(
    delayed(simulate_vanilla_lapse)(i) for i in tqdm(range(N_sim))
)

print('\\nSimulating NORM model...')
norm_sim_results = Parallel(n_jobs=-2, verbose=5)(
    delayed(simulate_norm)(i) for i in tqdm(range(N_sim))
)

print('\\nSimulating NORM+LAPSE model...')
norm_lapse_sim_results = Parallel(n_jobs=-2, verbose=5)(
    delayed(simulate_norm_lapse)(i) for i in tqdm(range(N_sim))
)

# Convert to DataFrames
vanilla_sim_df = pd.DataFrame(vanilla_sim_results)
vanilla_lapse_sim_df = pd.DataFrame(vanilla_lapse_sim_results)
norm_sim_df = pd.DataFrame(norm_sim_results)
norm_lapse_sim_df = pd.DataFrame(norm_lapse_sim_results)

# Compute RT relative to stimulus
vanilla_sim_df['RTwrtStim'] = vanilla_sim_df['rt'] - vanilla_sim_df['t_stim']
vanilla_lapse_sim_df['RTwrtStim'] = vanilla_lapse_sim_df['rt'] - vanilla_lapse_sim_df['t_stim']
norm_sim_df['RTwrtStim'] = norm_sim_df['rt'] - norm_sim_df['t_stim']
norm_lapse_sim_df['RTwrtStim'] = norm_lapse_sim_df['rt'] - norm_lapse_sim_df['t_stim']

# Filter if truncation is enabled
if DO_RIGHT_TRUNCATE:
    vanilla_sim_df = vanilla_sim_df[vanilla_sim_df['RTwrtStim'] < 1].copy()
    vanilla_lapse_sim_df = vanilla_lapse_sim_df[vanilla_lapse_sim_df['RTwrtStim'] < 1].copy()
    norm_sim_df = norm_sim_df[norm_sim_df['RTwrtStim'] < 1].copy()
    norm_lapse_sim_df = norm_lapse_sim_df[norm_lapse_sim_df['RTwrtStim'] < 1].copy()

print(f'\\nSimulation complete!')
print(f'Vanilla: {len(vanilla_sim_df):,} trials')
print(f'Vanilla+Lapse: {len(vanilla_lapse_sim_df):,} trials')
print(f'Norm: {len(norm_sim_df):,} trials')
print(f'Norm+Lapse: {len(norm_lapse_sim_df):,} trials')

# %%
# Plot 1: RT Distributions by ABL and Absolute ILD
ABL_arr = np.sort(df_valid_animal_filtered['ABL'].unique())
ILD_arr = np.sort(df_valid_animal_filtered['ILD'].unique())
abs_ILD_arr = np.sort(np.unique(np.abs(df_valid_animal_filtered['ILD'])))

# Add abs_ILD column to all dataframes
vanilla_sim_df['abs_ILD'] = np.abs(vanilla_sim_df['ILD'])
vanilla_lapse_sim_df['abs_ILD'] = np.abs(vanilla_lapse_sim_df['ILD'])
norm_sim_df['abs_ILD'] = np.abs(norm_sim_df['ILD'])
norm_lapse_sim_df['abs_ILD'] = np.abs(norm_lapse_sim_df['ILD'])
df_valid_animal_filtered['abs_ILD'] = np.abs(df_valid_animal_filtered['ILD'])

n_abl = len(ABL_arr)
n_abs_ild = len(abs_ILD_arr)

fig, axes = plt.subplots(n_abl, n_abs_ild, figsize=(4*n_abs_ild, 3*n_abl), 
                         sharex=True, sharey=True)
if n_abl == 1 and n_abs_ild == 1:
    axes = np.array([[axes]])
elif n_abl == 1:
    axes = axes.reshape(1, -1)
elif n_abs_ild == 1:
    axes = axes.reshape(-1, 1)

bins = np.arange(0, 1 if DO_RIGHT_TRUNCATE else max_rt, 0.01)

for row_idx, abl in enumerate(ABL_arr):
    for col_idx, abs_ild in enumerate(abs_ILD_arr):
        ax = axes[row_idx, col_idx]
        
        # Filter data for this ABL and absolute ILD
        vanilla_subset = vanilla_sim_df[(vanilla_sim_df['ABL'] == abl) & (vanilla_sim_df['abs_ILD'] == abs_ild)]
        vanilla_lapse_subset = vanilla_lapse_sim_df[(vanilla_lapse_sim_df['ABL'] == abl) & (vanilla_lapse_sim_df['abs_ILD'] == abs_ild)]
        norm_subset = norm_sim_df[(norm_sim_df['ABL'] == abl) & (norm_sim_df['abs_ILD'] == abs_ild)]
        norm_lapse_subset = norm_lapse_sim_df[(norm_lapse_sim_df['ABL'] == abl) & (norm_lapse_sim_df['abs_ILD'] == abs_ild)]
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['abs_ILD'] == abs_ild)
        ]
        
        # Plot as step histograms
        ax.hist(vanilla_subset['RTwrtStim'], bins=bins, density=True, 
                histtype='step', color='blue',label='Vanilla', alpha=0.8)
        ax.hist(vanilla_lapse_subset['RTwrtStim'], bins=bins, density=True, 
                histtype='step', color='red',  label='Vanilla+Lapse', alpha=0.8)
        ax.hist(norm_subset['RTwrtStim'], bins=bins, density=True, 
                histtype='step', color='green',  label='Norm', alpha=0.8)
        ax.hist(norm_lapse_subset['RTwrtStim'], bins=bins, density=True, 
                histtype='step', color='orange',  label='Norm+Lapse', alpha=0.8)
        ax.hist(empirical_subset['RTwrtStim'], bins=bins, density=True, 
                histtype='step', color='black', linewidth=2, label='Data', alpha=0.4)
        
        # Titles and labels
        if row_idx == 0:
            ax.set_title(f'|ILD| = {abs_ild} dB', fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(f'ABL={abl} dB\\nDensity', fontsize=10)
        if row_idx == n_abl - 1:
            ax.set_xlabel('RT w.r.t. Stim (s)', fontsize=10)
        
        # Legend only on first subplot
        if row_idx == 0 and col_idx == n_abs_ild - 1:
            ax.legend(fontsize=8, loc='upper right')
        
        ax.grid(True, alpha=0.3)

plt.suptitle(f'RT Distributions by ABL and |ILD| - All Four Models\\nBatch {batch_name}, Animal {animal}', 
             fontsize=13, y=0.995)
plt.tight_layout()
plt.show()

# %%
# Plot 2: Psychometric Curves by ABL
ABL_arr = np.sort(df_valid_animal_filtered['ABL'].unique())
ILD_arr = np.sort(df_valid_animal_filtered['ILD'].unique())

fig, axes = plt.subplots(1, len(ABL_arr), figsize=(5*len(ABL_arr), 4), sharey=True)
if len(ABL_arr) == 1:
    axes = [axes]

for idx, abl in enumerate(ABL_arr):
    ax = axes[idx]
    
    # Compute P(right) for each ILD for all four models
    ILD_vals = []
    vanilla_p_right = []
    vanilla_lapse_p_right = []
    norm_p_right = []
    norm_lapse_p_right = []
    empirical_p_right = []
    
    for ild in ILD_arr:
        ILD_vals.append(ild)
        
        # Vanilla
        vanilla_subset = vanilla_sim_df[(vanilla_sim_df['ABL'] == abl) & (vanilla_sim_df['ILD'] == ild)]
        vanilla_p_right.append(np.mean(vanilla_subset['choice'] == 1) if len(vanilla_subset) > 0 else np.nan)
        
        # Vanilla+Lapse
        vanilla_lapse_subset = vanilla_lapse_sim_df[(vanilla_lapse_sim_df['ABL'] == abl) & (vanilla_lapse_sim_df['ILD'] == ild)]
        vanilla_lapse_p_right.append(np.mean(vanilla_lapse_subset['choice'] == 1) if len(vanilla_lapse_subset) > 0 else np.nan)
        
        # Norm
        norm_subset = norm_sim_df[(norm_sim_df['ABL'] == abl) & (norm_sim_df['ILD'] == ild)]
        norm_p_right.append(np.mean(norm_subset['choice'] == 1) if len(norm_subset) > 0 else np.nan)
        
        # Norm+Lapse
        norm_lapse_subset = norm_lapse_sim_df[(norm_lapse_sim_df['ABL'] == abl) & (norm_lapse_sim_df['ILD'] == ild)]
        norm_lapse_p_right.append(np.mean(norm_lapse_subset['choice'] == 1) if len(norm_lapse_subset) > 0 else np.nan)
        
        # Empirical
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        empirical_p_right.append(np.mean(empirical_subset['choice'] == 1) if len(empirical_subset) > 0 else np.nan)
    
    # Plot
    ax.plot(ILD_vals, vanilla_p_right, 'o-', color='blue', markersize=6, label='Vanilla', alpha=0.7)
    ax.plot(ILD_vals, vanilla_lapse_p_right, 'x-', color='red', markersize=8, markeredgewidth=2, label='Vanilla+Lapse', alpha=0.7)
    ax.plot(ILD_vals, norm_p_right, 's-', color='green', markersize=6, label='Norm', alpha=0.7)
    ax.plot(ILD_vals, norm_lapse_p_right, 'd-', color='orange', markersize=6, label='Norm+Lapse', alpha=0.7)
    ax.plot(ILD_vals, empirical_p_right, 'k^-', markersize=8, label='Data', linewidth=2, alpha=0.5)
    
    ax.set_title(f'ABL = {abl} dB', fontsize=12)
    ax.set_xlabel('ILD (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('P(choice = right)', fontsize=11)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

plt.suptitle(f'Psychometric Curves - All Four Models\\nBatch {batch_name}, Animal {animal}', fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Plot 3: Log Odds Comparison
fig, axes = plt.subplots(1, len(ABL_arr), figsize=(5*len(ABL_arr), 4), sharey=True)
if len(ABL_arr) == 1:
    axes = [axes]

for idx, abl in enumerate(ABL_arr):
    ax = axes[idx]
    
    # Compute log odds for each ILD
    ILD_vals = []
    vanilla_log_odds = []
    vanilla_lapse_log_odds = []
    norm_log_odds = []
    norm_lapse_log_odds = []
    empirical_log_odds = []
    
    for ild in ILD_arr:
        if ild == 0:
            continue  # Skip ILD=0 to avoid log(0)
        ILD_vals.append(ild)
        
        # Vanilla
        vanilla_subset = vanilla_sim_df[(vanilla_sim_df['ABL'] == abl) & (vanilla_sim_df['ILD'] == ild)]
        if len(vanilla_subset) > 0:
            p_right = np.mean(vanilla_subset['choice'] == 1)
            p_right = np.clip(p_right, 0.01, 0.99)
            vanilla_log_odds.append(np.log(p_right / (1 - p_right)))
        else:
            vanilla_log_odds.append(np.nan)
        
        # Vanilla+Lapse
        vanilla_lapse_subset = vanilla_lapse_sim_df[(vanilla_lapse_sim_df['ABL'] == abl) & (vanilla_lapse_sim_df['ILD'] == ild)]
        if len(vanilla_lapse_subset) > 0:
            p_right = np.mean(vanilla_lapse_subset['choice'] == 1)
            p_right = np.clip(p_right, 0.01, 0.99)
            vanilla_lapse_log_odds.append(np.log(p_right / (1 - p_right)))
        else:
            vanilla_lapse_log_odds.append(np.nan)
        
        # Norm
        norm_subset = norm_sim_df[(norm_sim_df['ABL'] == abl) & (norm_sim_df['ILD'] == ild)]
        if len(norm_subset) > 0:
            p_right = np.mean(norm_subset['choice'] == 1)
            p_right = np.clip(p_right, 0.01, 0.99)
            norm_log_odds.append(np.log(p_right / (1 - p_right)))
        else:
            norm_log_odds.append(np.nan)
        
        # Norm+Lapse
        norm_lapse_subset = norm_lapse_sim_df[(norm_lapse_sim_df['ABL'] == abl) & (norm_lapse_sim_df['ILD'] == ild)]
        if len(norm_lapse_subset) > 0:
            p_right = np.mean(norm_lapse_subset['choice'] == 1)
            p_right = np.clip(p_right, 0.01, 0.99)
            norm_lapse_log_odds.append(np.log(p_right / (1 - p_right)))
        else:
            norm_lapse_log_odds.append(np.nan)
        
        # Empirical
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right = np.mean(empirical_subset['choice'] == 1)
            p_right = np.clip(p_right, 0.01, 0.99)
            empirical_log_odds.append(np.log(p_right / (1 - p_right)))
        else:
            empirical_log_odds.append(np.nan)
    
    # Plot
    ax.plot(ILD_vals, vanilla_log_odds, 'o-', color='blue', markersize=6, label='Vanilla', alpha=0.7)
    ax.plot(ILD_vals, vanilla_lapse_log_odds, 'x-', color='red', markersize=8, markeredgewidth=2, label='Vanilla+Lapse', alpha=0.7)
    ax.plot(ILD_vals, norm_log_odds, 's-', color='green', markersize=6, label='Norm', alpha=0.7)
    ax.plot(ILD_vals, norm_lapse_log_odds, 'd-', color='orange', markersize=6, label='Norm+Lapse', alpha=0.7)
    ax.plot(ILD_vals, empirical_log_odds, 'k^-', markersize=8, label='Data', linewidth=2, alpha=0.5)
    
    ax.set_title(f'ABL = {abl} dB', fontsize=12)
    ax.set_xlabel('ILD (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('Log Odds [log(P(R)/P(L))]', fontsize=11)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

plt.suptitle(f'Log Odds - All Four Models\\nBatch {batch_name}, Animal {animal}', fontsize=14)
plt.tight_layout()
plt.show()

print('\\n\\n=== ANALYSIS COMPLETE ===')
