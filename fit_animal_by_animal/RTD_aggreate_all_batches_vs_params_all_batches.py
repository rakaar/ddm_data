# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn

# %%
bins = np.arange(0,1,0.02)
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, 'batch_csvs')

csv_files = glob.glob(os.path.join(csv_dir, 'batch_*_valid_and_aborts.csv'))

if not csv_files:
    print(f"No CSV files found in {csv_dir} matching the pattern 'batch_*_valid_and_aborts.csv'")
else:
    all_data_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            all_data_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data_list:
        print("No data could be read from the CSV files.")
    else:
        all_data = pd.concat(all_data_list, ignore_index=True)
        # Exclude LED6, LED7
        all_data = all_data[all_data['batch_name'] != 'LED6']
        all_data = all_data[all_data['batch_name'] != 'LED7']


        # Filter for valid trials (success is 1 or -1)
        valid_trials = all_data[all_data['success'].isin([1, -1])].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Filter for RTwrtStim <= 1
        # Ensure RTwrtStim is numeric, coercing errors to NaN
        valid_trials['RTwrtStim'] = pd.to_numeric(valid_trials['RTwrtStim'], errors='coerce')
        filtered_trials = valid_trials[valid_trials['RTwrtStim'] <= 1]

        if filtered_trials.empty:
            print("No trials found matching the criteria (valid and RTwrtStim <= 1).")
        else:
            # Compute abs_ILD if not present
            if 'abs_ILD' not in filtered_trials.columns:
                filtered_trials.loc[:, 'abs_ILD'] = filtered_trials['ILD'].abs().astype(float)

            # Filter for specified ABL and abs_ILD values
            ABL_values = [20, 40, 60]
            abs_ILD_values = [1., 2., 4., 8., 16.]
            filtered_stim = filtered_trials[
                filtered_trials['ABL'].isin(ABL_values) &
                filtered_trials['abs_ILD'].isin(abs_ILD_values)
            ]

            # Group by animal, ABL, abs_ILD and collect RTDs
            grouped = filtered_stim.groupby(['animal', 'ABL', 'abs_ILD'])['RTwrtStim'].apply(list)
            # You can now compute the mean RTD per group:
            mean_rtd = filtered_stim.groupby(['animal', 'ABL', 'abs_ILD'])['RTwrtStim'].mean().reset_index()

            # Example: print or use mean_rtd as needed
            print(mean_rtd)
            # If you want to store the full lists for later use:
            rtd_dict = grouped.to_dict()
            # rtd_dict[(animal, ABL, abs_ILD)] gives the list of RTDs for that combination
            print(f"Stored RTDs for {len(rtd_dict)} (animal, ABL, abs_ILD) combinations.")

# %%
# =====================
# Aggregate fit params from pickle files for aborts and vanilla_tied models
# =====================
import pickle

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34']

# Model configs: (model_key, param_keys, param_labels)
model_configs = [
    ('vbmc_aborts_results',
        ['V_A_samples', 'theta_A_samples', 't_A_aff_samp'],
        ['V_A', 'theta_A', 't_A_aff']),
    ('vbmc_vanilla_tied_results',
        ['rate_lambda_samples', 'T_0_samples', 'theta_E_samples', 'w_samples', 't_E_aff_samples', 'del_go_samples'],
        ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go']),
]

import glob
import numpy as np

# Find all animal pickle files from all batches
pkl_files = []  # List of (batch, animal_id, filename)
for fname in os.listdir(RESULTS_DIR):
    if fname.startswith('results_') and fname.endswith('.pkl'):
        for batch in BATCHES:
            prefix = f'results_{batch}_animal_'
            if fname.startswith(prefix):
                try:
                    animal_id = int(fname.split('_')[-1].replace('.pkl', ''))
                    pkl_files.append((batch, animal_id, fname))
                except Exception:
                    continue

all_param_means = {}
all_param_medians = {}

for model_key, param_keys, param_labels in model_configs:
    print(f'\nModel: {model_key}')
    param_values = {param: [] for param in param_keys}
    for batch, animal_id, fname in pkl_files:
        pkl_path = os.path.join(RESULTS_DIR, fname)
        try:
            with open(pkl_path, 'rb') as f:
                results = pickle.load(f)
        except Exception as e:
            print(f'Could not load {fname}: {e}')
            continue
        if model_key not in results:
            continue
        for param in param_keys:
            samples = np.asarray(results[model_key][param])
            param_values[param].append(np.mean(samples))
    for param, label in zip(param_keys, param_labels):
        arr = np.array(param_values[param])
        if arr.size == 0:
            print(f'  {label}: No data')
            continue
        mean_val = np.mean(arr)
        median_val = np.median(arr)
        all_param_means[label] = mean_val
        all_param_medians[label] = median_val
        print(f'  {label}: mean={mean_val:.4g}, median={median_val:.4g} (N={arr.size})')

# Assign mean values to variables for simulation
# Helper to get param or None
# %%
def get_param_or_none(param_dict, key):
    if key in param_dict:
        return param_dict[key]
    else:
        print(f"Parameter '{key}' not found in param dict. Setting to None.")
        return None

# Simulation settings
N_sim = 1_000_000
dt = 1e-3
N_print = N_sim // 5

# Prepare samples for simulation
if not all_data.empty:
    rng = np.random.default_rng()
    ABL_samples = rng.choice(all_data['ABL'].dropna().values, size=N_sim, replace=True)
    ILD_samples = rng.choice(all_data['ILD'].dropna().values, size=N_sim, replace=True)
    t_stim_samples = rng.choice(all_data['intended_fix'].dropna().values, size=N_sim, replace=True)
else:
    print("all_data is not available or empty. Cannot sample ABL, ILD, or intended_fix for simulation.")
    ABL_samples = ILD_samples = t_stim_samples = None

rate_norm_l = 0
print(f"rate_norm_l = {rate_norm_l}")


def simulate_rts(param_dict, ABL_samples, ILD_samples, t_stim_samples, rate_norm_l, N_sim, N_print, dt):
    V_A = get_param_or_none(param_dict, 'V_A')
    theta_A = get_param_or_none(param_dict, 'theta_A')
    t_a_aff = get_param_or_none(param_dict, 't_A_aff')
    rate_lambda = get_param_or_none(param_dict, 'rate_lambda')
    T_0 = get_param_or_none(param_dict, 'T_0')
    theta_E = get_param_or_none(param_dict, 'theta_E')
    w = get_param_or_none(param_dict, 'w')
    t_E_aff = get_param_or_none(param_dict, 't_E_aff')
    del_go = get_param_or_none(param_dict, 'del_go')
    if w is not None and theta_E is not None:
        Z_E = (w - 0.5) * 2 * theta_E
    else:
        print("Cannot compute Z_E because w or theta_E is None. Setting Z_E to None.")
        Z_E = None
    sim_results = Parallel(n_jobs=30)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
            V_A, theta_A, ABL_samples[iter_num], ILD_samples[iter_num], rate_lambda, T_0, theta_E, Z_E, t_a_aff, t_E_aff, del_go, 
            t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
        ) for iter_num in tqdm(range(N_sim))
    )
    sim_results_df = pd.DataFrame(sim_results)
    sim_results_df_valid = sim_results_df[sim_results_df['rt'] > sim_results_df['t_stim']]
    sim_results_df_valid_lt_1 = sim_results_df_valid[sim_results_df_valid['rt'] - sim_results_df_valid['t_stim'] <= 1]
    sim_rt = sim_results_df_valid_lt_1['rt'] - sim_results_df_valid_lt_1['t_stim']
    return sim_rt

# Run simulation for mean and median parameters
sim_rt_mean = simulate_rts(all_param_means, ABL_samples, ILD_samples, t_stim_samples, rate_norm_l, N_sim, N_print, dt)
sim_rt_median = simulate_rts(all_param_medians, ABL_samples, ILD_samples, t_stim_samples, rate_norm_l, N_sim, N_print, dt)

# %%
# Plot mean, median, and data RT distributions
plt.hist(sim_rt_mean, bins=bins, density=True, histtype='step', label='Simulated (mean)', color='r')
plt.hist(sim_rt_median, bins=bins, density=True, histtype='step', label='Simulated (median)', color='g')
plt.hist(filtered_trials['RTwrtStim'], bins=bins, density=True, histtype='step', label='Data', color='b')
plt.legend()
plt.show()



