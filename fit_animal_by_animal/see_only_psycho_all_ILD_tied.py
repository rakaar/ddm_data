# %%
"""
Unified analysis for psychometric curves using TIED models.
Set IS_NORM_TIED = True for normalized TIED, False for vanilla TIED.
"""
# %%
IS_NORM_TIED = True  # Set to False for vanilla TIED

from scipy.integrate import trapezoid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn
import pickle
import warnings
from types import SimpleNamespace
from animal_wise_plotting_utils import calculate_theoretical_curves
from time_vary_norm_utils import (
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn, 
    cum_pro_and_reactive_time_vary_fn, 
    rho_A_t_fn, 
    cum_A_t_fn
)
from collections import defaultdict
import random

# %%
def fit_psychometric_sigmoid(ild_values, right_choice_probs):
    from scipy.optimize import curve_fit
    import numpy as np

    def sigmoid(x, upper, lower, x0, k):
        """Sigmoid function with explicit upper and lower asymptotes."""
        return lower + (upper - lower) / (1 + np.exp(-k*(x-x0)))

    valid_idx = ~np.isnan(right_choice_probs)
    if np.sum(valid_idx) < 4:
        return None

    x = ild_values[valid_idx]
    y = right_choice_probs[valid_idx]

    min_psycho = np.min(y)
    max_psycho = np.max(y)
    # Initial guess for [upper, lower, x0, k]
    p0 = [max_psycho, min_psycho, np.median(x), 0.1]
    bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])

    try:
        popt, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return {
            'params': popt,
            'sigmoid_fn': lambda x: sigmoid(x, *popt)
        }
    except Exception as e:
        print(f"Error fitting sigmoid: {str(e)}")
        return None

# %%
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)

merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

# --- Print animal table ---
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))

print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")

if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch, animal in batch_animal_pairs:
        # Ensure animal is a string and we don't add duplicates
        animal_str = str(animal)
        if animal_str not in batch_to_animals[batch]:
            batch_to_animals[batch].append(animal_str)

    # Determine column widths for formatting
    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0

    # Header
    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")

    # Rows
    for batch in sorted(animal_strings.keys()):
        animals_str = animal_strings[batch]
        print(f"{batch:<{max_batch_len}}  {animals_str}")

# %%
def get_params_from_animal_pkl_file(batch_name, animal_id):
    pkl_file = f'results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    vbmc_vanilla_tied_param_keys_map = {
        'rate_lambda_samples': 'rate_lambda',
        'T_0_samples': 'T_0',
        'theta_E_samples': 'theta_E',
        'w_samples': 'w',
        't_E_aff_samples': 't_E_aff',
        'del_go_samples': 'del_go'
    }
    vbmc_norm_tied_param_keys_map = {
        'rate_lambda_samples': 'rate_lambda',
        'T_0_samples': 'T_0',
        'theta_E_samples': 'theta_E',
        'w_samples': 'w',
        't_E_aff_samples': 't_E_aff',
        'del_go_samples': 'del_go',
        'rate_norm_l_samples': 'rate_norm_l'
    }
    abort_keyname = "vbmc_aborts_results"
    vanilla_tied_keyname = "vbmc_vanilla_tied_results"
    norm_tied_keyname = "vbmc_norm_tied_results"
    abort_params = {}
    vanilla_tied_params = {}
    norm_tied_params = {}
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    if vanilla_tied_keyname in fit_results_data:
        vanilla_tied_samples = fit_results_data[vanilla_tied_keyname]
        for param_samples_name, param_label in vbmc_vanilla_tied_param_keys_map.items():
            vanilla_tied_params[param_label] = np.mean(vanilla_tied_samples[param_samples_name])
    if norm_tied_keyname in fit_results_data:
        norm_tied_samples = fit_results_data[norm_tied_keyname]
        for param_samples_name, param_label in vbmc_norm_tied_param_keys_map.items():
            norm_tied_params[param_label] = np.mean(norm_tied_samples[param_samples_name])
    if IS_NORM_TIED:
        return abort_params, norm_tied_params
    else:
        return abort_params, vanilla_tied_params

def get_P_A_C_A(batch, animal_id, abort_params):
    N_theory = int(1e3)
    file_name = f'batch_csvs/batch_{batch}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == animal_id]
    t_pts = np.arange(-2, 2, 0.001)
    P_A_mean, C_A_mean, t_stim_samples = calculate_theoretical_curves(
        df_animal, N_theory, t_pts, abort_params['t_A_aff'], abort_params['V_A'], abort_params['theta_A'], rho_A_t_fn
    )
    return P_A_mean, C_A_mean, t_stim_samples

# %%
ABL_arr = [20, 40, 60]
ILD_arr = [-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.]

# %%
def get_animal_psychometric_data(batch_name, animal_id, ABL):
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df = df[(df['animal'] == animal_id) & (df['ABL'] == ABL) & (df['success'].isin([1, -1]))]
    if df.empty:
        print(f"No data found for batch {batch_name}, animal {animal_id}, ABL {ABL}. Returning NaNs.")
        return None
    df = df[df['RTwrtStim'] <= 1]
    ild_values = sorted(df['ILD'].unique())
    right_choice_probs = []
    for ild in ild_values:
        ild_trials = df[df['ILD'] == ild]
        if len(ild_trials) > 0:
            right_prob = np.mean(ild_trials['choice'] == 1)
            right_choice_probs.append(right_prob)
        else:
            right_choice_probs.append(np.nan)
    return {
        'ild_values': np.array(ild_values),
        'right_choice_probs': np.array(right_choice_probs)
    }

def get_animal_psychometric_all_ABL(batch_name, animal_id):
    file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df = df[(df['animal'] == animal_id) & (df['success'].isin([1, -1]))]
    if df.empty:
        print(f"No data found for batch {batch_name}, animal {animal_id}. Returning NaNs.")
        return None
    df = df[df['RTwrtStim'] <= 1]
    ild_values = sorted(df['ILD'].unique())
    right_choice_probs = []
    for ild in ild_values:
        ild_trials = df[df['ILD'] == ild]
        if len(ild_trials) > 0:
            right_prob = np.mean(ild_trials['choice'] == 1)
            right_choice_probs.append(right_prob)
        else:
            right_choice_probs.append(np.nan)
    return {
        'ild_values': np.array(ild_values),
        'right_choice_probs': np.array(right_choice_probs)
    }

# %%
def process_batch_animal_psychometric(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    animal_id = int(animal_id)
    animal_psychometric_data = {}
    valid_abls = []
    included_animals_psychometric = []
    try:
        file_name = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
        df = pd.read_csv(file_name)
        df_animal = df[(df['animal'] == animal_id) & (df['success'].isin([1, -1]))]
        df_animal = df_animal[df_animal['RTwrtStim'] <= 1]
        all_ABL_psychometric_data = get_animal_psychometric_all_ABL(batch_name, animal_id)
        for abl in ABL_arr:
            print(f'Processing animal = {batch_name},{animal_id} for ABL={abl}')
            try:
                psychometric_data = get_animal_psychometric_data(batch_name, int(animal_id), abl)
                if psychometric_data is not None:
                    fit_result = fit_psychometric_sigmoid(
                        psychometric_data['ild_values'], 
                        psychometric_data['right_choice_probs']
                    )
                    if fit_result is not None and 'sigmoid_fn' in fit_result:
                        del fit_result['sigmoid_fn']  # Remove lambda before storing
                    animal_psychometric_data[abl] = {
                        'empirical': psychometric_data,
                        'fit': fit_result,
                        'all_ABL': all_ABL_psychometric_data
                    }
                    valid_abls.append(abl)
                else:
                    print(f"  No data for ABL={abl}. Skipping this ABL.")
            except Exception as e:
                print(f"  Error processing ABL={abl}: {str(e)}")
                continue
        if valid_abls:
            included_animals_psychometric.append(batch_animal_pair)
            print(f"Animal {batch_name},{animal_id} included with ABLs: {valid_abls}")
            return batch_animal_pair, animal_psychometric_data
        else:
            print(f"Animal {batch_name},{animal_id} has no valid ABLs, but returning empty psychometric data.")
            return batch_animal_pair, animal_psychometric_data
    except Exception as e:
        print(f"Error processing psychometric data for {batch_name}, animal {animal_id}: {str(e)}")
        return batch_animal_pair, {}

# %%
def run_psychometric_processing():
    n_jobs = max(1, os.cpu_count() - 1)
    print(f"Running psychometric processing with {n_jobs} parallel jobs")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_batch_animal_psychometric)(batch_animal_pair) for batch_animal_pair in batch_animal_pairs
    )
    psychometric_data = {}
    for batch_animal_pair, animal_psychometric_data in results:
        if animal_psychometric_data:
            psychometric_data[batch_animal_pair] = animal_psychometric_data
    print(f"Completed psychometric processing for {len(psychometric_data)} batch-animal pairs")
    return psychometric_data

# %%
def get_theoretical_psychometric_data(batch_name, animal_id, ABL):
    try:
        abort_params, tied_params = get_params_from_animal_pkl_file(batch_name, int(animal_id))
        p_a, c_a, ts_samp = get_P_A_C_A(batch_name, int(animal_id), abort_params)
        ild_values = np.array([-16., -8., -4., -2., -1., 1., 2., 4., 8., 16.])
        right_choice_probs = []
        for ild in ild_values:
            try:
                t_pts_0_1, up_mean, down_mean = get_theoretical_RTD_up_down(
                    p_a, c_a, ts_samp, abort_params, tied_params, ABL, ild
                )
                up_area = trapezoid(up_mean, t_pts_0_1)
                down_area = trapezoid(down_mean, t_pts_0_1)
                right_prob = up_area / (up_area + down_area)
                right_choice_probs.append(right_prob)
            except Exception as e:
                print(f"  Error calculating theoretical psychometric for ABL={ABL}, ILD={ild}: {str(e)}")
                right_choice_probs.append(np.nan)
        return {
            'ild_values': ild_values,
            'right_choice_probs': np.array(right_choice_probs)
        }
    except Exception as e:
        print(f"Error getting parameters for batch {batch_name}, animal {animal_id}: {str(e)}")
        return None

# %%
def get_theoretical_RTD_up_down(P_A_mean, C_A_mean, t_stim_samples, abort_params, tied_params, ABL, ILD):
    phi_params_obj = np.nan
    if IS_NORM_TIED:
        rate_norm_l = tied_params.get('rate_norm_l', np.nan)
        is_norm = True
    else:
        rate_norm_l = 0
        is_norm = False
    is_time_vary = False
    K_max = 10
    T_trunc = 0.3
    t_pts = np.arange(-2, 2, 0.001)
    trunc_fac_samples = np.zeros((len(t_stim_samples)))
    Z_E = (tied_params['w'] - 0.5) * 2 * tied_params['theta_E']
    for idx, t_stim in enumerate(t_stim_samples):
        trunc_fac_samples[idx] = cum_pro_and_reactive_time_vary_fn(
            t_stim + 1, T_trunc,
            abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
            t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
            phi_params_obj, rate_norm_l, 
            is_norm, is_time_vary, K_max) \
            - \
            cum_pro_and_reactive_time_vary_fn(
            t_stim, T_trunc,
            abort_params['V_A'], abort_params['theta_A'], abort_params['t_A_aff'],
            t_stim, ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'],
            phi_params_obj, rate_norm_l, 
            is_norm, is_time_vary, K_max) + 1e-10
    trunc_factor = np.mean(trunc_fac_samples)
    up_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
        t, 1,
        P_A_mean[i], C_A_mean[i],
        ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
        phi_params_obj, rate_norm_l, 
        is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
    down_mean = np.array([up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn(
        t, -1,
        P_A_mean[i], C_A_mean[i],
        ABL, ILD, tied_params['rate_lambda'], tied_params['T_0'], tied_params['theta_E'], Z_E, tied_params['t_E_aff'], tied_params['del_go'],
        phi_params_obj, rate_norm_l, 
        is_norm, is_time_vary, K_max) for i, t in enumerate(t_pts)])
    mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    t_pts_0_1 = t_pts[mask_0_1]
    up_mean_0_1 = up_mean[mask_0_1]
    down_mean_0_1 = down_mean[mask_0_1]
    up_theory_mean_norm = up_mean_0_1 / trunc_factor
    down_theory_mean_norm = down_mean_0_1 / trunc_factor
    return t_pts_0_1, up_theory_mean_norm, down_theory_mean_norm

# %%
def process_batch_animal_theoretical_psychometric(batch_animal_pair):
    batch_name, animal_id = batch_animal_pair
    print(f"Processing theoretical psychometric data for batch {batch_name}, animal {animal_id}")
    animal_theoretical_psychometric_data = {}
    for abl in [20, 40, 60]:
        try:
            psychometric_data = get_theoretical_psychometric_data(batch_name, int(animal_id), abl)
            if psychometric_data is not None:
                fit_result = fit_psychometric_sigmoid(
                    psychometric_data['ild_values'], 
                    psychometric_data['right_choice_probs']
                )
                animal_theoretical_psychometric_data[abl] = {
                    'theoretical': psychometric_data,
                    'fit': fit_result
                }
                print(f"  Processed theoretical ABL={abl}")
            else:
                animal_theoretical_psychometric_data[abl] = {
                    'theoretical': None,
                    'fit': None
                }
                print(f"  No theoretical data for ABL={abl}")
        except Exception as e:
            print(f"  Error processing theoretical ABL={abl}: {str(e)}")
            animal_theoretical_psychometric_data[abl] = {
                'theoretical': None,
                'fit': None
            }
    return batch_animal_pair, animal_theoretical_psychometric_data

# %%
def run_theoretical_psychometric_processing():
    n_jobs = max(1, os.cpu_count() - 1)
    print(f"Running theoretical psychometric processing with {n_jobs} parallel jobs")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_batch_animal_theoretical_psychometric)(batch_animal_pair) for batch_animal_pair in batch_animal_pairs
    )
    theoretical_psychometric_data = {}
    for batch_animal_pair, animal_theoretical_psychometric_data in results:
        if animal_theoretical_psychometric_data:
            theoretical_psychometric_data[batch_animal_pair] = animal_theoretical_psychometric_data
    print(f"Completed theoretical psychometric processing for {len(theoretical_psychometric_data)} batch-animal pairs")
    return theoretical_psychometric_data

# %%
def plot_theoretical_psychometric_data(theoretical_psychometric_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    ild_smooth = np.linspace(-16, 16, 100)
    abls = [20, 40, 60]
    abl_colors = ['b', 'g', 'r']
    for i, abl in enumerate(abls):
        ax = axes[i]
        individual_fits = []
        for batch_animal_pair, animal_data in theoretical_psychometric_data.items():
            if abl in animal_data and animal_data[abl]['fit'] is not None:
                fit = animal_data[abl]['fit']
                fit_values = [fit['sigmoid_fn'](x) for x in ild_smooth]
                individual_fits.append(fit_values)
                ax.plot(ild_smooth, fit_values, color=abl_colors[i], alpha=0.4, linewidth=1)
        if individual_fits:
            avg_fit = np.nanmean(individual_fits, axis=0)
            ax.plot(ild_smooth, avg_fit, color=abl_colors[i], linewidth=3, label=f'ABL={abl}')
        ax.set_title(f'Theoretical ABL = {abl} dB')
        ax.set_xlabel('ILD (dB)')
        if i == 0:
            ax.set_ylabel('P(right choice)')
        ax.set_xlim(-16, 16)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    plt.tight_layout()
    plt.savefig('theoretical_psychometric_by_abl.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# %%
# Get theoretical and empirical data
theoretical_psychometric_data = run_theoretical_psychometric_processing()
psychometric_data = run_psychometric_processing()
# print(f'len of theory psycho data = {len(theoretical_psychometric_data)}')
print(f'len of empirical psycho data = {len(psychometric_data)}')

# %%
# Save theoretical data in pickle file
# uncomment when no pkl files
import copy
theory_psycho_data_to_save = copy.deepcopy(theoretical_psychometric_data)
for batch_animal_pair, animal_data in theory_psycho_data_to_save.items():
    for abl, abl_data in animal_data.items():
        fit = abl_data.get('fit')
        if fit is not None and 'sigmoid_fn' in fit:
            del fit['sigmoid_fn']

pickle_filename = f"theoretical_psychometric_data_{'norm' if IS_NORM_TIED else 'vanilla'}.pkl"
with open(pickle_filename, 'wb') as f:
    pickle.dump(theory_psycho_data_to_save, f)
print(f"Saved theoretical psychometric data to {pickle_filename}")



# %%
# --- Print a summary of psychometric_data dictionary structure ---
print("\n--- Summary of psychometric_data ---")
print(f"Top-level keys (batch, animal) pairs: {list(psychometric_data.keys())[:3]} ... total: {len(psychometric_data)}")

# Show a sample entry
for batch_animal_pair, abl_dict in list(psychometric_data.items())[:1]:
    print(f"\nBatch-animal pair: {batch_animal_pair}")
    print(f"  ABLs: {list(abl_dict.keys())}")
    for abl, d in abl_dict.items():
        print(f"    ABL {abl}:")
        print(f"      empirical keys: {list(d['empirical'].keys())}")
        if d['fit'] is not None:
            print(f"      fit params: {d['fit'].get('params', None)}")
        else:
            print(f"      fit: None")
            

# %%
# --- Slope comparison plot: Vanilla, Data, Norm TIED ---
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load vanilla and norm model pickles
with open('theoretical_psychometric_data_vanilla.pkl', 'rb') as f:
    vanilla_psychometric_data = pickle.load(f)
with open('theoretical_psychometric_data_norm.pkl', 'rb') as f:
    norm_psychometric_data = pickle.load(f)

# Helper: extract slopes for a dict of psychometric data
def extract_slopes(data_dict):
    slopes = {}
    for batch_animal, abl_dict in data_dict.items():
        slopes[batch_animal] = {}
        for abl in [20, 40, 60]:
            fit = abl_dict.get(abl, {}).get('fit', None)
            if fit is not None and 'params' in fit:
                slopes[batch_animal][abl] = fit['params'][3]  # slope param
            else:
                slopes[batch_animal][abl] = np.nan
    return slopes

slopes_vanilla = extract_slopes(vanilla_psychometric_data)
slopes_norm = extract_slopes(norm_psychometric_data)
slopes_data = extract_slopes(psychometric_data)

# Get all batch-animal pairs present in all three, EXCLUDING animal 41
common_pairs = set(slopes_vanilla) & set(slopes_norm) & set(slopes_data)
# Remove animal 41 from the set
common_pairs = [ba for ba in common_pairs if str(ba[1]) != '41']

# Sort animals by average slope in data (ascending)
avg_slope_data = {ba: np.nanmean([slopes_data[ba][a] for a in [20,40,60]]) for ba in common_pairs}
common_pairs_sorted = sorted(common_pairs, key=lambda ba: avg_slope_data[ba])

abl_colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
abl_names = [20, 40, 60]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for idx, (slopes_dict, title) in enumerate(zip(
    [slopes_vanilla, slopes_data, slopes_norm],
    ["Vanilla", "Data", "Norm TIED"])):
    ax = axes[idx]
    x = np.arange(len(common_pairs_sorted))
    for abl in abl_names:
        y = [slopes_dict[ba][abl] for ba in common_pairs_sorted]
        ax.scatter(x, y, color=abl_colors[abl], label=f'ABL {abl}' if idx==0 else None, s=50, alpha=0.8)
    # Plot average slope per animal with thinner X marker
    y_avg = [np.nanmean([slopes_dict[ba][a] for a in abl_names]) for ba in common_pairs_sorted]
    ax.scatter(x, y_avg, color='k', label='Avg' if idx==0 else None, s=36, marker='X', zorder=2, alpha=0.6, linewidths=1.2)
    # Remove x-tick labels
    
    ax.set_xlabel('Rat #', fontsize=15)
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.tick_params(axis='both', which='major', labelsize=60)
    ax.set_title(title, fontsize=16, pad=15)
    # ax.set_yticklabels([0.1, 0.5, 0.9], fontdict = {'fontsize': 100, 'fontweight': 'bold'})
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.tick_params(axis='y', labelsize=50)  # reasonable large size
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    if idx==0:
        ax.set_ylabel('Slope', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.grid(True, axis='y', alpha=0.25)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.margins(x=0.01)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    # if idx==0:
        # ax.legend(loc='upper right', frameon=False, fontsize=12)
plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.subplots_adjust(wspace=0.15, left=0.05, right=0.98, top=0.90, bottom=0.23)
plt.show()

# %%
# --- Data vs Model Mean Slope Scatter Plots ---

# Calculate mean slopes for each animal (same order as common_pairs_sorted)
data_means = np.array([np.nanmean([slopes_data[ba][a] for a in abl_names]) for ba in common_pairs_sorted])
vanilla_means = np.array([np.nanmean([slopes_vanilla[ba][a] for a in abl_names]) for ba in common_pairs_sorted])
norm_means = np.array([np.nanmean([slopes_norm[ba][a] for a in abl_names]) for ba in common_pairs_sorted])

from sklearn.metrics import r2_score

# --- Figure 1: Data vs Vanilla ---
fig_vanilla, ax_vanilla = plt.subplots(figsize=(4, 4))
ax_vanilla.scatter(data_means, vanilla_means, color='k', marker='X', s=60, alpha=0.7)
ax_vanilla.set_xlabel('Data', fontsize=20)
ax_vanilla.set_ylabel('Model', fontsize=20)
ax_vanilla.set_xticks([0.1, 0.5, 0.9])
ax_vanilla.set_yticks([0.1, 0.5, 0.9])
ax_vanilla.set_xlim(0.1, 0.9)
ax_vanilla.set_ylim(0.1, 0.9)
ax_vanilla.tick_params(axis='both', labelsize=18)
ax_vanilla.spines['top'].set_visible(False)
ax_vanilla.spines['right'].set_visible(False)
ax_vanilla.plot([0.1, 0.9], [0.1, 0.9], color='grey', alpha=0.5, linestyle='--', linewidth=2, zorder=0)
r2_vanilla = r2_score(data_means, vanilla_means)
# ax_vanilla.legend([f'$R^2$ = {r2_vanilla:.2f}'], loc='upper left', frameon=False, fontsize=15)
plt.show()

# --- Figure 2: Data vs Norm ---
fig_norm, ax_norm = plt.subplots(figsize=(4, 4))
ax_norm.scatter(data_means, norm_means, color='k', marker='X', s=60, alpha=0.7)
ax_norm.set_xlabel('Data', fontsize=20)
ax_norm.set_ylabel('Model', fontsize=20)
ax_norm.set_xticks([0.1, 0.5, 0.9])
ax_norm.set_yticks([0.1, 0.5, 0.9])
ax_norm.set_xlim(0.1, 0.9)
ax_norm.set_ylim(0.1, 0.9)
ax_norm.tick_params(axis='both', labelsize=18)
ax_norm.spines['top'].set_visible(False)
ax_norm.spines['right'].set_visible(False)
ax_norm.plot([0.1, 0.9], [0.1, 0.9], color='grey', alpha=0.5, linestyle='--', linewidth=2, zorder=0)
r2_norm = r2_score(data_means, norm_means)
# ax.legend([f'$R^2$ = {r2_norm:.2f}'], loc='upper left', frameon=False, fontsize=15)
# %%
# --- Combined Plots: Absolute Difference and Data Slopes ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), sharey=True)

# --- Plot 1: Absolute Difference (Data vs Vanilla) ---
abs_diff = np.abs(data_means - vanilla_means)
mean_diff = np.mean(abs_diff)
sorted_indices_diff = np.argsort(abs_diff)
sorted_abs_diff = abs_diff[sorted_indices_diff]
sorted_labels_diff = [f"{b}-{a}" for b, a in np.array(common_pairs_sorted)[sorted_indices_diff]]

ax1.scatter(range(len(sorted_labels_diff)), sorted_abs_diff, color='gray', alpha=0.7, label='|Data - Vanilla|')
ax1.axhline(mean_diff, color='r', linestyle='-', linewidth=2, label=f'Mean = {mean_diff:.2f}')
percentiles = [5, 10, 50, 90, 95]
p_values = np.percentile(sorted_abs_diff, percentiles)
colors = ['blue', 'green', 'purple', 'green', 'blue']
for p, val, c in zip(percentiles, p_values, colors):
    ax1.axhline(val, color=c, linestyle=':', linewidth=1.5, label=f'{p}%ile = {val:.2f}')

ax1.set_ylabel('Absolute Slope Difference', fontsize=14)
ax1.set_title('Absolute Difference between Data and Vanilla Model Slopes (Sorted)', fontsize=16)
ax1.set_xticks(range(len(sorted_labels_diff)))
ax1.set_xticklabels(sorted_labels_diff, rotation=90, fontsize=16)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
# ax1.legend(frameon=False, fontsize=10)

# --- Plot 2: Data Slopes (Sorted) ---
sorted_indices_data = np.argsort(data_means)
sorted_data_slopes = data_means[sorted_indices_data]
sorted_labels_data = [f"{b}-{a}" for b, a in np.array(common_pairs_sorted)[sorted_indices_data]]

ax2.scatter(range(len(sorted_labels_data)), sorted_data_slopes, color='k', alpha=0.7, label='Data Slope')
mean_data = np.mean(sorted_data_slopes)
ax2.axhline(mean_data, color='r', linestyle='-', linewidth=2, label=f'Mean = {mean_data:.2f}')
p_values_data = np.percentile(sorted_data_slopes, percentiles)
for p, val, c in zip(percentiles, p_values_data, colors):
    ax2.axhline(val, color=c, linestyle=':', linewidth=1.5, label=f'{p}%ile = {val:.2f}')

ax2.set_ylabel('Data Slope', fontsize=14)
ax2.set_title('Data Slopes (Sorted)', fontsize=16)
ax2.set_xticks(range(len(sorted_labels_data)))
ax2.set_xticklabels(sorted_labels_data, rotation=90, fontsize=16)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# ax2.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import r2_score

# Assume the following variables are defined in the namespace:
# data_means, vanilla_means, norm_means, common_pairs_sorted

# Helper to extract animal number from animal id/name
def extract_animal_number(animal_id):
    match = re.search(r'(\d+)', str(animal_id))
    return match.group(1) if match else str(animal_id)

fig3, axes3 = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# Data vs Vanilla (with labels)
ax = axes3[0]
for i, ba in enumerate(common_pairs_sorted):
    x = data_means[i]
    y = vanilla_means[i]
    animal_label = extract_animal_number(ba)
    ax.text(x, y, animal_label, fontsize=14, ha='center', va='center', fontweight='bold')
ax.set_xlabel('Data', fontsize=20)
ax.set_ylabel('w/o Normalization', fontsize=20)
ax.set_xticks([0.1, 0.5, 0.9])
ax.set_yticks([0.1, 0.5, 0.9])
ax.set_xlim(0.1, 0.9)
ax.set_ylim(0.1, 0.9)
ax.tick_params(axis='both', labelsize=18)
plt.setp(ax.get_xticklabels())
plt.setp(ax.get_yticklabels())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot([0.1, 0.9], [0.1, 0.9], color='grey', alpha=0.5, linestyle='--', linewidth=2, zorder=0)
r2_vanilla = r2_score(data_means, vanilla_means)
# ax.legend([f'$R^2$ = {r2_vanilla:.2f}'], loc='upper left', frameon=False, fontsize=15)

# Data vs Norm (with labels)
ax = axes3[1]
for i, ba in enumerate(common_pairs_sorted):
    x = data_means[i]
    y = norm_means[i]
    animal_label = extract_animal_number(ba)
    ax.text(x, y, animal_label, fontsize=14, ha='center', va='center', fontweight='bold')
ax.set_xlabel('Data', fontsize=20)
ax.set_ylabel('Normalization', fontsize=20)
ax.set_xticks([0.1, 0.5, 0.9])
ax.set_yticks([0.1, 0.5, 0.9])
ax.set_xlim(0.1, 0.9)
ax.set_ylim(0.1, 0.9)
ax.tick_params(axis='both', labelsize=18)
plt.setp(ax.get_xticklabels())
plt.setp(ax.get_yticklabels())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot([0.1, 0.9], [0.1, 0.9], color='grey', alpha=0.5, linestyle='--', linewidth=2, zorder=0)
r2_norm = r2_score(data_means, norm_means)
ax.legend([f'$R^2$ = {r2_norm:.2f}'], loc='upper left', frameon=False, fontsize=15)

plt.tight_layout()
plt.show()

# %%
########  Plot psychometrics ########
# %%
# Compare vanilla/norm and data psychometric curves
# theoretical psychometric data can be vanilla or norm. Comment,Uncomment accordingly
# theoretical_psychometric_data = vanilla_psychometric_data
# theoretical_psychometric_data = norm_psychometric_data

# Get all animal keys and determine grid size
animal_keys = list(psychometric_data.keys())
total_animals = len(animal_keys)
# print(f'Total animals: {len(animal_keys)}')


# Calculate number of rows needed (5 animals per row)
row_count = (total_animals + 4) // 5  # Ceiling division to get number of rows
print(f'Number of animals: {total_animals}')
# Create a figure with subplots arranged in a grid, 5 per row
fig, axes = plt.subplots(row_count, 5, figsize=(20, 4*row_count))

# Make axes 2D if there's only one row
if row_count == 1:
    axes = axes.reshape(1, -1)

# Define colors for different ABLs
abl_colors = {20: 'blue', 40: 'green', 60: 'red'}

# Create plots for each animal
for i, key in enumerate(animal_keys):
    # Calculate row and column position
    row = i // 5
    col = i % 5
    
    # Get the appropriate axis
    ax = axes[row, col]
    
    psycho_animal = theoretical_psychometric_data[key]
    empirical_animal = psychometric_data[key]
    # Set title for each subplot
    ax.set_title(f'Animal {key}')
    
    # Plot each ABL
    for abl in [20, 40, 60]:
        if abl in psycho_animal:
            psycho_abl = psycho_animal[abl]
            ild_values = psycho_abl['theoretical']['ild_values']
            right_choice_probs = psycho_abl['theoretical']['right_choice_probs']
            ax.scatter(ild_values, right_choice_probs, color=abl_colors[abl], label=f'ABL {abl}')
            if abl in empirical_animal:
                ax.scatter(empirical_animal[abl]['empirical']['ild_values'], empirical_animal[abl]['empirical']['right_choice_probs'], color=abl_colors[abl], marker='x')    
    # Add reference lines
    ax.axhline(y=0.5, color='grey', alpha=0.5, linestyle='--')  # Horizontal line at 0.5
    ax.axvline(x=0, color='grey', alpha=0.5, linestyle='--')    # Vertical line at 0
    
    # Set axis labels and limits
    ax.set_xlabel('ILD (dB)')
    if col == 0:  # Only add y-label for leftmost plots
        ax.set_ylabel('P(right choice)')
    ax.set_xlim(-17, 17)
    ax.set_ylim(0, 1.02)
    
    # Add legend only for the first subplot
    if i == 0:
        ax.legend()

# Hide empty subplots
for i in range(total_animals, row_count * 5):
    row = i // 5
    col = i % 5
    axes[row, col].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

# %%
# === Start of new code for average psychometric plots ===
# Ensure numpy and pyplot are available (likely already imported as np and plt)
theory_agg = {}
empirical_agg = {}
for abl in [20, 40, 60]:
    theory_agg[abl] = np.full((len(animal_keys), len(ILD_arr)), np.nan)
    empirical_agg[abl] = np.full((len(animal_keys), len(ILD_arr)), np.nan)

for idx, key in enumerate(animal_keys):
    animal_data = psychometric_data[key]
    theory_data = theoretical_psychometric_data[key]
    
    for abl_key in [20, 40, 60]:
        if abl_key in theory_data:
            theory_abl_psycho = theory_data[abl_key]['theoretical']['right_choice_probs']
            theory_agg[abl_key][idx] = theory_abl_psycho
        if abl_key in animal_data:
            empirical = animal_data[abl_key]['empirical']
            # Restrict to ILD_arr only, in order
            ild_values = empirical['ild_values']
            right_choice_probs = empirical['right_choice_probs']
            # Build array for ILD_arr
            selected_probs = np.full(len(ILD_arr), np.nan)
            for i, ild in enumerate(ILD_arr):
                matches = np.where(ild_values == ild)[0]
                if len(matches) > 0:
                    selected_probs[i] = right_choice_probs[matches[0]]
            try:
                empirical_agg[abl_key][idx] = selected_probs
            except:
                print(key, abl_key)
                print('empirical ild_values:', ild_values)
                print('empirical right_choice_probs:', right_choice_probs)
                print('selected_probs:', selected_probs)
        
# %%


# Plot average psychometric curves for each ABL
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for i, abl in enumerate([20, 40, 60]):
    emp = empirical_agg[abl]  # shape: (n_animals, n_ilds)
    theo = theory_agg[abl]
    emp_mean = np.nanmean(emp, axis=0)
    emp_std = np.nanstd(emp, axis=0)
    theo_mean = np.nanmean(theo, axis=0)
    theo_std = np.nanstd(theo, axis=0)

    ax = axes[i]
    ilds = ILD_arr
    # Empirical: blue dots with error bars (std), no caps
    ax.errorbar(ilds, emp_mean, yerr=emp_std, fmt='o', color='blue', label='data', capsize=0)
    ax.plot(ilds, emp_mean, color='blue', linestyle='-', linewidth=2, alpha=0.7)  # Blue line joining dots
    # Theoretical: red dots with error bars (std), no caps
    ax.errorbar(ilds, theo_mean, yerr=theo_std, fmt='o', color='red', label='theory', capsize=0)
    ax.plot(ilds, theo_mean, color='red', linestyle='-', linewidth=2, alpha=0.7)  # Red line joining dots
    ax.set_title(f'ABL = {abl}')
    ax.set_xlabel('ILD (dB)')
    if i == 0:
        ax.set_ylabel('P(choice = right)')
    ax.set_xticks(ilds)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
plt.tight_layout()
plt.show()


# %%
# Plot all three ABLs in a single figure: data (dotted), theory (solid), each ABL a color
import matplotlib.pyplot as plt
import numpy as np

colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
plt.figure(figsize=(8, 6))
for abl in [20, 40, 60]:
    emp = empirical_agg[abl]
    theo = theory_agg[abl]
    emp_mean = np.nanmean(emp, axis=0)
    theo_mean = np.nanmean(theo, axis=0)
    ilds = ILD_arr
    # Empirical: dotted line
    plt.plot(ilds, emp_mean, linestyle=':', marker='o', color=colors[abl], label=f'Data ABL={abl}')
    # Theory: solid line
    plt.plot(ilds, theo_mean, linestyle='-', marker=None, color=colors[abl], label=f'Theory ABL={abl}')
plt.xlabel('ILD (dB)')
plt.ylabel('P(choice = right)')
plt.title('Average Psychometric Curves (All ABLs)')
plt.xticks([-15,-5,5,15])
plt.yticks([0, 0.5, 1])
plt.axvline(0, alpha=0.5, color='grey')
plt.axhline(0.5, alpha=0.5, color='grey')

plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Plot: data for all ABLs in one plot, theory for all ABLs in another plot
import matplotlib.pyplot as plt
import numpy as np

colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Data plot (left)
for abl in [20, 40, 60]:
    emp = empirical_agg[abl]
    emp_mean = np.nanmean(emp, axis=0)
    ilds = ILD_arr
    axes[0].plot(ilds, emp_mean, linestyle=':', marker='o', color=colors[abl], label=f'Data ABL={abl}')
axes[0].set_xlabel('ILD (dB)')
axes[0].set_ylabel('P(choice = right)')
axes[0].set_title('Empirical Data (All ABLs)')
axes[0].set_xticks([-15, -5, 5, 15])
axes[0].set_yticks([0, 0.5, 1])
axes[0].axvline(0, alpha=0.5, color='grey')
axes[0].axhline(0.5, alpha=0.5, color='grey')
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend()

# Theory plot (right)
for abl in [20, 40, 60]:
    theo = theory_agg[abl]
    theo_mean = np.nanmean(theo, axis=0)
    ilds = ILD_arr
    axes[1].plot(ilds, theo_mean, linestyle='-', marker='o', color=colors[abl], label=f'Theory ABL={abl}')
axes[1].set_xlabel('ILD (dB)')
axes[1].set_title('Theoretical (All ABLs)')
axes[1].set_xticks([-15, -5, 5, 15])
axes[1].set_yticks([0, 0.5, 1])
axes[1].axvline(0, alpha=0.5, color='grey')
axes[1].axhline(0.5, alpha=0.5, color='grey')
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend()

plt.tight_layout()
plt.show()
# %%
# Plot all three ABLs in a single figure: data (dotted), theory (solid), each ABL a color

import matplotlib.pyplot as plt
import numpy as np

colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}
plt.figure(figsize=(4, 3))  # Smaller figure for publication
for abl in [20, 40, 60]:
    emp = empirical_agg[abl]
    theo = theory_agg[abl]
    emp_mean = np.nanmean(emp, axis=0)
    theo_mean = np.nanmean(theo, axis=0)
    ilds = np.array(ILD_arr)
    theo_mean = np.array(theo_mean)
    # Empirical: dotted line
    n_emp = np.sum(~np.isnan(emp), axis=0)
    print(f'emp n valid: {n_emp}')
    emp_sem = np.nanstd(emp, axis=0) / np.sqrt(np.maximum(n_emp - 1, 1))
    print(f'denominator: {np.sqrt(np.maximum(n_emp - 1, 1))}')
    plt.errorbar(ilds, emp_mean, yerr=emp_sem, fmt='o', color=colors[abl], label=f'Data ABL={abl}', capsize=0, markersize=4)
    # Logistic fit to theory: solid line
    valid_idx = ~np.isnan(theo_mean)
    if np.sum(valid_idx) >= 4:
        try:
            from scipy.optimize import curve_fit
            def logistic(x, base, amplitude, inflection, slope):
                values = base + amplitude / (1 + np.exp(-slope * (x - inflection)))
                return np.clip(values, 0, 1)
            p0 = [0.0, 1.0, 0.0, 1.0]
            popt, _ = curve_fit(logistic, ilds[valid_idx], theo_mean[valid_idx], p0=p0)
            ilds_smooth = np.linspace(min(ilds), max(ilds), 200)
            fit_curve = logistic(ilds_smooth, *popt)
            plt.plot(ilds_smooth, fit_curve, linestyle='-', color=colors[abl], label=f'Logistic fit (Theory) ABL={abl}', lw=0.5)
        except Exception as e:
            print(f"Could not fit logistic for ABL={abl}: {e}")
    else:
        print(f"Not enough valid theory points for ABL={abl} to fit.")
plt.xlabel('ILD (dB)', fontsize=16)
plt.ylabel('P(choice = right)', fontsize=16)
# plt.title('Average Psychometric Curves (All ABLs)')
plt.xticks([-15,-5,5,15], fontsize=14)
plt.yticks([0, 0.5, 1], fontsize=14)
plt.axvline(0, alpha=0.5, color='grey', linestyle='--')
plt.axhline(0.5, alpha=0.5, color='grey', linestyle='--')

plt.ylim(-0.05, 1.05)
# Remove top and right spines for publication style
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# %%
# %%
# Plot grand average (across all ABLs) for both data and theory

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(4,3))

# Stack all empirical and theory psychometrics across ABLs
theo_all = []
for abl in [20, 40, 60]:
    theo = theory_agg[abl]    # shape: (n_animals, n_ilds)
    theo_all.append(theo)
theo_all = np.concatenate(theo_all, axis=0)  # shape: (n_animals * 3, n_ilds)

##### Correct:  emp all ###
emp_all = np.full((len(psychometric_data), len(ILD_arr)), np.nan)
for i, (batch, animal) in enumerate(psychometric_data.keys()):
    ANY_KEY = list(psychometric_data[(batch, animal)].keys())[0]
    ilds = psychometric_data[(batch, animal)][ANY_KEY]['all_ABL']['ild_values']
    all_ABL_psycho = psychometric_data[(batch, animal)][ANY_KEY]['all_ABL']['right_choice_probs']
    for j, ild in enumerate(ILD_arr):
        if ild in ilds:
            # find its index
            idx = np.where(ilds == ild)[0][0]
            emp_all[i, j] = all_ABL_psycho[idx]




# Data: mean and error bars
emp_mean = np.nanmean(emp_all, axis=0)
n_emp = np.sum(~np.isnan(emp_all), axis=0)
emp_sem = np.nanstd(emp_all, axis=0) / np.sqrt(np.maximum(n_emp - 1, 1))
# print(f'std dev = {np.nanstd(emp_all, axis=0)}')
# print(f'sqrt len - 1 = {np.sqrt(np.maximum(n_emp - 1, 1))}')
# print(f'emp_sem = {emp_sem}')

plt.errorbar(ILD_arr, emp_mean, yerr=emp_sem, fmt='o', color='tab:blue', label='Data (grand avg)', capsize=0, markersize=4)

# Theory: mean
theo_mean = np.nanmean(theo_all, axis=0)

# Logistic fit to theory mean
valid_idx = ~np.isnan(theo_mean)
ilds = np.array(ILD_arr)
if np.sum(valid_idx) >= 4:
    try:
        from scipy.optimize import curve_fit
        def logistic(x, base, amplitude, inflection, slope):
            values = base + amplitude / (1 + np.exp(-slope * (x - inflection)))
            return np.clip(values, 0, 1)
        p0 = [0.0, 1.0, 0.0, 1.0]
        popt, _ = curve_fit(logistic, ILD_arr, theo_mean[valid_idx], p0=p0)
        ilds_smooth = np.linspace(min(ilds), max(ilds), 200)
        fit_curve = logistic(ilds_smooth, *popt)
        plt.plot(ilds_smooth, fit_curve, linestyle='-', color='black', label='Logistic fit (Theory grand avg)', lw=0.5)
    except Exception as e:
        print(f"Could not fit logistic for theory grand avg: {e}")
else:
    print(f"Not enough valid theory points to fit.")

plt.xlabel('ILD (dB)', fontsize=16)
plt.ylabel('P(choice = right)', fontsize=16)
plt.xticks([-15,-5,5,15], fontsize=14)
plt.yticks([0, 0.5, 1], fontsize=14)
plt.axvline(0, alpha=0.5, color='grey', linestyle='--')
plt.axhline(0.5, alpha=0.5, color='grey', linestyle='--')

plt.ylim(-0.05, 1.05)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('N_26_psychometric_curve_vanila_data.pdf')
plt.show()
# %%
# === Empirical-only psychometric plot (mean and fit) ===
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(4,3))

# Use previously computed emp_mean and emp_sem
plt.errorbar(ILD_arr, emp_mean, yerr=emp_sem, fmt='o', color='tab:blue', label='Data (grand avg)', capsize=0, markersize=4)

# Fit psychometric (logistic) to empirical mean
valid_idx = ~np.isnan(emp_mean)
ilds = np.array(ILD_arr)
if np.sum(valid_idx) >= 4:
    try:
        from scipy.optimize import curve_fit
        def logistic(x, base, amplitude, inflection, slope):
            values = base + amplitude / (1 + np.exp(-slope * (x - inflection)))
            return np.clip(values, 0, 1)
        p0 = [0.0, 1.0, 0.0, 1.0]
        popt, _ = curve_fit(logistic, ILD_arr, emp_mean[valid_idx], p0=p0)
        ilds_smooth = np.linspace(min(ilds), max(ilds), 200)
        fit_curve = logistic(ilds_smooth, *popt)
        plt.plot(ilds_smooth, fit_curve, linestyle='-', color='black', label='Logistic fit (Empirical grand avg)', lw=0.5)
    except Exception as e:
        print(f"Could not fit logistic for empirical grand avg: {e}")
else:
    print(f"Not enough valid empirical points to fit.")

plt.xlabel('ILD (dB)', fontsize=16)
plt.ylabel('P(choice = right)', fontsize=16)
plt.xticks([-15,-5,5,15], fontsize=14)
plt.yticks([0, 0.5, 1], fontsize=14)
plt.axvline(0, alpha=0.5, color='grey', linestyle='--')
plt.axhline(0.5, alpha=0.5, color='grey', linestyle='--')

plt.ylim(-0.05, 1.05)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('empirical_psychometric_fit.pdf', bbox_inches='tight')
# plt.legend()
plt.show()