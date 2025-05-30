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
    # Define 4-parameter sigmoid function
    def sigmoid(x, base, amplitude, inflection, slope):
        values = base + amplitude / (1 + np.exp(-slope * (x - inflection)))
        return np.clip(values, 0, 1)
    p0 = [0.0, 1.0, 0.0, 1.0]
    valid_idx = ~np.isnan(right_choice_probs)
    if np.sum(valid_idx) < 4:
        return None
    x = ild_values[valid_idx]
    y = right_choice_probs[valid_idx]
    try:
        popt, _ = curve_fit(sigmoid, x, y, p0=p0)
        return {
            'params': popt,
            'sigmoid_fn': lambda x: np.clip(sigmoid(x, *popt), 0, 1)
        }
    except Exception as e:
        print(f"Error fitting sigmoid: {str(e)}")
        return None

# %%
DESIRED_BATCHES = ['Comparable', 'SD', 'LED2', 'LED1', 'LED34', 'LED6']
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, 'batch_csvs')
results_dir = base_dir

def find_batch_animal_pairs():
    pairs = []
    pattern = os.path.join(results_dir, 'results_*_animal_*.pkl')
    pickle_files = glob.glob(pattern)
    for pickle_file in pickle_files:
        filename = os.path.basename(pickle_file)
        parts = filename.split('_')
        if len(parts) >= 4:
            batch_index = parts.index('animal') - 1 if 'animal' in parts else 1
            animal_index = parts.index('animal') + 1 if 'animal' in parts else 2
            batch_name = parts[batch_index]
            animal_id = parts[animal_index].split('.')[0]
            if batch_name in DESIRED_BATCHES:
                pairs.append((batch_name, animal_id))
        else:
            print(f"Warning: Invalid filename format: {filename}")
    return pairs
batch_animal_pairs = find_batch_animal_pairs()
print(f"Found {len(batch_animal_pairs)} batch-animal pairs: {batch_animal_pairs}")

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
                        'fit': fit_result
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
# theoretical_psychometric_data = run_theoretical_psychometric_processing()
psychometric_data = run_psychometric_processing()
# print(f'len of theory psycho data = {len(theoretical_psychometric_data)}')
print(f'len of empirical psycho data = {len(psychometric_data)}')

# %%
# Save theoretical data in pickle file
# uncomment when no pkl files
# import copy
# theory_psycho_data_to_save = copy.deepcopy(theoretical_psychometric_data)
# for batch_animal_pair, animal_data in theory_psycho_data_to_save.items():
#     for abl, abl_data in animal_data.items():
#         fit = abl_data.get('fit')
#         if fit is not None and 'sigmoid_fn' in fit:
#             del fit['sigmoid_fn']

# pickle_filename = f"theoretical_psychometric_data_{'norm' if IS_NORM_TIED else 'vanilla'}.pkl"
# with open(pickle_filename, 'wb') as f:
#     pickle.dump(theory_psycho_data_to_save, f)
# print(f"Saved theoretical psychometric data to {pickle_filename}")

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
fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# Calculate mean slopes for each animal (same order as common_pairs_sorted)
data_means = np.array([np.nanmean([slopes_data[ba][a] for a in abl_names]) for ba in common_pairs_sorted])
vanilla_means = np.array([np.nanmean([slopes_vanilla[ba][a] for a in abl_names]) for ba in common_pairs_sorted])
norm_means = np.array([np.nanmean([slopes_norm[ba][a] for a in abl_names]) for ba in common_pairs_sorted])

from sklearn.metrics import r2_score

# Data vs Vanilla
ax = axes2[0]
ax.scatter(data_means, vanilla_means, color='k', marker='X', s=60, alpha=0.7)
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
ax.legend([f'$R^2$ = {r2_vanilla:.2f}'], loc='upper left', frameon=False, fontsize=15)

# Data vs Norm
ax = axes2[1]
ax.scatter(data_means, norm_means, color='k', marker='X', s=60, alpha=0.7)
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
# %%
