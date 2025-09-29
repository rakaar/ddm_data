# VBMC on simulated data - lapses, pro + TIED
# %%
import pickle
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from pyvbmc import VBMC 

# Set model type (False for vanilla TIED, True for norm TIED)
IS_NORM_TIED = False

# VBMC constants
T_trunc = 0.3
phi_params_obj = np.nan
rate_norm_l = 0
is_norm = False
is_time_vary = False
K_max = 10

# Function to read animal parameters from pickle file
def get_params_from_animal_pkl_file(batch_name, animal_id):
    pkl_file = f'../fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
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
def psiam_tied_data_gen_wrapper_rate_norm_fn(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go, \
                                t_stim, rate_norm_l, iter_num, N_print, dt, lapse_prob=0.0, T_lapse_max=1.0):

    if iter_num % N_print == 0:
        print(f'os id: {os.getpid()}, In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')

    choice, rt, is_act = simulate_psiam_tied_rate_norm(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, \
                                                       t_stim, t_A_aff, t_E_aff, del_go, rate_norm_l, dt, lapse_prob, T_lapse_max)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}

def simulate_psiam_tied_rate_norm(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, \
                                  t_A_aff, t_E_aff, del_go, rate_norm_l, dt, lapse_prob=0.0, T_lapse_max=1.0):

    # Lapse mechanism: with probability lapse_prob, generate random choice and RT
    if np.random.rand() < lapse_prob:
        choice = 1 if np.random.rand() >= 0.5 else -1
        rt = t_stim + np.random.uniform(0, T_lapse_max)  # Uniform distribution between 0 and T_lapse_max
        is_act = -1  # Mark as lapse
        return choice, rt, is_act

    # Normal simulation process (with probability 1 - lapse_prob)
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5

    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    # mu = (2*q_e/T_0) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
    # sigma = np.sqrt( (2*(q_e**2)/T_0) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )
    lambda_ABL_term = (10 ** (rate_lambda * (1 - rate_norm_l) * ABL / 20))
    lambda_ILD_arg = rate_lambda * ILD / chi
    lambda_ILD_L_arg = rate_lambda * rate_norm_l * ILD / chi
    mu = (1/T_0) * lambda_ABL_term * (np.sinh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg))
    sigma = np.sqrt( (1/T_0) * lambda_ABL_term * ( np.cosh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg) ) )

    is_act = 0
    while True:
        AI += V_A*dt + np.random.normal(0, dB)

        if t > t_stim + t_E_aff:
            DV += mu*dt + sigma*np.random.normal(0, dB)


        t += dt

        if DV >= theta:
            choice = +1; RT = t
            break
        elif DV <= -theta:
            choice = -1; RT = t
            break

        if AI >= theta_A:
            both_AI_hit_and_EA_hit = 0 # see if both AI and EA hit
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + del_go):
                if t > t_stim + t_E_aff:
                    DV += mu*dt + sigma*np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        both_AI_hit_and_EA_hit = 1
                        break
                    elif DV <= -theta:
                        DV = -theta
                        both_AI_hit_and_EA_hit = -1
                        break
                t += dt

            break


    if is_act == 1:
        RT = AI_hit_time
        if both_AI_hit_and_EA_hit != 0:
            choice = both_AI_hit_and_EA_hit
        else:
            randomly_choose_up = np.random.rand() >= 0.5
            if randomly_choose_up:
                choice = 1
            else:
                choice = -1

    return choice, RT, is_act

# %%
# read random animal params
animal_id = 112
batch_name = 'LED8'

# Read parameters for the animal
abort_params, tied_params = get_params_from_animal_pkl_file(batch_name, animal_id)

# Extract individual parameters for easier access
V_A = abort_params.get('V_A', np.nan)
theta_A = abort_params.get('theta_A', np.nan)
t_A_aff = abort_params.get('t_A_aff', np.nan)

rate_lambda = tied_params.get('rate_lambda', np.nan)
T_0 = tied_params.get('T_0', np.nan)
theta_E = tied_params.get('theta_E', np.nan)
w = tied_params.get('w', np.nan)
t_E_aff = tied_params.get('t_E_aff', np.nan)
del_go = tied_params.get('del_go', np.nan)

print(f"V_A = {V_A}, theta_A = {theta_A}, t_A_aff = {t_A_aff}")
print(f"rate_lambda = {rate_lambda}, T_0 = {T_0}, theta_E = {theta_E}, w = {w}")
print(f"t_E_aff = {t_E_aff}, del_go = {del_go}")


# %%
# For norm model, also get rate_norm_l
if IS_NORM_TIED:
    rate_norm_l = tied_params.get('rate_norm_l', np.nan)
else:
    rate_norm_l = 0

print(f"Loaded parameters for {batch_name} animal {animal_id}:")
print(f"Abort params: V_A={V_A}, theta_A={theta_A}, t_A_aff={t_A_aff}")
print(f"Tied params: rate_lambda={rate_lambda}, T_0={T_0}, theta_E={theta_E}, w={w}")
print(f"t_E_aff={t_E_aff}, del_go={del_go}, rate_norm_l={rate_norm_l}")

# %%
# Run simulations for lapses analysis
print("Starting simulations...")

# Read CSV data for the specific animal
file_name = f'../fit_animal_by_animal/batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
df = pd.read_csv(file_name)
df_animal = df[df['animal'] == animal_id]

# Filter for valid trials (successful responses with RT <= 1s)
# NOTE: temporaily test with removing <= 1 filter
# df_valid = df_animal[(df_animal['success'].isin([1, -1])) & (df_animal['RTwrtStim'] <= 1)]
df_valid = df_animal[(df_animal['success'].isin([1, -1]))]

# Set simulation parameters
N_sim = int(30e3)  # Number of simulations - increased for better statistics
dt = 1e-3    # Time step
N_print = int(N_sim/5) # Print progress every N_print iterations
lapse_prob = 0.1  # Probability of lapse (0.05 = 5% lapse rate)
T_lapse_max = 0.8  # Maximum RT for lapse trials (default 1.0 seconds)

# Sample data from the animal's trials
t_stim_samples = df_valid['intended_fix'].sample(N_sim, replace=True).values
ABL_samples = df_valid['ABL'].sample(N_sim, replace=True).values
ILD_samples = np.random.choice(df_valid['ILD'].values, size=N_sim, replace=True)

print(f"Running {N_sim} simulations with dt={dt}, lapse_prob={lapse_prob}...")

# Calculate Z_E from w and theta_E
Z_E = (w - 0.5) * 2 * theta_E

# Run simulations in parallel
sim_results = Parallel(n_jobs=30)(
    delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
        V_A, theta_A, ABL_samples[iter_num], ILD_samples[iter_num],
        rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go,
        t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt, lapse_prob, T_lapse_max
    ) for iter_num in tqdm(range(N_sim))
)

# Convert results to DataFrame
sim_results_df = pd.DataFrame(sim_results)
print(f"Simulation completed! Generated {len(sim_results_df)} simulation trials")

# Analyze results
print("\nSimulation Results Summary:")
print(f"Total simulations: {len(sim_results_df)}")
print(f"Mean RT: {sim_results_df['rt'].mean():.3f} s")
print(f"RT range: {sim_results_df['rt'].min():.3f} - {sim_results_df['rt'].max():.3f} s")
print(f"Choice distribution: Left={np.mean(sim_results_df['choice'] == -1):.3f}, Right={np.mean(sim_results_df['choice'] == 1):.3f}")
print(f"Lapse rate (is_act=1): {np.mean(sim_results_df['is_act'] == 1):.3f}")

# Filter trials where rt - t_stim is between 0 and 1 (valid response window)
sim_results_df['rt_minus_t_stim'] = sim_results_df['rt'] - sim_results_df['t_stim']
sim_results_df = sim_results_df[ ~( (sim_results_df['rt'] < sim_results_df['t_stim']) & (sim_results_df['rt'] < T_trunc) )]
# NOTE: temporaily test with removing <= 1 filter
# valid_rt_trials = sim_results_df[
#     (sim_results_df['rt_minus_t_stim'] >= 0) &
#     (sim_results_df['rt_minus_t_stim'] <= 1)
# ].copy()
valid_rt_trials = sim_results_df[
    (sim_results_df['rt_minus_t_stim'] >= 0)
].copy()

# print(f"Valid RT trials (0 < rt - t_stim < 1): {len(valid_rt_trials)} out of {len(sim_results_df)} total")
print(f"Valid RT trials (0 < rt - t_stim): {len(valid_rt_trials)} out of {len(sim_results_df)} total")

if len(valid_rt_trials) == 0:
    print("No valid RT trials found. Cannot create analysis plots.")
else:
    # Create 1x2 plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Define ABL colors
    abl_colors = {20: 'blue', 40: 'orange', 60: 'green'}

    # Left plot: RT - t_stim distribution by ABL
    ax1 = axes[0]

    # Bin edges for RT distribution
    bin_edges = np.arange(0, 3.005, 0.005)  # 0 to 1 with 0.05 step size
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for abl in [20, 40, 60]:
        abl_data = valid_rt_trials[valid_rt_trials['ABL'] == abl]
        if len(abl_data) > 0:
            rt_diffs = abl_data['rt_minus_t_stim'].values
            ax1.hist(rt_diffs, bins=bin_edges, density=True, histtype='step',
                    label=f'ABL {abl}', color=abl_colors[abl], linewidth=2)

    ax1.set_xlabel('RT - t_stim (s)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('RT Distribution by ABL (0 < RT - t_stim < 1)')
    ax1.legend()
    ax1.set_xlim(0, 1.2)

    # Right plot: Psychometric function by ABL
    ax2 = axes[1]

    ild_values = np.sort(valid_rt_trials['ILD'].unique())

    for abl in [20, 40, 60]:
        abl_data = valid_rt_trials[valid_rt_trials['ABL'] == abl]
        if len(abl_data) > 0:
            right_choice_probs = []
            for ild in ild_values:
                ild_trials = abl_data[abl_data['ILD'] == ild]
                if len(ild_trials) > 0:
                    prob_right = np.mean(ild_trials['choice'] == 1)
                    right_choice_probs.append(prob_right)
                else:
                    right_choice_probs.append(np.nan)

            ax2.plot(ild_values, right_choice_probs, 'o-',
                    label=f'ABL {abl}', color=abl_colors[abl], linewidth=2, markersize=6)

    ax2.set_xlabel('ILD (dB)')
    ax2.set_ylabel('P(Right Choice)')
    ax2.set_title('Psychometric Function by ABL')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlim(-17, 17)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'lapses_analysis_{batch_name}_{animal_id}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nSaved analysis plot to lapses_analysis_{batch_name}_{animal_id}.png")

    # Print summary statistics for valid trials
    print("\n=== Analysis Summary ===")
    print(f"Total valid RT trials: {len(valid_rt_trials)}")

    for abl in [20, 40, 60]:
        abl_data = valid_rt_trials[valid_rt_trials['ABL'] == abl]
        if len(abl_data) > 0:
            print(f"\nABL {abl}:")
            print(f"  Trials: {len(abl_data)}")
            print(f"  Mean RT - t_stim: {abl_data['rt_minus_t_stim'].mean():.3f} s")
            print(f"  P(Right choice): {np.mean(abl_data['choice'] == 1):.3f}")
            print(f"  Lapse rate: {np.mean(abl_data['is_act'] == -1):.3f}")

    # Overall lapse statistics
    lapse_trials = sim_results_df[sim_results_df['is_act'] == -1]
    normal_trials = sim_results_df[sim_results_df['is_act'] == 0]

    # Note: Lapse mechanism - with probability lapse_prob, trials have random choice and uniform RT (0-T_lapse_max s)
    # Normal trials follow the full DDM simulation process
    print("\n=== Lapse Analysis ===")
    print(f"Total lapse trials: {len(lapse_trials)} ({len(lapse_trials)/len(sim_results_df)*100:.1f}%)")
    print(f"Expected lapse rate (lapse_prob): {lapse_prob*100:.1f}%")
    print(f"Observed lapse rate: {len(lapse_trials)/len(sim_results_df)*100:.1f}%")

    if len(lapse_trials) > 0:
        print(f"Lapse RT distribution: {lapse_trials['rt'].min():.3f} - {lapse_trials['rt'].max():.3f} s")
        print(f"Mean lapse RT: {lapse_trials['rt'].mean():.3f} s")
        print(f"Lapse choice distribution: Left={np.mean(lapse_trials['choice'] == -1):.3f}, Right={np.mean(lapse_trials['choice'] == 1):.3f}")

    if len(normal_trials) > 0:
        print("\n=== Normal Trials Analysis ===")
        print(f"Normal trials: {len(normal_trials)} ({len(normal_trials)/len(sim_results_df)*100:.1f}%)")
        print(f"Normal RT range: {normal_trials['rt'].min():.3f} - {normal_trials['rt'].max():.3f} s")
        print(f"Mean normal RT: {normal_trials['rt'].mean():.3f} s")


# %%
# VBMC helper funcs
def compute_loglike_vanilla(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, lapse_prob, T_lapse_max):

    rt = row['rt']
    t_stim = row['t_stim']
    choice = row['choice']
    ILD = row['ILD']
    ABL = row['ABL']
    lapse_rt_window = T_lapse_max

    # Import the required functions
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fit_animal_by_animal'))
    from time_vary_norm_utils import up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn

    pdf = up_or_down_RTs_fit_fn(
            rt, choice,
            V_A, theta_A, t_A_aff,
            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
            phi_params_obj, rate_norm_l,
            is_norm, is_time_vary, K_max)

    # trunc_factor_p_joint = cum_pro_and_reactive_time_vary_fn(
    #                         t_stim + 1, T_trunc,
    #                         V_A, theta_A, t_A_aff,
    #                         t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
    #                         phi_params_obj, rate_norm_l,
    #                         is_norm, is_time_vary, K_max) \
    #                         - \
    #                         cum_pro_and_reactive_time_vary_fn(
    #                         t_stim, T_trunc,
    #                         V_A, theta_A, t_A_aff,
    #                         t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
    #                         phi_params_obj, rate_norm_l,
    #                         is_norm, is_time_vary, K_max)

    # NOTE: temporaily test with removing <= 1 filter
    trunc_factor_p_joint = 1 - cum_pro_and_reactive_time_vary_fn(
                            t_stim, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l,
                            is_norm, is_time_vary, K_max)

    pdf /= (trunc_factor_p_joint + 1e-20)

    # eg: lapse contribution = 0 when T_lapse max is 0.7, rt is 0.9
    in_lapse_window = (rt >= t_stim) and (rt < t_stim + lapse_rt_window)
    lapse_pdf = (0.5 / lapse_rt_window) if in_lapse_window else 0.0


    included_lapse_pdf = (1 - lapse_prob) * pdf + lapse_prob * lapse_pdf
    included_lapse_pdf = max(included_lapse_pdf, 1e-50)
    if np.isnan(included_lapse_pdf):
        print(f'nan pdf rt = {rt}, t_stim = {t_stim}')
        raise ValueError(f'nan pdf rt = {rt}, t_stim = {t_stim}')

    return np.log(included_lapse_pdf)


def vbmc_vanilla_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, T_lapse_max = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_vanilla)(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, lapse_prob, T_lapse_max)\
                                       for _, row in valid_rt_trials.iterrows() )
    return np.sum(all_loglike)


def vbmc_vanilla_tied_prior_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, T_lapse_max = params

    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fit_animal_by_animal'))
    from vbmc_animal_wise_fit_utils import trapezoidal_logpdf

    rate_lambda_logpdf = trapezoidal_logpdf(
        rate_lambda,
        vanilla_rate_lambda_bounds[0],
        vanilla_rate_lambda_plausible_bounds[0],
        vanilla_rate_lambda_plausible_bounds[1],
        vanilla_rate_lambda_bounds[1]
    )

    T_0_logpdf = trapezoidal_logpdf(
        T_0,
        vanilla_T_0_bounds[0],
        vanilla_T_0_plausible_bounds[0],
        vanilla_T_0_plausible_bounds[1],
        vanilla_T_0_bounds[1]
    )

    theta_E_logpdf = trapezoidal_logpdf(
        theta_E,
        vanilla_theta_E_bounds[0],
        vanilla_theta_E_plausible_bounds[0],
        vanilla_theta_E_plausible_bounds[1],
        vanilla_theta_E_bounds[1]
    )

    w_logpdf = trapezoidal_logpdf(
        w,
        vanilla_w_bounds[0],
        vanilla_w_plausible_bounds[0],
        vanilla_w_plausible_bounds[1],
        vanilla_w_bounds[1]
    )

    t_E_aff_logpdf = trapezoidal_logpdf(
        t_E_aff,
        vanilla_t_E_aff_bounds[0],
        vanilla_t_E_aff_plausible_bounds[0],
        vanilla_t_E_aff_plausible_bounds[1],
        vanilla_t_E_aff_bounds[1]
    )

    del_go_logpdf = trapezoidal_logpdf(
        del_go,
        vanilla_del_go_bounds[0],
        vanilla_del_go_plausible_bounds[0],
        vanilla_del_go_plausible_bounds[1],
        vanilla_del_go_bounds[1]
    )

    lapse_prob_logpdf = trapezoidal_logpdf(
        lapse_prob,
        vanilla_lapse_prob_bounds[0],
        vanilla_lapse_prob_plausible_bounds[0],
        vanilla_lapse_prob_plausible_bounds[1],
        vanilla_lapse_prob_bounds[1]
    )

    T_lapse_max_logpdf = trapezoidal_logpdf(
        T_lapse_max,
        vanilla_T_lapse_max_bounds[0],
        vanilla_T_lapse_max_plausible_bounds[0],
        vanilla_T_lapse_max_plausible_bounds[1],
        vanilla_T_lapse_max_bounds[1]
    )

    return (
        rate_lambda_logpdf +
        T_0_logpdf +
        theta_E_logpdf +
        w_logpdf +
        t_E_aff_logpdf +
        del_go_logpdf +
        lapse_prob_logpdf +
        T_lapse_max_logpdf
    )

def vbmc_vanilla_tied_joint_fn(params):
    priors = vbmc_vanilla_tied_prior_fn(params)
    loglike = vbmc_vanilla_tied_loglike_fn(params)
    return priors + loglike

# %%
# VBMC parameter bounds (tight around true values for easier convergence)
vanilla_rate_lambda_bounds = [0.05, 0.15]    # Tight around 0.097
vanilla_T_0_bounds = [0.00015, 0.00045]     # Tight around 0.000287
vanilla_theta_E_bounds = [25, 55]           # Tight around 40
vanilla_w_bounds = [0.45, 0.57]             # Tight around 0.508
vanilla_t_E_aff_bounds = [0.05, 0.10]       # Tight around 0.074
vanilla_del_go_bounds = [0.15, 0.24]        # Tight around 0.192
vanilla_lapse_prob_bounds = [0.07, 0.13]    # Tight around 0.1 (lapse probability)
vanilla_T_lapse_max_bounds = [0.7, 0.9]     # Tight around 0.8 (max lapse RT)

# Plausible bounds (tighter focus around true values)
vanilla_rate_lambda_plausible_bounds = [0.07, 0.12]    # Very tight around 0.097
vanilla_T_0_plausible_bounds = [0.0002, 0.0004]       # Very tight around 0.000287
vanilla_theta_E_plausible_bounds = [35, 45]            # Very tight around 40
vanilla_w_plausible_bounds = [0.48, 0.54]             # Very tight around 0.508
vanilla_t_E_aff_plausible_bounds = [0.065, 0.085]     # Very tight around 0.074
vanilla_del_go_plausible_bounds = [0.175, 0.210]       # Very tight around 0.192
vanilla_lapse_prob_plausible_bounds = [0.085, 0.115]   # Very tight around 0.1
vanilla_T_lapse_max_plausible_bounds = [0.75, 0.85]    # Very tight around 0.8

vanilla_tied_lb = np.array([
    vanilla_rate_lambda_bounds[0],
    vanilla_T_0_bounds[0],
    vanilla_theta_E_bounds[0],
    vanilla_w_bounds[0],
    vanilla_t_E_aff_bounds[0],
    vanilla_del_go_bounds[0],
    vanilla_lapse_prob_bounds[0],
    vanilla_T_lapse_max_bounds[0]
])

vanilla_tied_ub = np.array([
    vanilla_rate_lambda_bounds[1],
    vanilla_T_0_bounds[1],
    vanilla_theta_E_bounds[1],
    vanilla_w_bounds[1],
    vanilla_t_E_aff_bounds[1],
    vanilla_del_go_bounds[1],
    vanilla_lapse_prob_bounds[1],
    vanilla_T_lapse_max_bounds[1]
])

vanilla_plb = np.array([
    vanilla_rate_lambda_plausible_bounds[0],
    vanilla_T_0_plausible_bounds[0],
    vanilla_theta_E_plausible_bounds[0],
    vanilla_w_plausible_bounds[0],
    vanilla_t_E_aff_plausible_bounds[0],
    vanilla_del_go_plausible_bounds[0],
    vanilla_lapse_prob_plausible_bounds[0],
    vanilla_T_lapse_max_plausible_bounds[0]
])

vanilla_pub = np.array([
    vanilla_rate_lambda_plausible_bounds[1],
    vanilla_T_0_plausible_bounds[1],
    vanilla_theta_E_plausible_bounds[1],
    vanilla_w_plausible_bounds[1],
    vanilla_t_E_aff_plausible_bounds[1],
    vanilla_del_go_plausible_bounds[1],
    vanilla_lapse_prob_plausible_bounds[1],
    vanilla_T_lapse_max_plausible_bounds[1]
])

# %%
# VBMC fitting on simulated data
print("Starting VBMC fitting on simulated data...")

# Initialize parameters randomly from plausible bounds for better exploration
np.random.seed(42)  # For reproducibility

rate_lambda_0 = np.random.uniform(vanilla_rate_lambda_plausible_bounds[0], vanilla_rate_lambda_plausible_bounds[1])
T_0_0 = np.random.uniform(vanilla_T_0_plausible_bounds[0], vanilla_T_0_plausible_bounds[1])
theta_E_0 = np.random.uniform(vanilla_theta_E_plausible_bounds[0], vanilla_theta_E_plausible_bounds[1])
w_0 = np.random.uniform(vanilla_w_plausible_bounds[0], vanilla_w_plausible_bounds[1])
t_E_aff_0 = np.random.uniform(vanilla_t_E_aff_plausible_bounds[0], vanilla_t_E_aff_plausible_bounds[1])
del_go_0 = np.random.uniform(vanilla_del_go_plausible_bounds[0], vanilla_del_go_plausible_bounds[1])
lapse_prob_0 = np.random.uniform(vanilla_lapse_prob_plausible_bounds[0], vanilla_lapse_prob_plausible_bounds[1])
T_lapse_max_0 = np.random.uniform(vanilla_T_lapse_max_plausible_bounds[0], vanilla_T_lapse_max_plausible_bounds[1])

x_0 = np.array([
    rate_lambda_0,
    T_0_0,
    theta_E_0,
    w_0,
    t_E_aff_0,
    del_go_0,
    lapse_prob_0,
    T_lapse_max_0
])

print(f"Random initial parameters from plausible bounds:")
print(f"rate_lambda={rate_lambda_0:.3f}, T_0={T_0_0:.6f}, theta_E={theta_E_0:.1f}, w={w_0:.3f}")
print(f"t_E_aff={t_E_aff_0:.3f}, del_go={del_go_0:.3f}, lapse_prob={lapse_prob_0:.3f}, T_lapse_max={T_lapse_max_0:.3f}")

print(f"\nOriginal simulated parameters for comparison:")
print(f"rate_lambda: {rate_lambda:.3f}, T_0: {T_0:.6f}, theta_E: {theta_E:.1f}, w: {w:.3f}")
print(f"t_E_aff: {t_E_aff:.3f}, del_go: {del_go:.3f}")

# Run VBMC optimization
vbmc = VBMC(vbmc_vanilla_tied_joint_fn, x_0, vanilla_tied_lb, vanilla_tied_ub, vanilla_plb, vanilla_pub,
            options={'display': 'on'})
vp, results = vbmc.optimize()

# Save VBMC results
vbmc.save(f'vbmc_vanilla_tied_results_batch_{batch_name}_animal_{animal_id}_lapses.pkl', overwrite=True)

print("VBMC optimization completed!")
print(f"Results saved to vbmc_vanilla_tied_results_batch_{batch_name}_animal_{animal_id}_lapses.pkl")

# %%
# Sample from posterior and plot parameter distributions
vp_samples = vp.sample(int(1e6))[0]  # Sample 1000 points from posterior

print("\n=== VBMC Fitting Results ===")

# Create parameter plots with true values
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Parameter Posterior Distributions vs True Values', fontsize=16)

param_names = ['rate_lambda', 'T_0', 'theta_E', 'w', 't_E_aff', 'del_go', 'lapse_prob', 'T_lapse_max']
true_values = [rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob, T_lapse_max]

for i, (param_name, true_val) in enumerate(zip(param_names, true_values)):
    row, col = i // 3, i % 3
    ax = axes[row, col]

    # Plot histogram
    ax.hist(vp_samples[:, i], bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Add vertical line for true value
    ax.axvline(true_val, color='red', linestyle='--', linewidth=2, label=f'True: {true_val:.4f}')

    ax.set_title(f'{param_name} Distribution')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Density')
    ax.legend()

# Hide empty subplot if we have 8 parameters
if len(param_names) < 9:
    axes[2, 2].set_visible(False)

plt.tight_layout()
plt.savefig(f'parameter_posteriors_vs_true_batch_{batch_name}_animal_{animal_id}_N_{N_sim}.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("Parameter means from posterior samples:")
for i, param_name in enumerate(param_names):
    mean_val = np.mean(vp_samples[:, i])
    std_val = np.std(vp_samples[:, i])
    true_val = true_values[i]
    print(f"{param_name}: {mean_val:.4f} ± {std_val:.4f} (True: {true_val:.4f})")

print(f"\nOriginal simulated parameters:")
print(f"rate_lambda: {rate_lambda:.3f}, T_0: {T_0:.6f}, theta_E: {theta_E:.1f}, w: {w:.3f}")
print(f"t_E_aff: {t_E_aff:.3f}, del_go: {del_go:.3f}")
print(f"lapse_prob: {lapse_prob:.3f}, T_lapse_max: {T_lapse_max:.3f}")
# %%

