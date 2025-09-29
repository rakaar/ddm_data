# VBMC fit on exp data single animal - lapses + vanila model
# %%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from pyvbmc import VBMC
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from vbmc_animal_wise_fit_utils import trapezoidal_logpdf
from time_vary_norm_utils import up_or_down_RTs_fit_fn, cum_pro_and_reactive_time_vary_fn
# %%
T_trunc = 0.3
batch_name = 'LED6'
phi_params_obj = np.nan
rate_norm_l = np.nan
is_norm = False
is_time_vary = False
K_max = 10

# load animal data directly from preprocessed batch CSV
csv_filename = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
exp_df = pd.read_csv(csv_filename)

# Data is already processed and batch-filtered in the CSV
### DF - valid and aborts ###
df_valid_and_aborts = exp_df[
    (exp_df['success'].isin([1,-1])) |
    (exp_df['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]

# animal_ids = df_valid_and_aborts['animal'].unique()
animal_ids = [86]
# animal = animal_ids[-1]
# for animal_idx in [-1]:

print('####################################')
print(f'Aborts Truncation Time: {T_trunc}')
print('####################################')
# %%

# load proactive params

# %%
# VBMC helper funcs
def compute_loglike_vanilla(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, lapse_prob):
    
    rt = row['TotalFixTime']
    t_stim = row['intended_fix']
    
    
    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['choice']
    lapse_rt_window = max_rt

    pdf = up_or_down_RTs_fit_fn(
            rt, choice,
            V_A, theta_A, t_A_aff,
            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go,
            phi_params_obj, rate_norm_l, 
            is_norm, is_time_vary, K_max)

    trunc_factor_p_joint = 1 - cum_pro_and_reactive_time_vary_fn(
                            t_stim, T_trunc,
                            V_A, theta_A, t_A_aff,
                            t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_E_aff,
                            phi_params_obj, rate_norm_l, 
                            is_norm, is_time_vary, K_max)
                           

    pdf /= (trunc_factor_p_joint + 1e-20)

    included_lapse_pdf = (1 - lapse_prob) * pdf  + lapse_prob * (0.5 * (1/lapse_rt_window))
    included_lapse_pdf = max(included_lapse_pdf, 1e-50)
    if np.isnan(included_lapse_pdf):
        print(f'row["abort_event"] = {row["abort_event"]}')
        print(f'row["RTwrtStim"] = {row["RTwrtStim"]}')
        raise ValueError(f'nan pdf rt = {rt}, t_stim = {t_stim}')

    
    return np.log(included_lapse_pdf)


def vbmc_vanilla_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob = params
    Z_E = (w - 0.5) * 2 * theta_E
    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike_vanilla)(row, rate_lambda, T_0, theta_E, Z_E, t_E_aff, del_go, lapse_prob)\
                                       for _, row in df_valid_animal.iterrows() )
    return np.sum(all_loglike)


def vbmc_vanilla_tied_prior_fn(params):
    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob = params

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
    
    return (
        rate_lambda_logpdf +
        T_0_logpdf +
        theta_E_logpdf +
        w_logpdf +
        t_E_aff_logpdf +
        del_go_logpdf +
        lapse_prob_logpdf
    )

def vbmc_vanilla_tied_joint_fn(params):
    priors = vbmc_vanilla_tied_prior_fn(params)
    loglike = vbmc_vanilla_tied_loglike_fn(params)

    return priors + loglike


vanilla_rate_lambda_bounds = [0.01, 1]
vanilla_T_0_bounds = [0.1e-3, 2.2e-3]
vanilla_theta_E_bounds = [5, 65]
vanilla_w_bounds = [0.3, 0.7]
vanilla_t_E_aff_bounds = [0.01, 0.2]
vanilla_del_go_bounds = [0, 0.2]
vanilla_lapse_prob_bounds = [1e-4, 0.2]

vanilla_rate_lambda_plausible_bounds = [0.1, 0.3]
vanilla_T_0_plausible_bounds = [0.5e-3, 1.5e-3]
vanilla_theta_E_plausible_bounds = [15, 55]
vanilla_w_plausible_bounds = [0.4, 0.6]
vanilla_t_E_aff_plausible_bounds = [0.03, 0.09]
vanilla_del_go_plausible_bounds = [0.05, 0.15]
vanilla_lapse_prob_plausible_bounds = [1e-3, 0.1]


vanilla_tied_lb = np.array([
    vanilla_rate_lambda_bounds[0],
    vanilla_T_0_bounds[0],
    vanilla_theta_E_bounds[0],
    vanilla_w_bounds[0],
    vanilla_t_E_aff_bounds[0],
    vanilla_del_go_bounds[0],
    vanilla_lapse_prob_bounds[0]
])

vanilla_tied_ub = np.array([
    vanilla_rate_lambda_bounds[1],
    vanilla_T_0_bounds[1],
    vanilla_theta_E_bounds[1],
    vanilla_w_bounds[1],
    vanilla_t_E_aff_bounds[1],
    vanilla_del_go_bounds[1],
    vanilla_lapse_prob_bounds[1]
])

vanilla_plb = np.array([
    vanilla_rate_lambda_plausible_bounds[0],
    vanilla_T_0_plausible_bounds[0],
    vanilla_theta_E_plausible_bounds[0],
    vanilla_w_plausible_bounds[0],
    vanilla_t_E_aff_plausible_bounds[0],
    vanilla_del_go_plausible_bounds[0],
    vanilla_lapse_prob_plausible_bounds[0]
])

vanilla_pub = np.array([
    vanilla_rate_lambda_plausible_bounds[1],
    vanilla_T_0_plausible_bounds[1],
    vanilla_theta_E_plausible_bounds[1],
    vanilla_w_plausible_bounds[1],
    vanilla_t_E_aff_plausible_bounds[1],
    vanilla_del_go_plausible_bounds[1],
    vanilla_lapse_prob_plausible_bounds[1]
])
# %%
print(f'len of animal_ids = {len(animal_ids)}')

# %%
for animal_idx in [0]:
    animal = animal_ids[animal_idx]

    df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
    df_aborts_animal = df_aborts[df_aborts['animal'] == animal]

    df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
    # no right Truncation
    # df_valid_animal_less_than_1 = df_valid_animal[df_valid_animal['RTwrtStim'] < 1]
    # max value of RTwrtStim
    max_rt = df_valid_animal['RTwrtStim'].max()


    print(f'Batch: {batch_name},sample animal: {animal}')
    pdf_filename = f'results_{batch_name}_animal_{animal}_lapse_fit.pdf'
    pdf = PdfPages(pdf_filename)
    fig_text = plt.figure(figsize=(8.5, 11)) # Standard page size looks better
    fig_text.clf() # Clear the figure
    fig_text.text(0.1, 0.9, f"Analysis Report", fontsize=20, weight='bold')
    fig_text.text(0.1, 0.8, f"Batch Name: {batch_name}", fontsize=14)
    fig_text.text(0.1, 0.75, f"Animal ID: {animal}", fontsize=14)
    fig_text.gca().axis("off")
    pdf.savefig(fig_text, bbox_inches='tight')
    plt.close(fig_text)

    ABL_arr = df_all_trials_animal['ABL'].unique()
    ILD_arr = df_all_trials_animal['ILD'].unique()


    # sort ILD arr in ascending order
    ILD_arr = np.sort(ILD_arr)
    ABL_arr = np.sort(ABL_arr)

    pkl_file = f'results_{batch_name}_animal_{animal}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    abort_keyname = "vbmc_aborts_results"

    abort_samples = fit_results_data[abort_keyname]
    abort_params = {}
    for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
        abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    
    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']

    rate_lambda_0 = 0.17
    T_0_0 = 1.4 * 1e-3
    theta_E_0 = 20
    w_0 = 0.51
    t_E_aff_0 = 0.071
    del_go_0 = 0.13
    lapse_prob_0 = 0.02
    x_0 = np.array([
        rate_lambda_0,
        T_0_0,
        theta_E_0,
        w_0,
        t_E_aff_0,
        del_go_0,
        lapse_prob_0,
    ])
    
    vbmc = VBMC(vbmc_vanilla_tied_joint_fn, x_0, vanilla_tied_lb, vanilla_tied_ub, vanilla_plb, vanilla_pub, options={'display': 'on', 'max_fun_evals': 200 * (2 + 6)})
    vp, results = vbmc.optimize()

    vbmc.save(f'vbmc_vanilla_tied_results_batch_{batch_name}_animal_{animal}_lapses.pkl', overwrite=True)

# %%

vp_samples = vp.sample(int(1e6))[0]
# %%
#    rate_lambda, T_0, theta_E, w, t_E_aff, del_go, lapse_prob = params
rate_lambda_samples = vp_samples[:, 0]
T_0_samples = vp_samples[:, 1]
theta_E_samples = vp_samples[:, 2]
w_samples = vp_samples[:, 3]
t_E_aff_samples = vp_samples[:, 4]
del_go_samples = vp_samples[:, 5]
lapse_prob_samples = vp_samples[:, 6]



# print mean of each sample
print("rate_lambda_samples mean: ", np.mean(rate_lambda_samples))
print("T_0_samples mean (ms): ", 1000* np.mean(T_0_samples))
print("theta_E_samples mean: ", np.mean(theta_E_samples))
print("w_samples mean: ", np.mean(w_samples))
# Z_E = (w - 0.5) * 2 * theta_E
print(f'Z_E = {(np.mean(w_samples) - 0.5) * 2 * np.mean(theta_E_samples)}')
print("t_E_aff_samples mean (ms): ", 1000* np.mean(t_E_aff_samples))
print("del_go_samples mean (ms): ", 1000* np.mean(del_go_samples))
print("lapse_prob_samples mean: ", np.mean(lapse_prob_samples))
#  %%
# 
bins = np.arange(vanilla_lapse_prob_bounds[0], vanilla_lapse_prob_bounds[1], 1e-4)
mean_lapse_prob = np.mean(lapse_prob_samples)
plt.hist(lapse_prob_samples, bins=bins, density=True)

plt.xlabel('Lapse probability')
plt.ylabel('Density')
plt.title('Lapse probability distribution')
plt.xlim(0, 1e-2)
plt.axvline(x=mean_lapse_prob, color='r', linestyle='--', label=f'Mean: {mean_lapse_prob:.4f}')
plt.legend()
plt.show()
# %%