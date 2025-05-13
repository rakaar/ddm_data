# %% 
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# %%
N_sim = int(1e6)
t_stim_samples = np.random.exponential(0.4, N_sim) + 0.2
ABL_arr = [20, 40, 60]
ILD_arr = [-8, -6, -4, -2, 2, 4, 6, 8]
ABL_samples = np.random.choice(ABL_arr, N_sim)
ILD_samples = np.random.choice(ILD_arr, N_sim)
# %%
# Posterior Means
V_A = 3.3587
theta_A = 2.1944
# theta_A = 5000
t_A_aff = 0.0303

rate_lambda = 2.8036
T_0 = 82.5288 * 1e-3
theta_E = 2.24  
w = 0.53
Z_E = (w - 0.5) * 2 * theta_E
t_E_aff = 92.4553 * 1e-3
del_go = 0.1303
rate_norm_l = 0.9347
# rate_norm_l = 0

N_print = int(N_sim / 5)
dt = 1e-3

sim_results = Parallel(n_jobs=30)(
    delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
        V_A, theta_A, ABL_samples[iter_num], ILD_samples[iter_num], rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go, 
        t_stim_samples[iter_num], rate_norm_l, iter_num, N_print, dt
    ) for iter_num in tqdm(range(N_sim))
)
sim_results_df = pd.DataFrame(sim_results)


# %%
valid_df = sim_results_df[sim_results_df['rt'] > sim_results_df['t_stim']]
# valid_df = valid_df[valid_df['rt'] - valid_df['t_stim'] < 1]

def plot_psycho(df_1):
    prob_choice_dict = {}
    
    all_ABL = np.sort(df_1['ABL'].unique())
    all_ILD = np.sort(ILD_arr)
    
    for abl in all_ABL:
        filtered_df = df_1[df_1['ABL'] == abl]
        prob_choice_dict[abl] = [
            sum(filtered_df[filtered_df['ILD'] == ild]['choice'] == 1) / len(filtered_df[filtered_df['ILD'] == ild])
            if len(filtered_df[filtered_df['ILD'] == ild]) > 0 else np.nan
            for ild in all_ILD
        ]
    
    return prob_choice_dict

# %% 
valid_df_psycho = plot_psycho(valid_df)
for key in valid_df_psycho.keys():
    plt.scatter(ILD_arr, valid_df_psycho[key], label=f'ABL={key}')
plt.legend()
plt.title(f'Psychometric, theta_A = {theta_A}, pro wins')
plt.xlabel('ILD')
plt.ylabel('Probability of Choice 1')
plt.show()

# %%
