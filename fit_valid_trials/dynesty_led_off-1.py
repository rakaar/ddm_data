# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from pyvbmc import VBMC
import corner
from psiam_tied_dv_map_utils import rho_A_t_fn, up_RTs_fit_fn, down_RTs_fit_fn, up_RTs_fit_single_t_fn
import sys
import multiprocessing
from dynesty import NestedSampler
from dynesty import plotting as dyplot
import multiprocessing

# %%
# read out_LED.csv as dataframe
og_df = pd.read_csv('../out_LED.csv')

# %%
# chose non repeat trials - 0 or 2 or missing
df = og_df[ og_df['repeat_trial'].isin([0,2]) | og_df['repeat_trial'].isna() ]

# only session type 7
session_type = 7    
df = df[ df['session_type'].isin([session_type]) ]

# training level 16
training_level = 16
df = df[ df['training_level'].isin([training_level]) ]

# %%
# find ABL and ILD
ABL_arr = df['ABL'].unique()
ILD_arr = df['ILD'].unique()


# sort ILD arr in ascending order
ILD_arr = np.sort(ILD_arr)
ABL_arr = np.sort(ABL_arr)

print('ABL:', ABL_arr)
print('ILD:', ILD_arr)

# %%
# LED off rows
df_1 = df[ df['LED_trial'] == 0 ]
df_1 = df_1[ df_1['timed_fix'] > df_1['intended_fix'] ]

# %%
import pickle
with open('../fitting_aborts/post_led_censor_test_vbmc.pkl', 'rb') as f:
    vp = pickle.load(f)

# %%
vp_sample = vp.sample(int(1e6))[0]

# %%
V_A = np.mean(vp_sample[:,0])
theta_A = np.mean(vp_sample[:,1])
t_motor = 0.04
t_A_aff = np.mean(vp_sample[:,2]) - t_motor
# t_A_aff = 0.05 # NOTE: TEMP, to test if negative afferent delay is causing VBMC to not converge



# %% [markdown]
# # VBMC

# %% [markdown]
# ## loglike fn

# %%
def compute_loglike(row, rate_lambda, T_0, theta_E, t_E_aff, Z_E, L):
    timed_fix = row['timed_fix']
    intended_fix = row['intended_fix']
    
    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['response_poke']

    rt = timed_fix
    t_stim = intended_fix
    
    K_max = 10

    if choice == 3:
        likelihood = up_RTs_fit_fn([rt], V_A, theta_A, ABL, ILD, rate_lambda, T_0, \
                                    theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, K_max)[0]
    elif choice == 2:
        likelihood = down_RTs_fit_fn([rt], V_A, theta_A, ABL, ILD, rate_lambda, T_0,\
                                        theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, K_max)[0]


    if likelihood <= 0:
        likelihood = 1e-50

    
    return np.log(likelihood)    




# %%

def dynesty_psiam_tied_loglike_fn(params):
    rate_lambda, T_0, theta_E, t_E_aff, Z_E, L = params
    
    # Filter DataFrame rows first
    filtered_rows = df_1.loc[
        (df_1['timed_fix'] > df_1['intended_fix']) &
        (df_1['response_poke'].isin([2, 3]))
    ]
    
    # Prepare arguments for each row
    # We'll map over just the row data. The other parameters are closed over
    # from the outer scope.
    row_data = [row for _, row in filtered_rows.iterrows()]

    # Use multiprocessing Pool
    # You can adjust processes=30 to however many processes you need
    all_loglike = np.array([compute_loglike(row, rate_lambda, T_0, theta_E, t_E_aff, Z_E, L) for row in row_data])

    # Sum the log-likelihoods
    loglike = np.sum(all_loglike)

    # If loglike is inf or -inf or nan, return log(1e-50)
    if np.isnan(loglike) or np.isinf(loglike):
        return np.log(1e-50)
    else:
        return loglike


# %% [markdown]
# ## Bounds

# %%
rate_lambda_bounds = [0.01, 0.2]
theta_E_bounds = [30, 60]
T_0_bounds = [0.1*(1e-3), 1*(1e-3)]


t_E_aff_bounds = [0.001, 0.1]
Z_E_bounds = [-10, 10]
L_bounds = [0.1, 1.99]

# ---
rate_lambda_plausible_bounds =  [0.05, 0.09]
T_0_plausible_bounds = [0.15*(1e-3), 0.5*(1e-3)]
theta_E_plausible_bounds = [40, 55]

t_E_aff_plausible_bounds = [0.01, 0.05]
Z_E_plausible_bounds = [-5, 5]
L_plausible_bounds = [0.5, 1.5]




# %%
def trapezoidal_transform(u, a, b, c, d):
    """
    Maps a uniform random variable u ~ Uniform[0, 1] to a trapezoidal prior distribution.
    """
    area_left = (b - a) / 2
    area_middle = c - b
    area_right = (d - c) / 2
    total_area = area_left + area_middle + area_right

    # Normalize u to the total area
    u_scaled = u * total_area

    if u_scaled <= area_left:
        # Left ramp
        return a + (2 * u_scaled * (b - a))**0.5
    elif u_scaled <= area_left + area_middle:
        # Flat top
        return b + (u_scaled - area_left)
    else:
        # Right ramp
        return d - (2 * (total_area - u_scaled) * (d - c))**0.5

def prior_transform(u):
    """
    Transform a unit cube sample (u ~ Uniform[0,1]) to the parameter space using trapezoidal priors.
    """
    priors = np.zeros_like(u)

    # Map each parameter using the trapezoidal transform
    priors[0] = trapezoidal_transform(u[0], rate_lambda_bounds[0], rate_lambda_plausible_bounds[0], 
                                       rate_lambda_plausible_bounds[1], rate_lambda_bounds[1])
    priors[1] = trapezoidal_transform(u[1], T_0_bounds[0], T_0_plausible_bounds[0], 
                                       T_0_plausible_bounds[1], T_0_bounds[1])
    priors[2] = trapezoidal_transform(u[2], theta_E_bounds[0], theta_E_plausible_bounds[0], 
                                       theta_E_plausible_bounds[1], theta_E_bounds[1])
    priors[3] = trapezoidal_transform(u[3], t_E_aff_bounds[0], t_E_aff_plausible_bounds[0], 
                                       t_E_aff_plausible_bounds[1], t_E_aff_bounds[1])
    priors[4] = trapezoidal_transform(u[4], Z_E_bounds[0], Z_E_plausible_bounds[0], 
                                       Z_E_plausible_bounds[1], Z_E_bounds[1])
    priors[5] = trapezoidal_transform(u[5], L_bounds[0], L_plausible_bounds[0], 
                                       L_plausible_bounds[1], L_bounds[1])

    return priors




# %% [markdown]
# # dynesty

# %%
print("Starting dynesty run...")
N_processes = 30
pool = multiprocessing.Pool(processes=N_processes)
ndim = 6
sampler = NestedSampler(dynesty_psiam_tied_loglike_fn, prior_transform, ndim, pool=pool, queue_size=N_processes)
sampler.run_nested()
pool.close()
pool.join()

results = sampler.results

# Save with pickle
with open("dynesty_results.pkl", "wb") as f:
    pickle.dump(results, f)