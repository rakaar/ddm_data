# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.integrate import quad
import pandas as pd

from V_A_change_utils import PDF_hit_V_A_change, CDF_hit_V_A_change
from pyvbmc import VBMC

# %% [markdown]
# # Read data

# %%
# read out_LED.csv as dataframe
og_df = pd.read_csv('../out_LED.csv')

# chose non repeat trials - 0 or 2 or missing
df = og_df[ og_df['repeat_trial'].isin([0,2]) | og_df['repeat_trial'].isna() ].copy()

# only session type 7
session_type = 7    
df = df[ df['session_type'].isin([session_type]) ]

# training level 16
training_level = 16
df = df[ df['training_level'].isin([training_level]) ]

# find ABL and ILD
ABL_arr = df['ABL'].unique()
ILD_arr = df['ILD'].unique()


# sort ILD arr in ascending order
ILD_arr = np.sort(ILD_arr)
ABL_arr = np.sort(ABL_arr)

print('ABL:', ABL_arr)
print('ILD:', ILD_arr)

# %% [markdown]
# # df

# %%
# OFF
df_to_fit  = df[ df['LED_trial'] == 0 ]

# %% [markdown]
# # VBMC

# %%
T_trunc = 0.3

# %% [markdown]
# ## loglike fn

# %%
def compute_loglike(row, V_A_old, V_A_new, theta_A):
    t_stim = row['intended_fix']
    RT_wrt_fix = row['timed_fix']
    t_LED = row['LED_onset_time']
    RT_wrt_stim = RT_wrt_fix - t_stim

    if RT_wrt_fix < T_trunc:
        likelihood = 0
    elif RT_wrt_stim < 0: # abort
        # trunc factor
        prob_T_trunc_to_inf = quad(PDF_hit_V_A_change, T_trunc, 10, args=(V_A_old, V_A_new, theta_A, t_LED))[0]
        likelihood = PDF_hit_V_A_change(RT_wrt_fix, V_A_old, V_A_new, theta_A, t_LED)/prob_T_trunc_to_inf
    elif RT_wrt_stim > 0: # valid trial
        likelihood = quad(PDF_hit_V_A_change, t_stim, 10, args=(V_A_old, V_A_new, theta_A, t_LED))[0]
    else:
        print(f'RT_wrt_stim: {RT_wrt_stim}')


    if likelihood <= 0:
        likelihood = 1e-50
    
    return np.log(likelihood)    


def vbmc_loglike_abort_fn(params):
    V_A_old, V_A_new, theta_A = params # for now, lets ignore NDT

    


    # because there are less aborts, single job runs faster
    all_loglike = Parallel(n_jobs=-1)(delayed(compute_loglike)(row, V_A_old, V_A_new, theta_A)\
                                       for _, row in df_to_fit.iterrows() if not np.isnan(row['timed_fix'] - row['intended_fix']))

    loglike = np.sum(all_loglike)
    return loglike

# %% [markdown]
# # bounds, prior

# %%
V_A_old_bounds = [0.01, 5]
V_A_new_bounds = [0.01, 5]
theta_A_bounds = [0.01, 5]

V_A_old_plausible_bounds = [0.5,3]
V_A_new_plausible_bounds = [0.5,3]
theta_A_plausible_bounds = [0.5,3]

def trapezoidal_logpdf(x, a, b, c, d):
    if x < a or x > d:
        return -np.inf  # Logarithm of zero
    area = ((b - a) + (d - c)) / 2 + (c - b)
    h_max = 1.0 / area  # Height of the trapezoid to normalize the area to 1
    
    if a <= x <= b:
        pdf_value = ((x - a) / (b - a)) * h_max
    elif b < x < c:
        pdf_value = h_max
    elif c <= x <= d:
        pdf_value = ((d - x) / (d - c)) * h_max
    else:
        pdf_value = 0.0  # This case is redundant due to the initial check

    if pdf_value <= 0.0:
        return -np.inf
    else:
        return np.log(pdf_value)
    

def vbmc_prior_abort_fn(params):
    V_A_old, V_A_new, theta_A = params

    V_A_old_logpdf = trapezoidal_logpdf(V_A_old, V_A_old_bounds[0], V_A_old_plausible_bounds[0], V_A_old_plausible_bounds[1], V_A_old_bounds[1])
    V_A_new_logpdf = trapezoidal_logpdf(V_A_new, V_A_new_bounds[0], V_A_new_plausible_bounds[0], V_A_new_plausible_bounds[1], V_A_new_bounds[1])
    theta_A_logpdf = trapezoidal_logpdf(theta_A, theta_A_bounds[0], theta_A_plausible_bounds[0], theta_A_plausible_bounds[1], theta_A_bounds[1])

    return V_A_old_logpdf + V_A_new_logpdf + theta_A_logpdf



# %% [markdown]
# # loglike + prior

# %%
def vbmc_prior_plus_loglike_fn(params):
        return vbmc_loglike_abort_fn(params) + vbmc_prior_abort_fn(params)

# %% [markdown]
# # init and run vbmc

# %%
lb = [V_A_old_bounds[0], V_A_new_bounds[0], theta_A_bounds[0]]
ub = [V_A_old_bounds[1], V_A_new_bounds[1], theta_A_bounds[1]]

plb = [V_A_old_plausible_bounds[0], V_A_new_plausible_bounds[0], theta_A_plausible_bounds[0]]
pub = [V_A_old_plausible_bounds[1], V_A_new_plausible_bounds[1], theta_A_plausible_bounds[1]]

np.random.seed(42)
V_A_old_0 = np.random.uniform(V_A_old_plausible_bounds[0], V_A_old_plausible_bounds[1])
V_A_new_0 = np.random.uniform(V_A_new_plausible_bounds[0], V_A_new_plausible_bounds[1])
theta_A_0 = np.random.uniform(theta_A_plausible_bounds[0], theta_A_plausible_bounds[1])

x_0 = np.array([V_A_old_0, V_A_new_0, theta_A_0])

vbmc = VBMC(vbmc_prior_plus_loglike_fn, x_0, lb, ub, plb, pub, options={'display': 'off'})
vp, results = vbmc.optimize()


# %%
vbmc.save('vbmc_abort_no_NDT_LED_OFF.pkl')


