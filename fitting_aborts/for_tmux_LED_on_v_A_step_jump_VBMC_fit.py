# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.integrate import quad
import pandas as pd
import random
from pyvbmc import VBMC
from V_A_step_jump_fit_utils import PDF_hit_V_A_change, CDF_hit_V_A_change, rho_A_t_fn, cum_A_t_fn
from tqdm import tqdm


# %% [markdown]
# # data

# %%
# read out_LED.csv as dataframe
og_df = pd.read_csv('../out_LED.csv')

# chose non repeat trials - 0 or 2 or missing
df = og_df[ og_df['repeat_trial'].isin([0,2]) | og_df['repeat_trial'].isna() ]

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

# %%
df_to_fit = df[ df['LED_trial'] == 1 ] 

# %% [markdown]
# # VBMC

# %%
T_trunc = 0.3

# %%


def compute_loglike(row, base_V_A, new_V_A, theta_A, t_A_aff):
    rt = row['timed_fix']
    t_stim = row['intended_fix']
    t_led = row['intended_fix'] - row['LED_onset_time']

    
    if rt < T_trunc:
        likelihood = 0
    else:
        if t_led == 0: # only new V_A will be used
            pdf_trunc_factor = 1 - cum_A_t_fn(T_trunc - t_A_aff, new_V_A, theta_A)
            if rt < t_stim:
                likelihood =  rho_A_t_fn(rt - t_A_aff, new_V_A, theta_A) / pdf_trunc_factor
            elif rt > t_stim:
                if t_stim < T_trunc:
                    likelihood = 1
                else:
                    likelihood = ( 1 - cum_A_t_fn(t_stim - t_A_aff, new_V_A, theta_A) ) / pdf_trunc_factor
        else: # V_A change in middle
            trunc_factor = 1 - CDF_hit_V_A_change(T_trunc - t_A_aff, base_V_A, new_V_A, theta_A, t_led)
            if rt < t_stim:
                likelihood = PDF_hit_V_A_change(rt-t_A_aff, base_V_A, new_V_A, theta_A, t_led) / trunc_factor
            elif rt > t_stim:
                if t_stim < T_trunc: # if stim is before truncation, the abort prob = 0
                    likelihood = 1
                else:
                    likelihood = ( 1 - CDF_hit_V_A_change(t_stim - t_A_aff, base_V_A, new_V_A, theta_A, t_led) ) / trunc_factor

    if likelihood <= 0:
        likelihood = 1e-50

    
    return np.log(likelihood)    



def psiam_tied_loglike_fn(params):
    base_V_A, new_V_A, theta_A, t_A_aff = params

    all_loglike = Parallel(n_jobs=30)(delayed(compute_loglike)(row, base_V_A, new_V_A, theta_A, t_A_aff)\
                                      for _, row in df_to_fit.iterrows() \
                                        if (not np.isnan(row['timed_fix'] + row['intended_fix'] + row['LED_onset_time']))) 
                                   

    loglike = np.sum(all_loglike)
    return loglike

# %% [markdown]
# # bounds

# %%
base_V_A_bounds = [0.1, 5]
new_V_A_bounds = [0.1, 5]
theta_A_bounds = [0.1, 5]
t_A_aff_bounds = [-5, 0.1]

base_V_A_plausible_bounds = [0.5, 3]
new_V_A_plausible_bounds = [0.5, 3]
theta_A_plausible_bounds = [0.5, 3]
t_A_aff_plausible_bounds = [-2, 0.06]

# %% [markdown]
# # prior

# %%
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
    base_V_A, new_V_A, theta_A, t_A_aff = params

    base_V_A_logpdf = trapezoidal_logpdf(base_V_A, base_V_A_bounds[0], base_V_A_plausible_bounds[0], base_V_A_plausible_bounds[1], base_V_A_bounds[1])
    new_V_A_logpdf = trapezoidal_logpdf(new_V_A, new_V_A_bounds[0], new_V_A_plausible_bounds[0], new_V_A_plausible_bounds[1], new_V_A_bounds[1])
    theta_A_logpdf = trapezoidal_logpdf(theta_A, theta_A_bounds[0], theta_A_plausible_bounds[0], theta_A_plausible_bounds[1], theta_A_bounds[1])
    t_A_aff_logpdf = trapezoidal_logpdf(t_A_aff, t_A_aff_bounds[0], t_A_aff_plausible_bounds[0], t_A_aff_plausible_bounds[1], t_A_aff_bounds[1])
    return base_V_A_logpdf + new_V_A_logpdf + theta_A_logpdf + t_A_aff_logpdf

# %% [markdown]
# # joint

# %%
def vbmc_joint(params):
    return vbmc_prior_abort_fn(params) + psiam_tied_loglike_fn(params)

# %% [markdown]
# # run vbmc

# %%
lb = [base_V_A_bounds[0], new_V_A_bounds[0], theta_A_bounds[0], t_A_aff_bounds[0]]
ub = [base_V_A_bounds[1], new_V_A_bounds[1], theta_A_bounds[1], t_A_aff_bounds[1]]

plb = [base_V_A_plausible_bounds[0], new_V_A_plausible_bounds[0], theta_A_plausible_bounds[0], t_A_aff_plausible_bounds[0]]
pub = [base_V_A_plausible_bounds[1], new_V_A_plausible_bounds[1], theta_A_plausible_bounds[1], t_A_aff_plausible_bounds[1]]

np.random.seed(42)
base_V_A_0 = np.random.uniform(base_V_A_plausible_bounds[0], base_V_A_plausible_bounds[1])
new_V_A_0 = np.random.uniform(new_V_A_plausible_bounds[0], new_V_A_plausible_bounds[1])
theta_A_0 = np.random.uniform(theta_A_plausible_bounds[0], theta_A_plausible_bounds[1])
t_A_aff_0 = np.random.uniform(t_A_aff_plausible_bounds[0], t_A_aff_plausible_bounds[1])

x_0 = np.array([base_V_A_0, new_V_A_0, theta_A_0, t_A_aff_0])

# %%
vbmc = VBMC(vbmc_joint, x_0, lb, ub, plb, pub, options={'display': 'off'})
vp, results = vbmc.optimize()

# %%
# save vbmc
vp.save('V_A_step_jump_LED_on_vbmc.pkl', overwrite=True)
