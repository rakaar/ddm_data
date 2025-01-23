# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from pyvbmc import VBMC
import corner
from psiam_tied_dv_map_utils import rho_A_t_fn, up_RTs_fit_fn, down_RTs_fit_fn, up_RTs_fit_single_t_fn, psiam_tied_data_gen_wrapper, psiam_tied_data_gen_wrapper_V2
from psiam_tied_dv_map_utils import down_RTs_fit_TRUNC_fn, up_RTs_fit_TRUNC_fn
from psiam_tied_no_dv_map_utils import cum_A_t_fn


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

# %% [markdown]
# # remove data points from df_1 for no second peak

# %%
bins = np.arange(0, 1, 0.01)
rt = df_1['timed_fix'] - df_1['intended_fix']

# Create a weight array initialized to 1s
weights = np.ones_like(rt)

# Identify the bump region between 0.20 and 0.30 s, and down-weight by 0.9
bump_mask = (rt >= 0.21) & (rt <= 0.33)
weights[bump_mask] = 0.8
plt.figure(figsize=(5,5))
# Plot the original histogram
plt.hist(rt, bins=bins, color='b', label='Original (weight=1)',
         density=True, histtype='step')

# Plot the weighted histogram
plt.hist(rt, bins=bins, color='r', label='Weighted (0.8 in [0.2,0.3])',
         weights=weights, density=True, histtype='step')

plt.xlabel("Reaction Time (s)")
plt.ylabel("Density")
plt.title("Comparison of Original vs. Weighted Histogram")
plt.legend()
plt.grid(True)
plt.xlim(0,1)
# xticks = np.arange(0,0.4,0.01)
# plt.xticks(xticks);


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

# %%
t_A_aff

# %% [markdown]
# # VBMC

# %% [markdown]
# ## loglike fn

# %%
T_trunc_aborts = 0.3

# %%
def compute_likelihood_and_weight(row, rate_lambda, T_0, theta_E, t_E_aff, Z_E, L):
    """Compute the *unweighted* likelihood for this row, plus its weight w_i."""
    timed_fix = row['timed_fix']
    intended_fix = row['intended_fix']
    ILD = row['ILD']
    ABL = row['ABL']
    choice = row['response_poke']

    rt = timed_fix
    t_stim = intended_fix

    K_max = 10

    trunc_factor = 1 - cum_A_t_fn(T_trunc_aborts - t_A_aff, V_A, theta_A)

    # -- Call your up/down RT fit functions (unweighted) --
    if rt < t_stim:
        likelihood = rho_A_t_fn(rt - t_A_aff - t_motor, V_A, theta_A)
    else:
        if choice == 3:
            likelihood = up_RTs_fit_fn(
                [rt], V_A, theta_A, ABL, ILD, rate_lambda, T_0,
                theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, K_max
            )[0]
        elif choice == 2:
            likelihood = down_RTs_fit_fn(
                [rt], V_A, theta_A, ABL, ILD, rate_lambda, T_0,
                theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, K_max
            )[0]
        else:
            # If not choice 2 or 3, return a "null" result
            return 1e-50, 1.0

    # Clip very small likelihoods
    if likelihood <= 0:
        likelihood = 1e-50

    likelihood /= trunc_factor
    # -- Compute weight w_i for bump region --
    w = 0.8 if 0.21 <= (rt - t_stim) <= 0.33 else 1.0
    
    return likelihood, w


def psiam_tied_loglike_fn(params):
    """Compute total log-likelihood with down-weighting in [0.21, 0.33], 
       then re-normalization so total mass stays the same.
    """
    rate_lambda, T_0, theta_E, t_E_aff, Z_E, L = params

    # 1) First pass: collect (likelihood, weight) for each data point
    results = Parallel(n_jobs=30)(
        delayed(compute_likelihood_and_weight)(
            row, rate_lambda, T_0, theta_E, t_E_aff, Z_E, L
        )
        for _, row in df_1.iterrows()
        if (row['timed_fix'] > row['intended_fix']) and (row['response_poke'] in [2, 3])
    )

    # Separate into arrays
    L_array = np.array([r[0] for r in results])  # unweighted likelihoods
    W_array = np.array([r[1] for r in results])  # weights

    # 2) Compute total mass before and after weighting
    M_orig = np.sum(L_array)             # sum of unweighted likelihoods
    M_weighted = np.sum(L_array * W_array)  # sum after weighting

    # 3) Normalization factor Z so that sum of final likelihood == sum of original
    #    i.e., M_weighted / Z = M_orig  =>  Z = M_weighted / M_orig
    if M_orig < 1e-50:  
        # Edge case: if M_orig=0 or extremely small, return a large negative loglike
        return -1e10

    Z = M_weighted / M_orig

    # 4) Final likelihood for each data point = (w_i * L_i) / Z
    #    Then sum of log-likelihood
    L_final = (W_array * L_array) / Z
    # Avoid log(0)
    L_final = np.clip(L_final, 1e-50, None)
    loglike = np.sum(np.log(L_final))

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

# %% [markdown]
# ## prior

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
    

def psiam_tied_prior_fn(params):
    rate_lambda, T_0, theta_E, t_E_aff, Z_E, L = params

    rate_lambda_logpdf = trapezoidal_logpdf(rate_lambda, rate_lambda_bounds[0], rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1], rate_lambda_bounds[1])
    theta_E_logpdf = trapezoidal_logpdf(theta_E, theta_E_bounds[0], theta_E_plausible_bounds[0], theta_E_plausible_bounds[1], theta_E_bounds[1])
    T_0_logpdf = trapezoidal_logpdf(T_0, T_0_bounds[0], T_0_plausible_bounds[0], T_0_plausible_bounds[1], T_0_bounds[1])
    
    t_E_aff_logpdf = trapezoidal_logpdf(t_E_aff, t_E_aff_bounds[0], t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1], t_E_aff_bounds[1])
    Z_E_logpdf = trapezoidal_logpdf(Z_E, Z_E_bounds[0], Z_E_plausible_bounds[0], Z_E_plausible_bounds[1], Z_E_bounds[1])
    L_logpdf = trapezoidal_logpdf(L, L_bounds[0], L_plausible_bounds[0], L_plausible_bounds[1], L_bounds[1])

    return rate_lambda_logpdf + T_0_logpdf + theta_E_logpdf + t_E_aff_logpdf + Z_E_logpdf + L_logpdf


# %% [markdown]
# ## prior + loglike

# %%
def psiam_tied_joint_fn(params):
    priors = psiam_tied_prior_fn(params) 
    loglike = psiam_tied_loglike_fn(params)

    joint = priors + loglike
    return joint

# %% [markdown]
# ## run vbmc

# %%
lb = np.array([ rate_lambda_bounds[0], T_0_bounds[0], theta_E_bounds[0], \
               t_E_aff_bounds[0], Z_E_bounds[0], L_bounds[0]])
ub = np.array([ rate_lambda_bounds[1], T_0_bounds[1], theta_E_bounds[1], \
                t_E_aff_bounds[1], Z_E_bounds[1], L_bounds[1]])

plb = np.array([ rate_lambda_plausible_bounds[0], T_0_plausible_bounds[0], theta_E_plausible_bounds[0], \
                t_E_aff_plausible_bounds[0], Z_E_plausible_bounds[0], L_plausible_bounds[0]])

pub = np.array([rate_lambda_plausible_bounds[1], T_0_plausible_bounds[1], theta_E_plausible_bounds[1], \
                t_E_aff_plausible_bounds[1], Z_E_plausible_bounds[1], L_plausible_bounds[1]])


np.random.seed(42)
rate_lambda_0 = np.random.uniform(rate_lambda_plausible_bounds[0], rate_lambda_plausible_bounds[1])
T_0_0 = np.random.uniform(T_0_plausible_bounds[0], T_0_plausible_bounds[1])
theta_E_0 = np.random.uniform(theta_E_plausible_bounds[0], theta_E_plausible_bounds[1])

t_E_aff_0 = np.random.uniform(t_E_aff_plausible_bounds[0], t_E_aff_plausible_bounds[1])
Z_E_0 = np.random.uniform(Z_E_plausible_bounds[0], Z_E_plausible_bounds[1])
L_0 = np.random.uniform(L_plausible_bounds[0], L_plausible_bounds[1])

x_0 = np.array([rate_lambda_0, T_0_0, theta_E_0, t_E_aff_0, Z_E_0, L_0])

vbmc = VBMC(psiam_tied_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on'})
vp, results = vbmc.optimize()

# %%
vbmc.save('include_aborts_reduced_wt_like.pkl')
