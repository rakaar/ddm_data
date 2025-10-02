# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# %%
T_trunc = 0.3
batch_name = 'LED8'
animal_ids = [105]

DO_RIGHT_TRUNCATE = True
if DO_RIGHT_TRUNCATE:
    print(f'Right truncation at 1s')
else:
    print(f'No right truncation')

csv_filename = f'batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
exp_df = pd.read_csv(csv_filename)
df_valid_and_aborts = exp_df[
    (exp_df['success'].isin([1,-1])) |
    (exp_df['abort_event'] == 3)
].copy()

df_aborts = df_valid_and_aborts[df_valid_and_aborts['abort_event'] == 3]
animal_idx = 0
animal = animal_ids[animal_idx]
df_all_trials_animal = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
df_aborts_animal = df_aborts[df_aborts['animal'] == animal]
df_valid_animal = df_all_trials_animal[df_all_trials_animal['success'].isin([1,-1])]
max_rt = df_valid_animal['RTwrtStim'].max()

# Apply right truncation filter if enabled
if DO_RIGHT_TRUNCATE:
    df_valid_animal_filtered = df_valid_animal[
        (df_valid_animal['RTwrtStim'] > 0) & 
        (df_valid_animal['RTwrtStim'] <= 1.0)
    ].copy()
    print(f'Applied right truncation: {len(df_valid_animal)} -> {len(df_valid_animal_filtered)} trials')
else:
    df_valid_animal_filtered = df_valid_animal[df_valid_animal['RTwrtStim'] > 0].copy()
    print(f'No right truncation applied: {len(df_valid_animal_filtered)} trials')
df_valid_animal_filtered['abs_ILD'] = np.abs(df_valid_animal_filtered['ILD'])

# %%
ABL_vals = df_valid_animal_filtered['ABL'].unique()
ILD_vals = sorted(df_valid_animal_filtered['ILD'].unique())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Log-Odds: Empirical Data - Batch {batch_name}, Animal {animal_ids[0]}', 
             fontsize=14, fontweight='bold')

for idx, abl in enumerate(ABL_vals[:3]):  # 3 ABLs
    ax = axes[idx]
    
    empirical_log_odds = []
    
    for ild in ILD_vals:
        # Empirical data
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right = np.mean(empirical_subset['choice'] == 1)
            p_left = np.mean(empirical_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_empirical = np.log(p_right / p_left)
            else:
                log_odds_empirical = np.nan
        else:
            log_odds_empirical = np.nan
        empirical_log_odds.append(log_odds_empirical)
    
    # Plot
    ax.plot(ILD_vals, empirical_log_odds, 's', color='green', markersize=8, 
            label='Data', markerfacecolor='green', markeredgecolor='green', alpha=0.3)
    
    # Formatting
    ax.set_title(f'ABL = {abl} dB', fontsize=12)
    ax.set_xlabel('ILD (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('log(P(right) / P(left))', fontsize=11)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set consistent y-limits across subplots
    ax.set_ylim(-5, 5)

plt.tight_layout()
plt.show()
# %%
def log_sigmoid_v2(x, a, d, th):
    f =  th * np.tanh(d * x)
    p0 = 1.0 / (1.0 + (np.exp(-2*f)))  # sigma(f)
    pR = a/2.0 + (1.0 - a) * p0
    pL = a/2.0 + (1.0 - a) * (1.0 - p0)
    eps = 0
    return np.log((pR + eps) / (pL + eps))


def log_sigmoid_v3_lapse_biased(x, a, d, th, lapse_pR):
    f =  th * np.tanh(d * x)
    p0 = 1.0 / (1.0 + (np.exp(-2*f)))  # sigma(f)
    p_plus = a*lapse_pR + (1.0 - a) * p0
    p_minus = a*(1-lapse_pR) + (1.0 - a) * (1.0 - p0)
    eps = 0
    return np.log((p_plus + eps) / (p_minus + eps))

# %%
# Collect all log odds data across all 3 ABLs for fitting
all_x_data = []
all_y_data = []

for idx, abl in enumerate(ABL_vals[:3]):
    for ild in ILD_vals:
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right = np.mean(empirical_subset['choice'] == 1)
            p_left = np.mean(empirical_subset['choice'] == -1)
            if p_left > 0 and p_right > 0:
                log_odds_empirical = np.log(p_right / p_left)
                all_x_data.append(ild)
                all_y_data.append(log_odds_empirical)

all_x_data = np.array(all_x_data)
all_y_data = np.array(all_y_data)

# Remove any NaN values
valid_mask = ~np.isnan(all_y_data)
all_x_data = all_x_data[valid_mask]
all_y_data = all_y_data[valid_mask]

# Fit log_sigmoid_v2 to all data
bounds = ([0.0, 0.01, 0.01], [0.1, 10, 50])
# Fixed initialization
p0 = [0.02, 0.09, 3.0]  # [a, d, th]

try:
    popt, pcov = curve_fit(log_sigmoid_v2, all_x_data, all_y_data, p0=p0, bounds=bounds)
    a_fit, d_fit, th_fit = popt
    
    # Calculate R^2
    y_pred = log_sigmoid_v2(all_x_data, a_fit, d_fit, th_fit)
    r2 = r2_score(all_y_data, y_pred)
    
    print(f"Fitted parameters (across all 3 ABLs):")
    print(f"  a        = {a_fit:.6f}")
    print(f"  d        = {d_fit:.6f}")
    print(f"  th       = {th_fit:.6f}")
    print(f"  R²       = {r2:.4f}")
    
    # Plot the fitted curve on top of the empirical data
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(f'Log-Odds unbiased lapse- - Batch {batch_name}, Animal {animal_ids[0]} (R²={r2:.4f})', 
                 fontsize=14, fontweight='bold')
    
    x_model = np.linspace(-16, 16, 300)
    y_fitted = log_sigmoid_v2(x_model, a_fit, d_fit, th_fit)
    
    for idx, abl in enumerate(ABL_vals[:3]):
        ax = axes2[idx]
        
        empirical_log_odds = []
        empirical_ild_vals = []
        
        for ild in ILD_vals:
            empirical_subset = df_valid_animal_filtered[
                (df_valid_animal_filtered['ABL'] == abl) & 
                (df_valid_animal_filtered['ILD'] == ild)
            ]
            if len(empirical_subset) > 0:
                p_right = np.mean(empirical_subset['choice'] == 1)
                p_left = np.mean(empirical_subset['choice'] == -1)
                if p_left > 0 and p_right > 0:
                    log_odds_empirical = np.log(p_right / p_left)
                    empirical_log_odds.append(log_odds_empirical)
                    empirical_ild_vals.append(ild)
        
        # Plot empirical data
        ax.plot(empirical_ild_vals, empirical_log_odds, 's', color='green', markersize=8, 
                label='Data', markerfacecolor='green', markeredgecolor='green', alpha=0.3)
        
        # Plot fitted curve
        ax.plot(x_model, y_fitted, '-', color='red', linewidth=2, 
                label=f'Fit: a={a_fit:.3f}, d={d_fit:.2f}, th={th_fit:.2f}')
        
        # Formatting
        ax.set_title(f'ABL = {abl} dB', fontsize=12)
        ax.set_xlabel('ILD (dB)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('log(P(right) / P(left))', fontsize=11)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Fitting failed: {e}")

# %%
# Fit log_sigmoid_v3_lapse_biased (with biased lapse parameter)
# Bounds for [a, d, th, lapse_pR]
# a: (0, 0.1), d: (0.01, 10), th: (0.01, 50), lapse_pR: (0, 1)
bounds_v3 = ([0.0, 0.01, 0.01, 0.0], [0.1, 10, 50, 1.0])
# Fixed initialization
p0_v3 = [0.02, 0.09, 3.0, 0.5]  # [a, d, th, lapse_pR]

try:
    popt_v3, pcov_v3 = curve_fit(log_sigmoid_v3_lapse_biased, all_x_data, all_y_data, 
                                  p0=p0_v3, bounds=bounds_v3)
    a_fit_v3, d_fit_v3, th_fit_v3, lapse_pR_fit = popt_v3
    
    # Calculate R^2
    y_pred_v3 = log_sigmoid_v3_lapse_biased(all_x_data, a_fit_v3, d_fit_v3, th_fit_v3, lapse_pR_fit)
    r2_v3 = r2_score(all_y_data, y_pred_v3)
    
    print(f"\nFitted parameters for log_sigmoid_v3_lapse_biased (across all 3 ABLs):")
    print(f"  a        = {a_fit_v3:.6f}")
    print(f"  d        = {d_fit_v3:.6f}")
    print(f"  th       = {th_fit_v3:.6f}")
    print(f"  lapse_pR = {lapse_pR_fit:.6f}")
    print(f"  R²       = {r2_v3:.4f}")
    
    # Plot the fitted curve on top of the empirical data
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle(f'Log-Odds with Biased Lapse - Batch {batch_name}, Animal {animal_ids[0]} (R²={r2_v3:.4f})', 
                 fontsize=14, fontweight='bold')
    
    x_model = np.linspace(-16, 16, 300)
    y_fitted_v3 = log_sigmoid_v3_lapse_biased(x_model, a_fit_v3, d_fit_v3, th_fit_v3, lapse_pR_fit)
    
    for idx, abl in enumerate(ABL_vals[:3]):
        ax = axes3[idx]
        
        empirical_log_odds = []
        empirical_ild_vals = []
        
        for ild in ILD_vals:
            empirical_subset = df_valid_animal_filtered[
                (df_valid_animal_filtered['ABL'] == abl) & 
                (df_valid_animal_filtered['ILD'] == ild)
            ]
            if len(empirical_subset) > 0:
                p_right = np.mean(empirical_subset['choice'] == 1)
                p_left = np.mean(empirical_subset['choice'] == -1)
                if p_left > 0 and p_right > 0:
                    log_odds_empirical = np.log(p_right / p_left)
                    empirical_log_odds.append(log_odds_empirical)
                    empirical_ild_vals.append(ild)
        
        # Plot empirical data
        ax.plot(empirical_ild_vals, empirical_log_odds, 's', color='green', markersize=8, 
                label='Data', markerfacecolor='green', markeredgecolor='green', alpha=0.3)
        
        # Plot fitted curve
        ax.plot(x_model, y_fitted_v3, '-', color='blue', linewidth=2, 
                label=f'Fit: a={a_fit_v3:.3f}, d={d_fit_v3:.2f}, th={th_fit_v3:.2f}, pR={lapse_pR_fit:.3f}')
        
        # Formatting
        ax.set_title(f'ABL = {abl} dB', fontsize=12)
        ax.set_xlabel('ILD (dB)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('log(P(right) / P(left))', fontsize=11)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Fitting v3 (biased lapse) failed: {e}")


# %%
# psychometrics
# Psychometric curves: P(choice=1) vs ILD for each ABL
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Psychometric Curves: Empirical Data - Batch {batch_name}, Animal {animal_ids[0]}', 
             fontsize=14, fontweight='bold')

for idx, abl in enumerate(ABL_vals[:3]):  # 3 ABLs
    ax = axes[idx]
    
    empirical_p_right = []
    
    for ild in ILD_vals:
        # Empirical data
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right_empirical = np.mean(empirical_subset['choice'] == 1)
        else:
            p_right_empirical = np.nan
        empirical_p_right.append(p_right_empirical)
    
    # Plot
    ax.plot(ILD_vals, empirical_p_right, 's', color='green', markersize=8, 
            label='Data', markerfacecolor='green', markeredgecolor='green', alpha=0.3)
    
    # Formatting
    ax.set_title(f'ABL = {abl} dB', fontsize=12)
    ax.set_xlabel('ILD (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('P(choice = right)', fontsize=11)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set consistent y-limits
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# %%
def psyc_lapse_UNBIASED(x, a, d, th):
    f =  th * np.tanh(d * x)
    p0 = 1.0 / (1.0 + (np.exp(-2*f)))  # sigma(f)
    pR = a/2.0 + (1.0 - a) * p0
    return pR


def psyc_lapse_biased(x, a, d, th, lapse_pR):
    f =  th * np.tanh(d * x)
    p0 = 1.0 / (1.0 + (np.exp(-2*f)))  # sigma(f)
    p_plus = a*lapse_pR + (1.0 - a) * p0
    return p_plus

# %%
# Collect all psychometric data across all 3 ABLs for fitting
all_x_psyc = []
all_y_psyc = []

for idx, abl in enumerate(ABL_vals[:3]):
    for ild in ILD_vals:
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right_empirical = np.mean(empirical_subset['choice'] == 1)
            all_x_psyc.append(ild)
            all_y_psyc.append(p_right_empirical)

all_x_psyc = np.array(all_x_psyc)
all_y_psyc = np.array(all_y_psyc)

# Remove any NaN values
valid_mask_psyc = ~np.isnan(all_y_psyc)
all_x_psyc = all_x_psyc[valid_mask_psyc]
all_y_psyc = all_y_psyc[valid_mask_psyc]

# %%
# Fit psyc_lapse_UNBIASED (3 parameters)
bounds_psyc_unbiased = ([0.0, 0.01, 0.01], [0.1, 10, 50])
# Fixed initialization
p0_psyc_unbiased = [0.02, 0.09, 3.0]  # [a, d, th]

try:
    popt_psyc_unbiased, pcov_psyc_unbiased = curve_fit(psyc_lapse_UNBIASED, all_x_psyc, all_y_psyc, 
                                                         p0=p0_psyc_unbiased, bounds=bounds_psyc_unbiased)
    a_psyc_unbiased, d_psyc_unbiased, th_psyc_unbiased = popt_psyc_unbiased
    
    # Calculate R^2
    y_pred_psyc_unbiased = psyc_lapse_UNBIASED(all_x_psyc, a_psyc_unbiased, d_psyc_unbiased, th_psyc_unbiased)
    r2_psyc_unbiased = r2_score(all_y_psyc, y_pred_psyc_unbiased)
    
    print(f"\nFitted parameters for psyc_lapse_UNBIASED (across all 3 ABLs):")
    print(f"  a        = {a_psyc_unbiased:.6f}")
    print(f"  d        = {d_psyc_unbiased:.6f}")
    print(f"  th       = {th_psyc_unbiased:.6f}")
    print(f"  R²       = {r2_psyc_unbiased:.4f}")
    
    # Plot the fitted curve on top of the empirical data
    fig_psyc_unbiased, axes_psyc_unbiased = plt.subplots(1, 3, figsize=(15, 5))
    fig_psyc_unbiased.suptitle(f'Psychometric with Unbiased Lapse Fit - Batch {batch_name}, Animal {animal_ids[0]} (R²={r2_psyc_unbiased:.4f})', 
                                fontsize=14, fontweight='bold')
    
    x_model = np.linspace(-16, 16, 300)
    y_fitted_psyc_unbiased = psyc_lapse_UNBIASED(x_model, a_psyc_unbiased, d_psyc_unbiased, th_psyc_unbiased)
    
    for idx, abl in enumerate(ABL_vals[:3]):
        ax = axes_psyc_unbiased[idx]
        
        empirical_p_right = []
        empirical_ild_vals = []
        
        for ild in ILD_vals:
            empirical_subset = df_valid_animal_filtered[
                (df_valid_animal_filtered['ABL'] == abl) & 
                (df_valid_animal_filtered['ILD'] == ild)
            ]
            if len(empirical_subset) > 0:
                p_right_empirical = np.mean(empirical_subset['choice'] == 1)
                empirical_p_right.append(p_right_empirical)
                empirical_ild_vals.append(ild)
        
        # Plot empirical data
        ax.plot(empirical_ild_vals, empirical_p_right, 's', color='green', markersize=8, 
                label='Data', markerfacecolor='green', markeredgecolor='green', alpha=0.3)
        
        # Plot fitted curve
        ax.plot(x_model, y_fitted_psyc_unbiased, '-', color='red', linewidth=2, 
                label=f'Fit: a={a_psyc_unbiased:.3f}, d={d_psyc_unbiased:.2f}, th={th_psyc_unbiased:.2f}')
        
        # Formatting
        ax.set_title(f'ABL = {abl} dB', fontsize=12)
        ax.set_xlabel('ILD (dB)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('P(choice = right)', fontsize=11)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Fitting psyc_lapse_UNBIASED failed: {e}")

# %%
# Fit psyc_lapse_biased (4 parameters)
bounds_psyc_biased = ([0.0, 0.01, 0.01, 0.0], [0.1, 10, 50, 1.0])
# Fixed initialization
p0_psyc_biased = [0.02, 0.09, 3.0, 0.5]  # [a, d, th, lapse_pR]

try:
    popt_psyc_biased, pcov_psyc_biased = curve_fit(psyc_lapse_biased, all_x_psyc, all_y_psyc, 
                                                     p0=p0_psyc_biased, bounds=bounds_psyc_biased)
    a_psyc_biased, d_psyc_biased, th_psyc_biased, lapse_pR_psyc = popt_psyc_biased
    
    # Calculate R^2
    y_pred_psyc_biased = psyc_lapse_biased(all_x_psyc, a_psyc_biased, d_psyc_biased, th_psyc_biased, lapse_pR_psyc)
    r2_psyc_biased = r2_score(all_y_psyc, y_pred_psyc_biased)
    
    print(f"\nFitted parameters for psyc_lapse_biased (across all 3 ABLs):")
    print(f"  a        = {a_psyc_biased:.6f}")
    print(f"  d        = {d_psyc_biased:.6f}")
    print(f"  th       = {th_psyc_biased:.6f}")
    print(f"  lapse_pR = {lapse_pR_psyc:.6f}")
    print(f"  R²       = {r2_psyc_biased:.4f}")
    
    # Plot the fitted curve on top of the empirical data
    fig_psyc_biased, axes_psyc_biased = plt.subplots(1, 3, figsize=(15, 5))
    fig_psyc_biased.suptitle(f'Psychometric with Biased Lapse Fit - Batch {batch_name}, Animal {animal_ids[0]} (R²={r2_psyc_biased:.4f})', 
                              fontsize=14, fontweight='bold')
    
    x_model = np.linspace(-16, 16, 300)
    y_fitted_psyc_biased = psyc_lapse_biased(x_model, a_psyc_biased, d_psyc_biased, th_psyc_biased, lapse_pR_psyc)
    
    for idx, abl in enumerate(ABL_vals[:3]):
        ax = axes_psyc_biased[idx]
        
        empirical_p_right = []
        empirical_ild_vals = []
        
        for ild in ILD_vals:
            empirical_subset = df_valid_animal_filtered[
                (df_valid_animal_filtered['ABL'] == abl) & 
                (df_valid_animal_filtered['ILD'] == ild)
            ]
            if len(empirical_subset) > 0:
                p_right_empirical = np.mean(empirical_subset['choice'] == 1)
                empirical_p_right.append(p_right_empirical)
                empirical_ild_vals.append(ild)
        
        # Plot empirical data
        ax.plot(empirical_ild_vals, empirical_p_right, 's', color='green', markersize=8, 
                label='Data', markerfacecolor='green', markeredgecolor='green', alpha=0.3)
        
        # Plot fitted curve
        ax.plot(x_model, y_fitted_psyc_biased, '-', color='blue', linewidth=2, 
                label=f'Fit: a={a_psyc_biased:.3f}, d={d_psyc_biased:.2f}, th={th_psyc_biased:.2f}, pR={lapse_pR_psyc:.3f}')
        
        # Formatting
        ax.set_title(f'ABL = {abl} dB', fontsize=12)
        ax.set_xlabel('ILD (dB)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('P(choice = right)', fontsize=11)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Fitting psyc_lapse_biased failed: {e}")


# %%
# Comparison: Psychometric curves from both fitting methods
def psyc_from_logodds_params(x, a, d, th, lapse_pR):
    """Convert log-odds model parameters to P(right)"""
    f = th * np.tanh(d * x)
    p0 = 1.0 / (1.0 + np.exp(-2*f))
    p_plus = a*lapse_pR + (1.0 - a) * p0
    return p_plus

# Create comparison plot
fig_comp, axes_comp = plt.subplots(1, 3, figsize=(15, 5))
fig_comp.suptitle(f'Comparison: Psychometric vs Log-Odds Fits - Batch {batch_name}, Animal {animal_ids[0]}', 
                  fontsize=14, fontweight='bold')

x_model = np.linspace(-16, 16, 300)

# Psychometric curves from both fits
y_psyc_fit = psyc_lapse_biased(x_model, a_psyc_biased, d_psyc_biased, th_psyc_biased, lapse_pR_psyc)
y_logodds_fit = psyc_from_logodds_params(x_model, a_fit_v3, d_fit_v3, th_fit_v3, lapse_pR_fit)

for idx, abl in enumerate(ABL_vals[:3]):
    ax = axes_comp[idx]
    
    # Collect empirical data
    empirical_p_right = []
    empirical_ild_vals = []
    
    for ild in ILD_vals:
        empirical_subset = df_valid_animal_filtered[
            (df_valid_animal_filtered['ABL'] == abl) & 
            (df_valid_animal_filtered['ILD'] == ild)
        ]
        if len(empirical_subset) > 0:
            p_right_empirical = np.mean(empirical_subset['choice'] == 1)
            empirical_p_right.append(p_right_empirical)
            empirical_ild_vals.append(ild)
    
    # Plot empirical data
    ax.plot(empirical_ild_vals, empirical_p_right, 's', color='black', markersize=8, 
            label='Data', markerfacecolor='none', markeredgecolor='black', alpha=0.7, markeredgewidth=2)
    
    # Plot psychometric fit (P(right) space)
    ax.plot(x_model, y_psyc_fit, '-', color='blue', linewidth=2, 
            label=f'Psyc fit (pR={lapse_pR_psyc:.3f})', alpha=0.8)
    
    # Plot log-odds fit converted to P(right)
    ax.plot(x_model, y_logodds_fit, '--', color='red', linewidth=2, 
            label=f'Log-odds fit (pR={lapse_pR_fit:.3f})', alpha=0.8)
    
    # Formatting
    ax.set_title(f'ABL = {abl} dB', fontsize=12)
    ax.set_xlabel('ILD (dB)', fontsize=11)
    if idx == 0:
        ax.set_ylabel('P(choice = right)', fontsize=11)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# %%
# Print comparison
print("\n" + "="*60)
print("COMPARISON OF FITTED PARAMETERS")
print("="*60)
print(f"{'Parameter':<15} {'Psychometric Fit':<20} {'Log-Odds Fit':<20}")
print("-"*60)
print(f"{'lapse rate-a':<15} {a_psyc_biased:<20.6f} {a_fit_v3:<20.6f}")
print(f"{'lamda/chi-d':<15} {d_psyc_biased:<20.6f} {d_fit_v3:<20.6f}")
print(f"{'theta-th':<15} {th_psyc_biased:<20.6f} {th_fit_v3:<20.6f}")
print(f"{'lapse_pR':<15} {lapse_pR_psyc:<20.6f} {lapse_pR_fit:<20.6f}")
print(f"{'R²':<15} {r2_psyc_biased:<20.4f} {r2_v3:<20.4f}")
print("="*60)