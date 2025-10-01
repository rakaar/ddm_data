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
df_valid_animal_filtered = df_valid_animal[df_valid_animal['RTwrtStim'] > 0].copy()
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
def log_sigmoid_no_lapse(x, d, th, ILD_bias):
    """Log-odds with a=0 (no lapses)"""
    f =  th * np.tanh(d * (x + ILD_bias))
    p0 = 1.0 / (1.0 + (np.exp(-2*f)))  # sigma(f)
    # With a=0: pR = p0, pL = 1-p0
    eps = 1e-10  # small epsilon to avoid log(0)
    return np.log((p0 + eps) / ((1.0 - p0) + eps))

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

# Fit log_sigmoid_no_lapse to all data
# Bounds for [d, th, ILD_bias]
bounds = ([0.01, 0.01, -1.0], [10, 50, 1.0])
# Fixed initialization
p0 = [0.09, 3.0, -0.5]  # [d, th, ILD_bias]

try:
    popt, pcov = curve_fit(log_sigmoid_no_lapse, all_x_data, all_y_data, p0=p0, bounds=bounds)
    d_fit, th_fit, ILD_bias_fit = popt
    
    # Calculate R^2
    y_pred = log_sigmoid_no_lapse(all_x_data, d_fit, th_fit, ILD_bias_fit)
    r2 = r2_score(all_y_data, y_pred)
    
    print(f"Fitted parameters (across all 3 ABLs) - NO LAPSES (a=0):")
    print(f"  d        = {d_fit:.6f}")
    print(f"  th       = {th_fit:.6f}")
    print(f"  ILD_bias = {ILD_bias_fit:.6f}")
    print(f"  R²       = {r2:.4f}")
    
    # Plot the fitted curve on top of the empirical data
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(f'Log-Odds NO LAPSES (a=0) - Batch {batch_name}, Animal {animal_ids[0]} (R²={r2:.4f})', 
                 fontsize=14, fontweight='bold')
    
    x_model = np.linspace(-16, 16, 300)
    y_fitted = log_sigmoid_no_lapse(x_model, d_fit, th_fit, ILD_bias_fit)
    
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
                label=f'Fit (a=0): d={d_fit:.2f}, th={th_fit:.2f}, bias={ILD_bias_fit:.2f}')
        
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
# Note: With a=0 (no lapses), we don't need a separate biased lapse fit
# The above fit with a=0 is the only model needed for log-odds


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
def psyc_no_lapse(x, d, th, ILD_bias):
    """Psychometric function with a=0 (no lapses)"""
    f =  th * np.tanh(d * (x + ILD_bias))
    p0 = 1.0 / (1.0 + (np.exp(-2*f)))  # sigma(f)
    # With a=0: pR = p0
    return p0

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
# Fit psyc_no_lapse
# Bounds for [d, th, ILD_bias]
bounds_psyc = ([0.01, 0.01, -1.0], [10, 50, 1.0])
# Fixed initialization
p0_psyc = [0.09, 3.0, -0.5]  # [d, th, ILD_bias]

try:
    popt_psyc, pcov_psyc = curve_fit(psyc_no_lapse, all_x_psyc, all_y_psyc, 
                                     p0=p0_psyc, bounds=bounds_psyc)
    d_psyc, th_psyc, ILD_bias_psyc = popt_psyc
    
    # Calculate R^2
    y_pred_psyc = psyc_no_lapse(all_x_psyc, d_psyc, th_psyc, ILD_bias_psyc)
    r2_psyc = r2_score(all_y_psyc, y_pred_psyc)
    
    print(f"\nFitted parameters for psyc_no_lapse (across all 3 ABLs) - NO LAPSES (a=0):")
    print(f"  d        = {d_psyc:.6f}")
    print(f"  th       = {th_psyc:.6f}")
    print(f"  ILD_bias = {ILD_bias_psyc:.6f}")
    print(f"  R²       = {r2_psyc:.4f}")
    
    # Plot the fitted curve on top of the empirical data
    fig_psyc, axes_psyc = plt.subplots(1, 3, figsize=(15, 5))
    fig_psyc.suptitle(f'Psychometric NO LAPSES (a=0) - Batch {batch_name}, Animal {animal_ids[0]} (R²={r2_psyc:.4f})', 
                      fontsize=14, fontweight='bold')
    
    x_model = np.linspace(-16, 16, 300)
    y_fitted_psyc = psyc_no_lapse(x_model, d_psyc, th_psyc, ILD_bias_psyc)
    
    for idx, abl in enumerate(ABL_vals[:3]):
        ax = axes_psyc[idx]
        
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
        ax.plot(x_model, y_fitted_psyc, '-', color='red', linewidth=2, 
                label=f'Fit (a=0): d={d_psyc:.2f}, th={th_psyc:.2f}, bias={ILD_bias_psyc:.2f}')
        
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
    print(f"Fitting psyc_no_lapse failed: {e}")

# %%
# Note: With a=0 (no lapses), we don't need a separate biased lapse psychometric fit
# The above fit with a=0 is the only model needed for psychometric curves


# %%
# Comparison: Psychometric curves from log-odds fit vs direct psychometric fit
# Both should be identical since they use the same model (a=0, with ILD_bias)

# Create comparison plot
fig_comp, axes_comp = plt.subplots(1, 3, figsize=(20, 5))
fig_comp.suptitle(f'Comparison: Psychometric vs Log-Odds Fits (NO LAPSES) - Batch {batch_name}, Animal {animal_ids[0]}', 
                  fontsize=14, fontweight='bold')

x_model = np.linspace(-16, 16, 300)

# Psychometric curves from both fits
y_psyc_fit = psyc_no_lapse(x_model, d_psyc, th_psyc, ILD_bias_psyc)
y_logodds_fit = psyc_no_lapse(x_model, d_fit, th_fit, ILD_bias_fit)  # Convert log-odds params to P(right)

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
            label=f'Psyc fit: d={d_psyc:.2f}, th={th_psyc:.2f}, bias={ILD_bias_psyc:.2f}', alpha=0.8)
    
    # Plot log-odds fit converted to P(right)
    ax.plot(x_model, y_logodds_fit, '--', color='red', linewidth=2, 
            label=f'Log-odds fit: d={d_fit:.2f}, th={th_fit:.2f}, bias={ILD_bias_fit:.2f}', alpha=0.8)
    
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
print(f"{'lamda/chi-d':<15} {d_psyc:<20.6f} {d_fit:<20.6f}")
print(f"{'theta-th':<15} {th_psyc:<20.6f} {th_fit:<20.6f}")
print(f"{'ILD_bias':<15} {ILD_bias_psyc:<20.6f} {ILD_bias_fit:<20.6f}")
print(f"{'R²':<15} {r2_psyc:<20.4f} {r2:<20.4f}")
print("\nNote: a=0 (NO LAPSES) fixed for both fits")
print("="*60)