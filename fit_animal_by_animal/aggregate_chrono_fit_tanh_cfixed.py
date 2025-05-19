# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --- Load and preprocess data as in aggregate_chrono.py ---
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]
all_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
], ignore_index=True)

# Only valid trials (success in {1, -1})
valid = all_data[all_data['success'].isin([1, -1])].copy()
valid = valid[valid['RTwrtStim'] <= 1]

# Filter for ABLs 20, 40, 60
valid = valid[valid['ABL'].isin([20, 40, 60])]

# Add abs_ILD if missing
if 'abs_ILD' not in valid.columns:
    valid['abs_ILD'] = valid['ILD'].abs()

# Only keep abs_ILD in [1,2,4,8,16]
valid['abs_ILD'] = valid['abs_ILD'].astype(float)
valid = valid[valid['abs_ILD'].isin([1.0,2.0,4.0,8.0,16.0])]

# batch name is nan fill LED7
valid['batch_name'] = valid['batch_name'].fillna('LED7')

abs_ilds = [1.0,2.0,4.0,8.0,16.0]
abl_list = [20, 40, 60]
abl_colors = {20:'#1f77b4', 40:'#ff7f0e', 60:'#2ca02c'}

# --- Step 1: Fit c per animal (shared across all ABLs for that animal) ---
def chrono_func(ILD, a, b, c):
    ILD = np.array(ILD)
    safe_ILD = np.where(ILD == 0, 1e-6, ILD)
    return a * np.tanh(b * safe_ILD) / safe_ILD + c

def chrono_func_ab(ILD, a, b, c):
    ILD = np.array(ILD)
    safe_ILD = np.where(ILD == 0, 1e-6, ILD)
    return a * np.tanh(b * safe_ILD) / safe_ILD + c

animal_keys = valid[['batch_name', 'animal']].drop_duplicates()
fit_params = {abl: [] for abl in abl_list}  # abl -> list of (a, b, c)
chronos = {abl: [] for abl in abl_list}     # abl -> list of (xvals, yvals)

for _, row in animal_keys.iterrows():
    batch = row['batch_name']
    animal = row['animal']
    # Gather all chronometric data for this animal (across ABLs)
    x_c = []
    y_c = []
    for abl in abl_list:
        animal_df = valid[(valid['batch_name'] == batch) & (valid['animal'] == animal) & (valid['ABL'] == abl)]
        for abs_ild in abs_ilds:
            subset = animal_df[animal_df['abs_ILD'] == abs_ild]
            if len(subset) > 0:
                val = subset['RTwrtStim'].mean()
                x_c.append(abs_ild)
                y_c.append(val)
    # Fit c for this animal
    try:
        popt_c, _ = curve_fit(chrono_func, x_c, y_c, p0=[0.2, 0.2, 0.2], maxfev=10000)
        c_fixed = popt_c[2]
    except Exception as e:
        c_fixed = 0.1
    # For each ABL, fit a, b with c fixed
    for abl in abl_list:
        animal_df = valid[(valid['batch_name'] == batch) & (valid['animal'] == animal) & (valid['ABL'] == abl)]
        xvals = []
        yvals = []
        for abs_ild in abs_ilds:
            subset = animal_df[animal_df['abs_ILD'] == abs_ild]
            if len(subset) > 0:
                val = subset['RTwrtStim'].mean()
                xvals.append(abs_ild)
                yvals.append(val)
        if len(xvals) < 2:
            continue  # not enough points to fit
        def fit_func(ILD, a, b):
            return chrono_func_ab(ILD, a, b, c_fixed)
        try:
            popt, _ = curve_fit(fit_func, xvals, yvals, p0=[0.2, 0.2], maxfev=10000)
            a, b = popt
            fit_params[abl].append((a, b, c_fixed))
            chronos[abl].append((xvals, yvals))
        except Exception as e:
            continue

# --- Step 3: Plot ---
fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
for i, abl in enumerate(abl_list):
    ax = axs[i]
    # Plot all animal fits as light lines
    xfit = np.linspace(min(abs_ilds), max(abs_ilds), 200)
    all_yfit = []
    for (a, b, c_fixed) in fit_params[abl]:
        yfit = chrono_func(xfit, a, b, c_fixed)
        all_yfit.append(yfit)
        ax.plot(xfit, yfit, color=abl_colors[abl], alpha=0.4, linewidth=2)
    # Plot mean of all fits in black
    if all_yfit:
        mean_yfit = np.mean(all_yfit, axis=0)
        ax.plot(xfit, mean_yfit, color='black', linewidth=1, label='Mean fit')
    # Plot group data points and error bars (as in aggregate_chrono.py)
    # For each abs_ILD, collect per-animal means
    per_animal_means = {abs_ild: [] for abs_ild in abs_ilds}
    for (xvals, yvals) in chronos[abl]:
        for x, y in zip(xvals, yvals):
            per_animal_means[x].append(y)
    means = []
    stds = []
    for abs_ild in abs_ilds:
        vals = per_animal_means[abs_ild]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    ax.errorbar(abs_ilds, means, yerr=stds, fmt='o', color=abl_colors[abl], capsize=0, markersize=8, label='Group mean')
    ax.set_title(f'ABL {abl}', fontsize=18)
    ax.set_xlabel('|ILD|', fontsize=16)
    ax.set_xticks(abs_ilds)
    ax.tick_params(axis='both', labelsize=14)
    if i == 0:
        ax.set_ylabel('Mean RT', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 0.5])
    ax.set_xlim([0, 17])
# --- Step 4: Overlay plot of mean fits and group means ---
ax = axs[3]
for i, abl in enumerate(abl_list):
    xfit = np.linspace(min(abs_ilds), max(abs_ilds), 200)
    all_yfit = []
    for (a, b, c_fixed) in fit_params[abl]:
        yfit = chrono_func(xfit, a, b, c_fixed)
        all_yfit.append(yfit)
    # Plot mean of all fits (previously black, now colored)
    if all_yfit:
        mean_yfit = np.mean(all_yfit, axis=0)
        ax.plot(xfit, mean_yfit, color=abl_colors[abl], linewidth=3, label=f'ABL {abl} mean fit')
    # Group mean data points and error bars
    per_animal_means = {abs_ild: [] for abs_ild in abs_ilds}
    for (xvals, yvals) in chronos[abl]:
        for x, y in zip(xvals, yvals):
            per_animal_means[x].append(y)
    means = []
    stds = []
    for abs_ild in abs_ilds:
        vals = per_animal_means[abs_ild]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    ax.errorbar(abs_ilds, means, yerr=stds, fmt='o', color=abl_colors[abl], capsize=0, markersize=8, label=f'ABL {abl} group mean')
ax.set_title('Overlay: Mean fits & group means', fontsize=18)
ax.set_xlabel('|ILD|', fontsize=16)
ax.set_xticks(abs_ilds)
ax.tick_params(axis='both', labelsize=14)
ax.set_ylabel('Mean RT', fontsize=16)
ax.legend(fontsize=12)
ax.set_ylim([0, 0.5])
ax.set_xlim([0, 17])
plt.tight_layout()
plt.show()

