# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --- Load merged_valid as in mean_chrono_plot.py ---
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
], ignore_index=True)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
merged_valid['batch_name'] = merged_valid['batch_name'].fillna('LED7')

# remove RTs > 1 in valid trials
merged_valid = merged_valid[merged_valid['RTwrtStim'] <= 1]

# add abs_ILD column
merged_valid['abs_ILD'] = merged_valid['ILD'].abs()

# Remove ILD 10 and ILD 6, very few rows
merged_valid = merged_valid[~merged_valid['abs_ILD'].isin([6, 10])]

# Use mean or median RT (can change to 'TotalFixTime' if needed)
time_col = 'RTwrtStim'  # or 'TotalFixTime'

batch_names = merged_valid['batch_name'].unique()

# Collect animal list
animal_data = []
for batch_name in batch_names:
    batch_df = merged_valid[merged_valid['batch_name'] == batch_name]
    batch_animals = batch_df['animal'].unique()
    for animal in batch_animals:
        animal_df = batch_df[batch_df['animal'] == animal]
        animal_data.append({
            'batch_name': batch_name,
            'animal': animal,
            'df': animal_df
        })

# Define model function: E[T] = (a / (10**(b*ABL) * abs_ILD)) * tanh(c*abs_ILD) + t_nd
def chrono_func_full(X, a, b, c, t_nd):
    ABL, abs_ILD = X
    safe_abs_ILD = np.where(abs_ILD == 0, 1e-6, abs_ILD)
    return (a / (10 ** (b * ABL) * safe_abs_ILD)) * np.tanh(c * safe_abs_ILD) + t_nd

fit_results = []

for animal_info in animal_data:
    animal = animal_info['animal']
    batch_name = animal_info['batch_name']
    df = animal_info['df']
    # Compute mean RT for each (ABL, abs_ILD)
    grouped = df.groupby(['ABL', 'abs_ILD'])[time_col].mean().reset_index()
    # Remove rows with abs_ILD == 0 (to avoid division by zero)
    grouped = grouped[grouped['abs_ILD'] != 0]
    if len(grouped) < 4:
        print(f"Skipping animal {animal} (not enough data)")
        continue
    ABLs = grouped['ABL'].values
    abs_ILDs = grouped['abs_ILD'].values
    RTs = grouped[time_col].values
    # Initial guesses: a, b, c, t_nd
    a0 = 0.5
    b0 = 0.01
    c0 = 0.2
    tnd0 = 0.1
    try:
        popt, pcov = curve_fit(
            chrono_func_full,
            (ABLs, abs_ILDs),
            RTs,
            p0=[a0, b0, c0, tnd0],
            bounds=([0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, 0.5]),
            maxfev=10000
        )
        a, b, c, t_nd = popt
        fit_results.append({'animal': animal, 'batch_name': batch_name, 'a': a, 'b': b, 'c': c, 't_nd': t_nd})
        print(f"{animal} ({batch_name}): a={a:.4f}, b={b:.4f}, c={c:.4f}, t_nd={t_nd:.4f}")
    except Exception as e:
        print(f"Fit failed for animal {animal} ({batch_name}): {e}")

# Plotting: one subplot per animal, similar to fit_chrono_tanh_curve_c_fixed.py
num_animals = len(fit_results)
num_cols = 5
num_rows = (num_animals + num_cols - 1) // num_cols
subplot_size = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * subplot_size, num_rows * subplot_size))
axes = axes.flatten() if num_rows > 1 else [axes]

xticks = [0, 4, 8, 12, 16]
yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

for idx, fit_info in enumerate(fit_results):
    ax = axes[idx]
    animal = fit_info['animal']
    batch_name = fit_info['batch_name']
    a = fit_info['a']
    b = fit_info['b']
    c = fit_info['c']
    t_nd = fit_info['t_nd']
    # Get the grouped data for this animal
    df = merged_valid[(merged_valid['animal'] == animal) & (merged_valid['batch_name'] == batch_name)]
    grouped = df.groupby(['ABL', 'abs_ILD'])[time_col].mean().reset_index()
    grouped = grouped[grouped['abs_ILD'] != 0]
    # Plot data points for each ABL
    for abl in sorted(grouped['ABL'].unique()):
        abl_df = grouped[grouped['ABL'] == abl]
        xvals = abl_df['abs_ILD'].values
        yvals = abl_df[time_col].values
        ax.plot(xvals, yvals, 'o', label=f'ABL {abl}')
    # Overlay the fitted curve for each ABL (smooth)
    abs_ILD_grid = np.linspace(grouped['abs_ILD'].min(), grouped['abs_ILD'].max(), 200)
    for abl in sorted(grouped['ABL'].unique()):
        ABL_grid = np.full_like(abs_ILD_grid, abl)
        yfit = chrono_func_full((ABL_grid, abs_ILD_grid), a, b, c, t_nd)
        ax.plot(abs_ILD_grid, yfit, '-', label=f'Fit ABL {abl}')
    # Title with rounded parameters
    ax.set_title(f"{batch_name} | {animal}\na={a:.2f}, b={b:.2f}, c={c:.2f}, t_nd={t_nd:.2f}", fontsize=12)
    ax.set_xlim([grouped['abs_ILD'].min(), grouped['abs_ILD'].max()])
    ax.set_xticks(xticks)
    ax.set_ylim([0, 0.5])
    ax.set_yticks(yticks)
    if idx % num_cols != 0:
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.spines['left'].set_color('gray')
    else:
        ax.set_ylabel(f'{time_col} (mean)', fontsize=12)
        ax.spines['left'].set_color('black')
    if idx < (num_rows - 1) * num_cols:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel('| ILD |', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, frameon=True)

# Hide unused axes
for i in range(idx + 1, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Compute lambda, theta_E, T_0 for each animal using the fit parameters
results_table = []
for fit_info in fit_results:
    animal = fit_info['animal']
    batch_name = fit_info['batch_name']
    a = fit_info['a']
    b = fit_info['b']
    c = fit_info['c']
    t_nd = fit_info['t_nd']
    # Compute lambda
    lambd = 20 * b
    # Compute theta_E
    theta_E = c / lambd if lambd != 0 else np.nan
    # Compute T_0
    T_0 = (2 * a * lambd) / theta_E if theta_E != 0 else np.nan
    results_table.append({
        'animal': animal,
        'batch': batch_name,
        'lambda': round(lambd, 4),
        'theta_E': round(theta_E, 4),
        'T_0': round(T_0, 4),
        't_nd': round(t_nd, 4)
    })

# Print as a readable table
import pandas as pd
print("\nParameter table per animal:")
print(pd.DataFrame(results_table)[['animal', 'batch', 'lambda', 'theta_E', 'T_0', 't_nd']].to_string(index=False))

# %%
