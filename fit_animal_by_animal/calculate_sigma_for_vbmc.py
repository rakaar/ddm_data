# Calculate empirical standard deviations for VBMC sigma parameters
import numpy as np
import pandas as pd

# %%
# Data loading
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

ABL_vals = df_valid_animal_filtered['ABL'].unique()
ILD_vals = sorted(df_valid_animal_filtered['ILD'].unique())

# %%
# Collect log-odds data
print("="*60)
print("Calculating empirical standard deviations")
print("="*60)

all_x_logodds = []
all_y_logodds = []

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
                all_x_logodds.append(ild)
                all_y_logodds.append(log_odds_empirical)

all_x_logodds = np.array(all_x_logodds)
all_y_logodds = np.array(all_y_logodds)

# Remove NaN values
valid_mask = ~np.isnan(all_y_logodds)
all_x_logodds = all_x_logodds[valid_mask]
all_y_logodds = all_y_logodds[valid_mask]

# %%
# Collect psychometric data
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

# Remove NaN values
valid_mask_psyc = ~np.isnan(all_y_psyc)
all_x_psyc = all_x_psyc[valid_mask_psyc]
all_y_psyc = all_y_psyc[valid_mask_psyc]

# %%
# Calculate standard deviations
sigma_logodds = np.std(all_y_logodds)
sigma_psyc = np.std(all_y_psyc)

# Calculate additional statistics
mean_logodds = np.mean(all_y_logodds)
mean_psyc = np.mean(all_y_psyc)
range_logodds = np.max(all_y_logodds) - np.min(all_y_logodds)
range_psyc = np.max(all_y_psyc) - np.min(all_y_psyc)

# %%
# Print results
print(f"\nBatch: {batch_name}, Animal: {animal}")
print("-"*60)
print("\nLOG-ODDS DATA:")
print(f"  Number of data points: {len(all_y_logodds)}")
print(f"  Mean:                  {mean_logodds:.4f}")
print(f"  Std (sigma):           {sigma_logodds:.4f}")
print(f"  Range:                 {range_logodds:.4f}")
print(f"  Min:                   {np.min(all_y_logodds):.4f}")
print(f"  Max:                   {np.max(all_y_logodds):.4f}")

print("\nPSYCHOMETRIC DATA:")
print(f"  Number of data points: {len(all_y_psyc)}")
print(f"  Mean:                  {mean_psyc:.4f}")
print(f"  Std (sigma):           {sigma_psyc:.4f}")
print(f"  Range:                 {range_psyc:.4f}")
print(f"  Min:                   {np.min(all_y_psyc):.4f}")
print(f"  Max:                   {np.max(all_y_psyc):.4f}")

print("\n" + "="*60)
print("RECOMMENDED SIGMA VALUES FOR VBMC:")
print("="*60)
print(f"  sigma_logodds  = {sigma_logodds:.6f}")
print(f"  sigma_psyc     = {sigma_psyc:.6f}")
print("="*60)

# Save to file for reference
with open('empirical_sigma_values.txt', 'w') as f:
    f.write(f"Batch: {batch_name}, Animal: {animal}\n")
    f.write(f"sigma_logodds = {sigma_logodds:.6f}\n")
    f.write(f"sigma_psyc = {sigma_psyc:.6f}\n")

print("\nValues saved to: empirical_sigma_values.txt")
