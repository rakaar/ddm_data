# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal'
CSV_PATH = os.path.join(BASE_DIR, 'vanilla_norm_lapse_loglike_comparison_v2.csv')
LAPSE_PARAMS_PATH = os.path.join(BASE_DIR, 'lapse_parameters_all_animals.pkl')

# %%
df = pd.read_csv(CSV_PATH)
with open(LAPSE_PARAMS_PATH, 'rb') as handle:
    lapse_params = pickle.load(handle)

# %%
# Norm - (Vanilla + lapse) vs lapse rate
lapse_rates = []
loglike_diffs = []

for _, row in df.iterrows():
    batch = row.get('batch')
    animal = row.get('animal')
    if pd.isna(batch) or pd.isna(animal):
        continue
    key = (batch, int(animal))
    lapse_entry = lapse_params.get(key)
    if not lapse_entry:
        continue
    lapse_prob = lapse_entry.get('vanilla_lapse', {}).get('lapse_prob')
    if lapse_prob is None or pd.isna(lapse_prob):
        continue
    og_norm_ll = row.get('og_norm_loglike')
    vanilla_lapse_ll = row.get('vanilla_lapse_loglike')
    if pd.isna(og_norm_ll) or pd.isna(vanilla_lapse_ll):
        continue
    lapse_rates.append(lapse_prob * 100)
    loglike_diffs.append(og_norm_ll - vanilla_lapse_ll)

norm_minus_vanilla_lapse_data = {
    'lapse_rate_pct': np.array(lapse_rates),
    'loglike_diff': np.array(loglike_diffs),
    'x_label': 'Lapse rate (%)',
    'y_label': 'NPL - (IPL + lapses)\nLL',
}

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['green' if diff > 0 else 'red' for diff in loglike_diffs]
ax.scatter(lapse_rates, loglike_diffs, c=colors, alpha=0.7, s=45, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

ax.set_xlabel('Lapse rate (%)', fontsize=14)
ax.set_ylabel('NPL - (IPL + lapses)\nLL', fontsize=14)
ax.set_title('Lapse rate vs LL diff', fontsize=14)
ax.set_xticks([0, 6])
ax.set_yticks([-300, 0, 300])
ax.tick_params(axis='both', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

output_prefix = os.path.join(BASE_DIR, 'lapse_rate_vs_norm_minus_vanilla_lapse_loglike')
png_path = f'{output_prefix}.png'
pdf_path = f'{output_prefix}.pdf'
plt.tight_layout()
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.show()

print(f'Saved: {png_path}')
print(f'Saved: {pdf_path}')

# %%
# Norm + lapse - (Vanilla + lapse) vs lapse rate
lapse_rates = []
loglike_diffs = []

for _, row in df.iterrows():
    batch = row.get('batch')
    animal = row.get('animal')
    if pd.isna(batch) or pd.isna(animal):
        continue
    key = (batch, int(animal))
    lapse_entry = lapse_params.get(key)
    if not lapse_entry:
        continue
    lapse_prob = lapse_entry.get('vanilla_lapse', {}).get('lapse_prob')
    if lapse_prob is None or pd.isna(lapse_prob):
        continue
    norm_lapse_ll = row.get('norm_lapse_loglike')
    vanilla_lapse_ll = row.get('vanilla_lapse_loglike')
    if pd.isna(norm_lapse_ll) or pd.isna(vanilla_lapse_ll):
        continue
    lapse_rates.append(lapse_prob * 100)
    loglike_diffs.append(norm_lapse_ll - vanilla_lapse_ll)

norm_lapse_minus_vanilla_lapse_data = {
    'lapse_rate_pct': np.array(lapse_rates),
    'loglike_diff': np.array(loglike_diffs),
    'x_label': 'Lapse rate (%)',
    'y_label': '(NPL + lapses) - (IPL + lapses)\nLL',
}

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['green' if diff > 0 else 'red' for diff in loglike_diffs]
ax.scatter(lapse_rates, loglike_diffs, c=colors, alpha=0.7, s=45, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

ax.set_xlabel('Lapse rate (%)', fontsize=14)
ax.set_ylabel('(NPL + lapses) - (IPL + lapses)\nLL', fontsize=14)
ax.set_title('Lapse rate vs LL diff', fontsize=14)
ax.set_xticks([0, 6])
ax.set_yticks([0, 300])
ax.tick_params(axis='both', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

output_prefix = os.path.join(BASE_DIR, 'lapse_rate_vs_norm_lapse_minus_vanilla_lapse_loglike')
png_path = f'{output_prefix}.png'
pdf_path = f'{output_prefix}.pdf'
plt.tight_layout()
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.show()

print(f'Saved: {png_path}')
print(f'Saved: {pdf_path}')
# %%
plot_data = {
    'norm_minus_vanilla_lapse': norm_minus_vanilla_lapse_data,
    'norm_lapse_minus_vanilla_lapse': norm_lapse_minus_vanilla_lapse_data,
}
pkl_path = os.path.join(BASE_DIR, 'lapse_rate_loglike_diff_data.pkl')
with open(pkl_path, 'wb') as handle:
    pickle.dump(plot_data, handle)

print(f'Saved data: {pkl_path}')
# %%
