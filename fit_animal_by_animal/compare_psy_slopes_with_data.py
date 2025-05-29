# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# -rw-rw-r-- 1 ragha ragha      715 May 29 17:30 psychometric_logistic_slopes_NORM_TIED.pkl
# -rw-rw-r-- 1 ragha ragha      715 May 29 17:35 psychometric_logistic_slopes_VANILLA_TIED.pkl
#
import pickle
import os

# --- Plot comparison of slopes from VANILLA_TIED, NORM_TIED, and DATA ---

vanilla_path = 'psychometric_logistic_slopes_VANILLA_TIED.pkl'
norm_path = 'psychometric_logistic_slopes_NORM_TIED.pkl'

# Check if files exist
if not (os.path.exists(vanilla_path) and os.path.exists(norm_path)):
    raise FileNotFoundError(f"Required pickle files not found in current directory.\nExpected: {vanilla_path}, {norm_path}")

with open(vanilla_path, 'rb') as f:
    vanilla = pickle.load(f)
with open(norm_path, 'rb') as f:
    norm = pickle.load(f)

# Prepare data
ABLs = [20, 40, 60]
x_labels = ['VANILLA_TIED', 'NORM_TIED', 'DATA']
x_pos = np.arange(len(x_labels))
colors = {20: 'tab:blue', 40: 'tab:orange', 60: 'tab:green'}

plt.figure(figsize=(5, 4))
for abl in ABLs:
    y = []
    # Vanilla
    v_slope = vanilla['theory'][abl]['slope'] if vanilla['theory'][abl]['slope'] is not None else np.nan
    # Norm
    n_slope = norm['theory'][abl]['slope'] if norm['theory'][abl]['slope'] is not None else np.nan
    # Data: use from either file (should be the same)
    d_slope = vanilla['data'][abl]['slope'] if vanilla['data'][abl]['slope'] is not None else np.nan
    y = [v_slope, n_slope, d_slope]
    plt.scatter(x_pos, y, color=colors[abl], label=f'ABL={abl}' if abl==ABLs[0] else None, s=80, edgecolor='k', zorder=3)

plt.xticks(x_pos, x_labels)
plt.ylabel('Logistic Slope')
plt.title('Psychometric Slopes: Model vs Data')
# Custom legend to avoid duplicate labels
handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'ABL={abl}',
                      markerfacecolor=colors[abl], markeredgecolor='k', markersize=10) for abl in ABLs]
plt.legend(handles=handles, title='ABL', frameon=True)
# Remove grid lines
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Set y-ticks
plt.yticks([0.3, 0.4, 0.5])
plt.tight_layout()
plt.show()

