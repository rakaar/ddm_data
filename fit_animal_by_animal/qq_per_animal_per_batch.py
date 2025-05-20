# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# --- Load merged_valid as in mean_chrono_plot.py ---
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]
merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files
], ignore_index=True)

# Keep only valid trials (success 1 or -1)
merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()
merged_valid['batch_name'] = merged_valid['batch_name'].fillna('LED7')

# Remove RTwrtStim < 0.1s or > 1s
merged_valid = merged_valid[(merged_valid['RTwrtStim'] >= 0.1) & (merged_valid['RTwrtStim'] <= 1)]

# Add abs_ILD column
merged_valid['abs_ILD'] = merged_valid['ILD'].abs()

# Remove ILD 6 and 10
merged_valid = merged_valid[~merged_valid['abs_ILD'].isin([6, 10])]

# --- Prepare subplot grid ---
batch_names = merged_valid['batch_name'].unique()
plot_triplets = []  # (batch, animal, abs_ILD)
for batch_name in batch_names:
    batch_df = merged_valid[merged_valid['batch_name'] == batch_name]
    animals = batch_df['animal'].unique()
    for animal in animals:
        animal_df = batch_df[batch_df['animal'] == animal]
        abs_ILDs = np.sort(animal_df['abs_ILD'].unique())
        for abs_ILD in abs_ILDs:
            plot_triplets.append((batch_name, animal, abs_ILD))

num_cols = 5
num_rows = (len(plot_triplets) + num_cols - 1) // num_cols
subplot_size = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * subplot_size, num_rows * subplot_size))
axes = axes.flatten() if num_rows > 1 else [axes]

# --- Color scheme for ABLs ---
fixed_abl_colors = {
    20: '#1f77b4',   # blue
    40: '#ff7f0e',   # orange
    60: '#2ca02c',   # green
}
all_abls = sorted(set(merged_valid['ABL'].unique()))
tab10 = plt.get_cmap('tab10').colors
palette_indices = [i for i in range(len(tab10)) if i not in [0, 1, 2]]
palette = [tab10[i] for i in palette_indices]
other_abls = [abl for abl in all_abls if abl not in fixed_abl_colors]
other_abl_colors = {abl: palette[i % len(palette)] for i, abl in enumerate(other_abls)}
abl_color_map = {**fixed_abl_colors, **other_abl_colors}
def get_abl_color(abl):
    return abl_color_map.get(abl, '#888888')

# --- Q-Q plot parameters ---
percentiles = np.arange(5, 100, 10)
MIN_RT = 0.1
MAX_RT = 1

# --- Plot each subplot ---
for idx, (batch_name, animal, abs_ILD) in enumerate(plot_triplets):
    ax = axes[idx]
    df = merged_valid[(merged_valid['batch_name'] == batch_name) &
                      (merged_valid['animal'] == animal) &
                      (merged_valid['abs_ILD'] == abs_ILD)]
    if df.empty:
        ax.axis('off')
        continue
    RTwrtStim_pos = df[(df['RTwrtStim'] >= MIN_RT) & (df['RTwrtStim'] <= MAX_RT)]
    if RTwrtStim_pos.empty:
        ax.axis('off')
        continue
    ABLs = np.sort(RTwrtStim_pos['ABL'].unique())
    if len(ABLs) == 0:
        ax.axis('off')
        continue
    q_dict = {}
    for abl in ABLs:
        q_dict[abl] = np.percentile(RTwrtStim_pos[RTwrtStim_pos['ABL'] == abl]['RTwrtStim'], percentiles)
    abl_highest = ABLs.max()
    Q_highest = q_dict[abl_highest]
    for abl in ABLs:
        if abl == abl_highest:
            continue
        diff = q_dict[abl] - Q_highest
        n_rows = len(RTwrtStim_pos[RTwrtStim_pos['ABL'] == abl])
        ax.plot(Q_highest, diff, marker='o', label=f'ABL {abl} (N={n_rows})', color=get_abl_color(abl))
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel(f'Percentiles of RTwrtStim (ABL={abl_highest})')
    ax.set_ylabel('Q_ABL - Q_highest_ABL')
    total_rows = len(RTwrtStim_pos)
    ax.set_title(f'{batch_name} | {animal} | abs(ILD): {abs_ILD} (N={total_rows})', fontsize=10)
    ax.legend(title='ABL', fontsize=7, title_fontsize=8)
    # Set fixed x and y axis limits and ticks
    ax.set_xlim(0, 0.4)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.set_ylim(-0.05, 0.35)
    ax.set_yticks(np.arange(-0.05, 0.46, 0.1))

# Hide unused axes
for i in range(len(plot_triplets), len(axes)):
    axes[i].axis('off')

plt.tight_layout()

# Unified legend for all ABLs at bottom right
from matplotlib.lines import Line2D
handles = []
for abl in sorted(abl_color_map):
    handles.append(Line2D([0], [0], color=get_abl_color(abl), marker='o', label=f'ABL {abl}', linestyle='-', linewidth=2))
fig.legend(handles=handles, loc='lower right', fontsize=12, title='ABL', title_fontsize=10, frameon=True)

plt.show()

# Optionally, save to PDF
# pdf_path = "qq_per_animal_per_batch.pdf"
# with PdfPages(pdf_path) as pdf:
#     pdf.savefig(fig)
# print(f"PDF saved to {pdf_path}")
