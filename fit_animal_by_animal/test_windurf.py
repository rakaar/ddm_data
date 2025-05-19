# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory containing batch CSVs
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory containing batch CSVs
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f for f in os.listdir(batch_dir) if f.endswith('_valid_and_aborts.csv')]
n_batches = len(batch_files)

# Layout: up to 3 columns
ncols = min(3, n_batches)
nrows = (n_batches + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
color_map = plt.get_cmap('tab10')

for batch_idx, batch_file in enumerate(batch_files):
    batch_name = batch_file.replace('batch_', '').replace('_valid_and_aborts.csv', '')
    df = pd.read_csv(os.path.join(batch_dir, batch_file))
    valid_df = df[df['success'].isin([1, -1])].copy()
    valid_df = valid_df[valid_df['RTwrtStim'] <= 1]
    if 'abs_ILD' not in valid_df.columns:
        valid_df['abs_ILD'] = valid_df['ILD'].abs()
    valid_df = valid_df[~valid_df['abs_ILD'].isin([6, 10])]
    abls = sorted(valid_df['ABL'].unique())
    ax = axes[batch_idx // ncols, batch_idx % ncols]
    for i, abl in enumerate(abls):
        abl_df = valid_df[valid_df['ABL'] == abl]
        means = []
        errors = []
        abs_ilds = sorted(abl_df['abs_ILD'].unique())
        for abs_ild in abs_ilds:
            subset = abl_df[abl_df['abs_ILD'] == abs_ild]
            n = len(subset)
            mean = subset['RTwrtStim'].mean()
            std = subset['RTwrtStim'].std()
            err = std / np.sqrt(n-1) if n > 1 else 0
            means.append(mean)
            errors.append(err)
        ax.errorbar(abs_ilds, means, yerr=errors, marker='o', linestyle='-', color=color_map(i), label=f'ABL {abl}')
    ax.set_title(f'Batch: {batch_name}')
    ax.set_xlabel('|ILD| (dB)')
    ax.set_ylabel('Mean RTwrtStim (s)')
    ax.grid(True, alpha=0.3)
    ax.legend(title='ABL', fontsize=9)

# Hide unused axes if any
for idx in range(n_batches, nrows * ncols):
    fig.delaxes(axes[idx // ncols, idx % ncols])

plt.tight_layout()
plt.show()

