# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Parameters
csv_path = Path("/home/rlab/raghavendra/ddm_data/LED8_session8_training16_repeat_filtered.csv")
save_path = Path("/home/rlab/raghavendra/ddm_data/led8_session8_led_off_rtds.png")

rt_min = 0.0
rt_max = 1.0
n_bins = 40


# %% Load and filter data
df = pd.read_csv(csv_path)

df = df[df["success"].isin([1, -1])].copy()
df = df[df["RTwrtStim"].between(rt_min, rt_max)].copy()
df = df[df["LED_trial"].isna() | (df["LED_trial"] == 0)].copy()

abl_values = np.sort(df["ABL"].dropna().unique())
abs_ild_values = np.sort(df["abs_ILD"].dropna().unique())

if len(abl_values) != 3:
    raise ValueError(f"Expected 3 ABL values after filtering, found {len(abl_values)}: {abl_values}")

if len(abs_ild_values) != 5:
    raise ValueError(
        f"Expected 5 abs_ILD values after filtering, found {len(abs_ild_values)}: {abs_ild_values}"
    )

# bins = np.linspace(rt_min, rt_max, n_bins + 1)
bins = np.arange(0, 1, 0.02)

# %% Plot RTDs
fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharex=True, sharey=True)

for row_idx, abl in enumerate(abl_values):
    for col_idx, abs_ild in enumerate(abs_ild_values):
        ax = axes[row_idx, col_idx]
        rts = df.loc[(df["ABL"] == abl) & (df["abs_ILD"] == abs_ild), "RTwrtStim"].dropna()

        if len(rts) > 0:
            ax.hist(rts, bins=bins,  histtype='step')
        else:
            ax.text(0.5, 0.5, "No trials", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"ABL={abl}, abs ILD={abs_ild}")
        ax.set_xlim(rt_min, rt_max)

        if row_idx == len(abl_values) - 1:
            ax.set_xlabel("RTwrtStim")
        if col_idx == 0:
            ax.set_ylabel("Count")

fig.suptitle("Session type 8, LED off trials, RTDs")
fig.tight_layout(rect=(0, 0, 1, 0.96))

fig.savefig(save_path, dpi=300)
plt.show()


# %%
