# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Parameters
csv_path = Path("/home/rlab/raghavendra/ddm_data/LED8_session8_training16_repeat_filtered.csv")
save_dir = Path("/home/rlab/raghavendra/ddm_data/led8_session8_led_off_rtds_collapsed")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "led8_session8_led_off_rtds_1x3_abl_collapsed.png"

rt_min = 0.0
rt_max = 1.0
bin_width = 0.02


# %% Load and filter data
df = pd.read_csv(csv_path)

df = df[df["success"].isin([1, -1])].copy()
df = df[df["RTwrtStim"].between(rt_min, rt_max)].copy()
df = df[df["LED_trial"].isna() | (df["LED_trial"] == 0)].copy()

abl_values = np.sort(df["ABL"].dropna().unique())

if len(abl_values) != 3:
    raise ValueError(f"Expected 3 ABL values after filtering, found {len(abl_values)}: {abl_values}")

bins = np.append(np.arange(rt_min, rt_max, bin_width), rt_max)

# %% Plot RTDs collapsed across abs ILD
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

for col_idx, abl in enumerate(abl_values):
    ax = axes[col_idx]
    rts = df.loc[df["ABL"] == abl, "RTwrtStim"].dropna()

    if len(rts) > 0:
        ax.hist(rts, bins=bins, histtype='step')
    else:
        ax.text(0.5, 0.5, "No trials", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(f"ABL={abl}")
    ax.set_xlim(rt_min, rt_max)
    ax.set_xlabel("RTwrtStim")
    if col_idx == 0:
        ax.set_ylabel("Count")

fig.suptitle("Session type 8, LED off trials, RTDs (collapsed across abs ILD)")
fig.tight_layout(rect=(0, 0, 1, 0.96))

fig.savefig(save_path, dpi=300)
plt.show()

print(f"Saved plot to: {save_path}")

# %%
