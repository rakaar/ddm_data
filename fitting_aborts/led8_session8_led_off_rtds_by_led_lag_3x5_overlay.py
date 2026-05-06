# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Parameters
csv_path = Path("/home/rlab/raghavendra/ddm_data/LED8_session8_training16_repeat_filtered.csv")
save_dir = Path("/home/rlab/raghavendra/ddm_data/led8_session8_led_off_rtds_led_lag")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "led8_session8_led_off_rtds_by_led_lag_3x5_overlay.png"

rt_min = 0.0
rt_max = 1.0
bin_width = 0.03


# %% Load data
df = pd.read_csv(csv_path)

abl_values = np.sort(df["ABL"].dropna().unique())
abs_ild_values = np.sort(df["abs_ILD"].dropna().unique())
if len(abl_values) != 3:
    raise ValueError(f"Expected 3 ABL values, found {len(abl_values)}: {abl_values}")
if len(abs_ild_values) != 5:
    raise ValueError(f"Expected 5 abs_ILD values, found {len(abs_ild_values)}: {abs_ild_values}")

bins = np.append(np.arange(rt_min, rt_max, bin_width), rt_max)


# %% Compute distance back to most recent LED ON trial (on full data so LED ON trials exist)
def compute_led_on_distance(group):
    group = group.sort_values("trial").reset_index(drop=True)
    previous_led_on_trial = group["trial"].where(group["LED_trial"] == 1).ffill()
    group["dist_to_led_on"] = group["trial"] - previous_led_on_trial
    group.loc[group["LED_trial"] != 0, "dist_to_led_on"] = np.nan
    return group


df = pd.concat(
    [compute_led_on_distance(group) for _, group in df.groupby(["animal", "session"], sort=False)],
    ignore_index=True,
)

# Distances are float because of np.nan; convert valid ones to int
df["dist_to_led_on"] = df["dist_to_led_on"].astype("Int64")

# %% Now filter to valid-choice LED-off trials
df = df[df["success"].isin([1, -1])].copy()
df = df[df["RTwrtStim"].between(rt_min, rt_max)].copy()
df = df[df["LED_trial"].isna() | (df["LED_trial"] == 0)].copy()

# %% Build condition subsets
conditions = {
    "1 lag": df[df["dist_to_led_on"] == 1],
    "2 lag": df[df["dist_to_led_on"] == 2],
    "3+ lag": df[df["dist_to_led_on"] >= 3],
}

colors = {
    "1 lag": "tab:blue",
    "2 lag": "tab:orange",
    "3+ lag": "tab:green",
}

# %% Plot 3 x 5 grid with lag overlays
fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharex=True, sharey=True)

for row_idx, abl in enumerate(abl_values):
    for col_idx, abs_ild in enumerate(abs_ild_values):
        ax = axes[row_idx, col_idx]
        for cond_name, cond_df in conditions.items():
            rts = cond_df.loc[
                (cond_df["ABL"] == abl) & (cond_df["abs_ILD"] == abs_ild), "RTwrtStim"
            ].dropna()
            n_trials = len(rts)
            if n_trials > 0:
                ax.hist(
                    rts,
                    bins=bins,
                    histtype="step",
                    density=True,
                    color=colors[cond_name],
                    label=f"{cond_name} (n={n_trials})",
                )

        ax.set_title(f"ABL={abl}, abs ILD={abs_ild}")
        ax.set_xlim(rt_min, rt_max)
        if row_idx == 2:
            ax.set_xlabel("RTwrtStim")
        if col_idx == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=6, loc="upper right")

fig.suptitle("Session type 8 LED-off RTDs by lag from LED ON", fontsize=12, y=1.00)
fig.tight_layout()

fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {save_path}")

# %%
