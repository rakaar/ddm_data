# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Parameters
csv_path = Path("/home/rlab/raghavendra/ddm_data/LED8_session8_training16_repeat_filtered.csv")
save_dir = Path("/home/rlab/raghavendra/ddm_data/led8_session8_led_off_rtds_led_lag")
save_dir.mkdir(parents=True, exist_ok=True)

fig_save_path = save_dir / "led8_session8_led_off_rt_quantiles_by_led_lag_3x5.png"
csv_save_path = save_dir / "led8_session8_led_off_rt_quantiles_by_led_lag_3x5.csv"

rt_min = 0.0
rt_max = 1.0
quantile_levels = np.array([0.1, 0.5, 0.9])
n_bootstrap = 2000
bootstrap_ci = (2.5, 97.5)
min_trials = 20
random_seed = 20260506


# %% Load data
df = pd.read_csv(csv_path)

abl_values = np.sort(df["ABL"].dropna().unique())
abs_ild_values = np.sort(df["abs_ILD"].dropna().unique())
if len(abl_values) != 3:
    raise ValueError(f"Expected 3 ABL values, found {len(abl_values)}: {abl_values}")
if len(abs_ild_values) != 5:
    raise ValueError(f"Expected 5 abs_ILD values, found {len(abs_ild_values)}: {abs_ild_values}")


# %% Compute distance back to most recent LED ON trial using actual trial numbers
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
df["dist_to_led_on"] = df["dist_to_led_on"].astype("Int64")


# %% Filter to valid-choice LED-off trials and assign lag groups
df = df[df["success"].isin([1, -1])].copy()
df = df[df["RTwrtStim"].between(rt_min, rt_max)].copy()
df = df[df["LED_trial"].isna() | (df["LED_trial"] == 0)].copy()

lag_groups = {
    "1 lag": df["dist_to_led_on"] == 1,
    "2 lag": df["dist_to_led_on"] == 2,
    "3+ lag": df["dist_to_led_on"] >= 3,
}

df["lag_group"] = pd.NA
for lag_name, lag_mask in lag_groups.items():
    df.loc[lag_mask, "lag_group"] = lag_name
df = df[df["lag_group"].notna()].copy()

lag_order = ["1 lag", "2 lag", "3+ lag"]
quantile_labels = {0.1: "q10", 0.5: "median", 0.9: "q90"}
quantile_colors = {0.1: "tab:blue", 0.5: "black", 0.9: "tab:red"}
quantile_offsets = {0.1: -0.12, 0.5: 0.0, 0.9: 0.12}


# %% Bootstrap quantiles for each ABL x abs ILD x lag group
rng = np.random.default_rng(random_seed)
summary_rows = []

for abl in abl_values:
    for abs_ild in abs_ild_values:
        for lag_name in lag_order:
            rts = df.loc[
                (df["ABL"] == abl) & (df["abs_ILD"] == abs_ild) & (df["lag_group"] == lag_name),
                "RTwrtStim",
            ].dropna().to_numpy()

            n_trials = len(rts)
            if n_trials < min_trials:
                for q in quantile_levels:
                    summary_rows.append(
                        {
                            "ABL": abl,
                            "abs_ILD": abs_ild,
                            "lag_group": lag_name,
                            "quantile": q,
                            "quantile_label": quantile_labels[q],
                            "n_trials": n_trials,
                            "rt_quantile": np.nan,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                        }
                    )
                continue

            rt_quantiles = np.quantile(rts, quantile_levels)
            bootstrap_samples = rng.choice(rts, size=(n_bootstrap, n_trials), replace=True)
            bootstrap_quantiles = np.quantile(bootstrap_samples, quantile_levels, axis=1).T
            ci_low, ci_high = np.percentile(bootstrap_quantiles, bootstrap_ci, axis=0)

            for q_idx, q in enumerate(quantile_levels):
                summary_rows.append(
                    {
                        "ABL": abl,
                        "abs_ILD": abs_ild,
                        "lag_group": lag_name,
                        "quantile": q,
                        "quantile_label": quantile_labels[q],
                        "n_trials": n_trials,
                        "rt_quantile": rt_quantiles[q_idx],
                        "ci_low": ci_low[q_idx],
                        "ci_high": ci_high[q_idx],
                    }
                )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(csv_save_path, index=False)


# %% Plot 3 x 5 quantile summaries
fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharex=True, sharey=True)
x_base = np.arange(len(lag_order))

for row_idx, abl in enumerate(abl_values):
    for col_idx, abs_ild in enumerate(abs_ild_values):
        ax = axes[row_idx, col_idx]
        panel_df = summary_df[(summary_df["ABL"] == abl) & (summary_df["abs_ILD"] == abs_ild)]

        for q in quantile_levels:
            q_df = (
                panel_df[panel_df["quantile"] == q]
                .set_index("lag_group")
                .reindex(lag_order)
                .reset_index()
            )
            y = q_df["rt_quantile"].to_numpy()
            yerr = np.vstack(
                [
                    y - q_df["ci_low"].to_numpy(),
                    q_df["ci_high"].to_numpy() - y,
                ]
            )
            x = x_base + quantile_offsets[q]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                color=quantile_colors[q],
                markersize=4,
                capsize=2,
                linewidth=1,
                label=quantile_labels[q],
            )

        median_df = (
            panel_df[panel_df["quantile"] == 0.5]
            .set_index("lag_group")
            .reindex(lag_order)
            .reset_index()
        )
        n_text = ", ".join(
            f"{lag_name}: n={int(n_trials)}"
            for lag_name, n_trials in zip(lag_order, median_df["n_trials"])
        )
        ax.text(0.02, 0.96, n_text, transform=ax.transAxes, va="top", ha="left", fontsize=6)

        ax.set_title(f"ABL={abl}, abs ILD={abs_ild}", fontsize=9)
        ax.set_xticks(x_base)
        ax.set_xticklabels(lag_order)
        ax.set_xlim(-0.45, len(lag_order) - 0.55)
        ax.set_ylim(rt_min, rt_max)
        ax.grid(alpha=0.25, linewidth=0.5)
        if row_idx == 2:
            ax.set_xlabel("Lag from previous LED ON")
        if col_idx == 0:
            ax.set_ylabel("RTwrtStim quantile (s)")
        if row_idx == 0 and col_idx == 4:
            ax.legend(fontsize=7, loc="lower right")

fig.suptitle("Session type 8 LED-off RT quantiles by lag from LED ON", fontsize=12, y=1.00)
fig.tight_layout()

fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved quantile plot to: {fig_save_path}")
print(f"Saved quantile summary CSV to: {csv_save_path}")

# %%
