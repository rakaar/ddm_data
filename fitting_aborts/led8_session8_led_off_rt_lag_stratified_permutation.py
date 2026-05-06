# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Parameters
csv_path = Path("/home/rlab/raghavendra/ddm_data/LED8_session8_training16_repeat_filtered.csv")
save_dir = Path("/home/rlab/raghavendra/ddm_data/led8_session8_led_off_rtds_led_lag")
save_dir.mkdir(parents=True, exist_ok=True)

summary_save_path = save_dir / "led8_session8_led_off_rt_lag_stratified_permutation_summary.csv"
null_save_path = save_dir / "led8_session8_led_off_rt_lag_stratified_permutation_null.csv"
fig_save_path = save_dir / "led8_session8_led_off_rt_lag_stratified_permutation.png"

rt_min = 0.0
rt_max = 1.0
n_permutations = 5000
random_seed = 20260506
strata_cols = ["animal", "session", "ABL", "abs_ILD"]


# %% Load data
df = pd.read_csv(csv_path)


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

df["lag_group"] = pd.NA
df.loc[df["dist_to_led_on"] == 1, "lag_group"] = "1 lag"
df.loc[df["dist_to_led_on"] == 2, "lag_group"] = "2 lag"
df.loc[df["dist_to_led_on"] >= 3, "lag_group"] = "3+ lag"
df = df[df["lag_group"].notna()].copy()
df = df.reset_index(drop=True)

lag_order = ["1 lag", "2 lag", "3+ lag"]
lag_to_code = {lag_name: idx for idx, lag_name in enumerate(lag_order)}
df["lag_code"] = df["lag_group"].map(lag_to_code).astype(int)


# %% Residualize RT within animal/session/ABL/abs_ILD strata
df["stratum_mean_rt"] = df.groupby(strata_cols)["RTwrtStim"].transform("mean")
df["rt_resid"] = df["RTwrtStim"] - df["stratum_mean_rt"]

resid = df["rt_resid"].to_numpy()
lag_codes = df["lag_code"].to_numpy()

group_indices = []
for _, group in df.groupby(strata_cols, sort=False):
    group_indices.append(group.index.to_numpy())


# %% Observed contrasts and stratified permutation null
def compute_stats(values, labels):
    means = np.array([values[labels == lag_code].mean() for lag_code in range(len(lag_order))])
    counts = np.array([(labels == lag_code).sum() for lag_code in range(len(lag_order))])
    weighted_center = np.average(means, weights=counts)
    omnibus = np.sum(counts * (means - weighted_center) ** 2)
    return {
        "mean_1_lag": means[0],
        "mean_2_lag": means[1],
        "mean_3plus_lag": means[2],
        "contrast_1_minus_3plus": means[0] - means[2],
        "contrast_2_minus_3plus": means[1] - means[2],
        "omnibus_between_lag": omnibus,
    }


rng = np.random.default_rng(random_seed)
observed_stats = compute_stats(resid, lag_codes)

null_rows = []
permuted_labels = lag_codes.copy()
for perm_idx in range(n_permutations):
    for idx in group_indices:
        permuted_labels[idx] = rng.permutation(lag_codes[idx])

    perm_stats = compute_stats(resid, permuted_labels)
    null_rows.append({"permutation": perm_idx, **perm_stats})

null_df = pd.DataFrame(null_rows)
null_df.to_csv(null_save_path, index=False)


# %% Summarize two-sided permutation p-values
summary_rows = []
for stat_name, observed_value in observed_stats.items():
    null_values = null_df[stat_name].to_numpy()
    if stat_name == "omnibus_between_lag":
        p_value = (np.sum(null_values >= observed_value) + 1) / (len(null_values) + 1)
    else:
        p_value = (np.sum(np.abs(null_values) >= abs(observed_value)) + 1) / (len(null_values) + 1)

    summary_rows.append(
        {
            "stat": stat_name,
            "observed_s": observed_value,
            "observed_ms": 1000 * observed_value,
            "null_mean_s": null_values.mean(),
            "null_std_s": null_values.std(ddof=1),
            "null_ci_low_s": np.percentile(null_values, 2.5),
            "null_ci_high_s": np.percentile(null_values, 97.5),
            "p_value": p_value,
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(summary_save_path, index=False)

print("Lag group counts:")
print(df["lag_group"].value_counts().reindex(lag_order).to_string())
print("\nObserved residual mean RT by lag group, ms:")
for lag_name in lag_order:
    print(f"{lag_name}: {1000 * df.loc[df['lag_group'] == lag_name, 'rt_resid'].mean():.3f}")
print("\nPermutation summary:")
print(summary_df[["stat", "observed_ms", "p_value"]].to_string(index=False))


# %% Plot null distributions for the main contrasts
plot_stats = [
    ("contrast_1_minus_3plus", "1 lag - 3+ lag"),
    ("contrast_2_minus_3plus", "2 lag - 3+ lag"),
    ("omnibus_between_lag", "omnibus"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

for ax, (stat_name, title) in zip(axes, plot_stats):
    null_values = null_df[stat_name].to_numpy()
    observed_value = observed_stats[stat_name]

    if stat_name == "omnibus_between_lag":
        null_plot = null_values * 1e6
        observed_plot = observed_value * 1e6
        xlabel = "Permutation statistic (s^2 x 1e6)"
    else:
        null_plot = 1000 * null_values
        observed_plot = 1000 * observed_value
        xlabel = "Residual mean RT contrast (ms)"

    p_value = summary_df.loc[summary_df["stat"] == stat_name, "p_value"].iloc[0]
    ax.hist(null_plot, bins=40, color="0.75", edgecolor="white")
    ax.axvline(observed_plot, color="tab:red", linewidth=2, label=f"observed, p={p_value:.3f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Permutations")
    ax.legend(fontsize=8)
    if stat_name == "omnibus_between_lag":
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(8)

fig.suptitle("Stratified permutation test: LED-off RT residuals by lag from LED ON", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved summary CSV to: {summary_save_path}")
print(f"Saved null CSV to: {null_save_path}")
print(f"Saved permutation plot to: {fig_save_path}")

# %%
