# %%
"""
Evaluate SD t_E_aff extrapolation methods using non-SD held-out |ILD|=16.

For each non-SD animal, ABL, and signed ILD branch, fit/extrapolate from
|ILD| = 1, 2, 4, 8 and predict the held-out |ILD| = 16 delay. The same fitted
rules are then applied to SD animals to produce candidate extrapolated curves.
"""

# %%
# =============================================================================
# Parameters
# =============================================================================
from pathlib import Path
import os
import re
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs"
OUTPUT_DIR = OUTPUT_ROOT / "delay_extrapolation_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANIMAL_VALUES_CSV = OUTPUT_DIR / "delay_extrapolation_condition_values.csv"
HOLDOUT_PREDICTIONS_CSV = OUTPUT_DIR / "delay_extrapolation_holdout_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "delay_extrapolation_method_metrics.csv"
SD_CURVES_CSV = OUTPUT_DIR / "sd_delay_extrapolated_curves_by_method.csv"
NON_SD_CURVES_CSV = OUTPUT_DIR / "non_sd_delay_extrapolated_curves_by_method.csv"

MEAN_CURVES_PNG = OUTPUT_DIR / "delay_extrapolation_mean_curves_by_method.png"
PRED_VS_ACTUAL_PNG = OUTPUT_DIR / "delay_extrapolation_heldout16_pred_vs_actual.png"
METRIC_SUMMARY_PNG = OUTPUT_DIR / "delay_extrapolation_method_metric_summary.png"

EXPECTED_N_ANIMALS = 30
EXPECTED_N_CONDITION_ROWS = 864
EXPECTED_NON_SD_ANIMALS = 24
EXPECTED_SD_ANIMALS = 6

ABLS = [20, 40, 60]
TRAIN_ABS_ILDS = np.array([1.0, 2.0, 4.0, 8.0])
HOLDOUT_ABS_ILD = 16.0
CONTINUOUS_ABS_ILD_STEP = float(os.environ.get("DELAY_EXTRAP_CONTINUOUS_ABS_ILD_STEP", "0.1"))
CONTINUOUS_ABS_ILDS = np.round(
    np.arange(1.0, HOLDOUT_ABS_ILD + CONTINUOUS_ABS_ILD_STEP / 2, CONTINUOUS_ABS_ILD_STEP),
    6,
)

PLAUSIBLE_DELAY_MS_RANGE = (
    float(os.environ.get("DELAY_EXTRAP_PLAUSIBLE_DELAY_MS_MIN", "0")),
    float(os.environ.get("DELAY_EXTRAP_PLAUSIBLE_DELAY_MS_MAX", "200")),
)
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

METHODS = [
    {"method": "flat_hold_8", "label": "Flat hold at |ILD|=8", "space": "flat", "degree": None},
    {"method": "raw_poly1", "label": "Raw |ILD| linear", "space": "raw", "degree": 1},
    {"method": "raw_poly2", "label": "Raw |ILD| quadratic", "space": "raw", "degree": 2},
    {"method": "raw_poly3", "label": "Raw |ILD| cubic", "space": "raw", "degree": 3},
    {"method": "log2_poly1", "label": "log2 |ILD| linear", "space": "log2", "degree": 1},
    {"method": "log2_poly2", "label": "log2 |ILD| quadratic", "space": "log2", "degree": 2},
    {"method": "log2_poly3", "label": "log2 |ILD| cubic", "space": "log2", "degree": 3},
]
METHOD_LABELS = {method["method"]: method["label"] for method in METHODS}


# %%
# =============================================================================
# Helpers
# =============================================================================
def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def transform_abs_ild(abs_ild, space):
    abs_ild = np.asarray(abs_ild, dtype=float)
    if space == "raw":
        return abs_ild
    if space == "log2":
        return np.log2(abs_ild)
    raise ValueError(space)


def predict_delay_curve(train_abs_ild, train_delay_ms, query_abs_ild, method_spec):
    train_abs_ild = np.asarray(train_abs_ild, dtype=float)
    train_delay_ms = np.asarray(train_delay_ms, dtype=float)
    query_abs_ild = np.asarray(query_abs_ild, dtype=float)

    order = np.argsort(train_abs_ild)
    train_abs_ild = train_abs_ild[order]
    train_delay_ms = train_delay_ms[order]

    if method_spec["space"] == "flat":
        y_at_8 = float(train_delay_ms[np.where(np.isclose(train_abs_ild, 8.0))[0][0]])
        interp = np.interp(
            np.minimum(query_abs_ild, 8.0),
            train_abs_ild,
            train_delay_ms,
        )
        return np.where(query_abs_ild > 8.0, y_at_8, interp)

    x_train = transform_abs_ild(train_abs_ild, method_spec["space"])
    x_query = transform_abs_ild(query_abs_ild, method_spec["space"])
    degree = int(method_spec["degree"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs = np.polyfit(x_train, train_delay_ms, degree)
    return np.polyval(coeffs, x_query)


def is_implausible(values):
    low, high = PLAUSIBLE_DELAY_MS_RANGE
    values = np.asarray(values, dtype=float)
    return (~np.isfinite(values)) | (values < low) | (values > high)


def summarize_animal_abs_values(df, value_col):
    rows = []
    for (batch, animal, abl, abs_ild), group in df.groupby(["batch_name", "animal", "ABL", "abs_ild"], sort=True):
        values = group[value_col].to_numpy(dtype=float)
        rows.append(
            {
                "batch_name": batch,
                "animal": int(animal),
                "ABL": int(abl),
                "abs_ild": float(abs_ild),
                value_col: float(np.nanmean(values)),
            }
        )
    return pd.DataFrame(rows)


def summarize_curve_values(curve_df):
    rows = []
    for (method, batch, animal, abl, abs_ild), group in curve_df.groupby(
        ["method", "batch_name", "animal", "ABL", "abs_ild"],
        sort=True,
    ):
        values = group["pred_delay_ms"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "batch_name": batch,
                "animal": int(animal),
                "ABL": int(abl),
                "abs_ild": float(abs_ild),
                "pred_delay_ms": float(np.nanmean(values)),
            }
        )
    animal_abs = pd.DataFrame(rows)
    rows = []
    for (method, abl, abs_ild), group in animal_abs.groupby(["method", "ABL", "abs_ild"], sort=True):
        values = group["pred_delay_ms"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "method": method,
                "ABL": int(abl),
                "abs_ild": float(abs_ild),
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "sem": sem(finite),
                "n": int(finite.size),
            }
        )
    return pd.DataFrame(rows)


def summarize_observed_values(df):
    animal_abs = summarize_animal_abs_values(df, "t_E_aff_ms_mean")
    rows = []
    for (abl, abs_ild), group in animal_abs.groupby(["ABL", "abs_ild"], sort=True):
        values = group["t_E_aff_ms_mean"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "ABL": int(abl),
                "abs_ild": float(abs_ild),
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "sem": sem(finite),
                "n": int(finite.size),
            }
        )
    return pd.DataFrame(rows)


def metric_rows_for_predictions(prediction_df):
    rows = []
    for method in [method_spec["method"] for method_spec in METHODS]:
        method_df = prediction_df[prediction_df["method"] == method]
        for abl in ["all"] + ABLS:
            subset = method_df if abl == "all" else method_df[method_df["ABL"] == abl]
            actual = subset["actual_delay_ms"].to_numpy(dtype=float)
            pred = subset["pred_delay_ms"].to_numpy(dtype=float)
            finite = np.isfinite(actual) & np.isfinite(pred)
            diff = pred[finite] - actual[finite]
            if finite.sum() >= 2:
                corr = float(np.corrcoef(actual[finite], pred[finite])[0, 1])
            else:
                corr = np.nan
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "ABL": abl,
                    "n_cases": int(finite.sum()),
                    "rmse_ms": float(np.sqrt(np.mean(diff**2))) if diff.size else np.nan,
                    "mae_ms": float(np.mean(np.abs(diff))) if diff.size else np.nan,
                    "bias_ms": float(np.mean(diff)) if diff.size else np.nan,
                    "pearson_r": corr,
                    "max_abs_error_ms": float(np.max(np.abs(diff))) if diff.size else np.nan,
                    "n_implausible_pred": int(np.sum(is_implausible(pred))),
                }
            )
    return pd.DataFrame(rows)


# %%
# =============================================================================
# Load condition summaries
# =============================================================================
summary_paths = sorted(OUTPUT_ROOT.glob("*/*_big_gamma_omega_delay_condition_summary.csv"))
if len(summary_paths) != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} condition-summary CSVs, found {len(summary_paths)}")

animal_dfs = []
for summary_path in summary_paths:
    match = re.match(r"^(?P<batch>.+)_(?P<animal>\d+)$", summary_path.parent.name)
    if match is None:
        raise RuntimeError(f"Could not parse animal folder name: {summary_path.parent}")

    df = pd.read_csv(summary_path)
    required_cols = ["batch_name", "animal", "ABL", "ILD", "t_E_aff_ms_mean"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"{summary_path} missing columns: {missing_cols}")

    df = df[required_cols].copy()
    df["animal"] = df["animal"].astype(int)
    df["ABL"] = df["ABL"].astype(int)
    df["ILD"] = df["ILD"].astype(float)
    df["abs_ild"] = df["ILD"].abs()
    df["sign"] = np.sign(df["ILD"]).astype(int)
    animal_dfs.append(df)

condition_df = pd.concat(animal_dfs, ignore_index=True)
condition_df = condition_df.sort_values(["batch_name", "animal", "ABL", "ILD"]).reset_index(drop=True)

if len(condition_df) != EXPECTED_N_CONDITION_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_N_CONDITION_ROWS} condition rows, found {len(condition_df)}")
n_animals = condition_df[["batch_name", "animal"]].drop_duplicates().shape[0]
if n_animals != EXPECTED_N_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_N_ANIMALS} animals, found {n_animals}")
if not np.all(np.isfinite(condition_df["t_E_aff_ms_mean"].to_numpy(dtype=float))):
    raise RuntimeError("t_E_aff_ms_mean contains NaN/Inf.")

non_sd_df = condition_df[condition_df["batch_name"] != "SD"].copy()
sd_df = condition_df[condition_df["batch_name"] == "SD"].copy()
n_non_sd_animals = non_sd_df[["batch_name", "animal"]].drop_duplicates().shape[0]
n_sd_animals = sd_df[["batch_name", "animal"]].drop_duplicates().shape[0]
if n_non_sd_animals != EXPECTED_NON_SD_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_NON_SD_ANIMALS} non-SD animals, found {n_non_sd_animals}")
if n_sd_animals != EXPECTED_SD_ANIMALS:
    raise RuntimeError(f"Expected {EXPECTED_SD_ANIMALS} SD animals, found {n_sd_animals}")
if len(sd_df[sd_df["abs_ild"] == HOLDOUT_ABS_ILD]):
    raise RuntimeError("SD unexpectedly has |ILD|=16 condition rows.")

condition_df.to_csv(ANIMAL_VALUES_CSV, index=False)
print(f"Loaded animals: {n_animals} total, {n_non_sd_animals} non-SD, {n_sd_animals} SD")
print(f"Loaded condition rows: {len(condition_df)}")
print(f"Saved condition values: {ANIMAL_VALUES_CSV}")


# %%
# =============================================================================
# Hold-out validation on non-SD animals and SD extrapolated curves
# =============================================================================
prediction_rows = []
non_sd_curve_rows = []
sd_curve_rows = []

group_cols = ["batch_name", "animal", "ABL", "sign"]
expected_group_abs = set(TRAIN_ABS_ILDS.tolist() + [HOLDOUT_ABS_ILD])

for group_key, group in non_sd_df.groupby(group_cols, sort=True):
    batch, animal, abl, sign = group_key
    abs_set = set(group["abs_ild"].astype(float))
    if abs_set != expected_group_abs:
        raise RuntimeError(f"{group_key} has abs ILDs {sorted(abs_set)}, expected {sorted(expected_group_abs)}")

    train = group[group["abs_ild"].isin(TRAIN_ABS_ILDS)].sort_values("abs_ild")
    holdout = group[group["abs_ild"] == HOLDOUT_ABS_ILD]
    if len(train) != len(TRAIN_ABS_ILDS) or len(holdout) != 1:
        raise RuntimeError(f"{group_key} has invalid train/holdout row counts.")

    train_abs = train["abs_ild"].to_numpy(dtype=float)
    train_delay = train["t_E_aff_ms_mean"].to_numpy(dtype=float)
    actual_delay = float(holdout["t_E_aff_ms_mean"].iloc[0])

    for method_spec in METHODS:
        method = method_spec["method"]
        pred_holdout = float(predict_delay_curve(train_abs, train_delay, np.array([HOLDOUT_ABS_ILD]), method_spec)[0])
        pred_curve = predict_delay_curve(train_abs, train_delay, CONTINUOUS_ABS_ILDS, method_spec)

        prediction_rows.append(
            {
                "batch_name": batch,
                "animal": int(animal),
                "ABL": int(abl),
                "sign": int(sign),
                "method": method,
                "method_label": method_spec["label"],
                "actual_abs_ild": HOLDOUT_ABS_ILD,
                "actual_delay_ms": actual_delay,
                "pred_delay_ms": pred_holdout,
                "error_ms": pred_holdout - actual_delay,
                "abs_error_ms": abs(pred_holdout - actual_delay),
                "squared_error_ms2": (pred_holdout - actual_delay) ** 2,
                "implausible_pred": bool(is_implausible([pred_holdout])[0]),
                "curve_has_implausible": bool(np.any(is_implausible(pred_curve))),
            }
        )
        for abs_ild, pred_delay in zip(CONTINUOUS_ABS_ILDS, pred_curve):
            non_sd_curve_rows.append(
                {
                    "batch_name": batch,
                    "animal": int(animal),
                    "ABL": int(abl),
                    "sign": int(sign),
                    "method": method,
                    "method_label": method_spec["label"],
                    "abs_ild": float(abs_ild),
                    "pred_delay_ms": float(pred_delay),
                    "is_extrapolated": bool(abs_ild > TRAIN_ABS_ILDS.max()),
                    "implausible_pred": bool(is_implausible([pred_delay])[0]),
                }
            )

for group_key, group in sd_df.groupby(group_cols, sort=True):
    batch, animal, abl, sign = group_key
    abs_set = set(group["abs_ild"].astype(float))
    if abs_set != set(TRAIN_ABS_ILDS.tolist()):
        raise RuntimeError(f"{group_key} has abs ILDs {sorted(abs_set)}, expected {TRAIN_ABS_ILDS.tolist()}")

    train = group.sort_values("abs_ild")
    train_abs = train["abs_ild"].to_numpy(dtype=float)
    train_delay = train["t_E_aff_ms_mean"].to_numpy(dtype=float)

    for method_spec in METHODS:
        method = method_spec["method"]
        pred_curve = predict_delay_curve(train_abs, train_delay, CONTINUOUS_ABS_ILDS, method_spec)
        for abs_ild, pred_delay in zip(CONTINUOUS_ABS_ILDS, pred_curve):
            sd_curve_rows.append(
                {
                    "batch_name": batch,
                    "animal": int(animal),
                    "ABL": int(abl),
                    "sign": int(sign),
                    "method": method,
                    "method_label": method_spec["label"],
                    "abs_ild": float(abs_ild),
                    "pred_delay_ms": float(pred_delay),
                    "is_extrapolated": bool(abs_ild > TRAIN_ABS_ILDS.max()),
                    "implausible_pred": bool(is_implausible([pred_delay])[0]),
                }
            )

prediction_df = pd.DataFrame(prediction_rows)
non_sd_curve_df = pd.DataFrame(non_sd_curve_rows)
sd_curve_df = pd.DataFrame(sd_curve_rows)
metrics_df = metric_rows_for_predictions(prediction_df)

expected_holdout_rows = EXPECTED_NON_SD_ANIMALS * len(ABLS) * 2 * len(METHODS)
if len(prediction_df) != expected_holdout_rows:
    raise RuntimeError(f"Expected {expected_holdout_rows} holdout prediction rows, found {len(prediction_df)}")

prediction_df.to_csv(HOLDOUT_PREDICTIONS_CSV, index=False)
non_sd_curve_df.to_csv(NON_SD_CURVES_CSV, index=False)
sd_curve_df.to_csv(SD_CURVES_CSV, index=False)
metrics_df.to_csv(METRICS_CSV, index=False)

print(f"Saved holdout predictions: {HOLDOUT_PREDICTIONS_CSV}")
print(f"Saved non-SD curves: {NON_SD_CURVES_CSV}")
print(f"Saved SD extrapolated curves: {SD_CURVES_CSV}")
print(f"Saved metrics: {METRICS_CSV}")
print("\nOverall method metrics:")
overall_metrics = metrics_df[metrics_df["ABL"].astype(str) == "all"].copy()
print(
    overall_metrics[
        ["method", "n_cases", "rmse_ms", "mae_ms", "bias_ms", "pearson_r", "n_implausible_pred"]
    ].to_string(index=False)
)


# %%
# =============================================================================
# Plot 1: mean delay curves by method
# =============================================================================
observed_summary_df = summarize_observed_values(non_sd_df)
curve_summary_df = summarize_curve_values(non_sd_curve_df)

fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.5), sharex=True, sharey=True)
axes = axes.ravel()

obs_for_ylim = observed_summary_df["mean"].to_numpy(dtype=float)
curve_for_ylim = curve_summary_df["mean"].to_numpy(dtype=float)
plausible_mask = (curve_for_ylim >= PLAUSIBLE_DELAY_MS_RANGE[0]) & (curve_for_ylim <= PLAUSIBLE_DELAY_MS_RANGE[1])
ylim_values = np.concatenate([obs_for_ylim[np.isfinite(obs_for_ylim)], curve_for_ylim[np.isfinite(curve_for_ylim) & plausible_mask]])
y_min = float(np.nanmin(ylim_values) - 12)
y_max = float(np.nanmax(ylim_values) + 12)

for ax, method_spec in zip(axes[: len(METHODS)], METHODS):
    method = method_spec["method"]
    all_metric = overall_metrics[overall_metrics["method"] == method].iloc[0]
    for abl in ABLS:
        color = COLORS[abl]
        obs = observed_summary_df[observed_summary_df["ABL"] == abl].sort_values("abs_ild")
        train_obs = obs[obs["abs_ild"] <= TRAIN_ABS_ILDS.max()]
        held_obs = obs[obs["abs_ild"] == HOLDOUT_ABS_ILD]
        ax.errorbar(
            train_obs["abs_ild"],
            train_obs["mean"],
            yerr=train_obs["sem"],
            fmt="o",
            color=color,
            mfc="white",
            mec=color,
            capsize=2,
            ms=4.5,
            linestyle="none",
        )
        ax.errorbar(
            held_obs["abs_ild"],
            held_obs["mean"],
            yerr=held_obs["sem"],
            fmt="D",
            color=color,
            mfc=color,
            mec=color,
            capsize=2,
            ms=4.5,
            linestyle="none",
        )
        curve = curve_summary_df[
            (curve_summary_df["method"] == method)
            & (curve_summary_df["ABL"] == abl)
        ].sort_values("abs_ild")
        ax.plot(curve["abs_ild"], curve["mean"], color=color, lw=1.8)
        ax.fill_between(
            curve["abs_ild"].to_numpy(dtype=float),
            (curve["mean"] - curve["sem"]).to_numpy(dtype=float),
            (curve["mean"] + curve["sem"]).to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    ax.axvline(8, color="0.55", lw=0.9, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.22)
    ax.set_title(
        f"{method_spec['label']}\nRMSE={all_metric['rmse_ms']:.2f} ms, r={all_metric['pearson_r']:.2f}",
        fontsize=10,
    )
    ax.set_ylim(y_min, y_max)

for ax in axes[: len(METHODS)]:
    ax.set_xlabel("|ILD|")
for ax in axes[::4]:
    ax.set_ylabel("t_E_aff (ms)")

legend_ax = axes[-1]
legend_ax.axis("off")
handles = [Line2D([0], [0], color=COLORS[abl], lw=2, label=f"ABL {abl}") for abl in ABLS]
handles.extend(
    [
        Line2D([0], [0], marker="o", color="black", mfc="white", lw=0, label="train data +/- SEM"),
        Line2D([0], [0], marker="D", color="black", lw=0, label="held-out |ILD|=16 data"),
        Line2D([0], [0], color="black", lw=2, label="fit/extrapolated mean"),
        Line2D([0], [0], color="0.55", lw=1, ls="--", label="SD observed limit |ILD|=8"),
    ]
)
legend_ax.legend(handles=handles, loc="upper left", frameon=False)
metric_text = overall_metrics[["method", "rmse_ms", "pearson_r"]].copy()
metric_text["short_method"] = [
    "flat",
    "raw1",
    "raw2",
    "raw3",
    "log1",
    "log2",
    "log3",
]
metric_lines = ["method   RMSE      r"]
for row in metric_text.itertuples(index=False):
    metric_lines.append(f"{row.short_method:<6} {row.rmse_ms:6.2f} {row.pearson_r:6.2f}")
legend_ax.text(
    0.0,
    0.05,
    "\n".join(metric_lines),
    transform=legend_ax.transAxes,
    family="monospace",
    fontsize=8,
    va="bottom",
)

fig.suptitle(
    "Non-SD delay extrapolation validation: fit |ILD|=1,2,4,8; hold out |ILD|=16",
    y=1.02,
)
fig.tight_layout()
fig.savefig(MEAN_CURVES_PNG, dpi=300, bbox_inches="tight")
print(f"Saved mean-curve figure: {MEAN_CURVES_PNG}")


# %%
# =============================================================================
# Plot 2: held-out prediction vs actual
# =============================================================================
fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.8), sharex=True, sharey=True)
axes = axes.ravel()

all_actual = prediction_df["actual_delay_ms"].to_numpy(dtype=float)
all_pred = prediction_df["pred_delay_ms"].to_numpy(dtype=float)
finite_plot = np.isfinite(all_actual) & np.isfinite(all_pred) & (all_pred > -100) & (all_pred < 250)
axis_min = float(np.nanmin(np.concatenate([all_actual[finite_plot], all_pred[finite_plot]])) - 8)
axis_max = float(np.nanmax(np.concatenate([all_actual[finite_plot], all_pred[finite_plot]])) + 8)

for ax, method_spec in zip(axes[: len(METHODS)], METHODS):
    method = method_spec["method"]
    subset = prediction_df[prediction_df["method"] == method]
    metric = overall_metrics[overall_metrics["method"] == method].iloc[0]
    for abl in ABLS:
        curr = subset[subset["ABL"] == abl]
        ax.scatter(
            curr["actual_delay_ms"],
            curr["pred_delay_ms"],
            s=22,
            alpha=0.72,
            color=COLORS[abl],
            label=f"ABL {abl}",
        )
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="0.45", lw=1.0, ls="--")
    ax.set_title(
        f"{method_spec['label']}\nRMSE={metric['rmse_ms']:.2f} ms, r={metric['pearson_r']:.2f}",
        fontsize=10,
    )
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.grid(True, alpha=0.22)
    ax.set_aspect("equal", adjustable="box")

for ax in axes[4:7]:
    ax.set_xlabel("actual t_E_aff(16) ms")
for ax in [axes[0], axes[4]]:
    ax.set_ylabel("predicted t_E_aff(16) ms")

axes[-1].axis("off")
axes[-1].legend(handles=[Line2D([0], [0], marker="o", color=COLORS[abl], lw=0, label=f"ABL {abl}") for abl in ABLS], loc="upper left", frameon=False)
axes[-1].text(
    0.0,
    0.05,
    "Each point is one non-SD\nanimal x ABL x sign branch.\n\nTraining: |ILD|=1,2,4,8\nHeld out: |ILD|=16",
    fontsize=10,
    va="bottom",
)

fig.suptitle("Held-out |ILD|=16 delay prediction by extrapolation method", y=0.985)
fig.subplots_adjust(left=0.065, right=0.985, bottom=0.08, top=0.89, hspace=0.36, wspace=0.22)
fig.savefig(PRED_VS_ACTUAL_PNG, dpi=300, bbox_inches="tight")
print(f"Saved prediction-vs-actual figure: {PRED_VS_ACTUAL_PNG}")


# %%
# =============================================================================
# Plot 3: metric summary
# =============================================================================
method_order = [method["method"] for method in METHODS]
plot_metrics = overall_metrics.set_index("method").loc[method_order].reset_index()
x = np.arange(len(plot_metrics))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
metric_specs = [
    ("rmse_ms", "RMSE (ms)", False),
    ("pearson_r", "Pearson r", False),
]
short_labels = [
    "flat",
    "raw1",
    "raw2",
    "raw3",
    "log1",
    "log2",
    "log3",
]

for ax, (metric_col, ylabel, add_zero) in zip(axes, metric_specs):
    values = plot_metrics[metric_col].to_numpy(dtype=float)
    ax.bar(x, values, color="0.78", edgecolor="0.25", linewidth=0.8)
    for abl in ABLS:
        abl_metrics = metrics_df[(metrics_df["ABL"].astype(str) == str(abl))].set_index("method").loc[method_order]
        ax.plot(
            x,
            abl_metrics[metric_col].to_numpy(dtype=float),
            marker="o",
            lw=1.1,
            ms=4,
            color=COLORS[abl],
            label=f"ABL {abl}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.22)
    ax.set_title(ylabel)

axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Delay extrapolation held-out |ILD|=16 metrics", y=1.03)
fig.tight_layout()
fig.savefig(METRIC_SUMMARY_PNG, dpi=300, bbox_inches="tight")
print(f"Saved metric-summary figure: {METRIC_SUMMARY_PNG}")

plt.show()

# %%
