# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Parameters (edit here) ############
SHOW_PLOT = True

input_delay_csv_path = (
    Path(__file__).resolve().parent
    / "estimate_delays_from_RTDs_rise_by_abl_abs_ild_kde_slope"
    / "baseline_significant_slope_onset_mean_plus_5sigma_ms_by_ABL_abs_ILD.csv"
)

panel_width = 4.3
panel_height = 3.5
slice_panel_width = 4.8
slice_panel_height = 4.2
png_dpi = 300
show_plot = SHOW_PLOT

title_fontsize = 15
axis_label_fontsize = 13
tick_label_fontsize = 11
legend_fontsize = 10
annotation_fontsize = 11
suptitle_fontsize = 16
equation_fontsize = 10

SCRIPT_DIR = Path(__file__).resolve().parent
output_dir = SCRIPT_DIR / "fit_delay_surface_from_5sigma_csv_ild2"


# %%
############ Helpers ############
def build_linear_design_matrix(abl_values: np.ndarray, abs_ild_values: np.ndarray):
    return np.column_stack(
        [
            np.ones_like(abl_values, dtype=float),
            abl_values,
            abs_ild_values,
        ]
    )


def build_linear_plus_ild2_design_matrix(abl_values: np.ndarray, abs_ild_values: np.ndarray):
    # The input delay table is keyed by |ILD|, and ILD^2 equals |ILD|^2.
    return np.column_stack(
        [
            np.ones_like(abl_values, dtype=float),
            abl_values,
            abs_ild_values,
            abs_ild_values**2,
        ]
    )


def solve_ols_and_metrics(design_matrix: np.ndarray, target_values: np.ndarray):
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, target_values, rcond=None)
    predicted_values = design_matrix @ coefficients
    residual_values = target_values - predicted_values
    rmse_ms = float(np.sqrt(np.mean(residual_values**2)))
    target_mean = float(np.mean(target_values))
    total_sum_squares = float(np.sum((target_values - target_mean) ** 2))
    residual_sum_squares = float(np.sum(residual_values**2))
    r2 = np.nan if np.isclose(total_sum_squares, 0.0) else 1.0 - (residual_sum_squares / total_sum_squares)
    return coefficients, predicted_values, residual_values, rmse_ms, float(r2)


def annotate_matrix(ax, matrix_df: pd.DataFrame):
    matrix_values = matrix_df.to_numpy(dtype=float)
    value_midpoint = 0.5 * (np.nanmin(matrix_values) + np.nanmax(matrix_values))

    for row_idx in range(matrix_df.shape[0]):
        for col_idx in range(matrix_df.shape[1]):
            value = matrix_values[row_idx, col_idx]
            text_color = "white" if value > value_midpoint else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                color=text_color,
            )


def format_linear_equation(coefficients: np.ndarray):
    return (
        f"delay_ms = {coefficients[0]:.4f} "
        f"+ ({coefficients[1]:.4f}) * ABL "
        f"+ ({coefficients[2]:.4f}) * abs_ILD"
    )


def format_linear_plus_ild2_equation(coefficients: np.ndarray):
    return (
        f"delay_ms = {coefficients[0]:.4f} "
        f"+ ({coefficients[1]:.4f}) * ABL "
        f"+ ({coefficients[2]:.4f}) * abs_ILD "
        f"+ ({coefficients[3]:.4f}) * ILD^2"
    )


def predict_linear_delay_ms(abl_values: np.ndarray, abs_ild_values: np.ndarray, coefficients: np.ndarray):
    return build_linear_design_matrix(
        abl_values=np.asarray(abl_values, dtype=float),
        abs_ild_values=np.asarray(abs_ild_values, dtype=float),
    ) @ coefficients


def predict_linear_plus_ild2_delay_ms(abl_values: np.ndarray, abs_ild_values: np.ndarray, coefficients: np.ndarray):
    return build_linear_plus_ild2_design_matrix(
        abl_values=np.asarray(abl_values, dtype=float),
        abs_ild_values=np.asarray(abs_ild_values, dtype=float),
    ) @ coefficients


def format_equation_for_title(equation_text: str):
    return equation_text.replace("delay_ms = ", "")


# %%
############ Load delay matrix CSV and reshape to long table ############
if not input_delay_csv_path.exists():
    raise FileNotFoundError(f"Could not find input delay CSV: {input_delay_csv_path}")

output_dir.mkdir(parents=True, exist_ok=True)

delay_matrix_df = pd.read_csv(input_delay_csv_path, index_col=0)
if delay_matrix_df.isna().any().any():
    raise ValueError("Input delay CSV contains missing values. This script expects a complete matrix.")

delay_matrix_df.index = delay_matrix_df.index.astype(float).astype(int)
delay_matrix_df.columns = delay_matrix_df.columns.astype(float).astype(int)
delay_matrix_df = delay_matrix_df.sort_index().sort_index(axis=1)
delay_matrix_df.index.name = "ABL"
delay_matrix_df.columns.name = "abs_ILD"

delay_long_df = (
    delay_matrix_df
    .stack()
    .rename("delay_ms")
    .reset_index()
    .astype({"ABL": float, "abs_ILD": float, "delay_ms": float})
)

if len(delay_long_df) != len(delay_matrix_df.index) * len(delay_matrix_df.columns):
    raise ValueError("Reshaped delay table does not contain the expected full grid of points.")

abl_values = delay_long_df["ABL"].to_numpy(dtype=float)
abs_ild_values = delay_long_df["abs_ILD"].to_numpy(dtype=float)
delay_values_ms = delay_long_df["delay_ms"].to_numpy(dtype=float)

print(f"Loaded delay matrix CSV: {input_delay_csv_path}")
print(f"Delay matrix shape: {delay_matrix_df.shape[0]} ABL x {delay_matrix_df.shape[1]} abs_ILD")
print(f"Total fit points: {len(delay_long_df)}")


# %%
############ Fit linear and linear + ILD^2 delay surfaces ############
design_matrix_linear = build_linear_design_matrix(abl_values=abl_values, abs_ild_values=abs_ild_values)
design_matrix_linear_plus_ild2 = build_linear_plus_ild2_design_matrix(
    abl_values=abl_values,
    abs_ild_values=abs_ild_values,
)

linear_coefficients, linear_pred_ms, linear_residual_ms, linear_rmse_ms, linear_r2 = solve_ols_and_metrics(
    design_matrix=design_matrix_linear,
    target_values=delay_values_ms,
)
(
    linear_plus_ild2_coefficients,
    linear_plus_ild2_pred_ms,
    linear_plus_ild2_residual_ms,
    linear_plus_ild2_rmse_ms,
    linear_plus_ild2_r2,
) = solve_ols_and_metrics(
    design_matrix=design_matrix_linear_plus_ild2,
    target_values=delay_values_ms,
)

linear_equation = format_linear_equation(linear_coefficients)
linear_plus_ild2_equation = format_linear_plus_ild2_equation(linear_plus_ild2_coefficients)

print("Linear model equation:")
print(f"  {linear_equation}")
print("Linear + ILD^2 model equation:")
print(f"  {linear_plus_ild2_equation}")

metrics_df = pd.DataFrame(
    [
        {
            "model_name": "linear",
            "n_points": len(delay_long_df),
            "n_parameters": design_matrix_linear.shape[1],
            "rmse_ms": linear_rmse_ms,
            "r2": linear_r2,
        },
        {
            "model_name": "linear_plus_ild2",
            "n_points": len(delay_long_df),
            "n_parameters": design_matrix_linear_plus_ild2.shape[1],
            "rmse_ms": linear_plus_ild2_rmse_ms,
            "r2": linear_plus_ild2_r2,
        },
    ]
)

print("Model comparison metrics:")
print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

best_model_name = str(metrics_df.sort_values("r2", ascending=False).iloc[0]["model_name"])
print(f"Better model by R^2: {best_model_name}")


# %%
############ Save long-form predictions and metrics CSVs ############
predictions_df = delay_long_df.copy()
predictions_df = predictions_df.rename(columns={"delay_ms": "delay_ms_observed"})
predictions_df["delay_ms_linear_pred"] = linear_pred_ms
predictions_df["delay_ms_linear_plus_ild2_pred"] = linear_plus_ild2_pred_ms
predictions_df["linear_residual_ms"] = predictions_df["delay_ms_observed"] - predictions_df["delay_ms_linear_pred"]
predictions_df["linear_plus_ild2_residual_ms"] = (
    predictions_df["delay_ms_observed"] - predictions_df["delay_ms_linear_plus_ild2_pred"]
)

predictions_csv_path = output_dir / "delay_surface_fit_predictions_long.csv"
metrics_csv_path = output_dir / "delay_surface_fit_metrics.csv"
predictions_df.to_csv(predictions_csv_path, index=False)
metrics_df.to_csv(metrics_csv_path, index=False)

print(f"Saved predictions CSV: {predictions_csv_path}")
print(f"Saved metrics CSV: {metrics_csv_path}")


# %%
############ Build matrices for plotting ############
observed_matrix_df = delay_matrix_df.copy()
linear_pred_matrix_df = (
    predictions_df
    .pivot(index="ABL", columns="abs_ILD", values="delay_ms_linear_pred")
    .reindex(index=observed_matrix_df.index.astype(float), columns=observed_matrix_df.columns.astype(float))
)
linear_plus_ild2_pred_matrix_df = (
    predictions_df
    .pivot(index="ABL", columns="abs_ILD", values="delay_ms_linear_plus_ild2_pred")
    .reindex(index=observed_matrix_df.index.astype(float), columns=observed_matrix_df.columns.astype(float))
)
linear_residual_matrix_df = (
    predictions_df
    .pivot(index="ABL", columns="abs_ILD", values="linear_residual_ms")
    .reindex(index=observed_matrix_df.index.astype(float), columns=observed_matrix_df.columns.astype(float))
)
linear_plus_ild2_residual_matrix_df = (
    predictions_df
    .pivot(index="ABL", columns="abs_ILD", values="linear_plus_ild2_residual_ms")
    .reindex(index=observed_matrix_df.index.astype(float), columns=observed_matrix_df.columns.astype(float))
)

linear_pred_matrix_df.index = linear_pred_matrix_df.index.astype(int)
linear_pred_matrix_df.columns = linear_pred_matrix_df.columns.astype(int)
linear_plus_ild2_pred_matrix_df.index = linear_plus_ild2_pred_matrix_df.index.astype(int)
linear_plus_ild2_pred_matrix_df.columns = linear_plus_ild2_pred_matrix_df.columns.astype(int)
linear_residual_matrix_df.index = linear_residual_matrix_df.index.astype(int)
linear_residual_matrix_df.columns = linear_residual_matrix_df.columns.astype(int)
linear_plus_ild2_residual_matrix_df.index = linear_plus_ild2_residual_matrix_df.index.astype(int)
linear_plus_ild2_residual_matrix_df.columns = linear_plus_ild2_residual_matrix_df.columns.astype(int)

matrix_vmin = float(
    min(
        observed_matrix_df.to_numpy().min(),
        linear_pred_matrix_df.to_numpy().min(),
        linear_plus_ild2_pred_matrix_df.to_numpy().min(),
    )
)
matrix_vmax = float(
    max(
        observed_matrix_df.to_numpy().max(),
        linear_pred_matrix_df.to_numpy().max(),
        linear_plus_ild2_pred_matrix_df.to_numpy().max(),
    )
)
residual_abs_max = float(
    max(
        np.abs(linear_residual_matrix_df.to_numpy()).max(),
        np.abs(linear_plus_ild2_residual_matrix_df.to_numpy()).max(),
    )
)


# %%
############ Plot observed matrix, fitted matrices, residuals, and scatter ############
fig, axes = plt.subplots(
    2,
    3,
    figsize=(panel_width * 3, panel_height * 2),
    squeeze=False,
)

matrix_plot_specs = [
    (axes[0, 0], observed_matrix_df, "Observed Delay (ms)", "viridis", matrix_vmin, matrix_vmax),
    (axes[0, 1], linear_pred_matrix_df, "Linear Fit (ms)", "viridis", matrix_vmin, matrix_vmax),
    (axes[0, 2], linear_plus_ild2_pred_matrix_df, "Linear + ILD^2 Fit (ms)", "viridis", matrix_vmin, matrix_vmax),
    (axes[1, 0], linear_residual_matrix_df, "Linear Residual (ms)", "coolwarm", -residual_abs_max, residual_abs_max),
    (axes[1, 1], linear_plus_ild2_residual_matrix_df, "Linear + ILD^2 Residual (ms)", "coolwarm", -residual_abs_max, residual_abs_max),
]

for ax, matrix_df, title, cmap, vmin, vmax in matrix_plot_specs:
    image = ax.imshow(
        matrix_df.to_numpy(dtype=float),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_xticklabels([str(int(value)) for value in matrix_df.columns])
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_yticklabels([str(int(value)) for value in matrix_df.index])
    ax.set_xlabel("abs_ILD")
    ax.set_ylabel("ABL")
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax.xaxis.label.set_size(axis_label_fontsize)
    ax.yaxis.label.set_size(axis_label_fontsize)
    ax.title.set_fontsize(title_fontsize)
    annotate_matrix(ax, matrix_df)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=tick_label_fontsize)

scatter_ax = axes[1, 2]
scatter_ax.scatter(
    predictions_df["delay_ms_observed"],
    predictions_df["delay_ms_linear_pred"],
    s=48,
    color="tab:blue",
    alpha=0.85,
    marker="o",
    label=f"Linear (R^2 = {linear_r2:.3f})",
)
scatter_ax.scatter(
    predictions_df["delay_ms_observed"],
    predictions_df["delay_ms_linear_plus_ild2_pred"],
    s=56,
    color="tab:orange",
    alpha=0.85,
    marker="^",
    label=f"Linear + ILD^2 (R^2 = {linear_plus_ild2_r2:.3f})",
)

scatter_min = float(
    min(
        predictions_df["delay_ms_observed"].min(),
        predictions_df["delay_ms_linear_pred"].min(),
        predictions_df["delay_ms_linear_plus_ild2_pred"].min(),
    )
)
scatter_max = float(
    max(
        predictions_df["delay_ms_observed"].max(),
        predictions_df["delay_ms_linear_pred"].max(),
        predictions_df["delay_ms_linear_plus_ild2_pred"].max(),
    )
)
scatter_pad = 0.05 * (scatter_max - scatter_min)
scatter_line_min = scatter_min - scatter_pad
scatter_line_max = scatter_max + scatter_pad
scatter_ax.plot(
    [scatter_line_min, scatter_line_max],
    [scatter_line_min, scatter_line_max],
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="y = x",
)
scatter_ax.set_xlim(scatter_line_min, scatter_line_max)
scatter_ax.set_ylim(scatter_line_min, scatter_line_max)
scatter_ax.set_title("Observed vs Predicted")
scatter_ax.set_xlabel("Observed Delay (ms)")
scatter_ax.set_ylabel("Predicted Delay (ms)")
scatter_ax.tick_params(axis="both", labelsize=tick_label_fontsize)
scatter_ax.xaxis.label.set_size(axis_label_fontsize)
scatter_ax.yaxis.label.set_size(axis_label_fontsize)
scatter_ax.title.set_fontsize(title_fontsize)
scatter_ax.grid(alpha=0.2, linewidth=0.6)
scatter_ax.legend(fontsize=legend_fontsize)

fig.suptitle(
    f"Delay Surface Fits From 5σ Baseline-Significant CSV | Best R^2: {best_model_name}",
    y=1.02,
    fontsize=suptitle_fontsize,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])

plot_output_path = output_dir / "delay_surface_fit_comparison.png"
fig.savefig(plot_output_path, dpi=png_dpi, bbox_inches="tight")
print(f"Saved comparison figure PNG: {plot_output_path}")


# %%
############ Plot delay vs abs_ILD at fixed ABL values ############
fig_by_abl, axes_by_abl = plt.subplots(
    1,
    len(observed_matrix_df.index),
    figsize=(slice_panel_width * len(observed_matrix_df.index), slice_panel_height),
    sharey=True,
    squeeze=False,
)

abs_ild_grid = np.linspace(float(observed_matrix_df.columns.min()), float(observed_matrix_df.columns.max()), 400)
linear_title_text = format_equation_for_title(linear_equation)
linear_plus_ild2_title_text = format_equation_for_title(linear_plus_ild2_equation)

for col_idx, abl in enumerate(observed_matrix_df.index.astype(int)):
    ax = axes_by_abl[0, col_idx]
    condition_df = predictions_df[predictions_df["ABL"] == float(abl)].sort_values("abs_ILD")

    linear_grid_pred_ms = predict_linear_delay_ms(
        abl_values=np.full_like(abs_ild_grid, float(abl), dtype=float),
        abs_ild_values=abs_ild_grid,
        coefficients=linear_coefficients,
    )
    linear_plus_ild2_grid_pred_ms = predict_linear_plus_ild2_delay_ms(
        abl_values=np.full_like(abs_ild_grid, float(abl), dtype=float),
        abs_ild_values=abs_ild_grid,
        coefficients=linear_plus_ild2_coefficients,
    )

    ax.scatter(
        condition_df["abs_ILD"],
        condition_df["delay_ms_observed"],
        s=70,
        color="black",
        alpha=0.9,
        label="Observed",
        zorder=3,
    )
    ax.plot(
        abs_ild_grid,
        linear_grid_pred_ms,
        color="tab:blue",
        linewidth=2.2,
        label="Linear fit",
    )
    ax.plot(
        abs_ild_grid,
        linear_plus_ild2_grid_pred_ms,
        color="tab:orange",
        linewidth=2.2,
        label="Linear + ILD^2 fit",
    )

    ax.set_title(f"ABL = {abl}", fontsize=title_fontsize)
    ax.set_xlabel("abs_ILD", fontsize=axis_label_fontsize)
    if col_idx == 0:
        ax.set_ylabel("Delay (ms)", fontsize=axis_label_fontsize)
    ax.set_xticks(observed_matrix_df.columns.astype(int))
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=legend_fontsize)

fig_by_abl.suptitle(
    "Delay vs abs_ILD at fixed ABL",
    y=1.12,
    fontsize=suptitle_fontsize,
)
fig_by_abl.text(
    0.5,
    1.05,
    f"Linear: {linear_title_text}",
    ha="center",
    va="center",
    fontsize=equation_fontsize,
)
fig_by_abl.text(
    0.5,
    1.00,
    f"Linear + ILD^2: {linear_plus_ild2_title_text}",
    ha="center",
    va="center",
    fontsize=equation_fontsize,
)
fig_by_abl.tight_layout(rect=[0, 0, 1, 0.90])

plot_by_abl_output_path = output_dir / "delay_vs_abs_ILD_by_ABL.png"
fig_by_abl.savefig(plot_by_abl_output_path, dpi=png_dpi, bbox_inches="tight")
print(f"Saved ABL-slice figure PNG: {plot_by_abl_output_path}")


# %%
############ Plot delay vs ABL at fixed abs_ILD values ############
fig_by_abs_ild, axes_by_abs_ild = plt.subplots(
    1,
    len(observed_matrix_df.columns),
    figsize=(slice_panel_width * len(observed_matrix_df.columns), slice_panel_height),
    sharey=True,
    squeeze=False,
)

abl_grid = np.linspace(float(observed_matrix_df.index.min()), float(observed_matrix_df.index.max()), 400)

for col_idx, abs_ild in enumerate(observed_matrix_df.columns.astype(int)):
    ax = axes_by_abs_ild[0, col_idx]
    condition_df = predictions_df[predictions_df["abs_ILD"] == float(abs_ild)].sort_values("ABL")

    linear_grid_pred_ms = predict_linear_delay_ms(
        abl_values=abl_grid,
        abs_ild_values=np.full_like(abl_grid, float(abs_ild), dtype=float),
        coefficients=linear_coefficients,
    )
    linear_plus_ild2_grid_pred_ms = predict_linear_plus_ild2_delay_ms(
        abl_values=abl_grid,
        abs_ild_values=np.full_like(abl_grid, float(abs_ild), dtype=float),
        coefficients=linear_plus_ild2_coefficients,
    )

    ax.scatter(
        condition_df["ABL"],
        condition_df["delay_ms_observed"],
        s=70,
        color="black",
        alpha=0.9,
        label="Observed",
        zorder=3,
    )
    ax.plot(
        abl_grid,
        linear_grid_pred_ms,
        color="tab:blue",
        linewidth=2.2,
        label="Linear fit",
    )
    ax.plot(
        abl_grid,
        linear_plus_ild2_grid_pred_ms,
        color="tab:orange",
        linewidth=2.2,
        label="Linear + ILD^2 fit",
    )

    ax.set_title(f"|ILD| = {abs_ild}", fontsize=title_fontsize)
    ax.set_xlabel("ABL", fontsize=axis_label_fontsize)
    if col_idx == 0:
        ax.set_ylabel("Delay (ms)", fontsize=axis_label_fontsize)
    ax.set_xticks(observed_matrix_df.index.astype(int))
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=legend_fontsize)

fig_by_abs_ild.suptitle(
    "Delay vs ABL at fixed abs_ILD",
    y=1.12,
    fontsize=suptitle_fontsize,
)
fig_by_abs_ild.text(
    0.5,
    1.05,
    f"Linear: {linear_title_text}",
    ha="center",
    va="center",
    fontsize=equation_fontsize,
)
fig_by_abs_ild.text(
    0.5,
    1.00,
    f"Linear + ILD^2: {linear_plus_ild2_title_text}",
    ha="center",
    va="center",
    fontsize=equation_fontsize,
)
fig_by_abs_ild.tight_layout(rect=[0, 0, 1, 0.90])

plot_by_abs_ild_output_path = output_dir / "delay_vs_ABL_by_abs_ILD.png"
fig_by_abs_ild.savefig(plot_by_abs_ild_output_path, dpi=png_dpi, bbox_inches="tight")
print(f"Saved abs_ILD-slice figure PNG: {plot_by_abs_ild_output_path}")

if show_plot:
    plt.show()
else:
    plt.close(fig)
    plt.close(fig_by_abl)
    plt.close(fig_by_abs_ild)
