# %%
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


# %%
# =============================================================================
# Parameters
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay_no_choice"

fit_a_label = "115 ms fixed"
fit_a_results_pkl_path = (
    RESULTS_DIR
    / "results_norm_tied_batch_LED7_aggregate_ledoff_1_"
    "proactive_loaded_truncate_NOT_censor_ABL_delay_no_choice_"
    "trunc115ms_fixN_20-1300_40-2300_60-3400.pkl"
)
fit_b_label = "130 ms fixed"
fit_b_results_pkl_path = (
    RESULTS_DIR
    / "results_norm_tied_batch_LED7_aggregate_ledoff_1_"
    "proactive_loaded_truncate_NOT_censor_ABL_delay_no_choice_"
    "trunc130ms_fixN_20-1300_40-2300_60-3400.pkl"
)

output_dir = RESULTS_DIR / "posterior_comparisons"
output_dir.mkdir(parents=True, exist_ok=True)

n_bins = 60
hist_density = True
histtype = "step"
fit_a_color = "tab:blue"
fit_b_color = "tab:red"
hist_linewidth = 1.8
quantile_linewidth = 1.2
quantile_linestyle = "--"
show_plot = True

param_specs = [
    {"key": "rate_lambda_samples", "label": "rate_lambda", "scale": 1.0, "unit": "", "width_fmt": ".4f"},
    {"key": "T_0_samples", "label": "T_0", "scale": 1e3, "unit": "ms", "width_fmt": ".2f"},
    {"key": "theta_E_samples", "label": "theta_E", "scale": 1.0, "unit": "", "width_fmt": ".4f"},
    {"key": "w_samples", "label": "w", "scale": 1.0, "unit": "", "width_fmt": ".4f"},
    {"key": "t_E_aff_20_samples", "label": "t_E_aff_20", "scale": 1e3, "unit": "ms", "width_fmt": ".2f"},
    {"key": "t_E_aff_40_samples", "label": "t_E_aff_40", "scale": 1e3, "unit": "ms", "width_fmt": ".2f"},
    {"key": "t_E_aff_60_samples", "label": "t_E_aff_60", "scale": 1e3, "unit": "ms", "width_fmt": ".2f"},
    {"key": "del_go_samples", "label": "del_go", "scale": 1e3, "unit": "ms", "width_fmt": ".2f"},
    {"key": "rate_norm_l_samples", "label": "rate_norm_l", "scale": 1.0, "unit": "", "width_fmt": ".4f"},
]


# %%
# =============================================================================
# Helpers
# =============================================================================
def load_fit_payload(pkl_path: Path):
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing results pickle: {pkl_path}")

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    vbmc_results = payload.get("vbmc_norm_tied_results")
    if vbmc_results is None:
        raise KeyError(f"Missing 'vbmc_norm_tied_results' in {pkl_path}")

    fit_config = payload.get("fit_config", {})
    fit_trial_counts = payload.get("fit_trial_counts", {})
    return payload, vbmc_results, fit_config, fit_trial_counts


def get_scaled_samples(vbmc_results, param_key, scale):
    if param_key not in vbmc_results:
        raise KeyError(f"Missing posterior samples for {param_key}")

    samples = np.asarray(vbmc_results[param_key], dtype=float).ravel() * float(scale)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"No finite posterior samples found for {param_key}")
    return samples


def compute_plot_bins(x, y, n_bins):
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if np.isclose(lo, hi):
        pad = 1e-6 if np.isclose(lo, 0.0) else 0.05 * abs(lo)
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, int(n_bins) + 1)


def compute_interval(samples):
    q025, q975 = np.quantile(samples, [0.025, 0.975])
    return float(q025), float(q975), float(q975 - q025)


def format_param_label(label, unit):
    return f"{label} ({unit})" if unit else label


def fmt_value(value, fmt):
    return format(float(value), fmt)


# %%
# =============================================================================
# Load results
# =============================================================================
fit_a_payload, fit_a_results, fit_a_fit_config, fit_a_trial_counts = load_fit_payload(
    fit_a_results_pkl_path
)
fit_b_payload, fit_b_results, fit_b_fit_config, fit_b_trial_counts = load_fit_payload(
    fit_b_results_pkl_path
)

fit_a_run_tag = fit_a_fit_config.get("run_tag", "unknown_run_tag")
fit_b_run_tag = fit_b_fit_config.get("run_tag", "unknown_run_tag")
fit_a_n_trials = int(fit_a_trial_counts.get("valid_trials_used_for_fit", -1))
fit_b_n_trials = int(fit_b_trial_counts.get("valid_trials_used_for_fit", -1))

missing_param_keys = [
    spec["key"]
    for spec in param_specs
    if spec["key"] not in fit_a_results or spec["key"] not in fit_b_results
]
if missing_param_keys:
    raise KeyError(f"Missing posterior sample keys across the two fits: {missing_param_keys}")

output_base = (
    "posterior_compare_no_choice_"
    f"{fit_a_run_tag}_vs_{fit_b_run_tag}"
)
output_pdf_path = output_dir / f"{output_base}.pdf"
output_png_path = output_dir / f"{output_base}.png"

print(f"{fit_a_label} run tag: {fit_a_run_tag}")
print(f"{fit_b_label} run tag: {fit_b_run_tag}")
print(f"{fit_a_label} fitted trials: {fit_a_n_trials}")
print(f"{fit_b_label} fitted trials: {fit_b_n_trials}")


# %%
# =============================================================================
# Plot posterior comparison
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.ravel()
summary_rows = []

for ax, spec in zip(axes, param_specs):
    param_key = spec["key"]
    scale = spec["scale"]
    unit = spec["unit"]
    width_fmt = spec["width_fmt"]
    x_label = format_param_label(spec["label"], unit)

    fit_a_samples = get_scaled_samples(fit_a_results, param_key, scale)
    fit_b_samples = get_scaled_samples(fit_b_results, param_key, scale)
    bins = compute_plot_bins(fit_a_samples, fit_b_samples, n_bins)

    fit_a_q025, fit_a_q975, fit_a_width = compute_interval(fit_a_samples)
    fit_b_q025, fit_b_q975, fit_b_width = compute_interval(fit_b_samples)
    fit_a_mean = float(np.mean(fit_a_samples))
    fit_b_mean = float(np.mean(fit_b_samples))
    width_ratio = fit_b_width / fit_a_width if fit_a_width > 0 else np.nan

    ax.hist(
        fit_a_samples,
        bins=bins,
        density=hist_density,
        histtype=histtype,
        linewidth=hist_linewidth,
        color=fit_a_color,
        label=f"{fit_a_label} (n={fit_a_n_trials})",
    )
    ax.hist(
        fit_b_samples,
        bins=bins,
        density=hist_density,
        histtype=histtype,
        linewidth=hist_linewidth,
        color=fit_b_color,
        label=f"{fit_b_label} (n={fit_b_n_trials})",
    )

    for x in (fit_a_q025, fit_a_q975):
        ax.axvline(
            x=x,
            color=fit_a_color,
            linestyle=quantile_linestyle,
            linewidth=quantile_linewidth,
            alpha=0.9,
        )
    for x in (fit_b_q025, fit_b_q975):
        ax.axvline(
            x=x,
            color=fit_b_color,
            linestyle=quantile_linestyle,
            linewidth=quantile_linewidth,
            alpha=0.9,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(
        f"{x_label}\n"
        f"mean {fit_a_label}={fit_a_mean:.2f}, {fit_b_label}={fit_b_mean:.2f}\n"
        f"width {fit_a_label}={fmt_value(fit_a_width, width_fmt)}, {fit_b_label}={fmt_value(fit_b_width, width_fmt)}"
    )

    summary_rows.append(
        {
            "label": x_label,
            "fit_a_width": fit_a_width,
            "fit_b_width": fit_b_width,
            "width_ratio": width_ratio,
            "width_fmt": width_fmt,
        }
    )

for ax in axes[len(param_specs):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
fig.suptitle(
    "Posterior comparison, no-choice fit\n"
    f"blue={fit_a_run_tag} ({fit_a_label}, n={fit_a_n_trials}) | "
    f"red={fit_b_run_tag} ({fit_b_label}, n={fit_b_n_trials})",
    y=0.995,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(output_pdf_path, bbox_inches="tight")
fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
print(f"Saved posterior comparison plot (PDF): {output_pdf_path}")
print(f"Saved posterior comparison plot (PNG): {output_png_path}")

if show_plot:
    plt.show()
plt.close(fig)


# %%
# =============================================================================
# Print width summary
# =============================================================================
print()
print(
    f"{'Parameter':<20} {fit_a_label:>14} {fit_b_label:>14} "
    f"{'B-A':>14} {'B/A':>14}"
)
print("-" * 80)
for row in summary_rows:
    width_fmt = row["width_fmt"]
    width_diff = row["fit_b_width"] - row["fit_a_width"]
    width_ratio = row["width_ratio"]
    ratio_text = "nan" if not np.isfinite(width_ratio) else f"{width_ratio:.4f}"
    print(
        f"{row['label']:<20} "
        f"{fmt_value(row['fit_a_width'], width_fmt):>14} "
        f"{fmt_value(row['fit_b_width'], width_fmt):>14} "
        f"{fmt_value(width_diff, width_fmt):>14} "
        f"{ratio_text:>14}"
    )
