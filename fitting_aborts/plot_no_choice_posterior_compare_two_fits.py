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

allvalid_results_pkl_path = (
    RESULTS_DIR
    / "results_norm_tied_batch_LED7_aggregate_ledoff_1_"
    "proactive_loaded_truncate_NOT_censor_ABL_delay_no_choice_trunc130ms_allvalid.pkl"
)
fixedn_results_pkl_path = (
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
allvalid_color = "tab:blue"
fixedn_color = "tab:red"
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
allvalid_payload, allvalid_results, allvalid_fit_config, allvalid_trial_counts = load_fit_payload(
    allvalid_results_pkl_path
)
fixedn_payload, fixedn_results, fixedn_fit_config, fixedn_trial_counts = load_fit_payload(
    fixedn_results_pkl_path
)

allvalid_run_tag = allvalid_fit_config.get("run_tag", "unknown_run_tag")
fixedn_run_tag = fixedn_fit_config.get("run_tag", "unknown_run_tag")
allvalid_n_trials = int(allvalid_trial_counts.get("valid_trials_used_for_fit", -1))
fixedn_n_trials = int(fixedn_trial_counts.get("valid_trials_used_for_fit", -1))

missing_param_keys = [
    spec["key"]
    for spec in param_specs
    if spec["key"] not in allvalid_results or spec["key"] not in fixedn_results
]
if missing_param_keys:
    raise KeyError(f"Missing posterior sample keys across the two fits: {missing_param_keys}")

output_base = (
    "posterior_compare_trunc130ms_no_choice_"
    f"{allvalid_run_tag}_vs_{fixedn_run_tag}"
)
output_pdf_path = output_dir / f"{output_base}.pdf"
output_png_path = output_dir / f"{output_base}.png"

print(f"All-valid run tag: {allvalid_run_tag}")
print(f"Fixed-N run tag: {fixedn_run_tag}")
print(f"All-valid fitted trials: {allvalid_n_trials}")
print(f"Fixed-N fitted trials: {fixedn_n_trials}")


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

    allvalid_samples = get_scaled_samples(allvalid_results, param_key, scale)
    fixedn_samples = get_scaled_samples(fixedn_results, param_key, scale)
    bins = compute_plot_bins(allvalid_samples, fixedn_samples, n_bins)

    allvalid_q025, allvalid_q975, allvalid_width = compute_interval(allvalid_samples)
    fixedn_q025, fixedn_q975, fixedn_width = compute_interval(fixedn_samples)
    allvalid_mean = float(np.mean(allvalid_samples))
    fixedn_mean = float(np.mean(fixedn_samples))
    width_ratio = fixedn_width / allvalid_width if allvalid_width > 0 else np.nan

    ax.hist(
        allvalid_samples,
        bins=bins,
        density=hist_density,
        histtype=histtype,
        linewidth=hist_linewidth,
        color=allvalid_color,
        label=f"All trials (n={allvalid_n_trials})",
    )
    ax.hist(
        fixedn_samples,
        bins=bins,
        density=hist_density,
        histtype=histtype,
        linewidth=hist_linewidth,
        color=fixedn_color,
        label=f"Fixed trials (n={fixedn_n_trials})",
    )

    for x in (allvalid_q025, allvalid_q975):
        ax.axvline(
            x=x,
            color=allvalid_color,
            linestyle=quantile_linestyle,
            linewidth=quantile_linewidth,
            alpha=0.9,
        )
    for x in (fixedn_q025, fixedn_q975):
        ax.axvline(
            x=x,
            color=fixedn_color,
            linestyle=quantile_linestyle,
            linewidth=quantile_linewidth,
            alpha=0.9,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(
        f"{x_label}\n"
        f"mean all={allvalid_mean:.2f}, fixed={fixedn_mean:.2f}\n"
        f"width all={fmt_value(allvalid_width, width_fmt)}, fixed={fmt_value(fixedn_width, width_fmt)}"
    )

    summary_rows.append(
        {
            "label": x_label,
            "allvalid_width": allvalid_width,
            "fixedn_width": fixedn_width,
            "width_ratio": width_ratio,
            "width_fmt": width_fmt,
        }
    )

for ax in axes[len(param_specs):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
fig.suptitle(
    "Posterior comparison, 130 ms no-choice fit\n"
    f"blue={allvalid_run_tag} (n={allvalid_n_trials}) | "
    f"red={fixedn_run_tag} (n={fixedn_n_trials})",
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
    f"{'Parameter':<20} {'All width':>14} {'Fixed width':>14} "
    f"{'Fixed-All':>14} {'Fixed/All':>14}"
)
print("-" * 80)
for row in summary_rows:
    width_fmt = row["width_fmt"]
    width_diff = row["fixedn_width"] - row["allvalid_width"]
    width_ratio = row["width_ratio"]
    ratio_text = "nan" if not np.isfinite(width_ratio) else f"{width_ratio:.4f}"
    print(
        f"{row['label']:<20} "
        f"{fmt_value(row['allvalid_width'], width_fmt):>14} "
        f"{fmt_value(row['fixedn_width'], width_fmt):>14} "
        f"{fmt_value(width_diff, width_fmt):>14} "
        f"{ratio_text:>14}"
    )
