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
RESULTS_DIR = (
    SCRIPT_DIR / "norm_only_led_off_from_loaded_proactive_truncate_NOT_censor_ABL_delay_no_choice"
)

PKL_BASE = (
    "results_norm_tied_batch_LED7_aggregate_ledoff_1_"
    "proactive_loaded_truncate_NOT_censor_ABL_delay_no_choice_"
)

fit_specs = [
    {
        "label": "115 ms, allvalid",
        "suffix": "trunc115ms_allvalid",
        "color": "tab:blue",
    },
    {
        "label": "130 ms, allvalid",
        "suffix": "trunc130ms_allvalid",
        "color": "tab:red",
    },
]

output_dir = SCRIPT_DIR / "posterior_comparisons"
output_dir.mkdir(parents=True, exist_ok=True)

output_base = "posterior_compare_no_choice_115ms_allvalid_vs_130ms_allvalid"

n_bins = 60
hist_density = True
histtype = "step"
hist_linewidth = 1.8
quantile_linewidth = 1.0
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

    return {
        "payload": payload,
        "vbmc_results": vbmc_results,
        "fit_config": payload.get("fit_config", {}),
        "fit_trial_counts": payload.get("fit_trial_counts", {}),
    }


def get_scaled_samples(vbmc_results, param_key, scale):
    if param_key not in vbmc_results:
        raise KeyError(f"Missing posterior samples for {param_key}")

    samples = np.asarray(vbmc_results[param_key], dtype=float).ravel() * float(scale)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError(f"No finite posterior samples found for {param_key}")
    return samples


def compute_common_bins(all_samples_list, n_bins):
    lo = min(np.min(samples) for samples in all_samples_list)
    hi = max(np.max(samples) for samples in all_samples_list)
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


def build_title_lines(samples_by_fit, width_fmt):
    title_lines = []
    for fit_idx, samples in enumerate(samples_by_fit):
        mean_val = float(np.mean(samples))
        _, _, width_val = compute_interval(samples)
        fit_label = fits[fit_idx]["label"]
        title_lines.append(
            f"{fit_label}: mean={fmt_value(mean_val, width_fmt)}, "
            f"95% width={fmt_value(width_val, width_fmt)}"
        )
    return title_lines


# %%
# =============================================================================
# Load requested results
# =============================================================================
fits = []
for fit_spec in fit_specs:
    pkl_path = RESULTS_DIR / f"{PKL_BASE}{fit_spec['suffix']}.pkl"
    loaded = load_fit_payload(pkl_path)
    n_trials = int(loaded["fit_trial_counts"].get("valid_trials_used_for_fit", -1))
    run_tag = loaded["fit_config"].get("run_tag", fit_spec["suffix"])

    fit_entry = {
        "label": fit_spec["label"],
        "color": fit_spec["color"],
        "pkl_path": pkl_path,
        "run_tag": run_tag,
        "n_trials": n_trials,
        "payload": loaded["payload"],
        "vbmc_results": loaded["vbmc_results"],
        "fit_config": loaded["fit_config"],
        "fit_trial_counts": loaded["fit_trial_counts"],
    }
    fits.append(fit_entry)
    print(f"Loaded {fit_entry['label']:20s} run_tag={run_tag} n_trials={n_trials}")

for fit in fits:
    missing_keys = [spec["key"] for spec in param_specs if spec["key"] not in fit["vbmc_results"]]
    if missing_keys:
        raise KeyError(f"Missing keys in {fit['label']}: {missing_keys}")

same_trial_count = fits[0]["n_trials"] == fits[1]["n_trials"]
print(f"Same number of trials across fits: {same_trial_count}")

output_pdf_path = output_dir / f"{output_base}.pdf"
output_png_path = output_dir / f"{output_base}.png"


# %%
# =============================================================================
# Plot posterior comparison
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.ravel()
summary_rows = []

for ax, param_spec in zip(axes, param_specs):
    param_key = param_spec["key"]
    scale = param_spec["scale"]
    unit = param_spec["unit"]
    width_fmt = param_spec["width_fmt"]
    x_label = format_param_label(param_spec["label"], unit)

    samples_by_fit = [
        get_scaled_samples(fit["vbmc_results"], param_key, scale)
        for fit in fits
    ]
    bins = compute_common_bins(samples_by_fit, n_bins)

    row_info = {"label": x_label, "width_fmt": width_fmt}
    for fit_idx, (fit, samples) in enumerate(zip(fits, samples_by_fit)):
        q025, q975, width_val = compute_interval(samples)
        mean_val = float(np.mean(samples))

        ax.hist(
            samples,
            bins=bins,
            density=hist_density,
            histtype=histtype,
            linewidth=hist_linewidth,
            color=fit["color"],
            label=f"{fit['label']} (n={fit['n_trials']})",
        )

        for x in (q025, q975):
            ax.axvline(
                x=x,
                color=fit["color"],
                linewidth=quantile_linewidth,
                alpha=0.7,
            )

        row_info[f"mean_{fit_idx}"] = mean_val
        row_info[f"width_{fit_idx}"] = width_val

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("\n".join([x_label, *build_title_lines(samples_by_fit, width_fmt)]), fontsize=9)
    summary_rows.append(row_info)

for ax in axes[len(param_specs):]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.965),
    fontsize=9,
)

fig.suptitle(
    "Posterior comparison — no-choice fit\n"
    f"115 ms allvalid (n={fits[0]['n_trials']}) vs 130 ms allvalid (n={fits[1]['n_trials']})",
    y=0.995,
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(output_pdf_path, bbox_inches="tight")
fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
print(f"\nSaved PDF: {output_pdf_path}")
print(f"Saved PNG: {output_png_path}")

if show_plot:
    plt.show()
plt.close(fig)


# %%
# =============================================================================
# Print width summary table
# =============================================================================
header_labels = [fit["label"] for fit in fits]
col_w = 24

print()
print(f"{'Parameter':<20}" + "".join(f"{label:>{col_w}}" for label in header_labels))
print("-" * (20 + col_w * len(fits)))
for row in summary_rows:
    width_fmt = row["width_fmt"]
    width_cols = "".join(
        f"{fmt_value(row[f'width_{fit_idx}'], width_fmt):>{col_w}}"
        for fit_idx in range(len(fits))
    )
    print(f"{row['label']:<20}{width_cols}")
