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

PKL_BASE = (
    "results_norm_tied_batch_LED7_aggregate_ledoff_1_"
    "proactive_loaded_truncate_NOT_censor_ABL_delay_no_choice_"
)

# Define the fits to compare: (truncation_ms, fix_trial_ON, run_tag_suffix, label)
fit_specs = [
    {
        "trunc_ms": 100,
        "fixN": False,
        "suffix": "trunc100ms_allvalid",
        "label": "100 ms, allvalid",
    },
    {
        "trunc_ms": 115,
        "fixN": False,
        "suffix": "trunc115ms_allvalid",
        "label": "115 ms, allvalid",
    },
    {
        "trunc_ms": 115,
        "fixN": True,
        "suffix": "trunc115ms_fixN_20-1300_40-2300_60-3400",
        "label": "115 ms, fixN",
    },
    {
        "trunc_ms": 130,
        "fixN": False,
        "suffix": "trunc130ms_allvalid",
        "label": "130 ms, allvalid",
    },
    {
        "trunc_ms": 130,
        "fixN": True,
        "suffix": "trunc130ms_fixN_20-1300_40-2300_60-3400",
        "label": "130 ms, fixN",
    },
    {
        "trunc_ms": 145,
        "fixN": False,
        "suffix": "trunc145ms_allvalid",
        "label": "145 ms, allvalid",
    },
]

# Style mapping:
#   truncation -> color
#   fixN OFF (allvalid) -> faint, fixN ON -> opaque
COLOR_MAP = {100: "black", 115: "tab:blue", 130: "tab:red", 145: "tab:green"}
ALPHA_MAP = {False: 0.4, True: 1.0}

output_dir = RESULTS_DIR / "posterior_comparisons"
output_dir.mkdir(parents=True, exist_ok=True)

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


def compute_common_bins(all_samples_list, n_bins):
    lo = min(np.min(s) for s in all_samples_list)
    hi = max(np.max(s) for s in all_samples_list)
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


def build_mean_summary_lines(fits, row_info, items_per_line=2):
    summary_chunks = []
    ordered_truncations = list(dict.fromkeys(fit["spec"]["trunc_ms"] for fit in fits))
    for trunc_ms in ordered_truncations:
        case_parts = []
        for idx, fit in enumerate(fits):
            if fit["spec"]["trunc_ms"] != trunc_ms:
                continue
            case_label = "fix" if fit["spec"]["fixN"] else "all"
            case_parts.append(f"{case_label}:{row_info[f'mean_{idx}']:.2f}")
        summary_chunks.append(f"{trunc_ms}ms — " + ", ".join(case_parts))

    title_lines = []
    for idx in range(0, len(summary_chunks), items_per_line):
        title_lines.append(" | ".join(summary_chunks[idx:idx + items_per_line]))
    return title_lines


def build_color_summary(fits):
    ordered_truncations = list(dict.fromkeys(fit["spec"]["trunc_ms"] for fit in fits))
    return ", ".join(
        f"{trunc_ms} ms={COLOR_MAP[trunc_ms].replace('tab:', '')}"
        for trunc_ms in ordered_truncations
    )


# %%
# =============================================================================
# Load all requested results
# =============================================================================
fits = []
for fspec in fit_specs:
    pkl_path = RESULTS_DIR / f"{PKL_BASE}{fspec['suffix']}.pkl"
    payload, vbmc_results, fit_config, fit_trial_counts = load_fit_payload(pkl_path)
    run_tag = fit_config.get("run_tag", fspec["suffix"])
    n_trials = int(fit_trial_counts.get("valid_trials_used_for_fit", -1))
    fits.append({
        "spec": fspec,
        "vbmc_results": vbmc_results,
        "run_tag": run_tag,
        "n_trials": n_trials,
        "color": COLOR_MAP[fspec["trunc_ms"]],
        "alpha": ALPHA_MAP[fspec["fixN"]],
        "label": fspec["label"],
    })
    print(f"Loaded {fspec['label']:30s}  run_tag={run_tag}  n_trials={n_trials}")

# Verify all param keys present
for fit in fits:
    missing = [s["key"] for s in param_specs if s["key"] not in fit["vbmc_results"]]
    if missing:
        raise KeyError(f"Missing keys in {fit['label']}: {missing}")

output_base = "posterior_compare_no_choice_multi_fit"
output_pdf_path = output_dir / f"{output_base}.pdf"
output_png_path = output_dir / f"{output_base}.png"


# %%
# =============================================================================
# Plot posterior comparison — all selected fits
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.ravel()
summary_rows = []

for ax, pspec in zip(axes, param_specs):
    param_key = pspec["key"]
    scale = pspec["scale"]
    unit = pspec["unit"]
    width_fmt = pspec["width_fmt"]
    x_label = format_param_label(pspec["label"], unit)

    # Collect samples for all selected fits
    all_samples = []
    for fit in fits:
        samples = get_scaled_samples(fit["vbmc_results"], param_key, scale)
        all_samples.append(samples)

    bins = compute_common_bins(all_samples, n_bins)

    row_info = {"label": x_label, "width_fmt": width_fmt}
    for i, (fit, samples) in enumerate(zip(fits, all_samples)):
        q025, q975, width = compute_interval(samples)
        mean_val = float(np.mean(samples))

        ax.hist(
            samples,
            bins=bins,
            density=hist_density,
            histtype=histtype,
            linewidth=hist_linewidth,
            color=fit["color"],
            alpha=fit["alpha"],
            label=f"{fit['label']} (n={fit['n_trials']})",
        )

        # Quantile lines — same color/alpha as the histogram
        for x in (q025, q975):
            ax.axvline(
                x=x,
                color=fit["color"],
                linewidth=quantile_linewidth,
                alpha=fit["alpha"] * 0.7,
            )

        row_info[f"mean_{i}"] = mean_val
        row_info[f"width_{i}"] = width

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.2)
    title_lines = [x_label, *build_mean_summary_lines(fits, row_info)]
    ax.set_title("\n".join(title_lines), fontsize=9)
    summary_rows.append(row_info)

for ax in axes[len(param_specs):]:
    ax.axis("off")

# Build legend from one axis — deduplicate
handles, labels = axes[0].get_legend_handles_labels()
legend_ncol = min(len(fits), 4)
fig.legend(handles, labels, loc="upper center", ncol=legend_ncol, frameon=False,
           bbox_to_anchor=(0.5, 0.965), fontsize=9)

fig.suptitle(
    f"Posterior comparison — no-choice fit ({len(fits)} conditions)\n"
    f"{build_color_summary(fits)} | opaque = fixN ON, faint = allvalid",
    y=0.995, fontsize=11,
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
header_labels = [f["label"] for f in fits]
col_w = 18
print()
print(f"{'Parameter':<20}" + "".join(f"{h:>{col_w}}" for h in header_labels))
print("-" * (20 + col_w * len(fits)))
for row in summary_rows:
    wfmt = row["width_fmt"]
    cols = "".join(
        f"{fmt_value(row[f'width_{i}'], wfmt):>{col_w}}" for i in range(len(fits))
    )
    print(f"{row['label']:<20}{cols}")
