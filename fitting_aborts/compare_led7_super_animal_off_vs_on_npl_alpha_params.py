# %%
"""Compare accepted LED7 super-animal OFF and bilateral-ON NPL+alpha fits."""

# %%
# =============================================================================
# Editable paths and plotting settings
# =============================================================================
from pathlib import Path
import json
import os
import pickle


SCRIPT_DIR = Path(__file__).resolve().parent
FIT_ROOT = Path(
    os.environ.get(
        "LED7_VALID_NPL_AGG_COMPARE_FIT_ROOT",
        str(
            SCRIPT_DIR
            / "numpyro_svi_proactive_led_step_jump_npl_alpha_valid_led7_"
            "aggregate_patience12_min50k_restore_best_outputs"
            / "LED7_all_animals"
        ),
    )
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get(
        "LED7_VALID_NPL_AGG_COMPARE_OUTPUT_DIR",
        str(FIT_ROOT / "diagnostics"),
    )
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {"off": "tab:blue", "on_bilateral": "tab:red"}
MODEL_LABELS = {"off": "LED OFF", "on_bilateral": "bilateral LED ON"}
ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}

GLOBAL_PARAMETERS = [
    {
        "key": "rate_lambda",
        "label": r"$\lambda^\prime$",
        "unit": "",
        "scale": 1.0,
    },
    {"key": "T_0", "label": r"$T_0$", "unit": "ms", "scale": 1000.0},
    {
        "key": "theta_E",
        "label": r"$\theta_E$",
        "unit": "",
        "scale": 1.0,
    },
    {"key": "w", "label": r"$w$", "unit": "", "scale": 1.0},
    {
        "key": "del_go",
        "label": r"$\delta_{go}$",
        "unit": "ms",
        "scale": 1000.0,
    },
    {
        "key": "rate_norm_l",
        "label": r"$\ell$",
        "unit": "",
        "scale": 1.0,
    },
    {"key": "alpha", "label": r"$\alpha$", "unit": "", "scale": 1.0},
]


# %%
# =============================================================================
# Imports
# =============================================================================
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %%
# =============================================================================
# Load the explicitly accepted OFF and bilateral-ON branches
# =============================================================================
fit_payloads = {}
for category in ["off", "on_bilateral"]:
    selection_path = FIT_ROOT / category / "selection_summary.json"
    if not selection_path.exists():
        raise FileNotFoundError(selection_path)
    with selection_path.open() as handle:
        selection = json.load(handle)
    if selection.get("status") != "complete":
        raise RuntimeError(f"{category} selection is not complete: {selection}")

    accepted_dir = Path(selection["accepted_fit_root"])
    posterior_path = accepted_dir / "main_fullrank_posterior_samples.npz"
    condition_path = accepted_dir / "condition_table.csv"
    summary_path = accepted_dir / "run_summary.json"
    for path in [posterior_path, condition_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(posterior_path) as saved:
        posterior = {name: np.asarray(saved[name], dtype=float) for name in saved.files}
    required = [item["key"] for item in GLOBAL_PARAMETERS] + ["t_E_aff"]
    missing = sorted(set(required) - set(posterior))
    if missing:
        raise RuntimeError(f"{category} posterior is missing {missing}.")
    if not all(np.all(np.isfinite(posterior[name])) for name in required):
        raise RuntimeError(f"{category} posterior contains non-finite samples.")

    condition_table = pd.read_csv(condition_path).sort_values("condition_id")
    if len(condition_table) != 30:
        raise RuntimeError(f"{category}: expected 30 conditions.")
    if condition_table[["ABL", "ILD"]].duplicated().any():
        raise RuntimeError(f"{category}: duplicate stimulus conditions.")

    with summary_path.open() as handle:
        run_summary = json.load(handle)
    if run_summary.get("status") != "complete":
        raise RuntimeError(f"{category} accepted run is not complete.")

    fit_payloads[category] = {
        "selection": selection,
        "fit_dir": accepted_dir,
        "posterior_path": posterior_path,
        "posterior": posterior,
        "condition_table": condition_table,
        "run_summary": run_summary,
    }

print(f"Fit root: {FIT_ROOT.resolve()}")
for category, payload in fit_payloads.items():
    print(
        f"{category}: {payload['selection']['accepted_initialization_mode']} -> "
        f"{payload['fit_dir'].resolve()}"
    )


# %%
# =============================================================================
# Global-parameter summaries and independent posterior differences
# =============================================================================
rng = np.random.default_rng(20260829)
global_rows = []
for parameter in GLOBAL_PARAMETERS:
    key = parameter["key"]
    scale = parameter["scale"]
    off_samples = fit_payloads["off"]["posterior"][key].reshape(-1) * scale
    on_samples = (
        fit_payloads["on_bilateral"]["posterior"][key].reshape(-1) * scale
    )
    n_pairs = min(len(off_samples), len(on_samples))
    off_indices = rng.permutation(len(off_samples))[:n_pairs]
    on_indices = rng.permutation(len(on_samples))[:n_pairs]
    difference_samples = on_samples[on_indices] - off_samples[off_indices]

    row = {
        "parameter": key,
        "plot_label": parameter["label"],
        "unit": parameter["unit"],
    }
    for category, samples in [
        ("off", off_samples),
        ("on_bilateral", on_samples),
    ]:
        row[f"{category}_mean"] = float(np.mean(samples))
        row[f"{category}_q025"] = float(np.quantile(samples, 0.025))
        row[f"{category}_q500"] = float(np.quantile(samples, 0.500))
        row[f"{category}_q975"] = float(np.quantile(samples, 0.975))
    row["on_minus_off_mean"] = float(np.mean(difference_samples))
    row["on_minus_off_q025"] = float(np.quantile(difference_samples, 0.025))
    row["on_minus_off_q500"] = float(np.quantile(difference_samples, 0.500))
    row["on_minus_off_q975"] = float(np.quantile(difference_samples, 0.975))
    global_rows.append(row)

global_summary = pd.DataFrame(global_rows)


# %%
# =============================================================================
# Align the 30 delay posteriors by ABL and signed ILD
# =============================================================================
off_conditions = fit_payloads["off"]["condition_table"][
    ["condition_id", "ABL", "ILD"]
].rename(columns={"condition_id": "off_condition_id"})
on_conditions = fit_payloads["on_bilateral"]["condition_table"][
    ["condition_id", "ABL", "ILD"]
].rename(columns={"condition_id": "on_condition_id"})
aligned_conditions = off_conditions.merge(
    on_conditions,
    on=["ABL", "ILD"],
    how="inner",
    validate="one_to_one",
).sort_values(["ABL", "ILD"])
if len(aligned_conditions) != 30:
    raise RuntimeError("OFF and bilateral-ON fits do not share exactly 30 conditions.")

off_delays = fit_payloads["off"]["posterior"]["t_E_aff"] * 1000.0
on_delays = fit_payloads["on_bilateral"]["posterior"]["t_E_aff"] * 1000.0
delay_rows = []
for row in aligned_conditions.itertuples(index=False):
    off_samples = off_delays[:, int(row.off_condition_id)]
    on_samples = on_delays[:, int(row.on_condition_id)]
    n_pairs = min(len(off_samples), len(on_samples))
    off_indices = rng.permutation(len(off_samples))[:n_pairs]
    on_indices = rng.permutation(len(on_samples))[:n_pairs]
    difference_samples = on_samples[on_indices] - off_samples[off_indices]
    delay_rows.append(
        {
            "ABL": float(row.ABL),
            "ILD": float(row.ILD),
            "off_mean_ms": float(np.mean(off_samples)),
            "off_q025_ms": float(np.quantile(off_samples, 0.025)),
            "off_q500_ms": float(np.quantile(off_samples, 0.500)),
            "off_q975_ms": float(np.quantile(off_samples, 0.975)),
            "on_bilateral_mean_ms": float(np.mean(on_samples)),
            "on_bilateral_q025_ms": float(np.quantile(on_samples, 0.025)),
            "on_bilateral_q500_ms": float(np.quantile(on_samples, 0.500)),
            "on_bilateral_q975_ms": float(np.quantile(on_samples, 0.975)),
            "on_minus_off_mean_ms": float(np.mean(difference_samples)),
            "on_minus_off_q025_ms": float(
                np.quantile(difference_samples, 0.025)
            ),
            "on_minus_off_q500_ms": float(
                np.quantile(difference_samples, 0.500)
            ),
            "on_minus_off_q975_ms": float(
                np.quantile(difference_samples, 0.975)
            ),
        }
    )

delay_summary = pd.DataFrame(delay_rows)
delay_r = float(
    np.corrcoef(
        delay_summary["off_mean_ms"], delay_summary["on_bilateral_mean_ms"]
    )[0, 1]
)
delay_rmse_ms = float(
    np.sqrt(
        np.mean(
            (
                delay_summary["on_bilateral_mean_ms"]
                - delay_summary["off_mean_ms"]
            )
            ** 2
        )
    )
)
delay_mean_difference_ms = float(delay_summary["on_minus_off_mean_ms"].mean())


# %%
# =============================================================================
# 2 x 4 global-parameter and condition-delay comparison
# =============================================================================
figure_path = OUTPUT_DIR / "led7_super_animal_off_vs_on_npl_alpha_parameters.png"
fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.4))
axes_flat = axes.ravel()

for ax, parameter, summary_row in zip(
    axes_flat[:7], GLOBAL_PARAMETERS, global_summary.itertuples(index=False)
):
    means = np.array([summary_row.off_mean, summary_row.on_bilateral_mean])
    lows = np.array([summary_row.off_q025, summary_row.on_bilateral_q025])
    highs = np.array([summary_row.off_q975, summary_row.on_bilateral_q975])
    for x, category, mean, low, high in zip(
        [0, 1], ["off", "on_bilateral"], means, lows, highs
    ):
        ax.errorbar(
            x,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="o",
            markersize=6.0,
            markeredgewidth=0.9,
            color=MODEL_COLORS[category],
            ecolor=MODEL_COLORS[category],
            elinewidth=1.4,
            capsize=3.0,
            zorder=3,
        )
    interval_low = float(np.min(lows))
    interval_high = float(np.max(highs))
    padding = max(0.08 * (interval_high - interval_low), 1e-6)
    ax.set_ylim(interval_low - padding, interval_high + padding)
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1], ["OFF", "bilateral\nON"])
    title = parameter["label"]
    if parameter["unit"]:
        title += f" ({parameter['unit']})"
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.90", linewidth=0.7)

delay_ax = axes_flat[7]
for ABL in [20, 40, 60]:
    subset = delay_summary[delay_summary["ABL"] == ABL]
    x = subset["off_mean_ms"].to_numpy()
    y = subset["on_bilateral_mean_ms"].to_numpy()
    xerr = np.vstack(
        [x - subset["off_q025_ms"].to_numpy(), subset["off_q975_ms"].to_numpy() - x]
    )
    yerr = np.vstack(
        [
            y - subset["on_bilateral_q025_ms"].to_numpy(),
            subset["on_bilateral_q975_ms"].to_numpy() - y,
        ]
    )
    delay_ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        markersize=4.5,
        markeredgewidth=0.7,
        color=ABL_COLORS[ABL],
        ecolor=ABL_COLORS[ABL],
        elinewidth=0.65,
        capsize=0,
        alpha=0.68,
        label=f"ABL {ABL}",
    )

delay_min = float(
    min(delay_summary["off_q025_ms"].min(), delay_summary["on_bilateral_q025_ms"].min())
)
delay_max = float(
    max(delay_summary["off_q975_ms"].max(), delay_summary["on_bilateral_q975_ms"].max())
)
delay_padding = 0.04 * (delay_max - delay_min)
delay_limits = (delay_min - delay_padding, delay_max + delay_padding)
delay_ax.plot(delay_limits, delay_limits, color="0.25", linestyle="--", linewidth=1.0)
delay_ax.set_xlim(delay_limits)
delay_ax.set_ylim(delay_limits)
delay_ax.set_aspect("equal", adjustable="box")
delay_ax.set_xlabel(r"OFF $t_{E,aff}$ (ms)")
delay_ax.set_ylabel(r"bilateral ON $t_{E,aff}$ (ms)")
delay_ax.set_title("condition delays")
delay_ax.text(
    0.04,
    0.96,
    f"r = {delay_r:.3f}\nRMSE = {delay_rmse_ms:.1f} ms\n"
    f"mean delta = {delay_mean_difference_ms:+.1f} ms",
    transform=delay_ax.transAxes,
    ha="left",
    va="top",
    fontsize=8.0,
)
delay_ax.spines[["top", "right"]].set_visible(False)
delay_ax.legend(frameon=False, fontsize=7.5, loc="lower right")

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        color=MODEL_COLORS[category],
        markersize=6,
        label=MODEL_LABELS[category],
    )
    for category in ["off", "on_bilateral"]
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=2,
    frameon=False,
)
fig.suptitle(
    "LED7 super-animal NPL+alpha posterior: LED OFF versus bilateral LED ON",
    y=1.035,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(figure_path, dpi=220, bbox_inches="tight")


# %%
# =============================================================================
# Save reusable summaries and provenance
# =============================================================================
global_csv = OUTPUT_DIR / "led7_super_animal_off_vs_on_global_params.csv"
delay_csv = OUTPUT_DIR / "led7_super_animal_off_vs_on_condition_delays.csv"
payload_path = OUTPUT_DIR / "led7_super_animal_off_vs_on_npl_alpha_parameters.pkl"
global_summary.to_csv(global_csv, index=False)
delay_summary.to_csv(delay_csv, index=False)
with payload_path.open("wb") as handle:
    pickle.dump(
        {
            "schema_version": 1,
            "fit_root": str(FIT_ROOT.resolve()),
            "accepted_fit_roots": {
                category: str(payload["fit_dir"].resolve())
                for category, payload in fit_payloads.items()
            },
            "accepted_initialization_modes": {
                category: payload["selection"]["accepted_initialization_mode"]
                for category, payload in fit_payloads.items()
            },
            "global_summary": global_summary,
            "delay_summary": delay_summary,
            "delay_metrics": {
                "pearson_r": delay_r,
                "rmse_ms": delay_rmse_ms,
                "mean_on_minus_off_ms": delay_mean_difference_ms,
            },
        },
        handle,
    )

print(global_summary.to_string(index=False))
print(
    f"Delay posterior means: r={delay_r:.6f}, RMSE={delay_rmse_ms:.6f} ms, "
    f"mean ON-OFF={delay_mean_difference_ms:+.6f} ms"
)
print(f"Figure: {figure_path.resolve()}")
print(f"Global summary: {global_csv.resolve()}")
print(f"Delay summary: {delay_csv.resolve()}")
print(f"Payload: {payload_path.resolve()}")
