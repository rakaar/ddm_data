# %%
"""
LED7 condition-delay summaries for both completed RT-only NPL+alpha fits.

Each animal contributes its posterior-mean t_E_aff for every signed ABL/ILD
condition. Error bars in the figures are SEM across the six animals.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIT_ROOT = REPO_ROOT / "fit_animal_by_animal"

BATCH_NAME = "LED7"
ANIMALS = (92, 93, 98, 99, 100, 103)
ABLS = (20, 40, 60)
SIGNED_ILDS = (-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0)
PLOT_DPI = 300

OUTPUT_DIR = SCRIPT_DIR / "led7_npl_alpha_rt_only_fit_aligned_rtds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_CONFIGS = (
    {
        "mode": "proactive_reactive",
        "label": "proactive + reactive",
        "root": FIT_ROOT
        / (
            "numpyro_svi_npl_alpha_rt_only_proactive_reactive_0_to_1s_"
            "condition_delay_patience12_min50k_restore_best_outputs"
        ),
        "figure": OUTPUT_DIR
        / "led7_npl_alpha_rt_only_proactive_reactive_t_E_aff_vs_ild_by_abl.png",
    },
    {
        "mode": "reactive_only",
        "label": "reactive only",
        "root": FIT_ROOT
        / (
            "numpyro_svi_npl_alpha_rt_only_reactive_only_0_to_1s_"
            "condition_delay_patience12_min50k_restore_best_outputs"
        ),
        "figure": OUTPUT_DIR
        / "led7_npl_alpha_rt_only_reactive_only_t_E_aff_vs_ild_by_abl.png",
    },
)

ANIMAL_DELAY_CSV = OUTPUT_DIR / "led7_npl_alpha_rt_only_t_E_aff_by_animal.csv"
SUMMARY_CSV = OUTPUT_DIR / "led7_npl_alpha_rt_only_t_E_aff_summary.csv"


# %%
# =============================================================================
# Load posterior delay means and intervals
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "Nimbus Sans",
            "Helvetica Neue",
            "Arial",
            "Liberation Sans",
            "sans-serif",
        ],
    }
)

expected_conditions = pd.MultiIndex.from_product(
    [ABLS, SIGNED_ILDS],
    names=["ABL", "ILD"],
).to_frame(index=False)

delay_rows = []
for fit_config in FIT_CONFIGS:
    fit_root = fit_config["root"]
    ledger_csv = fit_root / "_batch_logs" / "batch_run_status.csv"
    if not ledger_csv.exists():
        raise FileNotFoundError(ledger_csv)
    ledger_df = pd.read_csv(ledger_csv).sort_values("run_index")
    if tuple(ledger_df["animal"].astype(int)) != ANIMALS:
        raise RuntimeError(f"Unexpected animal order in {ledger_csv}.")
    if not ledger_df["status"].eq("completed").all():
        raise RuntimeError(f"Not all fits are completed in {fit_root}.")

    print(f"\n{fit_config['label']} fit root: {fit_root}")
    for animal in ANIMALS:
        fit_dir = fit_root / f"{BATCH_NAME}_{animal}"
        posterior_path = fit_dir / "main_fullrank_posterior_samples.npz"
        condition_path = fit_dir / "condition_table.csv"
        finite_path = fit_dir / "main_fullrank_posterior_finite_report.csv"
        for required_path in (posterior_path, condition_path, finite_path):
            if not required_path.exists():
                raise FileNotFoundError(required_path)

        condition_df = (
            pd.read_csv(condition_path)
            .sort_values("condition_id")
            .reset_index(drop=True)
        )
        if len(condition_df) != len(expected_conditions) or not np.allclose(
            condition_df[["ABL", "ILD"]].to_numpy(dtype=float),
            expected_conditions[["ABL", "ILD"]].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0,
        ):
            raise RuntimeError(f"Unexpected condition table for LED7/{animal}.")

        finite_df = pd.read_csv(finite_path)
        if not (
            finite_df["n_total"].to_numpy(dtype=int)
            == finite_df["n_finite"].to_numpy(dtype=int)
        ).all():
            raise RuntimeError(f"Non-finite posterior report for LED7/{animal}.")

        with np.load(posterior_path) as posterior:
            if "t_E_aff" not in posterior.files:
                raise KeyError(f"Missing t_E_aff in {posterior_path}.")
            delay_samples_s = np.asarray(posterior["t_E_aff"], dtype=float)

        if (
            delay_samples_s.ndim != 2
            or delay_samples_s.shape[1] != len(condition_df)
            or not np.isfinite(delay_samples_s).all()
        ):
            raise RuntimeError(
                f"Invalid t_E_aff posterior for LED7/{animal}: "
                f"{delay_samples_s.shape}."
            )

        delay_mean_ms = np.mean(delay_samples_s, axis=0) * 1e3
        delay_ci_ms = np.quantile(
            delay_samples_s,
            [0.025, 0.975],
            axis=0,
        ) * 1e3
        for condition, mean_ms, ci_low_ms, ci_high_ms in zip(
            condition_df.itertuples(index=False),
            delay_mean_ms,
            delay_ci_ms[0],
            delay_ci_ms[1],
        ):
            delay_rows.append(
                {
                    "process_mode": fit_config["mode"],
                    "animal": animal,
                    "ABL": int(condition.ABL),
                    "ILD": float(condition.ILD),
                    "posterior_mean_ms": float(mean_ms),
                    "posterior_ci_low_ms": float(ci_low_ms),
                    "posterior_ci_high_ms": float(ci_high_ms),
                }
            )

        print(
            f"  LED7/{animal}: {delay_samples_s.shape[0]:,} samples; "
            f"mean range {delay_mean_ms.min():.1f}-"
            f"{delay_mean_ms.max():.1f} ms"
        )

delay_df = pd.DataFrame(delay_rows)
delay_df.to_csv(ANIMAL_DELAY_CSV, index=False)

delay_summary_df = (
    delay_df.groupby(["process_mode", "ABL", "ILD"], as_index=False)
    .agg(
        mean_ms=("posterior_mean_ms", "mean"),
        std_ms=("posterior_mean_ms", "std"),
        n_animals=("animal", "nunique"),
    )
    .sort_values(["process_mode", "ABL", "ILD"])
    .reset_index(drop=True)
)
delay_summary_df["sem_ms"] = (
    delay_summary_df["std_ms"]
    / np.sqrt(delay_summary_df["n_animals"])
)

expected_summary_rows = len(FIT_CONFIGS) * len(ABLS) * len(SIGNED_ILDS)
if len(delay_summary_df) != expected_summary_rows:
    raise RuntimeError(
        f"Expected {expected_summary_rows} summary rows, "
        f"found {len(delay_summary_df)}."
    )
if not np.all(delay_summary_df["n_animals"].to_numpy() == len(ANIMALS)):
    raise RuntimeError("Every delay point must contain all six LED7 animals.")

delay_summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"\nSaved animal posterior means and intervals: {ANIMAL_DELAY_CSV}")
print(f"Saved across-animal summaries: {SUMMARY_CSV}")


# %%
# =============================================================================
# Plot matched across-animal delay summaries
# =============================================================================
abl_colors = {
    20: "#1f77b4",
    40: "#ff7f0e",
    60: "#2ca02c",
}

for fit_config in FIT_CONFIGS:
    mode_summary = delay_summary_df[
        delay_summary_df["process_mode"].eq(fit_config["mode"])
    ]
    mode_lows = mode_summary["mean_ms"] - mode_summary["sem_ms"]
    mode_highs = mode_summary["mean_ms"] + mode_summary["sem_ms"]
    mode_y_low = float(mode_lows.min())
    mode_y_high = float(mode_highs.max())
    mode_y_pad = max(1.0, 0.06 * (mode_y_high - mode_y_low))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for abl in ABLS:
        sub = mode_summary[mode_summary["ABL"].eq(abl)]
        ax.errorbar(
            sub["ILD"],
            sub["mean_ms"],
            yerr=sub["sem_ms"],
            fmt="o-",
            color=abl_colors[abl],
            ecolor=abl_colors[abl],
            linewidth=1.3,
            elinewidth=1.1,
            capsize=3,
            markersize=5,
            label=f"ABL {abl}",
        )

    ax.axvline(0, color="0.82", linewidth=1, zorder=0)
    ax.set_xticks(SIGNED_ILDS)
    ax.set_xticklabels(
        [f"{ild:g}" for ild in SIGNED_ILDS],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(mode_y_low - mode_y_pad, mode_y_high + mode_y_pad)
    ax.set_xlabel("ILD (dB)")
    ax.set_ylabel(r"$t_{E,\mathrm{aff}}$ (ms)")
    ax.set_title(
        f"LED7 RT-only NPL+alpha SVI condition delays: "
        f"{fit_config['label']}"
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(fit_config["figure"], dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved figure: {fit_config['figure']}")

# %%
