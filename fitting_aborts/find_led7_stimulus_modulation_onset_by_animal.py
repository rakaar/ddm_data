# %%
"""
Estimate the LED7 RT-stimulus modulation onset separately for each animal.

The paper method compares the strongest and weakest stimulus RT distributions
after discarding RTs above each candidate time. Here, |ILD|=16 and |ILD|=1 are
the strongest and weakest available LED7 conditions. The primary estimate adds
two safeguards against unstable tiny-sample onsets: at least 20 trials from
each strength and p < 0.05 for three consecutive 5 ms candidate times.
"""

# %%
# =============================================================================
# Editable parameters
# =============================================================================
from pathlib import Path
import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

BATCH_NAME = "LED7"
ANIMALS = (92, 93, 98, 99, 100, 103)
ABLS = (20, 40, 60)
ABS_ILDS = (1, 2, 4, 8, 16)
WEAKEST_ABS_ILD = 1
STRONGEST_ABS_ILD = 16

RT_MIN_S = 0.0
RT_MAX_S = 1.0
ONSET_SEARCH_MAX_S = 0.300
ONSET_STEP_S = 0.005
KS_ALPHA = 0.05

MIN_TRIALS_PER_STRENGTH = 20
N_CONSECUTIVE_SIGNIFICANT = 3
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 20260730

CDF_PLOT_MAX_S = 0.250
TIME_DELAY_MIN_CDF = 0.005
TIME_DELAY_MAX_CDF = 0.995
PLOT_DPI = 300

DATA_CSV = (
    REPO_ROOT
    / "raw_data"
    / "batch_csvs"
    / "batch_LED7_valid_and_aborts.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "led7_stimulus_modulation_onset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KS_GRID_PNG = OUTPUT_DIR / "led7_stimulus_modulation_onset_ks_by_animal.png"
DIAGNOSTIC_PNG = OUTPUT_DIR / "led7_stimulus_modulation_onset_diagnostic.png"
SUMMARY_CSV = OUTPUT_DIR / "led7_stimulus_modulation_onset_summary.csv"
KS_CURVES_CSV = OUTPUT_DIR / "led7_stimulus_modulation_onset_ks_curves.csv"
TIME_DELAY_CSV = OUTPUT_DIR / "led7_stimulus_modulation_time_delay_curves.csv"
EARLY_CHOICE_CSV = OUTPUT_DIR / "led7_stimulus_modulation_early_choice.csv"
OUTPUT_PKL = OUTPUT_DIR / "led7_stimulus_modulation_onset_payload.pkl"

PAPER_URL = "https://www.nature.com/articles/s41467-021-27302-8"
PAPER_CODE_URL = "https://osf.io/3qe59/"


# %%
# =============================================================================
# Imports and plotting defaults
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

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
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)

ABL_COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
ILD_COLORS = {
    1: "#000000",
    2: "#0072B2",
    4: "#009E73",
    8: "#E69F00",
    16: "#D55E00",
}


# %%
# =============================================================================
# Reused statistical operations
# =============================================================================
def sequential_ks_curve(weak_rts, strong_rts, candidate_times):
    """Paper-style one-tailed KS curve after retaining only RT <= candidate time."""
    weak_rts = np.asarray(weak_rts, dtype=float)
    strong_rts = np.asarray(strong_rts, dtype=float)

    rows = []
    for candidate_s in candidate_times:
        weak_before = weak_rts[weak_rts <= candidate_s]
        strong_before = strong_rts[strong_rts <= candidate_s]
        if len(weak_before) == 0 or len(strong_before) == 0:
            statistic = np.nan
            p_value = np.nan
        else:
            method = (
                "exact"
                if max(len(weak_before), len(strong_before)) <= 100
                else "asymp"
            )
            result = ks_2samp(
                weak_before,
                strong_before,
                alternative="less",
                method=method,
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        rows.append(
            {
                "candidate_s": float(candidate_s),
                "candidate_ms": float(candidate_s * 1000.0),
                "ks_statistic": statistic,
                "p_value": p_value,
                "n_weak": int(len(weak_before)),
                "n_strong": int(len(strong_before)),
            }
        )

    return pd.DataFrame(rows)


def find_ks_onset(curve_df, min_trials, consecutive_bins):
    """Return the first candidate starting a sustained significant run."""
    eligible = (
        curve_df["p_value"].lt(KS_ALPHA)
        & curve_df["n_weak"].ge(min_trials)
        & curve_df["n_strong"].ge(min_trials)
    ).to_numpy(dtype=bool)

    if consecutive_bins == 1:
        starts = np.flatnonzero(eligible)
    else:
        running_count = np.convolve(
            eligible.astype(int),
            np.ones(consecutive_bins, dtype=int),
            mode="valid",
        )
        starts = np.flatnonzero(running_count == consecutive_bins)

    if len(starts) == 0:
        return None
    return curve_df.iloc[int(starts[0])]


def empirical_cdf(samples, grid):
    samples = np.sort(np.asarray(samples, dtype=float))
    return np.searchsorted(samples, grid, side="right") / len(samples)


def empirical_time_delay(reference_rts, condition_rts, grid):
    """Horizontal CDF displacement Tr - T using the weak condition as reference."""
    reference_rts = np.sort(np.asarray(reference_rts, dtype=float))
    condition_cdf = empirical_cdf(condition_rts, grid)
    reference_times = np.quantile(
        reference_rts,
        np.clip(condition_cdf, 0.0, 1.0),
        method="linear",
    )
    delay_s = reference_times - grid
    valid = (
        (condition_cdf >= TIME_DELAY_MIN_CDF)
        & (condition_cdf <= TIME_DELAY_MAX_CDF)
    )
    return condition_cdf, np.where(valid, delay_s, np.nan)


# %%
# =============================================================================
# Load the exact valid-trial LED7 fitting pool
# =============================================================================
if not DATA_CSV.exists():
    raise FileNotFoundError(DATA_CSV)

raw_df = pd.read_csv(DATA_CSV)
required_columns = ["animal", "success", "RTwrtStim", "ABL", "ILD"]
missing_columns = [
    column for column in required_columns if column not in raw_df.columns
]
if missing_columns:
    raise KeyError(f"Missing columns in {DATA_CSV}: {missing_columns}")

valid_df = raw_df[
    raw_df["animal"].astype(int).isin(ANIMALS)
    & raw_df["success"].isin([1, -1])
    & raw_df["RTwrtStim"].ge(RT_MIN_S)
    & raw_df["RTwrtStim"].lt(RT_MAX_S)
    & raw_df["ABL"].isin(ABLS)
    & raw_df["ILD"].abs().isin(ABS_ILDS)
].copy()
valid_df = valid_df.dropna(subset=required_columns)
valid_df["animal"] = valid_df["animal"].astype(int)
valid_df["ABL"] = valid_df["ABL"].astype(int)
valid_df["ILD"] = valid_df["ILD"].astype(float)
valid_df["abs_ILD"] = valid_df["ILD"].abs().astype(int)
valid_df["correct"] = valid_df["success"].eq(1).astype(int)
valid_df["log2_abs_ILD"] = np.log2(valid_df["abs_ILD"].astype(float))

observed_animals = tuple(sorted(valid_df["animal"].unique()))
if observed_animals != tuple(sorted(ANIMALS)):
    raise RuntimeError(f"Expected animals {ANIMALS}, found {observed_animals}.")

observed_abs_ilds = tuple(sorted(valid_df["abs_ILD"].unique()))
if observed_abs_ilds != tuple(sorted(ABS_ILDS)):
    raise RuntimeError(
        f"Expected absolute ILDs {ABS_ILDS}, found {observed_abs_ilds}."
    )

candidate_times_s = np.arange(
    RT_MIN_S,
    ONSET_SEARCH_MAX_S + 0.5 * ONSET_STEP_S,
    ONSET_STEP_S,
)

print(f"Data CSV: {DATA_CSV}")
print(f"Valid LED7 rows in [0, 1) s: {len(valid_df):,}")
print(f"Animals: {ANIMALS}")
print(
    "Cutoff adaptation: compare |ILD|=1 versus |ILD|=16 with a one-tailed "
    "KS test on RTs <= each candidate time."
)
print(
    "Primary robustness rule: "
    f">={MIN_TRIALS_PER_STRENGTH} trials/strength and "
    f"{N_CONSECUTIVE_SIGNIFICANT} consecutive p<{KS_ALPHA:g} bins."
)


# %%
# =============================================================================
# Animal-wise pooled and ABL-specific onset estimates
# =============================================================================
summary_rows = []
ks_curve_rows = []
bootstrap_onsets_by_animal = {}
rng = np.random.default_rng(BOOTSTRAP_SEED)

for animal in ANIMALS:
    animal_df = valid_df[valid_df["animal"].eq(animal)].copy()
    weak_rts = animal_df.loc[
        animal_df["abs_ILD"].eq(WEAKEST_ABS_ILD),
        "RTwrtStim",
    ].to_numpy(dtype=float)
    strong_rts = animal_df.loc[
        animal_df["abs_ILD"].eq(STRONGEST_ABS_ILD),
        "RTwrtStim",
    ].to_numpy(dtype=float)

    pooled_curve = sequential_ks_curve(
        weak_rts,
        strong_rts,
        candidate_times_s,
    )
    pooled_curve["animal"] = animal
    pooled_curve["scope"] = "pooled_ABL"
    pooled_curve["ABL"] = np.nan
    ks_curve_rows.append(pooled_curve)

    raw_onset = find_ks_onset(
        pooled_curve,
        min_trials=1,
        consecutive_bins=1,
    )
    robust_onset = find_ks_onset(
        pooled_curve,
        min_trials=MIN_TRIALS_PER_STRENGTH,
        consecutive_bins=N_CONSECUTIVE_SIGNIFICANT,
    )

    abl_onsets_ms = {}
    for abl in ABLS:
        abl_df = animal_df[animal_df["ABL"].eq(abl)]
        abl_weak_rts = abl_df.loc[
            abl_df["abs_ILD"].eq(WEAKEST_ABS_ILD),
            "RTwrtStim",
        ].to_numpy(dtype=float)
        abl_strong_rts = abl_df.loc[
            abl_df["abs_ILD"].eq(STRONGEST_ABS_ILD),
            "RTwrtStim",
        ].to_numpy(dtype=float)
        abl_curve = sequential_ks_curve(
            abl_weak_rts,
            abl_strong_rts,
            candidate_times_s,
        )
        abl_curve["animal"] = animal
        abl_curve["scope"] = f"ABL_{abl}"
        abl_curve["ABL"] = abl
        ks_curve_rows.append(abl_curve)

        abl_onset = find_ks_onset(
            abl_curve,
            min_trials=MIN_TRIALS_PER_STRENGTH,
            consecutive_bins=N_CONSECUTIVE_SIGNIFICANT,
        )
        abl_onsets_ms[abl] = (
            float(abl_onset["candidate_ms"])
            if abl_onset is not None
            else np.nan
        )

    bootstrap_onsets = []
    bootstrap_strata = {}
    for abl in ABLS:
        for abs_ild in (WEAKEST_ABS_ILD, STRONGEST_ABS_ILD):
            values = animal_df.loc[
                animal_df["ABL"].eq(abl)
                & animal_df["abs_ILD"].eq(abs_ild),
                "RTwrtStim",
            ].to_numpy(dtype=float)
            bootstrap_strata[(abl, abs_ild)] = values

    for _ in range(N_BOOTSTRAP):
        bootstrap_weak = []
        bootstrap_strong = []
        for abl in ABLS:
            weak_values = bootstrap_strata[(abl, WEAKEST_ABS_ILD)]
            strong_values = bootstrap_strata[(abl, STRONGEST_ABS_ILD)]
            bootstrap_weak.append(
                rng.choice(weak_values, size=len(weak_values), replace=True)
            )
            bootstrap_strong.append(
                rng.choice(
                    strong_values,
                    size=len(strong_values),
                    replace=True,
                )
            )

        bootstrap_curve = sequential_ks_curve(
            np.concatenate(bootstrap_weak),
            np.concatenate(bootstrap_strong),
            candidate_times_s,
        )
        bootstrap_onset = find_ks_onset(
            bootstrap_curve,
            min_trials=MIN_TRIALS_PER_STRENGTH,
            consecutive_bins=N_CONSECUTIVE_SIGNIFICANT,
        )
        bootstrap_onsets.append(
            float(bootstrap_onset["candidate_ms"])
            if bootstrap_onset is not None
            else np.nan
        )

    bootstrap_onsets = np.asarray(bootstrap_onsets, dtype=float)
    bootstrap_onsets_by_animal[animal] = bootstrap_onsets
    finite_bootstrap = bootstrap_onsets[np.isfinite(bootstrap_onsets)]

    if robust_onset is None:
        robust_cutoff_s = np.nan
        express_df = animal_df.iloc[0:0].copy()
    else:
        robust_cutoff_s = float(robust_onset["candidate_s"])
        express_df = animal_df[
            animal_df["RTwrtStim"].lt(robust_cutoff_s)
        ].copy()

    if len(express_df) > 0 and express_df["correct"].nunique() == 2:
        early_choice_fit = smf.glm(
            "correct ~ log2_abs_ILD + C(ABL)",
            data=express_df,
            family=sm.families.Binomial(),
        ).fit()
        choice_strength_coef = float(
            early_choice_fit.params["log2_abs_ILD"]
        )
        choice_strength_p = float(
            early_choice_fit.pvalues["log2_abs_ILD"]
        )
    else:
        choice_strength_coef = np.nan
        choice_strength_p = np.nan

    summary_rows.append(
        {
            "batch_name": BATCH_NAME,
            "animal": animal,
            "n_valid_trials": len(animal_df),
            "n_weak_total": len(weak_rts),
            "n_strong_total": len(strong_rts),
            "paper_style_first_significant_ms": (
                float(raw_onset["candidate_ms"])
                if raw_onset is not None
                else np.nan
            ),
            "paper_style_n_weak": (
                int(raw_onset["n_weak"])
                if raw_onset is not None
                else 0
            ),
            "paper_style_n_strong": (
                int(raw_onset["n_strong"])
                if raw_onset is not None
                else 0
            ),
            "robust_cutoff_ms": (
                float(robust_onset["candidate_ms"])
                if robust_onset is not None
                else np.nan
            ),
            "robust_cutoff_p": (
                float(robust_onset["p_value"])
                if robust_onset is not None
                else np.nan
            ),
            "robust_n_weak": (
                int(robust_onset["n_weak"])
                if robust_onset is not None
                else 0
            ),
            "robust_n_strong": (
                int(robust_onset["n_strong"])
                if robust_onset is not None
                else 0
            ),
            "bootstrap_q025_ms": (
                float(np.quantile(finite_bootstrap, 0.025))
                if len(finite_bootstrap)
                else np.nan
            ),
            "bootstrap_median_ms": (
                float(np.quantile(finite_bootstrap, 0.5))
                if len(finite_bootstrap)
                else np.nan
            ),
            "bootstrap_q975_ms": (
                float(np.quantile(finite_bootstrap, 0.975))
                if len(finite_bootstrap)
                else np.nan
            ),
            "bootstrap_finite_fraction": (
                float(len(finite_bootstrap) / N_BOOTSTRAP)
            ),
            "ABL20_cutoff_ms": abl_onsets_ms[20],
            "ABL40_cutoff_ms": abl_onsets_ms[40],
            "ABL60_cutoff_ms": abl_onsets_ms[60],
            "n_express_before_robust_cutoff": len(express_df),
            "express_fraction": (
                float(len(express_df) / len(animal_df))
                if len(animal_df)
                else np.nan
            ),
            "early_choice_log2_abs_ILD_coef": choice_strength_coef,
            "early_choice_log2_abs_ILD_p": choice_strength_p,
        }
    )

summary_df = pd.DataFrame(summary_rows)
ks_curves_df = pd.concat(ks_curve_rows, ignore_index=True)

if summary_df["robust_cutoff_ms"].isna().any():
    missing_animals = summary_df.loc[
        summary_df["robust_cutoff_ms"].isna(),
        "animal",
    ].tolist()
    raise RuntimeError(
        f"No robust modulation onset found for animals {missing_animals}."
    )


# %%
# =============================================================================
# Time-delay curves and early-choice summaries
# =============================================================================
time_delay_rows = []
early_choice_rows = []

for animal in ANIMALS:
    animal_df = valid_df[valid_df["animal"].eq(animal)]
    cutoff_s = float(
        summary_df.loc[
            summary_df["animal"].eq(animal),
            "robust_cutoff_ms",
        ].iloc[0]
        / 1000.0
    )
    reference_rts = animal_df.loc[
        animal_df["abs_ILD"].eq(WEAKEST_ABS_ILD),
        "RTwrtStim",
    ].to_numpy(dtype=float)

    for abs_ild in ABS_ILDS:
        condition_rts = animal_df.loc[
            animal_df["abs_ILD"].eq(abs_ild),
            "RTwrtStim",
        ].to_numpy(dtype=float)
        condition_cdf, delay_s = empirical_time_delay(
            reference_rts,
            condition_rts,
            candidate_times_s,
        )
        for candidate_s, cdf_value, current_delay_s in zip(
            candidate_times_s,
            condition_cdf,
            delay_s,
        ):
            time_delay_rows.append(
                {
                    "batch_name": BATCH_NAME,
                    "animal": animal,
                    "abs_ILD": abs_ild,
                    "candidate_s": candidate_s,
                    "candidate_ms": candidate_s * 1000.0,
                    "condition_cdf": cdf_value,
                    "time_delay_s": current_delay_s,
                    "time_delay_ms": current_delay_s * 1000.0,
                }
            )

        early_condition = animal_df[
            animal_df["abs_ILD"].eq(abs_ild)
            & animal_df["RTwrtStim"].lt(cutoff_s)
        ]
        n_early = len(early_condition)
        n_correct = int(early_condition["correct"].sum())
        if n_early:
            ci_low, ci_high = proportion_confint(
                n_correct,
                n_early,
                alpha=0.05,
                method="wilson",
            )
            accuracy = n_correct / n_early
        else:
            ci_low = np.nan
            ci_high = np.nan
            accuracy = np.nan

        early_choice_rows.append(
            {
                "batch_name": BATCH_NAME,
                "animal": animal,
                "cutoff_ms": cutoff_s * 1000.0,
                "abs_ILD": abs_ild,
                "n_early": n_early,
                "n_correct": n_correct,
                "accuracy": accuracy,
                "accuracy_q025": ci_low,
                "accuracy_q975": ci_high,
            }
        )

time_delay_df = pd.DataFrame(time_delay_rows)
early_choice_df = pd.DataFrame(early_choice_rows)


# %%
# =============================================================================
# Compact 3 x 2 cutoff figure
# =============================================================================
fig_ks, axes_ks = plt.subplots(
    3,
    2,
    figsize=(8.0, 8.6),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for ax, animal in zip(axes_ks.flat, ANIMALS):
    animal_curves = ks_curves_df[ks_curves_df["animal"].eq(animal)]
    pooled_curve = animal_curves[
        animal_curves["scope"].eq("pooled_ABL")
    ]
    ax.plot(
        pooled_curve["candidate_ms"],
        np.clip(pooled_curve["p_value"], 1e-8, 1.0),
        color="black",
        lw=1.5,
        label="Pooled ABL",
    )
    for abl in ABLS:
        abl_curve = animal_curves[
            animal_curves["scope"].eq(f"ABL_{abl}")
        ]
        ax.plot(
            abl_curve["candidate_ms"],
            np.clip(abl_curve["p_value"], 1e-8, 1.0),
            color=ABL_COLORS[abl],
            lw=1.0,
            alpha=0.75,
            label=f"ABL {abl}",
        )

    animal_summary = summary_df[
        summary_df["animal"].eq(animal)
    ].iloc[0]
    cutoff_ms = float(animal_summary["robust_cutoff_ms"])
    ax.axhline(KS_ALPHA, color="0.4", ls=":", lw=1.0)
    ax.axvline(cutoff_ms, color="#D55E00", ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlim(0, CDF_PLOT_MAX_S * 1000.0)
    ax.set_ylim(1e-8, 1.05)
    ax.set_title(
        f"LED7/{animal}: cutoff {cutoff_ms:.0f} ms "
        f"[{animal_summary['bootstrap_q025_ms']:.0f}, "
        f"{animal_summary['bootstrap_q975_ms']:.0f}]"
    )
    ax.set_xlabel("Candidate cutoff (ms)")
    ax.set_ylabel("One-tailed KS p")

axes_ks.flat[0].legend(loc="lower left", frameon=False, ncol=2)
fig_ks.savefig(KS_GRID_PNG, dpi=PLOT_DPI, bbox_inches="tight")


# %%
# =============================================================================
# Figure 2-style diagnostic: CDF, time delay, cutoff test, and early choice
# =============================================================================
fig, axes = plt.subplots(
    len(ANIMALS),
    4,
    figsize=(14.0, 18.0),
    constrained_layout=True,
)

for row_idx, animal in enumerate(ANIMALS):
    animal_df = valid_df[valid_df["animal"].eq(animal)]
    animal_summary = summary_df[
        summary_df["animal"].eq(animal)
    ].iloc[0]
    cutoff_ms = float(animal_summary["robust_cutoff_ms"])
    cutoff_s = cutoff_ms / 1000.0

    ax_cdf, ax_delay, ax_ks, ax_choice = axes[row_idx]

    cdf_grid = candidate_times_s
    for abs_ild in ABS_ILDS:
        condition_rts = animal_df.loc[
            animal_df["abs_ILD"].eq(abs_ild),
            "RTwrtStim",
        ].to_numpy(dtype=float)
        ax_cdf.plot(
            cdf_grid * 1000.0,
            empirical_cdf(condition_rts, cdf_grid),
            color=ILD_COLORS[abs_ild],
            lw=1.2,
            label=rf"$|ILD|={abs_ild}$",
        )
    ax_cdf.axvline(cutoff_ms, color="#D55E00", ls="--", lw=1.0)
    ax_cdf.set_xlim(0, CDF_PLOT_MAX_S * 1000.0)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlabel("RT (ms)")
    ax_cdf.set_ylabel("RT CDF")
    ax_cdf.set_title(f"LED7/{animal}: RT distributions")

    animal_delay = time_delay_df[
        time_delay_df["animal"].eq(animal)
    ]
    for abs_ild in ABS_ILDS:
        current_delay = animal_delay[
            animal_delay["abs_ILD"].eq(abs_ild)
        ]
        ax_delay.plot(
            current_delay["candidate_ms"],
            current_delay["time_delay_ms"],
            color=ILD_COLORS[abs_ild],
            lw=1.2,
        )
    ax_delay.axhline(0, color="0.5", ls=":", lw=0.8)
    ax_delay.axvline(cutoff_ms, color="#D55E00", ls="--", lw=1.0)
    ax_delay.set_xlim(0, CDF_PLOT_MAX_S * 1000.0)
    ax_delay.set_xlabel("RT (ms)")
    ax_delay.set_ylabel(r"Time delay $\Delta t$ (ms)")
    ax_delay.set_title(r"Horizontal CDF delay vs $|ILD|=1$")

    animal_curves = ks_curves_df[ks_curves_df["animal"].eq(animal)]
    pooled_curve = animal_curves[
        animal_curves["scope"].eq("pooled_ABL")
    ]
    ax_ks.plot(
        pooled_curve["candidate_ms"],
        np.clip(pooled_curve["p_value"], 1e-8, 1.0),
        color="black",
        lw=1.5,
        label="Pooled ABL",
    )
    for abl in ABLS:
        abl_curve = animal_curves[
            animal_curves["scope"].eq(f"ABL_{abl}")
        ]
        ax_ks.plot(
            abl_curve["candidate_ms"],
            np.clip(abl_curve["p_value"], 1e-8, 1.0),
            color=ABL_COLORS[abl],
            lw=1.0,
            alpha=0.75,
            label=f"ABL {abl}",
        )
    ax_ks.axhline(KS_ALPHA, color="0.4", ls=":", lw=1.0)
    ax_ks.axvline(cutoff_ms, color="#D55E00", ls="--", lw=1.0)
    ax_ks.set_yscale("log")
    ax_ks.set_xlim(0, CDF_PLOT_MAX_S * 1000.0)
    ax_ks.set_ylim(1e-8, 1.05)
    ax_ks.set_xlabel("Candidate cutoff (ms)")
    ax_ks.set_ylabel("One-tailed KS p")
    ax_ks.set_title(
        f"Robust cutoff {cutoff_ms:.0f} ms; "
        f"raw first {animal_summary['paper_style_first_significant_ms']:.0f} ms"
    )

    animal_choice = early_choice_df[
        early_choice_df["animal"].eq(animal)
    ].sort_values("abs_ILD")
    yerr = np.vstack(
        [
            animal_choice["accuracy"]
            - animal_choice["accuracy_q025"],
            animal_choice["accuracy_q975"]
            - animal_choice["accuracy"],
        ]
    )
    ax_choice.errorbar(
        animal_choice["abs_ILD"],
        animal_choice["accuracy"],
        yerr=yerr,
        color="black",
        marker="o",
        ms=4,
        lw=1.0,
        capsize=2,
    )
    ax_choice.axhline(0.5, color="0.5", ls=":", lw=0.8)
    ax_choice.set_xscale("log", base=2)
    ax_choice.set_xticks(ABS_ILDS)
    ax_choice.set_xticklabels([str(value) for value in ABS_ILDS])
    ax_choice.set_ylim(0.35, 1.0)
    ax_choice.set_xlabel(r"$|ILD|$")
    ax_choice.set_ylabel("Accuracy before cutoff")
    ax_choice.set_title(
        f"Early choice strength p="
        f"{animal_summary['early_choice_log2_abs_ILD_p']:.2g}; "
        f"n={int(animal_summary['n_express_before_robust_cutoff'])}"
    )

axes[0, 0].legend(loc="lower right", frameon=False, ncol=2)
axes[0, 2].legend(loc="lower left", frameon=False, ncol=2)
fig.savefig(DIAGNOSTIC_PNG, dpi=PLOT_DPI, bbox_inches="tight")


# %%
# =============================================================================
# Save reusable tables and payload
# =============================================================================
summary_df.to_csv(SUMMARY_CSV, index=False)
ks_curves_df.to_csv(KS_CURVES_CSV, index=False)
time_delay_df.to_csv(TIME_DELAY_CSV, index=False)
early_choice_df.to_csv(EARLY_CHOICE_CSV, index=False)

payload = {
    "batch_name": BATCH_NAME,
    "animals": ANIMALS,
    "abls": ABLS,
    "abs_ilds": ABS_ILDS,
    "weakest_abs_ild": WEAKEST_ABS_ILD,
    "strongest_abs_ild": STRONGEST_ABS_ILD,
    "candidate_times_s": candidate_times_s,
    "ks_alpha": KS_ALPHA,
    "min_trials_per_strength": MIN_TRIALS_PER_STRENGTH,
    "n_consecutive_significant": N_CONSECUTIVE_SIGNIFICANT,
    "n_bootstrap": N_BOOTSTRAP,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "summary": summary_df,
    "ks_curves": ks_curves_df,
    "time_delay_curves": time_delay_df,
    "early_choice": early_choice_df,
    "bootstrap_onsets_ms_by_animal": bootstrap_onsets_by_animal,
    "data_csv": str(DATA_CSV),
    "paper_url": PAPER_URL,
    "paper_code_url": PAPER_CODE_URL,
    "interpretation": (
        "The cutoff marks the first detected stimulus-strength modulation of "
        "RT. RTs before it are candidate express/proactive responses, but "
        "proactive responses can also occur after the cutoff."
    ),
}
with OUTPUT_PKL.open("wb") as handle:
    pickle.dump(payload, handle)

print("\nAnimal-wise LED7 stimulus-modulation onset estimates:")
print(
    summary_df[
        [
            "animal",
            "paper_style_first_significant_ms",
            "paper_style_n_weak",
            "paper_style_n_strong",
            "robust_cutoff_ms",
            "bootstrap_q025_ms",
            "bootstrap_q975_ms",
            "ABL20_cutoff_ms",
            "ABL40_cutoff_ms",
            "ABL60_cutoff_ms",
            "express_fraction",
            "early_choice_log2_abs_ILD_p",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.4g}")
)
print(f"\nSaved: {KS_GRID_PNG}")
print(f"Saved: {DIAGNOSTIC_PNG}")
print(f"Saved: {SUMMARY_CSV}")
print(f"Saved: {OUTPUT_PKL}")
