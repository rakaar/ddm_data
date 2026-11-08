# %%
"""Compare raw and rolling LED7 fixation-abort fractions across the s7→s8 switch."""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR.parent / "raw_data"
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
WINDOWS = (1, 3, 5)  # One-session stride; 1 is the unsmoothed reference.
POST_ONSET_SPLIT_S = 0.100
LOW_TRIAL_THRESHOLD = 100
Y_LIMITS = ((0, 0.45), (0, 0.15), (0, 0.25))
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
TYPE_STYLES = {7: "-", 8: "--"}
FIGURE_DPI = 220
OUTPUT_STEM = "led7_s7_s8_fixation_abort_transition"
ANIMAL_AUDIT_CSV = SCRIPT_DIR / f"{OUTPUT_STEM}_animal_windows_audit.csv"
MEAN_AUDIT_CSV = SCRIPT_DIR / f"{OUTPUT_STEM}_equal_animal_mean_sem_audit.csv"

EXPECTED_SOURCE_ROWS = {7: 138_440, 8: 148_428}
EXPECTED_SIX_ANIMAL_ROWS = {7: 100_251, 8: 113_941}
EXPECTED_LED_COUNTS = {
    7: {0: 66_226, 1: 34_025},
    8: {0: 70_892, 1: 43_049},
}
EXPECTED_EVENT3_COUNTS = {7: 16_389, 8: 17_725}
EXPECTED_MISSING_ABORT_TIMES = {7: 2, 8: 2}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


# %%
############ Rebuild the session-level numerator and denominator audit ############
source_audits = []
for session_type in SESSION_TYPES:
    source_path = RAW_DATA_DIR / f"LED7_s{session_type}.csv"
    source = pd.read_csv(source_path, float_precision="round_trip")
    assert len(source) == EXPECTED_SOURCE_ROWS[session_type]
    assert len(source.columns) == 52
    assert source.columns.tolist().count("LED_onset_time") == 1
    assert source["training_level"].eq(16).all()
    assert source["session_type"].eq(session_type).all()
    assert (source["repeat_trial"].isin([0, 2]) | source["repeat_trial"].isna()).all()
    assert source["LED_trial"].isin([0, 1]).all()
    assert not source.duplicated(["animal", "session", "trial"]).any()

    data = source.loc[source["animal"].isin(ANIMALS)].copy()
    assert len(data) == EXPECTED_SIX_ANIMAL_ROWS[session_type]
    assert tuple(sorted(data["animal"].astype(int).unique())) == ANIMALS
    assert data["LED_trial"].value_counts().to_dict() == EXPECTED_LED_COUNTS[session_type]
    assert np.isfinite(data[["animal", "session", "trial", "intended_fix", "LED_onset_time"]]).all().all()
    assert data["LED_onset_time"].ge(0).all()
    assert data["LED_onset_time"].le(data["intended_fix"]).all()

    event3 = data["abort_event"].eq(3)
    finite_abort_time = np.isfinite(data["timed_fix"])
    assert data.loc[event3 & finite_abort_time, "timed_fix"].lt(
        data.loc[event3 & finite_abort_time, "intended_fix"]
    ).all()

    # The standardized CSV's sole onset column is already fixation-referenced.
    # For LED OFF it is the scheduled/counterfactual onset.
    abort_from_onset = data["timed_fix"] - data["LED_onset_time"]
    data["event3_total"] = event3
    data["event3_pre_onset"] = event3 & finite_abort_time & abort_from_onset.lt(0)
    data["event3_0_to_100ms"] = (
        event3 & finite_abort_time
        & abort_from_onset.ge(0) & abort_from_onset.lt(POST_ONSET_SPLIT_S)
    )
    data["event3_100ms_plus"] = event3 & finite_abort_time & abort_from_onset.ge(POST_ONSET_SPLIT_S)
    data["event3_missing_time"] = event3 & ~finite_abort_time
    assert data[
        ["event3_pre_onset", "event3_0_to_100ms", "event3_100ms_plus", "event3_missing_time"]
    ].sum(axis=1).eq(data["event3_total"]).all()

    session_audit = (
        data.groupby(["animal", "session", "LED_trial"], sort=True)
        .agg(
            all_condition_trials=("trial", "size"),
            event3_total=("event3_total", "sum"),
            event3_pre_onset=("event3_pre_onset", "sum"),
            event3_0_to_100ms=("event3_0_to_100ms", "sum"),
            event3_100ms_plus=("event3_100ms_plus", "sum"),
            event3_missing_time=("event3_missing_time", "sum"),
        )
        .reset_index()
    )
    for column in ("animal", "session", "LED_trial"):
        session_audit[column] = session_audit[column].astype(int)
    session_audit.insert(0, "session_type", session_type)
    assert int(session_audit["all_condition_trials"].sum()) == len(data)
    assert int(session_audit["event3_total"].sum()) == EXPECTED_EVENT3_COUNTS[session_type]
    assert int(session_audit["event3_missing_time"].sum()) == EXPECTED_MISSING_ABORT_TIMES[session_type]
    assert session_audit.groupby("LED_trial")["all_condition_trials"].sum().to_dict() == EXPECTED_LED_COUNTS[session_type]
    source_audits.append(session_audit)
    print(f"Read {source_path}: {len(data):,} selected six-animal trials", flush=True)

session_audit = pd.concat(source_audits, ignore_index=True)
assert not session_audit.duplicated(["animal", "session_type", "session", "LED_trial"]).any()
assert session_audit["event3_total"].eq(
    session_audit[["event3_pre_onset", "event3_0_to_100ms", "event3_100ms_plus", "event3_missing_time"]]
    .sum(axis=1)
).all()

transitions = {}
for animal in ANIMALS:
    s7 = session_audit.loc[
        session_audit["animal"].eq(animal) & session_audit["session_type"].eq(7), "session"
    ]
    s8 = session_audit.loc[
        session_audit["animal"].eq(animal) & session_audit["session_type"].eq(8), "session"
    ]
    last_s7, first_s8 = int(s7.max()), int(s8.min())
    assert first_s8 == last_s7 + 1
    transitions[animal] = (last_s7, first_s8)

session_audit["relative_session"] = session_audit["session"] - session_audit["animal"].map(
    {animal: first_s8 for animal, (_, first_s8) in transitions.items()}
)
assert session_audit.loc[session_audit["session_type"].eq(7), "relative_session"].lt(0).all()
assert session_audit.loc[session_audit["session_type"].eq(8), "relative_session"].ge(0).all()
for animal in ANIMALS:
    animal_sessions = session_audit.loc[session_audit["animal"].eq(animal)]
    assert animal_sessions.loc[
        animal_sessions["session_type"].eq(7), "relative_session"
    ].max() == -1
    assert animal_sessions.loc[
        animal_sessions["session_type"].eq(8), "relative_session"
    ].min() == 0


# %%
############ Centered moving windows within each uninterrupted type/LED run ############
count_columns = ("event3_total", "event3_0_to_100ms", "event3_100ms_plus")
fraction_columns = ("overall_fraction", "fraction_0_to_100ms", "fraction_100ms_plus")
animal_window_rows = []
for window in WINDOWS:
    assert window > 0 and window % 2 == 1
    half_window = window // 2
    for (animal, session_type, led), condition in session_audit.groupby(
        ["animal", "session_type", "LED_trial"], sort=True
    ):
        condition = condition.sort_values("session").reset_index(drop=True)
        session_ids = condition["session"].to_numpy(dtype=int)
        run_starts = np.flatnonzero(np.r_[True, np.diff(session_ids) != 1])
        run_ends = np.r_[run_starts[1:], len(condition)]
        for run_start, run_end in zip(run_starts, run_ends):
            run = condition.iloc[run_start:run_end].reset_index(drop=True)
            assert np.all(np.diff(run["session"].to_numpy(dtype=int)) == 1)
            for position, target in run.iterrows():
                low = max(0, position - half_window)
                high = min(len(run), position + half_window + 1)
                used = run.iloc[low:high]
                denominator = int(used["all_condition_trials"].sum())
                numerators = {column: int(used[column].sum()) for column in count_columns}
                assert denominator > 0 and len(used) <= window
                animal_window_rows.append({
                    "window_sessions": window,
                    "animal": int(animal),
                    "session_type": int(session_type),
                    "LED_trial": int(led),
                    "session": int(target["session"]),
                    "relative_session": int(target["relative_session"]),
                    "window_first_session": int(used["session"].iloc[0]),
                    "window_last_session": int(used["session"].iloc[-1]),
                    "n_sessions_used": len(used),
                    "window_condition_trials": denominator,
                    **numerators,
                    "overall_fraction": numerators["event3_total"] / denominator,
                    "fraction_0_to_100ms": numerators["event3_0_to_100ms"] / denominator,
                    "fraction_100ms_plus": numerators["event3_100ms_plus"] / denominator,
                    "raw_condition_trials": int(target["all_condition_trials"]),
                    "raw_overall_fraction": target["event3_total"] / target["all_condition_trials"],
                    "raw_fraction_0_to_100ms": target["event3_0_to_100ms"] / target["all_condition_trials"],
                    "raw_fraction_100ms_plus": target["event3_100ms_plus"] / target["all_condition_trials"],
                })

animal_windows = pd.DataFrame(animal_window_rows)
assert len(animal_windows) == len(session_audit) * len(WINDOWS)
assert not animal_windows.duplicated(
    ["window_sessions", "animal", "session_type", "session", "LED_trial"]
).any()
assert animal_windows["n_sessions_used"].le(animal_windows["window_sessions"]).all()
assert animal_windows["window_last_session"].sub(animal_windows["window_first_session"]).add(1).eq(
    animal_windows["n_sessions_used"]
).all()
assert animal_windows["session"].between(
    animal_windows["window_first_session"], animal_windows["window_last_session"]
).all()
assert animal_windows[list(fraction_columns)].ge(0).all().all()
assert animal_windows[list(fraction_columns)].le(1).all().all()
raw_rows = animal_windows.loc[animal_windows["window_sessions"].eq(1)]
assert raw_rows["n_sessions_used"].eq(1).all()
assert raw_rows["window_condition_trials"].eq(raw_rows["raw_condition_trials"]).all()
for fraction in fraction_columns:
    np.testing.assert_allclose(raw_rows[fraction], raw_rows[f"raw_{fraction}"], rtol=0, atol=0)
animal_windows.to_csv(ANIMAL_AUDIT_CSV, index=False)
print(f"Animal-window audit: {ANIMAL_AUDIT_CSV}", flush=True)


# %%
############ Equal-animal mean and across-animal SEM at each aligned session ############
mean_rows = []
for keys, group in animal_windows.groupby(
    ["window_sessions", "session_type", "relative_session", "LED_trial"], sort=True
):
    window, session_type, relative_session, led = keys
    n_animals = len(group)
    assert n_animals == group["animal"].nunique() and 1 <= n_animals <= len(ANIMALS)
    row = {
        "window_sessions": int(window),
        "session_type": int(session_type),
        "relative_session": int(relative_session),
        "LED_trial": int(led),
        "n_animals": n_animals,
        "contributing_animals": ",".join(map(str, sorted(group["animal"].astype(int)))),
    }
    for fraction in fraction_columns:
        values = group[fraction].to_numpy(dtype=float)
        # Grouping drops absent animals, so this is the nanmean of the full
        # six-animal vector, with one vote per animal rather than per trial.
        row[f"mean_{fraction}"] = float(np.mean(values))
        row[f"sem_{fraction}"] = (
            float(np.std(values, ddof=1) / np.sqrt(n_animals)) if n_animals >= 2 else np.nan
        )
    mean_rows.append(row)

mean_audit = pd.DataFrame(mean_rows)
assert not mean_audit.duplicated(
    ["window_sessions", "session_type", "relative_session", "LED_trial"]
).any()
assert mean_audit.loc[mean_audit["n_animals"].eq(1),
                      [f"sem_{fraction}" for fraction in fraction_columns]].isna().all().all()
assert mean_audit.loc[mean_audit["n_animals"].ge(2),
                      [f"sem_{fraction}" for fraction in fraction_columns]].notna().all().all()
for fraction in fraction_columns:
    assert mean_audit[f"mean_{fraction}"].between(0, 1).all()
    assert mean_audit[f"sem_{fraction}"].dropna().ge(0).all()
for fraction, (_, upper_limit) in zip(fraction_columns, Y_LIMITS):
    assert animal_windows[fraction].max() < upper_limit
    assert (
        mean_audit[f"mean_{fraction}"] + mean_audit[f"sem_{fraction}"].fillna(0)
    ).max() < upper_limit
mean_audit.to_csv(MEAN_AUDIT_CSV, index=False)
print(f"Equal-animal mean/SEM audit: {MEAN_AUDIT_CSV}", flush=True)


# %%
############ Three shared-axis figures: raw, rolling 3, and rolling 5 ############
metric_titles = ("Overall event-3 aborts", "0–100 ms after onset", "≥100 ms after onset")
relative_min = int(session_audit["relative_session"].min())
relative_max = int(session_audit["relative_session"].max())
for window in WINDOWS:
    window_rows = animal_windows.loc[animal_windows["window_sessions"].eq(window)]
    window_mean = mean_audit.loc[mean_audit["window_sessions"].eq(window)]
    fig, axes = plt.subplots(len(ANIMALS) + 1, 3, figsize=(17.0, 21.5), sharex=True)

    for row_index, animal in enumerate(ANIMALS):
        animal_rows = window_rows.loc[window_rows["animal"].eq(animal)]
        for column_index, (title, fraction) in enumerate(zip(metric_titles, fraction_columns)):
            ax = axes[row_index, column_index]
            for session_type in SESSION_TYPES:
                for led in (0, 1):
                    panel = animal_rows.loc[
                        animal_rows["session_type"].eq(session_type)
                        & animal_rows["LED_trial"].eq(led)
                    ].sort_values("relative_session")
                    x = panel["relative_session"].to_numpy(dtype=int)
                    if window > 1:
                        ax.scatter(
                            x, panel[f"raw_{fraction}"], s=14,
                            color=LED_COLORS[led], alpha=0.22, zorder=2,
                        )
                    grid = np.arange(x[0], x[-1] + 1)
                    values = np.full(len(grid), np.nan)
                    values[x - x[0]] = panel[fraction].to_numpy(dtype=float)
                    assert int(np.isfinite(values).sum()) == len(panel)
                    ax.plot(
                        grid, values, color=LED_COLORS[led], linestyle=TYPE_STYLES[session_type],
                        linewidth=1.6 if window > 1 else 1.25, zorder=3,
                    )
                    if window == 1:
                        sparse = panel["raw_condition_trials"].to_numpy(dtype=int) < LOW_TRIAL_THRESHOLD
                        ax.scatter(x[~sparse], panel.loc[~sparse, fraction], color=LED_COLORS[led], s=16, zorder=4)
                        ax.scatter(
                            x[sparse], panel.loc[sparse, fraction], s=38,
                            facecolors="white", edgecolors=LED_COLORS[led], linewidths=1.1, zorder=5,
                        )

            ax.axvline(-0.5, color="0.4", linestyle=":", linewidth=1.0, zorder=1)
            ax.set_xlim(relative_min - 1, relative_max + 1)
            ax.set_ylim(*Y_LIMITS[column_index])
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.grid(axis="y", color="0.9", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8.5)
            if row_index == 0:
                ax.set_title(title, fontsize=11.5, pad=7)
            if column_index == 0:
                last_s7, first_s8 = transitions[animal]
                ax.set_ylabel(
                    f"Animal {animal}\ns7 #{last_s7} → s8 #{first_s8}\nFraction of trials", fontsize=9
                )

    for column_index, fraction in enumerate(fraction_columns):
        ax = axes[-1, column_index]
        for session_type in SESSION_TYPES:
            for led in (0, 1):
                panel = window_mean.loc[
                    window_mean["session_type"].eq(session_type)
                    & window_mean["LED_trial"].eq(led)
                ].sort_values("relative_session")
                x = panel["relative_session"].to_numpy(dtype=int)
                grid = np.arange(x[0], x[-1] + 1)
                means = np.full(len(grid), np.nan)
                sems = np.full(len(grid), np.nan)
                means[x - x[0]] = panel[f"mean_{fraction}"].to_numpy(dtype=float)
                sems[x - x[0]] = panel[f"sem_{fraction}"].to_numpy(dtype=float)
                ax.fill_between(
                    grid, np.maximum(0, means - sems), means + sems,
                    color=LED_COLORS[led], alpha=0.14, linewidth=0, zorder=2,
                )
                ax.plot(
                    grid, means, color=LED_COLORS[led], linestyle=TYPE_STYLES[session_type],
                    linewidth=2.1, zorder=3,
                )
                one_animal = panel.loc[panel["n_animals"].eq(1)]
                ax.scatter(
                    one_animal["relative_session"], one_animal[f"mean_{fraction}"],
                    facecolors="white", edgecolors=LED_COLORS[led], s=38, linewidths=1.1, zorder=4,
                )

        ax.axvline(-0.5, color="0.4", linestyle=":", linewidth=1.0, zorder=1)
        ax.set_xlim(relative_min - 1, relative_max + 1)
        ax.set_ylim(*Y_LIMITS[column_index])
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5)
        ax.set_xlabel("Sessions from first type-8 session (0)", fontsize=10)
        if column_index == 0:
            ax.set_ylabel("Equal-animal mean\n± SEM", fontsize=9.5)

    version_label = "raw sessions" if window == 1 else f"{window}-session rolling counts/trials"
    fig.suptitle(
        f"LED7 fixation-abort fractions across the session-type switch · {version_label}",
        fontsize=15, y=0.996,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=LED_COLORS[0], lw=1.8, label="LED OFF"),
            Line2D([0], [0], color=LED_COLORS[1], lw=1.8, label="LED ON"),
            Line2D([0], [0], color="0.35", linestyle=TYPE_STYLES[7], lw=1.8, label="Type 7"),
            Line2D([0], [0], color="0.35", linestyle=TYPE_STYLES[8], lw=1.8, label="Type 8"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False,
    )
    fig.text(
        0.5, 0.010,
        "Each animal window pools event-3 counts / all LED-condition trials; centered, one-session stride, "
        "no crossing type or missing-session gaps.\n"
        "Bottom row: mean ± across-animal SEM (band for n≥2; hollow mean points for n=1). "
        "LED OFF onset is scheduled; missing abort times enter overall only.",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.947), h_pad=1.5, w_pad=0.9)
    version = "raw" if window == 1 else f"roll{window}"
    figure_path = SCRIPT_DIR / f"{OUTPUT_STEM}_{version}_7x3.png"
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {figure_path}", flush=True)

print(
    mean_audit.groupby(["window_sessions", "session_type"])["n_animals"]
    .agg(["min", "max", "median"]).to_string()
)
