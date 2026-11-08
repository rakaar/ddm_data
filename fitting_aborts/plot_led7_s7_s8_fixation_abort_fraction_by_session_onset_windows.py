# %%
"""LED7 event-3 abort fractions by session and time from scheduled LED onset."""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR.parent / "raw_data"
SESSION_TYPES = (7, 8)
ANIMALS = (92, 93, 98, 99, 100, 103)
POST_ONSET_SPLIT_S = 0.100
LOW_TRIAL_THRESHOLD = 100
FIGURE_DPI = 220
Y_LIMITS = ((0, 0.45), (0, 0.15), (0, 0.25))
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
AUDIT_CSV = SCRIPT_DIR / "led7_s7_s8_fixation_abort_fraction_by_session_onset_windows_audit.csv"

EXPECTED_SOURCE_ROWS = {7: 138_440, 8: 148_428}
EXPECTED_SIX_ANIMAL_ROWS = {7: 100_251, 8: 113_941}
EXPECTED_LED_COUNTS = {
    7: {0: 66_226, 1: 34_025},
    8: {0: 70_892, 1: 43_049},
}
EXPECTED_EVENT3_COUNTS = {7: 16_389, 8: 17_725}
EXPECTED_PRE_ONSET_COUNTS = {7: 9_666, 8: 9_517}
EXPECTED_0_TO_100MS_COUNTS = {7: 3_039, 8: 2_891}
EXPECTED_100MS_PLUS_COUNTS = {7: 3_682, 8: 5_315}
EXPECTED_MISSING_ABORT_TIMES = {7: 2, 8: 2}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


# %%
############ Read the standardized all-trial CSVs and count aborts ############
audits = []
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
    assert np.equal(data[["animal", "session", "trial"]], np.floor(data[["animal", "session", "trial"]])).all().all()
    assert data["LED_onset_time"].ge(0).all()
    assert data["LED_onset_time"].le(data["intended_fix"]).all()

    event3 = data["abort_event"].eq(3)
    finite_abort_time = np.isfinite(data["timed_fix"])
    assert data.loc[event3 & finite_abort_time, "timed_fix"].lt(
        data.loc[event3 & finite_abort_time, "intended_fix"]
    ).all()

    # LED_onset_time is already fixation-referenced in these CSVs. For OFF
    # trials, it is a scheduled/counterfactual onset, not actual illumination.
    abort_from_onset = data["timed_fix"] - data["LED_onset_time"]
    data["event3_total"] = event3
    data["event3_pre_onset"] = event3 & finite_abort_time & abort_from_onset.lt(0)
    data["event3_0_to_100ms"] = (
        event3 & finite_abort_time
        & abort_from_onset.ge(0) & abort_from_onset.lt(POST_ONSET_SPLIT_S)
    )
    data["event3_100ms_plus"] = (
        event3 & finite_abort_time & abort_from_onset.ge(POST_ONSET_SPLIT_S)
    )
    data["event3_missing_time"] = event3 & ~finite_abort_time
    assert data[
        ["event3_pre_onset", "event3_0_to_100ms", "event3_100ms_plus", "event3_missing_time"]
    ].sum(axis=1).eq(data["event3_total"]).all()

    audit = (
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
        audit[column] = audit[column].astype(int)
    audit.insert(0, "session_type", session_type)
    audit["overall_fraction"] = audit["event3_total"] / audit["all_condition_trials"]
    audit["fraction_0_to_100ms"] = audit["event3_0_to_100ms"] / audit["all_condition_trials"]
    audit["fraction_100ms_plus"] = audit["event3_100ms_plus"] / audit["all_condition_trials"]
    audit["below_trial_threshold"] = audit["all_condition_trials"].lt(LOW_TRIAL_THRESHOLD)

    assert not audit.duplicated(["session_type", "animal", "session", "LED_trial"]).any()
    assert int(audit["all_condition_trials"].sum()) == len(data)
    assert int(audit["event3_total"].sum()) == EXPECTED_EVENT3_COUNTS[session_type]
    assert int(audit["event3_pre_onset"].sum()) == EXPECTED_PRE_ONSET_COUNTS[session_type]
    assert int(audit["event3_0_to_100ms"].sum()) == EXPECTED_0_TO_100MS_COUNTS[session_type]
    assert int(audit["event3_100ms_plus"].sum()) == EXPECTED_100MS_PLUS_COUNTS[session_type]
    assert int(audit["event3_missing_time"].sum()) == EXPECTED_MISSING_ABORT_TIMES[session_type]
    assert audit["event3_total"].eq(
        audit[["event3_pre_onset", "event3_0_to_100ms", "event3_100ms_plus", "event3_missing_time"]].sum(axis=1)
    ).all()
    assert audit.groupby("LED_trial")["all_condition_trials"].sum().to_dict() == EXPECTED_LED_COUNTS[session_type]
    assert audit[["overall_fraction", "fraction_0_to_100ms", "fraction_100ms_plus"]].le(1).all().all()
    assert audit[["overall_fraction", "fraction_0_to_100ms", "fraction_100ms_plus"]].ge(0).all().all()
    for fraction, y_limits in zip(
        ("overall_fraction", "fraction_0_to_100ms", "fraction_100ms_plus"), Y_LIMITS
    ):
        assert audit[fraction].max() < y_limits[1], f"{fraction} would exceed its axis limit."

    print(f"Source: {source_path}")
    print(
        f"s{session_type}: {len(data):,} trials; "
        f"event 3 = {int(audit['event3_total'].sum()):,}, "
        f"pre-onset = {int(audit['event3_pre_onset'].sum()):,}, "
        f"0–100 ms = {int(audit['event3_0_to_100ms'].sum()):,}, "
        f"100+ ms = {int(audit['event3_100ms_plus'].sum()):,}, "
        f"missing time = {int(audit['event3_missing_time'].sum())}"
    )
    audits.append(audit)

audit = pd.concat(audits, ignore_index=True)
audit.to_csv(AUDIT_CSV, index=False)
print(f"Audit: {AUDIT_CSV}")


# %%
############ Plot fractions against each animal's actual session IDs ############
columns = (
    ("Overall event-3 aborts", "overall_fraction"),
    ("0–100 ms after onset", "fraction_0_to_100ms"),
    ("≥100 ms after onset", "fraction_100ms_plus"),
)
for session_type in SESSION_TYPES:
    type_audit = audit.loc[audit["session_type"].eq(session_type)]
    fig, axes = plt.subplots(len(ANIMALS), len(columns), figsize=(16.5, 19.5))
    for row, animal in enumerate(ANIMALS):
        animal_audit = type_audit.loc[type_audit["animal"].eq(animal)]
        first_session = int(animal_audit["session"].min())
        last_session = int(animal_audit["session"].max())
        full_session_ids = np.arange(first_session, last_session + 1)

        for col, (title, fraction) in enumerate(columns):
            ax = axes[row, col]
            for led in (0, 1):
                panel = animal_audit.loc[animal_audit["LED_trial"].eq(led)].sort_values("session")
                session_ids = panel["session"].to_numpy(dtype=int)
                values = panel[fraction].to_numpy(dtype=float)
                sparse = panel["below_trial_threshold"].to_numpy(dtype=bool)

                # A NaN breaks the line at a missing session ID or missing LED
                # condition; no observation is silently plotted as zero.
                complete_values = np.full(len(full_session_ids), np.nan)
                complete_values[session_ids - first_session] = values
                assert int(np.isfinite(complete_values).sum()) == len(panel)
                ax.plot(full_session_ids, complete_values, color=LED_COLORS[led], linewidth=1.3)
                ax.scatter(session_ids[~sparse], values[~sparse], color=LED_COLORS[led], s=18, zorder=3)
                ax.scatter(
                    session_ids[sparse], values[sparse],
                    facecolors="white", edgecolors=LED_COLORS[led],
                    linewidths=1.2, s=40, zorder=4,
                )

            ax.set_xlim(first_session - 0.7, last_session + 0.7)
            ax.set_ylim(*Y_LIMITS[col])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
            ax.grid(axis="y", color="0.88", linewidth=0.65)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9)
            if row == 0:
                ax.set_title(title, fontsize=12, pad=8)
            if col == 0:
                ax.set_ylabel(f"Animal {animal}\nFraction of trials", fontsize=10)
            ax.set_xlabel("Session number", fontsize=10)

    fig.suptitle(f"LED7 session type {session_type}: fixation-abort fractions by session", fontsize=16, y=0.995)
    fig.legend(
        handles=[
            Line2D([0], [0], color=LED_COLORS[0], marker="o", lw=1.3, label="LED OFF"),
            Line2D([0], [0], color=LED_COLORS[1], marker="o", lw=1.3, label="LED ON"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False,
    )
    fig.text(
        0.5, 0.014,
        "Each point = event-3 aborts in the stated window / all trials in that session and LED condition. "
        "Hollow points: <100 trials.\n"
        "LED OFF onset is scheduled/counterfactual; pre-onset aborts are included only in overall. "
        "Missing abort times remain in overall only.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.95), h_pad=1.7, w_pad=1.0)
    figure_path = SCRIPT_DIR / f"led7_s{session_type}_fixation_abort_fraction_by_session_onset_windows_6x3.png"
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {figure_path}")
