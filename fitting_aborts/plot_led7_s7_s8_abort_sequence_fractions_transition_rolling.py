# %%
"""Three-session LED7 abort-sequence fractions across the session-type-7→8 switch.

Reuse the saved first/last/isolated selections, verifying their actual trial IDs
against the all-trial CSVs. No timing filter is applied to these count fractions.
"""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR.parent / "raw_data"
MANIFEST_PATH = SCRIPT_DIR / "led7_s7_s8_abort_rates_by_sequence_position_sequence_manifest.csv"
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
POSITIONS = ("first", "last", "isolated")
TITLES = ("First abort in a run", "Last abort in a run", "Isolated: valid–abort–valid")
WINDOW_SESSIONS = 3
Y_LIMITS = (0, 0.20)
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
TYPE_STYLES = {7: "-", 8: "--"}
FIGURE_DPI = 220
OUTPUT_STEM = f"led7_s7_s8_abort_sequence_fractions_transition_roll{WINDOW_SESSIONS}"
FIGURE_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_7x3.png"
SESSION_AUDIT_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_session_audit.csv"
WINDOW_AUDIT_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_animal_windows_audit.csv"
MEAN_AUDIT_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_equal_animal_mean_sem_audit.csv"

EXPECTED_SOURCE_ROWS = {7: 138_440, 8: 148_428}
EXPECTED_COHORT_ROWS = {7: 100_251, 8: 113_941}
EXPECTED_LED_COUNTS = {7: {0: 66_226, 1: 34_025}, 8: {0: 70_892, 1: 43_049}}
EXPECTED_ABORTS = {7: 16_389, 8: 17_725}
EXPECTED_SELECTED = {
    7: {"first": 2_381, "last": 2_381, "isolated": 9_863},
    8: {"first": 2_593, "last": 2_593, "isolated": 10_511},
}
EXPECTED_SESSION_GROUPS = {7: 286, 8: 325}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

assert WINDOW_SESSIONS > 0 and WINDOW_SESSIONS % 2 == 1
trial_keys = ["session_type", "animal", "session", "trial"]
condition_keys = ["animal", "session_type", "session", "LED_trial"]
count_columns = [f"{position}_count" for position in POSITIONS]
fraction_columns = [f"{position}_fraction" for position in POSITIONS]


# %%
############ Verify the saved manifest against CSV rows and actual trial adjacency ############
manifest = pd.read_csv(MANIFEST_PATH, float_precision="round_trip")
assert not manifest.duplicated(trial_keys).any()
assert manifest["session_type"].isin(SESSION_TYPES).all()
assert manifest["animal"].isin(ANIMALS).all()
assert len(manifest) == sum(EXPECTED_ABORTS.values())
source_audits = []
for session_type in SESSION_TYPES:
    source_path = RAW_DATA_DIR / f"LED7_s{session_type}.csv"
    source = pd.read_csv(source_path, float_precision="round_trip")
    assert source.shape == (EXPECTED_SOURCE_ROWS[session_type], 52)
    assert source["training_level"].eq(16).all()
    assert source["session_type"].eq(session_type).all()
    assert (source["repeat_trial"].isin([0, 2]) | source["repeat_trial"].isna()).all()
    assert source["LED_trial"].isin([0, 1]).all()
    assert not source.duplicated(trial_keys).any()
    source["source_csv_row_index"] = np.arange(len(source))
    data = source.loc[source["animal"].isin(ANIMALS)].sort_values(trial_keys, kind="stable").copy()
    assert len(data) == EXPECTED_COHORT_ROWS[session_type]
    assert tuple(sorted(data["animal"].astype(int).unique())) == ANIMALS
    assert data["LED_trial"].value_counts().to_dict() == EXPECTED_LED_COUNTS[session_type]
    for column in [*trial_keys, "LED_trial"]:
        assert data[column].notna().all() and data[column].eq(np.floor(data[column])).all()
        data[column] = data[column].astype(int)

    selected_manifest = manifest.loc[manifest["session_type"].eq(session_type)].sort_values(trial_keys).reset_index(drop=True)
    source_aborts = data.loc[data["abort_event"].eq(3)].reset_index(drop=True)
    assert len(source_aborts) == len(selected_manifest) == EXPECTED_ABORTS[session_type]
    compare_columns = [
        *trial_keys, "source_csv_row_index", "training_level", "repeat_trial", "LED_trial",
        "abort_event", "success", "timed_fix", "intended_fix", "LED_onset_time",
    ]
    np.testing.assert_allclose(
        source_aborts[compare_columns], selected_manifest[compare_columns],
        rtol=0, atol=0, equal_nan=True,
    )
    assert selected_manifest["source_csv"].eq(str(source_path.relative_to(SCRIPT_DIR.parent))).all()

    # Independently check the saved selections from full session histories.
    # Shift only finds candidate neighbors: differences in the `trial` values
    # determine adjacency. LED is not a grouping key for sequence detection.
    groups = data.groupby(["animal", "session"], sort=False)
    previous = groups[["trial", "abort_event", "success"]].shift(1)
    following = groups[["trial", "abort_event", "success"]].shift(-1)
    previous_adjacent = previous["trial"].eq(data["trial"] - 1)
    next_adjacent = following["trial"].eq(data["trial"] + 1)
    abort_mask = data["abort_event"].eq(3)
    starts_run = abort_mask & ~(previous_adjacent & previous["abort_event"].eq(3))
    check = data.loc[abort_mask, trial_keys].copy()
    check["run"] = starts_run.cumsum().loc[abort_mask]
    check["previous_adjacent"] = previous_adjacent.loc[abort_mask]
    check["next_adjacent"] = next_adjacent.loc[abort_mask]
    check["previous_valid"] = previous.loc[abort_mask, "success"].isin([-1, 1])
    check["next_valid"] = following.loc[abort_mask, "success"].isin([-1, 1])
    check_groups = check.groupby("run", sort=False)
    check["position"] = check_groups.cumcount() + 1
    check["length"] = check_groups["trial"].transform("size")
    check["complete"] = (
        check_groups["previous_adjacent"].transform("first")
        & check_groups["next_adjacent"].transform("last")
    )
    isolated = (
        check["complete"] & check["length"].eq(1)
        & check["previous_valid"] & check["next_valid"]
    )
    multi = check["complete"] & check["length"].ge(2)
    expected_selection = np.select(
        [multi & check["position"].eq(1), multi & check["position"].eq(check["length"]),
         multi, isolated, ~check["complete"]],
        ["first", "last", "interior", "isolated", "boundary_or_gap"],
        default="singleton_other",
    )
    np.testing.assert_array_equal(expected_selection, selected_manifest["selection"])
    np.testing.assert_array_equal(check["position"], selected_manifest["run_position"])
    np.testing.assert_array_equal(check["length"], selected_manifest["observed_run_length"])
    np.testing.assert_array_equal(check["complete"], selected_manifest["complete_run"])
    np.testing.assert_array_equal(
        check["run"].ne(check["run"].shift()),
        selected_manifest["run_id"].ne(selected_manifest["run_id"].shift()),
    )

    # Join selections onto all eligible trials, so valid/other-outcome rows
    # still contribute to the denominator and sessions with zero events remain.
    data = data.merge(
        selected_manifest[trial_keys + ["selection"]], on=trial_keys,
        how="left", validate="one_to_one", indicator=True,
    )
    assert data["_merge"].eq("both").equals(data["abort_event"].eq(3))
    data["event3_total"] = data["abort_event"].eq(3)
    data["event3_missing_timing"] = data["event3_total"] & ~np.isfinite(data["timed_fix"])
    assert int(data["event3_missing_timing"].sum()) == 2
    aggregations = {
        "all_condition_trials": ("trial", "size"),
        "event3_total": ("event3_total", "sum"),
        "event3_missing_timing": ("event3_missing_timing", "sum"),
    }
    for position in POSITIONS:
        data[f"{position}_count"] = data["selection"].eq(position)
        data[f"{position}_missing_timing"] = data[f"{position}_count"] & ~np.isfinite(data["timed_fix"])
        aggregations[f"{position}_count"] = (f"{position}_count", "sum")
        aggregations[f"{position}_missing_timing"] = (f"{position}_missing_timing", "sum")
    audit = data.groupby(condition_keys, sort=True).agg(**aggregations).reset_index()
    assert len(audit) == EXPECTED_SESSION_GROUPS[session_type]
    assert int(audit["all_condition_trials"].sum()) == EXPECTED_COHORT_ROWS[session_type]
    assert int(audit["event3_total"].sum()) == EXPECTED_ABORTS[session_type]
    assert audit.groupby("LED_trial")["all_condition_trials"].sum().to_dict() == EXPECTED_LED_COUNTS[session_type]
    for position in POSITIONS:
        assert int(audit[f"{position}_count"].sum()) == EXPECTED_SELECTED[session_type][position]
        audit[f"{position}_fraction"] = audit[f"{position}_count"] / audit["all_condition_trials"]
    assert audit[count_columns].sum(axis=1).le(audit["event3_total"]).all()
    source_audits.append(audit)
    print(f"s{session_type}: verified source and sequence manifest; {len(data):,} trials, {len(audit)} session/LED groups", flush=True)

session_audit = pd.concat(source_audits, ignore_index=True)
assert len(session_audit) == 611 and not session_audit.duplicated(condition_keys).any()
assert session_audit["all_condition_trials"].gt(0).all()
assert session_audit[fraction_columns].ge(0).all().all()
assert session_audit[fraction_columns].le(1).all().all()


# %%
############ Align sessions to the first s8 session, retaining actual IDs and gaps ############
transitions = {}
for animal in ANIMALS:
    animal_sessions = session_audit.loc[session_audit["animal"].eq(animal)]
    last_s7 = int(animal_sessions.loc[animal_sessions["session_type"].eq(7), "session"].max())
    first_s8 = int(animal_sessions.loc[animal_sessions["session_type"].eq(8), "session"].min())
    assert first_s8 == last_s7 + 1
    transitions[animal] = (last_s7, first_s8)
session_audit["relative_session"] = session_audit["session"] - session_audit["animal"].map(
    {animal: first_s8 for animal, (_, first_s8) in transitions.items()}
)
assert session_audit.loc[session_audit["session_type"].eq(7), "relative_session"].lt(0).all()
assert session_audit.loc[session_audit["session_type"].eq(8), "relative_session"].ge(0).all()


# %%
############ Centered count/trial windows; move one session at a time ############
half_window = WINDOW_SESSIONS // 2
window_records = []
window_count_columns = ["event3_total", "event3_missing_timing", *count_columns]
window_count_columns += [f"{position}_missing_timing" for position in POSITIONS]
for (animal, session_type, led), condition in session_audit.groupby(
    ["animal", "session_type", "LED_trial"], sort=True
):
    condition = condition.sort_values("session").reset_index(drop=True)
    session_ids = condition["session"].to_numpy(dtype=int)
    run_starts = np.flatnonzero(np.r_[True, np.diff(session_ids) != 1])
    run_ends = np.r_[run_starts[1:], len(condition)]
    for run_start, run_end in zip(run_starts, run_ends):
        run = condition.iloc[run_start:run_end].reset_index(drop=True)
        for index, target in run.iterrows():
            used = run.iloc[max(0, index - half_window):min(len(run), index + half_window + 1)]
            used_ids = used["session"].to_numpy(dtype=int)
            assert np.all(np.diff(used_ids) == 1)
            assert np.all(np.abs(used_ids - int(target["session"])) <= half_window)
            denominator = int(used["all_condition_trials"].sum())
            row = {
                "animal": int(animal), "session_type": int(session_type), "LED_trial": int(led),
                "session": int(target["session"]), "relative_session": int(target["relative_session"]),
                "window_sessions": WINDOW_SESSIONS, "n_sessions_used": len(used),
                "window_first_session": int(used_ids[0]), "window_last_session": int(used_ids[-1]),
                "window_session_ids": ",".join(map(str, used_ids)),
                "window_condition_trials": denominator,
                "raw_condition_trials": int(target["all_condition_trials"]),
            }
            for column in window_count_columns:
                row[column] = int(used[column].sum())
                row[f"raw_{column}"] = int(target[column])
            for position in POSITIONS:
                row[f"{position}_fraction"] = row[f"{position}_count"] / denominator
                row[f"raw_{position}_fraction"] = target[f"{position}_fraction"]
            window_records.append(row)

animal_windows = pd.DataFrame(window_records)
assert len(animal_windows) == 611 and not animal_windows.duplicated(condition_keys).any()
assert animal_windows["n_sessions_used"].between(1, WINDOW_SESSIONS).all()
assert animal_windows["window_last_session"].sub(animal_windows["window_first_session"]).add(1).eq(animal_windows["n_sessions_used"]).all()
assert animal_windows[fraction_columns].ge(0).all().all() and animal_windows[fraction_columns].le(1).all().all()

# Reconstruct the centered window from session-ID lookup rather than array
# slices. This also verifies that missing IDs/conditions cannot be bridged.
session_lookup = session_audit.set_index(condition_keys)
for row in animal_windows.itertuples(index=False):
    expected_ids = [row.session]
    for direction in (-1, 1):
        for distance in range(1, half_window + 1):
            candidate = row.session + direction * distance
            if (row.animal, row.session_type, candidate, row.LED_trial) not in session_lookup.index:
                break
            expected_ids.append(candidate)
    expected_ids = sorted(expected_ids)
    assert expected_ids == [int(value) for value in row.window_session_ids.split(",")]
    members = session_lookup.loc[[(row.animal, row.session_type, s, row.LED_trial) for s in expected_ids]]
    assert int(members["all_condition_trials"].sum()) == row.window_condition_trials
    target = session_lookup.loc[(row.animal, row.session_type, row.session, row.LED_trial)]
    for position in POSITIONS:
        numerator = int(members[f"{position}_count"].sum())
        assert numerator == getattr(row, f"{position}_count")
        assert numerator / row.window_condition_trials == getattr(row, f"{position}_fraction")
        assert target[f"{position}_fraction"] == getattr(row, f"raw_{position}_fraction")


# %%
############ Equal-animal mean and SEM after smoothing; no pooled row ############
mean_records = []
for (session_type, relative_session, led), group in animal_windows.groupby(
    ["session_type", "relative_session", "LED_trial"], sort=True
):
    n = len(group)
    assert n == group["animal"].nunique() and 1 <= n <= len(ANIMALS)
    record = {
        "session_type": int(session_type), "relative_session": int(relative_session),
        "LED_trial": int(led), "window_sessions": WINDOW_SESSIONS,
        "n_animals": n, "contributing_animals": ",".join(map(str, sorted(group["animal"].astype(int)))),
    }
    for position in POSITIONS:
        values = group[f"{position}_fraction"].to_numpy(dtype=float)
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(n)) if n >= 2 else np.nan
        full_animal_vector = group.set_index("animal")[f"{position}_fraction"].reindex(ANIMALS).to_numpy()
        assert np.isclose(mean, np.nanmean(full_animal_vector), rtol=0, atol=1e-14)
        if n >= 2:
            expected_sem = np.sqrt(np.sum((values - mean) ** 2) / (n * (n - 1)))
            assert np.isclose(sem, expected_sem, rtol=0, atol=1e-14)
        record[f"mean_{position}_fraction"] = mean
        record[f"sem_{position}_fraction"] = sem
    mean_records.append(record)
mean_audit = pd.DataFrame(mean_records)
assert not mean_audit.duplicated(["session_type", "relative_session", "LED_trial"]).any()
for position in POSITIONS:
    assert session_audit[f"{position}_fraction"].max() < Y_LIMITS[1]
    assert animal_windows[f"{position}_fraction"].between(*Y_LIMITS).all()
    assert mean_audit[f"mean_{position}_fraction"].between(*Y_LIMITS).all()
    assert (mean_audit[f"mean_{position}_fraction"] + mean_audit[f"sem_{position}_fraction"].fillna(0)).max() < Y_LIMITS[1]
    assert mean_audit.loc[mean_audit["n_animals"].eq(1), f"sem_{position}_fraction"].isna().all()
    assert mean_audit.loc[mean_audit["n_animals"].ge(2), f"sem_{position}_fraction"].notna().all()

for audit, output_path in ((session_audit, SESSION_AUDIT_PATH), (animal_windows, WINDOW_AUDIT_PATH), (mean_audit, MEAN_AUDIT_PATH)):
    audit.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(audit):,} rows)", flush=True)
print("\nCounts by session type:")
print(session_audit.groupby("session_type")[["all_condition_trials", *count_columns]].sum().to_string())
print("\nContributing animals:")
print(mean_audit.groupby("session_type")["n_animals"].agg(["min", "max", "median"]).to_string())


# %%
############ Combined session-type plot: six animals and equal-animal mean ± SEM ############
relative_min, relative_max = int(session_audit["relative_session"].min()), int(session_audit["relative_session"].max())
fig, axes = plt.subplots(len(ANIMALS) + 1, len(POSITIONS), figsize=(17.0, 21.5), sharex=True, sharey=True)
for row_index, animal in enumerate(ANIMALS):
    animal_rows = animal_windows.loc[animal_windows["animal"].eq(animal)]
    for column, position in enumerate(POSITIONS):
        ax = axes[row_index, column]
        for session_type in SESSION_TYPES:
            for led in (0, 1):
                panel = animal_rows.loc[animal_rows["session_type"].eq(session_type) & animal_rows["LED_trial"].eq(led)].sort_values("relative_session")
                x = panel["relative_session"].to_numpy(dtype=int)
                ax.scatter(x, panel[f"raw_{position}_fraction"], s=14, color=LED_COLORS[led], alpha=0.22, zorder=2)
                grid = np.arange(x[0], x[-1] + 1)
                values = np.full(len(grid), np.nan)
                values[x - x[0]] = panel[f"{position}_fraction"].to_numpy()
                assert np.isfinite(values).sum() == len(panel)
                ax.plot(grid, values, color=LED_COLORS[led], linestyle=TYPE_STYLES[session_type], linewidth=1.7, zorder=3)
        if column == 0:
            last_s7, first_s8 = transitions[animal]
            ax.set_ylabel(f"Animal {animal}\ns7 #{last_s7} → s8 #{first_s8}\nFraction of trials", fontsize=9.5)

for column, position in enumerate(POSITIONS):
    ax = axes[-1, column]
    for session_type in SESSION_TYPES:
        for led in (0, 1):
            panel = mean_audit.loc[mean_audit["session_type"].eq(session_type) & mean_audit["LED_trial"].eq(led)].sort_values("relative_session")
            x = panel["relative_session"].to_numpy(dtype=int)
            grid = np.arange(x[0], x[-1] + 1)
            means, sems = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
            means[x - x[0]] = panel[f"mean_{position}_fraction"].to_numpy()
            sems[x - x[0]] = panel[f"sem_{position}_fraction"].to_numpy()
            ax.fill_between(grid, np.maximum(0, means - sems), means + sems, color=LED_COLORS[led], alpha=0.15, linewidth=0, zorder=2)
            ax.plot(grid, means, color=LED_COLORS[led], linestyle=TYPE_STYLES[session_type], linewidth=2.1, zorder=3)
            single_animal = panel.loc[panel["n_animals"].eq(1)]
            ax.scatter(single_animal["relative_session"], single_animal[f"mean_{position}_fraction"], facecolors="white", edgecolors=LED_COLORS[led], s=38, linewidths=1.1, zorder=4)
    ax.set_xlabel("Sessions from first type-8 session (0)", fontsize=10)
    if column == 0:
        ax.set_ylabel("Equal-animal mean\n± SEM", fontsize=10)

for ax in axes.flat:
    ax.axvline(-0.5, color="0.4", linestyle=":", linewidth=1.0, zorder=1)
    ax.set_xlim(relative_min - 1, relative_max + 1)
    ax.set_ylim(*Y_LIMITS)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
for column, title in enumerate(TITLES):
    axes[0, column].set_title(title, fontsize=12, pad=9)
fig.suptitle(f"LED7 abort-sequence fractions across sessions · {WINDOW_SESSIONS}-session rolling", fontsize=16, y=0.992)
fig.legend(
    handles=[
        Line2D([0], [0], color=LED_COLORS[0], lw=1.8, label="LED OFF"),
        Line2D([0], [0], color=LED_COLORS[1], lw=1.8, label="LED ON"),
        Line2D([0], [0], color="0.35", linestyle=TYPE_STYLES[7], lw=1.8, label="Type 7"),
        Line2D([0], [0], color="0.35", linestyle=TYPE_STYLES[8], lw=1.8, label="Type 8"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.978), ncol=4, frameon=False, fontsize=11,
)
fig.text(
    0.5, 0.013,
    "Faint dots: raw session fractions. Curves: selected aborts / all LED-condition trials in centered windows, one-session stride.\n"
    "Windows stay within session type and consecutive session IDs; edges use shorter windows. First/last refer to complete runs of ≥2 event-3 aborts.\n"
    "Bottom row: equal-animal mean ± SEM; hollow points indicate one contributing animal. Zero marks the first type-8 session.",
    ha="center", fontsize=9, linespacing=1.45,
)
fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.949), h_pad=1.6, w_pad=0.9)
fig.savefig(FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure: {FIGURE_PATH}", flush=True)
print("Source/sequence, session counts, centered-window boundaries, fractions, and equal-animal SEM checks passed.", flush=True)
