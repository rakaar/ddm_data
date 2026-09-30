# %%
"""LED7 conditional abort probabilities across the s7→s8 training transition.

Columns 1–2 condition their denominators on the preceding trial's outcome/LED.
Column 3 reproduces the existing isolated-abort / all-trials fraction exactly.
"""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR.parent / "raw_data"
MANIFEST_PATH = SCRIPT_DIR / "led7_s7_s8_abort_rates_by_sequence_position_sequence_manifest.csv"
REFERENCE_STEM = "led7_s7_s8_abort_sequence_fractions_transition_roll3"
ANIMALS = (92, 93, 98, 99, 100, 103)
SESSION_TYPES = (7, 8)
METRICS = ("previous_abort", "previous_same_LED_abort", "isolated")
TITLES = ("Abort given previous abort", "Abort given previous same-LED abort", "Isolated: valid–abort–valid")
WINDOW_SESSIONS = 3
Y_LIMITS = ((0, 0.70), (0, 0.70), (0, 0.20))
MEAN_Y_LIMITS = (0, 0.35)
LED_COLORS = {0: "tab:blue", 1: "tab:red"}
TYPE_STYLES = {7: "-", 8: "--"}
FIGURE_DPI = 220
OUTPUT_STEM = f"led7_s7_s8_conditional_abort_fractions_transition_roll{WINDOW_SESSIONS}"
FIGURE_PATH = SCRIPT_DIR / f"{OUTPUT_STEM}_7x3.png"

EXPECTED_SOURCE_ROWS = {7: 138_440, 8: 148_428}
EXPECTED_COHORT_ROWS = {7: 100_251, 8: 113_941}
EXPECTED_LED_COUNTS = {7: {0: 66_226, 1: 34_025}, 8: {0: 70_892, 1: 43_049}}
EXPECTED_ABORTS = {7: 16_389, 8: 17_725}
# Tuples are (current aborts, eligible current trials), independently by LED.
EXPECTED_CONDITIONAL = {
    7: {"previous_abort": {0: (1_528, 9_020), 1: (1_569, 7_348)},
        "previous_same_LED_abort": {0: (1_528, 9_020), 1: (1_361, 6_329)}},
    8: {"previous_abort": {0: (963, 6_450), 1: (2_463, 11_251)},
        "previous_same_LED_abort": {0: (963, 6_450), 1: (1_936, 8_532)}},
}
EXPECTED_ISOLATED = {7: 9_863, 8: 10_511}

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
metric_keys = ["metric", *condition_keys]
mean_keys = ["session_type", "relative_session", "LED_trial"]


# %%
############ Read sources and derive current-trial eligibility from actual trial IDs ############
manifest = pd.read_csv(MANIFEST_PATH, float_precision="round_trip")
assert not manifest.duplicated(trial_keys).any()
assert len(manifest) == sum(EXPECTED_ABORTS.values())
assert manifest["animal"].isin(ANIMALS).all()
assert manifest["session_type"].isin(SESSION_TYPES).all()
session_parts = []
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

    saved = manifest.loc[manifest["session_type"].eq(session_type)].sort_values(trial_keys).reset_index(drop=True)
    aborts = data.loc[data["abort_event"].eq(3)].reset_index(drop=True)
    assert len(aborts) == len(saved) == EXPECTED_ABORTS[session_type]
    compare = [*trial_keys, "source_csv_row_index", "LED_trial", "abort_event", "success",
               "repeat_trial", "timed_fix", "intended_fix", "LED_onset_time"]
    np.testing.assert_allclose(aborts[compare], saved[compare], rtol=0, atol=0, equal_nan=True)
    assert saved["source_csv"].eq(str(source_path.relative_to(SCRIPT_DIR.parent))).all()

    grouped = data.groupby(["animal", "session"], sort=False)
    previous = grouped[["trial", "abort_event", "LED_trial", "success"]].shift(1)
    following = grouped[["trial", "success"]].shift(-1)
    adjacent_previous = previous["trial"].eq(data["trial"] - 1)
    adjacent_next = following["trial"].eq(data["trial"] + 1)
    current_abort = data["abort_event"].eq(3)
    previous_abort = adjacent_previous & previous["abort_event"].eq(3)
    same_LED = previous_abort & previous["LED_trial"].eq(data["LED_trial"])

    # Independent keyed lookup verifies that filtered-out trials and session
    # boundaries cannot accidentally turn adjacent CSV rows into trial pairs.
    lookup_keys = ["animal", "session", "trial"]
    previous_keys = pd.MultiIndex.from_arrays(
        [data["animal"], data["session"], data["trial"] - 1], names=lookup_keys,
    )
    keyed_previous = data.set_index(lookup_keys)[["abort_event", "LED_trial"]].reindex(previous_keys)
    np.testing.assert_array_equal(previous_abort, keyed_previous["abort_event"].eq(3))
    np.testing.assert_array_equal(
        same_LED,
        keyed_previous["abort_event"].eq(3).to_numpy()
        & (keyed_previous["LED_trial"].to_numpy() == data["LED_trial"].to_numpy()),
    )

    # Only the isolated category looks at the following trial. Reuse saved
    # selection identities, and verify them directly from valid neighbors.
    isolated_keys = pd.MultiIndex.from_frame(saved.loc[saved["selection"].eq("isolated"), trial_keys])
    is_isolated = pd.Series(pd.MultiIndex.from_frame(data[trial_keys]).isin(isolated_keys), index=data.index)
    expected_isolated = (
        current_abort & adjacent_previous & adjacent_next
        & previous["success"].isin([-1, 1]) & following["success"].isin([-1, 1])
    )
    np.testing.assert_array_equal(is_isolated, expected_isolated)
    assert int(is_isolated.sum()) == EXPECTED_ISOLATED[session_type]

    # No timed_fix restriction: a coded abort is a count observation even if
    # its timestamp is missing, and all eligible current outcomes enter N.
    for metric in METRICS:
        eligible = {"previous_abort": previous_abort, "previous_same_LED_abort": same_LED,
                    "isolated": pd.Series(True, index=data.index)}[metric]
        numerator = is_isolated if metric == "isolated" else eligible & current_abort
        data["eligible"] = eligible
        data["counted_abort"] = numerator
        data["eligible_missing_timing"] = eligible & ~np.isfinite(data["timed_fix"])
        data["counted_abort_missing_timing"] = numerator & ~np.isfinite(data["timed_fix"])
        audit = data.groupby(condition_keys, sort=True).agg(
            all_condition_trials=("trial", "size"),
            eligible_trials=("eligible", "sum"), aborts=("counted_abort", "sum"),
            eligible_missing_timing=("eligible_missing_timing", "sum"),
            aborts_missing_timing=("counted_abort_missing_timing", "sum"),
        ).reset_index()
        audit.insert(0, "metric", metric)
        audit["fraction"] = audit["aborts"].div(audit["eligible_trials"].where(audit["eligible_trials"].gt(0)))
        audit["defined"] = audit["eligible_trials"].gt(0)
        assert len(audit) == {7: 286, 8: 325}[session_type]
        assert int(audit["all_condition_trials"].sum()) == EXPECTED_COHORT_ROWS[session_type]
        if metric != "isolated":
            for led, expected in EXPECTED_CONDITIONAL[session_type][metric].items():
                cell = audit.loc[audit["LED_trial"].eq(led)]
                assert (int(cell["aborts"].sum()), int(cell["eligible_trials"].sum())) == expected
        session_parts.append(audit)
    print(f"s{session_type}: source, keyed trial adjacency, conditional counts, and isolated selections verified.", flush=True)

session_audit = pd.concat(session_parts, ignore_index=True)
assert len(session_audit) == 611 * 3 and not session_audit.duplicated(metric_keys).any()
assert session_audit["aborts"].between(0, session_audit["eligible_trials"]).all()
assert session_audit["eligible_trials"].le(session_audit["all_condition_trials"]).all()
assert session_audit["fraction"].isna().equals(~session_audit["defined"])
zero_eligible = session_audit.loc[~session_audit["defined"]]
assert len(zero_eligible) == 2
assert zero_eligible[condition_keys].eq([98, 8, 68, 1]).all().all()
assert set(zero_eligible["metric"]) == set(METRICS[:2])
assert session_audit.loc[session_audit["defined"] & session_audit["aborts"].eq(0), "fraction"].eq(0).all()


# %%
############ Align actual session IDs to first s8 session (0) ############
transitions = {}
for animal in ANIMALS:
    animal_rows = session_audit.loc[session_audit["animal"].eq(animal)]
    last_s7 = int(animal_rows.loc[animal_rows["session_type"].eq(7), "session"].max())
    first_s8 = int(animal_rows.loc[animal_rows["session_type"].eq(8), "session"].min())
    assert first_s8 == last_s7 + 1
    transitions[animal] = (last_s7, first_s8)
session_audit["relative_session"] = session_audit["session"] - session_audit["animal"].map(
    {animal: first_s8 for animal, (_, first_s8) in transitions.items()}
)
assert session_audit.loc[session_audit["session_type"].eq(7), "relative_session"].lt(0).all()
assert session_audit.loc[session_audit["session_type"].eq(8), "relative_session"].ge(0).all()


# %%
############ Centered numerator/eligible-denominator windows, separately per metric ############
half_window = WINDOW_SESSIONS // 2
window_records = []
for (metric, animal, session_type, led), condition in session_audit.groupby(
    ["metric", "animal", "session_type", "LED_trial"], sort=True
):
    # Remove zero-denominator targets before identifying consecutive-session
    # runs. Their undefined audit rows are restored below, not interpolated.
    valid = condition.loc[condition["defined"]].sort_values("session").reset_index(drop=True)
    run_starts = np.flatnonzero(np.r_[True, np.diff(valid["session"].to_numpy(dtype=int)) != 1]) if len(valid) else []
    run_ends = np.r_[run_starts[1:], len(valid)] if len(valid) else []
    for start, end in zip(run_starts, run_ends):
        run = valid.iloc[start:end].reset_index(drop=True)
        for index, target in run.iterrows():
            used = run.iloc[max(0, index - half_window):min(len(run), index + half_window + 1)]
            ids = used["session"].to_numpy(dtype=int)
            assert np.all(np.diff(ids) == 1)
            numerator, denominator = int(used["aborts"].sum()), int(used["eligible_trials"].sum())
            window_records.append({
                "metric": metric, "animal": int(animal), "session_type": int(session_type),
                "LED_trial": int(led), "session": int(target["session"]),
                "relative_session": int(target["relative_session"]), "window_sessions": WINDOW_SESSIONS,
                "defined": True, "n_sessions_used": len(used),
                "window_first_session": int(ids[0]), "window_last_session": int(ids[-1]),
                "window_session_ids": ",".join(map(str, ids)),
                "window_aborts": numerator, "window_eligible_trials": denominator,
                "fraction": numerator / denominator,
                "raw_aborts": int(target["aborts"]), "raw_eligible_trials": int(target["eligible_trials"]),
                "raw_fraction": target["fraction"],
            })
    for target in condition.loc[~condition["defined"]].itertuples(index=False):
        window_records.append({
            "metric": metric, "animal": int(animal), "session_type": int(session_type),
            "LED_trial": int(led), "session": int(target.session),
            "relative_session": int(target.relative_session), "window_sessions": WINDOW_SESSIONS,
            "defined": False, "n_sessions_used": 0, "window_first_session": np.nan,
            "window_last_session": np.nan, "window_session_ids": "",
            "window_aborts": 0, "window_eligible_trials": 0, "fraction": np.nan,
            "raw_aborts": int(target.aborts), "raw_eligible_trials": 0, "raw_fraction": np.nan,
        })

windows = pd.DataFrame(window_records)
assert len(windows) == len(session_audit) and not windows.duplicated(metric_keys).any()
assert windows["fraction"].isna().equals(~windows["defined"])
session_lookup = session_audit.set_index(metric_keys)
for row in windows.itertuples(index=False):
    target = session_lookup.loc[(row.metric, row.animal, row.session_type, row.session, row.LED_trial)]
    if not row.defined:
        assert target.eligible_trials == 0 and row.n_sessions_used == 0
        continue
    # Independent ID lookup checks both ordinary gaps and zero-denominator
    # breaks, so neighboring windows cannot borrow across undefined targets.
    expected_ids = [row.session]
    for direction in (-1, 1):
        for distance in range(1, half_window + 1):
            candidate = row.session + direction * distance
            key = (row.metric, row.animal, row.session_type, candidate, row.LED_trial)
            if key not in session_lookup.index or not session_lookup.loc[key, "defined"]:
                break
            expected_ids.append(candidate)
    expected_ids.sort()
    assert expected_ids == [int(value) for value in row.window_session_ids.split(",")]
    assert len(expected_ids) == row.n_sessions_used <= WINDOW_SESSIONS
    members = session_lookup.loc[[(row.metric, row.animal, row.session_type, s, row.LED_trial) for s in expected_ids]]
    assert int(members["aborts"].sum()) == row.window_aborts
    assert int(members["eligible_trials"].sum()) == row.window_eligible_trials
    assert row.fraction == row.window_aborts / row.window_eligible_trials
    assert row.raw_fraction == target.fraction


# %%
############ Equal-animal mean/SEM, excluding undefined fractions ############
mean_records = []
for (metric, session_type, relative_session, led), group in windows.groupby(["metric", *mean_keys], sort=True):
    defined = group.loc[group["defined"]]
    values = defined["fraction"].to_numpy(dtype=float)
    n = len(values)
    assert n == defined["animal"].nunique()
    mean = float(values.mean()) if n else np.nan
    sem = float(values.std(ddof=1) / np.sqrt(n)) if n >= 2 else np.nan
    if n:
        vector = group.set_index("animal")["fraction"].reindex(ANIMALS).to_numpy()
        assert np.isclose(mean, np.nanmean(vector), rtol=0, atol=1e-14)
    if n >= 2:
        assert np.isclose(sem, np.sqrt(np.sum((values - mean) ** 2) / (n * (n - 1))), rtol=0, atol=1e-14)
    mean_records.append({
        "metric": metric, "session_type": int(session_type), "relative_session": int(relative_session),
        "LED_trial": int(led), "window_sessions": WINDOW_SESSIONS,
        "n_animals": n, "contributing_animals": ",".join(map(str, sorted(defined["animal"].astype(int)))),
        "mean_fraction": mean, "sem_fraction": sem,
        # Diagnostic totals only: the plotted mean is not their ratio.
        "sum_window_aborts": int(defined["window_aborts"].sum()),
        "sum_window_eligible_trials": int(defined["window_eligible_trials"].sum()),
    })
means = pd.DataFrame(mean_records)
assert not means.duplicated(["metric", *mean_keys]).any()
assert means.loc[means["n_animals"].lt(2), "sem_fraction"].isna().all()
for metric, limits in zip(METRICS, Y_LIMITS):
    raw = session_audit.loc[session_audit["metric"].eq(metric), "fraction"].dropna()
    rolling = windows.loc[windows["metric"].eq(metric), "fraction"].dropna()
    summary = means.loc[means["metric"].eq(metric)]
    assert raw.between(*limits).all() and rolling.between(*limits).all()
    assert (summary["mean_fraction"] + summary["sem_fraction"].fillna(0)).dropna().le(limits[1]).all()


# %%
############ Exact regression checks: OFF columns identical, isolated column unchanged ############
for table, keys, fields in (
    (session_audit, condition_keys, ["aborts", "eligible_trials", "fraction"]),
    (windows, condition_keys, ["window_aborts", "window_eligible_trials", "fraction", "n_sessions_used"]),
    (means, mean_keys, ["mean_fraction", "sem_fraction", "n_animals"]),
):
    off = table.loc[table["LED_trial"].eq(0)]
    left = off.loc[off["metric"].eq(METRICS[0])].set_index(keys).sort_index()
    right = off.loc[off["metric"].eq(METRICS[1])].set_index(keys).sort_index()
    assert left.index.equals(right.index)
    np.testing.assert_allclose(left[fields], right[fields], rtol=0, atol=0, equal_nan=True)

if WINDOW_SESSIONS == 3:
    # These prior CSVs are reference inputs; do not rerun the older plotting
    # script or overwrite its figures to perform this comparison.
    comparisons = (
        (session_audit, "session_audit", condition_keys,
         {"aborts": "isolated_count", "eligible_trials": "all_condition_trials", "fraction": "isolated_fraction"}),
        (windows, "animal_windows_audit", condition_keys,
         {"window_aborts": "isolated_count", "window_eligible_trials": "window_condition_trials",
          "fraction": "isolated_fraction", "raw_fraction": "raw_isolated_fraction",
          "n_sessions_used": "n_sessions_used", "window_first_session": "window_first_session",
          "window_last_session": "window_last_session"}),
        (means, "equal_animal_mean_sem_audit", mean_keys,
         {"mean_fraction": "mean_isolated_fraction", "sem_fraction": "sem_isolated_fraction", "n_animals": "n_animals"}),
    )
    for table, suffix, keys, fields in comparisons:
        reference = pd.read_csv(SCRIPT_DIR / f"{REFERENCE_STEM}_{suffix}.csv", float_precision="round_trip")
        old = reference.set_index(keys).sort_index()
        new = table.loc[table["metric"].eq("isolated")].set_index(keys).sort_index()
        assert new.index.equals(old.index)
        np.testing.assert_allclose(new[list(fields)], old[list(fields.values())], rtol=0, atol=0, equal_nan=True)
    print("Exact regression checks passed: OFF columns 1/2 identical; isolated session/rolling/mean/SEM values unchanged.", flush=True)

for table, suffix in ((session_audit, "session_audit"), (windows, "animal_windows_audit"), (means, "equal_animal_mean_sem_audit")):
    path = SCRIPT_DIR / f"{OUTPUT_STEM}_{suffix}.csv"
    table.to_csv(path, index=False)
    print(f"Saved {path} ({len(table):,} rows)", flush=True)
print("\nNumerators and eligible denominators before rolling:")
print(session_audit.groupby(["session_type", "metric", "LED_trial"])[["aborts", "eligible_trials"]].sum().to_string())
print("\nUndefined groups (retained as gaps):")
print(session_audit.loc[~session_audit["defined"], metric_keys + ["all_condition_trials", "eligible_trials"]].to_string(index=False))


# %%
############ Plot conditional fractions and unchanged isolated fractions ############
relative_min, relative_max = int(session_audit["relative_session"].min()), int(session_audit["relative_session"].max())
fig, axes = plt.subplots(7, 3, figsize=(17.0, 21.5), sharex=True)
for row_index, animal in enumerate(ANIMALS):
    for column, metric in enumerate(METRICS):
        ax = axes[row_index, column]
        for session_type in SESSION_TYPES:
            for led in (0, 1):
                panel = windows.loc[
                    windows["animal"].eq(animal) & windows["metric"].eq(metric)
                    & windows["session_type"].eq(session_type) & windows["LED_trial"].eq(led)
                ].sort_values("relative_session")
                x = panel["relative_session"].to_numpy(dtype=int)
                ax.scatter(x, panel["raw_fraction"], s=14, color=LED_COLORS[led], alpha=0.22, zorder=2)
                grid = np.arange(x[0], x[-1] + 1)
                values = np.full(len(grid), np.nan)
                values[x - x[0]] = panel["fraction"].to_numpy()
                assert np.isfinite(values).sum() == int(panel["defined"].sum())
                ax.plot(grid, values, color=LED_COLORS[led], linestyle=TYPE_STYLES[session_type], linewidth=1.7, zorder=3)
        if column == 0:
            last_s7, first_s8 = transitions[animal]
            ax.set_ylabel(f"Animal {animal}\ns7 #{last_s7} → s8 #{first_s8}\nConditional abort fraction", fontsize=9.5)

for column, metric in enumerate(METRICS):
    ax = axes[-1, column]
    for session_type in SESSION_TYPES:
        for led in (0, 1):
            panel = means.loc[means["metric"].eq(metric) & means["session_type"].eq(session_type) & means["LED_trial"].eq(led)].sort_values("relative_session")
            x = panel["relative_session"].to_numpy(dtype=int)
            grid = np.arange(x[0], x[-1] + 1)
            y, sem = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
            y[x - x[0]] = panel["mean_fraction"].to_numpy()
            sem[x - x[0]] = panel["sem_fraction"].to_numpy()
            ax.fill_between(grid, np.maximum(0, y - sem), y + sem, color=LED_COLORS[led], alpha=0.15, linewidth=0, zorder=2)
            ax.plot(grid, y, color=LED_COLORS[led], linestyle=TYPE_STYLES[session_type], linewidth=2.1, zorder=3)
            single = panel.loc[panel["n_animals"].eq(1)]
            ax.scatter(single["relative_session"], single["mean_fraction"], facecolors="white", edgecolors=LED_COLORS[led], s=38, linewidths=1.1, zorder=4)
    ax.set_xlabel("Sessions from first type-8 session (0)", fontsize=10)
    if column == 0:
        ax.set_ylabel("Equal-animal mean\n± SEM", fontsize=10)

for column, (title, limits) in enumerate(zip(TITLES, Y_LIMITS)):
    axes[0, column].set_title(title, fontsize=11.5, pad=9)
    for row_index, ax in enumerate(axes[:, column]):
        ax.axvline(-0.5, color="0.4", linestyle=":", linewidth=1, zorder=1)
        ax.set_xlim(relative_min - 1, relative_max + 1)
        ax.set_ylim(*(MEAN_Y_LIMITS if row_index == len(ANIMALS) else limits))
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(0.1 if column < 2 and row_index < len(ANIMALS) else 0.05))
        ax.tick_params(labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.90", linewidth=0.6)
fig.suptitle(f"LED7 conditional abort fractions across sessions · {WINDOW_SESSIONS}-session rolling", fontsize=16, y=0.992)
fig.legend(
    handles=[
        Line2D([0], [0], color=LED_COLORS[0], lw=1.8, label="Current trial OFF"),
        Line2D([0], [0], color=LED_COLORS[1], lw=1.8, label="Current trial ON"),
        Line2D([0], [0], color="0.35", linestyle=TYPE_STYLES[7], lw=1.8, label="Type 7"),
        Line2D([0], [0], color="0.35", linestyle=TYPE_STYLES[8], lw=1.8, label="Type 8"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.978), ncol=4, frameon=False, fontsize=11,
)
fig.text(
    0.5, 0.011,
    "Faint dots: raw session fractions. Curves: summed aborts / summed eligible trials; centered windows, one-session stride.\n"
    "Columns 1–2 condition on an adjacent previous abort (actual trial IDs differ by 1); column 2 also requires the same LED flag.\n"
    "Column 3 retains isolated aborts / all condition trials. Windows do not cross type switches, session gaps, or zero-denominator groups.\n"
    "Bottom row: equal-animal mean ± SEM; hollow points indicate one animal. Zero marks the first type-8 session.",
    ha="center", fontsize=8.8, linespacing=1.4,
)
fig.tight_layout(rect=(0.02, 0.055, 0.99, 0.949), h_pad=1.6, w_pad=0.9)
fig.savefig(FIGURE_PATH, dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure: {FIGURE_PATH}", flush=True)
print("All source, adjacency, conditional-denominator, rolling, SEM, and regression checks passed.", flush=True)
