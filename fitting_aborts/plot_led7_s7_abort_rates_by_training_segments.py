# %%
"""LED7 session-type-7/8 fixation-abort rates across equal-trial training segments."""

from pathlib import Path
import os


# %%
############ Editable parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
SESSION_TYPE = int(os.environ.get("LED7_SESSION_TYPE", "7"))
assert SESSION_TYPE in (7, 8)
BIN_WIDTH_MS = int(os.environ.get("LED7_BIN_WIDTH_MS", "40"))
N_GROUPS = tuple(int(value) for value in os.environ.get("LED7_N_GROUPS", "3,4,5").split(","))
INCLUDE_POOLED_ROW = os.environ.get("LED7_INCLUDE_POOLED_ROW", "0") == "1"
assert BIN_WIDTH_MS in (20, 40)
assert N_GROUPS and len(set(N_GROUPS)) == len(N_GROUPS) and set(N_GROUPS).issubset({3, 4, 5})
SOURCE_CSV = SCRIPT_DIR.parent / "raw_data" / f"LED7_s{SESSION_TYPE}.csv"
OUTPUT_PREFIX = f"led7_s{SESSION_TYPE}_abort_rates_by_training_segments_{BIN_WIDTH_MS}ms_with_all"
if INCLUDE_POOLED_ROW:
    OUTPUT_PREFIX += "_and_pooled"
AUDIT_CSV = SCRIPT_DIR / f"{OUTPUT_PREFIX}_audit.csv"

ANIMALS = (92, 93, 98, 99, 100, 103)
BIN_WIDTH_S = BIN_WIDTH_MS / 1000
FULL_RANGE_S = (-2.0, 2.0)
DISPLAY_RANGE_S = (-0.4, 0.5)
FIGURE_DPI = 220
LED_COLORS = {0: "tab:blue", 1: "tab:red"}

EXPECTED_SOURCE_ROWS = {7: 138_440, 8: 148_428}[SESSION_TYPE]
EXPECTED_ANIMAL_ROWS = {7: 100_251, 8: 113_941}[SESSION_TYPE]
EXPECTED_LED_COUNTS = {
    7: {0: 66_226, 1: 34_025},
    8: {0: 70_892, 1: 43_049},
}[SESSION_TYPE]
EXPECTED_CODED_ABORTS = {7: 16_389, 8: 17_725}[SESSION_TYPE]
EXPECTED_MISSING_ABORT_TIMES = 2

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

bins = np.linspace(
    FULL_RANGE_S[0], FULL_RANGE_S[1],
    round((FULL_RANGE_S[1] - FULL_RANGE_S[0]) / BIN_WIDTH_S) + 1,
)
bin_centers = (bins[:-1] + bins[1:]) / 2
visible_bins = (bin_centers >= DISPLAY_RANGE_S[0]) & (bin_centers <= DISPLAY_RANGE_S[1])


# %%
############ Read the standardized, fixation-referenced session CSV ############
source = pd.read_csv(SOURCE_CSV, float_precision="round_trip")
assert len(source) == EXPECTED_SOURCE_ROWS and len(source.columns) == 52
assert source.columns.tolist().count("LED_onset_time") == 1
assert source["training_level"].eq(16).all()
assert source["session_type"].eq(SESSION_TYPE).all()
assert (source["repeat_trial"].isin([0, 2]) | source["repeat_trial"].isna()).all()
assert source["LED_trial"].isin([0, 1]).all()
assert not source.duplicated(["animal", "session", "trial"]).any()

data = source.loc[source["animal"].isin(ANIMALS)].copy()
assert len(data) == EXPECTED_ANIMAL_ROWS
assert tuple(sorted(data["animal"].astype(int).unique())) == ANIMALS
assert data["LED_trial"].value_counts().to_dict() == EXPECTED_LED_COUNTS
assert np.isfinite(data[["session", "trial", "intended_fix", "LED_onset_time"]]).all().all()
assert np.equal(data[["session", "trial"]], np.floor(data[["session", "trial"]])).all().all()
assert data["LED_onset_time"].ge(0).all()
assert data["LED_onset_time"].le(data["intended_fix"]).all()

is_abort = data["abort_event"].eq(3)
has_abort_time = np.isfinite(data["timed_fix"])
assert int(is_abort.sum()) == EXPECTED_CODED_ABORTS
assert int((is_abort & ~has_abort_time).sum()) == EXPECTED_MISSING_ABORT_TIMES
assert data.loc[is_abort & has_abort_time, "timed_fix"].lt(
    data.loc[is_abort & has_abort_time, "intended_fix"]
).all()
print(
    f"LED7 s{SESSION_TYPE}: {len(data):,} trials in animals {ANIMALS}; "
    f"{int(is_abort.sum()):,} coded fixation aborts, "
    f"{int((is_abort & ~has_abort_time).sum())} missing abort times",
    flush=True,
)


# %%
############ Divide each animal's session/trial sequence into equal-trial groups ############
rates = {}
panels = {}
audit_rows = []
for animal in ANIMALS:
    ordered = (
        data.loc[data["animal"].eq(animal)]
        .sort_values(["session", "trial"], kind="stable")
        .reset_index(drop=True)
    )
    assert not ordered.empty

    for n_groups in N_GROUPS:
        # np.array_split gives consecutive groups, with at most one extra trial
        # in earlier groups. A boundary can fall inside an actual session.
        positions = np.array_split(np.arange(len(ordered)), n_groups)
        sizes = [len(part) for part in positions]
        assert min(sizes) > 0 and max(sizes) - min(sizes) <= 1
        assert np.array_equal(np.concatenate(positions), np.arange(len(ordered)))

        # The final panel repeats the same calculation over every session for
        # this animal, so it can be compared directly with the split panels.
        for group_index, part in enumerate([*positions, np.arange(len(ordered))], start=1):
            is_all_sessions = group_index == n_groups + 1
            group = ordered.iloc[part]
            first, last = group.iloc[0], group.iloc[-1]
            panels[n_groups, animal, group_index] = {
                "n_trials": len(group),
                "first_session": int(first["session"]),
                "first_trial": int(first["trial"]),
                "last_session": int(last["session"]),
                "last_trial": int(last["trial"]),
                "starts_mid_session": not is_all_sessions and group_index > 1
                and first["session"] == ordered.iloc[positions[group_index - 2][-1]]["session"],
                "ends_mid_session": not is_all_sessions and group_index < n_groups
                and last["session"] == ordered.iloc[positions[group_index][0]]["session"],
            }

            for led in (0, 1):
                condition = group.loc[group["LED_trial"].eq(led)]
                coded = condition.loc[condition["abort_event"].eq(3)]
                finite = coded.loc[np.isfinite(coded["timed_fix"])]
                assert len(condition) > 0

                # LED_onset_time is already measured from fixation t=0 in this CSV.
                # For LED OFF it is a scheduled/counterfactual onset.
                abort_times = (finite["timed_fix"] - finite["LED_onset_time"]).to_numpy()
                counts, _ = np.histogram(abort_times, bins=bins)
                assert int(counts.sum()) == len(finite), "Full bins omitted finite aborts."
                heights = counts / (len(condition) * BIN_WIDTH_S)
                area = float(np.sum(heights * np.diff(bins)))
                assert np.isclose(area, len(finite) / len(condition), rtol=0, atol=1e-14)
                rates[n_groups, animal, group_index, led] = heights

                audit_rows.append({
                    "n_groups": n_groups,
                    "animal": animal,
                    "group": group_index,
                    **panels[n_groups, animal, group_index],
                    "LED_trial": led,
                    "all_condition_trials": len(condition),
                    "coded_event3_aborts": len(coded),
                    "finite_plotted_aborts": len(finite),
                    "missing_timed_fix": len(coded) - len(finite),
                    "full_histogram_area": area,
                    "finite_abort_fraction": len(finite) / len(condition),
                    "aborts_outside_display": int(
                        ((abort_times < DISPLAY_RANGE_S[0])
                         | (abort_times > DISPLAY_RANGE_S[1])).sum()
                    ),
                })

audit = pd.DataFrame(audit_rows)
for n_groups in N_GROUPS:
    selection = audit.loc[audit["n_groups"].eq(n_groups) & audit["group"].le(n_groups)]
    whole = audit.loc[audit["n_groups"].eq(n_groups) & audit["group"].eq(n_groups + 1)]
    assert selection["all_condition_trials"].sum() == len(data)
    assert selection["coded_event3_aborts"].sum() == EXPECTED_CODED_ABORTS
    assert selection["missing_timed_fix"].sum() == EXPECTED_MISSING_ABORT_TIMES
    assert selection.groupby("LED_trial")["all_condition_trials"].sum().to_dict() == EXPECTED_LED_COUNTS
    assert whole["all_condition_trials"].sum() == len(data)
    assert whole["coded_event3_aborts"].sum() == EXPECTED_CODED_ABORTS
    assert whole["missing_timed_fix"].sum() == EXPECTED_MISSING_ABORT_TIMES
    for animal in ANIMALS:
        rows = selection.loc[selection["animal"].eq(animal)]
        assert rows.groupby("group")["n_trials"].first().sum() == int(data["animal"].eq(animal).sum())
        assert rows["all_condition_trials"].sum() == int(data["animal"].eq(animal).sum())
        for led in (0, 1):
            child = rows.loc[rows["LED_trial"].eq(led)]
            all_row = whole.loc[whole["animal"].eq(animal) & whole["LED_trial"].eq(led)].iloc[0]
            for field in ("all_condition_trials", "coded_event3_aborts", "finite_plotted_aborts", "missing_timed_fix"):
                assert child[field].sum() == all_row[field]
            weighted_rate = sum(
                rates[n_groups, animal, group_index, led]
                * int(child.loc[child["group"].eq(group_index), "all_condition_trials"].iloc[0])
                for group_index in range(1, n_groups + 1)
            ) / all_row["all_condition_trials"]
            np.testing.assert_allclose(
                weighted_rate, rates[n_groups, animal, n_groups + 1, led],
                rtol=0, atol=1e-14,
            )

if INCLUDE_POOLED_ROW:
    # Pool the *same* chronological group from each animal. Do not repartition
    # all six animals' overlapping session IDs into a new global sequence.
    pooled_rows = []
    for n_groups in N_GROUPS:
        for group_index in range(1, n_groups + 2):
            child = audit.loc[
                audit["n_groups"].eq(n_groups) & audit["group"].eq(group_index)
            ]
            n_trials = int(child.groupby("animal")["n_trials"].first().sum())
            panels[n_groups, "aggregate", group_index] = {
                "n_trials": n_trials,
                "first_session": np.nan,
                "first_trial": np.nan,
                "last_session": np.nan,
                "last_trial": np.nan,
                "starts_mid_session": np.nan,
                "ends_mid_session": np.nan,
            }
            for led in (0, 1):
                condition_rows = child.loc[child["LED_trial"].eq(led)]
                denominator = int(condition_rows["all_condition_trials"].sum())
                pooled_counts = sum(
                    rates[n_groups, animal, group_index, led]
                    * int(condition_rows.loc[
                        condition_rows["animal"].eq(animal), "all_condition_trials"
                    ].iloc[0])
                    * BIN_WIDTH_S
                    for animal in ANIMALS
                )
                heights = pooled_counts / (denominator * BIN_WIDTH_S)
                rates[n_groups, "aggregate", group_index, led] = heights
                finite_count = int(condition_rows["finite_plotted_aborts"].sum())
                area = float(np.sum(heights * np.diff(bins)))
                assert np.isclose(area, finite_count / denominator, rtol=0, atol=1e-14)
                pooled_rows.append({
                    "n_groups": n_groups,
                    "animal": "aggregate",
                    "group": group_index,
                    **panels[n_groups, "aggregate", group_index],
                    "LED_trial": led,
                    "all_condition_trials": denominator,
                    "coded_event3_aborts": int(condition_rows["coded_event3_aborts"].sum()),
                    "finite_plotted_aborts": finite_count,
                    "missing_timed_fix": int(condition_rows["missing_timed_fix"].sum()),
                    "full_histogram_area": area,
                    "finite_abort_fraction": finite_count / denominator,
                    "aborts_outside_display": int(condition_rows["aborts_outside_display"].sum()),
                })
            assert sum(row["all_condition_trials"] for row in pooled_rows[-2:]) == n_trials

        group_sizes = [
            panels[n_groups, "aggregate", group_index]["n_trials"]
            for group_index in range(1, n_groups + 1)
        ]
        assert sum(group_sizes) == EXPECTED_ANIMAL_ROWS
        assert max(group_sizes) - min(group_sizes) <= len(ANIMALS)
        assert panels[n_groups, "aggregate", n_groups + 1]["n_trials"] == EXPECTED_ANIMAL_ROWS
        print(f"s{SESSION_TYPE}, {n_groups} pooled training groups: {group_sizes}", flush=True)

    audit = pd.concat([audit, pd.DataFrame(pooled_rows)], ignore_index=True)

audit.to_csv(AUDIT_CSV, index=False)
print(f"Audit: {AUDIT_CSV}", flush=True)


# %%
############ Plot each group count with common axes for training comparisons ############
y_max = max(float(height[visible_bins].max()) for height in rates.values()) * 1.12
for n_groups in N_GROUPS:
    plot_rows = (*ANIMALS, "aggregate") if INCLUDE_POOLED_ROW else ANIMALS
    fig, axes = plt.subplots(
        len(plot_rows), n_groups + 1,
        figsize=(4.0 * (n_groups + 1) + 1.0, 16.5 + 2.4 * INCLUDE_POOLED_ROW),
        sharex=True, sharey=True,
    )
    for row, animal in enumerate(plot_rows):
        for col in range(n_groups + 1):
            group_index = col + 1
            ax = axes[row, col]
            if col == n_groups:
                ax.set_facecolor("#f7f7f7")
            info = panels[n_groups, animal, group_index]
            panel_audit = audit.loc[
                audit["n_groups"].eq(n_groups)
                & audit["animal"].eq(animal)
                & audit["group"].eq(group_index)
            ].set_index("LED_trial")

            for led in (0, 1):
                ax.stairs(
                    rates[n_groups, animal, group_index, led], bins,
                    color=LED_COLORS[led], linewidth=1.3,
                )
            ax.axvline(0, color="0.4", linestyle="--", linewidth=0.9, zorder=0)
            ax.set_xlim(DISPLAY_RANGE_S)
            ax.set_ylim(0, y_max)
            ax.grid(axis="y", color="0.9", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            column_title = "All sessions" if col == n_groups else f"Group {group_index}/{n_groups}"
            if animal == "aggregate":
                ax.set_title(
                    f"{column_title} · pooled n = {info['n_trials']:,}\nSix animals",
                    fontsize=9, pad=6,
                )
            else:
                ax.set_title(
                    f"{column_title} · n = {info['n_trials']:,}\n"
                    f"S{info['first_session']}/T{info['first_trial']} → "
                    f"S{info['last_session']}/T{info['last_trial']}",
                    fontsize=9, pad=6,
                )
            ax.text(
                0.03, 0.96,
                f"OFF {panel_audit.loc[0, 'finite_abort_fraction']:.1%}  "
                f"ON {panel_audit.loc[1, 'finite_abort_fraction']:.1%}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
            )
            if col == 0:
                row_label = "Aggregate" if animal == "aggregate" else f"Animal {animal}"
                ax.set_ylabel(f"{row_label}\nAbort rate (s$^{{-1}}$)", fontsize=9)
            if row == len(plot_rows) - 1:
                ax.set_xlabel("Time from LED onset (s)", fontsize=9)

    group_name = {3: "terciles", 4: "quartiles", 5: "quintiles"}[n_groups]
    fig.suptitle(
        f"LED7 session type {SESSION_TYPE}: fixation-abort rate across training ({group_name} + all sessions)",
        fontsize=15, y=0.995,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=LED_COLORS[0], lw=1.5, label="LED OFF"),
            Line2D([0], [0], color=LED_COLORS[1], lw=1.5, label="LED ON"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.973), ncol=2, frameon=False,
    )
    note = (
        "Each animal is split into consecutive, nearly equal trial counts by session/trial; "
        "the rightmost column pools all sessions.\n"
    )
    if INCLUDE_POOLED_ROW:
        note += "The bottom row pools corresponding animal groups by trial count. "
    note += (
        "Percentages are finite event-3 aborts / all condition trials; "
        f"{BIN_WIDTH_S * 1000:g} ms bins, full range −2 to +2 s. "
        "LED OFF onset is scheduled/counterfactual."
    )
    fig.text(0.5, 0.013, note, ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.945), h_pad=1.6, w_pad=0.7)
    figure_path = SCRIPT_DIR / f"{OUTPUT_PREFIX}_{len(plot_rows)}x{n_groups + 1}.png"
    fig.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {figure_path}", flush=True)

print(
    audit[["n_groups", "animal", "group", "n_trials", "first_session", "first_trial",
           "last_session", "last_trial", "LED_trial", "all_condition_trials",
           "coded_event3_aborts", "finite_plotted_aborts", "missing_timed_fix",
           "full_histogram_area"]].to_string(index=False)
)
