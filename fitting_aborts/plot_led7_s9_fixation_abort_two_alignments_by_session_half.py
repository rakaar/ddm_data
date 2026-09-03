# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
############ Parameters (edit here) ############
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

INPUT_CSV_PATH = REPO_ROOT / "raw_data" / "LED7_s9.csv"
FIRST_HALF_OUTPUT_PATH = (
    SCRIPT_DIR
    / "led7_s9_fixation_abort_two_alignments_within_animal_first_half_2x9.png"
)
SECOND_HALF_OUTPUT_PATH = (
    SCRIPT_DIR
    / "led7_s9_fixation_abort_two_alignments_within_animal_second_half_2x9.png"
)

TRAINING_LEVEL = 16
SESSION_TYPE = 9
ALLOWED_REPEAT_TRIALS = {0, 2}
EXPECTED_ANIMALS = [90, 92, 93, 98, 99, 100, 102, 103]
EXPECTED_SESSION_NUMBERS = list(range(73, 129))

EXPECTED_ROWS = 100_788
EXPECTED_LED_TRIAL_COUNTS = {0: 61_408, 1: 39_380}
EXPECTED_ABORT_EVENT_3_COUNTS = {0: 10_108, 1: 8_752}
EXPECTED_FINITE_ABORT_COUNTS = {0: 10_107, 1: 8_747}
EXPECTED_ANIMAL_SESSION_PAIR_COUNTS = {"first_half": 97, "second_half": 94}

BIN_WIDTH_S = 0.020
FULL_MIN_S = -1.1
FULL_MAX_S = 1.1
DISPLAY_MIN_S = -1.0
DISPLAY_MAX_S = 1.0
FULL_BINS = np.arange(
    FULL_MIN_S,
    FULL_MAX_S + BIN_WIDTH_S / 2,
    BIN_WIDTH_S,
)

LED_COLORS = {0: "tab:blue", 1: "tab:red"}
LED_LABELS = {0: "LED OFF", 1: "LED ON"}

OUTPUT_DPI = 250
SHOW_PLOTS = False


# %%
############ Load and validate the session-9 export ############
if not INPUT_CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find input CSV: {INPUT_CSV_PATH}")

df = pd.read_csv(INPUT_CSV_PATH)

required_columns = [
    "animal",
    "session",
    "training_level",
    "session_type",
    "repeat_trial",
    "LED_trial",
    "abort_event",
    "timed_fix",
    "intended_fix",
    "LED_onset_time",
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in {INPUT_CSV_PATH}: {missing_columns}")

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(f"Expected {EXPECTED_ROWS:,} rows, found {len(df):,}.")
if not df["training_level"].eq(TRAINING_LEVEL).all():
    raise RuntimeError(f"Input contains rows outside training_level == {TRAINING_LEVEL}.")
if not df["session_type"].eq(SESSION_TYPE).all():
    raise RuntimeError(f"Input contains rows outside session_type == {SESSION_TYPE}.")

repeat_values = set(df["repeat_trial"].dropna().astype(int).unique())
if not repeat_values.issubset(ALLOWED_REPEAT_TRIALS):
    raise RuntimeError(f"Unexpected repeat_trial values: {sorted(repeat_values)}")
if df[["animal", "session", "LED_trial"]].isna().any(axis=None):
    raise RuntimeError("animal, session, or LED_trial contains missing values.")

df["animal"] = df["animal"].astype(int)
df["session"] = df["session"].astype(int)
df["LED_trial"] = df["LED_trial"].astype(int)

animals = sorted(df["animal"].unique().tolist())
if animals != EXPECTED_ANIMALS:
    raise RuntimeError(f"Expected animals {EXPECTED_ANIMALS}, found {animals}.")

session_numbers = sorted(df["session"].unique().tolist())
if session_numbers != EXPECTED_SESSION_NUMBERS:
    raise RuntimeError(
        f"Expected session numbers {EXPECTED_SESSION_NUMBERS}, found {session_numbers}."
    )

led_values = set(df["LED_trial"].unique())
if led_values != {0, 1}:
    raise RuntimeError(f"Expected LED_trial values {{0, 1}}, found {sorted(led_values)}.")
led_trial_counts = df["LED_trial"].value_counts().sort_index().to_dict()
if led_trial_counts != EXPECTED_LED_TRIAL_COUNTS:
    raise RuntimeError(
        f"Expected LED trial counts {EXPECTED_LED_TRIAL_COUNTS}, found {led_trial_counts}."
    )


# %%
############ Split each animal's ordered sessions into its own halves ############
df["within_animal_session_half"] = ""
animal_half_sessions = {}
split_rows = []

for animal in animals:
    animal_sessions = sorted(
        df.loc[df["animal"].eq(animal), "session"].unique().tolist()
    )
    expected_contiguous_sessions = list(
        range(animal_sessions[0], animal_sessions[-1] + 1)
    )
    if animal_sessions != expected_contiguous_sessions:
        missing_sessions = sorted(
            set(expected_contiguous_sessions) - set(animal_sessions)
        )
        raise RuntimeError(
            f"Animal {animal} does not have a contiguous session sequence; "
            f"missing {missing_sessions}."
        )

    # Keep every session. For an odd count, the first half receives the midpoint.
    first_half_count = (len(animal_sessions) + 1) // 2
    first_half_sessions = animal_sessions[:first_half_count]
    second_half_sessions = animal_sessions[first_half_count:]
    if not first_half_sessions or not second_half_sessions:
        raise RuntimeError(f"Animal {animal} cannot be split into two nonempty halves.")

    animal_half_sessions[animal] = {
        "first_half": first_half_sessions,
        "second_half": second_half_sessions,
    }

    for half_key, half_sessions in animal_half_sessions[animal].items():
        row_mask = df["animal"].eq(animal) & df["session"].isin(half_sessions)
        df.loc[row_mask, "within_animal_session_half"] = half_key

    split_rows.append(
        {
            "animal": animal,
            "all_sessions": (
                f"{animal_sessions[0]}-{animal_sessions[-1]}"
            ),
            "n_all_sessions": len(animal_sessions),
            "first_half_sessions": (
                f"{first_half_sessions[0]}-{first_half_sessions[-1]}"
            ),
            "n_first_half_sessions": len(first_half_sessions),
            "second_half_sessions": (
                f"{second_half_sessions[0]}-{second_half_sessions[-1]}"
            ),
            "n_second_half_sessions": len(second_half_sessions),
        }
    )

split_df = pd.DataFrame(split_rows)

if df["within_animal_session_half"].eq("").any():
    raise RuntimeError("Some rows were not assigned to a within-animal session half.")
if not (
    split_df["n_first_half_sessions"] + split_df["n_second_half_sessions"]
).eq(split_df["n_all_sessions"]).all():
    raise RuntimeError("A within-animal split dropped or duplicated a session.")

half_definitions = [
    {
        "key": "first_half",
        "label": "First half",
        "output_path": FIRST_HALF_OUTPUT_PATH,
    },
    {
        "key": "second_half",
        "label": "Second half",
        "output_path": SECOND_HALF_OUTPUT_PATH,
    },
]

half_row_counts = df["within_animal_session_half"].value_counts().astype(int).to_dict()
if sum(half_row_counts.values()) != len(df):
    raise RuntimeError("The within-animal halves do not partition all rows.")

for half in half_definitions:
    half_df = df.loc[df["within_animal_session_half"].eq(half["key"])]
    observed_pair_count = half_df[["animal", "session"]].drop_duplicates().shape[0]
    expected_pair_count = EXPECTED_ANIMAL_SESSION_PAIR_COUNTS[half["key"]]
    if observed_pair_count != expected_pair_count:
        raise RuntimeError(
            f"Expected {expected_pair_count} animal-session pairs in "
            f"{half['label'].lower()}, found {observed_pair_count}."
        )
    if sorted(half_df["animal"].unique().tolist()) != animals:
        raise RuntimeError(f"Not every animal contributes to {half['label'].lower()}.")


# %%
############ Build aligned histograms for both within-animal halves ############
def abort_rate_histogram(values, n_total_trials):
    counts, _ = np.histogram(values, bins=FULL_BINS)
    heights = counts.astype(float) / (n_total_trials * BIN_WIDTH_S)
    return counts, heights


panel_results = {}
audit_rows = []
global_y_max = 0.0

for half in half_definitions:
    half_df = df.loc[df["within_animal_session_half"].eq(half["key"])].copy()
    group_definitions = [
        (f"Animal {animal}", half_df.loc[half_df["animal"].eq(animal)].copy())
        for animal in animals
    ]
    group_definitions.append(("Aggregate", half_df.copy()))

    for group_label, group_df in group_definitions:
        for led_trial in [0, 1]:
            condition_df = group_df.loc[group_df["LED_trial"].eq(led_trial)].copy()
            if condition_df.empty:
                raise RuntimeError(
                    f"{half['label']} {group_label} has no {LED_LABELS[led_trial]} trials."
                )

            coded_abort_df = condition_df.loc[
                condition_df["abort_event"].eq(3)
            ].copy()
            plotted_abort_df = coded_abort_df.loc[
                np.isfinite(
                    coded_abort_df[["timed_fix", "intended_fix", "LED_onset_time"]]
                ).all(axis=1)
            ].copy()

            n_total_trials = len(condition_df)
            n_coded_aborts = len(coded_abort_df)
            n_plotted_aborts = len(plotted_abort_df)

            aligned_to_led_onset = (
                plotted_abort_df["timed_fix"]
                - plotted_abort_df["LED_onset_time"]
            ).to_numpy(dtype=float)
            aligned_to_intended_minus_onset = (
                plotted_abort_df["timed_fix"]
                - (
                    plotted_abort_df["intended_fix"]
                    - plotted_abort_df["LED_onset_time"]
                )
            ).to_numpy(dtype=float)

            for alignment_label, values in [
                ("LED onset", aligned_to_led_onset),
                ("intended_fix - LED_onset_time", aligned_to_intended_minus_onset),
            ]:
                if not ((values >= FULL_BINS[0]) & (values <= FULL_BINS[-1])).all():
                    raise RuntimeError(
                        f"{half['label']} {group_label} {LED_LABELS[led_trial]} "
                        f"has {alignment_label} values outside "
                        f"[{FULL_BINS[0]}, {FULL_BINS[-1]}] s."
                    )

            led_onset_counts, led_onset_heights = abort_rate_histogram(
                aligned_to_led_onset,
                n_total_trials,
            )
            intended_minus_onset_counts, intended_minus_onset_heights = (
                abort_rate_histogram(
                    aligned_to_intended_minus_onset,
                    n_total_trials,
                )
            )

            full_abort_fraction = n_plotted_aborts / n_total_trials
            led_onset_full_area = float(led_onset_heights.sum() * BIN_WIDTH_S)
            intended_minus_onset_full_area = float(
                intended_minus_onset_heights.sum() * BIN_WIDTH_S
            )
            if not np.isclose(
                led_onset_full_area,
                full_abort_fraction,
                rtol=0,
                atol=1e-14,
            ):
                raise RuntimeError(
                    f"LED-onset histogram area mismatch for {half['label']} "
                    f"{group_label} {LED_LABELS[led_trial]}."
                )
            if not np.isclose(
                intended_minus_onset_full_area,
                full_abort_fraction,
                rtol=0,
                atol=1e-14,
            ):
                raise RuntimeError(
                    f"intended-minus-onset histogram area mismatch for "
                    f"{half['label']} {group_label} {LED_LABELS[led_trial]}."
                )

            led_onset_displayed_fraction = float(
                (
                    (aligned_to_led_onset >= DISPLAY_MIN_S)
                    & (aligned_to_led_onset <= DISPLAY_MAX_S)
                ).sum()
                / n_total_trials
            )
            intended_minus_onset_displayed_fraction = float(
                (
                    (aligned_to_intended_minus_onset >= DISPLAY_MIN_S)
                    & (aligned_to_intended_minus_onset <= DISPLAY_MAX_S)
                ).sum()
                / n_total_trials
            )

            panel_results[(half["key"], group_label, led_trial)] = {
                "led_onset_counts": led_onset_counts,
                "led_onset_heights": led_onset_heights,
                "intended_minus_onset_counts": intended_minus_onset_counts,
                "intended_minus_onset_heights": intended_minus_onset_heights,
                "n_total_trials": n_total_trials,
                "n_coded_aborts": n_coded_aborts,
                "n_plotted_aborts": n_plotted_aborts,
                "full_area": full_abort_fraction,
            }

            global_y_max = max(
                global_y_max,
                float(led_onset_heights.max(initial=0.0)),
                float(intended_minus_onset_heights.max(initial=0.0)),
            )

            audit_rows.append(
                {
                    "session_half": half["label"],
                    "group": group_label,
                    "condition": LED_LABELS[led_trial],
                    "n_animal_sessions": int(
                        condition_df[["animal", "session"]].drop_duplicates().shape[0]
                    ),
                    "n_total_trials": n_total_trials,
                    "n_abort_event_3": n_coded_aborts,
                    "n_plotted_aborts": n_plotted_aborts,
                    "n_missing_timed_fix": n_coded_aborts - n_plotted_aborts,
                    "full_abort_fraction": full_abort_fraction,
                    "led_onset_full_hist_area": led_onset_full_area,
                    "led_onset_displayed_fraction": (
                        led_onset_displayed_fraction
                    ),
                    "intended_minus_onset_full_hist_area": (
                        intended_minus_onset_full_area
                    ),
                    "intended_minus_onset_displayed_fraction": (
                        intended_minus_onset_displayed_fraction
                    ),
                    "led_onset_aligned_min_s": float(aligned_to_led_onset.min()),
                    "led_onset_aligned_max_s": float(aligned_to_led_onset.max()),
                    "intended_minus_onset_aligned_min_s": float(
                        aligned_to_intended_minus_onset.min()
                    ),
                    "intended_minus_onset_aligned_max_s": float(
                        aligned_to_intended_minus_onset.max()
                    ),
                }
            )

audit_df = pd.DataFrame(audit_rows)


# %%
############ Verify half accounting and pooled aggregates ############
for half in half_definitions:
    aggregate_rows = audit_df.loc[
        audit_df["session_half"].eq(half["label"])
        & audit_df["group"].eq("Aggregate")
    ]

    for led_trial in [0, 1]:
        condition_label = LED_LABELS[led_trial]
        animal_audit = audit_df.loc[
            audit_df["session_half"].eq(half["label"])
            & audit_df["group"].ne("Aggregate")
            & audit_df["condition"].eq(condition_label)
        ]
        aggregate_audit = aggregate_rows.loc[
            aggregate_rows["condition"].eq(condition_label)
        ].iloc[0]

        for column in [
            "n_animal_sessions",
            "n_total_trials",
            "n_abort_event_3",
            "n_plotted_aborts",
            "n_missing_timed_fix",
        ]:
            if int(animal_audit[column].sum()) != int(aggregate_audit[column]):
                raise RuntimeError(
                    f"{half['label']} aggregate {column} mismatch for "
                    f"{condition_label}."
                )

        for count_key in ["led_onset_counts", "intended_minus_onset_counts"]:
            animal_count_sum = sum(
                panel_results[(half["key"], f"Animal {animal}", led_trial)][
                    count_key
                ]
                for animal in animals
            )
            np.testing.assert_array_equal(
                animal_count_sum,
                panel_results[(half["key"], "Aggregate", led_trial)][count_key],
            )

split_event_counts = (
    audit_df.loc[audit_df["group"].eq("Aggregate")]
    .groupby("condition")["n_abort_event_3"]
    .sum()
    .astype(int)
    .to_dict()
)
split_finite_counts = (
    audit_df.loc[audit_df["group"].eq("Aggregate")]
    .groupby("condition")["n_plotted_aborts"]
    .sum()
    .astype(int)
    .to_dict()
)
if split_event_counts != {
    LED_LABELS[key]: value for key, value in EXPECTED_ABORT_EVENT_3_COUNTS.items()
}:
    raise RuntimeError("Within-animal halves do not recover all coded aborts.")
if split_finite_counts != {
    LED_LABELS[key]: value for key, value in EXPECTED_FINITE_ABORT_COUNTS.items()
}:
    raise RuntimeError("Within-animal halves do not recover all finite aborts.")


# %%
############ Plot one 2 x 9 figure per within-animal session half ############
y_max = 1.16 * global_y_max if global_y_max > 0 else 1.0
saved_figure_paths = []

for half in half_definitions:
    half_df = df.loc[df["within_animal_session_half"].eq(half["key"])].copy()
    group_labels = [f"Animal {animal}" for animal in animals] + ["Aggregate"]

    fig, axes = plt.subplots(
        2,
        len(group_labels),
        figsize=(3.15 * len(group_labels), 7.0),
        sharex=True,
        sharey=True,
    )

    for column_index, group_label in enumerate(group_labels):
        off_result = panel_results[(half["key"], group_label, 0)]
        on_result = panel_results[(half["key"], group_label, 1)]

        for led_trial in [0, 1]:
            result = panel_results[(half["key"], group_label, led_trial)]
            axes[0, column_index].stairs(
                result["led_onset_heights"],
                FULL_BINS,
                color=LED_COLORS[led_trial],
                linewidth=1.5,
                label=LED_LABELS[led_trial],
            )
            axes[1, column_index].stairs(
                result["intended_minus_onset_heights"],
                FULL_BINS,
                color=LED_COLORS[led_trial],
                linewidth=1.5,
                label=LED_LABELS[led_trial],
            )

        for row_index in [0, 1]:
            axes[row_index, column_index].axvline(
                0.0,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
            )
            axes[row_index, column_index].set_xlim(DISPLAY_MIN_S, DISPLAY_MAX_S)
            axes[row_index, column_index].set_ylim(0.0, y_max)
            axes[row_index, column_index].grid(
                axis="y",
                alpha=0.18,
                linewidth=0.6,
            )
            axes[row_index, column_index].tick_params(labelsize=8)
            axes[row_index, column_index].text(
                0.98,
                0.96,
                f"OFF area={off_result['full_area']:.3f}\n"
                f"ON area={on_result['full_area']:.3f}",
                transform=axes[row_index, column_index].transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
            )

        if group_label == "Aggregate":
            panel_title = "Pooled aggregate"
        else:
            animal = int(group_label.split()[-1])
            panel_sessions = animal_half_sessions[animal][half["key"]]
            panel_title = (
                f"Animal {animal}\n"
                f"sessions {panel_sessions[0]}-{panel_sessions[-1]}"
            )
        axes[0, column_index].set_title(panel_title, fontsize=10)
        axes[0, column_index].set_xlabel(
            "Abort time relative to LED onset (s)",
            fontsize=8,
        )
        axes[1, column_index].set_xlabel(
            "Abort time relative to intended_fix - LED_onset_time (s)",
            fontsize=8,
        )

    axes[0, 0].set_ylabel("LED OFF / ON\nAbort probability density (s$^{-1}$)")
    axes[1, 0].set_ylabel("LED OFF / ON\nAbort probability density (s$^{-1}$)")
    axes[0, -1].legend(loc="upper left", fontsize=8, frameon=False)
    axes[1, -1].legend(loc="upper left", fontsize=8, frameon=False)

    n_animal_sessions = half_df[["animal", "session"]].drop_duplicates().shape[0]
    fig.suptitle(
        f"LED7 session 9 fixation aborts: {half['label'].lower()} of each "
        "animal's sessions (training level 16)\n"
        f"All 8 animals, {n_animal_sessions} animal-session pairs; 20 ms bins; "
        "shared y-scale across halves",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.01, 0.01, 1.0, 0.93))
    fig.savefig(half["output_path"], dpi=OUTPUT_DPI, bbox_inches="tight")
    saved_figure_paths.append(half["output_path"])

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# %%
############ Print the numerical audit ############
print(f"Loaded: {INPUT_CSV_PATH}")
print(f"Rows: {len(df):,}")
print(f"Animals: {animals}")
print("\nWithin-animal session splits")
print(split_df.to_string(index=False))

for half in half_definitions:
    half_df = df.loc[df["within_animal_session_half"].eq(half["key"])]
    n_animal_sessions = half_df[["animal", "session"]].drop_duplicates().shape[0]
    aggregate_audit = audit_df.loc[
        audit_df["session_half"].eq(half["label"])
        & audit_df["group"].eq("Aggregate")
    ]
    print(
        f"\n{half['label']}: {n_animal_sessions} animal-session pairs, "
        f"{len(half_df):,} trials"
    )
    print(
        aggregate_audit[
            [
                "condition",
                "n_total_trials",
                "n_abort_event_3",
                "n_plotted_aborts",
                "n_missing_timed_fix",
                "full_abort_fraction",
                "led_onset_displayed_fraction",
                "intended_minus_onset_displayed_fraction",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )

print("\nSaved figures:")
for output_path in saved_figure_paths:
    print(output_path)
