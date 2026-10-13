# %%
"""Combine ASD cohorts by session type using the documented next-trial rule."""
import csv
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pandas as pd


# %% Parameters
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    REPO_ROOT / "SHANK3_cohort1.csv",
    REPO_ROOT / "SHANK3_cohort2.csv",
    REPO_ROOT / "CNTNAP2_cohort3.csv",
    REPO_ROOT / "CNTNAP2_cohort4.csv",
]
OUTPUT_DIR = REPO_ROOT / "asd_analysis/session_type_csvs"
METADATA_PATH = REPO_ROOT / "metadata.csv"
SESSION_TYPES = (1, 2, 23)
TRAINING_LEVEL = 16
VALID_SUCCESS = (1, -1)
FIXATION_ABORT = "Fixation"
NON_REPEAT_ABORTS = ("CNP", "Fixation")
SESSION_KEYS = ["animal", "source_file", "session"]
CHUNK_ROWS = 50_000


# %% Read animal genotypes and retain the union of source columns
metadata = pd.read_csv(METADATA_PATH, sep=";", dtype=str, keep_default_na=False)
if metadata["animal"].duplicated().any():
    raise ValueError("metadata.csv must contain one genotype row per animal.")
if not metadata["genotype"].isin(["wt", "hom", "het"]).all():
    raise ValueError("Unexpected genotype labels in metadata.csv.")
genotype_by_animal = metadata.set_index("animal")["genotype"]

output_columns = []
for source_path in SOURCE_FILES:
    with source_path.open(newline="") as stream:
        source_columns = next(csv.reader(stream))
    output_columns.extend(c for c in source_columns if c not in output_columns)
new_columns = ["filename", "genotype", "trial_is_repeat"]
if any(column in output_columns for column in new_columns):
    raise ValueError("A source already contains one of the new output columns.")
output_columns.extend(new_columns)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_paths = {
    session_type: OUTPUT_DIR / f"ASD_session_type_{session_type}.csv"
    for session_type in SESSION_TYPES
}
temporary_paths = {
    session_type: path.with_suffix(".csv.tmp")
    for session_type, path in output_paths.items()
}
audit = {
    "filters": {
        "training_level": TRAINING_LEVEL,
        "repeat_rule": (
            "Exclude a trial if the previous chronological trial has repeated_trial=True, "
            "except after abort_type CNP or Fixation. Compute before all other filters."
        ),
        "repeat_session_keys": SESSION_KEYS,
        "repeat_order": "numeric trial_start within each animal/source_file/session",
        "duplicate_trials": (
            "Records with the same session keys and trial_start share repeat status; "
            "all original records are preserved unless excluded by a requested filter."
        ),
        "outcome": "abort_type == 'Fixation' OR success in {1, -1}",
        "excluded_short_sound_rule": "session_type == 1 AND is_short_sound == 1; apply after deriving repeats on all source trials",
        "session_types": list(SESSION_TYPES),
    },
    "filename": "basename of the input cohort CSV; source_file retains the daily CSV name",
    "genotype": "metadata.csv genotype matched by animal; wt/hom/het",
    "trial_is_repeat": "derived current-trial repeat status; retained output rows are False",
    "columns": output_columns,
    "sources": {},
}
output_counts = Counter()


# %% Filter in chunks, retaining source text values and original row order
with ExitStack() as stack:
    output_streams = {
        session_type: stack.enter_context(path.open("w", newline=""))
        for session_type, path in temporary_paths.items()
    }
    for stream in output_streams.values():
        csv.writer(stream).writerow(output_columns)

    for source_path in SOURCE_FILES:
        source_counts = Counter()
        source_session_counts = {}
        source_genotype_counts = {str(t): Counter() for t in SESSION_TYPES}

        # Calculate repeat status on ALL source trials before any selection.
        # Rows may be out of time order, and trial numbers can restart. Copies of
        # the same timestamped trial must share status rather than repeat each other.
        trial_info = pd.read_csv(
            source_path,
            usecols=SESSION_KEYS + ["trial", "trial_start", "repeated_trial", "abort_type"],
            dtype=str,
            keep_default_na=False,
        )
        trial_info["trial_start_order"] = pd.to_numeric(trial_info["trial_start"])
        if trial_info[SESSION_KEYS + ["trial_start"]].eq("").any().any():
            raise ValueError(f"Missing session keys or trial_start in {source_path.name}.")
        repeated_trial = trial_info["repeated_trial"].str.strip().str.lower()
        if not repeated_trial.isin(["true", "false", "", "nan"]).all():
            raise ValueError(f"Unexpected repeated_trial values in {source_path.name}.")
        missing_animals = set(trial_info["animal"]) - set(genotype_by_animal.index)
        if missing_animals:
            raise ValueError(f"Animals missing from metadata.csv: {sorted(missing_animals)}")

        event_keys = SESSION_KEYS + ["trial_start_order"]
        duplicated = trial_info.duplicated(event_keys, keep=False)
        if duplicated.any():
            conflicts = (
                trial_info.loc[duplicated]
                .groupby(event_keys)[["trial", "repeated_trial", "abort_type"]]
                .nunique(dropna=False)
                .gt(1)
            )
            if conflicts.any().any():
                raise ValueError(f"Conflicting duplicate trial records in {source_path.name}.")

        chronological = trial_info.drop_duplicates(event_keys).sort_values(
            event_keys, kind="stable"
        ).copy()
        triggers_repeat = (
            chronological["repeated_trial"].str.strip().str.lower().eq("true")
            & ~chronological["abort_type"].isin(NON_REPEAT_ABORTS)
        )
        chronological["trial_is_repeat"] = triggers_repeat.groupby(
            [chronological[key] for key in SESSION_KEYS], sort=False
        ).shift(1, fill_value=False)
        # A left merge restores the original source order and maps the same
        # derived status to every copy of each timestamped trial.
        repeat_flags = trial_info[event_keys].merge(
            chronological[event_keys + ["trial_is_repeat"]],
            on=event_keys,
            how="left",
            sort=False,
            validate="many_to_one",
        )["trial_is_repeat"].to_numpy(dtype=bool)
        repeat_audit = {
            "source_sessions": int(chronological.groupby(SESSION_KEYS).ngroups),
            "distinct_timestamped_trials": len(chronological),
            "duplicate_records_beyond_first": len(trial_info) - len(chronological),
            "derived_repeat_rows": int(repeat_flags.sum()),
        }
        del trial_info, chronological
        source_row_offset = 0

        # Text reads preserve numeric strings and existing values. Missing source
        # columns become blank cells when aligned to the combined output schema.
        for chunk in pd.read_csv(
            source_path, dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS
        ):
            training_level = pd.to_numeric(chunk["training_level"], errors="coerce")
            session_type = pd.to_numeric(chunk["session_type"], errors="coerce")
            success = pd.to_numeric(chunk["success"], errors="coerce")
            short_sound = pd.to_numeric(chunk["is_short_sound"], errors="coerce")
            exclude_short_sound = session_type.eq(1) & short_sound.eq(1)
            repeated_trial = chunk["repeated_trial"].str.strip().str.lower()
            known_repeat = repeated_trial.isin(["true", "false", "", "nan"])
            if not known_repeat.all():
                raise ValueError(
                    f"Unexpected repeated_trial values in {source_path.name}: "
                    f"{sorted(repeated_trial[~known_repeat].unique())}"
                )

            chunk_repeat_flags = repeat_flags[source_row_offset : source_row_offset + len(chunk)]
            keep_repeat = ~pd.Series(chunk_repeat_flags, index=chunk.index)
            source_row_offset += len(chunk)
            keep_training = training_level.eq(TRAINING_LEVEL)
            keep_outcome = chunk["abort_type"].eq(FIXATION_ABORT) | success.isin(
                VALID_SUCCESS
            )
            keep = keep_training & keep_repeat & keep_outcome
            source_counts.update(
                {
                    "input_rows": len(chunk),
                    "repeated_trial_true": int(repeated_trial.eq("true").sum()),
                    "repeated_trial_false": int(repeated_trial.eq("false").sum()),
                    "repeated_trial_missing": int(repeated_trial.isin(["", "nan"]).sum()),
                }
            )

            for selected_type in SESSION_TYPES:
                type_mask = session_type.eq(selected_type)
                before_short_sound = type_mask & keep
                selected_mask = before_short_sound & ~exclude_short_sound
                counts = source_session_counts.setdefault(str(selected_type), Counter())
                counts.update(
                    {
                        "input_rows": int(type_mask.sum()),
                        "after_training": int((type_mask & keep_training).sum()),
                        "after_repeat": int((type_mask & keep_training & keep_repeat).sum()),
                        "repeat_rows_removed_at_level16": int(
                            (type_mask & keep_training & ~keep_repeat).sum()
                        ),
                        "output_rows": int(selected_mask.sum()),
                        "output_rows_before_short_sound_exclusion": int(before_short_sound.sum()),
                        "short_sound_rows_removed": int((before_short_sound & exclude_short_sound).sum()),
                        "fixation_abort_rows": int(
                            (selected_mask & chunk["abort_type"].eq(FIXATION_ABORT)).sum()
                        ),
                        "success_1_rows": int((selected_mask & success.eq(1)).sum()),
                        "success_minus1_rows": int((selected_mask & success.eq(-1)).sum()),
                    }
                )
                if selected_mask.any():
                    selected = chunk.loc[selected_mask].copy()
                    selected["filename"] = source_path.name
                    selected["genotype"] = selected["animal"].map(genotype_by_animal)
                    selected["trial_is_repeat"] = False
                    source_genotype_counts[str(selected_type)].update(
                        selected["genotype"].value_counts().to_dict()
                    )
                    selected.reindex(columns=output_columns, fill_value="").to_csv(
                        output_streams[selected_type], index=False, header=False
                    )
                    output_counts[selected_type] += len(selected)

        audit["sources"][source_path.name] = {
            **dict(source_counts),
            **repeat_audit,
            "session_types": {
                key: {**dict(value), "genotype_counts": dict(source_genotype_counts[key])}
                for key, value in source_session_counts.items()
            },
        }
        print(
            f"{source_path.name}: {source_counts['input_rows']:,} input rows; "
            + ", ".join(
                f"type {key}: {value['output_rows']:,} retained"
                for key, value in source_session_counts.items()
            ),
            flush=True,
        )


# %% Publish completed CSVs and their count/provenance manifest
# The previous verification describes the previous CSVs and must be refreshed.
(OUTPUT_DIR / "verification.json").unlink(missing_ok=True)
audit["outputs"] = {}
for session_type, output_path in output_paths.items():
    temporary_paths[session_type].replace(output_path)
    audit["outputs"][str(session_type)] = {
        "path": str(output_path.relative_to(REPO_ROOT)),
        "rows": output_counts[session_type],
        "bytes": output_path.stat().st_size,
    }
    print(f"Saved {output_path}: {output_counts[session_type]:,} rows", flush=True)
manifest_path = OUTPUT_DIR / "filter_manifest.json"
manifest_path.write_text(json.dumps(audit, indent=2) + "\n")
print(f"Saved {manifest_path}", flush=True)
