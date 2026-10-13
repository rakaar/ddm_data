# %%
"""Independently verify the ASD CSVs using the standard-library CSV parser."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


# %% Parameters
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "asd_analysis/session_type_csvs"
SOURCE_FILES = [
    "SHANK3_cohort1.csv",
    "SHANK3_cohort2.csv",
    "CNTNAP2_cohort3.csv",
    "CNTNAP2_cohort4.csv",
]
SESSION_TYPES = ("1", "2", "23")


# %% Independently establish the expected schema and genotype map
manifest = json.loads((OUTPUT_DIR / "filter_manifest.json").read_text())
columns = []
for name in SOURCE_FILES:
    with (REPO_ROOT / name).open(newline="") as stream:
        header = next(csv.reader(stream))
    columns.extend(column for column in header if column not in columns)
columns.extend(["filename", "genotype", "trial_is_repeat"])
assert manifest["columns"] == columns
with (REPO_ROOT / "metadata.csv").open(newline="") as stream:
    metadata = list(csv.DictReader(stream, delimiter=";"))
genotypes = {row["animal"]: row["genotype"] for row in metadata}
assert len(genotypes) == len(metadata)

expected_counts = Counter()
actual_counts = Counter()
expected_hashes = {
    (t, name): hashlib.sha256() for t in SESSION_TYPES for name in SOURCE_FILES
}
actual_hashes = {
    (t, name): hashlib.sha256() for t in SESSION_TYPES for name in SOURCE_FILES
}
fixation_counts = Counter()
genotype_counts = {t: Counter() for t in SESSION_TYPES}


# %% Reconstruct repeat status with chronological session-by-session iteration
for name in SOURCE_FILES:
    groups = defaultdict(list)
    with (REPO_ROOT / name).open(newline="") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            group = (row["animal"], row["source_file"], row["session"])
            groups[group].append(
                (
                    float(row["trial_start"]),
                    row_index,
                    row["repeated_trial"].strip().lower() == "true",
                    row["abort_type"],
                    row["trial"],
                )
            )

    excluded_rows = set()
    for records in groups.values():
        previous_time = None
        previous_trigger = False
        current_is_repeat = False
        previous_signature = None
        for timestamp, row_index, flag, abort_type, trial in sorted(records):
            signature = (flag, abort_type, trial)
            if timestamp != previous_time:
                current_is_repeat = previous_trigger
                previous_trigger = flag and abort_type not in {"CNP", "Fixation"}
                previous_time = timestamp
                previous_signature = signature
            else:
                # Multiple records of one trial share its eligibility.
                assert signature == previous_signature
            if current_is_repeat:
                excluded_rows.add(row_index)
    assert len(excluded_rows) == manifest["sources"][name]["derived_repeat_rows"]
    del groups

    # Read the original source again in its original order. No source cells are
    # transformed; each expected row gains only the three documented columns.
    with (REPO_ROOT / name).open(newline="") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            t = row["session_type"]
            retain = (
                t in SESSION_TYPES
                and row["training_level"] == "16"
                and row_index not in excluded_rows
                and (row["abort_type"] == "Fixation" or row["success"] in {"1", "-1"})
                and not (t == "1" and float(row["is_short_sound"] or 0) == 1)
            )
            if retain:
                row["filename"] = name
                row["genotype"] = genotypes[row["animal"]]
                row["trial_is_repeat"] = "False"
                values = [row.get(column, "") for column in columns]
                expected_hashes[(t, name)].update(("\0".join(values) + "\0\n").encode())
                expected_counts[(t, name)] += 1
    print(f"Source repeat rule and expected rows checked: {name}", flush=True)


# %% Check every written row, including all preserved cells and source order
for t in SESSION_TYPES:
    path = OUTPUT_DIR / f"ASD_session_type_{t}.csv"
    n_rows = 0
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == columns
        for row in reader:
            assert None not in row
            assert row["session_type"] == t
            assert row["training_level"] == "16"
            assert row["trial_is_repeat"] == "False"
            assert row["abort_type"] == "Fixation" or row["success"] in {"1", "-1"}
            assert not (t == "1" and float(row["is_short_sound"] or 0) == 1)
            assert row["genotype"] == genotypes[row["animal"]]
            key = (t, row["filename"])
            assert key in actual_hashes
            values = [row[column] for column in columns]
            actual_hashes[key].update(("\0".join(values) + "\0\n").encode())
            actual_counts[key] += 1
            fixation_counts[t] += row["abort_type"] == "Fixation"
            genotype_counts[t][row["genotype"]] += 1
            n_rows += 1
    assert n_rows == manifest["outputs"][t]["rows"]
    print(f"Output checked: {path.name}: {n_rows:,} rows", flush=True)

assert actual_counts == expected_counts
for key in expected_hashes:
    assert actual_hashes[key].digest() == expected_hashes[key].digest(), key
for name in SOURCE_FILES:
    for t in SESSION_TYPES:
        assert actual_counts[(t, name)] == manifest["sources"][name]["session_types"][t]["output_rows"]

result = {
    "status": "passed",
    "rows_checked": sum(actual_counts.values()),
    "columns_checked": len(columns),
    "checks": [
        "next chronological trial exclusion with CNP/Fixation exceptions",
        "session boundaries and duplicate trial status",
        "training/session/outcome filters",
        "session-type-1 short-sound exclusion after repeat derivation",
        "source filename and genotype",
        "source row counts and order",
        "every preserved cell value",
    ],
    "fixation_abort_rows": dict(fixation_counts),
    "genotype_counts": {t: dict(counts) for t, counts in genotype_counts.items()},
    "counts_by_session_and_source": {
        f"{t}/{name}": count for (t, name), count in sorted(actual_counts.items())
    },
}
(OUTPUT_DIR / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
