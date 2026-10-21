# %% Imports and parameters
"""Independently compare both cleaned CSVs with their archived originals."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_csvs"
report = json.loads((OUTPUT / "speaker_calibration_cleanup.json").read_text())
assert report["status"] == "passed"


# %% Independently check cells, byte-for-byte line identity, and removed records
checked = {}
expected_locations = []
for relative, info in report["files"].items():
    active = REPO / relative
    archived = REPO / info["archive_csv"]
    removed = REPO / info["removed_csv"]
    before_animals = set()
    after_animals = set()
    removed_counts = Counter()
    total = retained = excluded = 0
    with archived.open("rb") as original, active.open("rb") as cleaned, removed.open("rb") as discarded:
        header = original.readline()
        assert header == cleaned.readline() == discarded.readline()
        columns = next(csv.reader([header.decode()]))
        for csv_row_number, line in enumerate(original, start=2):
            row = dict(zip(columns, next(csv.reader([line.decode()])), strict=True))
            before_animals.add(row["animal"])
            total += 1
            # Numeric session, rather than session_type; other cohorts survive.
            affected = row.get("filename", "CNTNAP2_cohort4.csv") == "CNTNAP2_cohort4.csv"
            if affected and float(row["session"]) < 13:
                assert line == discarded.readline()
                excluded += 1
                removed_counts[row["animal"]] += 1
                expected_locations.append([
                    relative, info["archive_csv"], str(csv_row_number),
                    *[row[c] for c in ["animal", "session", "source_file", "trial", "trial_start", "training_level", "session_type"]],
                ])
            else:
                assert line == cleaned.readline(), (relative, csv_row_number)
                retained += 1
                after_animals.add(row["animal"])
        assert cleaned.read() == discarded.read() == b""
    assert before_animals == after_animals
    assert (total, excluded, retained) == (info["before_rows"], info["removed_rows"], info["after_rows"])
    assert dict(removed_counts) == info["removed_rows_by_animal"]
    for path, expected_hash in [(active, info["after"]["sha256"]), (archived, info["before"]["sha256"])]:
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected_hash
    checked[relative] = {
        "before_rows": total, "removed_rows": excluded, "retained_rows": retained,
        "animals_retained": len(after_animals), "affected_animals": len(removed_counts),
        "every_retained_byte_cell_and_row_order_matches_archive": True,
    }


# %% Complete removed-row audit and untouched-file checks
with gzip.open(REPO / report["removed_row_locations"], "rt", newline="") as stream:
    reader = csv.reader(stream)
    next(reader)
    for expected, actual in zip_longest(expected_locations, reader):
        assert expected == actual
for relative, expected in report["protected_files"].items():
    path = REPO / relative
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"]
    assert path.stat().st_size == expected["bytes"]
    assert path.stat().st_mtime_ns == expected["mtime_ns"]
prepared = json.loads((OUTPUT / "verification.json").read_text())
assert prepared["status"] == "passed"
assert prepared["rows_checked"] == 1843386
result = {
    "status": "passed", "files": checked,
    "removed_row_locations_checked": len(expected_locations),
    "protected_hashes_sizes_mtimes_unchanged": True,
    "independent_preparation_verification": prepared,
}
(OUTPUT / "speaker_calibration_independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
