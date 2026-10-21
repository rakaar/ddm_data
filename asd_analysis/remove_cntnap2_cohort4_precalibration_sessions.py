# %% Imports
"""Archive and remove CNTNAP2 cohort 4 sessions 1–12 from both active CSVs.

Retain each surviving CSV line byte for byte. A fresh preparation build in a
temporary folder supplies current manifests and must reproduce all three
prepared CSVs cell for cell before its manifest is adopted.
"""
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
PREPARED = REPO / "asd_analysis/session_type_csvs"
ARCHIVE = PREPARED / "archive/pre_cntnap2_cohort4_session13_cleanup"
REPORT = PREPARED / "speaker_calibration_cleanup.json"
STAGING = PREPARED / ".speaker_calibration_rebuild"
COHORT = "CNTNAP2_cohort4.csv"
FIRST_SESSION = 13
TARGETS = [REPO / COHORT, PREPARED / "ASD_session_type_1.csv"]
EXPECTED_ROWS = [(618712, 96082, 522630), (642370, 31767, 610603)]
EXPECTED_ANIMALS = [(20, 20), (53, 10)]  # retained animals, affected animals
PROTECTED = [REPO / name for name in [
    "SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv",
]] + [PREPARED / f"ASD_session_type_{t}.csv" for t in [2, 23]]


def fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns}


def cell_digest(path):
    digest = hashlib.sha256()
    rows = 0
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        for row in reader:
            assert len(row) == len(columns)
            digest.update(("\0".join(row) + "\0\n").encode())
            rows += 1
    return columns, rows, digest.hexdigest()


# %% Archive the complete original files and preparation history before editing
if ARCHIVE.exists() or STAGING.exists():
    raise SystemExit("Cleanup archive or staging folder exists; do not overwrite it.")
protected_before = {str(p.relative_to(REPO)): fingerprint(p) for p in PROTECTED}
before = {str(p.relative_to(REPO)): fingerprint(p) for p in TARGETS}
ARCHIVE.mkdir(parents=True)
for path in TARGETS:
    shutil.copy2(path, ARCHIVE / path.name)
    assert fingerprint(ARCHIVE / path.name)["sha256"] == before[str(path.relative_to(REPO))]["sha256"]
for name in ["filter_manifest.json", "verification.json", "short_sound_cleanup.json"]:
    shutil.copy2(PREPARED / name, ARCHIVE / name)
(ARCHIVE / "archive_manifest.json").write_text(json.dumps({
    "reason": "Speaker calibration was incorrect until numeric session 13 in CNTNAP2 cohort 4.",
    "files_before_cleanup": before,
    "protected_files": protected_before,
}, indent=2) + "\n")


# %% Stage byte-preserving replacements and complete removed-row provenance
files = {}
locations_path = ARCHIVE / "removed_row_locations.csv.gz"
with gzip.open(locations_path, "wt", newline="") as locations:
    location_writer = csv.writer(locations)
    location_writer.writerow([
        "active_csv", "archived_csv", "archived_csv_row_number", "animal",
        "session", "source_file", "trial", "trial_start", "training_level", "session_type",
    ])
    for path, expected, animal_counts in zip(TARGETS, EXPECTED_ROWS, EXPECTED_ANIMALS):
        relative = str(path.relative_to(REPO))
        archived = ARCHIVE / path.name
        temporary = path.with_suffix(".csv.session13.tmp")
        removed_path = ARCHIVE / f"removed_{path.name}"
        expected_bytes = hashlib.sha256()
        removed_by_animal = Counter()
        retained_animals = set()
        total = retained = 0
        with archived.open("rb") as original, temporary.open("wb") as output, removed_path.open("wb") as removed:
            header = original.readline()
            columns = next(csv.reader([header.decode("utf-8")]))
            index = {name: i for i, name in enumerate(columns)}
            output.write(header)
            removed.write(header)
            expected_bytes.update(header)
            for csv_row_number, line in enumerate(original, start=2):
                values = next(csv.reader([line.decode("utf-8")]))
                assert len(values) == len(columns), "Unexpected multiline CSV record."
                is_cohort = "filename" not in index or values[index["filename"]] == COHORT
                exclude = is_cohort and float(values[index["session"]]) < FIRST_SESSION
                total += 1
                if exclude:
                    removed.write(line)
                    removed_by_animal[values[index["animal"]]] += 1
                    location_writer.writerow([
                        relative, str(archived.relative_to(REPO)), csv_row_number,
                        *[values[index[name]] for name in [
                            "animal", "session", "source_file", "trial", "trial_start", "training_level", "session_type",
                        ]],
                    ])
                else:
                    output.write(line)
                    expected_bytes.update(line)
                    retained_animals.add(values[index["animal"]])
                    retained += 1
        assert (total, sum(removed_by_animal.values()), retained) == expected
        assert (len(retained_animals), len(removed_by_animal)) == animal_counts
        assert set(removed_by_animal) <= retained_animals
        assert fingerprint(temporary)["sha256"] == expected_bytes.hexdigest()
        files[relative] = {
            "archive_csv": str(archived.relative_to(REPO)),
            "removed_csv": str(removed_path.relative_to(REPO)),
            "before": before[relative], "after": fingerprint(temporary),
            "before_rows": total, "removed_rows": sum(removed_by_animal.values()),
            "after_rows": retained, "retained_animals": len(retained_animals),
            "affected_animals": len(removed_by_animal),
            "removed_rows_by_animal": dict(sorted(removed_by_animal.items())),
            "retained_bytes_and_row_order_preserved": True,
        }
        print(f"Verified replacement: {path.name}: {total:,} - {total-retained:,} = {retained:,} rows", flush=True)

for path in TARGETS:
    assert fingerprint(path) == before[str(path.relative_to(REPO))]
    path.with_suffix(".csv.session13.tmp").replace(path)
    files[str(path.relative_to(REPO))]["after"] = fingerprint(path)

report = {
    "status": "csv_cleanup_passed_preparation_verification_pending",
    "date": "2026-10-17",
    "rule": "filename == CNTNAP2_cohort4.csv AND numeric session < 13; no training/session-type restriction",
    "reason": "Speaker calibration was incorrect before session 13.",
    "archive_directory": str(ARCHIVE.relative_to(REPO)),
    "removed_row_locations": str(locations_path.relative_to(REPO)),
    "files": files, "protected_files": protected_before,
    "previous_figures": "Preserved as historical pre-calibration-exclusion results; not recomputed.",
    "old_row_numbers": "Earlier source/prepared row numbers refer to the archived copies, not the shortened active CSVs.",
}
REPORT.write_text(json.dumps(report, indent=2) + "\n")


# %% Rebuild separately to refresh source counts and verify prepared cells/order
environment = os.environ.copy()
environment["ASD_SESSION_TYPE_OUTPUT_DIR"] = str(STAGING)
subprocess.run([
    str(REPO / ".venv/bin/python"), str(REPO / "asd_analysis/build_session_type_csvs.py"),
], env=environment, cwd=REPO, check=True)
manifest = json.loads((STAGING / "filter_manifest.json").read_text())
for t in [1, 2, 23]:
    active = PREPARED / f"ASD_session_type_{t}.csv"
    assert cell_digest(active) == cell_digest(STAGING / active.name), active.name
    manifest["outputs"][str(t)].update({
        "path": str(active.relative_to(REPO)),
        "bytes": active.stat().st_size,
        "sha256": fingerprint(active)["sha256"],
    })
    print(f"Fresh preparation matches every cell and row: {active.name}", flush=True)
manifest["cleanup_manifest"] = str((PREPARED / "short_sound_cleanup.json").relative_to(REPO))
manifest["calibration_cleanup_manifest"] = str(REPORT.relative_to(REPO))
(PREPARED / "filter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(PREPARED / "verification.json").unlink()
shutil.rmtree(STAGING)


# %% Independently verify repeat eligibility and every prepared output cell
subprocess.run([
    str(REPO / ".venv/bin/python"), str(REPO / "asd_analysis/verify_session_type_csvs.py"),
], cwd=REPO, check=True)
protected_after = {str(p.relative_to(REPO)): fingerprint(p) for p in PROTECTED}
assert protected_after == protected_before
report.update({
    "status": "passed", "protected_files_unchanged": True,
    "fresh_preparation_matches_all_retained_cells_and_order": True,
    "independent_preparation_verification": "asd_analysis/session_type_csvs/verification.json",
})
REPORT.write_text(json.dumps(report, indent=2) + "\n")
print(f"Cleanup complete. Audit: {REPORT}", flush=True)
