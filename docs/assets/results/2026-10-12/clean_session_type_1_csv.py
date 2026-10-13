# %% Imports
"""Archive the former prepared CSV and remove its 807 short-sound records once.

Future rebuilds use the same exclusion in build_session_type_csvs.py.
The original cohorts and prepared session types 2/23 are read-only here.
"""
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


# %% Parameters
REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "asd_analysis/session_type_csvs"
INPUT = OUTPUT / "ASD_session_type_1.csv"
ARCHIVE = OUTPUT / "archive/pre_short_sound_cleanup"
EXPECTED_BEFORE = 643177
EXPECTED_REMOVED = 807
EXPECTED_AFTER = 642370
PROTECTED = [REPO / name for name in [
    "SHANK3_cohort1.csv", "SHANK3_cohort2.csv",
    "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv",
]] + [OUTPUT / f"ASD_session_type_{t}.csv" for t in [2, 23]]


def fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}


# %% Archive before replacing any data or manifests
if ARCHIVE.exists():
    raise SystemExit("Cleanup archive already exists; do not overwrite its historical records.")
before = fingerprint(INPUT)
protected_before = {str(p.relative_to(REPO)): fingerprint(p) for p in PROTECTED}
ARCHIVE.mkdir(parents=True)
for name in [INPUT.name, "filter_manifest.json", "verification.json"]:
    shutil.copy2(OUTPUT / name, ARCHIVE / name)
assert fingerprint(ARCHIVE / INPUT.name)["sha256"] == before["sha256"]


# %% Remove only the requested records; retain every cell and original row order
temporary = INPUT.with_suffix(".csv.tmp")
removed = Counter()
removed_fixation = Counter()
removed_success = Counter()
removed_genotype = Counter()
animals = set()
expected_digest = hashlib.sha256()
removed_rows = []
with (ARCHIVE / INPUT.name).open(newline="") as original, temporary.open("w", newline="") as output:
    reader = csv.DictReader(original)
    columns = reader.fieldnames
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    before_count = after_count = 0
    for csv_row_number, row in enumerate(reader, start=2):
        before_count += 1
        flagged = float(row["session_type"]) == 1 and float(row["is_short_sound"] or 0) == 1
        if flagged:
            assert float(row["short_duration"]) in [8, 16, 32, 64]
            removed[(row["filename"], row["animal"])] += 1
            removed_fixation[row["filename"]] += row["abort_type"] == "Fixation"
            removed_success[(row["filename"], row["success"])] += 1
            removed_genotype[(row["filename"], row["genotype"])] += 1
            removed_rows.append({"archived_csv_row_number": csv_row_number, **row})
        else:
            writer.writerow(row)
            expected_digest.update(("\0".join(row[c] for c in columns) + "\0\n").encode())
            animals.add(row["animal"])
            after_count += 1
assert before_count == EXPECTED_BEFORE and after_count == EXPECTED_AFTER
assert removed == Counter({("CNTNAP2_cohort3.csv", "ASD0026"): 330, ("SHANK3_cohort1.csv", "ASD0045"): 477})
assert sum(removed.values()) == EXPECTED_REMOVED and len(animals) == 53
assert {"ASD0026", "ASD0045"} <= animals
with (ARCHIVE / "removed_rows.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=["archived_csv_row_number"] + columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(removed_rows)


# %% Independently reread the replacement before publishing it
actual_digest = hashlib.sha256()
actual_count = 0
with temporary.open(newline="") as output:
    reader = csv.DictReader(output)
    assert reader.fieldnames == columns
    for row in reader:
        assert not (float(row["session_type"]) == 1 and float(row["is_short_sound"] or 0) == 1)
        actual_digest.update(("\0".join(row[c] for c in columns) + "\0\n").encode())
        actual_count += 1
assert actual_count == after_count and actual_digest.digest() == expected_digest.digest()
temporary.replace(INPUT)


# %% Refresh preparation counts; the full independent verifier runs afterwards
manifest = json.loads((ARCHIVE / "filter_manifest.json").read_text())
manifest["filters"]["excluded_short_sound_rule"] = (
    "session_type == 1 AND is_short_sound == 1; apply after deriving repeats on all source trials"
)
for filename, source in manifest["sources"].items():
    counts = source["session_types"]["1"]
    n_removed = sum(n for (name, _), n in removed.items() if name == filename)
    counts["output_rows_before_short_sound_exclusion"] = counts["output_rows"]
    counts["short_sound_rows_removed"] = n_removed
    counts["output_rows"] -= n_removed
    counts["fixation_abort_rows"] -= removed_fixation[filename]
    counts["success_1_rows"] -= removed_success[(filename, "1")]
    counts["success_minus1_rows"] -= removed_success[(filename, "-1")]
    for genotype in counts["genotype_counts"]:
        counts["genotype_counts"][genotype] -= removed_genotype[(filename, genotype)]
manifest["outputs"]["1"].update({"rows": after_count, "bytes": INPUT.stat().st_size})
manifest["cleanup_manifest"] = str((OUTPUT / "short_sound_cleanup.json").relative_to(REPO))
(OUTPUT / "filter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(OUTPUT / "verification.json").unlink()
protected_after = {str(p.relative_to(REPO)): fingerprint(p) for p in PROTECTED}
assert protected_after == protected_before
report = {
    "status": "passed", "rule": "session_type == 1 AND is_short_sound == 1",
    "input_csv": str(INPUT.relative_to(REPO)),
    "archive_csv": str((ARCHIVE / INPUT.name).relative_to(REPO)),
    "removed_rows_csv": str((ARCHIVE / "removed_rows.csv").relative_to(REPO)),
    "before": before, "after": fingerprint(INPUT),
    "before_rows": before_count, "removed_rows": sum(removed.values()), "after_rows": after_count,
    "removed_by_animal": {animal: n for (_, animal), n in removed.items()},
    "animals_retained": len(animals), "columns": len(columns),
    "every_retained_cell_and_row_order_verified": True,
    "retained_cells_sha256": actual_digest.hexdigest(),
    "protected_files": protected_after, "protected_hashes_sizes_mtimes_unchanged": True,
}
(OUTPUT / "short_sound_cleanup.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
