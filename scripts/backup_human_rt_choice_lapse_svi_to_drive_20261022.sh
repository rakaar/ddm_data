#!/usr/bin/env bash
set -euo pipefail

# Snapshot created and SHA256-checked against the completed local fit root.
REPO_DIR="/home/rlab/raghavendra/ddm_data"
SOURCE_DIR="${REPO_DIR}/backup_staging/human_rt_choice_lapse_svi_20261022_011519"
DRIVE_PARENT="raga:ddm_fit_backups_20261022_human_rt_choice_lapse_svi_lavos"
LOG_PREFIX="${REPO_DIR}/logs/gdrive_human_rt_choice_lapse_20261022_011519"
export SOURCE_DIR DRIVE_PARENT LOG_PREFIX

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/logs"
exec > >(tee -a "${LOG_PREFIX}.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "${LOG_PREFIX}.exit_status"' EXIT

echo "Human RT + choice with lapses; backup started: $(date --iso-8601=seconds)"
echo "Local snapshot: ${SOURCE_DIR}"
echo "Destination: ${DRIVE_PARENT}"

rclone mkdir "${DRIVE_PARENT}" --contimeout 10s --timeout 60s
rclone copy "${SOURCE_DIR}" "${DRIVE_PARENT}" \
  --transfers 4 --checkers 8 --drive-chunk-size 64M \
  --contimeout 10s --timeout 60s --retries 3 --low-level-retries 3 \
  --stats 30s --log-level INFO

rclone check "${SOURCE_DIR}" "${DRIVE_PARENT}" \
  --one-way --checkers 8 --contimeout 10s --timeout 60s \
  --combined "${LOG_PREFIX}.check.txt" --log-level INFO
rclone lsjson "${DRIVE_PARENT}" --recursive --files-only --hash \
  --contimeout 10s --timeout 60s > "${LOG_PREFIX}.remote_files.json"
rclone size "${DRIVE_PARENT}" --json > "${LOG_PREFIX}.remote_size.json"

.venv/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

snapshot = Path(os.environ['SOURCE_DIR'])
prefix = os.environ['LOG_PREFIX']
remote = json.loads(Path(prefix+'.remote_files.json').read_text())
observed = {item['Path']: item for item in remote}
local = {str(p.relative_to(snapshot)): p for p in snapshot.rglob('*') if p.is_file()}
assert set(local) == set(observed), (set(local)-set(observed), set(observed)-set(local))
for name, path in local.items():
    item = observed[name]
    assert path.stat().st_size == item['Size'], name
    hashes = {key.lower(): value for key, value in item['Hashes'].items()}
    assert hashlib.md5(path.read_bytes()).hexdigest() == hashes['md5'], name
size = json.loads(Path(prefix+'.remote_size.json').read_text())
expected_bytes = sum(p.stat().st_size for p in local.values())
assert size['count'] == len(local) and size['bytes'] == expected_bytes
manifest = json.loads((snapshot/'config/backup_manifest.json').read_text())
original = Path(manifest['source_path'])
fit_name = original.name
for path in original.rglob('*'):
    if path.is_file():
        copied = snapshot/fit_name/path.relative_to(original)
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(copied.read_bytes()).digest(), path
report = {'status':'passed', 'verified_utc':datetime.now(timezone.utc).isoformat(),
    'destination':os.environ['DRIVE_PARENT'], 'local_snapshot':str(snapshot),
    'source_path':str(original), 'files':len(local), 'bytes':expected_bytes,
    'fit_files':manifest['fit_files'], 'fit_bytes':manifest['fit_bytes'],
    'all_remote_paths_sizes_MD5_match':True, 'original_fit_files_unchanged':True,
    'copy_only_no_sync_or_deletions':True}
Path(prefix+'.verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY

echo "Human lapse fit backup and hash verification completed: $(date --iso-8601=seconds)"
