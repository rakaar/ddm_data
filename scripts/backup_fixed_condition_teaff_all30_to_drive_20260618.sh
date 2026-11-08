#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/rlab/raghavendra/ddm_data"
SOURCE_DIR="${REPO_DIR}/fit_animal_by_animal/NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30"
DRIVE_PARENT="raga:ddm_fit_backups_20260618_fixed_condition_t_E_aff_all30_lavos"
DRIVE_RESULTS_DIR="${DRIVE_PARENT}/NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30"
DRIVE_CONFIG_DIR="${DRIVE_PARENT}/config"
DRIVE_LEDGER_DIR="raga:ddm_fit_backup_ledgers"

cd "${REPO_DIR}"

echo "Started: $(date --iso-8601=seconds)"
echo "Source: ${SOURCE_DIR}"
echo "Drive parent: ${DRIVE_PARENT}"
echo

echo "Local source counts before upload:"
find "${SOURCE_DIR}" -type f | wc -l
du -sh "${SOURCE_DIR}"
echo "Result PKLs: $(find "${SOURCE_DIR}" -maxdepth 1 -name 'results_*_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pkl' | wc -l)"
echo "Result PDFs: $(find "${SOURCE_DIR}" -maxdepth 1 -name 'results_*_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pdf' -size +0c | wc -l)"
echo "VBMC PKLs: $(find "${SOURCE_DIR}" -maxdepth 1 -name 'vbmc_PKL_file_norm_alpha_condition_t_E_aff_fixed_delay_tied_results_batch_*_FROM_ABORTS.pkl' | wc -l)"
echo "Logs: $(find "${SOURCE_DIR}/logs" -type f -name '*.log' | wc -l)"
echo

rclone mkdir "${DRIVE_RESULTS_DIR}"
rclone mkdir "${DRIVE_CONFIG_DIR}"
rclone mkdir "${DRIVE_LEDGER_DIR}"

echo "Uploading consolidated results..."
rclone copy "${SOURCE_DIR}" "${DRIVE_RESULTS_DIR}" \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 64M \
  --progress \
  --stats 30s \
  --log-level INFO

echo
echo "Uploading ledgers to backup config and central Drive ledger folder..."
rclone copyto "${REPO_DIR}/FIT_BACKUP_LEDGER.md" "${DRIVE_CONFIG_DIR}/FIT_BACKUP_LEDGER.md"
rclone copyto "${REPO_DIR}/DISTRIBUTED_FIXED_DELAY_FIT_LEDGER.md" "${DRIVE_CONFIG_DIR}/DISTRIBUTED_FIXED_DELAY_FIT_LEDGER.md"
rclone copyto "${REPO_DIR}/FIT_BACKUP_LEDGER.md" "${DRIVE_LEDGER_DIR}/FIT_BACKUP_LEDGER.md"

echo
echo "Remote verification:"
echo "Remote result files: $(rclone lsf "${DRIVE_RESULTS_DIR}" --recursive --files-only | wc -l)"
echo "Remote config files: $(rclone lsf "${DRIVE_CONFIG_DIR}" --files-only | wc -l)"
rclone size "${DRIVE_PARENT}"

echo "Finished: $(date --iso-8601=seconds)"
