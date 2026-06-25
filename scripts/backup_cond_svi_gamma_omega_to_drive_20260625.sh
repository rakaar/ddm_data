#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/rlab/raghavendra/ddm_data"
COND_SOURCE_DIR="${REPO_DIR}/fit_each_condn/svi_gamma_omega_fixed_from_animal_svi_condition_delay_results"
COMPARISON_SOURCE_DIR="${REPO_DIR}/fit_each_condn/svi_condition_gamma_omega_vs_npl_alpha_svi_comparison"

DRIVE_PARENT="raga:ddm_fit_backups_20260625_cond_svi_gamma_omega_fixed_from_animal_svi_lavos"
DRIVE_COND_DIR="${DRIVE_PARENT}/svi_gamma_omega_fixed_from_animal_svi_condition_delay_results"
DRIVE_COMPARISON_DIR="${DRIVE_PARENT}/svi_condition_gamma_omega_vs_npl_alpha_svi_comparison"
DRIVE_CONFIG_DIR="${DRIVE_PARENT}/config"
DRIVE_LEDGER_DIR="raga:ddm_fit_backup_ledgers"

cd "${REPO_DIR}"

echo "Started: $(date --iso-8601=seconds)"
echo "Condition SVI source: ${COND_SOURCE_DIR}"
echo "Comparison source: ${COMPARISON_SOURCE_DIR}"
echo "Drive parent: ${DRIVE_PARENT}"
echo

echo "Local source counts before upload:"
echo "Condition SVI files: $(find "${COND_SOURCE_DIR}" -type f | wc -l)"
du -sh "${COND_SOURCE_DIR}"
for subdir in "${COND_SOURCE_DIR}"/*; do
  if [[ -d "${subdir}" ]]; then
    echo "  $(basename "${subdir}"): $(find "${subdir}" -type f | wc -l) files, $(du -sh "${subdir}" | awk '{print $1}')"
  fi
done
echo "Comparison output files: $(find "${COMPARISON_SOURCE_DIR}" -type f | wc -l)"
du -sh "${COMPARISON_SOURCE_DIR}"
echo

rclone mkdir "${DRIVE_COND_DIR}"
rclone mkdir "${DRIVE_COMPARISON_DIR}"
rclone mkdir "${DRIVE_CONFIG_DIR}"
rclone mkdir "${DRIVE_LEDGER_DIR}"

echo "Uploading condition SVI fit family..."
rclone copy "${COND_SOURCE_DIR}" "${DRIVE_COND_DIR}" \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 64M \
  --progress \
  --stats 30s \
  --log-level INFO

echo
echo "Uploading comparison outputs..."
rclone copy "${COMPARISON_SOURCE_DIR}" "${DRIVE_COMPARISON_DIR}" \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 64M \
  --progress \
  --stats 30s \
  --log-level INFO

echo
echo "Uploading config and provenance files..."
rclone copyto "${REPO_DIR}/FIT_BACKUP_LEDGER.md" "${DRIVE_CONFIG_DIR}/FIT_BACKUP_LEDGER.md"
rclone copyto "${REPO_DIR}/fit_each_condn/README.md" "${DRIVE_CONFIG_DIR}/fit_each_condn_README.md"
rclone copyto "${REPO_DIR}/fit_each_condn/benchmark_svi_3param_gamma_omega_t_E_aff.py" "${DRIVE_CONFIG_DIR}/benchmark_svi_3param_gamma_omega_t_E_aff.py"
rclone copyto "${REPO_DIR}/fit_each_condn/fit_svi_cond_by_cond_gamma_omega_fixed_from_animal_svi.py" "${DRIVE_CONFIG_DIR}/fit_svi_cond_by_cond_gamma_omega_fixed_from_animal_svi.py"
rclone copyto "${REPO_DIR}/fit_each_condn/consolidate_svi_gamma_omega_fixed_from_animal_svi_reruns.py" "${DRIVE_CONFIG_DIR}/consolidate_svi_gamma_omega_fixed_from_animal_svi_reruns.py"
rclone copyto "${REPO_DIR}/fit_each_condn/compare_svi_cond_gamma_omega_with_npl_alpha_svi.py" "${DRIVE_CONFIG_DIR}/compare_svi_cond_gamma_omega_with_npl_alpha_svi.py"
rclone copyto "${REPO_DIR}/fit_each_condn/compare_npl_svi_vs_mse_gamma_omega_alpha_params.py" "${DRIVE_CONFIG_DIR}/compare_npl_svi_vs_mse_gamma_omega_alpha_params.py"
rclone copyto "${REPO_DIR}/fit_each_condn/compare_npl_svi_vs_mse_params_rt_choice_loglike.py" "${DRIVE_CONFIG_DIR}/compare_npl_svi_vs_mse_params_rt_choice_loglike.py"
rclone copyto "${REPO_DIR}/scripts/backup_cond_svi_gamma_omega_to_drive_20260625.sh" "${DRIVE_CONFIG_DIR}/backup_cond_svi_gamma_omega_to_drive_20260625.sh"
rclone copyto "${REPO_DIR}/FIT_BACKUP_LEDGER.md" "${DRIVE_LEDGER_DIR}/FIT_BACKUP_LEDGER.md"

echo
echo "Remote verification:"
echo "Remote condition SVI files: $(rclone lsf "${DRIVE_COND_DIR}" --recursive --files-only | wc -l)"
echo "Remote comparison files: $(rclone lsf "${DRIVE_COMPARISON_DIR}" --recursive --files-only | wc -l)"
echo "Remote config files: $(rclone lsf "${DRIVE_CONFIG_DIR}" --files-only | wc -l)"
rclone size "${DRIVE_PARENT}"

echo "Finished: $(date --iso-8601=seconds)"
