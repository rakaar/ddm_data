#!/usr/bin/env bash
set -euo pipefail

cd /home/rlab/raghavendra/ddm_data

RCLONE_REMOTE="${RCLONE_REMOTE:-raga:}"
DRIVE_PARENT="${DRIVE_PARENT:-ddm_fit_backups_20260605_ganon_LED34}"

UPSTREAM_LOCAL="ganon_NPL_alpha_ABL_specific_ILD2_delay_fit_results"
COND_LOCAL="ganon_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2"

UPSTREAM_DRIVE="${RCLONE_REMOTE}${DRIVE_PARENT}/ganon_LED34_NPL_alpha_ABL_specific_ILD2_delay_fit_results"
COND_DRIVE="${RCLONE_REMOTE}${DRIVE_PARENT}/ganon_LED34_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2"

UPSTREAM_FILTER=(
  "--filter=+ *LED34_animal_57*"
  "--filter=+ *LED34_animal_59*"
  "--filter=+ *LED34_animal_61*"
  "--filter=+ *LED34_animal_63*"
  "--filter=- *"
)
COND_FILTER=(
  "--filter=+ *LED34_57_*"
  "--filter=+ *LED34_59_*"
  "--filter=+ *LED34_61_*"
  "--filter=+ *LED34_63_*"
  "--filter=- *"
)

echo "$(date): starting ganon LED34 Google Drive backup"
echo "remote: ${RCLONE_REMOTE}"
echo "parent: ${DRIVE_PARENT}"
echo "upstream drive: ${UPSTREAM_DRIVE}"
echo "condition drive: ${COND_DRIVE}"

echo "$(date): local LED34 source counts"
find "${UPSTREAM_LOCAL}" -maxdepth 1 -type f \( -name "*LED34_animal_57*" -o -name "*LED34_animal_59*" -o -name "*LED34_animal_61*" -o -name "*LED34_animal_63*" \) | wc -l
find "${COND_LOCAL}/pkl_files" -maxdepth 1 -type f \( -name "*LED34_57_*" -o -name "*LED34_59_*" -o -name "*LED34_61_*" -o -name "*LED34_63_*" \) | wc -l
find "${COND_LOCAL}/corner_plots" -maxdepth 1 -type f \( -name "*LED34_57_*" -o -name "*LED34_59_*" -o -name "*LED34_61_*" -o -name "*LED34_63_*" \) | wc -l

echo "$(date): creating destination folders"
rclone mkdir "${RCLONE_REMOTE}${DRIVE_PARENT}"
rclone mkdir "${UPSTREAM_DRIVE}"
rclone mkdir "${COND_DRIVE}/pkl_files"
rclone mkdir "${COND_DRIVE}/corner_plots"

echo "$(date): uploading LED34 upstream NPL+alpha+ABL-specific ILD2 results"
rclone copy "${UPSTREAM_LOCAL}" "${UPSTREAM_DRIVE}" \
  "${UPSTREAM_FILTER[@]}" \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 64M \
  --progress \
  --stats 30s \
  --log-level INFO

echo "$(date): uploading LED34 condition-by-condition pickle files"
rclone copy "${COND_LOCAL}/pkl_files" "${COND_DRIVE}/pkl_files" \
  "${COND_FILTER[@]}" \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 64M \
  --progress \
  --stats 30s \
  --log-level INFO

echo "$(date): uploading LED34 condition-by-condition corner plots"
rclone copy "${COND_LOCAL}/corner_plots" "${COND_DRIVE}/corner_plots" \
  "${COND_FILTER[@]}" \
  --transfers 4 \
  --checkers 8 \
  --drive-chunk-size 64M \
  --progress \
  --stats 30s \
  --log-level INFO

echo "$(date): remote counts"
rclone lsf "${UPSTREAM_DRIVE}" --files-only | wc -l
rclone lsf "${COND_DRIVE}/pkl_files" --files-only | wc -l
rclone lsf "${COND_DRIVE}/corner_plots" --files-only | wc -l

echo "$(date): remote size summary"
rclone size "${RCLONE_REMOTE}${DRIVE_PARENT}"

echo "$(date): done"
