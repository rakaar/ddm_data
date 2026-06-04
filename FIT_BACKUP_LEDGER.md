# Fit Backup Ledger

Last updated: 2026-06-04 18:40 WEST

## Current Fit Campaign

Model family:
- Upstream fit: NPL + alpha + ABL-specific ILD2 delay.
- Condition fit: condition-by-condition gamma, omega, t_E_aff with w and del_go fixed from the upstream fit.

## Machines And Local Paths

### ganon

Machine:
- SSH: `rlab@ganon`
- Project path: `/home/rlab/raghavendra/npl_alpha_ild2_abl_specific_delay`

Animals completed on ganon:
- `LED34_even/48`, `LED34_even/52`, `LED34_even/56`, `LED34_even/60`
- `LED7/92`, `LED7/93`, `LED7/98`, `LED7/99`, `LED7/100`, `LED7/103`
- `LED8/105`, `LED8/107`, `LED8/108`, `LED8/109`, `LED8/112`

Remote result paths:
- Upstream results: `/home/rlab/raghavendra/npl_alpha_ild2_abl_specific_delay/fit_animal_by_animal/NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- Condition pickles: `/home/rlab/raghavendra/npl_alpha_ild2_abl_specific_delay/fit_each_condn/each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_pkl_files/`
- Condition corner plots: `/home/rlab/raghavendra/npl_alpha_ild2_abl_specific_delay/fit_each_condn/each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_corner_plots/`

Copied to lavos at:
- Upstream results: `/home/rlab/raghavendra/ddm_data/ganon_NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- Condition results: `/home/rlab/raghavendra/ddm_data/ganon_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2/`

Copied local counts:
- Upstream files: 45 files, 3.5G
- Condition pickles: 450 files
- Condition corner plots: 450 files
- Condition folder size: 3.0G

Convergence status:
- Upstream fits: 15/15 converged.
- Condition fits: 15/15 animals have 30/30 converged conditions.

### lavos

Machine:
- Local host path: `/home/rlab/raghavendra/ddm_data`

Lavos run batches:
- Current foreground upstream/watcher session was launched for `LED6,SD`.
- It will not automatically run remaining `LED34` animals unless restarted with that batch included.

Known completed upstream fits at last check:
- `LED34/45`
- `LED6/81`, `LED6/82`, `LED6/84`, `LED6/86`
- `SD/48`

Known in-progress upstream fit at last check:
- `SD/49`

Known remaining or not yet complete at last check:
- `LED34/57`, `LED34/59`, `LED34/61`, `LED34/63`
- `SD/49`, `SD/50`, `SD/52`, `SD/53`, `SD/55`

Local result paths:
- Upstream results: `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- Condition pickles: `/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_pkl_files/`
- Condition corner plots: `/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_corner_plots/`

## Google Drive Backups

rclone remote:
- `raga:`

### ganon backup started 2026-06-04

Drive parent folder:
- `raga:ddm_fit_backups_20260604_ganon/`

Drive subfolders:
- `raga:ddm_fit_backups_20260604_ganon/ganon_NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- `raga:ddm_fit_backups_20260604_ganon/ganon_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2/pkl_files/`
- `raga:ddm_fit_backups_20260604_ganon/ganon_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2/corner_plots/`

Upload command wrapper:
- `/home/rlab/raghavendra/ddm_data/scripts/backup_ganon_fit_results_to_drive.sh`

Upload tmux session:
- `gdrive_ganon_backup`

Upload log:
- `gdrive_ganon_backup_20260604_184217.log`

Expected upload counts:
- Upstream folder: 45 files
- Condition pkl folder: 450 `.pkl` files
- Condition corner folder: 450 files

Status:
- Started in tmux on 2026-06-04 18:42 WEST.
