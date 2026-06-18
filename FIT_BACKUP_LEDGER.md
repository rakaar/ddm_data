# Fit Backup Ledger

Last updated: 2026-06-18 00:41 WEST

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
- `LED34/57`, `LED34/59`, `LED34/61`, `LED34/63` (follow-up run)
- `SD/52`, `SD/53` (follow-up run)
- `LED7/92`, `LED7/93`, `LED7/98`, `LED7/99`, `LED7/100`, `LED7/103`
- `LED8/105`, `LED8/107`, `LED8/108`, `LED8/109`, `LED8/112`

Remote result paths:
- Upstream results: `/home/rlab/raghavendra/npl_alpha_ild2_abl_specific_delay/fit_animal_by_animal/NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- Condition pickles: `/home/rlab/raghavendra/npl_alpha_ild2_abl_specific_delay/fit_each_condn/each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_pkl_files/`
- Condition corner plots: `/home/rlab/raghavendra/npl_alpha_ild2_abl_specific_delay/fit_each_condn/each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_corner_plots/`

Copied to lavos at:
- Upstream results: `/home/rlab/raghavendra/ddm_data/ganon_NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- Condition results: `/home/rlab/raghavendra/ddm_data/ganon_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2/`
- SD52/SD53 follow-up staging upstream: `/home/rlab/raghavendra/ddm_data/ganon_SD52_SD53_NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- SD52/SD53 follow-up staging condition results: `/home/rlab/raghavendra/ddm_data/ganon_SD52_SD53_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2/`
- SD52/SD53 follow-up files were also copied into the standard lavos result folders listed under `lavos`.
- Pre-merge backup of old local SD52/SD53 upstream files: `/home/rlab/raghavendra/ddm_data/pre_merge_lavos_nonstable_SD52_SD53_20260605_171314/`

Copied local counts:
- Original ganon copy: 45 upstream files, 450 condition pickles, 450 condition corner plots.
- LED34 follow-up copy on 2026-06-05: 12 upstream files, 120 condition pickles, 120 condition corner plots.
- Combined copied ganon folders after LED34 follow-up: 57 upstream files, 570 condition pickles, 570 condition corner plots.
- LED34 follow-up copied sizes: upstream 1.2G, condition pickles 713M, condition corner plots 79M.
- SD52/SD53 follow-up copy on 2026-06-05: 6 upstream files, 48 condition pickles, 48 condition corner plots.
- SD52/SD53 follow-up staged sizes: upstream 857M, condition results 293M.
- SD52/SD53 pre-merge local backup counts: 6 upstream files, 0 condition pickles, 0 condition corner plots.

Convergence status:
- Original ganon split: upstream fits 15/15 converged; condition fits 15/15 animals have 30/30 converged conditions.
- LED34 follow-up: files present for 4/4 animals with 30 condition pickles and 30 corner plots per animal.
- SD52/SD53 follow-up: upstream fits 2/2 ended with stable VBMC messages; condition watcher logs show 24 completed fits per animal, 6 low-trial skips per animal, 0 errors, and return code 0.

### lavos

Machine:
- Local host path: `/home/rlab/raghavendra/ddm_data`

Lavos run batches:
- Local/imported fit set is complete for the 30-animal campaign.

Known completed upstream fits:
- `LED34/45`
- `LED6/81`, `LED6/82`, `LED6/84`, `LED6/86`
- `SD/48`, `SD/49`, `SD/50`, `SD/55`
- `SD/52`, `SD/53` imported from ganon follow-up run

Local result paths:
- Upstream results: `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- Condition pickles: `/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_pkl_files/`
- Condition corner plots: `/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_corner_plots/`

### Consolidated all-30 lavos artifacts

Canonical local folders for analysis:
- Upstream results: `/home/rlab/raghavendra/ddm_data/all_30_NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- Condition pickles: `/home/rlab/raghavendra/ddm_data/all_30_cond_by_cond_gamma_omega_t_E_aff_fix_w_del_go_from_NPL_alpha_ABL_specific_ILD2_fit_results/pkl_files/`
- Condition corner plots: `/home/rlab/raghavendra/ddm_data/all_30_cond_by_cond_gamma_omega_t_E_aff_fix_w_del_go_from_NPL_alpha_ABL_specific_ILD2_fit_results/corner_plots/`
- Consolidation manifest: `/home/rlab/raghavendra/ddm_data/all_30_fit_consolidation_manifest.md`

Consolidated counts:
- Upstream files: 90 files, 11G.
- Condition pickles: 864 files.
- Condition corner plots: 864 files.
- Condition folder size: 5.6G.

Consolidated validation:
- Upstream result pickles represent 30/30 animals and all 30 have stable NPL + alpha + ABL-specific ILD2 VBMC messages.
- Condition outputs represent 30/30 animals: non-SD animals have 30 conditions each, and SD animals have 24 fitted conditions each because `ILD=+/-16` had no trials.
- Duplicate source outputs were resolved by fixed source selection; the short local `LED34/57` PDF was excluded in favor of the ganon PDF.
- On 2026-06-16, condition fits `LED7/92, ABL=20, ILD=+1` and `LED7/92, ABL=20, ILD=-1` were rerun locally on lavos and replaced in the consolidated all-30 condition folders. The `ILD=-1` rerun fixed a boundary-mode `t_E_aff` fit; refreshed posterior mean was about 53 ms.

## Google Drive Backups

rclone remote:
- `raga:`

Central Drive ledger:
- `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`

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
- Completed on 2026-06-04 18:54 WEST.
- Observed remote counts: 45 upstream files, 450 condition pickles, 450 condition corner plots.
- Observed remote size: 945 objects, 6.397 GiB.

### ganon LED34 follow-up backup started 2026-06-05

Drive parent folder:
- `raga:ddm_fit_backups_20260605_ganon_LED34/`

Drive subfolders:
- `raga:ddm_fit_backups_20260605_ganon_LED34/ganon_LED34_NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- `raga:ddm_fit_backups_20260605_ganon_LED34/ganon_LED34_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2/pkl_files/`
- `raga:ddm_fit_backups_20260605_ganon_LED34/ganon_LED34_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2/corner_plots/`

Upload command wrapper:
- `/home/rlab/raghavendra/ddm_data/scripts/backup_ganon_led34_fit_results_to_drive.sh`

Upload tmux session:
- `gdrive_ganon_led34_backup`

Upload log:
- `gdrive_ganon_led34_backup_20260605_004642.log`
- `gdrive_ganon_led34_backup_20260605_005029_retry.log`

Expected upload counts:
- Upstream folder: 12 files
- Condition pkl folder: 120 `.pkl` files
- Condition corner folder: 120 files

Status:
- Copied from ganon to lavos on 2026-06-05 with observed local counts 12/120/120.
- Google Drive upload completed on 2026-06-05 00:51 WEST.
- Observed remote counts: 12 upstream files, 120 condition pickles, 120 condition corner plots.
- Observed remote size: 252 objects, 1.949 GiB.
- Note: first upload log completed upstream and condition pickles, then exited before corner plots because the wrapper was edited while the shell was still reading it. The retry log completed corner plots and final remote verification.

### ganon SD52/SD53 follow-up backup completed 2026-06-05

Drive parent folder:
- `raga:ddm_fit_backups_20260605_ganon_SD52_SD53/`

Drive subfolders:
- `raga:ddm_fit_backups_20260605_ganon_SD52_SD53/ganon_SD52_SD53_NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- `raga:ddm_fit_backups_20260605_ganon_SD52_SD53/ganon_SD52_SD53_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2/pkl_files/`
- `raga:ddm_fit_backups_20260605_ganon_SD52_SD53/ganon_SD52_SD53_cond_by_cond_gamma_omega_t_E_aff_fit_w_del_go_fixed_from_NPL_alpha_ABL_specific_ILD2/corner_plots/`

Upload log:
- `gdrive_ganon_sd52_sd53_backup_20260605_172117.log`

Expected upload counts:
- Upstream folder: 6 files
- Condition pkl folder: 48 `.pkl` files
- Condition corner folder: 48 files

Status:
- Copied from ganon to lavos on 2026-06-05 with observed local counts 6/48/48.
- Google Drive upload completed on 2026-06-05 17:23 WEST.
- Observed remote counts: 6 upstream files, 48 condition pickles, 48 condition corner plots.
- Observed remote size: 102 objects, 1.122 GiB.

### all-30 consolidated backup completed 2026-06-05

Drive parent folder:
- `raga:ddm_fit_backups_20260605_all_30_animals_ABL_specific_ILD2/`

Drive subfolders:
- `raga:ddm_fit_backups_20260605_all_30_animals_ABL_specific_ILD2/all_30_NPL_alpha_ABL_specific_ILD2_delay_fit_results/`
- `raga:ddm_fit_backups_20260605_all_30_animals_ABL_specific_ILD2/all_30_cond_by_cond_gamma_omega_t_E_aff_fix_w_del_go_from_NPL_alpha_ABL_specific_ILD2_fit_results/pkl_files/`
- `raga:ddm_fit_backups_20260605_all_30_animals_ABL_specific_ILD2/all_30_cond_by_cond_gamma_omega_t_E_aff_fix_w_del_go_from_NPL_alpha_ABL_specific_ILD2_fit_results/corner_plots/`
- `raga:ddm_fit_backups_20260605_all_30_animals_ABL_specific_ILD2/all_30_fit_consolidation_manifest.md`

Upload tmux session:
- `gdrive_all30_abl_specific_backup`

Upload log:
- `gdrive_all30_abl_specific_backup_20260605_203021.log`

Expected upload counts:
- Upstream folder: 90 files
- Condition pkl folder: 864 `.pkl` files
- Condition corner folder: 864 files
- Manifest: 1 file

Status:
- Consolidated locally on lavos on 2026-06-05 with observed local counts 90/864/864 plus manifest.
- Google Drive upload completed on 2026-06-05 20:51 WEST.
- Observed remote counts: 90 upstream files, 864 condition pickles, 864 condition corner plots, 1 manifest.
- Observed remote size: 1819 objects, 16.116 GiB.

Verification on 2026-06-16 10:01 WEST:
- Local lavos upstream folder has 90 files, including 30 result `.pkl` files.
- Local lavos condition folders have 864 condition `.pkl` files and 864 corner plot files.
- Local consolidation manifest is present.
- Google Drive all-30 backup has 90 upstream files, 864 condition `.pkl` files, 864 condition corner plot files, and `all_30_fit_consolidation_manifest.md`.
- Google Drive all-30 backup size is 1819 objects, 16.116 GiB.
- Status for the four requested checks: all 30 upstream fits are present on lavos; upstream fits are backed up on Google Drive; all 30 animals' condition-by-condition outputs are present on lavos; condition-by-condition outputs are backed up on Google Drive.

Replacement update on 2026-06-16 13:14 WEST:
- Local replacement source: `/home/rlab/raghavendra/ddm_data/fit_each_condn/each_animal_cond_fit_3_params_fix_w_del_go_from_abl_specific_ild2_pkl_files/` and matching corner plot folder.
- Consolidated replacement destination: `/home/rlab/raghavendra/ddm_data/all_30_cond_by_cond_gamma_omega_t_E_aff_fix_w_del_go_from_NPL_alpha_ABL_specific_ILD2_fit_results/`.
- Replaced files:
  - `vbmc_cond_by_cond_LED7_92_20_ILD_1_FIX_w_del_go_FROM_ABL_SPECIFIC_ILD2_3_params.pkl`
  - `vbmc_cond_by_cond_LED7_92_20_ILD_-1_FIX_w_del_go_FROM_ABL_SPECIFIC_ILD2_3_params.pkl`
  - `corner_cond_by_cond_LED7_92_20_ILD_1_FIX_w_del_go_FROM_ABL_SPECIFIC_ILD2_3_params.png`
  - `corner_cond_by_cond_LED7_92_20_ILD_-1_FIX_w_del_go_FROM_ABL_SPECIFIC_ILD2_3_params.png`
- Google Drive backup updated in place with `rclone copy`, not `rclone sync`.
- Upload log: `gdrive_all30_led7_92_condition_replacement_20260616_131345.log`.
- Observed remote counts after replacement: 864 condition pickles, 864 condition corner plots.
- Observed all-30 remote size after replacement: 1819 objects, 17,291,656,095 bytes.
- Refreshed analysis cache and figures in `/home/rlab/raghavendra/ddm_data/fit_each_condn/abl_specific_ild2_delay_agreement/`.

### MkDocs result book backup completed 2026-06-16

Drive parent folder:
- `raga:ddm_result_book_backups_20260616_lavos/`

Drive subfolders:
- `raga:ddm_result_book_backups_20260616_lavos/docs/`
- `raga:ddm_result_book_backups_20260616_lavos/config/`
- `raga:ddm_result_book_backups_20260616_lavos/codex_skills/update-result-book/`

Upload tmux session:
- `gdrive_result_book_backup_20260616_125035`

Upload log:
- `gdrive_result_book_backup_20260616_125035.log`

Expected upload counts:
- MkDocs docs folder: 6 files
- Config files: 3 files (`mkdocs.yml`, `RESULT_BOOK_AGENT_SETUP.md`, `FIT_BACKUP_LEDGER.md`)
- Codex `update-result-book` skill: 3 files

Status:
- Backed up from lavos on 2026-06-16.
- Local source paths: `/home/rlab/raghavendra/ddm_data/docs`, `/home/rlab/raghavendra/ddm_data/mkdocs.yml`, `/home/rlab/raghavendra/ddm_data/RESULT_BOOK_AGENT_SETUP.md`, `/home/rlab/raghavendra/ddm_data/FIT_BACKUP_LEDGER.md`, `/home/rlab/.codex/skills/update-result-book`.
- Google Drive upload completed on 2026-06-16 12:51 WEST.
- Observed remote count: 12 files.
- Observed remote size: 12 objects, 1.056 MiB.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`, so Drive-side files were not deletion-mirrored.

Refresh on 2026-06-16 13:17 WEST:
- Updated the result-book figure after replacing the rerun `LED7/92, ABL=20, ILD=+/-1` condition fits and regenerating the delay-agreement analysis.
- Refreshed Drive folder: `raga:ddm_result_book_backups_20260616_lavos/`.
- Upload log: `gdrive_result_book_refresh_20260616_131642.log`.
- Used `rclone copy` / `rclone copyto`; explicitly deleted only the two superseded result-book image assets from the Drive backup:
  - `cond_t_E_aff_vs_npl_alpha_abl_specific_ild2_delay_observed_range.png`
  - `cond_t_E_aff_vs_npl_alpha_abl_specific_ild2_delay_full_range.png`
- Observed remote count after refresh: 12 files.
- Observed remote size after refresh: 12 objects, 450,073 bytes.

Refresh on 2026-06-16 13:38 WEST:
- Added the Gamma/Omega comparison figure generated by `fit_each_condn/compare_cond_gamma_omega_with_npl_alpha_abl_specific_ild2_delay.py`.
- Result-book asset added locally and on Drive: `cond_gamma_omega_vs_npl_alpha_abl_specific_ild2_delay.png`.
- Refreshed Drive folder: `raga:ddm_result_book_backups_20260616_lavos/`.
- Upload log: `gdrive_result_book_gamma_omega_refresh_20260616_133812.log`.
- Used `rclone copy` / `rclone copyto`, not `rclone sync`.
- Observed remote count after refresh: 13 files.
- Observed remote size after refresh: 13 objects, 727,557 bytes.

Refresh on 2026-06-16 16:58 WEST:
- Added the unconstrained MSE delay comparison figure generated by `fit_each_condn/compare_t_E_aff_with_npl_alpha_abl_specific_ild2_and_mse_delay.py`.
- Result-book asset added locally and on Drive: `cond_t_E_aff_vs_npl_alpha_abl_specific_ild2_and_mse_delay.png`.
- The plotted MSE delay fits use the same ABL-specific `bias + abs(ILD) + ILD^2` form as the NPL delay but are fit by unconstrained least squares separately for each animal and ABL.
- Refreshed Drive folder: `raga:ddm_result_book_backups_20260616_lavos/`.
- Upload log: `gdrive_result_book_mse_delay_refresh_20260616_165824.log`.
- Used `rclone copy` / `rclone copyto`, not `rclone sync`.
- Observed remote count after refresh: 15 files.
- Observed remote size after refresh: 15 objects, 1,344,087 bytes.

### MkDocs result book daily cron backup configured 2026-06-16

Cron schedule:
- `30 2 * * * /usr/bin/bash /home/rlab/raghavendra/ddm_data/scripts/backup_result_book_to_drive >/dev/null 2>&1`

Backup script:
- `/home/rlab/raghavendra/ddm_data/scripts/backup_result_book_to_drive`

Daily Drive parent pattern:
- `raga:ddm_result_book_daily_backups_lavos/YYYYMMDD/`

Manual test run on 2026-06-16:
- Destination: `raga:ddm_result_book_daily_backups_lavos/20260616/`
- Observed remote count: 12 files.
- Observed remote size after ledger refresh: 12 objects, about 334 KiB.
- Log pattern: `logs/result_book_drive_backup_YYYYMMDD_HHMMSS.log`
- Upload uses `rclone copy` / `rclone copyto`, not `rclone sync`.

The daily backup includes:
- MkDocs docs folder.
- `mkdocs.yml`.
- `RESULT_BOOK_AGENT_SETUP.md`.
- `FIT_BACKUP_LEDGER.md`.
- `scripts/backup_result_book_to_drive`.
- Codex `update-result-book` skill folder.

### all-30 fixed condition t_E_aff backup completed 2026-06-18

Fit family:
- NPL + alpha animal-wise refit with condition-by-condition `t_E_aff` fixed from the condition cache.
- Fitted parameters: `rate_lambda`, `T_0`, `theta_E`, `w`, `del_go`, `rate_norm_l`, `alpha`.

Source machines:
- lavos supplied 3 stable fits: `LED34/45`, `LED34/57`, `LED34/59`.
- ganon supplied 27 stable fits:
  - original ganon split: `LED34_even/48`, `LED34_even/52`, `LED34_even/56`, `LED34_even/60`, `LED8/105`, `LED8/107`, `LED8/108`, `LED8/109`, `LED8/112`, `SD/48`, `SD/49`, `SD/50`, `SD/52`, `SD/53`, `SD/55`
  - lavos remaining animals moved to ganon: `LED34/61`, `LED34/63`, `LED6/81`, `LED6/82`, `LED6/84`, `LED6/86`, `LED7/92`, `LED7/93`, `LED7/98`, `LED7/99`, `LED7/100`, `LED7/103`

Canonical local lavos folder:
- `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30/`

Drive parent folder:
- `raga:ddm_fit_backups_20260618_fixed_condition_t_E_aff_all30_lavos/`

Drive subfolders:
- `raga:ddm_fit_backups_20260618_fixed_condition_t_E_aff_all30_lavos/NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30/`
- `raga:ddm_fit_backups_20260618_fixed_condition_t_E_aff_all30_lavos/config/`

Upload command wrapper:
- `/home/rlab/raghavendra/ddm_data/scripts/backup_fixed_condition_teaff_all30_to_drive_20260618.sh`

Upload tmux session:
- `gdrive_fixed_teaff_all30_backup_20260618_003958`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_fixed_teaff_all30_backup_20260618_003958.log`

Expected local counts:
- Result pickles: 30
- Non-empty result PDFs: 30
- VBMC posterior files: 30
- Fit logs: 3
- Consolidation summary: 1
- Total files in consolidated result folder: 94

Status:
- Local convergence check: 30/30 stable, 0 missing, 0 unstable.
- Local consolidated folder size: 2.3G.
- Google Drive upload completed on 2026-06-18 00:41 WEST.
- Observed remote result files: 94.
- Observed remote config files: 3 (`FIT_BACKUP_LEDGER.md`, `DISTRIBUTED_FIXED_DELAY_FIT_LEDGER.md`, `backup_fixed_condition_teaff_all30_to_drive_20260618.sh`).
- Observed remote total: 97 objects, 2.238 GiB.
- A central Drive copy of this ledger was created/updated at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.
