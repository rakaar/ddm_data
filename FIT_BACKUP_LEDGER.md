# Fit Backup Ledger

Last updated: 2026-07-07 10:54 WEST

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

### NumPyro SVI condition-delay backup completed 2026-06-24

Fit family:
- NumPyro SVI NPL + alpha model with shared animal-wise parameters and per-condition `t_E_aff`.
- Main completed guide label: `main_fullrank`.
- Repository commit containing scripts/result-book entries: `024894f`.

Source machine:
- lavos.

Local source paths:
- Per-animal SVI outputs: `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/numpyro_svi_npl_alpha_condition_delay_single_animal_outputs/`
- All-animal SVI diagnostics: `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/numpyro_svi_npl_alpha_condition_delay_all_animals_diagnostics/`
- Fixed-condition `t_E_aff` versus ABL-specific ILD2 parameter comparison outputs: `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/fixed_condition_t_E_aff_vs_abl_specific_ild2_param_comparison/`

Drive parent folder:
- `raga:ddm_fit_backups_20260624_numpyro_svi_condition_delay_lavos/`

Drive subfolders:
- `raga:ddm_fit_backups_20260624_numpyro_svi_condition_delay_lavos/numpyro_svi_npl_alpha_condition_delay_single_animal_outputs/`
- `raga:ddm_fit_backups_20260624_numpyro_svi_condition_delay_lavos/numpyro_svi_npl_alpha_condition_delay_all_animals_diagnostics/`
- `raga:ddm_fit_backups_20260624_numpyro_svi_condition_delay_lavos/fixed_condition_t_E_aff_vs_abl_specific_ild2_param_comparison/`
- `raga:ddm_fit_backups_20260624_numpyro_svi_condition_delay_lavos/config/`

Upload tmux session:
- `gdrive_numpyro_svi_backup_20260624_164544`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_numpyro_svi_backup_20260624_164544.log`

Expected local counts and sizes:
- Per-animal SVI outputs: 505 files, 439M.
- All-animal SVI diagnostics: 7 files, 27M.
- Parameter comparison outputs: 7 files, 1.6M.

Status:
- Google Drive upload completed on 2026-06-24 16:51 WEST.
- Observed remote per-animal SVI output files: 505.
- Observed remote all-animal diagnostic files: 7.
- Observed remote parameter-comparison output files: 7.
- Observed remote total before config ledger copy: 519 objects, 465.334 MiB.
- Readable config copies: `config/gdrive_numpyro_svi_backup_20260624_164544.log` and `config/FIT_BACKUP_LEDGER_20260624.md`.
- Observed remote total after config copies: 522 objects, 465.695 MiB.
- A central Drive copy of this ledger was updated at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### Condition-by-condition Gamma/Omega SVI backup completed 2026-06-25

Fit family:
- NumPyro SVI condition-by-condition Gamma/Omega fits with `w`, `del_go`, and each condition's `t_E_aff` fixed from the matching animal-wise NPL+alpha condition-delay SVI posterior means.
- Includes the original all-observed run, selected extra-step reruns, and the consolidated `all_observed_with_30k_reruns` result folder used for the Gamma/Omega comparison figures.

Source machine:
- lavos.

Local source paths:
- Condition SVI outputs: `/home/rlab/raghavendra/ddm_data/fit_each_condn/svi_gamma_omega_fixed_from_animal_svi_condition_delay_results/`
- Gamma/Omega and likelihood comparison outputs: `/home/rlab/raghavendra/ddm_data/fit_each_condn/svi_condition_gamma_omega_vs_npl_alpha_svi_comparison/`

Drive parent folder:
- `raga:ddm_fit_backups_20260625_cond_svi_gamma_omega_fixed_from_animal_svi_lavos/`

Drive subfolders:
- `raga:ddm_fit_backups_20260625_cond_svi_gamma_omega_fixed_from_animal_svi_lavos/svi_gamma_omega_fixed_from_animal_svi_condition_delay_results/`
- `raga:ddm_fit_backups_20260625_cond_svi_gamma_omega_fixed_from_animal_svi_lavos/svi_condition_gamma_omega_vs_npl_alpha_svi_comparison/`
- `raga:ddm_fit_backups_20260625_cond_svi_gamma_omega_fixed_from_animal_svi_lavos/config/`

Upload command wrapper:
- `/home/rlab/raghavendra/ddm_data/scripts/backup_cond_svi_gamma_omega_to_drive_20260625.sh`

Upload tmux session:
- `gdrive_cond_svi_backup_20260625_171806`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_cond_svi_backup_20260625_171806.log`

Expected local counts and sizes:
- Condition SVI outputs: 451 files, 661M.
  - `all_observed`: 184 files, 317M.
  - `all_observed_with_30k_reruns`: 186 files, 324M.
  - `custom_conditions`: 10 files, 2.2M.
  - `nonstable_extra_steps_30k`: 71 files, 18M.
- Gamma/Omega and likelihood comparison outputs: 22 files, 3.1M.

Status:
- Google Drive upload completed on 2026-06-25 17:22 WEST.
- Observed remote condition SVI files: 451.
- Observed remote comparison output files: 22.
- Observed remote config files after final ledger/log refresh: 10.
- Observed remote total after final ledger/log refresh: 483 objects, 662.736 MiB.
- The 2026-06-24 NPL SVI backup was rechecked on 2026-06-25 before this upload: 505 per-animal SVI files and 522 total objects were present at `raga:ddm_fit_backups_20260624_numpyro_svi_condition_delay_lavos/`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### Big Gamma/Omega/delay SVI backup completed 2026-06-26

Fit family:
- NumPyro SVI direct Gamma/Omega model with condition-wise `gamma`, `omega`, and `t_E_aff`, plus global `w` and `del_go`.
- Includes the completed all-30-animal run, the LED8/105 single-animal smoke fit, all-animal condition-parameter summaries, and the per-animal MSE alpha-model comparison outputs derived from the big SVI condition means.

Source machine:
- lavos.

Local source paths:
- All-animal big SVI outputs: `/home/rlab/raghavendra/ddm_data/fit_each_condn/svi_big_gamma_omega_delay_all_animals_outputs/`
- Single-animal smoke outputs: `/home/rlab/raghavendra/ddm_data/fit_each_condn/svi_big_gamma_omega_delay_single_animal_outputs/`

Drive parent folder:
- `raga:ddm_fit_backups_20260626_big_gamma_omega_delay_svi_lavos/`

Drive subfolders:
- `raga:ddm_fit_backups_20260626_big_gamma_omega_delay_svi_lavos/svi_big_gamma_omega_delay_all_animals_outputs/`
- `raga:ddm_fit_backups_20260626_big_gamma_omega_delay_svi_lavos/svi_big_gamma_omega_delay_single_animal_outputs/`
- `raga:ddm_fit_backups_20260626_big_gamma_omega_delay_svi_lavos/config/`

Upload tmux session:
- `gdrive_big_gamma_omega_delay_svi_backup_20260626_122514`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_big_gamma_omega_delay_svi_backup_20260626_122514.log`

Expected local counts and sizes:
- All-animal big SVI outputs: 323 files, 430M.
- Single-animal smoke outputs: 9 files, 7.6M.

Status:
- Google Drive upload completed on 2026-06-26 12:28 WEST.
- Observed remote all-animal big SVI output files: 323.
- Observed remote single-animal smoke output files: 9.
- Observed remote total before config ledger/log copies: 332 objects, 436.283 MiB.
- Observed remote config files after ledger/log copy: 2.
- Observed remote total after config ledger/log copies: 334 objects, 436.514 MiB.
- Readable config copies: `config/gdrive_big_gamma_omega_delay_svi_backup_20260626_122514.log` and `config/FIT_BACKUP_LEDGER_20260626.md`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### Big Gamma/Omega/delay SVI patience12 backup completed 2026-06-27

Fit family:
- NumPyro SVI direct Gamma/Omega model with condition-wise `gamma`, `omega`, and `t_E_aff`, plus global `w` and `del_go`.
- Includes the all-30-animal patience12 restore-best rerun and the six-animal 50k convergence-audit outputs used to choose/check the stopping rule.

Source machine:
- lavos.

Local source paths:
- Patience12 all-animal big SVI outputs: `/home/rlab/raghavendra/ddm_data/fit_each_condn/svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs/`
- Convergence-audit outputs: `/home/rlab/raghavendra/ddm_data/fit_each_condn/svi_big_gamma_omega_delay_convergence_audit_outputs/`

Drive parent folder:
- `raga:ddm_fit_backups_20260627_big_gamma_omega_delay_patience12_lavos/`

Drive subfolders:
- `raga:ddm_fit_backups_20260627_big_gamma_omega_delay_patience12_lavos/svi_big_gamma_omega_delay_patience12_restore_best_all_animals_outputs/`
- `raga:ddm_fit_backups_20260627_big_gamma_omega_delay_patience12_lavos/svi_big_gamma_omega_delay_convergence_audit_outputs/`
- `raga:ddm_fit_backups_20260627_big_gamma_omega_delay_patience12_lavos/config/`

Upload tmux session:
- `gdrive_big_gamma_patience12_backup_20260627_163700`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_big_gamma_patience12_backup_20260627_163700.log`

Expected local counts and sizes:
- Patience12 all-animal big SVI outputs: 310 files, 479M.
- Convergence-audit outputs: 78 files, 97M.

Status:
- Google Drive upload completed on 2026-06-27 16:40 WEST.
- Observed remote patience12 all-animal output files: 310.
- Observed remote convergence-audit output files: 78.
- Observed remote total before config ledger/log copies: 388 objects, 573.549 MiB.
- Readable config copies: `config/gdrive_big_gamma_patience12_backup_20260627_163700.log` and `config/FIT_BACKUP_LEDGER_20260627.md`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### NPL+alpha condition-delay SVI patience12 backup completed 2026-06-30

Fit family:
- NumPyro SVI NPL+alpha condition-delay model with shared per-animal NPL parameters, global `w` and `del_go`, and condition-wise `t_E_aff`.
- Includes the latest all-30-animal patience12 restore-best outputs, the six replaced `min_steps=50000` reruns already consolidated into the output root, all-animal loss/summary figures, and the generated comparisons against the patience12 92-parameter big Gamma/Omega/delay SVI.

Source machine:
- lavos.

Local source path:
- `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs/`

Drive parent folder:
- `raga:ddm_fit_backups_20260630_numpyro_svi_condition_delay_patience12_lavos/`

Drive subfolders:
- `raga:ddm_fit_backups_20260630_numpyro_svi_condition_delay_patience12_lavos/numpyro_svi_npl_alpha_condition_delay_patience12_restore_best_outputs/`
- `raga:ddm_fit_backups_20260630_numpyro_svi_condition_delay_patience12_lavos/config/`

Upload tmux session:
- `gdrive_npl37_patience12_backup_20260630_122521`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_npl37_patience12_backup_20260630_122521.log`

Expected local counts and sizes:
- Latest 37-param patience12 NPL SVI output root: 505 files, 425M.
- Top-level output folders: 30 animal folders plus `_batch_logs`, `summary_figures`, `comparison_with_big_gamma_omega_delay_patience12`, and `three_npl_param_source_comparison`.

Status:
- Google Drive upload completed on 2026-06-30 12:30 WEST.
- Observed remote 37-param patience12 output files before config copies: 505.
- Observed remote total before config copies: 505 objects, 423.232 MiB.
- Observed remote total after config copies: 507 objects, 423.574 MiB.
- Readable config copies: `config/gdrive_npl37_patience12_backup_20260630_122521.log` and `config/FIT_BACKUP_LEDGER_20260630.md`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### Vanilla/IPL condition-delay SVI patience12 min50k backup completed 2026-07-02

Fit family:
- NumPyro SVI vanilla/IPL condition-delay model with shared per-animal `rate_lambda`, `T_0`, `theta_E`, global `w` and `del_go`, and condition-wise `t_E_aff`.
- Includes the all-30-animal patience12 min50k restore-best outputs, all-animal loss/summary figures, the comparison against NPL and 92-parameter big Gamma/Omega/delay SVI fits, and the one-row Fig 2-style IPL diagnostic.

Source machine:
- lavos.

Local source path:
- `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/numpyro_svi_vanilla_condition_delay_patience12_min50k_restore_best_outputs/`

Deleted local obsolete path before backup:
- `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/numpyro_svi_vanilla_condition_delay_patience12_restore_best_outputs/`
- This was the earlier 20k IPL/vanilla run and was intentionally not backed up to avoid confusion with the 50k run.

Drive parent folder:
- `raga:ddm_fit_backups_20260702_vanilla_ipl_condition_delay_patience12_min50k_lavos/`

Drive subfolders:
- `raga:ddm_fit_backups_20260702_vanilla_ipl_condition_delay_patience12_min50k_lavos/numpyro_svi_vanilla_condition_delay_patience12_min50k_restore_best_outputs/`
- `raga:ddm_fit_backups_20260702_vanilla_ipl_condition_delay_patience12_min50k_lavos/config/`

Upload tmux session:
- `gdrive_ipl50k_backup_20260702_122439`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_ipl50k_backup_20260702_122439.log`

Expected local counts and sizes:
- Latest vanilla/IPL patience12 min50k SVI output root: 345 files, 229M.
- Top-level output folders: 30 animal folders plus `_batch_logs`, `summary_figures`, `comparison_with_npl_and_big_gamma_omega_delay_patience12`, and `fig2_like_diagnostics`.

Status:
- Google Drive upload completed on 2026-07-02 12:29 WEST.
- Observed remote vanilla/IPL min50k output files before config copies: 345.
- Observed remote total before config copies: 345 objects, 228.044 MiB.
- Observed remote total after config copies: 347 objects, 228.285 MiB.
- Readable config copies: `config/gdrive_ipl50k_backup_20260702_122439.log` and `config/FIT_BACKUP_LEDGER_20260702.md`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### NPL+alpha+lapse condition-delay SVI patience12 min50k backup completed 2026-07-06

Fit family:
- NumPyro SVI NPL+alpha+lapse condition-delay model with shared per-animal NPL+alpha parameters, global `w` and `del_go`, global lapse parameters, and condition-wise `t_E_aff`.
- This is the all-30-animal patience12 min50k restore-best output root with atypical early restored-best loss curves for some animals.
- This is the analysis root to use going forward for this fit family; do not substitute the random-plausible 100k rerun root or the refreshed VBMC scalar-`t_E_aff` runs for this result.

Source machine:
- lavos.

Local source path:
- `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/numpyro_svi_npl_alpha_lapse_condition_delay_patience12_min50k_restore_best_outputs/`

Drive parent folder:
- `raga:ddm_fit_backups_20260706_npl_alpha_lapse_condition_delay_patience12_min50k_lavos/`

Drive subfolders:
- `raga:ddm_fit_backups_20260706_npl_alpha_lapse_condition_delay_patience12_min50k_lavos/numpyro_svi_npl_alpha_lapse_condition_delay_patience12_min50k_restore_best_outputs/`
- `raga:ddm_fit_backups_20260706_npl_alpha_lapse_condition_delay_patience12_min50k_lavos/config/`

Upload tmux session:
- `gdrive_npl_alpha_lapse_min50k_backup_20260706_125058`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_npl_alpha_lapse_min50k_backup_20260706_125058.log`

Expected local counts and sizes:
- NPL+alpha+lapse patience12 min50k SVI output root: 373 files, 244M.
- Top-level output folders: 30 animal folders plus `_batch_logs`, `comparison_with_npl_no_lapse`, and `summary_figures`.

Status:
- Google Drive upload completed on 2026-07-06 12:54 WEST.
- Observed remote NPL+alpha+lapse min50k output files before config copies: 373.
- Observed remote total before config copies: 373 objects, 242.973 MiB.
- Observed remote total after config copies: 375 objects, 243.231 MiB.
- Readable config copies: `config/gdrive_npl_alpha_lapse_min50k_backup_20260706_125058.log` and `config/FIT_BACKUP_LEDGER_20260706.md`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### Gamma/Omega/delay+lapse condition SVI patience12 backup completed 2026-07-07

Fit family:
- NumPyro SVI condition-wise Gamma/Omega/delay+lapse model with condition-wise `gamma`, `omega`, and `t_E_aff`, plus shared per-animal `w`, `del_go`, `lapse_rate`, and `lapse_prob_right`.
- This is the all-30-animal patience12 restore-best condition-fit output root used to compare condition-wise Gamma/Omega fits with and without lapses.

Source machine:
- lavos.

Local source path:
- `/home/rlab/raghavendra/ddm_data/fit_each_condn/svi_big_gamma_omega_delay_lapse_patience12_restore_best_all_animals_outputs/`

Drive parent folder:
- `raga:ddm_fit_backups_20260707_big_gamma_omega_delay_lapse_condition_fit_patience12_lavos/`

Drive subfolders:
- `raga:ddm_fit_backups_20260707_big_gamma_omega_delay_lapse_condition_fit_patience12_lavos/svi_big_gamma_omega_delay_lapse_patience12_restore_best_all_animals_outputs/`
- `raga:ddm_fit_backups_20260707_big_gamma_omega_delay_lapse_condition_fit_patience12_lavos/config/`

Upload tmux session:
- `gdrive_gamma_omega_lapse_condfit_backup_20260707_104913`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_gamma_omega_lapse_condfit_backup_20260707_104913.log`

Expected local counts and sizes:
- Gamma/Omega/delay+lapse condition SVI output root: 307 files, 488M.
- Top-level output folders: 30 animal folders plus `_batch_logs` and `summary_figures`.
- Structured counts: 30 fit bundles, 30 posterior sample archives, and 30 condition summary CSVs.

Status:
- Google Drive upload completed on 2026-07-07 10:52 WEST.
- Observed remote condition-fit output files before config copies: 307.
- Observed remote total before config copies: 307 objects, 486.371 MiB.
- Readable config copies: `config/gdrive_gamma_omega_lapse_condfit_backup_20260707_104913.log` and `config/FIT_BACKUP_LEDGER_20260707.md`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### WL normalization v2 Figure 2 reproducibility backup completed 2026-07-07

Artifact family:
- Compact reproduction folder for the Fig. 2-style direct IPL/vanilla condition-delay SVI diagnostic.
- Includes the plotting script, compact panel-data pickles, bundled validation pkl, final PNG/PDF, and local README.
- Does not include the full upstream all-animal IPL SVI output root; the included plot script redraws the figure from the compact panel pickles.

Source machine:
- lavos.

Local source path:
- `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/figure2_v2/`

Drive folder:
- `raga:WL_normalization_v2/figure 2/`

Drive config folder:
- `raga:WL_normalization_v2/config/`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_wl_normalization_v2_figure2_backup_20260707_162303.log`

Expected local counts and sizes:
- Figure 2 v2 compact folder: 10 files, 3.7M.

Status:
- Google Drive upload completed on 2026-07-07 16:23 WEST.
- Observed remote `figure 2` folder after removing generated `__pycache__`: 10 files, 3.602 MiB.
- Readable config copies: `config/gdrive_wl_normalization_v2_figure2_backup_20260707_162303.log` and `config/FIT_BACKUP_LEDGER_20260707.md`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### WL normalization v2 Figure 2 and Figure 6 reproducibility backup completed 2026-07-08

Artifact families:
- Updated compact reproduction folder for the Fig. 2-style direct IPL/vanilla condition-delay SVI diagnostic.
- New compact reproduction folder for the Fig. 6 / lapses supplementary 2 x 4 SVI diagnostic.
- Includes plotting/build scripts, compact panel-data pickles/CSVs, rendered PNG/PDF outputs, and local READMEs.
- Does not include the full upstream all-animal SVI output roots; the included scripts redraw the figures from compact products and documented upstream fit roots.

Source machine:
- lavos.

Local source paths:
- `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/figure2_v2/`
- `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/supplementary_lapses_v2/`

Drive folders:
- `raga:WL_normalization_v2/figure 2/figure2_v2/`
- `raga:WL_normalization_v2/figure 6/supplementary_lapses_v2/`

Drive config folder:
- `raga:WL_normalization_v2/config/`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_wl_normalization_v2_figures_backup_20260708_122108.log`

Expected local counts and sizes:
- Figure 2 v2 compact folder: 17 files, 4.8M.
- Figure 6 / supplementary lapses v2 compact folder: 11 files, 784K.

Status:
- Google Drive upload completed on 2026-07-08 12:21 WEST.
- Observed remote `figure 2/figure2_v2` folder: 17 files, 4.723 MiB.
- Observed remote `figure 6/supplementary_lapses_v2` folder: 11 files, 751.851 KiB.
- Observed remote `WL_normalization_v2` total after upload: 40 objects, 8.954 MiB.
- Readable config copies: `config/gdrive_wl_normalization_v2_figures_backup_20260708_122108.log` and `config/FIT_BACKUP_LEDGER_20260708.md`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### WL normalization v2 Figure 4 reproducibility backup completed 2026-07-09

Artifact family:
- Compact reproduction folder for the Fig. 4-style NPL+alpha diagnostic using
  per-animal Gamma+Omega MSE parameters from the patience12 92-parameter big
  Gamma/Omega/delay SVI fit.
- Includes plotting/build scripts, compact panel-data pickles, bundled
  validation pkl, rendered PNG/PDF outputs, and local README.
- Does not include the full upstream 92-parameter big SVI output root; the
  included scripts redraw the figures from compact products and documented
  upstream fit roots.

Source machine:
- lavos.

Local source path:
- `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/figure4_v2/`

Drive folder:
- `raga:WL_normalization_v2/figure 4/figure4_v2/`

Drive config folder:
- `raga:WL_normalization_v2/config/`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_wl_normalization_v2_figure4_backup_20260709_123216.log`

Expected local counts and sizes:
- Figure 4 v2 compact folder: 13 files, 4.8M.

Status:
- Google Drive upload completed on 2026-07-09 12:32 WEST.
- Observed remote `figure 4/figure4_v2` folder: 13 files, 4.752 MiB.
- Observed remote `WL_normalization_v2` total after upload: 55 objects, 13.747 MiB.
- Readable config copies: `config/gdrive_wl_normalization_v2_figure4_backup_20260709_123216.log` and `config/FIT_BACKUP_LEDGER_20260709.md`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### WL normalization v2 direct NPL SVI supplementary Figure 4 backup completed 2026-07-10

Artifact family:
- Compact supplementary Figure 4 reproduction using the direct patience12
  37-parameter NPL+alpha condition-delay SVI fit.
- Psychometric, RT-quantile, Gamma, Omega, and slope predictions use the direct
  NPL SVI parameters; Gamma/Omega scatter targets remain the patience12
  92-parameter descriptive condition-fit posterior means.
- Includes the builder and plotting scripts, compact posterior/panel-data
  bundle, rendered PNG/PDF, and local README.
- Does not include the full upstream all-animal NPL SVI output root.

Source machine:
- lavos.

Local source path:
- `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/figure4_npl_svi_supplementary/`

Drive folder:
- `raga:WL_normalization_v2/figure 4/figure4_npl_svi_supplementary/`

Drive config folder:
- `raga:WL_normalization_v2/config/`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_wl_normalization_v2_figure4_npl_svi_supplementary_backup_20260710_155241.log`

Expected local counts and sizes:
- Supplementary Figure 4 compact reproducibility folder: 6 intended artifacts,
  2.6M locally; generated `__pycache__` files were excluded from the final
  remote folder.

Status:
- Google Drive upload completed on 2026-07-10 15:52 WEST.
- Observed remote supplementary Figure 4 folder: 6 files, 2.471 MiB.
- Observed remote `WL_normalization_v2` total after upload: 63 objects,
  16.266 MiB.
- Readable config copies: `config/gdrive_wl_normalization_v2_figure4_npl_svi_supplementary_backup_20260710_155241.log` and `config/FIT_BACKUP_LEDGER_20260710.md`.
- A central Drive copy of this ledger was refreshed at `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Upload used `rclone copy` / `rclone copyto`, not `rclone sync`.

### WL normalization v2 curated PDF collection created 2026-07-10

Artifact family:
- Publication-facing copies of the final Figure 2 v2, Figure 4 v2, Figure 6
  v2, and direct-NPL supplementary Figure 4 PDFs.
- Files were copied server-side from their existing verified
  `WL_normalization_v2` reproduction folders and given short stable names.

Source machine:
- lavos initiated the Drive-side copies.

Drive folder:
- `raga:WL_normalization_v2/PDFs/`

Source-to-destination mapping:
- `figure 2/figure2_v2/figure2_v2_ipl_svi_condition_delay.pdf` ->
  `PDFs/Figure_2_v2.pdf`.
- `figure 4/figure4_v2/figure4_v2_mse_gamma_omega_npl_alpha_with_upper_corner.pdf`
  -> `PDFs/Figure_4_v2.pdf`.
- `figure 6/supplementary_lapses_v2/outputs/svi_lapses_supp_v2_2x4.pdf`
  -> `PDFs/Figure_6_v2.pdf`.
- `figure 4/figure4_npl_svi_supplementary/figure4_supplementary_npl_svi_patience12.pdf`
  -> `PDFs/Supplementary_Figure_4_NPL_SVI_v2.pdf`.

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_wl_normalization_v2_pdfs_20260710_160414.log`

Status:
- Drive-side copies completed on 2026-07-10 16:04 WEST.
- Observed curated folder: 4 PDF files, 299.413 KiB.
- MD5 checksums matched between every source and destination PDF.
- Observed remote `WL_normalization_v2` total after the copies: 69 objects,
  16.601 MiB.
- Readable config copies: `config/gdrive_wl_normalization_v2_pdfs_20260710_160414.log`
  and the refreshed `config/FIT_BACKUP_LEDGER_20260710.md`.
- A central Drive copy of this ledger was refreshed at
  `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Copies used `rclone copyto`, not `rclone sync`.

### Figure 2 v2 PDF symbol-rendering repair completed 2026-07-10

Issue and fix:
- The Figure 2 v2 PDF used literal Unicode Gamma, omega, and minus glyphs while
  `pdf.use14corefonts=True` selected WinAnsi Helvetica, causing those glyphs to
  render as question marks.
- The plotting script now uses Matplotlib math text for `Gamma` and `omega`
  and ASCII hyphen-minus tick labels through `axes.unicode_minus=False`.

Replaced Drive files:
- `raga:WL_normalization_v2/figure 2/figure2_v2/figure2_v2_ipl_svi_condition_delay.pdf`
- `raga:WL_normalization_v2/PDFs/Figure_2_v2.pdf`

Upload log:
- `/home/rlab/raghavendra/ddm_data/logs/gdrive_wl_normalization_v2_figure2_pdf_greek_fix_20260710_160946.log`

Status:
- Corrected PDF regenerated and visually checked from a PDF raster on
  2026-07-10 16:09 WEST.
- PDF text extraction contains no question-mark replacement characters, and
  the rendered labels show `Gamma`, `omega`, and negative tick signs correctly.
- Local and both remote MD5 values match:
  `bb3c4b9cab59a45eb1c9a08f44ef1fa7`.
- Readable config copies: `config/gdrive_wl_normalization_v2_figure2_pdf_greek_fix_20260710_160946.log`
  and the refreshed `config/FIT_BACKUP_LEDGER_20260710.md`.
- A central Drive copy of this ledger was refreshed at
  `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Replacements used `rclone copyto`, not `rclone sync`.
