# Distributed Fixed-Delay NPL Alpha Fit Ledger

Created: 2026-06-17

Fit script:

`fit_animal_by_animal/animal_wise_norm_alpha_tied_fit_from_abort_params_condition_t_E_aff_fixed_delay.py`

Fit type:

NPL + alpha + ILD2/ABL model refit with `t_E_aff` fixed from condition-by-condition fits. Fitted parameters are `rate_lambda`, `T_0`, `theta_E`, `w`, `del_go`, `rate_norm_l`, and `alpha`.

Condition delay cache:

`fit_each_condn/abl_specific_ild2_delay_agreement/condition_t_E_aff_extraction_cache.csv`

## Machine Assignments

| Machine | Remote folder | Batches | Animals | Count | Status |
|---|---|---:|---|---:|---|
| lavos | `/home/rlab/raghavendra/ddm_data` | `LED34`, `LED6`, `LED7` | `LED34: 45,57,59,61,63`; `LED6: 81,82,84,86`; `LED7: 92,93,98,99,100,103` | 15 | Stopped after 3 stable: `LED34/45`, `LED34/57`, `LED34/59` |
| ganon | `/home/rlab/raghavendra/npl_alpha_condition_t_E_aff_fixed_delay` | `LED34_even`, `LED8`, `SD` | `LED34_even: 48,52,56,60`; `LED8: 105,107,108,109,112`; `SD: 48,49,50,52,53,55` | 15 | Complete; all 15 stable as of 2026-06-17 |
| ganon | `/home/rlab/raghavendra/npl_alpha_condition_t_E_aff_fixed_delay` | remaining from lavos assignment: `LED34`, `LED6`, `LED7` | `LED34: 61,63`; `LED6: 81,82,84,86`; `LED7: 92,93,98,99,100,103` | 12 | Complete; all 12 stable as of 2026-06-18 |

## Output Folders

lavos:

`/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/NPL_alpha_condition_t_E_aff_fixed_delay_fit_results`

ganon:

`/home/rlab/raghavendra/npl_alpha_condition_t_E_aff_fixed_delay/fit_animal_by_animal/NPL_alpha_condition_t_E_aff_fixed_delay_fit_results`

## Run Commands

lavos:

```bash
cd /home/rlab/raghavendra/ddm_data
DESIRED_BATCHES_OVERRIDE=LED34,LED6,LED7 \
SKIP_FINISHED_FITS=1 \
RUN_DIAGNOSTICS_AFTER_FIT=0 \
.venv/bin/python -u fit_animal_by_animal/animal_wise_norm_alpha_tied_fit_from_abort_params_condition_t_E_aff_fixed_delay.py \
2>&1 | tee fit_animal_by_animal/NPL_alpha_condition_t_E_aff_fixed_delay_fit_results/lavos_LED34_LED6_LED7_$(date +%Y%m%d_%H%M%S).log
```

ganon:

```bash
cd /home/rlab/raghavendra/npl_alpha_condition_t_E_aff_fixed_delay
DESIRED_BATCHES_OVERRIDE=LED34_even,LED8,SD \
SKIP_FINISHED_FITS=1 \
RUN_DIAGNOSTICS_AFTER_FIT=0 \
.venv/bin/python -u fit_animal_by_animal/animal_wise_norm_alpha_tied_fit_from_abort_params_condition_t_E_aff_fixed_delay.py \
2>&1 | tee logs/fixed_condition_delay_ganon_LED34even_LED8_SD_$(date +%Y%m%d_%H%M%S).log
```

ganon, remaining animals moved from lavos:

```bash
cd /home/rlab/raghavendra/npl_alpha_condition_t_E_aff_fixed_delay
DESIRED_BATCHES_OVERRIDE=LED34,LED6,LED7 \
TEST_BATCH_ANIMAL_PAIRS=LED34:61,LED34:63,LED6:81,LED6:82,LED6:84,LED6:86,LED7:92,LED7:93,LED7:98,LED7:99,LED7:100,LED7:103 \
SKIP_FINISHED_FITS=1 \
RUN_DIAGNOSTICS_AFTER_FIT=0 \
.venv/bin/python -u fit_animal_by_animal/animal_wise_norm_alpha_tied_fit_from_abort_params_condition_t_E_aff_fixed_delay.py \
2>&1 | tee logs/fixed_condition_delay_ganon_lavos_remaining_$(date +%Y%m%d_%H%M%S).log
```

## Validation Notes

ganon transfer validation on 2026-06-17:

- New isolated workspace created at `/home/rlab/raghavendra/npl_alpha_condition_t_E_aff_fixed_delay`.
- `.venv` is symlinked to `/home/rlab/raghavendra/npl_alpha_ild2_abl_specific_delay/.venv`.
- Remote `py_compile` passed.
- Remote dry-run with `DESIRED_BATCHES_OVERRIDE=LED34_even,LED8,SD DRY_RUN_ONLY=1` found 15 animals.
- Remote dry-run validated 864 condition-cache rows and 414 observed fit conditions for selected animals.
- LED7/92 cache sanity values printed as `53.051 ms` for ABL 20 ILD -1 and `45.648 ms` for ABL 20 ILD +1.

## Collection Checklist

- [ ] Confirm lavos run started.
- [x] Confirm ganon run started.
- [x] Collect lavos result PKLs/PDFs/logs.
- [x] Collect ganon result PKLs/PDFs/logs. Ganon original split complete; result folder was about 1.2G before remaining-lavos outputs.
- [x] Collect ganon remaining-lavos result PKLs/PDFs/logs after `npl_alpha_fixed_delay_lavos_remaining` completes.
- [x] Merge all 30 animals into the local result folder or a consolidation folder.
- [x] Back up consolidated all-30 folder to Google Drive.
- [ ] Record any failed or unstable VBMC fits before rerunning.

## Progress Snapshots

2026-06-17 16:20 ganon:

- Stable completed: `LED34_even/48`, `LED34_even/52`, `LED34_even/56`, `LED34_even/60`, `LED8/105`, `LED8/107`, `LED8/108`, `LED8/109`, `LED8/112`, `SD/48`, `SD/49`, `SD/50`.
- Current log shows active fit at `SD/52`.
- Remaining after current fit: `SD/53`, `SD/55`.

2026-06-17 ganon completion check:

- Result PKLs: 15/15.
- Non-empty PDFs: 15/15.
- Stable completed: all ganon animals.
- Unstable/missing fits: none.
- No fixed-delay fit Python process running after completion.
- Result folder size: 1.2G.

2026-06-17 lavos partial run:

- Stable completed on lavos: `LED34/45`, `LED34/57`, `LED34/59`.
- Missing/stopped before stable result PKL: `LED34/61`, `LED34/63`, `LED6/81`, `LED6/82`, `LED6/84`, `LED6/86`, `LED7/92`, `LED7/93`, `LED7/98`, `LED7/99`, `LED7/100`, `LED7/103`.
- `LED34/61` had only a small partial PDF on lavos and no stable result PKL.

2026-06-17 ganon remaining-lavos transfer/run:

- Transferred batch CSVs for `LED34`, `LED6`, and `LED7`.
- Transferred abort-source PKLs for the 12 missing/stopped animals.
- Dry-run with `TEST_BATCH_ANIMAL_PAIRS=LED34:61,LED34:63,LED6:81,LED6:82,LED6:84,LED6:86,LED7:92,LED7:93,LED7:98,LED7:99,LED7:100,LED7:103` found 12 animals and 360 observed fit conditions.
- Started tmux session `npl_alpha_fixed_delay_lavos_remaining`.
- Log: `/home/rlab/raghavendra/npl_alpha_condition_t_E_aff_fixed_delay/logs/fixed_condition_delay_ganon_lavos_remaining_20260617_173341.log`.

2026-06-18 all-30 completion/consolidation/backup:

- Stable completed: 30/30.
- Local consolidated folder on lavos: `/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30/`.
- Consolidated local counts: 30 result pickles, 30 non-empty PDFs, 30 VBMC posterior files, 3 logs, 1 consolidation summary; 94 files total, 2.3G.
- Google Drive parent: `raga:ddm_fit_backups_20260618_fixed_condition_t_E_aff_all30_lavos/`.
- Google Drive result folder: `raga:ddm_fit_backups_20260618_fixed_condition_t_E_aff_all30_lavos/NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30/`.
- Google Drive config folder contains `FIT_BACKUP_LEDGER.md`, `DISTRIBUTED_FIXED_DELAY_FIT_LEDGER.md`, and `backup_fixed_condition_teaff_all30_to_drive_20260618.sh`.
- Central Drive fit-backup ledger: `raga:ddm_fit_backup_ledgers/FIT_BACKUP_LEDGER.md`.
- Drive verification: 94 result files, 3 config files, 97 objects total, 2.238 GiB.
