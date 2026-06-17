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
| lavos | `/home/rlab/raghavendra/ddm_data` | `LED34`, `LED6`, `LED7` | `LED34: 45,57,59,61,63`; `LED6: 81,82,84,86`; `LED7: 92,93,98,99,100,103` | 15 | Assigned; use command below unless already running |
| ganon | `/home/rlab/raghavendra/npl_alpha_condition_t_E_aff_fixed_delay` | `LED34_even`, `LED8`, `SD` | `LED34_even: 48,52,56,60`; `LED8: 105,107,108,109,112`; `SD: 48,49,50,52,53,55` | 15 | Files transferred and dry-run validated; not started by Codex |

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
- [ ] Confirm ganon run started.
- [ ] Collect lavos result PKLs/PDFs/logs.
- [ ] Collect ganon result PKLs/PDFs/logs.
- [ ] Merge all 30 animals into the local result folder or a consolidation folder.
- [ ] Record any failed or unstable VBMC fits before rerunning.
