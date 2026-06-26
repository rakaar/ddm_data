# NPL Alpha Fixed Condition t_E_aff All-30 Consolidation

Created: 2026-06-18 on lavos.

Source fit:

`fit_animal_by_animal/animal_wise_norm_alpha_tied_fit_from_abort_params_condition_t_E_aff_fixed_delay.py`

Model:

NPL + alpha animal-wise refit with condition-by-condition `t_E_aff` fixed from `fit_each_condn/abl_specific_ild2_delay_agreement/condition_t_E_aff_extraction_cache.csv`. Fitted parameters are `rate_lambda`, `T_0`, `theta_E`, `w`, `del_go`, `rate_norm_l`, and `alpha`.

Local consolidated folder:

`/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/NPL_alpha_condition_t_E_aff_fixed_delay_fit_results_all_30`

## Source Machines

- lavos supplied 3 stable animals: `LED34/45`, `LED34/57`, `LED34/59`.
- ganon supplied 27 stable animals:
  - `LED34_even/48`, `LED34_even/52`, `LED34_even/56`, `LED34_even/60`
  - `LED8/105`, `LED8/107`, `LED8/108`, `LED8/109`, `LED8/112`
  - `SD/48`, `SD/49`, `SD/50`, `SD/52`, `SD/53`, `SD/55`
  - `LED34/61`, `LED34/63`
  - `LED6/81`, `LED6/82`, `LED6/84`, `LED6/86`
  - `LED7/92`, `LED7/93`, `LED7/98`, `LED7/99`, `LED7/100`, `LED7/103`

## Validation

- Result pickles: 30/30.
- Non-empty PDFs: 30/30.
- VBMC posterior files: 30/30.
- Fit logs: 3.
- Stable VBMC result messages: 30/30.
- Missing result pickles: none.
- Unstable result pickles: none.

## Contents

- `results_*_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pkl`
- `results_*_NORM_ALPHA_CONDITION_T_E_AFF_FIXED_DELAY_FROM_ABORTS.pdf`
- `vbmc_PKL_file_norm_alpha_condition_t_E_aff_fixed_delay_tied_results_batch_*_FROM_ABORTS.pkl`
- `logs/lavos/lavos_LED34_LED6_LED7_20260617_143809.log`
- `logs/ganon/fixed_condition_delay_ganon_LED34even_LED8_SD_20260617_152430.log`
- `logs/ganon/fixed_condition_delay_ganon_lavos_remaining_20260617_173341.log`
