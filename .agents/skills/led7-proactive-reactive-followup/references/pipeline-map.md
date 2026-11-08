# LED7 Pipeline And Provenance Map

This map records the accepted routes and the historical branches that are easy to confuse with them. Verify roots on disk before a new run because fit outputs can be moved after Drive consolidation.

## A. Current Abort And Censored Fit

Model: analytic proactive drift step plus exponential lapse, no 300 ms truncation.

Core utility:

- `fitting_aborts/numpyro_proactive_led_step_jump_no_trunc_exp_lapse_svi_utils.py`

Aggregate driver:

- `fitting_aborts/numpyro_svi_proactive_led_step_jump_no_trunc_exp_lapse_aggregate_led7.py`

Accepted animal-wise sources:

- Audited fits for animals 92, 99, 100, and 103:
  `fitting_aborts/numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_patience12_min150k_restore_best_audit_outputs/LED7_<animal>/`
- Original completed fits for animals 93 and 98:
  `fitting_aborts/numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_patience12_min50k_restore_best_outputs/LED7_<animal>/`

Accepted pooled aggregate source:

- `fitting_aborts/numpyro_svi_proactive_led_step_jump_all_on_no_trunc_exp_lapse_aggregate_patience12_min150k_restore_best_outputs/LED7_all_animals/`

Aggregate data audit:

| Item | Count |
|---|---:|
| Total retained rows | 94,433 |
| LED ON rows | 30,910 |
| LED OFF rows | 63,523 |
| LED ON abort / censored | 6,336 / 24,574 |
| LED OFF abort / censored | 10,051 / 53,472 |
| Abort rows below 300 ms retained | 3,081 |

Per-animal retained totals:

| Animal | Rows |
|---|---:|
| 92 | 21,608 |
| 93 | 16,486 |
| 98 | 14,055 |
| 99 | 14,570 |
| 100 | 8,727 |
| 103 | 18,987 |

The aggregate fit completed by patience after checking 150,000 steps and restored the best checkpoint at 101,000 steps. Its posterior-mean lapse rate is about 4.36%, and its posterior-mean exponential rate is about 4.04 per second. Re-read the saved summary for exact intervals.

Scope details:

- LED ON includes unilateral and bilateral LED trials.
- Abort rows are `abort_event == 3`; censored rows are successful-choice trials.
- `abort_event == 4` is not part of this abort-fit dataset.
- LED ON and OFF likelihood terms are unweighted.

Backup:

- `raga:ddm_fit_backups_20260814_proactive_led_no_trunc_lapse_svi_lavos/`
- See `FIT_BACKUP_LEDGER.md` for the copied roots and verification record.

## B. Combined Likelihood Validation

Use these before changing the proactive-lapse plus reactive likelihood:

- Likelihood utility: `fitting_aborts/numpyro_proactive_led_step_jump_npl_alpha_utils.py`
- Simulator: `fitting_aborts/simulate_proactive_led_step_jump_npl_alpha.py`
- Validation script: `fitting_aborts/validate_proactive_led_step_jump_npl_alpha_likelihood.py`
- Choice-collapsed RTD check: `fitting_aborts/plot_proactive_led_step_jump_npl_alpha_lapse_choice_collapsed_rtd.py`
- Validation outputs: `fitting_aborts/proactive_led_step_jump_npl_alpha_validation/`

These checks established separate LED OFF and LED ON likelihoods, intact timing-pair use, choice-resolved/full-density agreement, and total CDF mass of one. Keep the 64-node quadrature reference unless a new accuracy check justifies changing it.

## C. Current Valid-Trial NPL+Alpha Fits

Reusable fit wrapper:

- `fitting_aborts/numpyro_proactive_led_step_jump_npl_alpha_valid_svi_utils.py`

Reactive parameter definitions and priors:

- `fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py`

Current pooled super-animal output root:

- `fitting_aborts/numpyro_svi_proactive_led_step_jump_npl_alpha_valid_led7_aggregate_patience12_min50k_restore_best_outputs/LED7_all_animals/`

The proactive and lapse parameters are fixed at posterior means from the aggregate abort fit in section A. The two reactive fits are independent:

| Category | Valid rows | Signed-condition range | Current accepted start |
|---|---:|---:|---|
| LED OFF | 52,799 | 1,665-1,844 | displaced |
| Bilateral LED ON | 8,138 | 224-294 | displaced |

The original automatic selection summaries chose OFF-displaced and ON-warm using best-window negative ELBO followed by posterior-mean likelihood for an ELBO tie within 0.1%. The explicit paper decision is to use displaced for both categories when results are next regenerated because its stopping interpretation is clearer. The OFF-displaced branch checked 58,000 steps and restored 46,000; ON-displaced checked 50,000 and restored 14,000. Keep the original automatic selection summaries unchanged for audit, and treat existing warm-ON figures as historical until regeneration.

Pilot LED7/98 roots:

- Main pilot: `fitting_aborts/numpyro_svi_proactive_led_step_jump_npl_alpha_valid_led7_98_pilot_outputs/`
- Deliberately displaced recovery diagnostic: `fitting_aborts/numpyro_svi_proactive_led_step_jump_npl_alpha_valid_led7_98_displaced_init_recovery_outputs/`

Pilot fit counts are 7,791 OFF and 1,170 bilateral ON valid trials. Treat the pilot as a method check, not the current pooled paper estimate.

Full-RTD diagnostic counts differ from successful-trial fit counts because they add abort events for display:

- OFF: 63,744 rows, including event 3, event 4, and successful trials; 221 event-4 rows are included.
- Bilateral ON: 10,313 rows, including event 3 and successful trials; event 4 is excluded.

Backup:

- `raga:ddm_fit_backups_20260829_proactive_lapse_npl_alpha_valid_svi_lavos/`
- This backup includes the LED7/98 pilot and super-animal roots. Check `FIT_BACKUP_LEDGER.md` for verification and any central-ledger limitation.

## D. Historical And Comparison Routes

Paper-era proactive VBMC:

- `aborts_ipl_npl_time_fit_results/results_LED7_animal_<animal>.pkl`
- Three proactive parameters with a 300 ms truncated likelihood.
- Corrected LED7/93 replacement provenance:
  `fitting_aborts/vbmc_old_truncated_proactive_led7_93_more_evals/canonical_replacement/replacement_provenance.json`

Intermediate truncated, bilateral, no-lapse SVI:

- `fitting_aborts/numpyro_svi_proactive_led_step_jump_bilateral_unweighted_patience12_min50k_restore_best_outputs/`
- `fitting_aborts/numpyro_svi_proactive_led_step_jump_bilateral_unweighted_min150k_restore_best_audit_outputs/`
- `fitting_aborts/numpyro_svi_proactive_led_step_jump_bilateral_unweighted_aggregate_patience12_min50k_restore_best_outputs/`

Do not use historical `vbmc_real_data_proactive_LED_fit_all_animals_agg_CORR_ID_no_trunc_exp_lapse_PYDDM*` scripts as substitutes for the current model. They implement a mechanistically different PyDDM route rather than the analytic step jump.
