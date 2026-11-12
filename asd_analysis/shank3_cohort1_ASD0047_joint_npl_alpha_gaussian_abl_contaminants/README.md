# ASD0047 single-stage proactive/reactive/contaminant fit

ASD0047 is the original SHANK3 cohort 1 WT animal with two-stage proactive delay −715.546 ms. Its same 18,771 valid RT [0,1] choices are fitted jointly with 3,004 pre-onset aborts and 11 onset-censored late fixation-labelled trials. The 161 long valid responses are excluded; no prepared repeat filter is reapplied.

The initial model jointly fits 20 parameters: seven shared NPL+alpha/readout parameters, Gaussian reactive delay means/SDs by ABL shared across signed ILDs, three proactive parameters, and contaminant probability/right bias/exponential weight/rate. The selected nested model fixes the exponential weight d=1 and jointly fits the remaining **19 parameters**. `selected_fit.json` identifies `exponential_only/round_00/seed_0/`; both original mixture starts and both reduced-model starts remain preserved. The exact common priors initially match this animal's final two-stage priors, including proactive delay [−3,0.3] s. Reactive trial delays remain Gaussian truncated below zero. Initial mixture starts use the old physical posterior means and a distinct near-zero-delay start; reduced-model starts use the respective mixture posterior means with fresh guides/optimizers. Full-rank float64 SVI uses the established 1,000-step windows, 50,000-step minimum, 25-window confirmation and restore-best rule, with diagnostic-only parameter stability.

The complete joint mixture conditions on no response before fixation and the selected upper endpoint onset+1 s. Abort choices are unobserved and summed out. Contaminants are a biased-choice exponential/uniform mixture on fixation-time [0,3.3] s. These differ from the earlier unconditioned proactive stage and untruncated exponential, so this is not a change of estimation order alone.

`run_shank3_asd0047_joint.py` runs validated GPU fits, independently verifies them, and expands artificial hard bounds/flat intervals when flagged. It never widens physical probabilities past 0 or 1. A selected mean exponential weight ≥0.95, or upper-edge pressure with mode ≥0.95 and median ≥0.90, triggers the user-authorized nested pure-exponential refit. The mode/median extension was introduced after inspecting the first completed start; see `contaminant_policy_revision.json`. Original physical-bound flags remain visible; this simplification is documented, not represented as a Bayes-factor comparison. Both mixture and reduced-model fits are preserved.

`finish_shank3_asd0047_joint.py` observes the verified live controller, then generates numerical predictions, a 3×5 RTD PNG, psychometric PNG, both comparison PNGs, full and focused corners, and a four-page diagnostics PDF. It runs independent prediction, posterior, restart, histogram/count and PDF validation and stops for actual visual inspection. It does not claim goal completion from numerical receipts alone. PDFs are copied to `shan3_coh_1_animal_Wise_fitting/ASD0047_joint/`.

See `task_requirements.json` for the deliverable audit; `completion.json` and `postprocessing_completion.json` track the separate fitting and rendering stages. The source CSV and previous animal fit are protected by hashes.

## Completed result and interpretation

The selected proactive delay is −778.05 ms, central 95% variational credible interval [−872.40, −685.27] ms, compared with −715.55 ms in the original two-stage fit. The other competitive joint start gives −573.36 ms [−666.45, −483.97] ms. Joint fitting did not eliminate the negative shift; separated posteriors across starts indicate initialization sensitivity or weak identification. This is not a controlled experiment changing estimation order alone: contaminant support/right bias, entry conditioning and treatment of long valid responses also differ.

All 19 free parameters in both final starts pass the saved hard-bound and prior-taper audits. The original mixture's d posterior is broad despite its upper-edge mode; fixing d=1 was an exploratory simplification, not evidence that a uniform component is ruled out. The final abort diagnostic retains a visible early-time shape residual.

ASD0047 is a **WT-genotype individual**. WT in its figure titles does not indicate pooling. `joint_vs_two_stage_parameters/` compares only this animal's saved joint and two-stage draws: 13 reactive estimates and six proactive/lapse panels, including the old fixed right-choice probability 0.5 without an interval. The same 18,771 valid choices are independently matched between fits. These are posterior credible intervals, not across-animal SEMs.

## Sources and verification

All entrypoints below live in the parent `asd_analysis/` directory and use `# %%` blocks with editable settings. Use the repository's `.venv/bin/python`.

- `shank3_asd0047_joint_utils.py`, `prepare_shank3_asd0047_joint.py`, `fit_shank3_asd0047_joint.py` and `run_shank3_asd0047_joint.py`: animal-only selection, complete-mixture model, two-start fitting and bound/model decisions.
- `verify_shank3_asd0047_joint.py` and `verify_shank3_asd0047_joint_resume.py`: independent numerical/posterior/prediction checks and serialized restart verification.
- `plot_shank3_asd0047_joint.py`: 3×5 RTDs and psychometrics from saved estimates. RTDs use 20 ms data bins, 1 ms model evaluation, full [−1,1] s normalization and [−0.1,0.6] s display; psychometrics use actual numeric signed ILDs.
- `plot_shank3_asd0047_joint_comparison.py` and `verify_shank3_asd0047_joint_comparison.py`: both single-animal parameter PNGs and independent draw/trial/source checks.
- `export_shank3_asd0047_joint_corners_and_report.py` and `verify_shank3_asd0047_joint_exports.py`: full/focused corners with parameter-and-mean titles, four-page bookmarked diagnostics PDF, count and export validation.
- `finish_shank3_asd0047_joint.py`: numerical postprocessing after the live controller accepts the fit; actual visual inspection remains a separate step.
- `publish_shank3_asd0047_joint.py` and `verify_shank3_asd0047_joint_completion.py`: result-book copies, direct embeds, rendered-link checks and strict MkDocs build. `goal_completion_audit.json` records the completed checks; it does not establish a unique global solution.

The pure shared kernels `shank3_wt_joint_contaminant_utils.py`, `shank3_individual_adaptive_bounds_utils.py` and `shank3_individual_bounds_policy.py` are reused dependencies. Their original pooled/individual defaults do not override the ASD0047-only inputs or priors in this workflow.

Versioned artifacts include the four completed joint posterior sets, losses/summaries/convergence records, the exact selected two-stage ASD0047 reference posteriors, condition/membership counts, model prediction tables, parameter comparisons, source snapshots and compact numerical/visual receipts. Published PNGs, PDFs, audit downloads and source copies are in `docs/assets/results/2026-11-12/shank3_ASD0047_joint/`. Raw trial copies, the original prepared CSV, timing caches, optimizer checkpoints, logs and PDF-render previews remain local and are not added during wrap-up. Re-running raw-data or restart verification requires those original local inputs/checkpoints; the published summaries are not a replacement for them. No Google Drive backup is performed by this wrap-up.
