---
name: led7-proactive-reactive-followup
description: Map, run, diagnose, compare, publish, and back up the LED7 follow-up-paper fits that combine LED-ON/OFF proactive abort models with NPL+alpha valid-trial models. Use for questions about accepted fit roots, trial filters, likelihood conventions, SVI convergence, LED timing, RTD normalization, result-book figures, or provenance across historical and current LED7 analyses.
---

# LED7 Proactive-Reactive Follow-up

Use this skill for the LED7 follow-up-paper analysis. Keep the current analytic step-jump route distinct from historical paper fits, truncated diagnostics, and PyDDM experiments.

## Current Paper Route

1. Fit abort and censored trials with the analytic proactive step-jump plus exponential-lapse model.
   - Fit LED OFF and all selected LED ON rows jointly.
   - Fit seven parameters: `V_A_base`, `V_A_post_LED`, `theta_A`, `del_a_minus_del_LED`, `del_m_plus_del_LED`, `lapse_prob`, and `beta_lapse`.
   - Retain early aborts. There is no 300 ms truncation or truncation renormalization in this route.
2. Fix those seven proactive/lapse parameters at posterior means from the pooled aggregate abort fit.
3. Fit successful valid trials separately for LED OFF and bilateral LED ON with NPL+alpha RT+choice likelihoods.
   - Fit seven shared reactive parameters plus 30 condition-specific `t_E_aff` values.
   - Use successful trials with `0 <= RTwrtStim <= 1 s`.
   - Treat the pooled six-animal fit as the current super-animal result. The LED7/98 fit is a pilot.

Read [pipeline-map.md](references/pipeline-map.md) before selecting any fit root. Read [data-likelihood-conventions.md](references/data-likelihood-conventions.md) before changing filters, likelihoods, or normalization. Read [result-book-map.md](references/result-book-map.md) before publishing figures.

## First Checks

Before editing or running anything:

1. Inspect `git status --short`; fit roots are often intentionally untracked and must not be deleted or reverted.
2. Confirm whether the request concerns abort/censored data, successful valid trials, or a full RTD diagnostic containing both.
3. Confirm LED scope: LED OFF, every LED ON trial, or bilateral LED ON only.
4. Name the exact source root in scripts, captions, CSVs, and reports.
5. Check the available `run_summary.json` or `run_status.json`, posterior summary, and selection metadata rather than inferring convergence from a loss image.
6. Preserve exact trial-wise `(t_LED, t_stim)` pairs. Do not independently sample or bin these timings for fitting.

## Non-Negotiable Distinctions

- **Abort fit versus valid fit:** abort fits use abort-density and censored-survival terms. Valid fits use RT+choice likelihood conditioned on the valid RT window.
- **Fit data versus RTD display data:** the valid fit uses successful trials only. A full RTD diagnostic may add abort events for visualization; document the denominator and event codes explicitly.
- **All LED ON versus bilateral ON:** the current abort fit uses all LED ON trials. The current valid LED ON fit uses bilateral trials only.
- **No truncation versus historical truncation:** the accepted no-truncation+lapse abort route retains the 3,081 aborts below 300 ms. Historical paper VBMC and several intermediate fits used a 300 ms left truncation.
- **Step jump versus PyDDM:** the current proactive model has an analytic drift step after LED timing. Historical PyDDM scripts use a different mechanism and are not interchangeable.
- **Pooled versus equal-animal average:** the super-animal fit is trial pooled. Equal-animal averaging is a plotting choice and must be labeled separately.
- **Displayed mass versus total mass:** an RTD plotted over a finite x-range need not have area one. Verify full-support CDF mass before claiming a normalization bug.

## Fit And Convergence Rules

- Use `.venv/bin/python` for local scripts.
- Keep new `fitting_aborts/` scripts cell-wise with `# %%`, editable parameters near the top, and no `main()` wrapper unless explicitly requested.
- Follow the existing full-rank SVI configuration unless the experiment explicitly changes it: clipped Adam, 1,000-step windows, patience-based restore-best, saved complete loss traces, posterior samples, guide state, and provenance.
- A restored-best checkpoint and final checked step are different. Show both in convergence figures.
- If multiple starts are run, select from the saved selection summary. Do not choose by visual smoothness alone.
- The original automatic selector chose displaced OFF and warm ON. The explicit paper decision is to use displaced for both OFF and bilateral ON when diagnostics are next regenerated. Until then, existing warm-ON figures remain historical outputs; retain the original selector files for audit.
- Label max-step runs as inconclusive unless the project has explicitly accepted them after an audit.

## Diagnostics And Publication

For new fits, normally save:

- posterior samples or a reconstructable guide state;
- posterior mean and 95% interval summaries;
- complete negative-ELBO trace and convergence-window table;
- exact row counts and per-condition counts;
- fixed-parameter and initialization provenance;
- reusable RTD/psychometric payloads and displayed-mass audits;
- figures with source script and fit root in the result-book caption.

Run compilation, the relevant likelihood/normalization checks, and `mkdocs build --strict`. Use the `update-result-book` skill for publication, `google-drive-fit-backup` for fit roots and Drive ledger updates, and `git-wrap-up` only when the user asks.

## Historical Guardrail

Do not silently replace paper-era artifacts. The old three-parameter proactive VBMC files, the corrected LED7/93 replacement, truncated five-parameter fits, no-lapse fits, and PyDDM fits remain useful comparison baselines. Record any canonical replacement with an explicit provenance file.
