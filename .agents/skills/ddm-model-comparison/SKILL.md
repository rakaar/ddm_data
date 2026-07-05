---
name: ddm-model-comparison
description: Compare and diagnose DDM/TIED model families in the ddm_data repository. Use when working on IPL/vanilla TIED, NPL/normalized TIED, NPL+alpha, lapse variants, Gamma/Omega condition fits, SVI/VBMC fit comparisons, likelihood/ELBO checks, RTD/psychometric/quantile diagnostics, or result-book figures involving these model families.
---

# DDM Model Comparison

## Overview

Use this skill to keep DDM model-comparison work scientifically and procedurally consistent in `ddm_data`. The canonical model meaning lives in [paper_notes/model_taxonomy_and_paper_notes.md](../../../paper_notes/model_taxonomy_and_paper_notes.md); read it before making model-meaning claims or choosing formulas.

## First Checks

- Read `AGENTS.md` for repo coding style before editing.
- Read the relevant current README before broad search:
  - `fit_animal_by_animal/README.md` for animal-wise IPL/NPL/NPL+alpha/SVI/VBMC fits and Fig. 2/Fig. 4 diagnostics.
  - `fit_each_condn/README.md` for condition-wise Gamma/Omega/`t_E_aff` fits, big Gamma/Omega/delay SVI, and MSE comparisons.
- Prefer exact fit roots from the user request, README, ledger, or result-book caption. Do not silently use the newest-looking folder when results may have multiple stopping rules or reruns.
- Keep source fit roots configurable near the top of new scripts or through clear environment variables, and print the selected roots in script output.

## Model Names

- Treat `IPL`, `vanilla TIED`, `vanila`, and local typo-like variants such as `vanilla_tied`/`vanilla_toed` as the same model family unless the code path proves otherwise. `IPL` is the paper name that came later.
- Treat `NPL`, `norm TIED`, and `normalized TIED` as the same baseline normalized power-law family.
- Treat `NPL+alpha` as the current alpha-extended normalized family. Alpha is not in `WL_Normalization (6).pdf`; it is being tested because it may better match IC-style tuning curves and potentially fix Omega mismatch.
- Treat `big Gamma/Omega/delay SVI` as a condition-wise descriptive DDM fit, not as an IPL/NPL mechanistic parameterization.

## Formula Discipline

- Use current likelihood utilities for likelihood calculations, not simplified plotting formulas.
- For IPL/vanilla SVI likelihoods, start with `fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py`.
- For IPL/vanilla+lapse SVI likelihoods, start with `fit_animal_by_animal/numpyro_vanilla_lapse_condition_delay_svi_utils.py`.
- For NPL+alpha SVI likelihoods, start with `fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py`.
- For NPL+alpha+lapse SVI likelihoods, start with `fit_animal_by_animal/numpyro_npl_alpha_lapse_svi_utils.py`.
- For direct Gamma/Omega condition likelihoods, start with `fit_each_condn/svi_gamma_omega_likelihood_utils.py`.
- When comparing likelihoods across models, evaluate all models on the same trial set, RT window, truncation/censoring convention, and posterior-mean or posterior-sampling convention.

## Data Conventions

- Use `.venv/bin/python` for local Python commands.
- Use valid RTs in the same window as the fit being diagnosed; for current SVI animal-wise diagnostics this is usually `[0, 1]` seconds.
- Respect batch-specific abort truncation when RTDs include aborts: generally `<300 ms` removed, but `LED34_even` uses `<150 ms`.
- For RTD plotting from recent diagnostics, data curves include valid trials plus abort events 3 and 4 after truncation when that was the user-approved convention.
- For SD animals, avoid nonexistent `|ILD|=16` psychometric/slope model points. For RT-quantile continuity plots that need full ILD range, use the documented flat-delay rule beyond `|ILD|=8` only when the source script/request specifies it.

## Common Workflows

When asked to compare parameter sources:
- Identify whether each source is direct RT+choice likelihood fit, condition-wise Gamma/Omega fit, or MSE fit to condition posterior means.
- State that MSE fits are diagnostic fits to Gamma/Omega means, not RT+choice likelihood fits.
- If the question is "which is better", compute a common likelihood on the same data rather than relying on Gamma/Omega visual agreement alone.

When asked to make paper-style diagnostics:
- Use Fig. 2-style pipelines for IPL/vanilla diagnostics.
- Use Fig. 4-style pipelines for NPL/NPL+alpha diagnostics.
- Prefer reusing existing scripts listed in the READMEs over rewriting figure logic from scratch.
- Put exact fit roots and source scripts in result-book captions.

When asked about convergence:
- Distinguish the old `stable_3_windows` rule from patience restore-best rules.
- For patience12 restore-best runs, show both the restored-best checkpoint and final checked step.
- Parameter stability is the key test; ELBO curves alone are not enough.

## Result Book

Use the repo's `update-result-book` skill whenever the user asks to add generated figures to the result book. Do not run Google Drive backup unless the user explicitly asks for it.
