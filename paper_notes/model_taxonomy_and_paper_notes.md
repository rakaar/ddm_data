# DDM model taxonomy and paper notes

This note is the repo-owned context for model-comparison work in `ddm_data`. It is not a complete paper summary; it records the model meanings, source papers, and code conventions that should prevent future agents from mixing up IPL, NPL, alpha, lapses, Gamma/Omega fits, and likelihood comparisons.

## Primary sources

- Local draft: `WL_Normalization (6).pdf`
- IPL/TIED foundation: Pardo-Vazquez et al., 2019, Nature Neuroscience, "The mechanistic foundation of Weber's law", https://www.nature.com/articles/s41593-019-0439-7
- 2019 supplementary information, including TIED constraints and parameter uncertainty: https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-019-0439-7/MediaObjects/41593_2019_439_MOESM1_ESM.pdf
- Proactive process / PSIAM foundation: Hernandez-Navarro et al., 2021, Nature Communications, "Proactive and reactive accumulation-to-bound processes compete during perceptual decisions", https://www.nature.com/articles/s41467-021-27302-8

## Scientific frame

The 2019 paper introduced the TIED constraint: at fixed intensity ratio, changing absolute sound level should primarily rescale the effective time axis of the decision process. That constraint supports a bounded exact accumulation account with stimulus-dependent evidence mean and variance that scale together.

The current local `WL_Normalization` draft extends this story to a larger 30-animal dataset and broader ILD range. The central problem is that IPL can respect Weber/TIED structure but struggles when high-evidence ILDs expose saturation of discriminability. NPL solves this by adding divisive gain normalization, separating ILD-dependent saturation of discriminability from ABL-dependent time scaling.

The proactive process is separate from the sensory transduction question. It models action timing/abort/short-RT behavior as an Action Initiation process racing with Evidence Accumulation. In proactive trials, timing can be set by AI while choice is still read from accumulated evidence.

## Model names used in this repo

- `IPL`, `vanilla TIED`, `vanila`, and common local typos such as `vanilla_tied` or `vanilla_toed` refer to the same family. `IPL` is the paper-facing name; older scripts often use `vanilla`.
- `NPL`, `norm TIED`, and `normalized TIED` refer to normalized power-law sensory transduction.
- `NPL+alpha` is the current extension under active testing. It is not part of `WL_Normalization (6).pdf`. Alpha was introduced to better match IC-style tuning curves and to test whether that representational change can help with the Omega mismatch.
- `lapse` variants add stimulus-independent choice lapses, usually with a global lapse rate and right-choice probability. Lapses are a possible behavioral contaminant but should not be treated as the main explanation for Gamma saturation unless common-likelihood comparisons support it.
- `big Gamma/Omega/delay SVI` fits condition-wise `gamma`, `omega`, and `t_E_aff` directly plus global `w` and `del_go`. It is a descriptive condition-wise DDM reference, not an IPL/NPL mechanistic model.
- `MSE` fits to Gamma/Omega means are diagnostic parameter mappings. They fit posterior means of condition parameters, not the original RT+choice likelihood.

## Core parameter groups

The exact formula source is the active utility file, not this note.

- IPL/vanilla direct SVI currently fits `rate_lambda`, `T_0`, `theta_E`, `w`, `del_go`, and condition-wise `t_E_aff`.
- IPL/vanilla+lapse adds `lapse_prob` and `lapse_prob_right`.
- NPL+alpha direct SVI currently fits `rate_lambda`, `T_0`, `theta_E`, `w`, `del_go`, `rate_norm_l`, `alpha`, and condition-wise `t_E_aff`.
- NPL+alpha+lapse adds `lapse_prob` and `lapse_prob_right`.
- Big Gamma/Omega/delay SVI fits condition-wise `gamma`, `omega`, `t_E_aff` with global `w` and `del_go`.

Useful likelihood/formula entry points:

- `fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py`
- `fit_animal_by_animal/numpyro_vanilla_lapse_condition_delay_svi_utils.py`
- `fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py`
- `fit_animal_by_animal/numpyro_npl_alpha_lapse_svi_utils.py`
- `fit_each_condn/svi_gamma_omega_likelihood_utils.py`
- Legacy plotting comparison helpers: `fit_each_condn/compare_gamma_from_cond_fit_and_norm_model_fit_for_paper.py`

Important caution: some older plotting helpers use simplified or legacy Gamma/Omega expressions for display. Use the current likelihood utilities for common-likelihood comparisons.

## Interpretation of Gamma and Omega

`gamma` is the effective signed signal-to-noise/discriminability parameter of the dimensionless DDM. It is primarily what psychometric slope and choice accuracy care about.

`omega` is the condition-dependent time-scale parameter. It maps physical time into the dimensionless decision time and is therefore central for RT/quantile/RTD predictions.

IPL can keep accuracy level-invariant but tends to couple the power-law exponent needed for ILD saturation to the exponent controlling ABL-dependent time scaling. NPL breaks that coupling through normalization: the "naked" exponent shapes discriminability saturation while the renormalized exponent controls the time-scale effect.

## Current comparison logic

When comparing model families, first classify the comparison:

- Direct RT+choice likelihood fit vs direct RT+choice likelihood fit: compare common log likelihoods on exactly the same trial subset and truncation/censoring convention.
- Mechanistic model vs condition-wise Gamma/Omega fit: compare Gamma/Omega curves, then check whether the mechanistic parameters also perform well on RT+choice likelihood and paper-style diagnostics.
- MSE Gamma/Omega fit vs SVI posterior parameters: remember the MSE objective ignores trial counts, uncertainty, posterior covariance, and RT likelihood curvature. A visually better Gamma/Omega curve can still have worse RT+choice likelihood.

The current best default family has usually been the patience12 restore-best SVI outputs, but this is not a permanent rule. Scripts should expose the fit root near the top or through environment variables, print selected roots, and state exact roots in captions.

## Data and plotting conventions

- Use `.venv/bin/python` for repo-local Python.
- For current animal-wise SVI diagnostics, valid RTs are usually filtered to `[0, 1]` seconds.
- RTD data curves in recent diagnostics include valid trials plus abort events 3 and 4 after truncation when comparing to model RTDs.
- Abort truncation differs by batch: most batches remove aborts below 300 ms; `LED34_even` removes below 150 ms.
- For SD psychometric/slope diagnostics, do not add nonexistent `|ILD|=16` points.
- For RT quantile plots that require continuous model curves, SD `t_E_aff` beyond `|ILD|=8` has sometimes been flat-held at the `|ILD|=8` value to avoid discontinuities. Only use this when the specific figure/request calls for it.
- Model RTDs should generally be evaluated at 1 ms resolution when inspecting near-zero behavior.

## Open scientific questions to preserve

- Does alpha improve Omega because it better matches IC-like tuning curves, or does it mainly shift parameter tradeoffs?
- Are NPL+alpha+lapse SVI fits sensitive to initialization in a way that indicates local minima, especially for animals whose early-best ELBO occurs at 1k/2k?
- Do MSE-fitted parameters that visually match condition Gamma/Omega improve the common RT+choice likelihood, or are they only better under the MSE summary objective?
- Are delay differences meaningful after refitting TIED parameters, given that `lambda`, `theta_E`, and `T_0` are delay-dependent?
