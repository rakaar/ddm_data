# Final human model: fitting-code guide

This is the final **16-parameter-per-group** model used for the shared-Gaussian-delay human figures. Control and ASD are fitted separately. Code and original initialization/configuration files are versioned on GitHub; datasets, posterior arrays, checkpoints and figure outputs are kept separately.

## Read these files in this order

1. [fit_human_shared_gaussian_delay_svi.py](fit_human_shared_gaussian_delay_svi.py): command-line settings, initialization, NumPyro full-rank SVI, checkpointing, loss-only stopping and posterior export.
2. [human_shared_gaussian_delay_svi_utils.py](human_shared_gaussian_delay_svi_utils.py): the actual 16-parameter model, priors, ten stimulus conditions, scalar broadcasting and guide.
3. [human_gaussian_delay_svi_utils_wide.py](human_gaussian_delay_svi_utils_wide.py): joint RT/choice likelihood, Gaussian-delay mixture with uniform lapses and prediction generation. Its 83-parameter docstring describes an earlier general adapter; the final helper above broadcasts the final 16 parameters into it.
4. [gaussian_delay_likelihood_wide.py](gaussian_delay_likelihood_wide.py): delay convolution, truncated-Gaussian normalization, response-window normalization, FFT/interpolation and quadrature fallback.
5. [human_plain_ddm_diagnostics_utils.py](human_plain_ddm_diagnostics_utils.py): the DDM first-passage density/CDF calculations.

The remaining helper sources are included because the original code imports their data loaders, trapezoidal priors or first-passage primitives. Some describe older models in their docstrings; those older model drivers are not the final fitting entry point. Run the driver from the repository root so its existing imports and relative data/output paths work.

## Final model and data

- Human visual brightness task; simultaneous `Condition == "RT"`, `Valid == 1`, inclusive `0 <= RT_s <= 1.8` seconds.
- Pool raw trials across participants, sessions and ASLs (20 and 40 cd/m²) within each group.
- Control: **6,574 trials / 11 participants**. ASD: **4,770 / 8**. Both choices are retained, including all four RTs below 200 ms.
- Right choice is `+1`; left is `-1`. Signed `Delta_dB` is right minus left.
- Ten Gammas correspond, in order, to signed differences `[-2, -1.3, -0.65, -0.3, -0.1, 0.1, 0.3, 0.65, 1.3, 2]`.
- Each group has Gamma[10], one Omega, one Gaussian delay mean `delay_mu`, one Gaussian delay SD `delay_sigma`, starting bias `w`, lapse fraction `p=lapse_prob`, and lapse-right probability `q=lapse_prob_right`: **16 parameters**.
- Delay is drawn independently for each trial from Normal(mu, sigma) truncated below zero, with no upper cutoff. This variability is distinct from the variational posterior over parameters.
- There is no proactive process. Lapse RT is Uniform(0,1.8 s) in stored RT coordinates.

For observed RT t and choice y, the likelihood is

    L(t,y | D) = (1-p) h_y(t | D) / Z_D + p q_y / 1.8
    h_y(t | D) = integral_0^t f_y(t-d | Gamma_D, Omega, w) g(d) dd
    g(d) = phi((d-mu)/sigma) / [sigma * Phi(mu/sigma)], d >= 0
    Z_D = sum_y integral_0^1.8 h_y(t | D) dt
    q_right = q, q_left = 1-q

Thus both Gaussian truncation and each signed condition's retained either-bound mass are normalized. Here p is the lapse fraction among retained trials.

Independent trapezoidal priors have linear tails outside the flat region:

| Parameter | Hard limits | Flat region |
|---|---|---|
| `gamma_positive` | [-1.0, 20.0] | [0.0, 12.0] |
| `gamma_negative` | [-20.0, 1.0] | [-12.0, 0.0] |
| `omega` | [0.01, 15.0] | [0.1, 12.0] |
| `w` | [0.0, 1.0] | [0.4, 0.6] |
| `lapse_prob` | [0.0001, 0.4] | [0.001, 0.1] |
| `lapse_prob_right` | [0.001, 0.999] | [0.4, 0.6] |
| `delay_mu` | [0.0, 1.0] | [0.01, 0.2] |
| `delay_sigma` | [0.0005, 0.3] | [0.01, 0.15] |

## Required data files

Before running, restore the existing dataset files under `ASD_Human_Data/` in your repository checkout:

- `table_visualTask_15nov2026.mat` (original file; read here for its provenance hash).
- `simultaneous_rt_trials.csv` (the existing 11,400-row RT export; the fitter selects the final 11,344 rows).

`simultaneous_rt_extraction.json` is included and records both expected SHA-256 hashes. Use the exact existing files; the loader checks them. The filename's date is historical and does not affect the selection.

Original starting values are already included under the original 34-parameter initialization path. The final helper averages its ten delay mu values and ten sigma values into scalars. It does not initialize from the fitted posterior. The final output directory is `pooled_asl_shared_omega_shared_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms/`; saved result binaries are restored separately from the data/fit backup.

## Run from the repository root

The original environment used Python 3.12.13, JAX/JAXlib 0.4.38, NumPyro 0.15.3 and an RTX 4090 GPU. The driver defaults to GPU and disables memory preallocation. CPU operation is available by explicitly setting `JAX_PLATFORM_NAME=cpu`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r ASD_Human_Data/requirements_human_shared_delay_cpu.txt

# Show settings without starting a fit.
JAX_PLATFORM_NAME=cpu .venv/bin/python ASD_Human_Data/fit_human_shared_gaussian_delay_svi.py --help

# Validate data, likelihoods, gradients and guide shape; no optimization.
JAX_PLATFORM_NAME=cpu .venv/bin/python ASD_Human_Data/fit_human_shared_gaussian_delay_svi.py \
  --group Control --preflight-only --output-dir ASD_Human_Data/catarina_preflight

# Full CPU run of both groups (sequential); use a fresh output directory.
JAX_PLATFORM_NAME=cpu .venv/bin/python ASD_Human_Data/fit_human_shared_gaussian_delay_svi.py \
  --group all --output-dir ASD_Human_Data/catarina_fit
```

For GPU, use an existing GPU-enabled JAX environment and set `JAX_PLATFORM_NAME=gpu` instead. Use `--group Control` and `--group ASD` in separate processes/terminals with the same output directory to run the groups concurrently. CPU and GPU runs need separate output directories because the saved run configuration records the backend. To resume an interrupted fit on the same backend, repeat the command with `--resume-existing`.

Fit settings: float64, full-batch AutoMultivariateNormal, one ELBO particle, seed 0, guide scale 0.1, ClippedAdam learning rate 0.0002 and gradient clip 1. At least 50,000 and at most 1,000,000 updates, checked in nonoverlapping 1,000-update windows. Every strictly lower window mean resets patience; stop after 25 subsequent windows without improvement and restore the best checkpoint. There is no posterior-stability stopping gate. Loss confirmation does not establish a global optimum.

The completed source run restored Control at 82,000 updates (finished 107,000) and ASD at 61,000 (finished 86,000). Each saved 10,000 aligned joint posterior draws. Intervals describe a pooled-trial variational approximation, not participant-to-participant variability.

## Validation and sharing

The source handoff was checked in an isolated directory with the existing dataset supplied separately. It reproduced the 11,344 retained trial identities, original initialization, 16 free parameters per group and joint log likelihoods at saved posterior means. No model was refitted for this handoff. The final run records Control joint log likelihood 758.1589063 and ASD −509.0318001 at the arithmetic posterior means.

For Catarina, start with the fitting driver, final model helper, Gaussian-delay likelihood and this guide. The driver imports the remaining helpers in the same repository; clone the repository rather than downloading only one Python file.
