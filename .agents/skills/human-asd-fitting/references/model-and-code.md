# Reference model and code for the agent

The human should not need to edit these files or install libraries themselves.
Inspect the imports and create a suitable isolated Python environment. The
reference source environment used Python 3.12.13, JAX/JAXlib 0.4.38 and NumPyro
0.15.3. Version changes require import/numerical checks, not a request for the
human to debug dependencies. Set `JAX_PLATFORM_NAME=cpu` before imports when
using CPU; the historical fitting driver otherwise defaults to GPU.

## Where the code lives

`../assets/reference_code/` relative to this reference directory contains a
snapshot preserving the original project layout. In a skill installation the
path is `assets/reference_code/` relative to `SKILL.md`. Work from that project
root when running reference drivers. The main files are:

| Path under reference_code | Role |
|---|---|
| `ASD_Human_Data/fit_human_shared_gaussian_delay_svi.py` | Reference SVI driver, checkpointing and posterior export |
| `ASD_Human_Data/human_shared_gaussian_delay_svi_utils.py` | Final 16-parameter sharing and guide/model definitions |
| `ASD_Human_Data/human_gaussian_delay_svi_utils_wide.py` | Joint/marginal likelihood and predictions, after shared values are broadcast |
| `ASD_Human_Data/gaussian_delay_likelihood_wide.py` | Truncated Gaussian delay convolution and retained-window normalization |
| `ASD_Human_Data/human_plain_ddm_diagnostics_utils.py` | DDM density/CDF and independent NumPy reference |
| `ASD_Human_Data/plot_human_shared_gaussian_delay_svi.py` | Data/model, posterior-boundary and parameter-comparison reporting |
| `ASD_Human_Data/verify_human_shared_gaussian_delay_svi.py` | Detailed source/quadrature/posterior audit examples |

Other included Python modules are transitive local imports, not additional
fitting entry points. Some describe older 61/63/83-parameter models in their
docstrings. The final helper's actual sampling sites define the reference
16-parameter fit. `reference_manifest.json` lists included files and SHA-256s.

The bundle includes the original small initialization records and extraction
provenance, not participant trial files, posterior draws or checkpoints. Raw
data must be supplied separately by the collaborator. The historical audit
script also expects archived baseline trial exports for byte comparisons;
reuse/adapt those checks rather than pretending the archives are bundled.

**These are reference sources, not a generic configuration-driven fitter.**
The driver CLI exposes group, seed, output, step limits, confirmation windows,
resume and preflight. Data/hash/count checks, pooling, priors and model sharing
are set in helper code. For a different agreed model the agent must update the
relevant loader, model, predictions and validation together in its working copy.
Do not ask the human to edit code or bypass failed checks without replacing
them with checks appropriate to the agreed selection.

## Dataset conventions to inspect and confirm

- Human visual task: choose the brighter of two squares. `RT` is simultaneous
  free-response, `SD` is simultaneous short-duration (30 ms here), `CB` is
  sequential contraction-bias. These are different conditions.
- `Choice=+1` means right; `−1` means left. `Delta_dB` is right minus left visual
  dB. Do not treat these numbers as auditory ILDs or impose an auditory conversion.
- `ASL_cd` labels mean luminance levels 20/40 cd/m². `Mean_dB` is the arithmetic
  mean of `Stim_left_dB` and `Stim_right_dB`, a different coordinate. They are not
  interchangeable column names.
- The reference CSV `ASD_Human_Data/simultaneous_rt_trials.csv` contains 11,400
  valid RT trials from 19 participants: 6,600 Control/11 people, 4,800 ASD/8.
  The reference loader verifies its hash and that of
  `ASD_Human_Data/table_visualTask_15nov2026.mat` using the bundled
  `simultaneous_rt_extraction.json`.
- Historical final selection: `Condition == 'RT'`, `Valid == 1`, inclusive
  `0 <= RT_s <= 1.8`. This retains 6,574 Control and 4,770 ASD; excludes 56;
  retains four RTs below 200 ms. These are reference counts, not mandatory
  counts for a changed filter. The export already contains retained trials;
  do not infer an acquisition deadline from its maximum RT.
- Ten signed differences are `[-2,-1.3,-0.65,-0.3,-0.1,0.1,0.3,0.65,1.3,2]`.
  Two separate ASLs yield 20 stimulus combinations; pooling ASLs yields ten.

## Historical final model: explain it before offering it

Control and ASD are separate group-pooled fits. Within each group:

| Parameter | Number | Sharing |
|---|---:|---|
| Gamma_D | 10 | Independent signed differences; ASLs pooled |
| Omega | 1 | All stimuli |
| Gaussian delay mu, sigma | 2 | All stimuli |
| Starting bias w | 1 | All stimuli |
| Lapse fraction p, lapse-right q | 2 | All stimuli |
| Total | 16 | Per group |

Gamma/Omega use normalized bounds −1,+1, start `2w−1`, drift Gamma and normalized
time `u=Omega*t_decision`. For raw `dX=v dt+sigma_noise dW` and bounds ±a,
`Gamma=v*a/sigma_noise²`, `Omega=sigma_noise²/a²`. The two-parameter base DDM
assumes bias/delay are fixed. For centered start, no lapses and unlimited time,
`P(right) = logistic(2 Gamma) = (1+tanh Gamma)/2`. Use the actual integrated,
retained-window predictions for the fitted model rather than this shortcut.

Delay per trial is Normal(mu,sigma) truncated at zero with no upper truncation.
Mu/sigma are the underlying Gaussian's parameters, not its truncated moments.
They are not fixed per-trial delays. There is no proactive process. Lapse RTs
are Uniform(0,T) in stored RT coordinates, independently of a right choice with
probability q. Here T=1.8 and p means lapse fraction among retained trials.

For chosen bound y and condition D:

```
g(d) = Normal_pdf(d; mu,sigma) / Phi(mu/sigma), d >= 0
h_y(t|D) = integral_0^t f_y(t-d|Gamma_D,Omega,w) g(d) dd
Z_D = sum_y integral_0^T h_y(t|D) dt
p(t,y|D, retained) = (1-p) h_y(t|D)/Z_D + p*q_y/T
q_right=q; q_left=1-q
p(t|D, retained) = p(t,right|D,retained) + p(t,left|D,retained)
```

For a nonzero lower RT bound or a different lapse family, rederive the selected
likelihood. Choosing p after retention versus before retention changes mixture
weights. Do not simply substitute a cutoff into one density while leaving the
normalizer or lapse support unchanged.

Historical independent trapezoidal priors, with linear tails:

| Parameter | Hard limits | Flat region |
|---|---|---|
| Gamma, positive D | [−1,20] | [0,12] |
| Gamma, negative D | [−20,1] | [−12,0] |
| Omega (s⁻¹) | [0.01,15] | [0.1,12] |
| mu (s) | [0,1] | [0.01,0.2] |
| sigma (s) | [0.0005,0.3] | [0.01,0.15] |
| w | [0,1] | [0.4,0.6] |
| p | [0.0001,0.4] | [0.001,0.1] |
| q | [0.001,0.999] | [0.4,0.6] |

Gamma priors favor the expected sign but permit small reversals. Do not impose
exact symmetry or a fixed sign unless agreed. Priors and sharing are modeling
decisions to explain, not hidden technical defaults.

## Useful checks without starting a fit

The agent can check imports/CLI with CPU set before execution:

```bash
JAX_PLATFORM_NAME=cpu .venv/bin/python ASD_Human_Data/fit_human_shared_gaussian_delay_svi.py --help
```

After the reference data are supplied, `--preflight-only` checks numerical
likelihoods, gradients and the guide. It is not an independent simulation or
recovery fit. The standalone `scripts/check_reference_simulation.py` in this
skill checks independent simulated RT/choice distributions against the bundled
Gaussian-delay likelihood without reading any participant data. Its defaults
are explicitly illustrative, not proposed priors or fitted human parameters.

For changed models, use corresponding simulator/prediction/likelihood components
and verify the same selection. Preserve the user-approved meaning of parameters
when adapting these sources.
