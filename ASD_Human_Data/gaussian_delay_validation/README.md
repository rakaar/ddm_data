# Gaussian trial-varying delay: synthetic likelihood validation

Implemented and checked on 22 October 2026. This is a synthetic validation,
not a fit to human data or a parameter-recovery experiment.

## Model

Every trial draws one independent delay from a Gaussian **truncated at zero,
with no upper cutoff**, as requested. This is variability between trials.
The Gaussian parameters `delay_mu` and `delay_sigma` are the underlying
Gaussian location and scale, not posterior uncertainty. Truncation changes
the actual delay mean and SD; both are saved in `parameter_cases.csv`.

The dimensionless diffusion has bounds −1/+1, start `2*w-1`, drift Gamma,
and time `s = Omega * decision_time`. In physical time:

```text
dX = Gamma * Omega * dt + sqrt(Omega) * dW
RT = delay + decision_time
h(delay) = phi((delay-mu)/sigma) / [sigma * Phi(mu/sigma)], delay >= 0
f_y(RT) = integral_0^RT f_DDM,y(RT-delay) * h(delay) d(delay)
```

`f_y` is a **joint time-and-choice density**; the left and right densities
together integrate to one. The RT-only density is their sum. Delays and
diffusion paths are independent. There is no proactive process or lapse.

When retaining an RT window `[L,U]`, the objective divides either density
by the same either-bound mass `Z = sum_y [F_y(U)-F_y(L)]`. Integrating over
random delays happens **before** this normalization. Normalizing each delay
draw separately would define a different model. This validation checks the
0–1.8 s window as well as unconditional distributions.

## Three tests

| Case | Gamma | Omega (s⁻¹) | w | Gaussian mu (ms) | Gaussian sigma (ms) | Actual delay mean / SD (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Zero drift | 0 | 2.5 | 0.50 | 280 | 120 | 283.18 / 116.19 |
| Positive drift | 1.8 | 3.5 | 0.38 | 240 | 180 | 272.48 / 153.45 |
| Negative drift | −3 | 1.4 | 0.62 | 400 | 200 | 411.05 / 188.30 |

For each case, simulate 200,000 paths at each of two dimensionless timesteps
(0.001 and 0.0005), giving 1.2 million independent trials across six runs.
The independent simulator steps the stochastic differential equation; it
does not sample from the tested density/CDF. Brownian-bridge crossing
probabilities catch exits between sampled endpoints. The detected interval's
midpoint supplies the crossing time, and halving the step checks that
remaining discretization error. No simulated paths are time-censored.

The figure uses the finer simulations. Grey steps show empirical RTDs;
black curves show the random-delay likelihood. Green dashed curves use a
**fixed delay at the same actual mean**, making the effect of variability
visible without changing the mean. The left/right panels retain both choices
in the denominator; they are not separately normalized histograms.
Plots are display zooms, with no renormalization to the displayed interval.

## Implementation and accuracy

- `../gaussian_delay_likelihood.py` provides reusable, differentiable JAX
  density/CDF and joint/RT-only log-likelihood functions, in float64.
- The fast method computes each condition's first-passage density once on a
  0.5 ms grid, then uses zero-padded FFT convolution and linear interpolation.
  Zero padding prevents circular wraparound. Trapezoidal integration includes
  the nonzero truncated-Gaussian density at delay zero correctly.
- Its image/eigenfunction kernel uses the established dimensionless-time
  switch at 0.6. The fast default, K=8, is checked against K=40 and K=80,
  including all objective gradients. A separate kernel check covers 64
  parameter sets, including Gamma ±20 and bias near either boundary.
- Near-empty FFT tails can contain floating-point cancellation. The objective
  falls back to positive log-space Gauss–Legendre quadrature there rather
  than using an artificial density floor as the likelihood.
- Independent adaptive SciPy integrals check joint densities and bound CDFs.
  Grid refinement checks 1, 0.5 and 0.25 ms. All five parameter derivatives
  are compared with finite differences of the independent integrals.
- Other checks cover eventual bound masses, conditional unit mass, delay
  sample means, retained choice probabilities, simulation CDF envelopes,
  RT-only choice-label invariance, and finite gradients for tiny densities.

Observed worst absolute density discrepancy against adaptive integration is
approximately `5.1e-6 s^-1`; worst CDF discrepancy is `4.3e-7`. Maximum absolute
gradient discrepancy is below `1e-6`. The synthetic 6,600-trial log-likelihood
differs from 96-node direct quadrature by about `9.8e-8` per trial.
The simulation CDF discrepancies are within the recorded sampling envelopes.

The saved timing record uses eight CPU cores and a 20-condition, 6,600-trial
synthetic workload with 81 physical parameters (20 Gamma, Omega, delay means,
delay SDs, plus one shared bias). Warmed K=8 likelihood-and-gradient evaluation
is approximately **65–70 ms**, compared with approximately 270–280 ms at K=40.
Compilation and value-only timings are recorded separately. These are
likelihood benchmarks, **not full SVI-update timings**.

This establishes accuracy in the three tested broad-delay regimes. Before a
fit with new parameter supports, repeat resolution checks for the proposed
range, especially very narrow delay SDs, extreme bias or very fast decisions.
The kernel stress test does not by itself validate convolution resolution at
every combination of those parameters.

## Reproduce

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-7 .venv/bin/python -u ASD_Human_Data/validate_gaussian_delay_likelihood.py
```

Use `--skip-simulation` to reuse saved simulations with matching parameters,
seed and timestep. `--trials`, `--seed`, `--threads` and `--output-dir` are
configurable. The three parameter sets are together near the top of the
validation script.

## Artifacts

- `gaussian_delay_simulation_validation.png` / `.pdf`: three-case comparison.
- `validation_report.json`: gates, numerical errors, timing, settings and hashes.
- `parameter_cases.csv`: Gaussian parameters and actual truncated moments.
- `case_{1,2,3}_{coarse,fine}_simulation.npz`: RT, choice, sampled delay,
  decision time, parameters, seed and dimensionless timestep.
- `independent_quadrature.csv`: independent density/CDF comparisons.
- `simulation_histograms.csv`: empirical counts, denominators and expected counts.
- `likelihood_curves.csv`: joint densities/CDFs and fixed-mean-delay comparison.

![Three-case validation](gaussian_delay_simulation_validation.png)
