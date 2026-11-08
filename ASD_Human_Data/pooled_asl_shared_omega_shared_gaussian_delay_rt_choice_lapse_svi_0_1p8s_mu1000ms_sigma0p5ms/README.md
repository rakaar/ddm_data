# Human RT with one Gaussian delay distribution per group

Control and ASD are fitted independently with ASLs pooled during fitting.
Condition RT, Valid 1, inclusive 0–1.8 s: Control 6574 trials / 11 participants;
ASD 4770 / 8 participants. Both choices and all four RTs below 200 ms are kept.
The selected/excluded source identities match the completed 34-parameter run.

Each group has 16 parameters: Gamma[10], scalar Omega, scalar delay_mu,
scalar delay_sigma, scalar w, lapse_prob and lapse_prob_right. Gamma is ordered
by signed Delta_dB = -2,-1.3,-.65,-.3,-.1,.1,.3,.65,1.3,2. All parameters are
shared across ASLs; Gaussian mu/sigma also are shared across all differences.
Trial delays are independent Gaussian draws truncated below zero, without an
upper cutoff. This Gaussian describes trial variability. The guide describes
posterior uncertainty about its parameters.

Likelihood: (1-p) h_choice(rt) / Z_condition + p q_choice / 1.8.
The Gaussian density includes normalization by Phi(mu/sigma). Z_condition is
the either-bound response probability in 0–1.8 s and remains condition-specific.
The lapse component is Uniform(0,1.8 s) in stored RT, with p the fraction among
retained trials. The mu and sigma priors each contribute once per group.
Priors retain the 34-parameter hard/flat ranges; mu hard [0,1] s, sigma hard
[0.0005,0.3] s. See run_config.json for every normalized trapezoidal prior.

Initialization copies the original saved Gamma/scalar centers from the
34-parameter run, and averages its ten original mu values and sigma values.
No fitted posterior is used to initialize. Mean delay starts at 102.787 ms
Control / 173.426 ms ASD, sigma at 50 ms, p=.02 and q=.5.

Both groups run concurrently on GPU. Float64 full-batch AutoMultivariateNormal,
one ELBO particle, seed 0, guide scale .1, ClippedAdam lr .0002 and clip 1.
Loss-only stopping checks nonoverlapping 1000-update means; any strict new
minimum resets patience. Require >=50000 updates and 25 subsequent windows
without improvement, with cap1000000; restore the lowest-loss checkpoint.
There is no posterior-mean/SD stability stopping gate. An unconfirmed cap is flagged.

The fitter fit_human_shared_gaussian_delay_svi.py exposes --group, --seed,
--output-dir, --initial-root, --min-steps, --max-steps, --confirmation-windows,
--resume-existing and --preflight-only. Use .venv/bin/python. GPU memory
preallocation is disabled. Checkpoints include complete optimizer/RNG state.

posterior_samples.npz stores 10000 aligned, unsorted joint draws per group.
Gamma has shape (10000,10), all six shared parameters have shape (10000,).
run_config.json records the parameter order of the physical 16x16 posterior
covariance/correlation matrices. Draw rows align within each group.

Plot and verify with plot_human_shared_gaussian_delay_svi.py and
verify_human_shared_gaussian_delay_svi.py. Figures are evaluated at arithmetic
physical posterior means; intervals are equal-tailed 95% variational credible
intervals for pooled trials, not participant variability. The book section
is Human RT: one shared Gaussian delay per group, on 23 October 2026.
Loss confirmation does not establish a global optimum.
