# Human ASL-pooled Gamma/Omega/Gaussian-delay fits

Control and ASD are fitted separately. All retained raw trials enter the joint
RT+choice likelihood, pooling ASL 20 and 40 at each signed Delta_dB. Selection is
Condition RT, Valid 1, 0 <= RT_s <= 1.8: Control 6574/11 participants and ASD
4770/8 participants. Both choices and all four RTs below 200 ms are included.

Each group has 34 parameters: Gamma[10], scalar Omega, Gaussian delay mu[10]
and sigma[10], scalar w, lapse_prob and lapse_prob_right. The delay is drawn
independently per trial from the Gaussian truncated at zero, without an upper
cutoff. The likelihood includes both Gaussian truncation and 0–1.8 s response
window normalization. Uniform lapse RT is on 0–1.8 s in stored RT coordinates.

Initialization uses the original saved centers in the preceding 64-parameter
fit's initialization.json files. Matching ASL Gamma/mu/sigma pairs are averaged
arithmetically; the original scalar values are copied. No completed posterior
is used to initialize these fits. All priors retain the approved hard/flat ranges.

Both groups run concurrently on GPU. Full-batch float64 AutoMultivariateNormal,
one ELBO particle, seed 0, guide scale 0.1, ClippedAdam learning rate 0.0002,
gradient clip 1. Check nonoverlapping 1000-update average losses: each strict new
minimum resets patience. Stop after >=50000 updates and 25 subsequent windows
without improvement, capped at 1000000. Restore the saved lowest-window-loss
checkpoint. There is no posterior parameter-stability stopping gate.

The fitter is ASD_Human_Data/fit_human_pooled_asl_shared_omega_svi.py; it exposes
--group, --seed, --output-dir, --initial-root, --min-steps, --max-steps,
--confirmation-windows, --resume-existing, and --preflight-only. Always invoke
with the repository .venv/bin/python. GPU memory preallocation is disabled.

posterior_samples.npz stores 10000 unsorted, aligned joint draws per group.
Gamma, delay_mu and delay_sigma arrays have shape (10000,10); Omega, w and the
two lapse arrays have shape (10000,). Signed differences increase from -2,
-1.3, -0.65, -0.3, -0.1, +0.1, +0.3, +0.65, +1.3, +2. parameter_names in
run_config.json gives the 34-dimensional covariance/correlation ordering.

Use plot_human_pooled_asl_shared_omega_svi.py and
verify_human_pooled_asl_shared_omega_svi.py for figures and independent audits.
Posterior intervals are pooled-trial variational approximations and exclude
participant-to-participant variability. Confirmation does not prove a global
optimum. Cap stops without confirmation are explicitly flagged.
