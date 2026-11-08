# Human RT: one shared Omega per group

This is a new 64-parameter joint RT/choice fit, with independent Control and ASD models.
Each has 20 Gamma, one shared Omega, 20 Gaussian delay mu, 20 Gaussian delay sigma,
one starting bias, one lapse fraction and one lapse-right probability.
All parameters are refitted. Omega is shared across both ASLs and all ten signed differences.

Selection: Condition == RT, Valid == 1, inclusive RT_s in [0, 1.8] seconds.
Control: 6,574 trials / 11 participants. ASD: 4,770 trials / 8 participants.
Pool raw trials, including both choices and all four RTs below 200 ms.

The Gaussian delay is truncated at zero with no upper cutoff. Its density includes
normalization by Phi(mu/sigma). Each observed-choice density is divided by the
condition-specific either-bound retained mass. Lapse RT is Uniform(0, 1.8 seconds),
unshifted; lapse probability is the fraction among retained trials.

The Omega trapezoidal prior is applied once per group: hard [0.01, 15], flat [0.1, 12].
All other priors, numerical methods and SVI settings match the widened-bound
83-parameter Gaussian-delay fit. Saved original initialization is reused, replacing
20 initial Omegas with their arithmetic mean: Control 0.7168707536348256,
ASD 0.8118221105605528. No completed posterior is used for initialization.

Sources are the fit_human_shared_omega_gaussian_delay_svi.py,
human_shared_omega_gaussian_delay_svi_utils.py,
plot_human_shared_omega_gaussian_delay_svi.py,
verify_human_shared_omega_gaussian_delay_svi.py and
publish_human_shared_omega_gaussian_delay_svi.py scripts in ASD_Human_Data/.
Use the repository .venv/bin/python interpreter. Fit CLI exposes --group, --seed,
--min-steps, --max-steps, --confirmation-windows, --output-dir,
--initial-root, --resume-existing and --preflight-only.

Each posterior_samples.npz contains 10,000 unsorted joint draws after fitting;
rows are aligned across all parameter arrays. Gamma/mu/sigma have shape
(10000,20); Omega/w/lapse_prob/lapse_prob_right have shape (10000,).
Conditions are ASL 20 then 40, signed Delta_dB ascending within each ASL.
The run configuration records the 64-parameter covariance/correlation ordering.

Status is recorded in each group convergence_summary.json when completed.
The stopping gate uses terminal stability; local stability at the restored best
checkpoint is reported separately. Passing either does not establish a global optimum.
Posterior intervals are pooled-trial variational approximations, not participant variability.

## CPU-to-GPU continuation

The user approved GPU continuation after a float64 benchmark. Control resumes
at 25,000 updates and ASD at 32,000, with optimizer/RNG states and all completed
loss history preserved. Both use the RTX 4090 concurrently; memory preallocation
is disabled. The likelihood, priors, seed and convergence settings are unchanged.
`gpu_resume.json`, `cpu_gpu_benchmark.json`, `source_history/`, and each group's
`before_gpu/` record the transition. The fitter accepts `--allow-platform-resume`
only for an audited CPU-to-GPU continuation with matching mathematical settings.
The 100-update benchmark agreed to 5.23e-12 in losses and 3.32e-14 in optimizer
state; warm GPU timing was 0.474 s versus 19.04 s CPU under concurrent load.

A second GPU continuation at Control 43,000 / ASD 56,000 updates reuses the
compiled 1,000-update `stable_update` scan. Its output matches the prior
`SVI.run` scan to float64 tolerance on both copied checkpoints (maximum loss
difference 2.51e-12, state difference 8.20e-14). No optimizer, RNG, likelihood,
prior or stopping settings changed. See `compiled_loop_resume.json`,
`gpu_compiled_loop_benchmark.json` and `before_compiled_loop/`. The explicit
`--allow-loop-resume` flag validates the archived source/configuration.

## Final loss-based Control selection

After 289,000 Control updates, the user requested stopping at the loss plateau
and restoring the lowest saved 1,000-update mean loss. The selected checkpoint
is 95,000 updates (mean negative ELBO -572.160617), with 194 subsequent windows
without improvement. The final, higher-loss checkpoint is preserved for resuming
but is not used for posterior summaries or predictions. Control did not pass
the original parameter-stability gate and remains explicitly flagged.
ASD stopped at 83,000 updates under the original rule and restored 52,000.

`user_requested_finalize.json` records this authorized stopping change;
`control/before_user_requested_finalize/` preserves the coherent pre-stop files.
`finalize_human_shared_omega_saved_best.py` draws the posterior from the saved
best guide without performing further optimizer updates. Both final terminal
and restored-checkpoint stability values are reported in the group summaries.
