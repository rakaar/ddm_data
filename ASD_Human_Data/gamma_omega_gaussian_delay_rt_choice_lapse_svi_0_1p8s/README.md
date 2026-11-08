# Human Gaussian-delay Gamma/Omega SVI

Joint RT + choice + uniform lapse; 83 parameters per group. Saved output and methodology: [22 October result-book page](../../docs/results/2026-10-22.md#human-rt-trial-varying-gaussian-delay-choice-and-uniform-lapses).

Reproduction sources are the fit_human_gaussian_delay_svi.py, plot_human_gaussian_delay_svi.py, verify_human_gaussian_delay_svi.py and publish_human_gaussian_delay_svi.py scripts one directory above. Use a separate --output-dir for a new fit; --resume-existing continues saved checkpoints. run_config.json freezes likelihood source hashes, priors, original initialization and numerical settings.

The original pre-extension Gaussian prototype source is archived in synthetic_validation_original_sources/. The old three-case simulation reports retain their original source hashes. Current independent verification includes extreme-bias and narrow-delay cases and posterior numerical refinement checks.
