# Human Gaussian-delay Gamma/Omega SVI

Joint RT + choice + uniform lapse; 83 parameters per group. Saved output and methodology: [22 October result-book page](../../docs/results/2026-10-22.md#human-rt-wider-gaussian-delay-bounds-choice-and-uniform-lapses).

Reproduction sources are the fit_human_gaussian_delay_svi_wide.py, plot_human_gaussian_delay_svi_wide.py, verify_human_gaussian_delay_svi_wide.py and publish_human_gaussian_delay_svi_wide.py scripts one directory above. Use a separate --output-dir for a new fit; --resume-existing continues saved checkpoints. run_config.json freezes likelihood source hashes, priors, original initialization and numerical settings.

Earlier Gaussian fit and simulation outputs remain unchanged. Current independent verification includes extreme-bias and narrow-delay cases and posterior numerical refinement checks.
