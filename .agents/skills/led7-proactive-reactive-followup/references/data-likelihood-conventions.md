# LED7 Data And Likelihood Conventions

## Timing Coordinates

- `timed_fix` is fixation-time coordinate.
- `intended_fix` is the stimulus-onset time, also used as `t_stim` in current scripts.
- `RTwrtStim = timed_fix - intended_fix`.
- Current LED scripts compute `t_LED = intended_fix - LED_onset_time`.
- Keep each observed `(t_LED, t_stim)` pair intact. Do not independently resample the two columns.
- Fits use exact timings, not binned LED times. Binning or Monte Carlo timing draws are visualization choices only.

## Shared Behavioral Filters

For the current LED7 analyses, confirm the exact script before reuse. Typical common filters are:

- animals 92, 93, 98, 99, 100, and 103;
- session type 7 and training level 16;
- repeat-trial values 0, 2, or missing where the historical FCT-compatible abort route requires it;
- finite timing and stimulus columns.

Do not assume abort and valid-trial filters are identical.

## Abort And Censored Likelihood

The current no-truncation+lapse route fits the mixture

`(1 - lapse_prob) * proactive_process + lapse_prob * exponential_lapse_process`.

For an abort at fixation time `t`, the lapse component is

`beta_lapse * exp(-beta_lapse * t)`.

For a censored/successful trial with stimulus time `t_stim`, its lapse survival is

`exp(-beta_lapse * t_stim)`.

Thus the stable likelihood terms are:

- abort density: `(1-p) * f_A(t) + p * beta * exp(-beta*t)`;
- censored survival: `(1-p) * (1-F_A(t_stim)) + p * exp(-beta*t_stim)`.

Evaluate mixtures in log space. There is no `1-F(0.3)` denominator in the accepted route, and early aborts are retained.

LED OFF uses the baseline proactive branch. LED ON uses the analytic post-LED drift step and exact trial timing. The accepted model is an upward-step model; do not introduce a downward-jump branch without a new hypothesis decision.

## Valid RT+Choice Likelihood

The current valid-trial fits:

- include successful choices only;
- condition on `0 <= RTwrtStim <= 1 s`, inclusive;
- fit LED OFF and bilateral LED ON independently;
- fix all seven proactive/lapse parameters at aggregate-abort posterior means;
- use the NPL+alpha reactive model with 30 signed `(ABL, ILD)` delays.

Choice coding is poke 3 to `+1` and poke 2 to `-1`. The lapse component is unbiased over choices, contributing half of its density to each bound.

Fitted shared reactive parameters:

- `rate_lambda`
- `T_0`
- `theta_E`
- `w`
- `del_go`
- `rate_norm_l`
- `alpha`

Condition-specific parameter:

- `t_E_aff` for each of the 30 signed ABL/ILD conditions.

## LED Scope

- **Abort model LED ON:** all selected `LED_trial == 1` rows, including unilateral and bilateral powers.
- **Valid LED ON model:** bilateral rows only, requiring positive left and right LED powers.
- **LED OFF:** LED flag zero or missing according to the source script.

State this distinction in every fit caption and comparison table.

## RTD Display Rules

A diagnostic RTD can use more rows than the valid-trial fit:

- current OFF full RTD includes abort event 3, abort event 4, and successful trials;
- current bilateral-ON full RTD includes abort event 3 and successful trials, excluding event 4.

Scale by the declared eligible-trial denominator. Do not normalize each panel or displayed window to area one unless the figure is explicitly a conditional-density plot.

For a finite plotting window, report displayed mass as the probability in that window. It can be below one because of omitted tails. Before diagnosing normalization, compare:

1. full-support CDF mass;
2. PDF integral over the plotted window;
3. CDF difference over the same window.

The existing validation found full total mass equal to one and PDF/CDF window agreement within numerical tolerance.

## Pooled And Averaged Results

- A super-animal fit pools trials; animals with more trials contribute more likelihood.
- An equal-animal curve averages completed animal-level curves with equal weight.
- A pooled diagnostic histogram combines raw trials.
- A trial-count-weighted mixture of animal-specific model curves is not the same as fitting one aggregate parameter vector.

Use these labels literally rather than calling all four constructions “average.”

## Convergence And Initialization

- Negative ELBO is minimized.
- Loss-window means are diagnostics; posterior samples come from the restored-best optimizer state.
- Record both restored-best and final-checked steps.
- Warm starts are initialization aids, not fixed parameters.
- A deliberately displaced start tests recovery and local-minimum sensitivity.
- When branch losses are within the configured ELBO tie tolerance, compare posterior-mean likelihood on identical data and preserve the selection record.
- A fit reaching the step limit without patience convergence is inconclusive unless separately audited and accepted.

## Common Failure Modes

- Applying the historical 300 ms truncation to the accepted no-truncation fit.
- Filtering all successful responses below 300 ms rather than filtering only the intended abort population in a truncated analysis.
- Treating event 4 identically in fit and display datasets without checking the script.
- Mixing unilateral LED ON into the bilateral valid fit.
- Breaking `(t_LED, t_stim)` pairing during Monte Carlo theory evaluation.
- Comparing raw ELBOs across different datasets or likelihood definitions.
- Choosing a start because its curve looks smoother despite a saved objective-based selection summary.
- Inferring total mass from an `xlim`-cropped RTD.
