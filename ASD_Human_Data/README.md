# Visual-task human data: first simultaneous analysis

Source: `table_visualTask_15nov2026.mat`, MATLAB variable `Ttrials` (a saved
`table`). Run `.venv/bin/python ASD_Human_Data/prepare_simultaneous_rt_trials.py`
from the repository root to regenerate `simultaneous_rt_trials.csv` and
`simultaneous_rt_extraction.json`. The CSV has `source_row` (one-based row in
`Ttrials`) so each exported row can be traced to the MAT file. The JSON records
the source and output hashes, source columns, and row counts. CSV files and MAT
files are ignored by this repository's `.gitignore`.

## Selection and definitions

The first analysis uses **`Condition == "RT"` only**: 11,400 rows from 19
participants (8 ASD, 11 Control). `SD` also has simultaneous left/right
stimuli, but its 6,080 rows have `Stim_duration_s == 0.03` and are excluded.
The 11,400 `CB` rows are sequential and excluded. All 28,880 source rows have
`Valid == 1`; the extractor applies no additional correctness or RT cutoff.

For the selected trials:

- `Mean_dB = (Stim_left_dB + Stim_right_dB) / 2` is the requested average
  brightness on the **dB scale**.
- Stored `Delta_dB = Stim_right_dB - Stim_left_dB`; positive values mean the
  right square has the larger stored dB value. The meeting notes call this
  "ILD" by analogy, but these are visual stimuli, so the export keeps the
  source name `Delta_dB`.
- `Choice == 1` means right and `Choice == -1` means left, as confirmed by the
  stored `ChoseRight` column. `Correct` agrees with choosing the side with
  larger `Stim_*_dB` on every row.
- `RT_s` is the stored reaction time. The meeting notes say its origin is
  stimulus presentation, but no event timestamps in this table independently
  establish that origin. In selected RT rows it ranges from 0.111190 to
  2.499536 s.
- `ASL_cd` is the stored nominal brightness level: 20 or 40 for RT rows. It
  is kept separate from `Mean_dB`; an arithmetic mean of the two cd/m² values
  can differ from `ASL_cd`.

The file itself reveals the numerical luminance conversion. For **every**
left/right stimulus value in all 28,880 rows (57,760 values total),
`Stim_*_cdm2 = 7.72 + 10^(Stim_*_dB/10)` to floating-point precision. Hence
`Stim_*_dB = 10*log10((Stim_*_cdm2 - 7.72)/1)` numerically, where both 7.72
and the implicit reference 1 carry cd/m² units. The 7.72 offset's physical
meaning is not supplied in this table. It may reflect a display background,
but that needs confirmation from the experimenter. `ASL_cd` also equals
`7.72 + 10^(Mean_dB/10)` to floating-point precision. This is a
**dataset-specific mapping**: a decibel value is a logarithmic ratio and
requires a stated reference, rather than being a universal conversion from
cd/m² ([NIST guidance](https://www.nist.gov/pml/special-publication-330/sp-330-section-4)).

Useful row identity fields are `Participant`, `GroupID`, `Group`, `Trial`,
`Session`, `Block`, and `source_row`. Useful stimulus and response fields are
`ASL_cd`, `Stim_left_dB`, `Stim_right_dB`, `Mean_dB`, `Delta_dB`, `RT_s`,
`Choice`, `ChoseRight`, and `Correct`. The export also preserves the two
`Stim_*_cdm2` values, `Delta_cdm2`, `Stim_duration_s`, `Condition`, and `Valid`
for checking units and filters. `GroupID == 0` maps to Control and
`GroupID == 1` maps to ASD in every saved row.

## Structure checked in the saved table

- Each participant has 1,520 valid rows: two simultaneous sessions with 460
  rows each and two CB sessions with 300 rows each. Each session has four
  blocks.
- Each simultaneous block has 75 RT and 40 SD rows, all at one `ASL_cd`.
  Each simultaneous session uses both 20 and 40 cd/m², with two blocks at
  each level. Thus the RT-only export has 75 rows per selected block.
- CB blocks have 75 rows and mix five `ASL_cd` values. `FirstSide == 1` means
  left first, `FirstSide == 2` means right first: `Stim_first_dB` matches that
  side on all CB rows. The source field is `ChoseSecond` (not `ChooseSecond`);
  `ChoseSecond == 1` agrees with choosing the second side on all CB rows.
  These fields are missing in RT and SD rows.
- Session 1 is simultaneous for all 19 participants. Session 2 is
  simultaneous for 10 participants and CB for 9. This matches the note that
  the second session's task is not fixed.

## RT tail and remaining questions

The meeting notes say that rows with RT above 3 s were removed. Among the
11,400 retained RT-condition rows, 4 have `RT_s >= 2.4`, 1 has
`RT_s >= 2.49`, and none has `RT_s >= 2.4999`. Their maximum is
2.499536 s. Across all conditions, the maxima are RT 2.499536, SD 2.493084,
and CB 2.487314 s. There are no rows with `RT_s >= 2.5` in the MAT file.
There is no cluster of values at 2.5 s; the observed upper bound could
reflect a response window or upstream trial selection. The valid-trial-only
table cannot distinguish these possibilities or show the discarded >3 s
trials. Ask the experimenter how the 3 s rule relates to `RT_s` measured
from stimulus presentation and whether another 2.5 s bound was applied.

The data were received on **15 October 2026**, per the recipient, and the
MAT header also says it was created that day. The filename contains
`15nov2026`; retain the filename for provenance, but do not treat it as the
data's date or infer trial dates from it.

The extractor reads the saved MATLAB table through SciPy's private MAT reader
because the normal `loadmat` result is opaque. Its internal cell indices are
specific to this file's serialization; rerun its checks if the source is
re-exported. A plain MATLAB `writetable` CSV would avoid that dependency.

## JND and slope figures

Run `.venv/bin/python ASD_Human_Data/plot_simultaneous_rt_metric_deviations.py`
and then `.venv/bin/python ASD_Human_Data/verify_simultaneous_rt_metric_deviations.py`.
The output folder is `ASD_Human_Data/simultaneous_rt_metric_deviations/`.
It contains PNG and PDF pairs, 380 measured condition probabilities,
38 fitted curves and JND crossings, metric-specific eligibility, the plotted
values, population summary statistics, PDF mark coordinates, and two
validation manifests. The plotting script checks the extracted CSV's hash
against `simultaneous_rt_extraction.json` before fitting.
The horizontal participant-mean marks are blue for Control and red for ASD
in both figures; the ASL-specific choice estimates remain blue/orange.
The vertical bars at the right use the same group colors and show each
group's mean ±1 population SD. The upper bars summarize participant means
within group; the lower bars summarize both signed ASL deviations from each
eligible participant in that group. `group_summary_statistics.csv` records
those eight group-by-metric summaries. Each bar's group name, mean, and SD
appear directly above it to three decimals. The black dotted upper line and
`summary_statistics.csv` still give the pooled overall summaries.

The figures use `ASL_cd` (20 and 40 cd/m²) as the nominal brightness level
and signed `Delta_dB` (right minus left) as the horizontal psychometric
coordinate. Each participant/ASL curve fits ten measured `P(right)` values
with equal weight. It uses the same four-parameter sigmoid, parameter bounds,
initial values, and crossing-based JND as the linked animal figures. Slope is
the fitted parameter `k` in dB⁻¹. Each participant's metric mean is the
arithmetic average of the 20 and 40 cd/m² estimates; signed deviations are
the ASL estimate minus this mean. The JND and slope eligibility rules are
applied independently, and an undefined JND is saved as missing rather than
imputed. For the present dataset all 19 participants are eligible for both.

## Participant 47 fit-rule diagnostic

Run `.venv/bin/python ASD_Human_Data/diagnose_participant47_psychometrics.py`
after the main plotting script. It reconstructs the 20 measured probabilities
from Participant 47's 600 RT trials and plots the same points under three
curve-fitting rules. `participant47_psychometric_fit_comparison.png` and its
PDF, plus the standalone `participant47_current_psychometric.png`/PDF,
`participant47_condition_counts.csv`, and
`participant47_fit_rule_comparison.csv`, are saved in the metric-deviations
output folder. This is a sensitivity check, not a replacement JND policy.

The current unweighted four-parameter fit gives JND 0.279 dB at ASL 20 and
0.111 dB at ASL 40. A four-parameter binomial-likelihood fit to those exact
same trials gives 0.246 and 0.272 dB. The latter is much closer to the
friend's screenshot labels of 0.28 dB at both levels, but no tested fit rule
reproduces all four of that screenshot's PSE/JND labels. Its source code is
needed to identify the precise fitting and crossing rules.

## Per-participant distribution pages

Run `.venv/bin/python ASD_Human_Data/plot_simultaneous_rt_subject_distributions.py`
and `.venv/bin/python ASD_Human_Data/verify_simultaneous_rt_subject_distributions.py`.
The output folder `simultaneous_rt_subject_distributions/` contains a 19-page
PDF, one full-size PNG per participant, calculation tables, and validation
manifests. Participants are ordered Control IDs increasing, then ASD IDs
increasing. Every page uses that participant's 600 `Condition == "RT"`,
`Valid == 1` trials, pooled across simultaneous sessions. All left/right
choices and all recorded RTs are retained. The earlier published
four-parameter psychometric fits are reused without changing their fitting
policy.

The upper row follows the animal PDF's empirical definitions: `P(right)` by
signed `Delta_dB`; 20 ms RTD histograms normalized across all 300 RTs per ASL;
and mean RT ± sample SEM after pooling 30 positive and 30 negative difference
trials at each `|Delta_dB|` and ASL. The full RTD area is one; 0–1.5 s is
only a display zoom.

The lower row shows 1 ms CDFs, `|F20−F40|`, empirical 10/30/50/70/90% RT
quantiles, and a quantile-scaling overlay. CDF and quantile calculations use
all 600 recorded RTs per participant, including the 945 selected RTs above
one second. CDFs are normalized across all 60 signed-pooled RTs per
participant/ASL/absolute difference. The 0–1.5 s CDF view is a display zoom.
The two-ASL difference has one pair rather than the animal PDF's three-pair
mean. The first 10 ms forward-minus-backward `|ΔCDF|` rise ≥0.005 supplies a
cutoff; the current bin is omitted. Rise values are rounded to 12 decimal
places before comparison so an exact 0.005 is stable under floating-point
summation. ASL 40 is the higher-level reference, corresponding to ABL 60 in
the animal PDF. For each participant/absolute difference, the quantile scale
factor uses the animal Figure 1 formula on linearly interpolated 1–99%
reference quantiles at or above the cutoff, without the animal's 0.3 s upper
fit limit. A failed cutoff or scale fit remains unavailable; none fails in
this dataset. These are descriptive empirical transformations, not TIED model
fits. With only 60 trials per CDF, cutoff locations may be noisy.

## Control and ASD pooled as two groups

Run `.venv/bin/python ASD_Human_Data/plot_simultaneous_rt_group_pooled_distributions.py`
and `.venv/bin/python ASD_Human_Data/verify_simultaneous_rt_group_pooled_distributions.py`.
`simultaneous_rt_group_pooled_distributions/` contains a single four-row PNG
and PDF: two rows for Control above two rows for ASD, following the individual
pages' panel layout. Control pools 6,600 trials from 11 participants; ASD pools
4,800 from 8 participants. Select `Condition == "RT"`, retain both choices,
and impose no additional RT cutoff.

Combine raw trials across participants and sessions before calculating each
group's probabilities, RT distributions, means, CDFs, cutoffs, quantiles, and
scale factors. Fit four new sigmoids to the pooled ten-point psychometrics
using the established unweighted four-parameter procedure. Control has 330
choices per signed difference/ASL and 660 RTs per absolute difference/ASL;
ASD has 240 and 480. Chronometric bars show sample SEM across pooled RTs.
All axes use matching ranges across the two groups. The same all-RT CDF and
quantile rules, 1.5 s display zoom, and ASL-40-reference scaling apply.
The output tables retain group membership/counts and all newly calculated
quantities; four fits and ten pooled cutoffs/scaling factors are available.

## SD-only pooled Control and ASD figure

The extractor, pooled plotting script, and pooled verifier accept
`--condition {RT,SD}`; omitting it retains the existing RT workflow. Run:

```bash
.venv/bin/python ASD_Human_Data/prepare_simultaneous_rt_trials.py --condition SD
.venv/bin/python ASD_Human_Data/plot_simultaneous_rt_group_pooled_distributions.py --condition SD
.venv/bin/python ASD_Human_Data/verify_simultaneous_rt_group_pooled_distributions.py --condition SD
```

The SD export is `simultaneous_sd_trials.csv` with
`simultaneous_sd_extraction.json`; figures and calculation tables are in
`simultaneous_sd_group_pooled_distributions/`. The source-row and hash
provenance follows the RT export. The verifier checks every selected source
row against the original MAT table, including unchanged stored `RT_s` values.

Select only `Condition == "SD"`, with `Valid == 1` and
`Stim_duration_s == 0.03` on all 6,080 rows. Control combines 3,520 trials from
11 participants and ASD combines 2,560 trials from 8 participants. Every
participant contributes 320 trials: 16 choices per signed difference/ASL.
After pooling, Control has 176 choices per signed difference/ASL and 352 RTs
per absolute difference/ASL; ASD has 128 and 256. The two Control rows appear
above the two ASD rows, labeled **SD — 30 ms stimulus presentation**.

All probabilities, four-parameter fits, RT densities, pooled-trial means/SEMs,
CDFs, cutoffs, quantiles, and scaling factors are recalculated from the raw
SD trials. Both correct and incorrect choices and every recorded RT are
retained; `RT_s` is used without a time shift or added cutoff. The ASLs,
signed differences, 20 ms RTD bins, 1 ms CDF bins, 10/30/50/70/90% quantiles,
ASL-40-reference scaling, and 0–1.5 s display zoom match the RT pooled method.
All four fits and ten cutoff/scaling calculations are available. Separate
SD filenames preserve the earlier RT exports and figures.

## SD RTDs separately by absolute difference

Run `.venv/bin/python ASD_Human_Data/plot_simultaneous_group_rtd_by_abs_delta.py`
(default `--condition SD`). The PNG/PDF and bin/count/validation tables are
saved in `simultaneous_sd_rtd_by_abs_delta/`. The figure has Control on the
top row, ASD below, and five columns for `|Delta_dB| = 0.1, 0.3, 0.65, 1.3, 2`.
Each panel overlays ASL 20 and 40 cd/m² in blue and orange. Pool actual
positive and negative difference trials across participants and sessions
within group/ASL/absolute difference: 352 Control or 256 ASD trials per
curve. Use both choices and all stored RTs, 20 ms bins, full-range area-one
normalization, a 0–1.5 s display zoom, and matching axes across all panels.
Every 20 ms bin count is checked against the previously verified 1 ms
condition audit aggregated into 20 ms bins.

## Participant JND figure with separate RT and SD rows

Run `.venv/bin/python ASD_Human_Data/plot_simultaneous_rt_sd_jnd_deviations.py`
and `.venv/bin/python ASD_Human_Data/verify_simultaneous_rt_sd_jnd_deviations.py`.
`simultaneous_rt_sd_jnd_deviations/` contains one four-row PNG/PDF: RT JND,
RT signed ΔJND, SD JND, and SD signed ΔJND. All rows share the original
participant order (11 Control, then 8 ASD, IDs increasing within group),
ASL colors, dot/star markers, and participant mean marks. Each row has four
right-side mean ±1 population SD bars: Control ASL 20/40, then ASD ASL 20/40.
Control bars are blue and ASD bars red; ASL 20 is solid and ASL 40 dotted.
Labels above every bar give group, ASL, mean, and SD. Panels use individual
y-ranges appropriate to the displayed condition and metric.

RT parameters are reused exactly from the published fit audit. SD is fit
separately for every participant/ASL with the same unweighted four-parameter
sigmoid and initial values/bounds. Fit all 11,400 RT and 6,080 SD choices,
including errors; pool sessions within each participant/condition and apply
no added RT restriction. RT has 30 choices per signed difference/ASL and SD
has 16. JND remains half the distance between crossings at `.75*upper` and
`.25*(1-lower)`. Means and signed deviations are calculated separately
within each condition using the two ASL estimates.

All 76 fits converge, and 75 individual JNDs are defined. SD Participant 52
at ASL 20 has a fitted lower asymptote about 0.2655, above the lower target
about 0.1836, so that crossing and its JND are unavailable. Its valid ASL-40
JND is retained in the fit audit. Following the paired-ASL rule, both SD
values for P52 are omitted from the plotted pair, mean, deltas, and summary
bars; its position remains labeled N/A. Thus RT summarizes 19 complete
pairs (11 Control/8 ASD), and SD summarizes 18 (11 Control/7 ASD), using
5,760 choices from eligible SD participants. The original RT figures and
their numerical outputs remain unchanged.

The four summary bars use each ASL separately across eligible participants:
raw rows summarize ASL-specific JNDs, and delta rows summarize ASL-specific
signed deviations. For each group, delta means at ASL 20 and 40 are equal
and opposite, and their population SDs are equal. RT bars each use 11 Control
or 8 ASD participants; SD bars each use 11 Control or 7 ASD participants.
`group_asl_summary_statistics.csv` records all 16 displayed summaries.

## Reactive-only TIED SVI for pooled RT trials (20 October 2026)

This is a separate joint RT-and-choice model, using all **11,400 RT rows**:
Control 6,600 from 11 participants, ASD 4,800 from 8. Pool raw trials and
both simultaneous sessions within group, including errors and participants
whose descriptive JND is unavailable in other analyses. SD and CB are
excluded; there is no new RT cutoff or time shift in the input data.

There is **one parameter vector per group**, jointly fit across ASL 20/40
and all ten signed differences. Its five free parameters are `rate_lambda`,
`T_0`, `theta_E`, `w`, and scalar `t_E_aff`. This differs from fitting each
ASL separately: ASL dependence is constrained by the vanilla/IPL TIED
equations. Model brightness input is `Mean_dB` (approximately 10.892/15.089
visual dB); 20/40 cd/m² remain the nominal `ASL_cd` labels. The other input
is signed `Delta_dB = right - left`, and `Choice = 1/-1` means right/left.

The human adapter imports the existing vanilla density/CDF primitives from
`fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py`.
It evaluates the bound-specific density at `RT_s - t_E_aff`, with
`Z_E = (w - .5)*2*theta_E`, and divides it by the probability of reaching
**either** bound in the user-confirmed **0–3 s** retention window. This same
normalization is used for every prediction, including the 0–1.5 s display
zoom. Choice masses are not normalized separately. The scalar delay is an
effective nondecision offset in this reactive-only model. The fitted
baseline has no proactive process, lapse mixture, or extra condition-wise
parameters.

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u ASD_Human_Data/fit_human_reactive_tied_svi.py --group all
.venv/bin/python ASD_Human_Data/plot_human_reactive_tied_svi.py
.venv/bin/python ASD_Human_Data/verify_human_reactive_tied_svi.py
```

`--group {all,Control,ASD}`, `--seed`, `--min-steps`, `--max-steps`, and
`--output-dir` configure the fitter. Default output:
`reactive_tied_rt_svi/`. Completed posterior files are protected from
accidental reruns; choose a different output directory for a new run.

Priors use natural logs: `rate_lambda ~ LogNormal(log(.2),1.5)`,
`T_0 ~ LogNormal(log(.001),2)`, `theta_E ~ LogNormal(log(30),1.5)`,
`w ~ Uniform(.05,.95)`, and `t_E_aff ~ Uniform(0,min(group RT)-1e-6)`.
Initial values are `.2, .001, 30, .5, .05` in that parameter order. The
delay upper bounds are **0.111188799848944 s Control** and
**0.181610000094563 s ASD**, derived from the fastest retained trials.

Use CPU float64, `K_max=40`, full-batch NumPyro `AutoMultivariateNormal`,
one ELBO particle, seed 0 and one initialization per group. ClippedAdam
uses learning rate .0002 and clip 1; guide initialization scale is .1.
Check 1,000-step loss windows, with minimum 50,000/maximum 150,000 steps,
12-window patience and .001 minimum relative best-window improvement.
Also require the largest pairwise posterior-mean movement over the last
three checkpoints to be below .1 posterior SD for every parameter. The
same 4,096 guide draws are used at each checkpoint to reduce Monte Carlo
noise in this stability check. Restore the checkpoint with the smallest
finite mean window loss. Record stability of both terminal and restored
checkpoints, and flag instability rather than adding model flexibility.

Save 10,000 constrained draws, means/SD/central 95% intervals, physical
covariance/correlation matrices, guide parameters, best/latest optimizer
checkpoints, loss traces, convergence records, and corner plots. These
are **variational approximations**. Pooled-trial posterior uncertainty
does not represent participant-to-participant variability. A single
initialization cannot establish global optimality.

Comparison figures evaluate the model at the **arithmetic means of the
five physical parameters**, not at an average of posterior curves.
Psychometric predictions are conditional right-bound masses. RTD
mixtures combine actual signed-condition frequencies; 20 ms empirical
histograms and 1 ms model curves retain 0–3 s normalization. Full and
zoomed versions of the 2×2 psychometric/RTD and 2×5 absolute-difference
RTD comparisons are saved as PNG/PDF. Tables include joint likelihood,
psychometric errors and per-signed-condition empirical/model CDF errors.

The independent verifier decodes the original MAT without rewriting the
export, checks every selected source row, reconstructs all empirical
counts, and checks the JAX primitives against separate SciPy image-CDF
and image/eigenfunction-density calculations. It tests `K_max=40/80`,
finite gradients, density/CDF agreement, conditional mass, physical
posterior support, and frequency-weighted mixtures at initialization and
posterior parameter values. The 20 October result-book entry records the
actual convergence, fit quality, visual inspection and strict docs build.

### Continuation with confirmation after the actual best loss window

The subsequent stopping audit found that the original relative threshold
allowed patience to increase despite new lower window means. Both original
runs ended at their lowest observed window, without post-best confirmation.
The separate continuation preserves their data and posterior artifacts:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u ASD_Human_Data/continue_human_reactive_tied_svi.py
.venv/bin/python ASD_Human_Data/plot_human_reactive_tied_svi.py --output-dir ASD_Human_Data/reactive_tied_rt_svi_confirm25
.venv/bin/python ASD_Human_Data/compare_human_reactive_tied_continuation.py
.venv/bin/python ASD_Human_Data/verify_human_reactive_tied_svi.py --output-dir ASD_Human_Data/reactive_tied_rt_svi_confirm25
.venv/bin/python ASD_Human_Data/verify_human_reactive_tied_continuation.py
```

The continuation restores **Adam moments, optimizer iteration, RNG state,
and guide parameters** from each original latest checkpoint: Control
150,000 and ASD 127,000 updates. It uses the same five-parameter model,
priors, full-batch data, learning rate, clipping, guide, and likelihood.
Program initialization only reconstructs transforms; the fresh optimizer
state is discarded before continuing the saved state. There is no new
statistical initialization.

Every lower 1,000-step mean negative ELBO now resets patience, including
improvements smaller than 0.1%. Require **25 subsequent windows** without
a new best (at least 25,000 updates), plus terminal and restored-best
posterior-mean stability. Restore the lowest finite window checkpoint.
A safety cap of **1,000,000 total updates per group** leaves any unfinished
confirmation explicitly flagged. This is an observed-window stopping
criterion, not proof of a global optimum. `--confirmation-windows`,
`--max-total-steps`, `--source-dir`, `--output-dir`, and `--group` configure
the continuation; `--resume-existing` continues an interrupted output run.

Outputs are under `reactive_tied_rt_svi_confirm25/`. They include the full
original loss prefix, continued checkpoints/posteriors, fresh predictions
and fit-quality tables, and a `comparison/` folder with original-versus-new
figures, parameter changes, condition-wise CDF changes, and a 256-common-draw
ELBO comparison. The verifier replays every strict patience reset, checks
the exact loss prefix and trial identities, and confirms original artifact
hashes. The actual continuation outcome belongs in the 20 October result
book alongside the original results.

### Sensitivity refit with RT <= 1.5 s

Use **Condition = RT and RT_s <= 1.5**, with likelihood and predictions
conditional on **0--1.5 s**, rather than a display zoom of the original model.
This retains 6,531 Control trials (11 people) and 4,704 ASD trials (8 people),
removing 69/96 long responses. Keep correct/error choices, both ASLs, all signed
differences and all sessions. Counts now vary by stimulus cell; use actual
retained frequencies for RTD mixtures. The fastest RTs and data-derived shared
delay supports remain unchanged.

The separate `human_reactive_tied_window_utils.py` imports the original
bound-specific density/CDF primitives and uses the either-bound 0--1.5 s
retention denominator. The model has the same five shared parameters and
unchanged priors. One warm start per group uses the full guide from
`reactive_tied_rt_svi_confirm25/`; reset Adam, RNG (seed 0), losses and patience
for the new objective. Its initial covariance is the saved guide covariance.
Run at least 50k and at most 1M new updates, with strict 25-window confirmation
and terminal/restored mean stability. Group, seed, min/max steps, confirmation
windows, source/output roots, and interrupted-run resumption are configurable.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u ASD_Human_Data/fit_human_reactive_tied_window_svi.py
.venv/bin/python ASD_Human_Data/plot_human_reactive_tied_window_svi.py
.venv/bin/python ASD_Human_Data/compare_human_reactive_tied_window.py
.venv/bin/python ASD_Human_Data/verify_human_reactive_tied_window_svi.py
```

Output: `reactive_tied_rt_svi_rt_le_1p5s/`. The comparison recomputes the previous
confirmed fit's predictions and likelihood on **the same selected rows and
0--1.5 s denominator**. Do not compare full-data training losses directly with
the new fit's losses. Save excluded source rows, selection counts, posteriors,
strict convergence records, common-window metrics and PNG/PDFs. After visual
inspection, `publish_human_reactive_tied_window.py` adds the verified result
under a new asset root on 20 October, preserving earlier fits.

### RT-only likelihood and a 200 ms delay prior

The next analysis fits **RT distributions only**, retaining ASL and signed
Delta_dB as covariates while summing left/right first-passage densities.
Observed response side is absent from the fitter's data dictionary and does
not affect the likelihood or gradients. Five group-shared parameters remain.

To allow `t_E_aff ~ Uniform(0, .200 s)`, the user approved excluding the four
responses below 200 ms (3 Control, 1 ASD). Without that selection change,
positive likelihood still forces delays below the original 111/182 ms minima.
Fit **Condition = RT, 0.2 <= RT_s <= 1.5**, including all retained correct/error
choices: **6,528 Control and 4,703 ASD** trials from all 19 participants. Keep
the other four priors unchanged. Divide summed densities by the either-bound
mass between **0.2 and 1.5 s**, including the lower-CDF subtraction.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u ASD_Human_Data/fit_human_reactive_tied_rt_only_svi.py
.venv/bin/python ASD_Human_Data/plot_human_reactive_tied_rt_only_svi.py
.venv/bin/python ASD_Human_Data/verify_human_reactive_tied_rt_only_svi.py
```

Output: `reactive_tied_rt_only_svi_0p2_1p5s_delay200ms/`. CLI exposes group,
seed, min/max updates, confirmation windows, delay upper bound, retained RT
window, source/output roots, and interrupted-run resumption. Transfer the
previous 1.5-s joint-fit guide, mapping its delay center and local covariance
to the wider prior support; reset Adam/RNG/loss history. Preserve the original
physical delay center during this coordinate change. Use the same full-batch,
float64 SVI and strict 25-window/local-stability checks as the preceding run.

Plot pooled RTDs/CDFs and the 2x5 RTD layout; report parameter posteriors and
mass near the requested 200 ms cap. No psychometric score is optimized or
plotted. Compare old/new parameters using **marginal RT likelihoods on the same
new rows and 0.2--1.5 s window**; old joint training losses are not comparable.
The likelihood, four early rows, and delay prior all change in this run, so any
improvement cannot be attributed to ignoring choice alone. The independent
verifier checks source rows, label invariance, the wider support, normalization,
numerics, warm-start mapping, strict stopping, mixtures and figure bounds.
After visual inspection, `publish_human_reactive_tied_rt_only_svi.py` adds the
results under a separate asset root on 20 October.

### RT-only sensitivity fit with a 250 ms delay prior

The user approved increasing the delay support to `Uniform(0, .250 s)` and
retaining **Condition = RT, 0.25 <= RT_s <= 1.5**. Relative to the 200-ms-cap
RT-only fit, this removes four additional Control responses (Participant 36,
source rows 6418/6492/6987/6992; 226--243 ms). Retain **6,524 Control and 4,703
ASD** trials, with all 19 participants represented. The fastest retained RTs
are 257.898/277.840 ms, so both groups can explore the full new delay range.
These removals are a user-approved sensitivity selection, not newly detected
invalid trials. Summed left/right RT densities are conditional on **0.25--1.5
s**, including the lower-CDF subtraction. Observed choice is still ignored.
The five group-shared parameters and four other priors remain unchanged.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u ASD_Human_Data/fit_human_reactive_tied_rt_only_svi.py \
  --source-dir ASD_Human_Data/reactive_tied_rt_only_svi_0p2_1p5s_delay200ms \
  --output-dir ASD_Human_Data/reactive_tied_rt_only_svi_0p25_1p5s_delay250ms \
  --rt-lower .25 --rt-upper 1.5 --delay-upper .25
.venv/bin/python ASD_Human_Data/plot_human_reactive_tied_rt_only_delay250_svi.py
.venv/bin/python ASD_Human_Data/verify_human_reactive_tied_rt_only_delay250_svi.py
```

Use one warm start per group from the saved 200-ms RT-only guide, with the
existing delay center/Jacobian mapping and reset optimizer, RNG and loss
history. Keep the existing float64 full-batch SVI configuration and strict
25-window plus terminal/restored local-stability checks. Save 10,000
constrained variational draws per group and report any capped/unconfirmed run.

The separate plotting/verifying/publishing scripts preserve the earlier
200-ms sources and artifacts. Compare both saved parameter sets on the same
new trials and **0.25--1.5 s normalization**, using actual retained stimulus
counts. Evaluate curves at arithmetic physical posterior means. Plot pooled
RTDs/CDFs and a 2x5 RTD layout, with full and zoom views, posterior corners,
convergence traces, and delay-boundary summaries. Histograms use 20-ms bins
with a final 10-ms bin from 1.49 to 1.5 s. Independently reconstruct MATLAB
source identities, selection counts, likelihoods, CDFs, mixtures and bins;
check finite gradients, K=40/80, conditional mass, prior supports, warm-start
mapping and stopping records. After visual inspection,
`publish_human_reactive_tied_rt_only_delay250_svi.py` adds a separate section
and assets to the 20 October result page. This sensitivity test changes both
the delay cap and the lower retained RT; interpretation must retain that
distinction.

### RT-only 0--1.8 s refit with the requested 0--300 ms delay prior

Use **Condition = RT and 0 <= RT_s <= 1.8**, retaining **6,574 Control and
4,770 ASD** trials from all 19 participants. All four responses below 200 ms
are included. Only 56 responses above 1.8 s are excluded (26 Control, 30 ASD).
Pool raw trials across participants and sessions. Retain correct/error trials;
observed response side is marginalized. Normalize densities and predictions
over **0--1.8 s**. Keep the same five shared parameters and four other priors.

The actual fitted model uses **t_E_aff ~ Uniform(0, .300 s)**. Pure reactive
likelihood is zero for delays at or beyond the earliest retained group RT:
111.1898 ms Control / 181.6110 ms ASD. Do not remove fast trials to enlarge
that region. `human_reactive_tied_rt_only_support_utils.py` provides a separate
guide-only prototype: its AutoMultivariateNormal transform is confined to
the positive-likelihood interval. The SVI model still evaluates the requested
Uniform(0, .300) density, including its normalization. The prototype's prior
does not enter the fitted joint density. Verify the exact log-density offset
between model and prototype is log(causal_upper / .300), and verify delays
outside causal support have -infinite likelihood while remaining inside the
requested prior support.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u ASD_Human_Data/fit_human_reactive_tied_rt_only_delay300_svi.py
.venv/bin/python ASD_Human_Data/plot_human_reactive_tied_rt_only_delay300_svi.py
.venv/bin/python ASD_Human_Data/verify_human_reactive_tied_rt_only_delay300_svi.py
```

Output: `reactive_tied_rt_only_svi_0_1p8s_delay300ms/`. One warm start per group
uses the earlier confirmed joint fit `reactive_tied_rt_svi_rt_le_1p5s/`, which
retained the same earliest responses. Later 200/250-ms RT-only fits have delays
outside the new posterior's causal support. Preserve the physical delay center
and map local covariance to the guide interval; reset Adam/RNG/loss history for
the new objective. Use the existing float64 full-batch, one-particle SVI and
10,000 constrained variational draws per group.

Stopping follows the original plan: at least 50k updates, at least 25 subsequent
1,000-step windows after the lowest observed window mean, and terminal mean
stability below 0.1 posterior SD over three checkpoints, then restore the
selected minimum. Any lower window resets patience. Record movement at the
selected checkpoint as a diagnostic. The earlier 200/250-ms scripts imposed an
additional selected-checkpoint stability gate; that extra gate is removed only
for this new run. Save the correction, original configuration/source, and the
optimizer/RNG-preserving checkpoint continuation. Keep the 1M maximum and flag
any run that fails the required checks at its cap.

Show data and the model at arithmetic physical posterior means, with full and
zoomed pooled RTD/CDF comparisons, a 2x5 RTD layout, posterior means/intervals,
correlations, corners and convergence traces. Label requested prior and causal
limits separately. Histograms use 20-ms bins and curves a 1-ms grid. Independently
audit source identities, true prior and guide supports, normalization, finite
gradients, K=40/80 convergence, mixtures, histograms, posteriors and stopping
records. Publish through `publish_human_reactive_tied_rt_only_delay300_svi.py`
to a new section/assets on 20 October, preserving the prior analyses.

During the stopping-rule continuation, the interrupted original Control worker
continued writing its trace after the resumed fit had accepted step 81k and
restored step 40k. The leftover worker was terminated using its exact host PID;
its extra iterations and overwritten files were archived. The audit helper
`reactive_tied_rt_only_svi_0_1p8s_delay300ms/repair_control_checkpoints.py`
deterministically replays the original initialization/RNG through the accepted
stop, checks every loss and checkpoint mean, reproduces all 10k posterior draws,
and restores consistent best/latest checkpoints. It preserves the posterior
and guide files byte for byte. Require `checkpoint_repair.json` to pass before
the numerical verifier or publisher accepts this run. Archive this repair and
the discarded trace for provenance.

### Human TIED/DDM debugging on unchanged 0--1.8 s trials

`reactive_ddm_debug_0_1p8s_delay300ms/` contains direct-likelihood diagnostic
fits, not new SVI posteriors: the same five-parameter TIED model, a standard
23-parameter DDM (20 stimulus drifts plus shared Omega/start/delay), and a
42-parameter DDM (20 Gamma, 20 Omega, shared start/delay). Run RT-only and
joint-choice controls on exactly the existing 11,344 rows, keep all four RTs
below 200 ms, retain the requested 0--300 ms delay bounds, and normalize each
stimulus condition on 0--1.8 s. The actual causal delay support remains below
the minimum group RT. Multiple L-BFGS-B starts and float64 gradients test
optimization separately from the old variational approximation and priors.

Run `debug_human_reactive_ddm.py`, `debug_human_ddm_joint.py`, then
`debug_human_ddm_shared_scale.py` with `.venv/bin/python`. Analyze with
`analyze_human_reactive_ddm_debug.py` and independently verify with
`verify_human_reactive_ddm_debug.py`. All scripts live in `ASD_Human_Data/`.
The stable diagnostic Wiener kernels are in `human_plain_ddm_diagnostics_utils.py`;
check them against the paper/current primitives and independent SciPy before
interpreting a score. Publish verified six-panel-type PNG/PDF outputs with
`publish_human_reactive_ddm_debug.py` on the 21 October result page. The bundle
includes every start/trace, exact rows, parameters, predictions and baseline
posterior samples used for a joint-draw posterior-predictive check.

Plain DDM RT-only fits improve shape but imply nearly deterministic choices
at small differences; joint fits recover graded choices and broader RTDs.
This isolates a timing/choice and cross-stimulus-constraint issue, without
claiming global optimization, predictive superiority, or failure of all DDMs.
The original baseline and earlier analyses remain reference results.

## Condition-wise Gamma–Omega–delay SVI for human RT and choice

`fit_human_gamma_omega_svi.py` adapts the repository's big Gamma/Omega
workflow to a pure reactive human DDM. Each pooled group has 20 independent
Gamma values, 20 Omega values, 20 effective delays, and one shared fractional
starting bias `w`: **61 parameters per group**. Gamma is dimensionless and
Omega is in s⁻¹. There is no proactive race or `del_go`, and no symmetry is
imposed between signed differences.

The selection is `Condition == "RT"`, `Valid == 1`, and
`0 <= RT_s <= 1.8`. Control retains **6,574 trials / 11 participants**;
ASD retains **4,770 trials / 8 participants**. Both correct and incorrect
choices are retained, including all four RTs below 200 ms. The 56 RTs above
1.8 s are excluded. Pool raw trials across participants and sessions.
`ASL_cd` labels the 20/40 cd/m² levels; signed `Delta_dB` is right minus
left visual dB. `Choice == 1/-1` means right/left.

`human_gamma_omega_svi_utils.py` uses the existing stable Gamma/Omega
Wiener kernel and repository trapezoidal-prior helpers. The likelihood is
the **joint observed RT and choice density**, normalized by the either-bound
mass on 0–1.8 s. Each signed condition is normalized before forming any
displayed RTD mixture.

| Parameter | Hard support | Flat part of trapezoidal prior |
|---|---|---|
| Gamma, positive difference | −1 to 20 | 0 to 12 |
| Gamma, negative difference | −20 to 1 | −12 to 0 |
| Omega | 0.01–15 s⁻¹ | 0.1–12 s⁻¹ |
| Effective delay | 0–300 ms | 10–200 ms |
| Shared bias | 0–1 | 0.4–0.6 |

Gamma's sign is softly preferred, not enforced. Each guide's delay support
ends at `min(0.3, condition minimum RT - 1e-6)`; the actual target prior
remains the specified 0–300 ms trapezoid. The same-trial joint MLE in
`reactive_ddm_debug_0_1p8s_delay300ms/` supplies the initialization; its
shared delay is repeated into the 20 delay centers.

The fitter uses full-batch, float64 CPU NumPyro `AutoMultivariateNormal`,
one ELBO particle, seed 0, ClippedAdam learning rate 0.0002 / clip 1, and
guide initial scale 0.1. Minimum 50k and maximum 1M updates; every lower
1,000-update window resets patience. Stop only after 25 later windows and
posterior-mean ranges below 0.1 SD across the last three checkpoints for all
61 parameters. Restore the selected best checkpoint and save 10,000 draws.

Control and ASD can run as separate CPU processes, as requested for this
analysis. Run the first two commands in separate terminals; each group has
its own checkpoints. Never also leave an all-groups launcher running.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u ASD_Human_Data/fit_human_gamma_omega_svi.py --group Control --resume-existing
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u ASD_Human_Data/fit_human_gamma_omega_svi.py --group ASD --resume-existing
.venv/bin/python ASD_Human_Data/plot_human_gamma_omega_svi.py
.venv/bin/python ASD_Human_Data/verify_human_gamma_omega_svi.py
.venv/bin/python ASD_Human_Data/publish_human_gamma_omega_svi.py
```

The default output root is
`gamma_omega_rt_choice_svi_0_1p8s_delay300ms/`. CLI options expose group,
seed, minimum/maximum steps, confirmation windows, initialization root,
output root, preflight-only checks and resume. Use a separate output root
for changed settings. The publication command protects existing entries.

`plot_human_gamma_omega_svi.py` produces separate Control and ASD pages:
2×5 RTD panels (ASL rows, absolute-difference columns) plus a psychometric
panel. Analytic curves use arithmetic posterior means; empirical RTDs use
20 ms bins. Opposite signs are mixed using actual retained counts.
The parameter figure has Gamma, Omega and delay rows, ASL columns, and a
small shared-bias panel; Control uses blue circles and ASD red stars. Bars are equal-tailed
95% **variational credible intervals**, not participant variability.
Convergence/correlation diagnostics and all numerical calculation tables
are saved alongside the three main PNG/PDF pairs.

`verify_human_gamma_omega_svi.py` independently re-reads the original MAT
rows, reconstructs all 40 probabilities and 20 RTD mixtures, verifies all
122 posterior summaries and covariance/correlation matrices, compares
NumPy/SciPy first passage with the fitted JAX calculations, integrates
conditional unit mass, checks K=40 against K=80 and finite gradients, and
reconstructs the patience/stability rule. Earlier human artifacts are
hash-checked. Publication targets the **20 October 2026** result-book page
as explicitly requested, with an audit ZIP containing posterior samples,
checkpoints, calculation tables, initialization caches and source snapshots.

### Delay-prior sensitivity: 300 to 500 ms

The separate `*_gamma_omega_delay500_svi.py` sources preserve the completed
300 ms runs and their source hashes. Only the delay hard upper prior changes
to **500 ms**; its flat region remains **10–200 ms**. All retained trials,
61 physical parameters, other priors, original MLE initialization, seed,
optimizer and confirmation/stability rule are identical. Control and ASD
run in separate parallel CPU processes, each with fresh optimizer state.

The 300 ms audit finds upper 95% endpoints above 295 ms for 2 Control and
8 ASD conditions. No additional trials are removed. Each guide's delay
support is `min(0.5, condition minimum RT - 1e-6)`; with the same data,
the largest causal limits are 347.71 ms for Control and 391.00 ms for ASD.
The physical prior remains the requested 0–500 ms trapezoid.

`human_gamma_omega_delay500_svi_utils.py` holds the priors, causal guide
supports, 20-condition mapping, initialization and normalized likelihood.
`fit_human_gamma_omega_delay500_svi.py` runs the two SVI fits;
`plot_human_gamma_omega_delay500_svi.py` renders the model/data and parameter
figures. `compare_human_gamma_omega_delay_bounds.py` audits the 300/500 ms
change; `verify_human_gamma_omega_delay500_svi.py` independently reconstructs
the calculations. `publish_human_gamma_omega_delay500_svi.py` copies the
figures, sources, tables and audit bundle into the result book.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u ASD_Human_Data/fit_human_gamma_omega_delay500_svi.py --group Control --resume-existing
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u ASD_Human_Data/fit_human_gamma_omega_delay500_svi.py --group ASD --resume-existing
.venv/bin/python ASD_Human_Data/plot_human_gamma_omega_delay500_svi.py
.venv/bin/python ASD_Human_Data/compare_human_gamma_omega_delay_bounds.py
.venv/bin/python ASD_Human_Data/verify_human_gamma_omega_delay500_svi.py
.venv/bin/python ASD_Human_Data/publish_human_gamma_omega_delay500_svi.py
```

Run the first two commands in separate terminals, and the remaining commands
after both finish. Outputs use
`gamma_omega_rt_choice_svi_0_1p8s_delay500ms/`. The comparison script checks
identical trial/exclusion identities and physical initial values, exports
all 122 old/new parameter summaries, and compares delay posteriors and pooled
RTDs directly. The verifier independently reconstructs the old/new mixtures.
New figures and audit links are added to the 21 October 2026 result-book
page; the original 20 October entry and 300 ms artifacts are preserved.

### Latest published figures and display update

The 21 October page embeds these three 500 ms-prior figures from
`gamma_omega_rt_choice_svi_0_1p8s_delay500ms/`:

- `human_control_gamma_omega_rtd_psychometric.png` / `.pdf`: Control RTDs
  and psychometric, 6,574 retained trials from 11 participants.
- `human_asd_gamma_omega_rtd_psychometric.png` / `.pdf`: ASD with the same
  layout and axes, 4,770 retained trials from 8 participants.
- `human_gamma_omega_parameter_comparison.png` / `.pdf`: Gamma, Omega,
  condition delays and shared bias, with 95% variational credible intervals.

All three use simultaneous `Condition == "RT"`, `Valid == 1`, and
`0 <= RT_s <= 1.8`, retaining both choices. The two psychometric panels
show empirical choice proportions as dots without error bars. The fitted
probabilities at the measured stimuli remain joined as visual guides;
no additional interpolation or fit was added. The display update leaves
all fitted parameters, numerical tables and RTD drawing pixels unchanged.

For another display-only update, run
`.venv/bin/python ASD_Human_Data/plot_human_gamma_omega_delay500_svi.py --group-figures-only`.
Then use `publish_human_gamma_omega_delay500_svi.py --update-existing`
after checking the revised images and visual-inspection manifest. The
publisher replaces only its own result section, preserves any following
sections, and refreshes source snapshots and the audit ZIP.
`psychometric_display_revision/` keeps the previous displays, source
snapshots and the hash/pixel-comparison audit.

Git wrap-up includes the human analysis sources and published 20–21 October
result-book assets, including linked PDFs, CSV tables and audit ZIPs. Raw
MAT/trial exports and local optimizer/posterior files remain in their
analysis output folders under the repository's existing ignore policy;
the published audit bundles contain the corresponding fit evidence.
