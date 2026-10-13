# ASD Analysis Scripts

- `build_session_type_csvs.py` - Combines the four supplied ASD cohort CSVs into session-type 1, 2, and 23 CSVs using the documented next-trial repeat rule; preserves source columns and adds `filename`, `genotype`, and derived `trial_is_repeat`.
- `verify_session_type_csvs.py` - Independently reconstructs repeat status with the standard-library CSV parser and verifies every output row and preserved cell against its source.
- `clean_session_type_1_csv.py` - One-time cleanup: archives the previous prepared CSV and manifests, removes 807 flagged session-type-1 rows, and verifies retained cells/order and protected file hashes.
- `plot_session_type_1_rtds.py` - Plots session-type-1 RTDs by cohort CSV and animal, with 20 ms window-normalized histograms, an eight-page PDF, PNG pages, count/bin audits, and complete short-sound row locations.
- `plot_session_type_1_genotype_rtds.py` - Pools session-type-1 trials by filename + genotype (12 rows, two pages) and gene family + genotype (six rows, one page), and checks every pooled histogram bin against the existing animal histogram counts.
- `plot_session_type_1_psychometrics.py` - Empirical P(right) scatter plots by filename + genotype (12 panels) and genotype alone (three panels, both gene families combined), pooling correct/error choices without an RT restriction.
- `plot_session_type_1_family_psychometrics.py` - Additional six-panel P(right) scatter image with SHANK3 and CNTNAP2 separate, pooling cohorts within each family + genotype and checking counts against the existing psychometric audits.
- `plot_session_type_1_ild_pooled_rtds.py` - Three RTD images matching the psychometric layouts, pooling all ten signed ILDs within each ABL; 20 ms bins, 0–0.6 s display with 0–1 s normalization, and independent checks against existing animal histogram counts.
- `plot_session_type_1_slope_jnd.py` - Eight scatter images of fitted sigmoid k and Figure 1 JND: individual animals, filename + genotype, family + genotype, and genotype only; fits ten equally weighted ILD probabilities separately at each ABL.
- `verify_session_type_1_slope_jnd.py` - Independently reconstructs raw choice counts/memberships, checks numerical sigmoid crossings, and verifies saved PDF colors/marker counts and preserved existing artifacts.
- `explore_asd_dataset.py` - Filters ASD dataset by genotype/quality criteria and generates exploratory RTD/CDF/quantile plots.
- `explore_valid_trials.py` - Utility exploration script for valid-trial subsets in the ASD dataset.
- `asd_wt_all_animals_rtd_cdf.py` - Computes ASD-WT RTD histograms and CDF summaries across animals by ABL and |ILD|.
- `asd_wt_scaling_quantiles_not_RTD_for_paper_fig1.py` - Quantile-based ASD-WT scaling analysis (per-animal slopes, scaled/unscaled quantile figures).
- `asd_wt_rtd_scaling_for_paper.py` - ASD-WT adaptation of RTD scaling pipeline (aggregate RTD, QQ-derived scaling, rescaled RTDs).
- `asd_wt_rtd_scaling_for_paper_Q_scale_match.py` - ASD-WT RTD scaling variant aligned to quantile-style slope estimation, with debug comparisons.
- `asd_wt_rtd_scaling_for_paper_quantile_avg_match.py` - ASD-WT RTD scaling variant using per-animal quantile averaging to match quantile-analysis assumptions.

## Session-type CSVs

Run `.venv/bin/python asd_analysis/build_session_type_csvs.py` from the repository root,
then `.venv/bin/python asd_analysis/verify_session_type_csvs.py` to verify the outputs.
Outputs are `session_type_csvs/ASD_session_type_{1,2,23}.csv`, with a
`filter_manifest.json` containing source counts and sequential filter counts.

Each output pools SHANK3 cohorts 1 and 2 and CNTNAP2 cohorts 3 and 4, selecting:

- The output's `session_type` (1, 2, or 23).
- `training_level == 16`.
- The current trial is **not** the trial following a `repeated_trial=True` trial,
  except that a True flag on a CNP or fixation abort does not exclude the next trial.
- `abort_type == 'Fixation'` **or** `success` is 1 or -1.
- For session type 1 only, exclude `is_short_sound == 1` after deriving repeat eligibility on all original trials.

The ASD source fields are Boolean `repeated_trial`, text `abort_type`, and numeric
`success`. The requested `abort_event == 3` condition maps to fixation aborts.
A current row's raw `repeated_trial=True` flag does not itself exclude that row.
Repeat status is derived from the preceding distinct timestamped trial, following
the rule in `animal_structure_columns.txt:6` and confirmed by the user.

"N+1" means the next chronological trial within the same input cohort CSV,
`animal`, `source_file`, and `session`, ordered by numeric `trial_start`.
This handles rows that are out of order and trial-number restarts. Repeat status
is calculated on all source trials before training, outcome, or session-type
filters are applied. Duplicate records with the same session keys and trial-start
timestamp share repeat status; all source records remain eligible under the
requested filters, without deduplication of the written data.

`filename` identifies the cohort input, for example `SHANK3_cohort1.csv`.
The existing `source_file` continues to identify the original daily CSV.
`genotype` is matched by `animal` from semicolon-delimited `metadata.csv`, with
values `wt`, `hom`, and `het`. `trial_is_repeat` records the derived current-trial
repeat status and is False for every retained row; the raw `repeated_trial`
column remains unchanged and may be True.
All source columns are preserved; columns absent from a cohort are blank in its
output rows. Apart from the session-type-1 short-sound exclusion, no further duration, session-number, genotype, RT-quality, or
missing-timing filters are applied, and rows are not deduplicated.

`filter_manifest.json` records counts after each selection step, retained outcomes
and genotypes by source, and the grouping/ordering rules. A successful independent
verification writes `session_type_csvs/verification.json`; rebuilding clears the
previous verification so it cannot be mistaken for a check of the new files.

The 2026-10-12 cleanup and independent verification produced:

| Output | Rows | Fixation aborts |
| --- | ---: | ---: |
| `ASD_session_type_1.csv` | 642,370 | 146,059 |
| `ASD_session_type_2.csv` | 547,720 | 139,413 |
| `ASD_session_type_23.csv` | 685,063 | 190,209 |
| Total | 1,875,153 | 475,681 |

All 1,875,153 rows and 109 output columns passed the independent source-value,
repeat-rule, provenance, genotype, count, and order checks. Fixation aborts remain
eligible when their own raw flag is True; an abort can still be removed if it is
the actual repeated trial following a different, repeat-triggering trial.

The prepared session-type-1 CSV was shortened from 643,177 to 642,370 rows by
removing 807 `session_type == 1 AND is_short_sound == 1` records (330 ASD0026,
477 ASD0045). All 53 animals remain. The source cohorts and prepared session
types 2/23 are unchanged. Retained cells, columns, and order were preserved.
The former CSV, preparation manifest, verification, and removed records are in
`session_type_csvs/archive/pre_short_sound_cleanup/`; `short_sound_cleanup.json`
records before/after hashes and verification. The cleanup script deliberately
does not overwrite an existing archive; future rebuilds use the generator's rule.

## Session-type-1 RTDs

Run `.venv/bin/python asd_analysis/plot_session_type_1_rtds.py` from the repository
root. Parameters are near the top of the script; outputs are in
`asd_analysis/session_type_1_rtds/`. Input CSVs remain unchanged.

The prepared CSV excludes rows with `session_type == 1 AND is_short_sound == 1`.
The requested `is_short_duration` flag is named `is_short_sound` in the CSVs;
its flagged rows exactly match positive `short_duration` records. This removes
807 rows during preparation (763 in the displayed stimulus domain), and retains ASD0026
and ASD0045 with their remaining standard-sound trials. All 53 animals are
included. Plotting applies zero additional short-sound exclusions and leaves its
input files unchanged. Plotting maps ABL 59 to 60 and selects ABL
20/40/60 and |ILD| 1/2/4/8/16. Every filename+animal row has five correct/error
panels (`success` in {1, -1}, finite recorded `timed_rt` in [0, 1] seconds) and
one pooled correct/error-plus-fixation panel (finite `timed_rt` in [-1, 1]
seconds, restricted to the same displayed stimuli). All genotypes are included.
Signed ILDs and sessions are pooled by their observed trials; the pooled panel
also combines conditions directly by trial.

Each curve uses 20 ms bins and density `bin_count / (window_trials * 0.020)`;
every nonempty curve has area 1 within its original normalization window. The
display is zoomed to [0, 0.6] seconds in the first five panels and [-0.6, 0.6]
seconds in the pooled panel. The original [0, 1] / [-1, 1] second windows still
determine the normalization and printed sample counts; the zoomed curves are
not renormalized. Colors are
blue/orange/green for ABL 20/40/60 and black for the pooled panel. The first five
panels share their y limits within each animal row.

The outputs cover 53 filename+animal groups and 610,234 selected rows, with
466,978 correct/error trials in the first-five-panel windows and 598,864 trials
in the pooled windows. `curve_counts.csv`, `histogram_bins.csv`,
`animal_counts.csv`, `pages.csv`, and `validation.json` record the counts,
densities, layout, and provenance. `flagged_row_locations.csv` contains all 836
flagged original records and all 807 flagged archived prepared records, including rows
outside the plotted conditions. CSV row numbers count the header as row 1;
flagged rows are interleaved, so their first-to-last spans are not exclusion lists.
Historical prepared row numbers point to the archived CSV, not the cleaned file.
Regenerated RTD counts, densities, windows, and all 11 PNG pages are identical
to the preceding plots; input hashes and exclusion provenance were refreshed.
The figures and audits are documented in `docs/results/2026-10-12.md`.

## Session-type-1 RTDs pooled by genotype

Run `.venv/bin/python asd_analysis/plot_session_type_1_genotype_rtds.py` from the
repository root. Outputs are in `asd_analysis/session_type_1_genotype_rtds/`.
It reads the cleaned prepared input, retains the same filters, verifies zero
remaining short-sound rows, and includes ASD0026 and ASD0045
with their remaining trials. It reads the updated animal audits for independent
checks and preserves those figures and all input CSVs during the pooled run.

The filename + genotype PDF has two six-row pages: SHANK3 cohorts 1/2, then
CNTNAP2 cohorts 3/4. The family + genotype PDF has six rows on one page,
combining the two cohorts within each family. Each group uses WT, heterozygous,
homozygous row order, and labels the contributing animal count.

Raw trials are pooled before histogramming, so animals with more eligible
trials contribute more weight. The six columns, stimulus selection, colors,
20 ms bins, original [0, 1] / [-1, 1] second normalization windows, and
[0, 0.6] / [-0.6, 0.6] second display limits match the animal analysis.
Each view covers 53 animals and 610,234 selected rows, with 466,978 correct/error
window trials and 598,864 pooled-with-fixation window trials.

The `genotype_rtd_*.csv` audits record group membership, counts, histogram bins,
and page assignments. The validation manifests check every pooled bin and
eligibility/outcome count against sums of the existing animal audits, and verify
that all input CSVs and existing animal artifacts remain unchanged. The new PDFs,
three matching PNG pages, plotting source, and audits are added to the same
2026-10-12 result-book entry.

## Session-type-1 psychometric scatter plots

Run `.venv/bin/python asd_analysis/plot_session_type_1_psychometrics.py`.
Outputs are in `asd_analysis/session_type_1_psychometrics/`: two single-page
PDFs, matching PNGs, `psychometric_points.csv`, group counts/membership,
and validation manifests.

Use the cleaned prepared input, training level 16, and confirmed repeat rule.
Retain `success in {1, -1}` without any RT restriction, map ABL 59 to 60,
and select ABL 20/40/60 and signed ILD ±1/2/4/8/16. Every point is right
choices (`response_poke == 1`) divided by all recorded right/left choices at
that group's ABL and signed ILD. All 471,416 trials have a known response side;
241,539 are right and 229,877 left. The 2,188 trials with missing RT remain
eligible because these plots depend on choices rather than timing.

The filename + genotype figure has four cohort rows and three genotype columns
(WT, heterozygous, homozygous). The genotype-only figure has three panels,
combining SHANK3 and CNTNAP2 across all cohorts, with 16 WT, 18 heterozygous,
and 19 homozygous animals. Trials are pooled directly; animals with more eligible
trials contribute more weight. All three ABLs share each panel as blue/orange/green
scatter series, without fitted curves, connecting lines, or error bars.
The point audit includes right/left counts and denominators. Both views and
PDF downloads are documented in the 2026-10-12 result book.

For the additional separate-family view, run
`.venv/bin/python asd_analysis/plot_session_type_1_family_psychometrics.py`.
The six panels place SHANK3 above CNTNAP2, with WT, heterozygous, and homozygous
columns. Pool raw trials across cohorts within each family and genotype using
the same selection, response-side denominator, and ABL colors above. This view
uses the same 53 animals and 471,416 choices; all 180 points match sums of the
independently verified cohort counts, and their counts sum to the existing
combined-genotype view. The two existing figures remain available unchanged.
The additional PNG, single-page PDF, `family_psychometric_*.csv` count/membership
audits, and validation manifests share the existing output directory. All three
images and their downloads are included in the 2026-10-12 result book.

## Session-type-1 RTDs with all ILDs pooled

Run `.venv/bin/python asd_analysis/plot_session_type_1_ild_pooled_rtds.py`.
Outputs are in `asd_analysis/session_type_1_ild_pooled_rtds/`: three PNG images,
matching single-page PDFs, histogram/count/membership audits, and validation
manifests. Layouts match the psychometrics: 12 filename + genotype panels,
three genotype-only panels combining both gene families, and six family +
genotype panels with SHANK3 above CNTNAP2. Genotype columns are WT, heterozygous,
and homozygous.

Use the cleaned prepared session-type-1 CSV, training level 16, and confirmed
next-trial repeat rule. All 53 animals remain; the 807 short-sound rows were
removed during preparation, with zero additional animal/short-sound exclusions.
Select correct/error trials (`success in {1, -1}`), ABL 20/40/60 after mapping
59→60, and signed ILDs ±1/2/4/8/16. Unlike the psychometrics, these RTDs require
finite recorded `timed_rt` in inclusive [0,1] s: 466,978 of the 471,416 eligible
choices enter each view, omitting 2,188 nonfinite RTs and 2,250 outside the window.

Pool raw trials across all ten signed ILDs and sessions within each ABL/group;
animals and conditions contribute their observed window trials. Solid, unfilled
blue/orange/green steps show ABL 20/40/60. Each of the 63 curves uses 20 ms bins
and `bin_count / (curve_window_n × 0.020 s)`, with unit area over [0,1] s. The
display is zoomed to 0–0.6 s without renormalizing. Legends still show each
curve's 0–1 s normalization count;
titles show contributing animals and total RTs. All panels and views share the
y scale. All 3,150 histogram bins match summed existing animal/|ILD| counts,
and coarser groupings match sums of the filename + genotype counts. Existing
figures and input data are preserved. The new figures are embedded directly
as images in the 2026-10-12 result book.

## Session-type-1 sigmoid k and Figure 1 JND

Run `.venv/bin/python asd_analysis/plot_session_type_1_slope_jnd.py`, then
`.venv/bin/python asd_analysis/verify_session_type_1_slope_jnd.py`.
Outputs are in `asd_analysis/session_type_1_slope_jnd/`: eight PNGs and matching
single-page PDFs, condition/group/membership audits, fitted parameters,
JND thresholds and crossings, fit-status counts, and validation manifests.
The figures are embedded in `docs/results/2026-10-13.md`.

The cleaned prepared CSV and selection match the existing psychometrics:
session type 1, training level 16, the confirmed repeat rule, correct/error
choices (`success in {1, -1}`), ABL 59→60, ABL 20/40/60, and signed ILDs
±1/2/4/8/16, with **no RT restriction**. All 53 animals remain, including
ASD0026/ASD0045; 807 short-sound records were removed during preparation.
Each view contains 471,416 choices (241,539 right; 229,877 left), including
2,188 choices with nonfinite RT. P(right) is `response_poke == 1` divided
by all recorded right/left choices at the group's ABL and signed ILD.

Fit `lower + (upper-lower)/(1+exp(-k*(ILD-x0)))`, with parameter order
`[upper, lower, x0, k]`, using unweighted `curve_fit` on all ten ILD
probability points. Match Figure 1: initial guesses
`[max(Pright), min(Pright), median(ILD), 0.1]`, upper/lower bounded 0–1,
x0 unrestricted, k nonnegative, and `maxfev=10000`. Pool raw choice counts
at each condition before fitting pooled groups. More trials contribute more
weight to each condition's probability, while each of the ten probabilities
gets equal weight in the fit. Do not average animal probabilities or parameters.

Slope is exactly **k**, in dB⁻¹. The same fit supplies Figure 1 JND in dB:
`y_high=0.75*upper`, `y_low=0.25*(1-lower)`, then
`JND=(inverse_sigmoid(y_high)-inverse_sigmoid(y_low))/2`.
Require an increasing fit, both thresholds strictly within the asymptotes,
and finite positive crossing spacing. Keep valid k if JND alone is undefined;
store unavailable JND as a blank value with its reason. Each individually
available crossing is retained in the parameter audit. With zero lapses,
this formula reduces to `ln(3)/k`.

The four views contain 53/12/6/3 groups and 159/36/18/9 fits, totaling 222
fits and 2,220 condition points. Each metric has one image per view. The animal
image uses one wide axis sorted by cohort, WT/heterozygous/homozygous, then ID,
with three offset ABL points per animal. Each pooled image uses one axis with
12 filename + genotype, six family + genotype, or three genotype positions.
Each position has three offset ABL dots, a category label rotated 45°, and
contributing animal/choice counts. Category spacing and label rotation are
editable near the top of the script. ABL 20/40/60 use blue/orange/green
scatter markers, without connecting curves, error bars, or mean markers.
Each image uses a single y-scale and retains all valid estimates. This layout
revision preserves the original fitted parameters and numerical audits.

All 222 k estimates are available. Individual JND has 155/159 available
estimates; the low threshold lies below the lower asymptote for ASD0035 at
ABL 20, ASD0044 at ABL 40, and ASD0048 at ABL 40/60. All 63 pooled JNDs
are available. Counts match existing psychometric audits; independent checks
reconstruct all conditions, memberships, residuals, 440 available crossings,
and 218 JNDs. Saved PDF marker counts/colors and PNG dimensions are checked.
Original/prepared CSVs and all existing October 12 analysis/figure artifacts
remain unchanged by SHA-256.

## Versioned artifacts and local data

The ASD analysis scripts, compact count/membership/histogram/fit audits, and
JSON preparation/validation manifests are versioned with the result-book pages.
Published PNGs, PDF downloads, source-script copies, and downloadable audits
are in `docs/assets/results/2026-10-12/` and `2026-10-13/`. These published
downloads and the compact analysis CSV audits are explicitly staged despite
the repository's general CSV/PDF ignore rules.

The original cohort CSVs, `metadata.csv`, the source ZIP, the three large
prepared session-type CSVs, and the archived pre-cleanup CSV remain local.
They are not included in this analysis commit; preparation manifests record
their provenance and hashes. Native PNG/PDF exports in the analysis output
directories also remain local under the existing ignore rules; their published
copies are included in the result book.
