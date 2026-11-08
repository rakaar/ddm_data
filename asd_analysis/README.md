# ASD Analysis Scripts

- `build_session_type_csvs.py` - Combines the four supplied ASD cohort CSVs into session-type 1, 2, and 23 CSVs using the documented next-trial repeat rule; preserves source columns and adds `filename`, `genotype`, and derived `trial_is_repeat`.
- `verify_session_type_csvs.py` - Independently reconstructs repeat status with the standard-library CSV parser and verifies every output row and preserved cell against its source.
- `clean_session_type_1_csv.py` - One-time cleanup: archives the previous prepared CSV and manifests, removes 807 flagged session-type-1 rows, and verifies retained cells/order and protected file hashes.
- `remove_cntnap2_cohort4_precalibration_sessions.py` - One-time speaker-calibration cleanup: archives the source cohort 4 CSV, prepared session-type-1 CSV, and manifests; removes numeric sessions below 13 from those two CSVs while preserving every retained byte; compares a separate fresh preparation build and runs the independent source verifier.
- `verify_cntnap2_cohort4_session13_cleanup.py` - Independently checks both active CSVs against their archived originals, every removed-row location, unchanged animals, and hashes/sizes/mtimes of the other cohorts and prepared session types 2/23.
- `plot_session_type_1_rtds.py` - Plots session-type-1 RTDs by cohort CSV and animal, with 20 ms window-normalized histograms, an eight-page PDF, PNG pages, count/bin audits, and complete short-sound row locations.
- `plot_session_type_1_genotype_rtds.py` - Pools session-type-1 trials by filename + genotype (12 rows, two pages) and gene family + genotype (six rows, one page), and checks every pooled histogram bin against the existing animal histogram counts.
- `plot_session_type_1_psychometrics.py` - Empirical P(right) scatter plots by filename + genotype (12 panels) and genotype alone (three panels, both gene families combined), pooling correct/error choices without an RT restriction.
- `plot_session_type_1_family_psychometrics.py` - Additional six-panel P(right) scatter image with SHANK3 and CNTNAP2 separate, pooling cohorts within each family + genotype and checking counts against the existing psychometric audits.
- `plot_session_type_1_ild_pooled_rtds.py` - Three RTD images matching the psychometric layouts, pooling all ten signed ILDs within each ABL; 20 ms bins, 0–0.6 s display with 0–1 s normalization, and independent checks against existing animal histogram counts.
- `plot_session_type_1_genotype_overlaid_rtds.py` - Current cleaned-data 1×2 PNG: SHANK3 and CNTNAP2 separate, each overlaying WT (blue), HOM (green), HET (orange), pooling ABL 20/40/60 and signed ILDs ±1/2/4/8/16. Uses valid choices with finite timed_rt in inclusive [0,1] s; 20 ms bins, full 0–1 s display, area 1 per curve. The 446,949 RTs, six memberships/counts and all 300 bins are independently checked against source CSV parsing and pooled current CDF histograms; outputs are in `session_type_1_genotype_overlaid_rtds/`, directly embedded in the October 19 result book.
- `plot_session_type_1_slope_jnd.py` - Eight scatter images of fitted sigmoid k and Figure 1 JND: individual animals, filename + genotype, family + genotype, and genotype only; fits ten equally weighted ILD probabilities separately at each ABL.
- `verify_session_type_1_slope_jnd.py` - Independently reconstructs raw choice counts/memberships, checks numerical sigmoid crossings, and verifies saved PDF colors/marker counts and preserved existing artifacts.
- `plot_session_type_1_low_delta_psychometrics.py` - Two-by-five psychometric comparison: the five smallest mean absolute pairwise JND differences across ABL above the five smallest k differences. Both rankings use the same 50 JND-eligible animals from the 53-animal sample. Reuses saved probabilities/fits with no RT restriction; PNG/PDF, 53-animal eligibility/metrics, complete rankings, panel counts, and validation are in `session_type_1_low_delta_psychometrics/`. Published in the October 15 result book; previous figures and CSVs are preserved.
- `verify_session_type_1_low_delta_psychometrics.py` - Independently checks all pairwise differences, the two-thirds-range identity, common eligibility, rankings, 300 measured points, 30 reused fits and smooth curves, saved PDF colors/marker/curve counts, and protected artifact hashes.
- `plot_session_type_1_animal_metric_deviations.py` - Two Figure 1-style PNG/PDF pairs: JND and signed JND-minus-animal-mean deviations for 50 complete-JND animals, and fitted k with signed k deviations for all 53 animals. Reuses unfiltered-by-RT saved fits, orders animals by their equal-ABL means, and adds gray mean ±1 population-SD bars (`ddof=0`). Eligibility, 103 means, 309 values/deviations, summaries, and plot geometry are in `session_type_1_animal_metric_deviations/`; published in the October 15 result book.
- `verify_session_type_1_animal_metric_deviations.py` - Independently parses saved source fits, checks metric-specific eligibility, means, signed deltas, sorting, sample totals, `statistics.pstdev` summaries, all 618 saved-PDF data dots, four SD-bar endpoints, and preserved earlier artifacts.
- `plot_session_type_1_animal_metric_deviations_calibrated.py` - Updated Figure 1-style JND/signed-ΔJND and k/signed-Δk PNG/PDF pairs after CNTNAP2 cohort 4 sessions below 13 were removed. Refits all 159 animal/ABL psychometrics from 451,381 valid choices without an RT restriction; includes 49 complete-JND animals and all 53 complete-slope animals. Outputs, eligibility, fits, probabilities, and common-population before/after comparisons are in `session_type_1_animal_metric_deviations_calibrated/`; directly embedded in the October 17 result book. Historical fits and figures are preserved.
- `verify_session_type_1_animal_metric_deviations_calibrated.py` - Independently reconstructs current and archived choices, checks all 159 refits and 308 numerical JND crossings, confirms 129 unchanged fits for the 43 unaffected animals, and checks means, signed deltas, eight before/after summary rows, all 612 PDF data dots, four SD bars, and protected hashes.
- `plot_session_type_1_genotype_overlay_psychometrics.py` - Six WT/HET/HOM comparison figures with measured scatter points and four-parameter sigmoid fits, using trial pooling or equal averages of measured ABL probabilities.
- `verify_session_type_1_genotype_overlay_psychometrics.py` - Independently checks raw choice counts, memberships, both ABL combination formulas, unweighted fits, and saved PDF marker/curve colors and counts.
- `plot_session_type_1_genotype_overlay_psychometrics_rt_0_to_1.py` - Three additional trial-pooled genotype overlays restricted to finite measured `timed_rt` in inclusive [0,1] seconds, retaining all 53 animals and preserving the six unrestricted figures.
- `verify_session_type_1_genotype_overlay_psychometrics_rt_0_to_1.py` - Independently checks the measured-RT selection, pooled probabilities, memberships, fits, saved PDF markers/curves, and preservation of prior exports.
- `plot_animal_psychometric_and_rtd.py` - Two-panel single-animal figure (default ASD0054): three-ABL correct/error psychometrics and signed-ILD-pooled RTDs, using every finite RT with 20 ms bins and a 0–0.6 s display zoom. Parameters are at the top; outputs are in `session_type_1_animal_psychometric_rtd/<animal>/`.
- `plot_all_animals_psychometric_and_rtd.py` - Extends the same two-panel layout to all 53 animals, five animals per page (11 PDF/PNG pages), reusing verified animal sigmoid fits. RTDs normalize over all finite RTs, including values above 1 s; outputs and page/count/bin audits are in `session_type_1_all_animals_psychometric_rtd/`.
- `verify_all_animals_psychometric_and_rtd.py` - Independently parses the prepared CSV, checks every animal/ABL/ILD choice count and all-RT histogram, and verifies the 53-animal PDF's pagination, bookmarks, colors, scatter points, and fitted curves.
- `plot_all_animals_psychometric_rtd_chronometric.py` - Three-column version of the 53-animal PDF: unchanged psychometrics/RTDs plus mean RT ± SEM versus |ILD| after pooling +ILD and -ILD raw valid trials. Uses all finite RTs, five animals per page, 11 PDF/PNG pages with compact square chronometric panels at 28% of the other column widths on 18.24-inch pages, and preserves the two-column exports; outputs are in `session_type_1_all_animals_psychometric_rtd_chronometric/`.
- `verify_all_animals_psychometric_rtd_chronometric.py` - Independently verifies all 795 signed-ILD-pooled means/SEMs, count/weighted-mean consistency, unchanged psychometric/RTD audits, and every chronometric dot and SEM error bar in the saved PDF.
- `plot_all_animals_psychometric_rtd_chronometric_scaling.py` - Extended 53-animal PDF with two rows per animal: the saved psychometric/RTD/chronometric panels above CDFs, ΔCDF/onsets, three ABL quantile panels, and their scaled overlay. The new row uses 466,978 measured RTs in [0,1] s, 1 ms CDF bins, 265 freshly detected per-animal/|ILD| cutoffs, and 530 Figure 1 rescaling fits. Five animals per page, 11 taller pages; previous exports remain unchanged. Outputs and full count/bin/quantile/scaling audits are in `session_type_1_all_animals_psychometric_rtd_chronometric_scaling/`.
- `verify_all_animals_psychometric_rtd_chronometric_scaling.py` - Independently parses raw CSV trials, checks signed pooling, bisect-based histogram counts, CDF differences/onsets, order-statistic quantiles, scalar rescaling fits, synthetic missing/invalid cases, original means/SEMs, PDF cutoff/error-bar geometry, pagination/bookmarks, and preserved artifact hashes.
- `plot_family_genotype_psychometric_rtd_chronometric_scaling.py` - New two-page PDF and matching PNGs using the calibration-cleaned CSV: SHANK3 on page 1, CNTNAP2 on page 2, each with WT/HET/HOM blocks. Pools raw trials across cohorts/animals within family/genotype before recomputing psychometrics, full-range RTDs, mean RT ± pooled-trial SEM, CDF cutoffs, quantiles, and Figure 1 scaling. Published in the October 19 result book; outputs/audits are in `session_type_1_family_genotype_psychometric_rtd_chronometric_scaling/`.
- `verify_family_genotype_psychometric_rtd_chronometric_scaling.py` - Independently checks raw memberships/counts, 18 refits, 180 probabilities, full-range RTD bins/unit areas, 90 means/SEMs, 90,000 CDF bins, 30 cutoffs, 60 scaling fits, all quantiles/transforms, missing/invalid cases, PDF points/SEMs/cutoffs, two pages/six bookmarks, and protected file hashes.
- `plot_session_type_1_animal_rt_summary.py` - Three plots with all 53 animals on one categorical axis: mean RT ± sample SD, mean RT ± SEM, and median RT with a binomial/order-statistic 95% CI. Pools all selected signed ILDs and uses every finite valid-trial RT, including >1 s; statistics, membership, plot-point audits, and exports are in `session_type_1_animal_rt_summary/`.
- `verify_session_type_1_animal_rt_summary.py` - Independently checks RT selection and Python sample statistics, median CI endpoints/coverage, all 477 plotted estimates and error bars, and saved PDF identities/colors/geometry.
- `plot_session_type_1_accuracy_vs_mean_rt.py` - One scatter dot per animal/ABL (159 dots), plotting valid-choice accuracy against all-finite-RT arithmetic mean. ABL determines color; SHANK3 is hollow and CNTNAP2 filled; WT/HET/HOM are circles/squares/triangles. Pools signed ILDs and sessions; uses all valid choices for accuracy and all finite RTs for the RT mean, with both denominators in `session_type_1_accuracy_vs_mean_rt/animal_abl_points.csv`.
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
- For `CNTNAP2_cohort4.csv` only, exclude numeric `session < 13` because the speaker calibration was incorrect before session 13. Apply at every training level and session type; session 13 is retained. The source cohort CSV itself has also been shortened by this rule.

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
output rows. Apart from the session-type-1 short-sound exclusion and the CNTNAP2
cohort 4 calibration cutoff, no further duration, session-number, genotype, RT-quality, or
missing-timing filters are applied, and rows are not deduplicated.

`filter_manifest.json` records counts after each selection step, retained outcomes
and genotypes by source, and the grouping/ordering rules. A successful independent
verification writes `session_type_csvs/verification.json`; rebuilding clears the
previous verification so it cannot be mistaken for a check of the new files.

The historical 2026-10-12 cleanup and independent verification produced:

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
477 ASD0045). All 53 animals remained. At that cleanup, the source cohorts and
prepared session types 2/23 were unchanged. Retained cells, columns, and order were preserved.
The former CSV, preparation manifest, verification, and removed records are in
`session_type_csvs/archive/pre_short_sound_cleanup/`; `short_sound_cleanup.json`
records before/after hashes and verification. The cleanup script deliberately
does not overwrite an existing archive; future rebuilds use the generator's rule.

### Current CSVs after the 2026-10-17 speaker-calibration cleanup

The second CNTNAP2 cohort is stored as `CNTNAP2_cohort4.csv`. The user reported
incorrect speaker calibration before session 13 and requested removal from both
that source CSV and the prepared CSV used for analysis. This uses the numeric
`session` column, not `session_type`, and removes whole sessions 1–12. It does
not exclude whole animals or impose any RT restriction.

| Active CSV | Before | Removed | Retained | Animals retained |
| --- | ---: | ---: | ---: | ---: |
| `CNTNAP2_cohort4.csv` | 618,712 | 96,082 | 522,630 | 20 |
| `session_type_csvs/ASD_session_type_1.csv` | 642,370 | 31,767 | 610,603 | 53 |

The source removal affects 20 animals; the prepared removal affects 10 animals
(2 WT, 6 HET, 2 HOM). Every affected animal retains later sessions. Cohort 4
still contributes 17 animals to the 53-animal prepared analysis. The other
cohort CSVs and prepared session types 2/23 are byte-for-byte unchanged.

Current prepared output counts are **610,603 / 547,720 / 685,063** for session
types **1 / 2 / 23**, totaling **1,843,386 rows**. A fresh preparation build
matches all retained cells and row order, and the independent verifier checks
repeat eligibility on the surviving complete source sessions before selection.
Removing whole sessions does not change next-trial eligibility in retained sessions.
The generator and verifier also apply the cutoff explicitly, so restoring an
older source file cannot silently reintroduce these sessions into prepared data.

Complete pre-change CSVs, old manifests, exact removed rows, and a compressed
row-location audit are under
`session_type_csvs/archive/pre_cntnap2_cohort4_session13_cleanup/`.
`session_type_csvs/speaker_calibration_cleanup.json` and
`speaker_calibration_independent_verification.json` contain before/after hashes,
per-animal removals, protected-file checks, and independent verification.

**The figures, fits, and detailed analysis counts below and in the October
12–15 result-book entries remain historical results from before this calibration
exclusion.** Their previous prepared CSV is the
archived 642,370-row version. Earlier row numbers for either shortened CSV refer
to its archived copy. Existing fit-dependent scripts retain their original
validation expectations; the old fits must be refreshed before presenting them
as analyses of the current CSV.

The **October 17 Figure 1-style animal JND/signed-ΔJND and k/signed-Δk plots**
have now been recomputed in `session_type_1_animal_metric_deviations_calibrated/`
using the current CSV and the same unweighted four-parameter psychometric fits.
They use 451,381 valid choices (231,263 right; 220,118 left), with no RT restriction.
All 53 animals retain three k values. JND eligibility is now 49 animals rather
than 50: ASD0058's ABL-20 lower target falls below the fitted lower asymptote;
ASD0035/0044/0048 remain ineligible. The other historical figure sets and fit
tables have not been overwritten or refreshed.

The before/after audit includes a fixed population for each metric. Among the
same 49 animals, mean JND changes 3.188877 → 3.188179 dB and the SD of signed
ABL deviations changes 0.363051 → 0.383571 dB (+5.65%). Among the same 53
animals, mean k changes 0.365666 → 0.366198 dB⁻¹ and its deviation SD changes
0.038319 → 0.039098 dB⁻¹ (+2.03%). The 43 unaffected animals' 129 fits are
numerically unchanged; some of the 10 affected animals change more.

## Family/genotype PDF with CDFs and quantile scaling — 2026-10-19

Run `.venv/bin/python asd_analysis/plot_family_genotype_psychometric_rtd_chronometric_scaling.py`,
then `.venv/bin/python asd_analysis/verify_family_genotype_psychometric_rtd_chronometric_scaling.py`.
The new two-page PDF is the genotype-pooled counterpart of the original
53-animal extended PDF. SHANK3 and CNTNAP2 are separate, with WT/HET/HOM
on each family's page; cohorts and animals are pooled by observed trials.
The page width is 18.24 inches, height 22 inches, with the compact square
chronometric geometry and two rows/nine panels per genotype. Six bookmarks
identify the six family/genotype groups. The October 19 result book embeds
both PNG pages directly and provides the PDF, sources, and complete audits.

Use the current cleaned CSV: 451,381 valid choices (231,263 right; 220,118 left),
449,193 finite RTs, and 446,949 RTs in inclusive [0,1] seconds. Include all
53 animals irrespective of JND availability. The psychometric uses all valid
choices with no RT restriction; RTD/chronometric use all finite RTs, including
2,244 above 1 second. The lower CDF/quantile row alone uses [0,1] seconds.
The confirmed repeat rule is already applied and is not derived again.

Raw choice counts are pooled before the 18 unweighted four-parameter fits;
raw RTs are pooled before means, sample SDs, SEMs, CDFs, and quantiles.
SEM is sample SD divided by sqrt(pooled finite-trial n), describing pooled
trial variability. The RTDs have 20 ms bins, full-range normalization, and
0–0.6 s display. The CDFs use 1 ms bins; all 30 fresh thresholds are available
(37.5–79.5 ms), and all 60 Figure 1 scaling factors are valid. The previous
animal plots, fits, and source CSVs remain unchanged.

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

## Genotype-overlay psychometrics: two ways to combine ABLs

Run `.venv/bin/python asd_analysis/plot_session_type_1_genotype_overlay_psychometrics.py`,
then `.venv/bin/python asd_analysis/verify_session_type_1_genotype_overlay_psychometrics.py`.
The new output directory is `asd_analysis/session_type_1_genotype_overlay_psychometrics/`;
six PNGs and matching single-page PDFs are embedded in the 2026-10-14 result book.
Existing input CSVs, psychometrics, RTDs, and slope/JND figures remain unchanged.

Use the same 53 animals and 471,416 correct/error choices as the preceding
psychometrics (241,539 right; 229,877 left), without an RT restriction.
Keep session type 1, training 16, the confirmed next-trial repeat rule,
ABL 59→60, ABL 20/40/60, and signed ILDs ±1/2/4/8/16. The prepared CSV
already excludes 807 short-sound records; no further animal exclusions apply.
All 2,188 choices with nonfinite RT remain included.

- **Trial pooling:** At each group/genotype/ILD, sum right and left trial
  counts across ABLs, then calculate summed right divided by summed choices.
  ABLs contribute in proportion to their eligible trial counts at that ILD.
- **Equal ABL averaging:** First pool raw choices within each ABL and compute
  its measured P(right). At each ILD, average the three ABL probabilities
  equally. Plot those ten averaged points and fit afterward. Animals still
  contribute observed trials within each ABL; this is not an equal-animal
  average or an average of fitted curves/parameters.

Fit the same four-parameter sigmoid as the slope/JND analysis, with the same
initial guesses and bounds, using unweighted `curve_fit` on ten combined
ILD probabilities. Record upper/lower/x0/k, optimizer status/warnings,
unweighted residuals, and RMSE. Failed or non-increasing fits retain their
measured scatter points and are explicitly marked unavailable.

Each method has a 2×2 cohort figure, a 1×2 figure separating SHANK3/CNTNAP2,
and one combined-family/cohort subplot. WT, HET, and HOM share every subplot
as blue circles, orange squares, and green triangles with solid fitted curves.
Axes share −18–18 dB and P(right) 0–1 limits. Panel legends show contributing
animals and raw eligible choice counts; those counts are descriptive for
equal-ABL points, whose denominator is computed separately within each ABL.

Audits `overlay_abl_counts.csv`, `overlay_combined_points.csv`,
`overlay_group_counts.csv`, and `overlay_group_membership.csv` record the
630 ABL cells, 420 combined points, 21 genotype groups per method, and 159
membership records across three views. Combined points include every ABL's
counts, probability, and weight. `overlay_fits.csv` and
`overlay_fit_status_counts.csv` contain 42 fits; all 42 are increasing.
Validation manifests independently check raw CSV counts, all probabilities
and weights, fit residuals, saved PDF scatter/curve counts and colors, and
preserved input/previous-artifact SHA-256 hashes. The largest pointwise
difference between the methods is about 0.010848 in P(right).

## Trial-pooled genotype psychometrics with measured RT 0–1 seconds

Run `.venv/bin/python asd_analysis/plot_session_type_1_genotype_overlay_psychometrics_rt_0_to_1.py`,
then `.venv/bin/python asd_analysis/verify_session_type_1_genotype_overlay_psychometrics_rt_0_to_1.py`.
Outputs are in `asd_analysis/session_type_1_genotype_overlay_psychometrics_rt_0_to_1/`.
Three PNG/PDF pairs are appended to the 2026-10-14 result book; the preceding
six figures and all CSVs remain unchanged.

The requested `max_rt == 1` would select zero rows: the CSV's `max_rt` field
is a task setting of 10 seconds for every prepared row. As clarified with
the user, the new plots select finite **measured `timed_rt` in inclusive [0,1] s**.
Keep the preceding session/training/repeat/short-sound/outcome/stimulus rules.
This retains 466,978 choices (239,331 right; 227,647 left), excluding 2,188
nonfinite RTs and 2,250 RTs above 1 second from the 471,416 original choices.
All 53 animals remain, including all six SHANK3 cohort 2 animals
(ASD0072–ASD0077). The prepared CSV contains all six; no whole-animal removal
was applied. The historical short-sound cleanup involved ASD0045 in SHANK3
cohort 1 and ASD0026 in CNTNAP2 cohort 3, and retained their other trials.

Use trial pooling only: sum raw right/left counts across ABLs and animals at
each signed ILD within cohort/genotype, family/genotype, or combined genotype.
Fit the same bounded four-parameter sigmoid with unweighted ten-ILD points,
the same initial guesses, and `maxfev=10000`. WT/HET/HOM overlays retain the
2×2 cohort, 1×2 family, and single combined subplot layouts, colors, markers,
and x/y limits. Counts are the eligible RT-window choices.

The `rt_window_*.csv` audits cover 630 ABL cells, 210 pooled points, 21 group
counts/fits, and 159 membership records. All ABL normalization samples match
the existing all-ILD RTD counts. Independent checks verify raw CSV selection,
probability denominators, fit residuals, saved PDF marker/curve counts, and
unchanged prior-export and input hashes; manifests explicitly distinguish the
measured-RT filter from the unfiltered `max_rt` setting.

## Animal psychometrics, RT summaries, and chronometrics

The October 14 result-book entry includes the ASD0054 psychometric/RTD
example, the 53-animal two-column PDF, and its three-column extension with
chronometrics. All-animal PDFs use five animals per page (11 pages, three
animals on the last page). Original two-column exports remain available.
Chronometric plotting areas are square, about 1.76 × 1.76 inches, on
18.24-inch-wide pages; the psychometric/RTD panel widths are retained.

Psychometrics use all 471,416 valid choices without an RT restriction.
RTDs and RT statistics use all 469,228 finite `timed_rt` values, including
2,250 above one second; 2,188 missing RTs remain in the choice sample.
Full-range RTDs have 20 ms bins and area-one normalization, displaying only
0–0.6 seconds. Chronometrics pool actual +ILD/-ILD trials before calculating
mean RT and sample SEM at |ILD| 1/2/4/8/16; 795 points cover the same finite
RTs. No repeat filter is reapplied to the prepared CSV.

The three animal RT-summary figures show mean ± sample SD, mean ± SEM, and
median with a binomial/order-statistic 95% CI. The animal/ABL accuracy versus
mean-RT scatter uses all valid choices for accuracy and finite RTs for its
mean: ABL determines color, SHANK3/CNTNAP2 use hollow/filled markers, and
WT/HET/HOM use circles/squares/triangles. Each analysis has count/statistic
audits and independent validation reports in its named output folder.

## Animals with the smallest changes across ABL

The October 15 entry displays a two-by-five psychometric comparison, ranking
animals by `(|M20-M40| + |M40-M60| + |M20-M60|) / 3` for JND and fitted k.
Both rankings use the same 50 animals with complete paper-defined JNDs;
ASD0035/ASD0044/ASD0048 remain in the CSV and 53-animal audit, but are
ineligible for either ranking. No threshold definition or fit is changed.
The common set covers 444,718 choices. Top ΔJND animals are
ASD0061/ASD0055/ASD0042/ASD0063/ASD0027; top Δk animals are
ASD0050/ASD0055/ASD0051/ASD0064/ASD0039. The rows contain 43,490/41,474
choices, 30 curves, and 300 measured points, with ASD0055 shown twice.
Rank by unrounded values with animal-ID tie breaks. ΔJND and Δk have
different units and describe ABL stability; raw score sizes cannot be
compared directly. Full eligibility, pairwise differences, rankings, and
selected point/fit/count audits accompany the PNG/PDF and validation reports.

## Per-animal CDF cutoffs and quantile scaling

The extended all-animal PDF in the October 15 result book adds six panels
beneath each animal: CDFs, ΔCDFs, ABL 20/40/60 quantiles, and a rescaled
overlay. All 53 animals are included, regardless of JND eligibility. The
upper row reuses the previous psychometric/RTD/chronometric audits exactly.
The new row alone selects `timed_rt` in inclusive [0,1] seconds, yielding
466,978 valid-choice RTs. Pool raw +ILD/-ILD trials across sessions first.

CDFs use 1 ms bins normalized by each animal/ABL/|ILD| window count. For
each animal/|ILD|, ΔCDF is the mean of the three absolute ABL-pair differences.
The cutoff is the first bin center where the next ten bins' mean minus the
preceding ten bins' mean is at least 0.005, excluding the current bin.
All 265 cutoffs are available (10.5–104.5 ms); they replace the old paper's
hard-coded cutoffs for this new analysis only. Both CDF displays zoom to
0–0.15 seconds; cutoff markers appear only on ΔCDF panels.

Display five empirical quantiles (10/30/50/70/90%) against |ILD| 1/2/4/8/16.
For ABL 20/40, fit 1–99% quantiles against ABL 60 where its reference RT is
between the cutoff and 0.3 seconds. Match the original shifted-difference
fit: `x=Q60-t0; y=Qabl-Q60; m=sum(x*(y-yfirst))/sum(x*x)`. Above the cutoff,
use `t0+(Q-t0)/(1+m)`; earlier quantiles and the ABL-60 reference remain
unchanged. All 530 fits are valid. Missing cutoffs or invalid/nonpositive
factors are audited and omit affected scaled values, never silently replaced
with unscaled points. Full bin/CDF, fine-quantile, and fit-input audits use
gzip-compressed CSVs; other compact tables and JSON checks remain directly
readable. The book provides individual audit downloads and a complete bundle.

## Versioned artifacts and local data

The ASD analysis scripts, compact count/membership/histogram/fit audits, and
JSON preparation/validation manifests are versioned with the result-book pages.
Published PNGs, PDF downloads, source-script copies, and downloadable audits
are in the date folders under `docs/assets/results/`, including 2026-10-12,
2026-10-13, 2026-10-14, 2026-10-15, 2026-10-17, 2026-10-19, 2026-10-21,
and 2026-10-22. During git wrap-up, explicitly stage published
downloads and compact analysis CSV audits despite the repository's general
CSV/PDF ignore rules.

The original cohort CSVs, `metadata.csv`, the source ZIP, the three large
prepared session-type CSVs, and the archived pre-cleanup CSV remain local.
They are not included in this analysis commit; preparation manifests record
their provenance and hashes. Native PNG/PDF exports in the analysis output
directories also remain local under the existing ignore rules; their published
copies are included in the result book.

## Genotype chronometrics by cohort and gene family — October 21

Run `.venv/bin/python asd_analysis/plot_session_type_1_genotype_chronometrics.py`,
then `.venv/bin/python asd_analysis/verify_session_type_1_genotype_chronometrics.py`.
Both scripts use readable `# %%` blocks and editable parameters near the top.
Outputs are in `asd_analysis/session_type_1_genotype_chronometrics/`; the two
PNG images and matching single-page PDFs are embedded/downloadable in
`docs/results/2026-10-21.md`.

The current cleaned CSV contributes 446,949 valid-choice RTs from all 53
animals: `success in {1,-1}`, finite recorded `timed_rt` in inclusive [0,1]
seconds, ABL 59→60 then 20/40/60, and signed ILDs ±1/2/4/8/16. Prepared
training/repeat/short-sound/speaker-calibration rules remain applied; no
repeat filter is reapplied. This window excludes RTs above one second,
unlike the earlier all-finite-RT animal chronometrics.

Dots pool raw positive/negative ILD trials across sessions/animals before
calculating mean RT. Error bars use the sample SD of animal condition means
about their unweighted mean, divided by √animal count (`ddof=1`). The point
is trial weighted; animals have equal weight in the SEM, which is centered
on the pooled point as requested. This differs from earlier pooled-trial
SEM. SHANK3 cohort 2 WT/HOM each have one animal: retain their dots and
record SD/SEM as unavailable, with no bar and no zero substitution.

The four-cohort × three-ABL view has 180 points; the two-family × three-ABL
view has 90. All panels are square, share 0–0.45 s limits, and show filled
circular points with capped bars, no lines/fits. Genotype colors are WT blue,
HOM green, HET orange, offset −0.12/0/+0.12 at categorical |ILD| 1/2/4/8/16.
Audits include 795 animal-condition means, 270 pooled means, 240 available
SEMs, 30 unavailable SEMs, signed-trial counts, memberships, and PDF geometry.
The independent verifier reconstructs these statistics from the CSV, checks
the saved PDF dots/bar endpoints, and verifies unchanged inputs and prior
genotype exports.

## Genotype accuracy and JND by cohort and family — October 22

Run `.venv/bin/python asd_analysis/plot_session_type_1_genotype_accuracy_jnd.py`,
then `.venv/bin/python asd_analysis/verify_session_type_1_genotype_accuracy_jnd.py`.
The readable `# %%` plotter has editable parameters at the top. Output folder:
`asd_analysis/session_type_1_genotype_accuracy_jnd/`. Four PNGs and matching
single-page PDFs are directly embedded/downloadable in `docs/results/2026-10-22.md`.

Use the same 446,949 valid-choice RTs and all 53 animals as the October 21
chronometrics: `success in {1,-1}`, finite recorded `timed_rt` in inclusive
[0,1] seconds, ABL 59→60 then 20/40/60, signed ILDs ±1/2/4/8/16, and all
existing cleaned preparation rules. No repeat filter is reapplied.

The two accuracy figures retain cohort 4×3/family 2×3 layouts, ABL columns,
and categorical |ILD|. Dots pool actual ±ILD trials: correct / valid choices.
Bars use sample SD of animal accuracies around their unweighted mean / √N.
The 795 animal accuracies yield 270 pooled points, 240 SEM bars and 30 missing
single-animal bars. Accuracy limits are 0–1. Colors remain WT blue, HOM green,
HET orange, with offsets −0.12/0/+0.12 and no connecting lines.

The two JND figures retain those grids, with WT/HOM/HET as x categories.
Refit 159 animal and 54 pooled psychometrics using this RT window and the
unchanged unweighted four-parameter sigmoid and paper JND definition. All
54 pooled JNDs are defined; 154/159 individual JNDs are defined. Undefined:
ASD0035/0058 at ABL20, ASD0044 at ABL40, ASD0048 at ABL40/60. All their trials
remain in pooled fits. Bars use sample SD of defined animal JNDs / √defined N,
centered on the pooled-fit JND. They are not pooled-fit confidence intervals.
Show pooled and SEM-contributing animal counts separately. The 47 available
bars and seven missing bars are audited; no missing SEM is replaced by zero.
Both JND figures share limits covering every full bar.

Audits include raw condition/choice counts, memberships, animal/pooled
estimates, fit parameters/statuses, thresholds/crossings, SEMs and PDF geometry.
The independent verifier reconstructs selections and accuracies from the CSV,
refits all 213 curves, numerically checks 416 crossings, validates all 324
colored PDF dots/287 bars and missing bars, checks zero-lapse/invalid-fit
handling, and verifies unchanged data and prior-result hashes.


## SHANK3 cohort 1 WT RTDs by |ILD| (2026-10-22)

`plot_shank3_cohort1_wt_rtds.py` pools the five WT animals' actual trials from the cleaned session-type-1 CSV into a 1×5 |ILD|=1/2/4/8/16 figure. Each panel overlays ABL 20/40/60 (blue/orange/green; 59→60). Retain valid choices (`success` 1/−1) plus `abort_type=Fixation`, finite recorded RT in inclusive [−1,1] s. The 83,089 selected trials comprise 68,184 valid choices and 14,905 fixation aborts, with 14,820 negative RTs. No repeat eligibility is recalculated. Each 20 ms histogram is normalized by its ABL/|ILD| window count and bin width, area 1. Original signed trials and animals are pooled by their counts. All five animals contribute to every curve.

Run `.venv/bin/python asd_analysis/plot_shank3_cohort1_wt_rtds.py`. PNG/PDF, condition counts, membership, selected trial identities, histogram bins and independent cell-parse/count verification are in `shank3_cohort1_wt_rtds/`. The source CSV hash is unchanged. The PNG is directly embedded in the October 22 result book with source/audit downloads.


## SHANK3 cohort 1 fixation-abort durations by genotype (2026-10-22)

`plot_shank3_cohort1_fixation_abort_durations.py` plots recorded fixation duration (`fix_time` in milliseconds / 1000) for fixation aborts (`abort_type=Fixation`, ASD equivalent of abort_event 3) whose finite `timed_rt<0`, without a lower RT cutoff. The 1×3 panels are WT/HOM/HET (blue/green/orange). Retain ABL 20/40/60 after 59→60 and signed ILDs ±1/2/4/8/16, pooling actual trials across conditions, animals and sessions. The cleaned session-type-1 CSV already applies preparation exclusions; no repeat filter is reapplied.

Counts: WT 15,279 (5 animals), HOM 13,580 (6), HET 10,547 (3), total 39,406. Of these, 1,093 have timed_rt<−1 s and are retained. All recorded fixation durations are finite and nonnegative. Each 20 ms step histogram covers 0–2.1 s, includes every selected duration, and has area 1 using genotype-window count × 0.020 s as denominator. The x-axis is fixation duration; stimulus-relative RT is used only for selection.

Run `.venv/bin/python asd_analysis/plot_shank3_cohort1_fixation_abort_durations.py`. PNG/PDF and audits are in `shank3_cohort1_fixation_abort_durations/`; validation independently parses source cells, trial identities, fixation-duration conversion, bin counts and densities and checks an unchanged input hash. The PNG is directly embedded in the October 22 result book with source/audit downloads.


## SHANK3 cohort 1 proactive + exponential-lapse fits (2026-10-22)

The cleaned session-type-1 CSV is the sole fit input. Original documentation
identifies type 1 as standard RT; original SHANK3 cohort 1 contains types 1/2/23.
Retain training 16, ABL 20/40/60 after 59→60, and signed ILDs ±1/2/4/8/16.
Pool raw trials within WT/HOM/HET; do not reapply preparation/repeat exclusions
or restrict measured RT. There are 225,245 trials from 14 animals: WT 83,921
(5 animals), HOM 81,721 (6), HET 59,603 (3).

Use fixation-time coordinates, `fix_time/1000` and `intended_fix_time/1000` in
seconds. The 38,224 fixation aborts strictly before planned onset contribute
mixture density. All valid choices and the 1,381 late fixation-labelled rows
contribute mixture survival at planned onset: 187,021 censored observations.
The 26 valid choices with logged fixation duration slightly below onset stay
censored by outcome label. Post-stimulus response times are unused. This is
not the negative-RT selection used in the earlier fixation-duration histogram.

Five free parameters per genotype: drift, bound, combined delay, lapse
probability and exponential rate. Diffusion is one; no LED/reactive parameters.
Only the proactive process is delayed; exponential lapses start at fixation.
Use untruncated density/survival mixtures and the specified trapezoidal priors.

Readable cell-based scripts, with parameters at the top:

- `fit_shank3_cohort1_proactive_lapse.py`: preparation, two SVI starts per genotype,
  complete losses/checkpoints, 10,000 posterior draws and reproducible selection.
- `proactive_lapse_no_led_utils.py`: reusable float64 density/survival likelihood
  and priors; stable log survival and exact pure-component limits.
- `verify_shank3_cohort1_proactive_lapse.py`: independent CSV-cell reconstruction,
  SciPy/gradient/normalization checks, posterior and restoration verification.
- `plot_shank3_cohort1_proactive_lapse.py`: abort data/model, convergence, and five
  posterior-parameter panels; three PNGs with single-page PDFs.
- `publish_shank3_cohort1_proactive_lapse.py`: direct embeds and downloadable
  sources/audits in the October 22 result book, without replacing prior sections.

Run from the repository root:

```bash
.venv/bin/python asd_analysis/fit_shank3_cohort1_proactive_lapse.py --prepare-only
.venv/bin/python asd_analysis/verify_shank3_cohort1_proactive_lapse.py --pre-fit
.venv/bin/python asd_analysis/fit_shank3_cohort1_proactive_lapse.py
.venv/bin/python asd_analysis/plot_shank3_cohort1_proactive_lapse.py
.venv/bin/python asd_analysis/verify_shank3_cohort1_proactive_lapse.py
.venv/bin/python asd_analysis/publish_shank3_cohort1_proactive_lapse.py
.venv/bin/python -m mkdocs build --strict
```

For independent workers, specify `--genotype wt --seed 0` (and the other five
combinations); after all finish, use `--select-only`. Existing completed runs
are preserved; interrupted runs resume their saved complete window. Use a new
output folder when changing model or optimization parameters.

Outputs are in `shank3_cohort1_proactive_lapse/`. Float64 full-rank SVI uses
clipped Adam (0.0002; clip 1), init scale 0.1, 1,000-step windows, minimum
50,000 steps, patience 12, and absolute-best restoration. Budget is initially
150,000 steps with a 250,000 upper limit for the patience tail. Limit hits are
inconclusive. Lowest restored negative ELBO selects the start, with posterior-
mean likelihood resolving a 0.1% tie on identical trials.

Abort data bins are 20 ms, divided by all eligible genotype trials × width;
the area is the observed abort fraction, not one. Predictions apply every
trial's exact planned onset: density × fraction still exposed, and exact bin
mass `mean[F(min(b,c))-F(min(a,c))]`. Parameters use posterior means and central
95% variational intervals, which do not describe across-animal variability.
The fit root retains full optimizer/guide states; the result-book archive
includes trial IDs, classifications, timing discrepancies, memberships,
posterior draws, complete loss traces, convergence/selection tables and audits.

## SHANK3 cohort 1 WT: direct Gamma/Omega fits (2026-10-22)

`shank3_cohort1_wt_gamma_omega/` compares shared and condition-specific
reactive delays with the proactive and exponential-lapse components frozen
at the unrounded means of the selected WT seed-0 abort fit above.

Both objectives contain the same **68,184 valid choices from five animals**:
37,482 right and 30,702 left. Use cleaned session type 1, training 16, SHANK3
cohort 1 WT, `success in {1,-1}`, recorded `timed_rt` in inclusive [0,1] s,
ABL 20/40/60 after 59→60 and signed ILDs ±1/2/4/8/16. The 373 valid choices
above one second are excluded. Preserve preparation exclusions without
reapplying repeat filtering. Pool observed trials; do not average animals.

Each signed ABL–ILD condition has its own Gamma and Omega, with no ±ILD
symmetry. Both models fit global `w` and evidence-readout interval `del_go`.
One shared `t_E_aff` gives 63 free parameters; 30 condition delays give 92.
The complete ordinary-race/exponential-lapse choice density is divided by
its complete either-choice mass in `[c,c+1]`, where
`c=intended_fix_time/1000`. Response time is `c+timed_rt`; valid RT is not
reconstructed from `fix_time`. No aborts or censored observations enter these
new objectives. Half the fixed exponential-lapse density enters each choice.
Reactive delays are not capped at the minimum RT.

Scripts (readable `# %%` sections and editable parameters at the top):

- `fit_shank3_wt_gamma_omega.py`: immutable selection/provenance and two
  independently resumable full-batch, full-rank NumPyro fits.
- `shank3_wt_gamma_omega_utils.py`: stable direct Gamma/Omega evidence kernels,
  fixed proactive/lapse race, shared readout and joint-window normalizer.
- `plot_shank3_wt_gamma_omega.py`: four primary PNG/PDF figures, convergence,
  exact-parent-onset averages and in-sample numerical comparisons.
- `verify_shank3_wt_gamma_omega.py`: independent data/SciPy likelihood and
  posterior checks, exact signed pooling, RTD/CDF and psychometric audits.
- `publish_shank3_wt_gamma_omega.py`: verified-only October 22 direct image
  embeds, PDF/source downloads and an audit archive.
- `finish_shank3_wt_gamma_omega.py`: one-off continuation that waits for both
  running fits, then performs plotting, independent verification, publication
  and the strict build, stopping and recording any failed stage.

Preparation and preflight (the current prepared inputs already exist):

```bash
.venv/bin/python asd_analysis/fit_shank3_wt_gamma_omega.py --prepare-only
.venv/bin/python asd_analysis/verify_shank3_wt_gamma_omega.py --pre-fit
```

Run the following in separate workers for concurrent fitting. `ASDFIT_CPUS`
can specify disjoint comma-separated CPU sets. One initialization per model,
seed 0; full-batch float64 SVI, clipped Adam 0.0002/clip 1, guide scale 0.1.

```bash
.venv/bin/python asd_analysis/fit_shank3_wt_gamma_omega.py --variant shared_delay
.venv/bin/python asd_analysis/fit_shank3_wt_gamma_omega.py --variant condition_delay
```

Check 1,000-step windows, minimum 50,000 updates, and 25 windows after the best
loss. The user removed the parameter-mean movement below 0.1 SD as a stopping
requirement; retain it only as a diagnostic. `stopping_rule_change.json` records
the authorization, old/new rules and hashes of the archived configurations,
sources and states under `before_loss_only_stop/`. Both existing runs already
passed the revised rule: restore the best checkpoints at 49,000/141,000 updates
without taking more updates. Status `converged` refers to the revised loss rule,
not a passed parameter-stability test. The minimum step count applies to the
checked run rather than the restored best step.
Restore the absolute-best state. Maximum 1,000,000 updates; limit
hits are inconclusive. Save 10,000 posterior draws, complete losses and both
best/latest guide/optimizer states. Completed runs are preserved. Interrupted
runs resume only with identical source hashes and configuration.

The current runs migrated from CPU to the NVIDIA RTX 4090 after 28,000
shared-delay and 19,000 condition-delay updates. `gpu_migration.json` records
the exact optimizer/guide/RNG transfer and archived CPU files under
`cpu_before_gpu/`; only the configuration backend changed. Both continue in
float64 with the same source code, trials, optimization and convergence rules.
GPU launches use `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false`
before the commands above. The two processes share one GPU. CPU/GPU likelihood,
gradient and update timings are recorded in `gpu_benchmark.json`; final
posterior reproduction across devices allows only float64 rounding differences.

After both complete:

```bash
.venv/bin/python asd_analysis/plot_shank3_wt_gamma_omega.py
.venv/bin/python asd_analysis/verify_shank3_wt_gamma_omega.py
.venv/bin/python asd_analysis/publish_shank3_wt_gamma_omega.py
.venv/bin/python -m mkdocs build --strict
```

Alternatively, start `.venv/bin/python asd_analysis/finish_shank3_wt_gamma_omega.py`
alongside the fits to execute those four final steps automatically. Read
`completion.json`, the stage logs and each model's `run_summary.json` for
actual completion/convergence status; publication and final visual inspection
are recorded separately.

The RTD grids contain 81,934 observations: the 68,184 fitted choices plus
13,750 strict pre-onset fixation aborts within [−1,0), timed as
`fix_time/1000 - intended_fix_time/1000`. The negative observations are only
displayed. Late fixation-labelled rows are excluded from observed aborts.
Data use 20 ms bins; models use 1 ms evaluation. Pool signed conditions with
actual counts and normalize each data/model panel separately to area one
over [−1,1]. Full-model predictions average all 83,921 eligible parent
onset times and signed-condition frequencies; audits retain window masses.
Psychometric predictions integrate each trial's right density over [0,1],
divide by its complete window mass, then average the fitted valid onsets.
Lines connect only the ten fitted ILDs; no sigmoid is added.

Posterior intervals are conditional on frozen abort parameters, and the
single starts do not test initialization sensitivity. Likelihood/CDF/choice
comparisons are in-sample. The optional `--preview-initial` plotting and
verification modes use an isolated `initial_preview/` folder; those are
pipeline dry runs, not fitted results, and are never published.

## SHANK3 cohort 1 HOM: condition-specific Gamma/Omega/delay (2026-10-23)

`shank3_cohort1_hom_gamma_omega/` is a separate HOM analysis, preserving the
WT scripts/results and all source CSVs. Pool six cohort-1 HOM animals:
ASD0023, ASD0024, ASD0043, ASD0045, ASD0046, ASD0048. The cleaned session-type-1
selection uses training level 16, success ±1, finite recorded RT in [0,1] s,
ABL 20/40/60 after 59→60, and ten signed ILDs ±1/2/4/8/16. There are 67,715
fitted choices (35,320 right, 32,395 left); 357 valid choices above 1 s are
excluded. Prepared repeat and short-sound exclusions are retained.

Freeze the five unrounded HOM seed-0 proactive/exponential-lapse posterior
means. Reuse the unchanged `shank3_wt_gamma_omega_utils.py` likelihood and
priors, with 30 Gamma, 30 Omega, 30 delays, shared w and shared del_go:
92 parameters and no shared-delay fit. Use exact planned onset and normalize
the complete joint mixture over the valid [0,1] s window.

The independent, readable `# %%` scripts are `fit_shank3_hom_gamma_omega.py`,
`plot_shank3_hom_gamma_omega.py`, `verify_shank3_hom_gamma_omega.py`,
`publish_shank3_hom_gamma_omega.py`, and `finish_shank3_hom_gamma_omega.py`.
Parameters and source roots are near the top. Use `.venv/bin/python`.

Preparation and numerical preflight:

```bash
.venv/bin/python asd_analysis/fit_shank3_hom_gamma_omega.py --prepare-only
.venv/bin/python asd_analysis/verify_shank3_hom_gamma_omega.py --pre-fit
```

Run one seed-0, full-batch, float64 full-rank fit on the RTX 4090:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python asd_analysis/fit_shank3_hom_gamma_omega.py
```

Clipped Adam uses learning rate 0.0002, clip norm 1 and initial guide scale
0.1. Every strictly lower 1,000-update mean loss resets confirmation; after
at least 50,000 updates, stop following 25 windows without improvement and
restore the absolute-best checkpoint. The minimum applies to the checked
run, not its restored best step. Parameter stability is diagnostic only, at
both terminal and restored checkpoints. A 1,000,000-update limit without
confirmation is inconclusive. Save 10,000 posterior draws and 95% intervals;
uncertainty is conditional on frozen abort parameters. One initialization
does not assess start sensitivity.

The `--checkpoint-only-after 2000` execution option saves an unfinished
state without posterior finalization. Relaunch without it to continue the
same optimizer/guide/RNG state with identical mathematical configuration.
The initial controlled continuation is documented in `resume_validation.json`
and `condition_delay/before_resume_check/`; completed fits cannot be overwritten.

The RTD grid uses 80,989 observations: 67,715 fitted choices and 13,274
strict pre-onset fixation aborts in [-1,0), displayed only. It pools actual
signed trials, averages exact onset times across 81,721 parent trials, and
normalizes each pooled data/model panel separately over [-1,1] s. Empirical
steps use 20 ms bins and 1.4 pt width above the thinner 0.8 pt model evaluated
at 1 ms. Psychometrics use data scatter and 1.0 pt model lines at the measured
signed ILDs, with exact per-trial valid-window normalization before averaging.

The finishing script generates figures, independently verifies them, then
waits for a manually recorded `visual_inspection.json` before publishing and
running the strict MkDocs build. Read `completion.json` and stage logs for
actual status. The two primary PNGs/PDFs, supplementary convergence figure,
source files and audits are published in the October 23 result book.

Completed at 187,000 checked updates, restoring the 162,000-update best
checkpoint after exactly 25 confirmation windows. Mean negative ELBO at the
best window is -48617.280308. Both terminal and restored parameter-stability
diagnostics are false, as permitted by the requested loss-only rule. All
67,715 fitted trials, 15 RTD panels and 30 psychometric probabilities passed
independent checks; CPU/GPU reproduction of 10,000 posterior draws differed
by at most 6.44e-15. The two primary PNGs/PDFs and supplementary convergence
figure passed visual inspection and are embedded/linked in the
[October 23 entry](../docs/results/2026-10-23.md#shank3-cohort-1-hom-condition-specific-reactive-delays).

## SHANK3 cohort 1 WT/HOM parameter and chronometric comparisons (2026-10-23)

`shank3_cohort1_wt_hom_comparison/` contains seven PNGs and matching
single-page PDFs derived from the saved condition-delay fits, without
refitting. The three 1x3 Gamma/Omega/reactive-delay figures have ABL columns
and separate signed ILD categories; the 1x2 shared-parameter figure shows w
and del_go. WT is blue and HOM red. All 184 group/parameter estimates use
saved posterior means and 2.5th/97.5th percentiles of 10,000 draws. Delay
parameters are shown in milliseconds; Omega in inverse seconds. Intervals
are conditional variational uncertainty with proactive/lapse means frozen,
not animal-to-animal variability. Frozen parameters are excluded.

Three compact square chronometrics show WT ABL-separated, HOM ABL-separated,
and WT/HOM with ABLs pooled. The original fitted valid [0,1] s populations
are preserved: 68,184 WT choices / five animals and 67,715 HOM choices / six
animals. Data dots pool actual trials; bars use sample SD across animal
means / sqrt(animal count), centered on those pooled dots. For the pooled
ABL view, pool within each animal first, then calculate SEM. Positive and
negative ILDs are combined by observed trial counts. Models integrate the
complete density conditional on each trial's exact valid-window mass before
averaging; ABL/sign mixture weights use fitted valid-trial counts. The
earlier abort-plus-valid RTDs are not used to obtain chronometric means.
Model lines are 1.0 pt, all three panels use 0-0.30 s, and ABL-separated
colors are blue/orange/green for 20/40/60.

Sources and reproducible steps:

```bash
.venv/bin/python asd_analysis/plot_shank3_cohort1_wt_hom_comparison.py
.venv/bin/python asd_analysis/verify_shank3_cohort1_wt_hom_comparison.py
```

After visually inspecting all seven images and recording their hashes in
`visual_inspection.json`, publish and build:

```bash
.venv/bin/python asd_analysis/publish_shank3_cohort1_wt_hom_comparison.py
.venv/bin/python -m mkdocs build --strict
```

The independent verifier reconstructs original source trial identities,
all 184 posterior summaries, 60 signed model means, 40 plotted data/model
pairs and 40 SEMs, 165 animal/ABL/absolute-ILD means and 55 ABL-pooled animal
means. Primary density quadrature is checked against independent adaptive
SciPy conditional-survival integration and the saved CDF integral; the
largest density-versus-saved-CDF difference is 4.32e-8 s (tolerance 1e-6 s).
All source CSVs and saved fit files retain their baseline hashes. Audits
include condition mappings, pooling weights, memberships, means, intervals,
SEMs, integration checks and preserved source snapshots.

The two composite overview images are now directly embedded in the
[October 23 result book](../docs/results/2026-10-23.md#shank3-cohort-1-wt-vs-hom-parameters-and-chronometrics),
with PDF, source and audit downloads. The seven original individual PNGs/PDFs
remain unchanged and available as downloads; previous fit entries remain available.

### Combined WT/HOM overview figures

`plot_shank3_cohort1_wt_hom_composites.py` rearranges the verified numerical
tables without refitting or repeating model integration. New exports live
under `shank3_cohort1_wt_hom_comparison/composites/`:

- `SHANK3_cohort1_WT_HOM_all_parameters.png` / `.pdf`: 11 panels, with three
  Gamma/Omega/reactive-delay rows and ABL columns, then shared w and del_go.
  The 184 posterior means and central 95% credible intervals are unchanged.
- `SHANK3_cohort1_WT_HOM_behavior_comparison.png` / `.pdf`: four rows containing
  WT's five |ILD| RTDs, HOM's five RTDs, three compact square chronometrics,
  and separate WT/HOM psychometrics (15 panels). The requested "HET" row
  refers to HOM in these completed-fit comparisons.

RTDs overlay ABL 20/40/60 in blue/orange/green, with solid 1.4 pt empirical
steps and matching-color 0.8 pt dashed model curves. Each retains its own
normalization over [-1,1] s (20 ms data bins; 1 ms model sampling), with one
shared density scale. They include 81,934 WT observations (68,184 valid +
13,750 pre-onset aborts) and 80,989 HOM observations (67,715 valid + 13,274
aborts). Exact parent-trial onset/sign weights are retained for RTD models.
Chronometric/psychometric values instead use fitted valid choices in [0,1] s,
and the chronometric model normalizes at each valid trial's exact onset.
All prior means, equal-animal SEMs, probabilities, curves and fit files are
preserved. Model densities retain exact CDF window-mass normalization; the
1 ms trapezoid approximation differs from unit area by at most 1.58e-6.

Reproduce the composites, then independently check the actual plotted artists:

```bash
.venv/bin/python asd_analysis/plot_shank3_cohort1_wt_hom_composites.py
.venv/bin/python asd_analysis/verify_shank3_cohort1_wt_hom_composites.py
```

After inspecting both PNGs and recording their hashes in the composites'
`visual_inspection.json`, publish the two main embeds and build:

```bash
.venv/bin/python asd_analysis/publish_shank3_cohort1_wt_hom_composites.py
.venv/bin/python -m mkdocs build --strict
```

The verifier reconstructs posterior summaries from all 10,000 draws and
rebuilds RTD counts, valid-choice probabilities, pooled RT means and animal
SEMs directly from the saved trial populations. It checks every plotted
array, 11/15 panel layouts, square lower panels, limits, colors, both
single-page PDFs, and 285 preserved input/fit/previous-output/asset hashes.
The publisher edits only the existing WT/HOM section, retaining original
download links and all other October 23 entries. Its audit ZIP includes the
source scripts, original section, input tables, plotted arrays and checks.

### WT/HOM RTD display zoom: 0–0.6 s

`plot_shank3_cohort1_wt_hom_rtd_zoom.py` creates separate WT and HOM 3×5
ABL-by-|ILD| PNG/PDF grids under `shank3_cohort1_wt_hom_comparison/rtd_zoom_0_0p6/`.
It reuses the saved condition-delay RTD arrays and the combined figure's
common y-scale, changing the display to 0–0.6 s without filtering or
renormalizing. Normalization remains over [-1,1] s. Colored 1.4 pt data
steps are drawn above thin 0.7 pt black model lines. All 3,000 data bins
and 60,030 model samples in the new artists are checked exactly against
the previous composite's artists; prior fits, source data and figures are
preserved. Both images and PDF downloads are in the October 23 section
**WT and HOM: RTDs zoomed to 0–0.6 s**.

## SHANK3 cohort 1 WT/HOM: tied Omega and one reactive delay (2026-10-23)

`shank3_cohort1_wt_hom_tied_omega/` is a separate four-run analysis. Each
genotype has 30 independent signed-condition Gammas, one reactive delay,
shared w and del_go. `omega_by_abl` has three Omegas (36 free parameters);
`omega_global` has one (34 parameters). WT and HOM fit independently, using
the exact original 68,184 WT / 67,715 HOM valid choices in [0,1] s, their
recorded RTs and exact planned onsets, and the same frozen five proactive/
exponential-lapse posterior means. Priors apply once per free parameter;
deterministic expansion to 30 conditions is not additional sampling.

All four fits completed on the RTX 4090 after the restart restored matching
NVIDIA driver 595.91.07. Each satisfied the loss-confirmation stopping rule
and restored its absolute-best checkpoint. Parameter stability remained a
diagnostic, as requested; it was not a stopping requirement.

| Genotype | Variant | Restored-best update | Final checked update |
|---|---|---:|---:|
| WT | Omega by ABL | 52,000 | 77,000 |
| HOM | Omega by ABL | 52,000 | 77,000 |
| WT | Global Omega | 129,000 | 154,000 |
| HOM | Global Omega | 70,000 | 95,000 |

Each run has 10,000 posterior draws. Independent posterior/checkpoint and
diagnostic checks passed, all four final PNGs were visually inspected,
and the matching PNG/PDF composites are published. For both genotypes,
Omega by ABL has higher joint log likelihood and smaller psychometric and
RT-CDF discrepancies than global Omega on the fitted trials. This is an
in-sample comparison with one initialization per fit; see
`in_sample_comparison.csv` for all four rows.

Scripts use editable `# %%` blocks. Preparation copies original inputs
byte-for-byte and records preserved hashes. The independent prefit verifier
rebuilds selection from the cleaned CSV, checks 36/34 free parameters,
prior multiplicities, initializations, SciPy likelihood agreement, finite
gradients and full-mixture normalization. Small 30-trial/15-update tests
validate interrupted/resumed states against uninterrupted updates, including
recovery from a corrupted derived history export. Those tests and the
separate `initial_preview/` figures are not scientific fit results.

```bash
.venv/bin/python asd_analysis/prepare_shank3_tied_omega.py
.venv/bin/python asd_analysis/verify_shank3_tied_omega.py --pre-fit
# Launch or resume the GPU pipeline:
.venv/bin/python asd_analysis/run_shank3_tied_omega_pipeline.py
```

The launcher verifies CUDA access, benchmarks disposable fresh states, then
runs four fits concurrently with GPU preallocation disabled (two at a time
if measured memory requires it). Benchmark states never seed the real fits.
Each fit uses seed 0, float64 full-batch full-rank SVI, clipped Adam 0.0002,
clip norm 1 and initial scale 0.1. Minimum 50,000 updates, 1,000-update loss
windows and 25 windows without a strictly lower loss determine stopping;
the best checkpoint is restored, and the cap is 1,000,000 updates.
Parameter stability is diagnostic only. Atomic checkpoints contain both
current/best optimizer/guide states and full histories for exact resume.

After all four fits, the pipeline verifies posterior draws/checkpoints,
generates predictions and four PNG/PDF composites, then independently checks
SciPy survival-integral means, choices, parent pooling, empirical statistics,
actual plotted artists, limits and download files. Each variant gets one
seven-panel parameter figure and one 15-panel behavior figure. RTDs display
[-0.1,0.6] s with 20 ms data bins and thin 0.7 pt model curves, retaining
[-1,1] s normalization; no empirical values are changed. Chronometric dots
pool trials; SEM weights animal means equally. Shared parameters are shown
once, and corresponding axes are common across variants.

The pipeline leaves completed figures at `figures_ready_for_visual_inspection`
until the four final PNG hashes have been visually checked and recorded in
`figures/visual_inspection.json`. Then publish and build:

```bash
.venv/bin/python asd_analysis/publish_shank3_tied_omega.py
.venv/bin/python -m mkdocs build --strict
```

The October 23 section **SHANK3 cohort 1 WT vs HOM: tied-Omega and shared-delay
refits** embeds all four final PNGs and provides matching PDFs, sources and
audit downloads. Publication requires real 10,000-draw posteriors, numerical
checks and visual inspection. Initial-value previews cannot be published
by the publisher.


## SHANK3 cohort 1 WT/HOM: Gamma shared across ABLs (2026-10-23)

`shank3_cohort1_wt_hom_gamma_by_ild/` contains a separate two-run analysis:
10 independent signed-ILD Gammas shared across ABLs, three ABL Omegas,
one reactive delay, shared w and del_go. These are **16 free parameters
per genotype**, expanded deterministically to the existing 30 stimulus
conditions, with each prior applied once per free parameter. WT and HOM
fit independently. Positive and negative ILDs are not constrained to be
symmetric.

Use exactly the previous 68,184 WT choices from five animals and 67,715 HOM
choices from six animals, valid success ±1 and finite recorded RT in [0,1] s.
Keep session type 1, training 16, ABL 59→60, existing signed ILDs and the same
frozen proactive/exponential-lapse parameters. The prepared filters are not
reapplied. All original inputs and previous fits/figures are protected by
hashes. Gamma initializes from half the pooled right-choice logit per signed
ILD; Omega initializes from trial-weighted condition initials per ABL.
Delay starts at 0.05 s, w at 0.5 and del_go at 0.08 s.

Both fresh seed-0 runs execute concurrently on GPU, with float64 full-batch
full-rank SVI, clipped Adam 0.0002, clip norm 1 and initial guide scale 0.1.
Use 1,000-update loss windows, minimum 50,000 updates, 25 windows without a
strictly lower loss, and restore the absolute best checkpoint. Parameter
stability is diagnostic only; an unconfirmed 1,000,000-update limit hit is
inconclusive. Save 10,000 posterior draws and complete resumable histories.

```bash
.venv/bin/python asd_analysis/prepare_shank3_gamma_by_ild.py
.venv/bin/python asd_analysis/verify_shank3_gamma_by_ild.py --pre-fit
.venv/bin/python asd_analysis/verify_shank3_gamma_by_ild_smoke.py
.venv/bin/python asd_analysis/run_shank3_gamma_by_ild_pipeline.py
```

The Gamma panel uses 50% of its original width (4.55 inches), preserving
height, axis limits and all estimates. `GAMMA_WIDTH_FRACTION` controls its width.

The five-panel parameter composite shows shared Gamma once, Omega by ABL,
reactive delay, w and del_go: WT blue/HOM red, posterior means and central
95% credible intervals, delays in ms. The 15-panel behavior composite retains
five WT RTDs, five HOM RTDs, three compact chronometrics and separate WT/HOM
psychometrics. RTD model curves are **solid, opaque, 1.2 pt above data steps
at alpha 0.5 and 1.4 pt**. ABL 20/40/60 use blue/orange/green. Display
[-0.1,0.6] s without changing 20 ms empirical bins, 1 ms model evaluation
or full [-1,1] s normalization.

Empirical values and pooling weights are unchanged. Valid-trial predictions
normalize each exact onset before averaging. Chronometric dots pool trials;
SEM weights animal means equally. Independent SciPy checks cover likelihoods,
normalization, means/choices and actual plotted arrays/styles. The new fits
are compared with the earlier 36-parameter Omega-by-ABL model on identical
trials; these are in-sample comparisons with one initialization per fit.

Both GPU fits completed at 74,000 updates, restoring their absolute-best
49,000-update checkpoints after 25 confirmation windows. Each has 10,000
saved posterior draws. Independent verification passed for all 32 parameter
summaries, 60 signed predictions and 40 chronometric pairs; maximum independent
mean-RT error was 8.1e-12 s. Both final fitted composites were visually inspected.

The 16-parameter fits preserve RT predictions closely while psychometric RMSE
increases compared with the 36-parameter reference: WT 0.01263 → 0.02687 and
HOM 0.01515 → 0.01925. Joint log likelihood decreases by 164.62 (WT) and
43.44 (HOM), evaluated on identical trials at the physical posterior means.
These are in-sample diagnostics, not held-out model selection.

The October 23 section **SHANK3 cohort 1 WT vs HOM: Gamma shared across ABLs**
embeds the two final PNGs with PDF/source/audit downloads. Posterior, numerical
and visual checks passed; initial previews remain isolated.


## SHANK3 cohort 1 WT: NPL+alpha with condition-specific delays (2026-11-04)

37-parameter fit using the exact 68,184 WT choices from the existing condition-delay reference, with five proactive/exponential-lapse parameters frozen. No prepared filters are reapplied. Full-rank GPU SVI uses minimum 50k updates and a strict-improvement 25-window confirmation tail, without a parameter-stability gate. Output: `shank3_cohort1_wt_npl_alpha_condition_delay/`.

Scripts: `prepare_shank3_wt_npl_alpha.py`, `verify_shank3_wt_npl_alpha.py --pre-fit`, `run_shank3_wt_npl_alpha_pipeline.py`, and (after visual inspection) `publish_shank3_wt_npl_alpha.py`. Two PNGs: 3×5 RTDs (full [-1,1] normalization; [-0.1,0.6] display) and ABL-colored psychometric points/model lines. The result book includes direct embeds, posterior/audit/source downloads and an in-sample comparison against the 92-parameter Gamma/Omega reference.


## SHANK3 cohort 1 HOM: NPL+alpha with condition-specific delays (2026-11-04)

37-parameter fit using the exact 67,715 HOM choices from the existing condition-delay reference, with five proactive/exponential-lapse parameters frozen. No prepared filters are reapplied. Full-rank GPU SVI uses minimum 50k updates and a strict-improvement 25-window confirmation tail, without a parameter-stability gate. Output: `shank3_cohort1_hom_npl_alpha_condition_delay/`.

Scripts: `prepare_shank3_hom_npl_alpha.py`, `verify_shank3_hom_npl_alpha.py --pre-fit`, `run_shank3_hom_npl_alpha_pipeline.py`, and (after visual inspection) `publish_shank3_hom_npl_alpha.py`. Two PNGs: 3×5 RTDs (full [-1,1] normalization; [-0.1,0.6] display) and ABL-colored psychometric points/model lines. The result book includes direct embeds, posterior/audit/source downloads and an in-sample comparison against the 92-parameter Gamma/Omega reference.


## WT/HOM NPL+alpha posterior-parameter comparison (2026-11-04)

`plot_shank3_wt_hom_npl_alpha_parameters.py` reads the two completed NPL+alpha seed-0 posteriors without refitting. One ten-panel PNG/PDF contains all 37 fitted parameters per genotype: seven shared-parameter panels and a 1×3 ABL delay row versus ten signed ILDs. Control/WT is black; ASD/HOM is red. All axes are equal 2.8-inch squares, with no grids. Error bars are central 95% credible intervals from 10,000 draws, conditional on fixed proactive/lapse values; times are shown in ms. Outputs and 74-estimate audit: `shank3_cohort1_wt_hom_npl_alpha_comparison/`. Verify with `verify_shank3_wt_hom_npl_alpha_parameters.py`. The November 4 result book directly embeds the PNG and provides the PDF, source and audits.

### NPL+alpha workflow files and saved artifacts

For each genotype, replace `<group>` with `wt` or `hom` in these filenames:

- `prepare_shank3_<group>_npl_alpha.py` copies the exact existing Gamma/Omega reference trials, fixed proactive/lapse values and condition maps, recording preservation hashes without reapplying preparation filters.
- `fit_shank3_<group>_npl_alpha.py` runs the fresh seed-0 full-rank GPU fit, with disposable timing and smoke-test options, atomic resumable checkpoints, strict loss-confirmation stopping and absolute-best restoration.
- `plot_shank3_<group>_npl_alpha.py` generates the 3×5 RTD and psychometric PNGs, signed-condition predictions, window masses and in-sample reference comparisons.
- `verify_shank3_<group>_npl_alpha.py` independently reconstructs trial identities, priors, mapping, SciPy likelihoods, normalization, checkpoint/posterior restoration and numerical plot values; `--pre-fit` runs the checks required before fitting.
- `publish_shank3_<group>_npl_alpha.py` embeds verified PNGs and downloadable source/audits in the November 4 result book while preserving existing entries.
- `run_shank3_<group>_npl_alpha_pipeline.py` sequences fitting, prediction and independent verification, recording failures and preserving completed checkpoints; visual inspection precedes publication.

`shank3_wt_npl_alpha_utils.py` defines the shared 37-parameter model, established trapezoidal priors, fresh initialization, NPL+alpha Gamma/Omega mapping and loss-window decision. `shank3_hom_npl_alpha_utils.py` reuses those pure functions with HOM-specific output/reference roots. Both use `shank3_wt_gamma_omega_utils.py` for stable evidence kernels and the complete-mixture conditional RT/choice likelihood, and `proactive_lapse_no_led_utils.py` for the untruncated inverse-Gaussian/exponential proactive model and priors. Existing `fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py` supplies the canonical NPL+alpha mapping; `ASD_Human_Data/human_plain_ddm_diagnostics_utils.py` supplies the independent SciPy evidence calculations.

The comparison workflow is `plot_shank3_wt_hom_npl_alpha_parameters.py`, then `verify_shank3_wt_hom_npl_alpha_parameters.py`, followed by actual visual inspection and `publish_shank3_wt_hom_npl_alpha_parameters.py`. It reads the two saved 10,000-draw posteriors and makes no model or data changes.

Versioned results include compact preparation/provenance records, condition/membership and numerical summaries, posterior draws and losses, plus the November 4 published images, parameter PDF and complete audit downloads. Raw cohort/prepared CSVs, copied trial-level inputs and optimizer/checkpoint pickle files remain local under the repository's existing ignore rules. Preparation and independent raw-data verification require those local datasets and the earlier Gamma/Omega/abort reference fit folders. The published per-genotype audit ZIPs retain trial tables, likelihood data, posterior draws, losses, numerical predictions and source snapshots.


## SHANK3 cohort 1 WT/HOM NPL+alpha Gaussian condition delays (2026-11-05)

67 free parameters per genotype: seven shared NPL+alpha parameters and 30 Gaussian delay means/SDs. The zero-truncated trial delay is integrated before complete-mixture normalization, retaining frozen genotype-specific exponential lapses. Fresh seed-0 random initialization within flat prior regions; loss-only confirmation and best restoration. Outputs: `shank3_cohort1_wt_hom_npl_alpha_gaussian_delay/`. Entrypoints: `prepare_shank3_npl_alpha_gaussian.py`, `run_shank3_npl_alpha_gaussian_pipeline.py`, and `publish_shank3_npl_alpha_gaussian.py`; the matching independent verifiers and plotters share the same analysis prefix. The November 5 result book embeds WT/HOM RTD/psychometric PNGs and the black/red 13-square-panel posterior comparison, with its PDF and source/audit downloads. Existing point-delay fits remain untouched.

Workflow files:

- `shank3_npl_alpha_gaussian_utils.py` defines the 67-parameter model, priors, random initialization, delay-marginalized race and complete-window normalizer. It reuses `ASD_Human_Data/gaussian_delay_likelihood.py` for truncated-Gaussian convolution and rescaled quadrature, together with the existing point-delay NPL and proactive utilities.
- `prepare_shank3_npl_alpha_gaussian.py` preserves reference trial identities and frozen values, and records fresh seed-0 initialization and provenance for both genotypes.
- `fit_shank3_npl_alpha_gaussian.py` runs or resumes one genotype's float64 full-rank SVI, saves best/latest checkpoints and 10,000 draws, and restores the absolute-best loss window.
- `run_shank3_npl_alpha_gaussian_pipeline.py` sequences fitting, predictions and independent verification with a completion record; publication follows visual inspection.
- `plot_shank3_npl_alpha_gaussian.py` creates RTD/psychometric PNGs and exact-onset predictions; `plot_shank3_npl_alpha_gaussian_parameters.py` creates the 13-square-panel posterior comparison and actual truncated-delay moment tables.
- `verify_shank3_npl_alpha_gaussian.py` independently checks selection, priors, initialization, SciPy convolution/likelihood agreement, gradients, normalization and posterior restoration. `verify_shank3_npl_alpha_gaussian_figures.py` reconstructs all displayed counts, pooling weights, predictions and intervals.
- `publish_shank3_npl_alpha_gaussian.py` embeds verified images and source/audit downloads in the November 5 result book.

Both fits converged after 116,000 updates and restored update 91,000, followed by 25 non-improving windows; parameter stability is diagnostic only. `source_migration.json` and `optimization_check.json` document an execution-only optimization at the 8,000-update checkpoints, preserving optimizer/RNG state and verified likelihood/gradient equivalence. `resume_validation.json` records an independent interrupted-versus-uninterrupted recovery check.

Versioned results include compact numerical/provenance tables, posterior draws, loss traces and published PNG/PDF/source downloads. The published audit ZIP additionally preserves copied trial inputs, numerical caches, resumable checkpoints and source history. Raw datasets, duplicate local caches, previews and smoke-run folders remain local. Raw-data verification and preparation require the local cleaned CSV and reference fits. Focused figure verification: `JAX_PLATFORMS=cpu .venv/bin/python asd_analysis/verify_shank3_npl_alpha_gaussian_figures.py`; result-book validation: `.venv/bin/python -m mkdocs build --strict`.


## WT sensory-delay effect in the NPL+alpha fit (2026-11-07)

`plot_shank3_sensory_delay_effect.py` reads saved Gaussian-delay posterior means (editable `GENOTYPE`, currently `wt`) and produces a **three-ABL-row × ten-signed-ILD-column** PNG/PDF, overlaying three continuous model-density curves at **1 ms** resolution in every panel: sensory delay/NDT (blue), complete model with exactly zero sensory delay (orange), and the fitted full model RT (green). There are no empirical data or histogram steps in this figure. Proactive timing, exponential lapses and all remaining NPL+alpha parameters and exact parent onsets remain fixed; no refitting or ±ILD pooling. Each RT curve normalizes independently over [-1,1] s; NDT keeps its full nonnegative normalization. All panels display [0,0.4] s with common density limits; the zoom does not change normalization.

`verify_shank3_sensory_delay_effect.py` checks all 90 plotted 1 ms arrays and curve-to-panel mappings against the original numerical export, reconstructs complete right/left densities with independent SciPy series and raw onset frequencies, and verifies unchanged fit inputs. Outputs and audits: `shank3_cohort1_wt_hom_npl_alpha_gaussian_delay/wt/sensory_delay_effect/overlay_1ms/`. The November 7 result book directly embeds this figure and links the PDF, sources and audits; the earlier 20 ms histogram layout remains downloadable. Reference 20 ms bin tables are retained only as numerical checks and are not used for plotting.


## NPL+alpha Gaussian delays shared across ILDs (2026-11-07)

The WT/HOM extension fits **13 parameters per genotype**: seven shared NPL+alpha parameters plus three ABL-specific Gaussian means and three SDs. Each ABL's pair is shared across all ten signed ILDs. Evidence retains all 30 conditions. The exact earlier WT/HOM trial identities and five fixed proactive/exponential-lapse values are reused; Gaussian delays are integrated before complete-mixture valid-window normalization. Priors apply once per free parameter; there is no additional lapse component or upper cutoff on delay draws.

Scripts use the `shank3_npl_alpha_gaussian_abl_delay` prefix:

- `_utils.py` defines sampling, 13-parameter initialization and deterministic expansion to the existing immutable 67-parameter numerical kernels.
- `prepare_…py` preserves reference inputs and hashes, records condition-to-ABL sharing maps and identical fresh seed-0 initial values for WT/HOM.
- `fit_…py` runs or resumes full-rank float64 GPU SVI, restoring the best window after at least 50,000 updates and 25 non-improving 1,000-update windows. Parameter stability is diagnostic only; the update cap is 1,000,000.
- `verify_…py` independently reconstructs raw selection, priors, shared-gradient sums, SciPy convolution, normalization and posterior/checkpoint restoration. Use `--pre-fit` before fitting.
- `run_…_pipeline.py` benchmarks both complete fits, selects concurrent/sequential execution by measured memory, then runs predictions, parameter plots and independent verification.
- `plot_…py`, `plot_…_parameters.py` and `verify_…_figures.py` generate/check four RTD/psychometric PNGs and a nine-square-panel black/red WT/HOM posterior comparison with a single-page PDF.
- `publish_…py` adds all five verified and visually inspected PNGs, source/audit downloads and in-sample comparisons against the 67-parameter reference to the November 7 result book.

Outputs: `shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl/`, separately under `wt/` and `hom/`. Each fit retains 10,000 posterior draws, losses, best/latest resumable states and provenance. RTDs use 20 ms data bins, 1 ms model evaluation, [-1,1] s normalization and [-0.1,0.6] s display. Trials, earlier fits and earlier figures remain unchanged. Posterior uncertainty is conditional on frozen proactive/lapse values; one initialization per genotype leaves initialization sensitivity untested.

`shank3_gaussian_batched_refinement.py` evaluates the same refined FFT grids in batches while retaining the original condition-wise direct-quadrature safeguard. `_batched_utils.py` and `fit_…_batched.py` select it for WT; HOM's completed original-kernel run stays unchanged. WT resumed update 33,000 with its exact guide/optimizer/RNG state. `source_migration.json`, `batched_refinement_validation.json`, `batched_svi_validation.json` and `source_history/WT_before_batched_refinement/` preserve the execution-only change and equivalence checks. Point likelihoods were identical, gradients agreed within 2e-9, and a full SVI update yielded identical loss and guide parameters.

Both ABL-delay fits stopped at 108,000 updates and restored their 83,000-update checkpoints after 25 confirmation windows. Versioned artifacts include posterior draws, loss traces, compact numerical/provenance exports and the published figures and audit ZIPs. The ZIPs preserve copied trial inputs, resumable states and source history; duplicate local caches, checkpoint pickles, smoke runs and exploratory previews remain outside this commit. Raw-data verification requires the local cleaned CSV and earlier reference fits.


## Archived 15-bound Gamma/Omega fits (2026-11-08)

Superseded in the result book. Use the current `_omega25` fits and comparisons described below; these older files are retained only for reproducibility.

SHANK3 cohort 1 WT/HOM fits, independently pooling five WT and six HOM animals using the exact saved 68,184/67,715 valid choices (session type 1, training 16, success ±1, measured RT in [0,1] s). Each genotype fits 68 parameters: 30 signed-condition Gamma, 30 signed-condition Omega, three Gaussian sensory-delay means and three SDs shared across ILDs within ABL, plus shared w and del_go. The NPL+alpha evidence mapping is not imposed. The same genotype-specific proactive/exponential-lapse values are frozen. Integrate zero-truncated delays before complete-mixture normalization at each exact planned onset.

Workflow prefix: `shank3_gamma_omega_gaussian_abl_delay`. `_utils.py` defines direct parameter sampling, signed priors, random starts, deterministic delay expansion and the Gaussian race; `prepare_…py` preserves inputs and records seed-0 initialization; `fit_…py` runs/resumes float64 full-rank GPU SVI and restores the absolute-best loss checkpoint. `verify_…py` checks raw selection, priors, SciPy convolution/likelihood, mapped NPL likelihood equivalence, gradients, normalization, stopping and posterior restoration. `plot_…py` makes RTDs/psychometrics and in-sample reference comparisons; `plot_…_parameters.py` makes 136-estimate, ten-square-panel WT-black/HOM-red comparisons; `verify_…_figures.py` independently checks figure arrays and intervals. `run_…_pipeline.py` benchmarks and orchestrates fitting/diagnostics, stopping before visual inspection; `publish_…py` publishes only after independent and actual visual checks.

Outputs: `shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl/`, separately under `wt/` and `hom/`. Every starting value is a fresh independent uniform prior-flat draw; seed 0 is reset so initial values match. No previous posterior is used to initialize these fits. Full-batch GPU SVI uses clipped Adam 0.0002, clip 1, guide scale 0.1, 1,000-update loss windows, minimum 50,000, then 25 windows without strict improvement; parameter stability is diagnostic only. The unconfirmed 1,000,000-update cap is inconclusive. Checkpoints, optimizer/RNG states, all losses, 10,000 posterior draws, counts, predictions, hashes and source audits are retained.

Diagnostic RTDs retain full [−1,1] s normalization, 20 ms empirical bins, 1 ms model evaluation, exact parent-onset weighting and [−0.1,0.6] s display. Counts remain 81,934 WT and 80,989 HOM observations including diagnostic-only pre-onset aborts. Psychometrics use individually normalized valid-trial onset predictions, with no sigmoid fit. The original publication has been retired from the result-book view; the current seven-figure publication uses the 25-bound fits below. Previous data/fits remain unchanged; uncertainty is conditional on fixed proactive/lapse values and a single initialization per genotype.

The direct Omega hard upper bound remains 15 s⁻¹, as specified; the Gaussian-by-ABL NPL+alpha reference derives values up to 22.68 (WT) and 20.86 (HOM). These are therefore not strictly nested parameter spaces. `posterior_boundary_audit.csv` records draw fractions within 1% of each hard-bound range; `reference_evidence_parameter_ranges.csv` records the reference ranges. Compare in-sample likelihoods without assuming the model with more parameters must improve them.

Both runs completed: WT stopped at 153,000 updates and restored 128,000; HOM stopped at 118,000 and restored 93,000. Each had 25 confirmation windows and 10,000 posterior draws. Independent mean log-likelihood discrepancies were 1.15e-9/trial (WT) and 2.17e-10/trial (HOM). All 136 parameter summaries, 30 RTD panels, 60 psychometric predictions, restart behavior, rendered figures, direct embeds/downloads and strict MkDocs build passed. The source data and 712 reference/data preservation hashes remain unchanged.


## Archived 15-bound Gaussian delay comparison (2026-11-08)

Superseded in the result book. Use the current `_omega25` fits and comparisons described below; these older files are retained only for reproducibility.

`plot_shank3_gaussian_abl_delay_model_comparison.py` reads completed 68-parameter direct and 13-parameter NPL+alpha fits, both with Gaussian delays shared across signed ILDs within ABL. It makes one 2×2 PNG/PDF: WT top/HOM bottom, underlying Gaussian μ left/σ right, black circles for direct Gamma/Omega and blue squares for NPL+alpha. Means and central 95% variational credible intervals are reconstructed from 10,000 draws, with equal square axes, no grids and shared y-limits per parameter column. No refitting or delay-moment transformation. `verify_shank3_gaussian_abl_delay_model_comparison.py` independently checks all 24 estimates and interval endpoints, identical retained populations/frozen values, unchanged source hashes and figure layout. Outputs: `shank3_cohort1_wt_hom_gaussian_delay_model_comparison/`, including point/mean-difference tables and provenance audits. This 15-bound figure is retired from the result-book view; use the `_omega25` comparison below.


## Archived 15-bound Gamma/Omega comparison (2026-11-08)

Superseded in the result book. Use the current `_omega25` fits and comparisons described below; these older files are retained only for reproducibility.

`plot_shank3_gamma_omega_model_comparison.py` compares the same completed 68-parameter direct and 13-parameter NPL+alpha fits with ABL-shared Gaussian delays. One 2×2 PNG/PDF has WT top/HOM bottom and Gamma left/Omega right. ABL-colored direct posterior means/CIs cover all 120 Gamma/Omega estimates; continuous NPL+alpha curves use the canonical float64 mapping at posterior-mean parameters on signed ILD −16:0.1:16. Gamma is ABL-invariant (one common black line); Omega curves are blue/orange/green for ABL 20/40/60. Dot x offsets ±0.18 dB are display-only. No refitting or curve uncertainty bands. `verify_shank3_gamma_omega_model_comparison.py` checks saved-draw intervals, all 3,852 mapped values using independent hyperbolic formulas, agreement at the 60 fitted conditions, source hashes, grid/units, source populations and figure mapping. Outputs: `shank3_cohort1_wt_hom_gamma_omega_model_comparison/`, with point/curve/parameter tables and source audits. This 15-bound figure is retired from the result-book view. Its archived direct Omega hard cap is 15 s⁻¹; the figure and caption expose its contrast with larger NPL+alpha-derived Omegas.

## Omega-upper-25 sensitivity refits (2026-11-08)

Separate 68-parameter WT/HOM fits in `shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl_omega25/` raise only Omega hard upper 15→25 s⁻¹; flat 0.1–12 and all other settings, exact trials, frozen values and physical seed-0 starts match the prior fits. Sources carry `_omega25`. This is the current result-book view; superseded 15-bound sections and figures have been removed. Seven PNGs are directly embedded in the November 8 result book: two RTD grids, two psychometrics, black/red WT/HOM parameter intervals, and the two 2×2 comparisons against unchanged NPL+alpha delays/evidence curves. The comparison folders end in `_omega25/`. Posterior draws, resumable checkpoints, independent numerical/resume/figure checks and before/after boundary-pressure and common-trial metric tables are saved. Existing fits, figures, sources and CSVs remain byte-identical. Parameter stability is diagnostic only.

### Reproducibility and backup for the current 25-bound fits

The current fit/diagnostic entrypoints use `shank3_gamma_omega_gaussian_abl_delay_omega25`; `verify_shank3_gamma_omega_omega25_resume.py` checks exact interrupted-run reproduction, and `verify_shank3_gamma_omega_omega25_sensitivity.py` checks the sole prior change, identical trials and initialization, all parameter changes and preservation of the reference fits. `publish_shank3_gamma_omega_omega25_comparisons.py` publishes only the current figures/tables and preserves old result-book anchors without redisplaying the retired results.

`scripts/backup_shank3_gaussian_omega25_to_drive_20261108.py` snapshots the full current fits, both current comparisons, historical 15-bound fits/comparisons, the unchanged NPL+alpha reference fits, WT/HOM proactive abort fits, sources and the current result-book publication. It uses `rclone copy`, validates every remote path, size and MD5 plus original-source SHA256, and saves its receipt/inventories under `logs/gdrive_shank3_gaussian_omega25_<timestamp>.*`. See `FIT_BACKUP_LEDGER.md` for the verified destination and counts.

Versioned numerical exports follow the existing Gaussian-fit convention: posterior draws/summaries, losses, convergence, condition/membership tables, model/data arrays and compact audits. Duplicate raw trial copies, fixed timing caches, checkpoint pickles, smoke runs and transfer logs remain local and in the complete Drive snapshot/audit downloads. Raw-data verification requires the local prepared CSV and restored reference folders. No fits are rerun during wrap-up.
