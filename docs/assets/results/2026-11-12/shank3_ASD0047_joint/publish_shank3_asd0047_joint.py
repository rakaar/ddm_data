# %% Publish only independently checked and visually inspected ASD0047 results
from datetime import datetime
import json
import shutil
import zipfile
from pathlib import Path
import pandas as pd

# %% Editable roots and result-book date
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'asd_analysis/shank3_cohort1_ASD0047_joint_npl_alpha_gaussian_abl_contaminants'
DAY = datetime.now().date().isoformat()
TITLE = 'SHANK3 cohort 1 ASD0047: joint proactive, NPL+alpha and contaminants'
validated = json.loads((BASE / 'deliverable_validation.json').read_text())
visual = json.loads((BASE / 'visual_inspection.json').read_text())
assert validated['status'] == visual['status'] == 'passed'
selected = json.loads((BASE / 'selected_fit.json').read_text())
assert selected['status'] == 'passed'
run = ROOT / selected['selected_run']
variant = run.parents[1]
report = variant / 'report'
exports = BASE / 'posterior_and_diagnostics'
comparison = BASE / 'joint_vs_two_stage_parameters'
assets = ROOT / 'docs/assets/results' / DAY / 'shank3_ASD0047_joint'
assets.mkdir(parents=True, exist_ok=True)
config = json.loads((run / 'configuration.json').read_text())
summary = pd.read_csv(run / 'posterior_summary.csv').set_index('parameter')
old_fixed = json.loads((ROOT / 'asd_analysis/shank3_cohort1_wt_hom_individual_adaptive_bounds/wt/ASD0047/fixed_proactive.json').read_text())
old_delay = old_fixed['values'][old_fixed['parameter_order'].index('t_nd')] * 1000
new_delay = summary.loc['t_nd', ['mean', 'q025', 'q975']].to_numpy(float) * 1000
decision = json.loads((BASE / 'contaminant_model_decision.json').read_text())
report_manifest = json.loads((report / 'report_manifest.json').read_text())
corners_manifest = json.loads((exports / 'export_manifest.json').read_text())
comparison_manifest = json.loads((comparison / 'plot_manifest.json').read_text())
files = [report / name for name in report_manifest['pngs']]
files += [comparison / name for name in comparison_manifest['figures']]
files += [exports / page['png'] for page in corners_manifest['corners']]
files += [Path(corners_manifest['corner_pdf']), Path(corners_manifest['diagnostic_pdf']),
          exports / 'SHANK3_cohort1_ASD0047_joint_abort_and_convergence.png']
audits = [BASE / 'selected_fit.json', BASE / 'contaminant_model_decision.json',
          BASE / 'deliverable_validation.json', BASE / 'visual_inspection.json',
          BASE / 'task_requirements.json', variant / 'preparation.json', variant / 'likelihood_policy.json',
          run / 'posterior_summary.csv', run / 'configuration.json',
          run.parent / 'selected_boundary_audit.csv', run.parent / 'all_start_boundary_audit.csv',
          run.parent / 'independent_validation.json', variant / 'exact_resume_validation.json',
          report / 'independent_model_prediction_checks.json', report / 'psychometric_probabilities.csv',
          report / 'rtd_window_masses.csv', comparison / 'posterior_parameter_comparison.csv',
          comparison / 'posterior_mean_changes.csv', comparison / 'independent_validation.json',
          exports / 'corner_parameter_summaries.csv', exports / 'posterior_correlations.csv',
          exports / 'all_start_parameter_estimates.csv', exports / 'proactive_delay_and_entry_summary.json',
          exports / 'abort_probability_audit.json', exports / 'censor_adjusted_abort_bins.csv']
sources = sorted((ROOT / 'asd_analysis').glob('*asd0047*.py'))
for source in files + audits + sources:
    name = source.name
    if source == run.parent / 'independent_validation.json':
        name = 'fit_independent_validation.json'
    elif source == comparison / 'independent_validation.json':
        name = 'comparison_independent_validation.json'
    shutil.copy2(source, assets / name)
archive = BASE / 'SHANK3_cohort1_ASD0047_joint_light_audits_and_sources.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
    for source in audits:
        zipped.write(source, 'audits/' + str(source.relative_to(BASE)))
    for source in sources:
        zipped.write(source, 'sources/' + source.name)
shutil.copy2(archive, assets / archive.name)
prefix = f'../assets/results/{DAY}/shank3_ASD0047_joint'
if selected['variant'] == 'exponential_only':
    lapse_note = (f'The initial exponential/uniform contaminant fit gave d={decision["d_mean"]:.4f} '
        f'[{decision["d_q025"]:.4f}, {decision["d_q975"]:.4f}]. Under the user-authorized exponential-tendency simplification, '
        'the model was refitted with **d=1**, removing the uniform branch. The selected reduced model fits '
        '**19 parameters jointly**. d is fixed, has no posterior interval and is omitted from corner axes. '
        'The tendency check includes a posterior mean ≥0.95, or upper-edge pressure with mode ≥0.95 and median ≥0.90. '
        'The mode/median extension was added after inspecting the first completed start, because a long lower tail can pull a bounded posterior mean away from its upper-edge peak. '
        'The broad mixture posterior does not rule out a uniform contribution. '
        'This is an exploratory nested simplification, not a predeclared model-selection or Bayes-factor test; the original mixture fit and its flags remain preserved. '
        'The reduced model uses both corresponding mixture-start posterior means as distinct warm starts, with fresh guides and optimizers.')
else:
    lapse_note = (f'The selected model retains the exponential/uniform contaminant mixture and fits **20 parameters jointly**. '
        f'The exponential weight is d={decision["d_mean"]:.4f} [{decision["d_q025"]:.4f}, {decision["d_q975"]:.4f}], '
        'without an exponential-edge tendency triggering the nested simplification. d is the exponential fraction within contaminants, '
        'not the overall lapse probability.')
reactive_png = next(name for name in comparison_manifest['figures'] if '_reactive_' in name)
proactive_png = next(name for name in comparison_manifest['figures'] if '_proactive_' in name)
focus_png = next(page['png'] for page in corners_manifest['corners'] if page['focus'] == 'proactive_and_contaminants')
full_corner = next(page['png'] for page in corners_manifest['corners'] if page['focus'] == 'all_parameters')
starts = pd.read_csv(exports / 'all_start_parameter_estimates.csv')
start_delays = starts.loc[starts.parameter.eq('t_nd')]
start_note = '; '.join(f'start {int(row.seed)}: {row.mean*1000:.2f} [{row.q025*1000:.2f}, {row.q975*1000:.2f}] ms'
                      for row in start_delays.itertuples(index=False))
entry = f'''
<!-- shank3-ASD0047-joint:start -->
## {TITLE}

This is **ASD0047 alone**, the SHANK3 cohort 1 WT animal whose earlier two-stage proactive delay was **{old_delay:.2f} ms**. Its proactive, reactive NPL+alpha and contaminant parameters are now fitted in **one joint likelihood**, without freezing the proactive or lapse estimates. There is no cross-animal pooling or hierarchy.

The new proactive delay is **{new_delay[0]:.2f} ms**, central 95% variational credible interval **[{new_delay[1]:.2f}, {new_delay[2]:.2f}] ms**. A change in this estimate does not by itself establish that the earlier value was caused by two-stage estimation: contaminant support/choice bias and likelihood conditioning also change, as described below.

**Initialization sensitivity:** proactive delay by completed start is {start_note}. Both starts and all parameter estimates remain available in [the start-by-start table]({prefix}/all_start_parameter_estimates.csv). The selected corner does not establish a unique solution. In particular, similar losses with separated delay posteriors indicate weak identification or different variational solutions; they must not be hidden by quoting only the selected interval.

### Model and exact population

The reactive model retains seven shared NPL+alpha/readout parameters and three underlying Gaussian μ/σ pairs, one per ABL 20/40/60 shared across all ten signed ILDs. Reactive trial delays are Gaussian truncated below zero, without an upper cutoff. The proactive delay alone may be negative. All shared parameter priors initially match this animal’s final two-stage priors, including proactive **t_nd ∈ [−3,0.3] s**; the old −700 ms value is not ruled out by the new prior. Any adaptive hard/flat prior changes are exported explicitly. No unresolved artificial-bound, physical-domain or prior-taper flag remains in the selected posterior.

{lapse_note}

| Observation | Count |
|---|---:|
| Valid RT [0,1] s choices | 18,771 |
| Right / left choices | 9,592 / 9,179 |
| Pre-onset aborts entering fitting | 3,004 |
| Late fixation-labelled onset-censored trials | 11 |
| Retained joint observations | 21,786 |
| Valid RT >1 s excluded | 161 |
| RTD observations: valid choices + aborts in [−1,0) s | 21,694 |

Session type 1, training level 16, success ±1 for valid choices, ABL 59→60 then 20/40/60, and signed ILDs ±1/2/4/8/16 remain unchanged. The same 18,771 valid trial identities and recorded values are independently matched to the prepared CSV and earlier animal fit. Prepared exclusions are retained without reapplying repeat filtering. All 3,004 pre-onset aborts enter fitting; no early-abort cutoff is imposed.

The complete joint mixture conditions on **no response before fixation** and the selected fixation-time window **[0,planned onset+1] s**, after Gaussian delay marginalization. Actual valid RT is planned onset plus recorded `timed_rt`. Late fixation-labelled rows retain onset-censoring semantics within that selected window. Contaminants use a common fixation-time **[0,3.3] s** support and a fitted right-choice probability. Abort choices are unobserved and summed out. There is no additional reactive lapse component.

Two starts use the old animal’s physical posterior means and a distinct near-zero proactive-delay initialization. Float64 full-batch full-rank variational inference uses 1,000-update mean-loss windows, minimum 50,000 updates, 25 subsequent windows without strict improvement and absolute-best checkpoint restoration. Parameter stability is diagnostic only. Each completed start retains 10,000 posterior draws and resumable states. The selected run is `{selected['selected_run']}`.

### Joint-fit RTDs and psychometric

![ASD0047 joint model and observed RT distributions]({prefix}/{report_manifest['pngs'][0]})

*3×5 ABL × |ILD| panels: 20 ms empirical steps above thin 0.9 pt black model curves evaluated at 1 ms. Data and model pool actual signed trials and are separately area-one normalized over **[−1,1] s**; only the display is zoomed to **−0.1–0.6 s**. Model predictions use each retained parent trial’s exact planned onset and normalization before pooling. The 81 aborts before −1 s and 11 onset-censored rows enter fitting but are not observed RTD points.*

![ASD0047 joint model and observed psychometric]({prefix}/{report_manifest['pngs'][1]})

*Measured P(right) dots and complete-mixture conditional model probabilities use the same fitted valid choices. ABL 20 blue, 40 orange, 60 green; actual numeric signed ILD positions. Predictions are normalized per exact onset over valid RT [0,1] s before trial-weighted averaging. Lines join measured ILDs; no sigmoid is fitted.*

### Two-stage versus joint parameter comparisons

![ASD0047 proactive and contaminant parameters, two-stage versus joint]({prefix}/{proactive_png})

![ASD0047 reactive parameters, two-stage versus joint]({prefix}/{reactive_png})

*Gray circles: this animal’s original two-stage fit; blue squares: the selected joint fit. Dots are posterior means and bars central 95% credible intervals from 10,000 saved draws. Open diamonds show fixed constants without intervals. The original right-lapse probability was fixed 0.5; it is estimated in the new joint fit. Underlying Gaussian μ/σ are shown in ms, with numeric ABL positions and additive −1/+1 dB offsets for readability. These are variational intervals, not across-animal SEMs. Two-stage reactive intervals omit uncertainty in the frozen proactive/lapse parameters; the joint fit estimates them together.*

**Comparison limits:** the original proactive fit used all 21,947 event-or-onset-censored trials, including the 161 long valid responses as onset-censored information. Its reactive stage used the same 18,771 valid choices with a complete-mixture valid-window normalization. It retained the original unconditioned proactive approximation and untruncated exponential with fixed right-choice probability 0.5. The new fit excludes those 161 long choices, uses entry and upper-window conditioning, and fits the contaminant right bias on finite support. Actual priors and any expansions are included in [the comparison table]({prefix}/posterior_parameter_comparison.csv). This is not a controlled experiment changing only estimation order.

### Corner plots and PDF downloads

![ASD0047 joint proactive and contaminant posterior corner]({prefix}/{focus_png})

*Every diagonal title gives the **parameter name and posterior mean**, with times in ms. Solid lines mark means, dashed lines central 95% limits; pairwise panels show 68%/95% joint contours. All 10,000 draws are retained inside padded display limits. These limits are not prior endpoints; actual hard/flat ranges are exported separately.*

- **[Diagnostics PDF, four pages]({prefix}/SHANK3_cohort1_ASD0047_joint_diagnostics.pdf):** 3×5 RTDs; psychometric; abort/convergence/parameter summary; proactive+contaminant corner.
- **[Full and focused posterior-corner PDF, two pages]({prefix}/SHANK3_cohort1_ASD0047_joint_posterior_corners.pdf)** · [Full-corner PNG]({prefix}/{full_corner}).
- [Abort and convergence PNG]({prefix}/SHANK3_cohort1_ASD0047_joint_abort_and_convergence.png) · [Posterior summaries]({prefix}/posterior_summary.csv) · [Selected bound audit]({prefix}/selected_boundary_audit.csv) · [Independent deliverable verification]({prefix}/deliverable_validation.json) · [Light audits and sources]({prefix}/{archive.name}).

The abort summary uses fixation-time coordinates and scales counts by all retained trials × bin width, so its area represents the abort fraction rather than one. Model predictions apply each exact onset censor and selection normalizer. Independent checks cover the likelihood, normalization, finite gradients, checkpoint restoration, loss-tail replay, reproducible restart, all posterior summaries/correlations, all 15 RTD histograms/unit areas, 30 psychometric probabilities and unchanged prior files. Every delivered PNG and PDF page has been visually inspected. These are in-sample diagnostics; variational convergence is not a guarantee of global identifiability or predictive validity.

**Visible residual:** the single-exponential model underpredicts the earliest fixation-abort peak and smooths the subsequent dip. A similar integrated abort fraction does not imply that it captures that early-time shape exactly. The observed data remain visible in the diagnostics PDF.

Sources: `asd_analysis/run_shank3_asd0047_joint.py`, `asd_analysis/plot_shank3_asd0047_joint.py`, `asd_analysis/plot_shank3_asd0047_joint_comparison.py`, `asd_analysis/export_shank3_asd0047_joint_corners_and_report.py`.

Previous animal fits: [November 12 two-stage analysis](2026-11-12.md#shank3-cohort-1-wthom-individual-refits-with-audited-expanded-bounds).
<!-- shank3-ASD0047-joint:end -->
'''
page = ROOT / 'docs/results' / f'{DAY}.md'
previous = page.read_text() if page.exists() else f'# Results: {DAY}\n'
start_marker, end_marker = '<!-- shank3-ASD0047-joint:start -->', '<!-- shank3-ASD0047-joint:end -->'
if start_marker in previous:
    first, last = previous.index(start_marker), previous.index(end_marker) + len(end_marker)
    page.write_text(previous[:first] + entry.strip() + previous[last:])
else:
    page.write_text(previous.rstrip() + '\n' + entry)
index = ROOT / 'docs/results/index.md'
index_text = index.read_text()
date_link = f'- [{DAY}]({DAY}.md)'
if date_link not in index_text:
    index_text = index_text.rstrip() + '\n' + date_link + '\n'
nav = ROOT / 'mkdocs.yml'
nav_text = nav.read_text()
nav_line = f'      - "{DAY}": results/{DAY}.md'
if nav_line not in nav_text:
    nav.write_text(nav_text.replace('      - Overview: results/index.md', '      - Overview: results/index.md\n' + nav_line))
    assert nav_line in nav.read_text()
index.write_text(index_text)
readme = ROOT / 'asd_analysis/README.md'
heading = '## ASD0047 single-stage proactive/reactive/contaminant fit'
if heading not in readme.read_text():
    with readme.open('a') as stream:
        stream.write(f'\n\n{heading}\n\nThe isolated SHANK3 cohort 1 WT animal with the former −716 ms proactive delay is refitted in `shank3_cohort1_ASD0047_joint_npl_alpha_gaussian_abl_contaminants/`. All proactive, NPL+alpha, Gaussian ABL delay and contaminant parameters update jointly. The saved animal-specific common priors initially match the final two-stage fit, with adaptive hard/flat audits and an authorized pure-exponential nested refit if d dominates. Exact choices, no-repeat-recalculation, entry/upper-window conditioning and the 161 excluded long valid responses are documented in the {DAY} result book. The output includes full/focused posterior corners with parameter-and-mean diagonal titles, a four-page diagnostics PDF and separate reactive/proactive comparison PNGs. `task_requirements.json`, `deliverable_validation.json` and `visual_inspection.json` record scope and verification. The old fit and CSV remain protected.\n')
(BASE / 'publication_manifest.json').write_text(json.dumps(dict(status='published_pending_strict_build',
    date=DAY, title=TITLE, page=str(page.relative_to(ROOT)), assets=str(assets.relative_to(ROOT)),
    primary_image_embeds=5, selected_run=selected['selected_run']), indent=2) + '\n')
print(str(page), flush=True)
