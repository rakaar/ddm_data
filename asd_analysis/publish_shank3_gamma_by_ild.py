# %% Publish this separate analysis only after numerical and visual verification
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile
from shank3_gamma_by_ild_utils import ROOT, OUT, sha256

DATE = '2026-10-23'
TITLE = 'SHANK3 cohort 1 WT vs HOM: Gamma shared across ABLs'
PAGE = ROOT / f'docs/results/{DATE}.md'
ASSETS = ROOT / f'docs/assets/results/{DATE}/shank3_cohort1_wt_hom_gamma_by_ild'
START, END = '<!-- shank3-gamma-by-ild:start -->', '<!-- shank3-gamma-by-ild:end -->'
SOURCES = [ROOT / 'asd_analysis' / name for name in [
    'prepare_shank3_gamma_by_ild.py', 'shank3_gamma_by_ild_utils.py', 'fit_shank3_gamma_by_ild.py',
    'verify_shank3_gamma_by_ild.py', 'verify_shank3_gamma_by_ild_smoke.py',
    'predict_shank3_gamma_by_ild.py', 'plot_shank3_gamma_by_ild.py',
    'verify_shank3_gamma_by_ild_diagnostics.py', 'run_shank3_gamma_by_ild_pipeline.py',
    'publish_shank3_gamma_by_ild.py', 'shank3_wt_gamma_omega_utils.py',
    'shank3_tied_omega_utils.py', 'shank3_tied_omega_prediction_utils.py', 'proactive_lapse_no_led_utils.py']]
SOURCES += [ROOT / 'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py']
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pending', action='store_true')
args = parser.parse_args()
for name in ['prefit_validation.json', 'smoke_validation.json']:
    assert json.loads((OUT / name).read_text())['status'] == 'passed'
ASSETS.mkdir(exist_ok=True)
base = f'../assets/results/{DATE}/shank3_cohort1_wt_hom_gamma_by_ild/'
reports = ['preparation.json', 'prefit_validation.json', 'smoke_validation.json',
           'condition_mapping.csv', 'preservation_baseline.json', 'gpu_status.json', 'launch_manifest.json']
if args.pending:
    status = '**Status: WT and HOM GPU fits are running.** Prefit likelihood, prior multiplicity, gradients, exact trial identities and interrupted/resumed checkpoint checks passed. The two final composites will be embedded after numerical verification and visual inspection; initial-value previews are not fitted results.'
    figures = ''
else:
    post = json.loads((OUT / 'postfit_validation.json').read_text())
    diagnostics = json.loads((OUT / 'diagnostic_validation.json').read_text())
    inspection = json.loads((OUT / 'figures/visual_inspection.json').read_text())
    manifest = json.loads((OUT / 'figures/plot_manifest.json').read_text())
    assert post['status'] == diagnostics['status'] == inspection['status'] == 'passed'
    assert not manifest['initial_preview'] and not diagnostics['initial_preview']
    assert diagnostics['figure_artifact_hashes'] == manifest['artifacts']
    for name, expected in manifest['artifacts'].items():
        assert sha256(OUT / 'figures' / name) == expected
        if name.endswith('.png'):
            assert inspection['figure_hashes'][name] == expected
        shutil.copy2(OUT / 'figures' / name, ASSETS / name)
    rows = []
    for group in ['wt', 'hom']:
        run = json.loads((OUT / group / 'gamma_by_ild/run_summary.json').read_text())
        rows.append(f'| {group.upper()} | {run["status"]} | {run["restored_best_step"]:,} | {run["final_checked_step"]:,} |')
    status = '''Both fits are complete under the requested loss-confirmation rule. Parameter stability is diagnostic only.

| Genotype | Status | Restored-best update | Final checked update |
|---|---|---:|---:|
''' + '\n'.join(rows)
    comparison_rows = []
    with (OUT / 'in_sample_comparison.csv').open() as stream:
        for row in csv.DictReader(stream):
            comparison_rows.append(f'| {row["group"].upper()} | {row["parameters"]} | '
                f'{float(row["joint_log_likelihood"]):,.2f} | {float(row["psychometric_rmse"]):.5f} | '
                f'{float(row["trial_weighted_rt_cdf_integrated_gap"]):.5f} |')
    comparison_table = '\n'.join(comparison_rows)
    stem = 'SHANK3_cohort1_WT_HOM_gamma_by_ild'
    figures = f'''
### RTDs, chronometrics and psychometrics

![WT/HOM behavior with Gamma shared across ABLs]({base}{stem}_behavior.png)

*Four rows: five WT RTDs, five HOM RTDs, three compact chronometrics, then separate WT/HOM psychometrics. ABL 20/40/60 are blue/orange/green. RTD model lines are solid, fully opaque, 1.2 pt and above the 1.4 pt empirical steps at alpha 0.5. RTDs display **−0.1 to 0.6 s**, retaining 20 ms empirical bins, 1 ms model evaluation and separate data/model normalization over the complete **[−1,1] s** window. The zoom is not renormalized. Data values are unchanged.*

[Behavior PNG]({base}{stem}_behavior.png) · [Behavior PDF]({base}{stem}_behavior.pdf)

### All reactive parameters

![WT/HOM shared Gamma and reactive parameter comparison]({base}{stem}_parameters.png)

*Five panels: ten signed-ILD Gammas, three ABL Omegas, one reactive delay, w and del_go. The Gamma panel is half its original width, retaining its height and limits for closer visual comparison. WT is blue; HOM is red. Dots are posterior means; capped bars are central 95% credible intervals from 10,000 saved draws. Gamma is shared across ABLs, with positive/negative ILDs independently fitted. Shared values appear once; delays are in milliseconds. These are conditional variational posterior intervals with the proactive/lapse values frozen, not animal-to-animal SEMs.*

[Parameter PNG]({base}{stem}_parameters.png) · [Parameter PDF]({base}{stem}_parameters.pdf)

### In-sample comparison with the 36-parameter reference

| Genotype | Parameters | Joint log likelihood | Psychometric RMSE | Trial-weighted RT-CDF integrated gap (s) |
|---|---:|---:|---:|---:|
{comparison_table}

The smaller model has slightly larger RT-CDF discrepancies and higher psychometric RMSE on the fitted trials, particularly for WT. Likelihoods and predictions use physical posterior means; empirical data, trial identities and fixed proactive/lapse values match across models.

[Same-trial in-sample comparison with the earlier 36-parameter Omega-by-ABL fits]({base}in_sample_comparison.csv): joint log likelihood, psychometric errors and RT-CDF discrepancies. This comparison does not test held-out predictions or initialization sensitivity.
'''
    reports += ['postfit_validation.json', 'diagnostic_validation.json', 'parameter_comparisons.csv',
                'signed_condition_means.csv', 'chronometric_points.csv', 'in_sample_comparison.csv']
    square = OUT / 'square_gamma_figures'
    if (square / 'validation.json').exists():
        checked = json.loads((square / 'validation.json').read_text())
        assert checked['status'] == 'passed'
        for name, expected in checked['artifact_hashes'].items():
            assert sha256(square / name) == expected
            shutil.copy2(square / name, ASSETS / name)
        shutil.copy2(square / 'validation.json', ASSETS / 'square_gamma_validation.json')
        alt = 'SHANK3_cohort1_WT_HOM_gamma_by_ild_parameters_square_gamma'
        extra = f'''### Square Gamma panel — alternative layout

![WT/HOM parameters with an exactly square Gamma panel]({base}{alt}.png)

*Alternative for visual comparison: Gamma's plotting area is 2.60 × 2.60 inches. All parameter estimates, intervals and axis limits are identical; the other four panels retain their layout. The preceding wider version remains available.*

[Square-panel PNG]({base}{alt}.png) · [Square-panel PDF]({base}{alt}.pdf) · [Layout verification]({base}square_gamma_validation.json)

'''
        figures = figures.replace('### In-sample comparison with the 36-parameter reference',
                                  extra + '### In-sample comparison with the 36-parameter reference')
    for name in ['visual_inspection.json', 'plot_manifest.json', 'plotted_series.json.gz']:
        shutil.copy2(OUT / 'figures' / name, ASSETS / name)
for name in reports:
    shutil.copy2(OUT / name, ASSETS / name)
for source in SOURCES:
    shutil.copy2(source, ASSETS / source.name)
archive_name = 'shank3_gamma_by_ild_prepared_audits.zip' if args.pending else 'shank3_gamma_by_ild_fit_audits.zip'
with zipfile.ZipFile(OUT / archive_name, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for source in SOURCES:
        archive.write(source, 'source/' + str(source.relative_to(ROOT)))
    for name in reports:
        archive.write(OUT / name, name)
    for group in ['wt', 'hom']:
        for path in (OUT / group).rglob('*'):
            if path.is_file() and path.suffix in ['.json', '.csv', '.npz', '.npy', '.gz', '.pkl']:
                if args.pending and str(path.relative_to(OUT)) not in json.loads((OUT / 'preparation.json').read_text())['copied_inputs']:
                    continue
                archive.write(path, str(path.relative_to(OUT)))
    if not args.pending and (OUT / 'square_gamma_figures/validation.json').exists():
        for path in (OUT / 'square_gamma_figures').iterdir():
            if path.is_file():archive.write(path, str(path.relative_to(OUT)))
shutil.copy2(OUT / archive_name, ASSETS / archive_name)
entry = f'''{START}
## {TITLE}

{status}

Fit WT and HOM independently: **10 Gamma values shared across ABLs at each signed ILD; three Omegas, one per ABL; one reactive delay; shared w and del_go = 16 free parameters per genotype**. Keep +ILD and −ILD independent. Deterministic expansion supplies the existing 30 ABL–ILD likelihood conditions; each prior is applied once per free parameter.

Reuse exactly **68,184 WT choices from five animals** (37,482 right / 30,702 left) and **67,715 HOM choices from six animals** (35,320 right / 32,395 left). Source: cleaned session-type-1 CSV, SHANK3 cohort 1, training 16, `success ∈ {{1,−1}}`, finite recorded `timed_rt` in inclusive **[0,1] s**, ABL 59→60 then 20/40/60, signed ILDs ±1/2/4/8/16. Do not derive repeat or preparation exclusions again. No aborts enter these valid-trial fitting objectives.

Freeze the same five genotype-specific seed-0 proactive/exponential-lapse posterior means. Retain the complete-mixture likelihood, exact planned onsets and per-trial normalization of the joint RT/choice mixture over [0,1] s. Each fit pools actual trials across animals and sessions. WT and HOM do not share parameter values.

Both runs start fresh at seed 0: Gamma uses half the logit of measured P(right) after pooling choice counts across ABLs for each signed ILD; Omega uses trial-weighted data-derived condition initials within ABL. Delay starts at 0.05 s, w at 0.5 and del_go at 0.08 s. Float64 full-batch full-rank SVI, clipped Adam 0.0002, clip norm 1, guide scale 0.1. Stop after **minimum 50,000 updates, 1,000-update loss windows and 25 consecutive windows without a strictly lower loss**, restoring the absolute-best checkpoint. Maximum 1,000,000 updates; an unconfirmed limit hit is inconclusive. Parameter stability is diagnostic only; one start does not test initialization sensitivity.

RTD diagnostics retain **81,934 WT** and **80,989 HOM** displayed observations over [−1,1] s, including 13,750/13,274 strict pre-onset fixation aborts. RTD predictions use exact eligible-parent onset/sign weights. Chronometrics and psychometrics instead normalize predictions separately at each fitted valid trial's onset before averaging, using observed valid-trial frequencies. Chronometric dots are pooled-trial means; bars are sample SD across animal-condition means / √animal count, centered on the pooled dot. Pool ABLs within each animal before calculating the ABL-pooled SEM.

{figures}
### Sources and audits

Output folder: `asd_analysis/shank3_cohort1_wt_hom_gamma_by_ild/`. Sources: [fit driver]({base}fit_shank3_gamma_by_ild.py), [shared-Gamma model]({base}shank3_gamma_by_ild_utils.py), [independent fit verifier]({base}verify_shank3_gamma_by_ild.py), [prediction source]({base}predict_shank3_gamma_by_ild.py), [plotting source]({base}plot_shank3_gamma_by_ild.py), [independent diagnostic verifier]({base}verify_shank3_gamma_by_ild_diagnostics.py). Exact source paths are `asd_analysis/` followed by those filenames.

[Prefit checks]({base}prefit_validation.json) · [Checkpoint/resume checks]({base}smoke_validation.json) · [Condition mapping]({base}condition_mapping.csv) · [Protected hashes]({base}preservation_baseline.json) · [Sources, trials, posteriors and audits ZIP]({base}{archive_name})

See the [earlier tied-Omega and shared-delay fits](#shank3-cohort-1-wt-vs-hom-tied-omega-and-shared-delay-refits). Previous fits, figures and CSVs are preserved.
{END}
'''
page = PAGE.read_text()
if START in page:
    before, rest = page.split(START, 1)
    _, after = rest.split(END, 1)
    PAGE.write_text(before + entry + after)
else:
    PAGE.write_text(page.rstrip() + '\n\n' + entry)
index_path = ROOT / 'docs/results/index.md'
index = index_path.read_text()
link = f'- [{TITLE}]({DATE}.md#shank3-cohort-1-wt-vs-hom-gamma-shared-across-abls)'
if link not in index:
    index_path.write_text(index.rstrip() + '\n' + link + '\n')
assert f'results/{DATE}.md' in (ROOT / 'mkdocs.yml').read_text()
for name, expected in json.loads((OUT / 'preservation_baseline.json').read_text()).items():
    assert sha256(ROOT / name) == expected, name
publication = dict(status='fits_running' if args.pending else 'published', section=TITLE,
    page=str(PAGE.relative_to(ROOT)), initial_previews_published=False,
    final_PNG_embeds=0 if args.pending else 2 + int((OUT / 'square_gamma_figures/validation.json').exists()),
    asset_hashes={str(p.relative_to(ROOT)):sha256(p) for p in ASSETS.iterdir() if p.is_file()})
(OUT / 'publication_manifest.json').write_text(json.dumps(publication, indent=2) + '\n')
print(publication['status'] + ': ' + TITLE, flush=True)
