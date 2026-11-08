# %% Editable publication settings; leave all previous exports untouched
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_comparison'
OUT = COMPARISON / 'composites'
DATE = '2026-10-23'
ASSETS = ROOT / 'docs/assets/results' / DATE / 'shank3_cohort1_wt_hom_comparison/composites'
PAGE = ROOT / 'docs/results' / f'{DATE}.md'
START = '<!-- shank3-wt-hom-comparison:start -->'
END = '<!-- shank3-wt-hom-comparison:end -->'
SOURCES = [ROOT / 'asd_analysis' / f'{prefix}_shank3_cohort1_wt_hom_composites.py'
           for prefix in ['plot', 'verify', 'publish']]
PARAMETER_STEM = 'SHANK3_cohort1_WT_HOM_all_parameters'
BEHAVIOR_STEM = 'SHANK3_cohort1_WT_HOM_behavior_comparison'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


manifest = json.loads((OUT / 'plot_manifest.json').read_text())
validation = json.loads((OUT / 'independent_validation.json').read_text())
inspection = json.loads((OUT / 'visual_inspection.json').read_text())
assert validation['status'] == inspection['status'] == 'passed'
assert validation['figure_artifact_hashes'] == manifest['artifacts']
assert digest(SOURCES[1]) == validation['source_sha256']
for name, expected in manifest['artifacts'].items():
    assert digest(OUT / name) == expected
    if name.endswith('.png'):
        assert inspection['figure_hashes'][name] == expected
baseline = json.loads((OUT / 'preservation_baseline.json').read_text())
for name, expected in baseline.items():
    assert digest(ROOT / name) == expected, name

# %% Keep the old captions/downloads and replace only this result-book section
page = PAGE.read_text()
before, rest = page.split(START, 1)
current_section, after = rest.split(END, 1)
snapshot = OUT / 'previous_section.md'
if not snapshot.exists():
    snapshot.write_text(current_section)
old_section = snapshot.read_text()
prefix = old_section.split('### Gamma by signed ILD', 1)[0]
old_intro = prefix.split('\n\n')[1]
assert old_intro.startswith('Seven comparisons')
prefix = prefix.replace(old_intro,
    'Two combined overview figures from the completed **condition-specific-delay fits**: '
    'all reactive parameters in one image, and RTDs/chronometrics/psychometrics in a second image. '
    'There is no refitting or change to the saved numerical results. WT has **68,184 fitted choices '
    'from five animals**; HOM has **67,715 choices from six animals**. The chronometric comparison '
    'uses **HOM** (the earlier “HET” wording referred to these WT/HOM plots). '
    'The seven original individual figures remain available below.')
base = f'../assets/results/{DATE}/shank3_cohort1_wt_hom_comparison/'
composite_base = base + 'composites/'
old_figures = [
    ('Gamma', 'SHANK3_cohort1_WT_HOM_gamma_comparison'),
    ('Omega', 'SHANK3_cohort1_WT_HOM_omega_comparison'),
    ('Reactive delay', 'SHANK3_cohort1_WT_HOM_t_E_aff_comparison'),
    ('Shared w and del_go', 'SHANK3_cohort1_WT_HOM_shared_parameter_comparison'),
    ('WT chronometric', 'SHANK3_cohort1_WT_ABL_chronometric'),
    ('HOM chronometric', 'SHANK3_cohort1_HOM_ABL_chronometric'),
    ('WT/HOM ABL-pooled chronometric', 'SHANK3_cohort1_WT_HOM_ABL_pooled_chronometric')]
individual_downloads = '\n'.join(f'- {title}: [PNG]({base}{stem}.png) · [PDF]({base}{stem}.pdf)'
                                 for title, stem in old_figures)
old_validation = old_section.split('### Validation and downloads', 1)[1]
archive_name = 'shank3_cohort1_wt_hom_composite_audits.zip'
new_section = f'''{prefix}### Combined reactive-parameter comparison

![All WT/HOM reactive parameters: Gamma, Omega, reactive delay, w and del_go]({composite_base}{PARAMETER_STEM}.png)

*Eleven panels: Gamma/Omega/reactive-delay rows with ABL 20/40/60 columns, then shared w and del_go. WT is blue and HOM red. All 184 posterior means and central 95% credible intervals are unchanged. Signed ILDs remain separate; delays are displayed in milliseconds. Intervals describe conditional variational posterior uncertainty with proactive/lapse parameters fixed.*

[Full-size PNG]({composite_base}{PARAMETER_STEM}.png) · [Single-page PDF]({composite_base}{PARAMETER_STEM}.pdf)

### Combined RTDs, chronometrics and psychometrics

![WT and HOM: two five-panel RTD rows, three chronometrics, and two psychometrics]({composite_base}{BEHAVIOR_STEM}.png)

*Fifteen panels in four rows. (a) WT and (b) HOM each have five |ILD| panels (1/2/4/8/16 dB), overlaying ABL 20 blue, 40 orange and 60 green. Solid 1.4 pt steps are data; thin 0.8 pt dashed curves are the matching models. (c) Compact square chronometrics show WT by ABL, HOM by ABL, and ABL-pooled WT blue/HOM red, with unchanged mean/animal-SEM bars and 1.0 pt model lines. (d) Separate WT/HOM psychometrics retain measured points and model lines at the ten independently fitted signed conditions; no extra sigmoid is fitted.*

**RTD population differs from the valid-choice-only lower rows.** RTDs add pre-onset fixation aborts in [−1,0) to the fitted choices in [0,1]: **81,934 WT observations** (68,184 valid + 13,750 aborts) and **80,989 HOM observations** (67,715 valid + 13,274 aborts). Each genotype/ABL/|ILD| data and model curve retains its own area-one normalization over **[−1,1] s**, with 20 ms empirical bins and 1 ms model sampling. ABLs are overlaid, not combined. Models retain exact eligible-parent onset/sign weights; late fixation-labelled observations are not displayed as aborts. No aborts enter the chronometric or psychometric data. The lower rows use fitted-valid-trial weights and the per-trial conditional [0,1] s model predictions described above. Across-ABL RTD axes share one density scale; chronometric axes stay at **0–0.30 s**.

[Full-size PNG]({composite_base}{BEHAVIOR_STEM}.png) · [Single-page PDF]({composite_base}{BEHAVIOR_STEM}.pdf)

### Original individual-figure downloads

The original seven PNGs and PDFs are preserved byte-for-byte:

{individual_downloads}

### Validation and downloads

The composites independently verify **11/15 panels**, **184 posterior summaries**, **3,000 empirical RTD bins**, **60,030 model RTD samples**, **40 chronometric data/model pairs and animal SEMs**, and **60 psychometric data/model probabilities**. All displayed arrays match the original verified exports. Every RTD histogram is reconstructed from eligible parent trials, and all means/SEMs are checked against fitted valid trials. Exact model CDF/window masses retain unit area; the largest 1 ms sampled-density trapezoid error is **{validation['max_1ms_model_trapezoid_area_error']:.3g}**. Both PNGs were visually inspected, both PDFs have one page, and **{validation['preserved_files']} prior input/fit/export/asset hashes** are unchanged.

[Composite numerical verification]({composite_base}independent_validation.json) · [Visual inspection]({composite_base}visual_inspection.json) · [Actual plotted arrays]({composite_base}plotted_series.json.gz) · [Panel geometry and figure manifest]({composite_base}plot_manifest.json) · [Preservation hashes]({composite_base}preservation_baseline.json) · [Composite source/input/audit ZIP]({composite_base}{archive_name})

Composite sources: [plotter]({composite_base}{SOURCES[0].name}), [independent verifier]({composite_base}{SOURCES[1].name}), [publisher]({composite_base}{SOURCES[2].name}). Exact repository paths are `asd_analysis/` followed by these filenames; new outputs are under `asd_analysis/shank3_cohort1_wt_hom_comparison/composites/`.

The original calculation checks and downloads are retained below:
{old_validation}'''
PAGE.write_text(before + START + new_section + END + after)
rewritten = PAGE.read_text()
assert rewritten.split(START, 1)[0] == before
assert rewritten.split(END, 1)[1] == after
assert new_section.count('![') == 2

# %% Publication assets and reproducible source/input archive
manifest['status'] = 'independently_verified_and_visually_inspected'
(OUT / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
ASSETS.mkdir(parents=True, exist_ok=True)
asset_names = list(manifest['artifacts']) + ['plot_manifest.json', 'independent_validation.json',
    'visual_inspection.json', 'preservation_baseline.json', 'plotted_series.json.gz']
for name in asset_names:
    shutil.copy2(OUT / name, ASSETS / name)
for source in SOURCES:
    shutil.copy2(source, ASSETS / source.name)
archive_files = [(OUT / name, name) for name in asset_names if not name.endswith(('.png', '.pdf'))]
archive_files += [(snapshot, snapshot.name)]
archive_files += [(source, 'source/' + str(source.relative_to(ROOT))) for source in SOURCES]
archive_files += [(p, 'inputs/comparison/' + p.name) for p in sorted(COMPARISON.glob('*.csv'))]
for group in ['wt', 'hom']:
    folder = ROOT / f'asd_analysis/shank3_cohort1_{group}_gamma_omega'
    for name in ['rtd_data_bins.csv', 'condition_delay/pooled_rtd_model_1ms.csv.gz',
                 'condition_delay/pooled_rtd_model_bins.csv', 'condition_delay/psychometric_predictions.csv']:
        archive_files.append((folder / name, f'inputs/{group}/' + name))
with zipfile.ZipFile(OUT / archive_name, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path, name in archive_files:
        archive.write(path, name)
shutil.copy2(OUT / archive_name, ASSETS / archive_name)

index_path = ROOT / 'docs/results/index.md'
index = index_path.read_text()
for label, anchor in [
        ('WT–HOM: all reactive parameters', 'combined-reactive-parameter-comparison'),
        ('WT–HOM: RTDs, chronometrics and psychometrics', 'combined-rtds-chronometrics-and-psychometrics')]:
    link = f'- [{label}]({DATE}.md#{anchor})'
    if link not in index:
        index = index.rstrip() + '\n' + link + '\n'
index_path.write_text(index)
assert f'results/{DATE}.md' in (ROOT / 'mkdocs.yml').read_text()
for name, expected in baseline.items():
    assert digest(ROOT / name) == expected, name
publication = dict(page=str(PAGE.relative_to(ROOT)),
    section='SHANK3 cohort 1 WT vs HOM: parameters and chronometrics',
    direct_PNG_embeds=2, composite_PDF_downloads=2, preserved_original_figure_download_pairs=7,
    unrelated_result_book_sections_unchanged=True,
    copied_asset_hashes={str(p.relative_to(ROOT)): digest(p) for p in ASSETS.iterdir() if p.is_file()},
    archive_members=len(archive_files), archive_sha256=digest(OUT / archive_name))
(OUT / 'publication_manifest.json').write_text(json.dumps(publication, indent=2) + '\n')
print('Published two composite PNGs and PDFs in:', publication['section'])
