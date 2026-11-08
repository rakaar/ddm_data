# %% Verified publication settings
"""Publish the seven saved WT/HOM comparisons after numerical and visual checks."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'asd_analysis/shank3_cohort1_wt_hom_comparison'
DATE='2026-10-23'
ASSETS=ROOT/'docs/assets/results'/DATE/'shank3_cohort1_wt_hom_comparison'
PAGE=ROOT/'docs/results'/f'{DATE}.md'
TITLE='SHANK3 cohort 1 WT vs HOM: parameters and chronometrics'
START,END='<!-- shank3-wt-hom-comparison:start -->','<!-- shank3-wt-hom-comparison:end -->'
SOURCES=[ROOT/'asd_analysis'/name for name in ['plot_shank3_cohort1_wt_hom_comparison.py',
    'verify_shank3_cohort1_wt_hom_comparison.py','publish_shank3_cohort1_wt_hom_comparison.py',
    'shank3_wt_gamma_omega_utils.py','proactive_lapse_no_led_utils.py']]
SOURCES.append(ROOT/'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py')
manifest=json.loads((OUT/'plot_manifest.json').read_text())
validation=json.loads((OUT/'independent_validation.json').read_text())
inspection=json.loads((OUT/'visual_inspection.json').read_text())
assert validation['status']==inspection['status']=='passed'
assert validation['figure_artifact_hashes']==manifest['artifacts']
ASSETS.mkdir(parents=True,exist_ok=True)
for name,digest in manifest['artifacts'].items():
    assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    if name.endswith('.png'):
        assert inspection['figure_hashes'][name]==digest
    shutil.copy2(OUT/name,ASSETS/name)
manifest['status']='independently_verified_and_visually_inspected'
(OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for p in OUT.iterdir():
    if p.suffix in ['.csv','.json'] and p.name not in ['publication_manifest.json','completion.json']:
        shutil.copy2(p,ASSETS/p.name)
for p in SOURCES:
    shutil.copy2(p,ASSETS/p.name)
archive_name='shank3_cohort1_wt_hom_comparison_audits.zip'
with zipfile.ZipFile(OUT/archive_name,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(OUT.iterdir()):
        if p.suffix in ['.csv','.json'] and p.name not in ['publication_manifest.json','completion.json']:
            archive.write(p,p.name)
    for p in SOURCES:
        archive.write(p,'source/'+str(p.relative_to(ROOT)))
shutil.copy2(OUT/archive_name,ASSETS/archive_name)

# %% Seven direct embeds, captions, PDFs and source/audit links
base=f'../assets/results/{DATE}/shank3_cohort1_wt_hom_comparison/'
figures=[
    ('Gamma by signed ILD','SHANK3_cohort1_WT_HOM_gamma_comparison',
     'Dimensionless Gamma: three ABL panels, with WT blue and HOM red. Each signed condition is retained separately.'),
    ('Omega by signed ILD','SHANK3_cohort1_WT_HOM_omega_comparison',
     'Omega in s⁻¹: three ABL panels, with WT blue and HOM red.'),
    ('Reactive delay by signed ILD','SHANK3_cohort1_WT_HOM_t_E_aff_comparison',
     'Condition-specific reactive delay (t_E_aff / del_E) in milliseconds: WT blue and HOM red.'),
    ('Shared starting bias and readout delay','SHANK3_cohort1_WT_HOM_shared_parameter_comparison',
     'Shared w is dimensionless; shared del_go is in milliseconds. Each genotype contributes one posterior mean and central 95% credible interval per panel.'),
    ('WT chronometric: ABLs separate','SHANK3_cohort1_WT_ABL_chronometric',
     'Five WT animals and 68,184 fitted valid choices. ABL 20 blue, 40 orange, 60 green. Dots show pooled-trial mean RT ± SEM across animal-condition means; 1.0 pt lines show the conditional model mean.'),
    ('HOM chronometric: ABLs separate','SHANK3_cohort1_HOM_ABL_chronometric',
     'Six HOM animals and 67,715 fitted valid choices. ABL colors and mean/SEM calculations match the WT figure; axes are identical.'),
    ('WT vs HOM chronometric: ABLs pooled','SHANK3_cohort1_WT_HOM_ABL_pooled_chronometric',
     'WT blue and HOM red. Pool all retained ABL and signed-ILD trials within each |ILD|. For the error bars, first pool ABLs within each animal, then calculate SEM across the five WT or six HOM animal means. Model component weights use the same retained valid-trial counts.')]
images='\n\n'.join(f'''### {title}

![{title}]({base}{stem}.png)

*{caption}*

[Full-size PNG]({base}{stem}.png) · [PDF]({base}{stem}.pdf)''' for title,stem,caption in figures)
text=f'''{START}
## {TITLE}

Seven comparisons from the completed **condition-specific-delay fits**: four posterior-parameter figures and three chronometric figures. There is no refitting. WT has **68,184 fitted choices from five animals**; HOM has **67,715 choices from six animals**. HOM is the mutant group meant by “ASD” in the pooled comparison; this is a SHANK3 cohort-1 comparison.

Source fits: `asd_analysis/shank3_cohort1_wt_gamma_omega/condition_delay/` and `asd_analysis/shank3_cohort1_hom_gamma_omega/condition_delay/`. Reuse their saved 10,000-draw posteriors, restored-best checkpoints and genotype-specific frozen proactive/exponential-lapse means. The WT shared-delay fit is not used. See [WT fit details](2026-10-22.md#shank3-cohort-1-wt-shared-versus-condition-specific-reactive-delays) and [HOM fit details](#shank3-cohort-1-hom-condition-specific-reactive-delays) for the stopping histories and fit qualifications.

### Selection, weighting and uncertainty

Select the same cleaned session-type-1 population: SHANK3 cohort 1, training level 16, `success ∈ {{1,−1}}`, finite recorded `timed_rt` in inclusive **[0,1] s**, ABL 20/40/60 after 59→60, and signed ILDs ±1/2/4/8/16. Preserve preparation exclusions without refiltering repeats. Correct and error choices both contribute; abort observations do not enter these chronometric means.

**Parameter bars are central 95% posterior credible intervals**, the 2.5th–97.5th percentiles of the saved draws, around arithmetic posterior means. All **92 reactive parameters per genotype** are represented: 30 Gamma, 30 Omega, 30 reactive delays, shared w and shared del_go. The five frozen proactive/lapse parameters are excluded. Signed ILDs use categorical spacing and slight genotype offsets; each condition-dependent parameter image shares one y-range across its ABL panels, including all intervals. These are conditional variational uncertainties, not animal-to-animal variability or population-level confidence intervals; uncertainty in the frozen abort parameters is not propagated.

**Chronometric dots are pooled-trial arithmetic means.** Their capped bars use `sample SD(animal-condition means, ddof=1) / sqrt(animal count)`, with equal animal weights in the SEM calculation; the SD is calculated around the unweighted animal-mean average. Bars are centered on the pooled dot. Trials weight the dot, while animals have equal weight in its SEM. For pooled ABLs, calculate each animal's mean from all its retained ABL/±ILD trials before calculating this SEM.

**Model lines are complete-mixture means conditional on [0,1] s**, evaluated at arithmetic physical posterior means. Divide by each exact planned-onset trial's either-choice window mass before averaging. Pool signs and ABLs with observed fitted valid-trial counts, including the fixed proactive and exponential-lapse contributions. The model mean integrates `t × normalized density`; an independent adaptive SciPy calculation integrates conditional survival, and the saved conditional-CDF identity is checked as well. The earlier abort-plus-valid display RTDs are not the source of the chronometric means. Model lines have no uncertainty bands. All three chronometrics use compact square panels, |ILD| categories 1/2/4/8/16 and common **0–0.30 s** y-limits.

{images}

### Validation and downloads

Independent checks cover **184 parameter means/intervals**, **60 signed-condition model means**, **40 plotted data/model pairs and 40 available SEM bars**, **165 animal–ABL–|ILD| means**, and **55 ABL-pooled animal–|ILD| means**. Every pooled mean and SEM is reconstructed from raw retained trials; model weights reproduce the condition and pooled averages. All seven PNGs and single-page PDFs were visually/numerically checked. The maximum density-quadrature versus saved-CDF mean difference is **{manifest['max_density_vs_saved_CDF_mean_error_s']:.3g} s**, below the 10⁻⁶ s tolerance. Source CSVs and previous fits/figures retain their recorded hashes. Comparisons describe these pooled fits and are not a held-out or animal-level genotype-effect test.

[Parameter means and credible intervals]({base}parameter_comparisons.csv) · [Signed-condition model/data means]({base}signed_condition_means.csv) · [Chronometric points and SEMs]({base}chronometric_points.csv) · [Trial pooling weights]({base}pooling_weights.csv) · [Memberships]({base}membership.csv)

[Animal means by ABL]({base}animal_ABL_abs_ILD_means.csv) · [Animal means with ABLs pooled]({base}animal_ABL_pooled_abs_ILD_means.csv) · [Condition mapping]({base}condition_mapping.csv) · [Independent validation]({base}independent_validation.json) · [Visual inspection]({base}visual_inspection.json) · [Source preservation hashes]({base}preservation_baseline.json) · [Figure manifest]({base}plot_manifest.json) · [Calculation/source audit ZIP]({base}{archive_name})

Sources: [plotter]({base}plot_shank3_cohort1_wt_hom_comparison.py), [independent verifier]({base}verify_shank3_cohort1_wt_hom_comparison.py), [publisher]({base}publish_shank3_cohort1_wt_hom_comparison.py). Exact paths are `asd_analysis/` followed by those filenames. Derived outputs: `asd_analysis/shank3_cohort1_wt_hom_comparison/`. The archive includes source snapshots and independent-reference code; original posterior/trial downloads remain available through the two fit entries above.
{END}
'''
page=PAGE.read_text()
if START in page:
    before,rest=page.split(START,1);_,after=rest.split(END,1)
    PAGE.write_text(before+text+after)
else:
    PAGE.write_text(page.rstrip()+'\n\n'+text)
index_path=ROOT/'docs/results/index.md';index=index_path.read_text()
label=f'- [{TITLE}]({DATE}.md#shank3-cohort-1-wt-vs-hom-parameters-and-chronometrics)'
if label not in index:
    index_path.write_text(index.rstrip()+'\n'+label+'\n')
assert f'results/{DATE}.md' in (ROOT/'mkdocs.yml').read_text()
copied={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ASSETS.rglob('*') if p.is_file()}
(OUT/'publication_manifest.json').write_text(json.dumps({'page':str(PAGE.relative_to(ROOT)),
    'direct_PNG_embeds':7,'PDF_downloads':7,'copied_asset_hashes':copied},indent=2)+'\n')
print('Published seven direct PNG embeds and matching PDFs:',PAGE)
