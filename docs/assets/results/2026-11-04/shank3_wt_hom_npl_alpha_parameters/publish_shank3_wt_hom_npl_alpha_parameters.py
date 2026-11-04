# %% Publish the independently and visually checked single composite
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_npl_alpha_comparison'
DATE = '2026-11-04'
TITLE = 'SHANK3 cohort 1 WT vs HOM: NPL+alpha posterior parameters'
STEM = 'SHANK3_cohort1_WT_HOM_NPL_alpha_parameter_comparison'
ASSETS = ROOT/'docs/assets/results'/DATE
AUDITS = ASSETS/'shank3_wt_hom_npl_alpha_parameters'
PAGE = ROOT/'docs/results'/f'{DATE}.md'
START,END = '<!-- shank3-wt-hom-npl-alpha-parameters:start -->','<!-- shank3-wt-hom-npl-alpha-parameters:end -->'
SOURCES = ['plot_shank3_wt_hom_npl_alpha_parameters.py','verify_shank3_wt_hom_npl_alpha_parameters.py',
           'publish_shank3_wt_hom_npl_alpha_parameters.py']
manifest = json.loads((OUT/'plot_manifest.json').read_text())
validation = json.loads((OUT/'independent_validation.json').read_text())
visual = json.loads((OUT/'visual_inspection.json').read_text())
assert validation['status']==visual['status']=='passed'
assert validation['artifacts']==visual['artifacts']==manifest['artifacts']
before = PAGE.read_text()
prior_sections = {group:before.split(f'<!-- shank3-{group}-npl-alpha:start -->',1)[1].split(f'<!-- shank3-{group}-npl-alpha:end -->',1)[0]
                  for group in ['wt','hom']}
caption = ('Black = Control (WT); red = ASD (HOM). Dots are posterior means and capped bars are central '
    '2.5th–97.5th percentile credible intervals from 10,000 saved draws per genotype. The first seven panels '
    'show shared fitted parameters; the bottom 1×3 row shows reactive delays separately at ABL 20/40/60 '
    'against the ten signed ILDs, without pooling signs. All time parameters are displayed in ms. '
    'Every plotting area is a 2.8-inch square; all delay panels share a y-scale. These are conditional '
    'variational-posterior intervals with proactive/exponential-lapse parameters fixed, rather than across-animal SEMs.')
if START not in before:
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'.agents/skills/update-result-book/scripts/update_result_book.py'),
        '--repo',str(ROOT),'--date',DATE,'--figure',str(OUT/f'{STEM}.png'),
        '--source',str(ROOT/'asd_analysis'/SOURCES[0]),'--title',TITLE,'--caption',caption],check=True)
AUDITS.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:
    path = OUT/f'{STEM}.{ext}'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['artifacts'][path.name]
    shutil.copy2(path,ASSETS/path.name)
for name in ['parameter_estimates.csv','plot_manifest.json','input_preservation.json','independent_validation.json','visual_inspection.json']:
    shutil.copy2(OUT/name,AUDITS/name)
for name in SOURCES:
    shutil.copy2(ROOT/'asd_analysis'/name,AUDITS/name)
archive = OUT/'shank3_wt_hom_npl_alpha_parameter_audits.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in ['parameter_estimates.csv','plot_manifest.json','input_preservation.json','independent_validation.json','visual_inspection.json']:
        z.write(OUT/name,name)
    for name in SOURCES:
        z.write(ROOT/'asd_analysis'/name,'source/asd_analysis/'+name)
shutil.copy2(archive,AUDITS/archive.name)
base = f'../assets/results/{DATE}/'
text = f'''{START}
## {TITLE}

![Control WT versus ASD HOM NPL+alpha posterior means and central 95% credible intervals, including a three-ABL delay row]({base}{STEM}.png)

*{caption}*

This single composite covers **all 37 fitted parameters per genotype (74 estimates)**: `rate_lambda`, `T_0`, `theta_E`, `rate_norm_l`, `alpha`, `w`, `del_go`, and 30 signed-condition `t_E_aff` delays. Control pools five WT animals and 68,184 valid choices; ASD pools six HOM animals and 67,715 valid choices. Both fits use session type 1, training level 16, success ±1, inclusive measured RT [0,1] s, ABL 59→60 and the existing stimulus selection and preparation exclusions. Actual trial counts determine the pooled fits. No fitting or new inference was performed for this figure.

The five proactive/exponential-lapse values were fixed in these fits and are not included among the 37 fitted parameters. The displayed intervals do not propagate uncertainty from those earlier abort fits. Delays remain distinct for positive/negative ILDs and use small horizontal genotype offsets for readability; the values and intervals are unchanged. `T_0`, `del_go` and `t_E_aff` are converted from seconds to milliseconds; other parameters are dimensionless.

[Full-size PNG]({base}{STEM}.png) · [Single-page PDF]({base}{STEM}.pdf) · [All 74 estimates (CSV)]({base}shank3_wt_hom_npl_alpha_parameters/parameter_estimates.csv) · [Numerical/source audit ZIP]({base}shank3_wt_hom_npl_alpha_parameters/{archive.name}) · [Independent validation]({base}shank3_wt_hom_npl_alpha_parameters/independent_validation.json)

Source: `asd_analysis/{SOURCES[0]}` ([download]({base}shank3_wt_hom_npl_alpha_parameters/{SOURCES[0]})); independent verifier: `asd_analysis/{SOURCES[1]}` ([download]({base}shank3_wt_hom_npl_alpha_parameters/{SOURCES[1]})). Output folder: `asd_analysis/shank3_cohort1_wt_hom_npl_alpha_comparison/`.

Sources are the completed [WT NPL+alpha fit](#shank3-cohort-1-wt-nplalpha-with-condition-specific-delays-and-fixed-exponential-lapses) and [HOM NPL+alpha fit](#shank3-cohort-1-hom-nplalpha-with-condition-specific-delays-and-fixed-exponential-lapses). The independent checks reconstruct all estimates from saved draws, verify signed-condition mappings, colors, actual error-bar endpoints, square geometry, unclipped limits and the single PDF page. Existing data, posterior files and diagnostic figures retain their recorded hashes.
{END}
'''
page = PAGE.read_text()
if START in page:
    first,rest = page.split(START,1)
    _,last = rest.split(END,1)
    PAGE.write_text(first+text+last)
else:
    first,rest = page.split('\n## '+TITLE,1)
    assert '\n## ' not in rest
    PAGE.write_text(first.rstrip()+'\n\n'+text)
after = PAGE.read_text()
for group,section in prior_sections.items():
    assert after.split(f'<!-- shank3-{group}-npl-alpha:start -->',1)[1].split(f'<!-- shank3-{group}-npl-alpha:end -->',1)[0]==section
index = ROOT/'docs/results/index.md'
label = f'- [{DATE}]({DATE}.md)'
description = ' — SHANK3 cohort 1 WT and HOM NPL+alpha fits, condition-specific delays and fixed exponential lapses; combined black/red posterior-parameter comparison.'
index.write_text('\n'.join(label+description if line.startswith(label) else line for line in index.read_text().splitlines())+'\n')
readme = ROOT/'asd_analysis/README.md'
heading = '## WT/HOM NPL+alpha posterior-parameter comparison (2026-11-04)'
if heading not in readme.read_text():
    with readme.open('a') as stream:
        stream.write(f'\n\n{heading}\n\n`{SOURCES[0]}` reads the two completed NPL+alpha seed-0 posteriors without refitting. One ten-panel PNG/PDF contains all 37 fitted parameters per genotype: seven shared-parameter panels and a 1×3 ABL delay row versus ten signed ILDs. Control/WT is black; ASD/HOM is red. All axes are equal 2.8-inch squares, with no grids. Error bars are central 95% credible intervals from 10,000 draws, conditional on fixed proactive/lapse values; times are shown in ms. Outputs and 74-estimate audit: `shank3_cohort1_wt_hom_npl_alpha_comparison/`. Verify with `{SOURCES[1]}`. The November 4 result book directly embeds the PNG and provides the PDF, source and audits.\n')
(OUT/'publication_manifest.json').write_text(json.dumps(dict(page=str(PAGE.relative_to(ROOT)),title=TITLE,
    png=str((ASSETS/f'{STEM}.png').relative_to(ROOT)),pdf=str((ASSETS/f'{STEM}.pdf').relative_to(ROOT)),
    prior_fit_sections_unchanged=True,published_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
print('Published composite with PDF, source and audits:',PAGE,flush=True)
