# %% Verified PNG publication; preserve previous date entries and fit artifacts
from pathlib import Path
import json
import shutil
import subprocess
import zipfile
import pandas as pd
from shank3_wt_npl_alpha_utils import ROOT, OUT, sha256

DATE = '2026-11-04'  # Date the approved analysis was started (local workstation).
TITLE = 'SHANK3 cohort 1 WT: NPL+alpha with condition-specific delays and fixed exponential lapses'
ASSETS = ROOT/'docs/assets/results'/DATE
AUDITS = ASSETS/'shank3_wt_npl_alpha'
PAGE = ROOT/'docs/results'/f'{DATE}.md'
START,END = '<!-- shank3-wt-npl-alpha:start -->','<!-- shank3-wt-npl-alpha:end -->'
STEMS = ['SHANK3_cohort1_WT_npl_alpha_RTDs','SHANK3_cohort1_WT_npl_alpha_psychometric']
SOURCES = ['shank3_wt_npl_alpha_utils.py','prepare_shank3_wt_npl_alpha.py','fit_shank3_wt_npl_alpha.py',
    'plot_shank3_wt_npl_alpha.py','verify_shank3_wt_npl_alpha.py','publish_shank3_wt_npl_alpha.py',
    'run_shank3_wt_npl_alpha_pipeline.py']
validation = json.loads((OUT/'independent_validation.json').read_text())
manifest = json.loads((OUT/'plot_manifest.json').read_text())
visual = json.loads((OUT/'visual_inspection.json').read_text())
assert validation['status']=='passed' and visual['status']=='passed'
assert validation['figure_artifact_hashes']==visual['figure_artifact_hashes']==manifest['artifacts']
for name,digest in manifest['artifacts'].items():
    assert sha256(OUT/name)==digest
run = json.loads((OUT/'seed_0/run_summary.json').read_text())
comparison = pd.read_csv(OUT/'in_sample_comparison.csv')
base = f'../assets/results/{DATE}/'
caption = ('SHANK3 cohort 1 WT, five animals pooled by observed trial counts. Colored steps are 20 ms data bins; '
    'thin black curves are the NPL+alpha proactive/reactive race with fixed exponential lapses, evaluated at 1 ms. '
    'Each panel pools actual signed ILDs before separately normalizing data and model over [-1,1] s; '
    'the displayed range is -0.1 to 0.6 s. The 81,934 displayed observations include 68,184 fitted valid choices '
    'and 13,750 diagnostic-only pre-onset fixation aborts. Model RTDs average all 83,921 parent trials at their exact planned onsets.')

# The result-book workflow helper registers the date, first image, index and nav.
if START not in (PAGE.read_text() if PAGE.exists() else ''):
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'.agents/skills/update-result-book/scripts/update_result_book.py'),
        '--repo',str(ROOT),'--date',DATE,'--figure',str(OUT/(STEMS[0]+'.png')),
        '--source',str(ROOT/'asd_analysis/plot_shank3_wt_npl_alpha.py'),'--title',TITLE,'--caption',caption],check=True)
ASSETS.mkdir(parents=True,exist_ok=True)
AUDITS.mkdir(exist_ok=True)
for stem in STEMS:
    shutil.copy2(OUT/(stem+'.png'),ASSETS/(stem+'.png'))
manifest['status']='independently_and_visually_verified'
(OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name in ['preparation.json','fixed_proactive.json','membership.csv','conditions.csv','prefit_validation.json',
             'independent_validation.json','visual_inspection.json','plot_manifest.json','in_sample_comparison.csv',
             'rtd_panel_audit.csv','psychometric_predictions.csv','gpu_benchmark.json']:
    shutil.copy2(OUT/name,AUDITS/name)
for name in ['run_summary.json','configuration.json','posterior_summary.csv','convergence.csv']:
    shutil.copy2(OUT/'seed_0'/name,AUDITS/name)
for name in SOURCES:
    shutil.copy2(ROOT/'asd_analysis'/name,AUDITS/name)
source_dir = OUT/'source'
source_dir.mkdir(exist_ok=True)
for name in SOURCES:
    shutil.copy2(ROOT/'asd_analysis'/name,source_dir/name)
archive = OUT/'shank3_wt_npl_alpha_audits.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUT.rglob('*')):
        if any(part in path.parts for part in ['validation_smoke','validation_smoke_resumed','initial_preview','source']):
            continue
        if path.is_file() and path.suffix in ['.csv','.gz','.json','.npz','.npy'] and path.name not in ['completion.json','publication_manifest.json']:
            z.write(path,path.relative_to(OUT))
    dependencies = [ROOT/'asd_analysis'/name for name in SOURCES]
    dependencies += [ROOT/'asd_analysis/shank3_wt_gamma_omega_utils.py',ROOT/'asd_analysis/proactive_lapse_no_led_utils.py',
        ROOT/'fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py',ROOT/'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py']
    for path in dependencies:
        z.write(path,'source/'+str(path.relative_to(ROOT)))
shutil.copy2(archive,AUDITS/archive.name)
table = '\n'.join(f'| {r.model} | {int(r.parameters)} | {r.joint_log_likelihood:,.3f} | {r.psychometric_rmse:.5f} | {r.mean_rt_cdf_max_gap_1ms:.5f} |'
    for r in comparison.itertuples())
text = f'''{START}
## {TITLE}

Five WT animals from `SHANK3_cohort1.csv` are pooled by actual trial counts. The objective uses **68,184 choices** (37,482 right and 30,702 left), `success` in {{1,-1}}, finite recorded `timed_rt` in inclusive **[0,1] s**, session type 1, training level 16, ABL 20/40/60 after 59→60, and signed ILDs ±1/2/4/8/16. The 373 valid choices above 1 s are excluded. Prepared repeat, short-sound and calibration rules are preserved without applying them again.

The five WT seed-0 proactive/exponential-lapse posterior means remain fixed. This fit estimates **37 parameters**: seven shared NPL+alpha parameters and 30 independent signed-condition reactive delays. Gamma/Omega are determined by NPL+alpha rather than sampled separately. Each condition uses a fixed delay point. Exponential lapses start at fixation onset and choose left/right with equal probability; there is no additional uniform or reactive-specific lapse. The complete joint RT/choice mixture is normalized together over the valid window at each trial's exact onset, using absolute response time `intended_fix_time/1000 + timed_rt`.

### RT distributions: model and data

![WT NPL+alpha model and data: three ABL rows and five absolute ILD columns]({base}{STEMS[0]}.png)

*{caption} Negative abort RT is `fix_time/1000 - intended_fix_time/1000`; late fixation-labelled trials are not shown as pre-onset aborts. These aborts are diagnostic only and do not enter the reactive-fit objective. ABL 20 is blue, 40 orange and 60 green.*

### Psychometric: model and data

![WT NPL+alpha measured choices and model psychometric predictions]({base}{STEMS[1]}.png)

*Dots show measured P(right); lines join the model predictions at the ten signed ILDs, with ABL colors matching the RTDs. Each prediction integrates right-choice density over [0,1] s, divides by the corresponding trial's complete window probability, then averages over that condition's fitted-trial onset times. No additional sigmoid is fitted.*

### Fit and validation

GPU float64, full-batch, full-rank NumPyro SVI used one fresh seed-0 start, clipped Adam 0.0002 and 1,000-update windows. Status: **{run['status']}**; restored-best update **{run['restored_best_step']:,}**, final checked update **{run['final_checked_step']:,}**. The stopping rule requires at least 50,000 updates and 25 consecutive windows without a strictly lower mean loss; parameter stability is diagnostic only. Central 95% intervals describe conditional variational uncertainty with the proactive/lapse values frozen. One-start initialization sensitivity is untested.

The following comparison is **in-sample** on identical trials, using physical posterior means. Differences do not establish held-out performance.

| Model | Parameters | Joint log likelihood | Psychometric RMSE | Mean RT-CDF maximum gap |
|---|---:|---:|---:|---:|
{table}

Independent checks reproduce the selection, frozen values, NPL mapping, prior multiplicities, SciPy likelihood, normalization, finite gradients, loss resets, checkpoint restoration and posterior draws. All 15 RTD panels and 30 psychometric predictions are audited; source CSVs and previous fits retain their recorded hashes.

[RTD full-size PNG]({base}{STEMS[0]}.png) · [Psychometric full-size PNG]({base}{STEMS[1]}.png) · [Complete numerical audits and source]({base}shank3_wt_npl_alpha/{archive.name}) · [Parameter summaries]({base}shank3_wt_npl_alpha/posterior_summary.csv) · [Independent validation]({base}shank3_wt_npl_alpha/independent_validation.json) · [Convergence]({base}shank3_wt_npl_alpha/convergence.csv)

Source: `asd_analysis/plot_shank3_wt_npl_alpha.py` ([download]({base}shank3_wt_npl_alpha/plot_shank3_wt_npl_alpha.py)); [fitter]({base}shank3_wt_npl_alpha/fit_shank3_wt_npl_alpha.py), [likelihood adapter]({base}shank3_wt_npl_alpha/shank3_wt_npl_alpha_utils.py), [independent verifier]({base}shank3_wt_npl_alpha/verify_shank3_wt_npl_alpha.py). Outputs: `asd_analysis/shank3_cohort1_wt_npl_alpha_condition_delay/`.

Reference: [WT condition-specific Gamma/Omega fit](2026-10-22.md#shank3-cohort-1-wt-shared-versus-condition-specific-reactive-delays).
{END}
'''
page = PAGE.read_text()
if START in page:
    before,rest = page.split(START,1)
    _,after = rest.split(END,1)
    PAGE.write_text(before+text+after)
else:
    # Replace only the newly appended helper entry, preserving earlier entries.
    before,after = page.split('\n## '+TITLE,1)
    assert '\n## ' not in after
    PAGE.write_text(before.rstrip()+'\n\n'+text)
index = ROOT/'docs/results/index.md'
label = f'- [{DATE}]({DATE}.md)'
index_text = index.read_text()
description = ' — SHANK3 cohort 1 WT NPL+alpha, condition-specific delays and fixed exponential lapses.'
if label+description not in index_text:
    index.write_text(index_text.replace(label,label+description))
readme = ROOT/'asd_analysis/README.md'
heading = f'## SHANK3 cohort 1 WT: NPL+alpha with condition-specific delays ({DATE})'
if heading not in readme.read_text():
    with readme.open('a') as stream:
        stream.write(f'\n\n{heading}\n\n37-parameter fit using the exact 68,184 WT choices from the existing condition-delay reference, with five proactive/exponential-lapse parameters frozen. No prepared filters are reapplied. Full-rank GPU SVI uses minimum 50k updates and a strict-improvement 25-window confirmation tail, without a parameter-stability gate. Output: `shank3_cohort1_wt_npl_alpha_condition_delay/`.\n\nScripts: `prepare_shank3_wt_npl_alpha.py`, `verify_shank3_wt_npl_alpha.py --pre-fit`, `run_shank3_wt_npl_alpha_pipeline.py`, and (after visual inspection) `publish_shank3_wt_npl_alpha.py`. Two PNGs: 3×5 RTDs (full [-1,1] normalization; [-0.1,0.6] display) and ABL-colored psychometric points/model lines. The result book includes direct embeds, posterior/audit/source downloads and an in-sample comparison against the 92-parameter Gamma/Omega reference.\n')
(OUT/'publication_manifest.json').write_text(json.dumps(dict(page=str(PAGE.relative_to(ROOT)),direct_png_embeds=2,
    title=TITLE,assets={str(p.relative_to(ROOT)):sha256(p) for p in ASSETS.rglob('*') if p.is_file()}),indent=2)+'\n')
print('Published two verified PNGs:',PAGE,flush=True)
