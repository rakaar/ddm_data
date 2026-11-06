# %% Publish only completed, numerically verified and visually inspected fits
from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import subprocess
import zipfile
import pandas as pd
from shank3_npl_alpha_gaussian_abl_delay_utils import ROOT,OUT,sha256

DATE='2026-11-07'
TITLE='SHANK3 cohort 1 WT/HOM: NPL+alpha with Gaussian delays shared across ILDs'
PAGE=ROOT/'docs/results'/f'{DATE}.md'
ASSETS=ROOT/'docs/assets/results'/DATE/'shank3_npl_alpha_gaussian_abl_delay'
START,END='<!-- shank3-gaussian-abl-delay:start -->','<!-- shank3-gaussian-abl-delay:end -->'
PREFIX='shank3_npl_alpha_gaussian_abl_delay'
SCRIPTS=[PREFIX+'_utils.py','prepare_'+PREFIX+'.py','fit_'+PREFIX+'.py',
    'verify_'+PREFIX+'.py','plot_'+PREFIX+'.py','plot_'+PREFIX+'_parameters.py',
    'verify_'+PREFIX+'_figures.py','run_'+PREFIX+'_pipeline.py','publish_'+PREFIX+'.py']
SCRIPTS+=['shank3_gaussian_batched_refinement.py',PREFIX+'_batched_utils.py','fit_'+PREFIX+'_batched.py']
ARTIFACTS=[]
for group in ['wt','hom']:
    validation=json.loads((OUT/group/'independent_validation.json').read_text())
    assert validation['status']=='passed' and validation['parameter_count']==13
    assert validation['verifier_sha256']==sha256(ROOT/'asd_analysis'/('verify_'+PREFIX+'.py'))
    m=json.loads((OUT/group/'plot_manifest.json').read_text())
    for name,digest in m['artifacts'].items():
        assert sha256(OUT/group/name)==digest
        ARTIFACTS.append(OUT/group/name)
pm=json.loads((OUT/'parameter_plot_manifest.json').read_text())
assert pm['estimates']==26 and pm['panels']==9
for name,digest in pm['artifacts'].items():
    assert sha256(OUT/name)==digest
    ARTIFACTS.append(OUT/name)
visual=json.loads((OUT/'visual_inspection.json').read_text())
assert visual['status']=='passed'
assert visual['artifacts']=={str(p.relative_to(OUT)):sha256(p) for p in ARTIFACTS}
assert json.loads((OUT/'figure_validation.json').read_text())['status']=='passed'
ASSETS.mkdir(parents=True,exist_ok=True)
for p in ARTIFACTS: shutil.copy2(p,ASSETS/p.name)
for group in ['wt','hom']:
    shutil.copy2(OUT/group/'in_sample_comparison.csv',ASSETS/f'{group}_in_sample_comparison.csv')
for name in SCRIPTS: shutil.copy2(ROOT/'asd_analysis'/name,ASSETS/name)
for name in ['parameter_estimates.csv','actual_truncated_delay_moments.csv','parameter_plot_manifest.json',
             'figure_validation.json','visual_inspection.json','resume_validation.json',
             'source_migration.json','batched_refinement_validation.json','batched_svi_validation.json']:
    shutil.copy2(OUT/name,ASSETS/name)
archive=OUT/'shank3_npl_alpha_gaussian_abl_delay_audits.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for group in ['wt','hom']:
        for p in (OUT/group).rglob('*'):
            if p.is_file() and p.suffix!='.log' and 'validation_smoke' not in str(p) and 'initial_preview' not in p.parts:
                z.write(p,str(p.relative_to(OUT)))
    for name in SCRIPTS: z.write(ROOT/'asd_analysis'/name,'source/asd_analysis/'+name)
    for name in ['asd_analysis/shank3_npl_alpha_gaussian_utils.py','ASD_Human_Data/gaussian_delay_likelihood.py',
        'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py','asd_analysis/shank3_wt_npl_alpha_utils.py',
        'asd_analysis/shank3_wt_gamma_omega_utils.py','asd_analysis/proactive_lapse_no_led_utils.py',
        'fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py']:
        z.write(ROOT/name,'source/'+name)
    for name in ['parameter_estimates.csv','actual_truncated_delay_moments.csv','figure_validation.json',
                 'visual_inspection.json','resume_validation.json','source_migration.json',
                 'batched_refinement_validation.json','batched_svi_validation.json','batched_svi_test_checkpoint.pkl']:
        z.write(OUT/name,name)
    for p in (OUT/'source_history').rglob('*'):
        if p.is_file(): z.write(p,str(p.relative_to(OUT)))
shutil.copy2(archive,ASSETS/archive.name)

# %% Five direct PNG embeds, model interpretation, pooling and convergence
base=f'../assets/results/{DATE}/shank3_npl_alpha_gaussian_abl_delay'
lines=[START,'## '+TITLE,'',
    'WT and HOM are fitted independently, pooling observed trials across five WT or six HOM animals. '
    '**13 parameters per genotype:** seven shared NPL+alpha parameters and three Gaussian means plus '
    'three Gaussian SDs, one μ/σ pair per ABL 20/40/60. Each pair is shared across all ten signed ILDs. '
    'The evidence process retains all 30 signed ABL–ILD conditions; Gamma/Omega are derived from '
    'NPL+alpha rather than fitted independently.','',
    '**Selection is unchanged:** cleaned session type 1, training 16, success ±1, finite recorded RT '
    'in inclusive [0,1] s, ABL 59→60 then 20/40/60 and signed ILDs ±1/2/4/8/16. '
    'WT: **68,184 choices, 37,482 right and 30,702 left, five animals**. '
    'HOM: **67,715 choices, 35,320 right and 32,395 left, six animals**. '
    'No preparation/repeat exclusions are reapplied.','',
    'The same five genotype-specific proactive/exponential-lapse posterior means remain frozen. '
    'Trial delays are independent Gaussian draws truncated below zero, without an upper cutoff; '
    'μ/σ describe the underlying Gaussian. Delays are integrated before the complete joint RT/choice '
    'mixture is normalized over [0,1] s separately for each exact planned onset. The proactive readout '
    'and fixation-referenced exponential lapse are retained; no additional reactive or uniform lapse is added.','',
    'The seven shared priors are unchanged. Gaussian μ: hard 0–500 ms, flat 10–200 ms; '
    'σ: hard 5–300 ms, flat 10–150 ms. Each prior is applied once per free parameter. '
    'All 13 initial values are fresh uniform draws within flat prior intervals, resetting seed 0 '
    'between genotypes. One initialization leaves initialization sensitivity untested.','',
    'Float64 full-batch, full-rank NumPyro SVI; clipped Adam 0.0002, clip 1, guide scale 0.1. '
    'Loss windows contain 1,000 updates; minimum 50,000; stop after 25 consecutive windows without '
    'strict improvement, restoring the absolute-best checkpoint. Parameter stability is diagnostic only.','',
    'WT resumed its saved 33,000-update checkpoint using a batched FFT refinement implementation, '
    'with optimizer, guide, RNG, losses and stopping history preserved. The grids, refinement factors '
    'and direct quadrature are unchanged. Likelihood/gradient comparisons passed, and an entire '
    'SVI update produced identical loss and updated guide parameters. HOM completed with the original '
    'kernel and was preserved. Original source/configuration/checkpoints and equivalence checks are archived.','',
    '| Group | Final updates | Restored best | Confirmation windows | Status |',
    '|---|---:|---:|---:|---|']
for group in ['wt','hom']:
    run=json.loads((OUT/group/'seed_0/run_summary.json').read_text())
    lines.append(f"| {group.upper()} | {run['final_checked_step']:,} | {run['restored_best_step']:,} | {run['confirmation_windows']} | {run['status']} |")
for group in ['wt','hom']:
    stem=f'SHANK3_cohort1_{group.upper()}_npl_alpha_gaussian_by_abl'
    lines+=['','### '+group.upper()+' data and model','',
        f'![{group.upper()} data/model RTDs with Gaussian delays shared across ILDs]({base}/{stem}_RTDs.png)','',
        '*RTDs: 3 ABL rows × 5 |ILD| columns; 20 ms empirical bins and 1 ms model curves. '
        'Data and model normalize separately over [−1,1] s; display is −0.1–0.6 s. Actual ±ILD '
        'trials and parent-weighted unnormalized model predictions are pooled before normalization. '
        'Each model uses every exact eligible parent onset. WT displays 81,934 observations '
        '(68,184 choices + 13,750 pre-onset aborts); HOM displays 80,989 (67,715 + 13,274). '
        'Aborts are diagnostic only. ABL-colored 1.4 pt data steps sit above thin 0.9 pt black model curves.*','',
        f'![{group.upper()} psychometric data/model with Gaussian delays shared across ILDs]({base}/{stem}_psychometric.png)','',
        '*Psychometrics: ABL-colored measured P(right) points and model lines at the ten signed ILDs. '
        'Right-choice density is integrated over [0,1] s and divided by each fitted trial’s complete '
        'window mass before averaging over exact onsets. No sigmoid fit is added.*','']
stem='SHANK3_cohort1_WT_HOM_NPL_alpha_gaussian_ABL_delay_parameter_comparison'
lines+=['### Posterior parameter comparison','',f'![WT HOM posterior parameter comparison]({base}/{stem}.png)','',
    '*WT black; HOM red. All 26 means and central 95% credible intervals come from 10,000 draws per '
    'genotype. Nine equal square panels show seven shared parameters, Gaussian μ versus ABL and '
    'Gaussian σ versus ABL. Time units are ms; no grids. Intervals describe conditional variational '
    'uncertainty with proactive/lapse estimates frozen, not across-animal SEM. Actual truncated-delay '
    'moments before RT-window selection are exported separately.*','',
    f'[Parameter PDF]({base}/{stem}.pdf) · [Parameter estimates]({base}/parameter_estimates.csv) · '
    f'[Actual truncated-delay moments]({base}/actual_truncated_delay_moments.csv) · '
    f'[Complete posterior/checkpoint/numerical/source audits]({base}/{archive.name}) · '
    f'[Figure verification]({base}/figure_validation.json)','',
    '### In-sample comparison with the 67-parameter Gaussian-delay reference','',
    '| Group | Model | Parameters | Joint log likelihood | Psychometric RMSE | Mean max RT-CDF gap |',
    '|---|---|---:|---:|---:|---:|']
for group in ['wt','hom']:
    table=pd.read_csv(OUT/group/'in_sample_comparison.csv')
    for row in table.loc[table.parameters.isin([67,13])].itertuples():
        label='Gaussian μ/σ per signed condition' if row.parameters==67 else 'Gaussian μ/σ per ABL'
        lines.append(f'| {group.upper()} | {label} | {row.parameters} | {row.joint_log_likelihood:,.2f} | {row.psychometric_rmse:.5f} | {row.mean_rt_cdf_max_gap_1ms:.5f} |')
lines+=['','Comparisons use identical trials, frozen values and physical posterior means. '
    'Likelihood sums trials; psychometric RMSE and mean maximum RT-CDF gap weight the 30 signed '
    'conditions equally. These are in-sample diagnostics, without a held-out model-selection claim.','',
    f'[WT numerical comparison]({base}/wt_in_sample_comparison.csv) · '
    f'[HOM numerical comparison]({base}/hom_in_sample_comparison.csv)','',
    'Sources: '+' · '.join(f'[{name}]({base}/{name})' for name in SCRIPTS)+'.','',
    'Output: `asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl/`. '
    'Reference: [November 5 Gaussian condition-delay fits](2026-11-05.md).',END,'']
page=PAGE.read_text()
block='\n'.join(lines)
if START in page:
    first,rest=page.split(START,1)
    _,last=rest.split(END,1)
    PAGE.write_text(first+block+last)
else:
    PAGE.write_text(page.rstrip()+'\n\n'+block)
readme=ROOT/'asd_analysis/README.md'
heading='## NPL+alpha Gaussian delays shared across ILDs (2026-11-07)'
if heading not in readme.read_text():
    with readme.open('a') as f:
        f.write('\n\n'+heading+'\n\n13 free parameters per genotype: seven shared NPL+alpha parameters plus three ABL-specific Gaussian means and three SDs. '
            'The exact previous WT/HOM valid-choice trials and fixed proactive/lapse values are reused. '
            'The original 67-parameter likelihood kernels operate on deterministically expanded delay arrays; priors apply once to each free parameter. '
            'Fresh seed-0 random starts, full-rank GPU SVI, strict 25-window loss confirmation and best restoration. '
            'Entrypoints use the `shank3_npl_alpha_gaussian_abl_delay` prefix: prepare, fit, plot, verify, parameter/figure checks, pipeline and publish scripts. '
            'Outputs: `shank3_cohort1_wt_hom_npl_alpha_gaussian_delay_by_abl/`. The November 7 result book directly embeds four RTD/psychometric PNGs and '
            'the nine-square-panel black/red posterior comparison, with its single-page PDF and complete source/audit downloads. '
            'Previous fits and numerical data remain unchanged.\n')
index=ROOT/'docs/results/index.md'
text=index.read_text()
line=next(x for x in text.splitlines() if x.startswith('- [2026-11-07]'))
if 'Gaussian delays shared across ILDs' not in line:
    index.write_text(text.replace(line,line+' WT/HOM NPL+alpha refits with Gaussian delays shared across ILDs.'))
(OUT/'publication_manifest.json').write_text(json.dumps(dict(page=str(PAGE.relative_to(ROOT)),title=TITLE,
    artifacts={str(p.relative_to(OUT)):sha256(p) for p in ARTIFACTS},published_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
subprocess.run([str(ROOT/'.venv/bin/python'),'-m','mkdocs','build','--strict'],cwd=ROOT,check=True)
(OUT/'completion.json').write_text(json.dumps(dict(status='published',date=DATE,title=TITLE,
    fits={g:json.loads((OUT/g/'seed_0/run_summary.json').read_text()) for g in ['wt','hom']},
    strict_mkdocs_build='passed',visual_inspection='passed'),indent=2)+'\n')
print('Published five direct PNG embeds, parameter PDF and audits:',PAGE,flush=True)
