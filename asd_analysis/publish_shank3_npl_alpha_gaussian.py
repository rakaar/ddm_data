# %% Publish the actual inspected Gaussian-delay fits; preserve old entries
from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import subprocess
import zipfile
import pandas as pd
from shank3_npl_alpha_gaussian_utils import ROOT, OUT, sha256

DATE='2026-11-05'
TITLE='SHANK3 cohort 1 WT/HOM: NPL+alpha with Gaussian condition delays'
PAGE=ROOT/'docs/results'/f'{DATE}.md'
ASSETS=ROOT/'docs/assets/results'/DATE/'shank3_npl_alpha_gaussian'
START,END='<!-- shank3-npl-alpha-gaussian:start -->','<!-- shank3-npl-alpha-gaussian:end -->'
SCRIPTS=['shank3_npl_alpha_gaussian_utils.py','prepare_shank3_npl_alpha_gaussian.py',
         'fit_shank3_npl_alpha_gaussian.py','verify_shank3_npl_alpha_gaussian.py',
         'plot_shank3_npl_alpha_gaussian.py','plot_shank3_npl_alpha_gaussian_parameters.py',
         'verify_shank3_npl_alpha_gaussian_figures.py',
         'run_shank3_npl_alpha_gaussian_pipeline.py','publish_shank3_npl_alpha_gaussian.py']
ARTIFACTS=[]
for group in ['wt','hom']:
    validation=json.loads((OUT/group/'independent_validation.json').read_text())
    assert validation['status']=='passed' and validation['verifier_sha256']==sha256(ROOT/'asd_analysis/verify_shank3_npl_alpha_gaussian.py')
    manifest=json.loads((OUT/group/'plot_manifest.json').read_text())
    for name,digest in manifest['artifacts'].items():
        assert sha256(OUT/group/name)==digest
        ARTIFACTS.append(OUT/group/name)
pm=json.loads((OUT/'parameter_plot_manifest.json').read_text())
for name,digest in pm['artifacts'].items():
    assert sha256(OUT/name)==digest
    ARTIFACTS.append(OUT/name)
visual=json.loads((OUT/'visual_inspection.json').read_text())
assert visual['status']=='passed'
assert visual['artifacts']=={str(p.relative_to(OUT)):sha256(p) for p in ARTIFACTS}
assert json.loads((OUT/'figure_validation.json').read_text())['status']=='passed'
if not PAGE.exists() or TITLE not in PAGE.read_text():
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'.agents/skills/update-result-book/scripts/update_result_book.py'),
        '--repo',str(ROOT),'--date',DATE,'--figure',str(ARTIFACTS[0]),
        '--source',str(ROOT/'asd_analysis/plot_shank3_npl_alpha_gaussian.py'),
        '--title',TITLE,'--caption','WT/HOM NPL+alpha with independently varying Gaussian condition delays.'],check=True)
ASSETS.mkdir(parents=True,exist_ok=True)
for path in ARTIFACTS: shutil.copy2(path,ASSETS/path.name)
for group in ['wt','hom']:
    shutil.copy2(OUT/group/'in_sample_comparison.csv',ASSETS/f'{group}_in_sample_comparison.csv')
for name in SCRIPTS: shutil.copy2(ROOT/'asd_analysis'/name,ASSETS/name)
for name in ['parameter_estimates.csv','actual_truncated_delay_moments.csv','parameter_plot_manifest.json',
             'figure_validation.json','visual_inspection.json','completion.json']:
    shutil.copy2(OUT/name,ASSETS/name)
archive=OUT/'shank3_npl_alpha_gaussian_audits.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for group in ['wt','hom']:
        for path in (OUT/group).rglob('*'):
            if path.is_file() and path.suffix!='.log' and 'validation_smoke' not in str(path) and 'initial_preview' not in path.parts:
                z.write(path,str(path.relative_to(OUT)))
    for name in SCRIPTS: z.write(ROOT/'asd_analysis'/name,'source/asd_analysis/'+name)
    for name in ['ASD_Human_Data/gaussian_delay_likelihood.py','ASD_Human_Data/human_plain_ddm_diagnostics_utils.py',
                 'asd_analysis/shank3_wt_npl_alpha_utils.py','asd_analysis/shank3_wt_gamma_omega_utils.py',
                 'asd_analysis/proactive_lapse_no_led_utils.py','fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py']:
        z.write(ROOT/name,'source/'+name)
    for name in ['parameter_estimates.csv','actual_truncated_delay_moments.csv','figure_validation.json','visual_inspection.json',
                 'resume_validation.json','source_migration.json','optimization_check.json']:
        z.write(OUT/name,name)
    for path in (OUT/'source_history').rglob('*'):
        if path.is_file(): z.write(path,str(path.relative_to(OUT)))
shutil.copy2(archive,ASSETS/archive.name)
base=f'../assets/results/{DATE}/shank3_npl_alpha_gaussian'
lines=[START,'## '+TITLE,'',
    'WT and HOM are fitted separately, pooling actual trial counts across five WT or six HOM animals. '
    'The model fits **67 parameters per genotype**: seven shared NPL+alpha parameters plus Gaussian μ/σ '
    'for each of 30 signed ABL–ILD conditions. Gamma/Omega follow the NPL+alpha mapping. Each trial draws '
    'an independent delay from its condition’s Normal(μ,σ²), truncated below zero with no upper cutoff. '
    'The Gaussian construction follows the human reference; the earlier rat reference used point delays.','',
    '**Selection:** cleaned session type 1, training level 16, success ±1, finite recorded RT in inclusive '
    '[0,1] s, ABL 59→60 then 20/40/60, and signed ILDs ±1/2/4/8/16. WT: **68,184 choices '
    '(37,482 right; 30,702 left)**. HOM: **67,715 (35,320 right; 32,395 left)**. Preparation exclusions '
    'are preserved without reapplying repeat eligibility.','',
    'The five genotype-specific proactive/exponential-lapse posterior means remain fixed. The exponential '
    'lapse starts at fixation onset and contributes half its density to each choice as an alternative '
    'to the full race. There is no added uniform or reactive-specific lapse. Random delays are integrated '
    'out **before** the complete mixture is normalized over the valid window for each exact onset. '
    'Valid absolute response time is planned onset plus recorded timed_rt.','',
    'Gaussian variability affects the reactive afferent delay and the proactive evidence readout. '
    'Proactive and exponential-lapse clocks retain their fixation-time definitions. The readout '
    'integrates the complete delay distribution, including its mass beyond the observed RT.','',
    'Delay marginalization uses a 0.5 ms grid through 1.2 s, adaptive FFT refinement and positive '
    'rescaled quadrature safeguards. Independent checks compare finer grids, longer series and SciPy '
    'quadrature. A verified execution optimization skips unused refinement checks; both fits resume '
    'from their 8,000-update checkpoints with optimizer/RNG states retained. Earlier source/configuration '
    'and checkpoints, plus likelihood/gradient equivalence checks, are included in the audits.','',
    '### Priors, initialization and convergence','',
    'Seven shared NPL+alpha trapezoidal priors are unchanged. Gaussian μ: hard 0–500 ms, flat 10–200 ms; '
    'σ: hard 5–300 ms, flat 10–150 ms. All 67 initial values are fresh independent uniform draws '
    'within prior flat regions, resetting NumPy seed 0 for each genotype. One start per genotype '
    'leaves initialization sensitivity untested.','',
    'Float64, full-batch, full-rank NumPyro SVI on RTX 4090; clipped Adam 0.0002, clip 1, guide scale 0.1. '
    'Loss windows contain 1,000 updates; minimum 50,000; stop after 25 windows without a strictly lower '
    'loss and restore the absolute-best checkpoint. Parameter stability is diagnostic only.','',
    '| Genotype | Final updates | Restored best | Confirmation windows | Status |',
    '|---|---:|---:|---:|---|']
for group in ['wt','hom']:
    run=json.loads((OUT/group/'seed_0/run_summary.json').read_text())
    lines.append(f"| {group.upper()} | {run['final_checked_step']:,} | {run['restored_best_step']:,} | {run['confirmation_windows']} | {run['status']} |")
for group in ['wt','hom']:
    lines+=['','### '+group.upper()+' model and data','']
    stem=f'SHANK3_cohort1_{group.upper()}_npl_alpha_gaussian'
    for kind in ['RTDs','psychometric']:
        lines += [f'![{group.upper()} Gaussian-delay NPL+alpha {kind}]({base}/{stem}_{kind}.png)','']
    lines += ['*RTDs use 20 ms data bins and 1 ms model evaluation, normalized separately over [−1,1] s; '
        'only the display is −0.1–0.6 s. They pool actual ±ILD trials and all exact eligible parent onset '
        'times, before normalization. WT displays 81,934 observations (68,184 choices + 13,750 pre-onset '
        'aborts); HOM displays 80,989 (67,715 + 13,274). Aborts do not enter the new fit. Data steps use '
        'ABL blue/orange/green and sit above a thin 0.9 pt black model. Psychometric dots are measured '
        'P(right); lines connect predictions at the ten measured signed ILDs, normalizing per valid-trial '
        'onset before averaging. No extra sigmoid is fitted.*','']
lines+=['### Posterior parameter comparison','',
    f'![WT HOM posterior comparison with Gaussian means and SDs]({base}/SHANK3_cohort1_WT_HOM_NPL_alpha_gaussian_parameter_comparison.png)','',
    '*Black Control/WT; red ASD/HOM. All 134 means and central 95% credible intervals come from 10,000 '
    'draws per genotype. Seven shared panels and two 1×3 ABL rows for μ and σ use 13 equally sized '
    'square axes without grids. Time parameters are shown in ms. μ/σ describe the underlying Gaussian; '
    'actual zero-truncated-delay moments before RT-window selection are exported separately. Bars describe conditional variational '
    'uncertainty with proactive/lapse parameters frozen, rather than animal-to-animal SEM.*','',
    f'[Parameter PDF]({base}/SHANK3_cohort1_WT_HOM_NPL_alpha_gaussian_parameter_comparison.pdf) · '
    f'[All parameter estimates]({base}/parameter_estimates.csv) · '
    f'[Actual truncated-delay moments]({base}/actual_truncated_delay_moments.csv) · '
    f'[Posterior/checkpoint/numerical/source audits]({base}/{archive.name}) · '
    f'[Figure verification]({base}/figure_validation.json)','',
    'Comparisons to the saved deterministic-delay NPL+alpha and 92-parameter Gamma/Omega fits use '
    'identical retained trials and physical posterior means. Likelihood, psychometric errors and '
    'RT-CDF discrepancies are **in-sample** diagnostics, without a model-selection or global-optimum claim.','',
    f'[WT in-sample comparison]({base}/wt_in_sample_comparison.csv) · '
    f'[HOM in-sample comparison]({base}/hom_in_sample_comparison.csv)','',
    'Sources: '+ ' · '.join(f'[{name}]({base}/{name})' for name in SCRIPTS) +'.','',
    'Output folder: `asd_analysis/shank3_cohort1_wt_hom_npl_alpha_gaussian_delay/`. '
    'See the [November 4 point-delay fits and parameter comparison](2026-11-04.md).',END,'']
position=next(i for i,line in enumerate(lines) if line.startswith('Sources:'))
comparison=['### In-sample comparison','',
    '| Genotype | Model | Parameters | Joint log likelihood | Psychometric RMSE | Mean max RT-CDF gap |',
    '|---|---|---:|---:|---:|---:|']
for group in ['wt','hom']:
    table=pd.read_csv(OUT/group/'in_sample_comparison.csv')
    for row in table.itertuples():
        label='Gamma/Omega point delays' if row.parameters==92 else ('NPL+alpha point delays' if row.parameters==37 else 'NPL+alpha Gaussian delays')
        comparison.append(f'| {group.upper()} | {label} | {row.parameters} | {row.joint_log_likelihood:,.2f} | {row.psychometric_rmse:.5f} | {row.mean_rt_cdf_max_gap_1ms:.5f} |')
comparison+=['','Compare models within genotype. Log likelihood sums all fitted trials; psychometric RMSE '
    'and mean maximum RT-CDF gap weight the 30 signed conditions equally. The Gaussian model improves '
    'these RT-CDF gaps and likelihoods in-sample; the direct Gamma/Omega reference retains the lowest '
    'psychometric RMSE.','']
lines[position:position]=comparison
page=PAGE.read_text()
block='\n'.join(lines)
if START in page:
    first,rest=page.split(START,1)
    _,last=rest.split(END,1)
    PAGE.write_text(first+block+last)
else:
    first,rest=page.split('\n## '+TITLE,1)
    assert '\n## ' not in rest
    PAGE.write_text(first.rstrip()+'\n\n'+block)
heading='## SHANK3 cohort 1 WT/HOM NPL+alpha Gaussian condition delays (2026-11-05)'
readme=ROOT/'asd_analysis/README.md'
if heading not in readme.read_text():
    with readme.open('a') as stream:
        stream.write('\n\n'+heading+'\n\n67 free parameters per genotype: seven shared NPL+alpha parameters and 30 Gaussian delay means/SDs. '
            'The zero-truncated trial delay is integrated before complete-mixture normalization, retaining frozen genotype-specific exponential lapses. '
            'Fresh seed-0 random initialization within flat prior regions; loss-only confirmation and best restoration. '
            'Outputs: `shank3_cohort1_wt_hom_npl_alpha_gaussian_delay/`. Entrypoints: `prepare_shank3_npl_alpha_gaussian.py`, '
            '`run_shank3_npl_alpha_gaussian_pipeline.py`, and `publish_shank3_npl_alpha_gaussian.py`; the matching independent verifiers '
            'and plotters share the same analysis prefix. The November 5 result book embeds WT/HOM RTD/psychometric PNGs and the black/red 13-square-panel '
            'posterior comparison, with its PDF and source/audit downloads. Existing point-delay fits remain untouched.\n')
index=ROOT/'docs/results/index.md'
label=f'- [{DATE}]({DATE}.md)'
index.write_text('\n'.join(label+' — SHANK3 cohort 1 WT/HOM NPL+alpha Gaussian condition-delay fits and diagnostics.' if line.startswith(label) else line for line in index.read_text().splitlines())+'\n')
(OUT/'publication_manifest.json').write_text(json.dumps(dict(page=str(PAGE.relative_to(ROOT)),title=TITLE,
    artifacts={str(p.relative_to(OUT)):sha256(p) for p in ARTIFACTS},published_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
subprocess.run([str(ROOT/'.venv/bin/python'),'-m','mkdocs','build','--strict'],cwd=ROOT,check=True)
(OUT/'completion.json').write_text(json.dumps(dict(status='published',date=DATE,title=TITLE,
    fits={g:json.loads((OUT/g/'seed_0/run_summary.json').read_text()) for g in ['wt','hom']},
    strict_mkdocs_build='passed',visual_inspection='passed'),indent=2)+'\n')
print('Published five direct PNG embeds, PDF and audits:',PAGE,flush=True)
