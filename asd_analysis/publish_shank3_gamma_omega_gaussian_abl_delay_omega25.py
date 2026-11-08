# %% Publish completed and independently checked direct Gamma/Omega fits
from pathlib import Path
from datetime import datetime,timezone
import json
import shutil
import subprocess
import zipfile
import re
import pandas as pd
from shank3_gamma_omega_gaussian_abl_delay_omega25_utils import ROOT,OUT,sha256

DATE='2026-11-08'
TITLE='SHANK3 cohort 1 WT/HOM: Gamma/Omega refits with Omega upper bound 25'
PREFIX='shank3_gamma_omega_gaussian_abl_delay_omega25'
PAGE=ROOT/'docs/results'/f'{DATE}.md'
ASSETS=ROOT/'docs/assets/results'/DATE/PREFIX
START,END='<!-- shank3-direct-gaussian-abl-omega25:start -->','<!-- shank3-direct-gaussian-abl-omega25:end -->'
SCRIPTS=[PREFIX+'_utils.py','prepare_'+PREFIX+'.py','fit_'+PREFIX+'.py','verify_'+PREFIX+'.py',
    'plot_'+PREFIX+'.py','plot_'+PREFIX+'_parameters.py','verify_'+PREFIX+'_figures.py',
    'run_'+PREFIX+'_pipeline.py','publish_'+PREFIX+'.py']
ARTIFACTS=[]
for group in ['wt','hom']:
    checked=json.loads((OUT/group/'independent_validation.json').read_text())
    assert checked['status']=='passed' and checked['parameter_count']==68
    assert checked['verifier_sha256']==sha256(ROOT/'asd_analysis'/('verify_'+PREFIX+'.py'))
    for name,digest in json.loads((OUT/group/'plot_manifest.json').read_text())['artifacts'].items():
        assert sha256(OUT/group/name)==digest
        ARTIFACTS.append(OUT/group/name)
pm=json.loads((OUT/'parameter_plot_manifest.json').read_text())
assert pm['estimates']==136 and pm['panels']==10
for name,digest in pm['artifacts'].items():
    assert sha256(OUT/name)==digest
    ARTIFACTS.append(OUT/name)
visual=json.loads((OUT/'visual_inspection.json').read_text())
assert visual['status']=='passed'
assert visual['artifacts']=={str(p.relative_to(OUT)):sha256(p) for p in ARTIFACTS}
assert json.loads((OUT/'figure_validation.json').read_text())['status']=='passed'
assert json.loads((OUT/'resume_validation.json').read_text())['status']=='passed'
ASSETS.mkdir(parents=True,exist_ok=True)
for p in ARTIFACTS: shutil.copy2(p,ASSETS/p.name)
for group in ['wt','hom']:
    shutil.copy2(OUT/group/'in_sample_comparison.csv',ASSETS/f'{group}_in_sample_comparison.csv')
for name in SCRIPTS: shutil.copy2(ROOT/'asd_analysis'/name,ASSETS/name)
for name in ['parameter_estimates.csv','actual_truncated_delay_moments.csv','parameter_plot_manifest.json',
             'figure_validation.json','visual_inspection.json','resume_validation.json','posterior_boundary_audit.csv','reference_evidence_parameter_ranges.csv']:
    shutil.copy2(OUT/name,ASSETS/name)
archive=OUT/(PREFIX+'_audits.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for group in ['wt','hom']:
        for p in (OUT/group).rglob('*'):
            if p.is_file() and p.suffix!='.log' and not any('smoke' in x or x=='initial_preview' for x in p.parts):
                z.write(p,str(p.relative_to(OUT)))
    for name in SCRIPTS: z.write(ROOT/'asd_analysis'/name,'source/asd_analysis/'+name)
    for name in ['asd_analysis/shank3_npl_alpha_gaussian_utils.py','ASD_Human_Data/gaussian_delay_likelihood.py',
        'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py','asd_analysis/shank3_wt_npl_alpha_utils.py',
        'asd_analysis/shank3_wt_gamma_omega_utils.py','asd_analysis/proactive_lapse_no_led_utils.py',
        'fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py']:
        z.write(ROOT/name,'source/'+name)
    for name in ['parameter_estimates.csv','actual_truncated_delay_moments.csv','figure_validation.json','visual_inspection.json','resume_validation.json','posterior_boundary_audit.csv','reference_evidence_parameter_ranges.csv']:
        z.write(OUT/name,name)
with zipfile.ZipFile(archive) as z: assert z.testzip() is None
shutil.copy2(archive,ASSETS/archive.name)

# %% Selection, model differences, normalization and conditional uncertainty
base=f'../assets/results/{DATE}/{PREFIX}'
lines=[START,
    '<span id="shank3-cohort-1-wthom-condition-specific-gammaomega-with-gaussian-delays-by-abl"></span>',
    '<span id="shank3-cohort-1-wthom-gaussian-delay-comparison-between-gammaomega-and-nplalpha"></span>',
    '<span id="shank3-cohort-1-wthom-condition-specific-gammaomega-versus-continuous-nplalpha"></span>',
    '', '## '+TITLE,'',
    'WT and HOM are fitted independently, pooling actual trials across five WT or six HOM animals. '
    '**68 free parameters per genotype:** 30 Gamma, 30 Omega, three Gaussian delay means and three SDs, '
    'shared w and shared del_go. Gamma/Omega are fitted directly at each signed ABL–ILD condition; '
    'the NPL+alpha evidence mapping is not imposed. Only Gaussian μ/σ are shared across the ten signed '
    'ILDs within each ABL 20/40/60.','',
    '**Unchanged selection:** cleaned session type 1, training 16, success ±1, finite measured RT in inclusive '
    '[0,1] s, ABL 59→60, ABL 20/40/60 and signed ILDs ±1/2/4/8/16. WT: **68,184 choices '
    '(37,482 right; 30,702 left), five animals**. HOM: **67,715 choices (35,320 right; 32,395 left), six animals**. '
    'Saved identities match the earlier direct and NPL+alpha fits exactly; preparation/repeat filters are not reapplied.','',
    'The same five genotype-specific proactive/exponential-lapse posterior means remain frozen. Independent '
    'trial delays follow a Gaussian truncated below zero with no upper cutoff. μ/σ describe its underlying '
    'Gaussian. Delay marginalization precedes complete-mixture RT/choice normalization over [0,1] s for '
    'each exact planned onset, retaining proactive evidence readout. The fixation-referenced exponential '
    'lapse contributes equally to both choices; no new uniform or reactive-specific lapse is added.','',
    '| Parameter | Hard bounds | Flat interval |',
    '|---|---|---|',
    '| Gamma, positive ILD | −1 to 20 | 0 to 12 |',
    '| Gamma, negative ILD | −20 to 1 | −12 to 0 |',
    '| Omega (s⁻¹) | 0.01 to 25 | 0.1 to 12 |',
    '| Gaussian μ (ms) | 0 to 500 | 10 to 200 |',
    '| Gaussian σ (ms) | 5 to 300 | 10 to 150 |',
    '| w | 0 to 1 | 0.4 to 0.6 |',
    '| del_go (ms) | 0 to 200 | 20 to 199 |','',
    'Priors apply once per free parameter. All 68 starts are independent uniform draws within flat intervals; '
    'seed 0 is reset between groups so starting values match. Float64, full-batch full-rank NumPyro SVI on GPU; '
    'clipped Adam 0.0002, clip norm 1 and guide scale 0.1. Losses are checked in 1,000-update windows, '
    'with a 50,000 minimum, then 25 consecutive windows without strict improvement and absolute-best '
    'checkpoint restoration. Parameter stability remains diagnostic only. A limit hit without confirmation is inconclusive.','',
    '| Group | Final updates | Restored best | Confirmation windows | Status |',
    '|---|---:|---:|---:|---|']
for group in ['wt','hom']:
    run=json.loads((OUT/group/'seed_0/run_summary.json').read_text())
    lines.append(f"| {group.upper()} | {run['final_checked_step']:,} | {run['restored_best_step']:,} | {run['confirmation_windows']} | {run['status']} |")
for group in ['wt','hom']:
    stem=f'SHANK3_cohort1_{group.upper()}_gamma_omega_gaussian_by_abl_omega25'
    lines+=['','### '+group.upper()+' data and model','',
        f'![{group.upper()} direct Gamma/Omega Gaussian-delay RTDs]({base}/{stem}_RTDs.png)','',
        '*Three ABL rows × five |ILD| columns. Actual ±ILD trials and parent-weighted unnormalized model '
        'predictions are pooled before data/model area-one normalization over [−1,1] s. Display: −0.1–0.6 s. '
        '20 ms data bins; 1 ms model evaluation. ABL-colored 1.4 pt data steps are drawn above 0.9 pt black '
        'model curves. Model timing averages every exact eligible parent onset. WT shows 81,934 observations '
        '(68,184 valid + 13,750 pre-onset aborts); HOM shows 80,989 (67,715 + 13,274). Aborts are diagnostic only.*','',
        f'![{group.upper()} direct Gamma/Omega Gaussian-delay psychometric]({base}/{stem}_psychometric.png)','',
        '*Measured P(right) dots and ABL-colored model lines at the ten signed ILDs. Each fitted trial’s '
        'right-choice window probability is divided by its complete [0,1] s mass before averaging over '
        'exact onsets. No sigmoid or interpolation beyond the measured-condition joins is fitted.*','']
stem='SHANK3_cohort1_WT_HOM_Gamma_Omega_gaussian_ABL_delay_parameter_comparison'
lines+=['### Posterior parameter comparison','',f'![WT HOM direct Gamma/Omega posterior comparison]({base}/{stem}.png)','',
    '*WT black; HOM red. Ten equal square panels: three Gamma and three Omega panels by ABL, μ and σ '
    'versus ABL, w and del_go. All 136 means and central 95% credible intervals use 10,000 draws per genotype. '
    'Delays are in milliseconds, Omega in s⁻¹, Gamma and w dimensionless. No grids. Intervals represent '
    'conditional variational uncertainty with proactive/lapse values frozen, not across-animal SEM. '
    'Actual truncated-delay moments before valid-window selection are exported separately.*','',
    f'[Parameter PDF]({base}/{stem}.pdf) · [Parameter estimates]({base}/parameter_estimates.csv) · '
    f'[Actual truncated-delay moments]({base}/actual_truncated_delay_moments.csv) · '
    f'[Complete posterior/checkpoint/numerical/source audits]({base}/{archive.name}) · '
    f'[Figure verification]({base}/figure_validation.json) · [Posterior boundary audit]({base}/posterior_boundary_audit.csv) · '
    f'[Reference Gamma/Omega ranges]({base}/reference_evidence_parameter_ranges.csv)','',
    '### In-sample comparison on identical trials','',
    '| Group | Model | Parameters | Joint log likelihood | Psychometric RMSE | Mean max RT-CDF gap |',
    '|---|---|---:|---:|---:|---:|']
labels={92:'Direct Gamma/Omega; fixed condition delays',67:'NPL+alpha; Gaussian delays by condition',13:'NPL+alpha; Gaussian delays by ABL',68:'Direct Gamma/Omega; Gaussian delays by ABL'}
for group in ['wt','hom']:
    table=pd.read_csv(OUT/group/'in_sample_comparison.csv')
    for row in table.loc[table.parameters.isin(labels)].itertuples():
        lines.append(f'| {group.upper()} | {labels[row.parameters]} | {row.parameters} | {row.joint_log_likelihood:,.2f} | {row.psychometric_rmse:.5f} | {row.mean_rt_cdf_max_gap_1ms:.5f} |')
lines+=['','Likelihood sums identical fitted trials at physical posterior means. Psychometric RMSE and '
    'mean maximum RT-CDF gap weight signed conditions equally. These are in-sample diagnostics, not held-out '
    'model selection. The direct Omega prior has a hard upper bound of 25 s⁻¹, whereas the Gaussian-by-ABL NPL+alpha reference derives values up to 22.68 (WT) and 20.86 (HOM). This raises only the previous hard ceiling of 15 to 25 s⁻¹; the Omega flat prior remains 0.1–12 s⁻¹ and all other settings and seed-0 physical initial values are unchanged. Single-start optimization and variational approximation remain relevant. Posterior uncertainty is conditional on frozen proactive/lapse parameters; '
    'one random initialization per genotype leaves initialization sensitivity untested.','',
    f'[WT comparison]({base}/wt_in_sample_comparison.csv) · [HOM comparison]({base}/hom_in_sample_comparison.csv)','',
    'Sources: '+' · '.join(f'[{name}]({base}/{name})' for name in SCRIPTS)+'.','',
    'References: [Gaussian delays per condition](2026-11-05.md) · [Gaussian delays per ABL](2026-11-07.md) · '
    '[Earlier direct Gamma/Omega fits](2026-10-23.md).','',
    'Output: `asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl_omega25/`.',END,'']
block='\n'.join(lines)
page=PAGE.read_text() if PAGE.exists() else f'# Results: {DATE}\n'
# The 25-bound figures are the user-selected current view. Keep historical files,
# but do not redisplay the superseded 15-bound fits or their two comparisons.
page=re.sub(r'(?ms)^<!-- shank3-direct-gaussian-abl:start -->\n.*?^<!-- shank3-direct-gaussian-abl:end -->\n*','',page)
for old_title in [
    'SHANK3 cohort 1 WT/HOM: Gaussian delay comparison between Gamma/Omega and NPL+alpha',
    'SHANK3 cohort 1 WT/HOM: condition-specific Gamma/Omega versus continuous NPL+alpha']:
    page=re.sub(r'(?ms)^## '+re.escape(old_title)+r'\n.*?(?=^## |^<!-- shank3-direct-gaussian-abl-omega25:start -->|\Z)','',page)
if START in page:
    first,rest=page.split(START,1);_,last=rest.split(END,1)
    PAGE.write_text(first+block+last)
else: PAGE.write_text(page.rstrip()+'\n\n'+block)
index=ROOT/'docs/results/index.md'
text=index.read_text()
if f'[{DATE}]' not in text:
    index.write_text(text.rstrip()+f'\n- [{DATE}]({DATE}.md) — SHANK3 WT/HOM direct Gamma/Omega fits with ABL-specific Gaussian sensory delays.\n')
nav=ROOT/'mkdocs.yml'
text=nav.read_text()
if f'results/{DATE}.md' not in text:
    anchor='      - "2026-11-07": results/2026-11-07.md'
    assert anchor in text
    nav.write_text(text.replace(anchor,anchor+f'\n      - "{DATE}": results/{DATE}.md'))
for link in re.findall(r'\]\(([^)]+)\)',PAGE.read_text()):
    if '://' not in link: assert (PAGE.parent/link.split('#')[0]).exists(),link
subprocess.run([str(ROOT/'.venv/bin/python'),'-m','mkdocs','build','--strict'],cwd=ROOT,check=True)
report=dict(status='passed',page=str(PAGE.relative_to(ROOT)),title=TITLE,png_embeds=5,
    strict_mkdocs_build='passed',audit_zip_crc='passed',visual_inspection='passed',
    artifacts={str(p.relative_to(OUT)):sha256(p) for p in ARTIFACTS},published_utc=datetime.now(timezone.utc).isoformat())
(OUT/'publication_validation.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'completion.json').write_text(json.dumps({**report,'status':'published',
    'fits':{g:json.loads((OUT/g/'seed_0/run_summary.json').read_text()) for g in ['wt','hom']}},indent=2)+'\n')
print('Published five PNG embeds and the parameter PDF:',PAGE,flush=True)
