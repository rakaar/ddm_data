# %%
"""Publish the requested human Gamma/Omega analysis on 20 October 2026."""
from pathlib import Path
import argparse
import json
import shutil
import zipfile
import numpy as np
import pandas as pd
from human_gamma_omega_svi_utils import ROOT, OUTPUT, PRIORS, sha256

# %% Requested date is explicit; do not replace it with the run's local date.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
args=parser.parse_args();out=args.output_dir
date='2026-10-20'
page=ROOT/f'docs/results/{date}.md'
assets=ROOT/f'docs/assets/results/{date}/human_gamma_omega_svi'
assets.mkdir(parents=True,exist_ok=True)
validation=json.loads((out/'independent_validation.json').read_text())
assert validation['status']=='passed'
assert json.loads((out/'visual_inspection.json').read_text())['status']=='passed'
heading='## Human RT: condition-wise Gamma–Omega–delay SVI, pooled Control and ASD'
old_page=page.read_text()
assert heading not in old_page,'This result is already published'
(out/'result_book_page_before.md').write_text(old_page)
plot=json.loads((out/'plot_manifest.json').read_text())
copy_names=plot['figures']+plot['tables']+['run_config.json','environment.json','independent_validation.json',
    'initial_independent_validation.json','visual_inspection.json','selection_counts.csv','plot_manifest.json','execution_schedule.json']
for name in copy_names: shutil.copy2(out/name,assets/name)
for group in ['control','asd']:
    dest=assets/group;dest.mkdir(exist_ok=True)
    for name in ['posterior_summary.csv','convergence_summary.json','convergence.csv','conditions.csv',
                 'data_manifest.json','initialization.json']:
        shutil.copy2(out/group/name,dest/name)
source_files=[
    'ASD_Human_Data/fit_human_gamma_omega_svi.py','ASD_Human_Data/human_gamma_omega_svi_utils.py',
    'ASD_Human_Data/plot_human_gamma_omega_svi.py','ASD_Human_Data/verify_human_gamma_omega_svi.py',
    'ASD_Human_Data/publish_human_gamma_omega_svi.py','ASD_Human_Data/human_plain_ddm_diagnostics_utils.py',
    'ASD_Human_Data/human_reactive_tied_svi_utils.py','ASD_Human_Data/prepare_simultaneous_rt_trials.py',
    'fit_each_condn/svi_gamma_omega_likelihood_utils.py',
    'fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py',
]
(assets/'source').mkdir(exist_ok=True)
for name in source_files: shutil.copy2(ROOT/name,assets/'source'/Path(name).name)
archive=assets/'human_gamma_omega_svi_audits.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(out.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='result_book_page_before.md':
            z.write(path,'analysis/'+str(path.relative_to(out)))
    for name in source_files:z.write(ROOT/name,'source/'+name)
    for group in ['control','asd']:
        for name in ['joint_fits.json','shared_scale_fits.json','fits.json']:
            z.write(Path(json.loads((out/'run_config.json').read_text())['initial_root'])/group/name,
                    f'initialization/{group}/{name}')
base=f'../assets/results/{date}/human_gamma_omega_svi'
lines=['',heading,'',
    '**Selection:** `Condition == "RT"`, `Valid == 1`, and **0 ≤ RT_s ≤ 1.8 s**. '
    '**Control: 6,574 trials from 11 participants; ASD: 4,770 trials from 8 participants.** '
    'All retained correct and incorrect choices are pooled as raw trials across participants and sessions. '
    'The four RTs below 200 ms are retained; 56 RTs above 1.8 s are excluded. SD and sequential CB are excluded.','',
    '**Model:** a reactive-only, descriptive two-bound DDM with **20 independent Gamma, 20 Omega, '
    '20 effective nondecision delays, and one shared fractional starting bias `w` per group (61 parameters)**. '
    'This replaces the earlier shared-bound/shared-delay restriction with condition-specific time scales and delays. '
    'There is no proactive process or `del_go`. Gamma is dimensionless, Omega is in s⁻¹. '
    '`Choice=1/-1` is right/left; `Delta_dB` is right minus left visual dB. ASL labels are 20/40 cd/m².','',
    'The likelihood is the observed bound-specific RT density divided by the probability of reaching **either** bound '
    'within 0–1.8 s. Predictions use the same normalization. Fits keep all signed conditions distinct; only displayed '
    'RTDs pool ±Δ using actual retained-trial frequencies. This is not an equal-participant average.','',
    '### Data and mean-parameter fits','']
for group in ['Control','ASD']:
    stem=f'human_{group.lower()}_gamma_omega_rtd_psychometric'
    lines += [f'![{group} RTDs by ASL and absolute difference, with fitted psychometric]({base}/{stem}.png)','',
        f'*{group}: ASL 20/40 occupy the two RTD rows; columns are |ΔdB| = 0.1, 0.3, 0.65, 1.3, 2. '
        'Steps are data in 20 ms bins; curves use arithmetic posterior means of physical Gamma, Omega, delay, and bias. '
        'Each RTD has unit area on the full 0–1.8 s window. The right panel shows empirical P(right) ± binomial SE '
        'and conditional fitted probabilities at the ten observed signed differences; connecting segments are visual guides.*','',
        f'[Full-size PNG]({base}/{stem}.png) · [PDF]({base}/{stem}.pdf)','']
lines += ['### Parameter comparison','',
    f'![Gamma, Omega, and delay by ASL and signed difference, plus shared bias]({base}/human_gamma_omega_parameter_comparison.png)','',
    '*Rows: Gamma, Omega, delay; columns: ASL 20/40. Control uses blue circles and ASD red stars. Points are posterior means '
    'and bars are equal-tailed **95% variational credible intervals**. Bias is shared across stimuli within each group. '
    'Small horizontal offsets separate group marks. These pooled-trial variational posteriors do not measure '
    'participant-to-participant variability.*','',
    f'[Full-size PNG]({base}/human_gamma_omega_parameter_comparison.png) · [PDF]({base}/human_gamma_omega_parameter_comparison.pdf) · '
    f'[All 122 parameter means, SDs, and intervals]({base}/parameter_summary.csv)','',
    '### Fit quality on the identical retained trials','',
    '| Group / model | Parameters | Joint log likelihood | Psychometric RMSE | Mean signed-condition CDF gap | Maximum CDF gap |',
    '|---|---:|---:|---:|---:|---:|']
bench=pd.read_csv(out/'benchmark_comparison.csv')
for row in bench.itertuples():
    lines.append(f'| {row.Group}: {row.model} | {int(row.free_parameters)} | {row.joint_log_likelihood:.2f} | '
                 f'{row.psychometric_rmse:.4f} | {row.mean_condition_cdf_gap:.4f} | {row.max_condition_cdf_gap:.4f} |')
changes=[]
for group in ['Control','ASD']:
    old=bench.loc[bench.Group.eq(group)&bench.free_parameters.eq(42)].iloc[0]
    new=bench.loc[bench.Group.eq(group)&bench.free_parameters.eq(61)].iloc[0]
    changes.append(f'**{group}: {old.mean_condition_cdf_gap:.4f} → {new.mean_condition_cdf_gap:.4f}**')
lines += ['', 'Allowing condition-specific delays changes the mean signed-condition RT CDF gap: '+ '; '.join(changes)+'. '
          'RTD agreement improves, but some fitted curves still underestimate the sharp empirical peaks. '
          'The remaining discrepancies are visible in the plots and maximum-gap column.']
lines += ['',
    'Likelihoods use the same rows and normalization. More flexible fits have more parameters; these in-sample '
    'scores do not establish predictive superiority. The earlier rows are maximum-likelihood diagnostics; '
    'the new SVI row is evaluated at arithmetic physical posterior means.','',
    f'[Fit-quality table]({base}/fit_quality.csv) · [Benchmark comparisons]({base}/benchmark_comparison.csv) · '
    f'[All signed-condition CDF gaps and probabilities]({base}/condition_fit_quality.csv)','',
    '### Priors, initialization and convergence','',
    '| Parameter | Hard interval | Flat interval of trapezoidal prior |','|---|---:|---:|']
for name,p in PRIORS.items():lines.append(f'| `{name}` | {p["hard"]} | {p["plausible"]} |')
lines += ['',
    'Linear tails connect each flat region to its hard limits. Gamma favors the stimulus-consistent sign but permits '
    'small reversals. The true delay prior remains 0–300 ms; each guide transform alone is confined below its '
    'condition’s earliest RT. All 40 causal limits are exported. Initialization uses the same-trial joint '
    'Gamma/Omega MLE with its shared delay copied into all conditions (Control 102.8 ms, ASD 173.4 ms).','',
    'NumPyro AutoMultivariateNormal; float64 CPU, full batch, one ELBO particle, seed 0, one initialization per group. '
    'ClippedAdam learning rate 0.0002, gradient clip 1, guide initial scale 0.1. Minimum 50,000 and maximum '
    '1,000,000 updates. Any lower 1,000-update window mean resets patience; stopping requires 25 later windows '
    'and posterior-mean ranges below 0.1 posterior SD over the last three checkpoints for all 61 parameters. '
    'The selected best checkpoint is restored; 10,000 joint posterior samples are retained. '
    'ASD ran in parallel with Control’s continuation using separate CPU processes and checkpoints.','',
    '| Group | Final updates | Restored best | Later windows | Terminal max shift (SD) | Best checkpoint locally stable | Status |',
    '|---|---:|---:|---:|---:|---|---|']
for group in ['Control','ASD']:
    r=json.loads((out/group.lower()/'convergence_summary.json').read_text())
    status='**PROVISIONAL / convergence unconfirmed**' if r['flagged'] else 'Confirmation and terminal stability passed'
    lines.append(f'| {group} | {r["completed_steps"]:,} | {r["restored_best_step"]:,} | {r["confirmation_windows"]} | '
                 f'{r["maximum_terminal_mean_change_sd"]:.3f} | {r["restored_checkpoint_parameter_stable"]} | {status} |')
lines += ['',
    f'![SVI loss confirmation and physical posterior correlation matrices]({base}/human_gamma_omega_convergence_correlations.png)','',
    'The mean-stability stopping gate applies to the final three checkpoints. The restored best checkpoint’s '
    'own local stability is recorded separately as a diagnostic. The checks establish observed-window confirmation, not a global minimum. '
    f'[Diagnostic PDF]({base}/human_gamma_omega_convergence_correlations.pdf) · '
    f'[Strongest posterior correlations]({base}/strongest_correlations.csv) · '
    f'[Hard-bound and causal-limit proximity]({base}/posterior_boundary_audit.csv)','',
    f'[Control conditions and causal limits]({base}/control/conditions.csv) · '
    f'[ASD conditions and causal limits]({base}/asd/conditions.csv) · '
    f'[Control convergence record]({base}/control/convergence_summary.json) · '
    f'[ASD convergence record]({base}/asd/convergence_summary.json) · '
    f'[Parallel execution record]({base}/execution_schedule.json)','',
    '### Boundary and dependence audit','']
boundaries=pd.read_csv(out/'posterior_boundary_audit.csv')
summary=pd.read_csv(out/'parameter_summary.csv')
boundaries=boundaries.merge(summary[['Group','parameter','ASL_cd','signed_delta']],on=['Group','parameter'],validate='one_to_one')
hard=np.maximum(boundaries.fraction_within_1pct_lower_hard,boundaries.fraction_within_1pct_upper_hard)
lines += [f'{int((hard>=.05).sum())} of 122 parameters place at least 5% of posterior draws within the outer 1% of a hard prior interval. '
          f'{int((boundaries.fraction_within_1pct_causal_upper.fillna(0)>=.05).sum())} condition delays meet that threshold near their causal upper limit. '
          'The support table separates prior limits from the data-implied causal limits.','',
          '| Group | Strongest physical posterior pair | Correlation |','|---|---|---:|']
correlations=pd.read_csv(out/'strongest_correlations.csv')
for group in ['Control','ASD']:
    for row in correlations.loc[correlations.Group.eq(group)].head(3).itertuples():
        lines.append(f'| {group} | `{row.parameter_1}` / `{row.parameter_2}` | {row.correlation:.3f} |')
lines += ['',
    'Parameter suffixes are zero-based condition indices listed in each group’s condition table. Strong correlations '
    'describe posterior trade-offs; they do not measure between-participant associations.','',
    '### Sources and audit','',f'Output root: `ASD_Human_Data/{out.name}/`.','']
for name in source_files[:5]:lines.append(f'- `{name}` ([source]({base}/source/{Path(name).name})).')
lines += ['',
    f'[Complete posterior/checkpoint/calculation/source audit ZIP]({base}/{archive.name}) · '
    f'[Run configuration and source hashes]({base}/run_config.json) · '
    f'[Independent verification]({base}/independent_validation.json) · '
    f'[Visual inspection]({base}/visual_inspection.json)','',
    'Verification independently reconstructs all retained MAT row identities, 40 psychometric probabilities, '
    '20 histograms and their signed mixtures, 122 posterior summaries, joint likelihoods, and CDF discrepancies. '
    'It checks true trapezoidal prior normalization, guide-only causal restrictions, finite gradients, '
    'K=40 versus K=80 accuracy, conditional unit mass, covariance/correlation matrices, and the stopping rule. '
    'Earlier human artifacts and unrelated modified tracked files are preserved.','']
page.write_text(old_page+'\n'.join(lines))
assert page.read_text().startswith(old_page)
index=ROOT/'docs/results/index.md'
old_index=index.read_text();new_index=[]
for line in old_index.splitlines():
    if line.startswith('- [2026-10-20]'):
        line+='; human condition-wise Gamma–Omega–delay RT + choice SVI'
    new_index.append(line)
index.write_text('\n'.join(new_index)+'\n')
assert 'results/2026-10-20.md' in (ROOT/'mkdocs.yml').read_text()
manifest=dict(result_page=str(page.relative_to(ROOT)),assets=str(assets.relative_to(ROOT)),
    requested_result_date=date,page_sha256=sha256(page),previous_page_sha256=sha256(out/'result_book_page_before.md'),
    prior_page_preserved_as_prefix=True,navigation='Existing 2026-10-20 entry verified',
    sources={p:sha256(ROOT/p) for p in source_files},archive_sha256=sha256(archive),
    figures={name:sha256(assets/name) for name in plot['figures']})
(out/'publication.json').write_text(json.dumps(manifest,indent=2))
print('Published',page)
