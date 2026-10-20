# %%
"""Publish the requested human Gamma/Omega analysis on 21 October 2026."""
from pathlib import Path
import argparse
import json
import re
import shutil
import zipfile
import numpy as np
import pandas as pd
from human_gamma_omega_delay500_svi_utils import ROOT, OUTPUT, PRIORS, sha256

# %% New sensitivity result uses the client's local date, 21 October 2026.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
parser.add_argument('--update-existing',action='store_true',help='Replace only this result section and its published assets')
args=parser.parse_args();out=args.output_dir
date='2026-10-21'
page=ROOT/f'docs/results/{date}.md'
assets=ROOT/f'docs/assets/results/{date}/human_gamma_omega_delay500_svi'
assets.mkdir(parents=True,exist_ok=True)
validation=json.loads((out/'independent_validation.json').read_text())
assert validation['status']=='passed'
assert json.loads((out/'visual_inspection.json').read_text())['status']=='passed'
heading='## Human RT: Gamma–Omega SVI with delay prior widened to 500 ms'
current_page=page.read_text();suffix=''
if heading in current_page:
    assert args.update_existing,'This result is already published; use --update-existing for a display revision'
    start=current_page.index(heading)
    following=re.search(r'\n## ',current_page[start+len(heading):])
    end=start+len(heading)+following.start() if following else len(current_page)
    old_page=current_page[:start];suffix=current_page[end:]
else:
    old_page=current_page
    (out/'result_book_page_before.md').write_text(old_page)
plot=json.loads((out/'plot_manifest.json').read_text())
comparison=json.loads((out/'delay_bound_comparison_manifest.json').read_text())
copy_names=plot['figures']+plot['tables']+comparison['figures']+comparison['tables']+[
    'run_config.json','environment.json','independent_validation.json','delay300_boundary_diagnosis.json',
    'delay_bound_comparison_manifest.json','source_variant_changes.diff',
    'visual_inspection.json','selection_counts.csv','plot_manifest.json','execution_schedule.json']
for name in copy_names: shutil.copy2(out/name,assets/name)
for group in ['control','asd']:
    dest=assets/group;dest.mkdir(exist_ok=True)
    for name in ['posterior_summary.csv','convergence_summary.json','convergence.csv','conditions.csv',
                 'data_manifest.json','initialization.json']:
        shutil.copy2(out/group/name,dest/name)
source_files=[
    'ASD_Human_Data/fit_human_gamma_omega_delay500_svi.py','ASD_Human_Data/human_gamma_omega_delay500_svi_utils.py',
    'ASD_Human_Data/plot_human_gamma_omega_delay500_svi.py','ASD_Human_Data/verify_human_gamma_omega_delay500_svi.py',
    'ASD_Human_Data/publish_human_gamma_omega_delay500_svi.py','ASD_Human_Data/human_plain_ddm_diagnostics_utils.py',
    'ASD_Human_Data/compare_human_gamma_omega_delay_bounds.py',
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
            if path.parent==out and path.name in ['publication.json','completion_audit.json']:
                continue
            z.write(path,'analysis/'+str(path.relative_to(out)))
    for name in source_files:z.write(ROOT/name,'source/'+name)
    for group in ['control','asd']:
        for name in ['joint_fits.json','shared_scale_fits.json','fits.json']:
            z.write(Path(json.loads((out/'run_config.json').read_text())['initial_root'])/group/name,
                    f'initialization/{group}/{name}')
    previous=ROOT/'ASD_Human_Data/gamma_omega_rt_choice_svi_0_1p8s_delay300ms'
    for name in ['run_config.json','parameter_summary.csv','fit_quality.csv','rtd_model_curves.csv','selected_trials.csv','excluded_trials.csv']:
        z.write(previous/name,f'comparison_baseline_300ms/{name}')
    for group in ['control','asd']:
        for name in ['posterior_samples.npz','conditions.csv','initialization.json','convergence_summary.json']:
            z.write(previous/group/name,f'comparison_baseline_300ms/{group}/{name}')
base=f'../assets/results/{date}/human_gamma_omega_delay500_svi'
lines=['',heading,'',
    '**Selection:** `Condition == "RT"`, `Valid == 1`, and **0 ≤ RT_s ≤ 1.8 s**. '
    '**Control: 6,574 trials from 11 participants; ASD: 4,770 trials from 8 participants.** '
    'All retained correct and incorrect choices are pooled as raw trials across participants and sessions. '
    'The four RTs below 200 ms are retained; 56 RTs above 1.8 s are excluded. SD and sequential CB are excluded.','',
    '**Model:** a reactive-only, descriptive two-bound DDM with **20 independent Gamma, 20 Omega, '
    '20 effective nondecision delays, and one shared fractional starting bias `w` per group (61 parameters)**. '
    'The model and all fit settings match the 300 ms fit; only the delay hard upper prior changes to 500 ms. '
    'There is no proactive process or `del_go`. Gamma is dimensionless, Omega is in s⁻¹. '
    '`Choice=1/-1` is right/left; `Delta_dB` is right minus left visual dB. ASL labels are 20/40 cd/m².','',
    'The likelihood is the observed bound-specific RT density divided by the probability of reaching **either** bound '
    'within 0–1.8 s. Predictions use the same normalization. Fits keep all signed conditions distinct; only displayed '
    'RTDs pool ±Δ using actual retained-trial frequencies. This is not an equal-participant average.','',
    '### Why widen the bound?','',
    'The 300 ms posteriors show upper 95% endpoints above 295 ms in **2 Control and 8 ASD conditions**, '
    'reaching 298.95 and 298.87 ms. The largest posterior means were 292.26 and 290.05 ms. '
    'This is pressure near the old prior cap; the continuous prior does not put point mass exactly at 300 ms. '
    'The flat part of the trapezoid remains **10–200 ms**.','',
    'With all trials retained, each delay must remain below its own condition’s fastest RT. '
    'Widening creates additional causal support for **5 Control and 16 ASD conditions**. '
    'The largest new effective limits are **347.71 ms Control and 391.00 ms ASD**; the 500 ms prior '
    'does not override these data-implied limits. No RT selection or response coding changes.','',
    f'[Original boundary audit]({base}/delay300_boundary_audit.csv) · '
    f'[Configuration and initialization comparison]({base}/delay300_boundary_diagnosis.json)','',
    '### Direct comparison with the 300 ms fits','',
    f'![Delay and RTD comparisons for 300 versus 500 ms priors]({base}/human_gamma_omega_delay_prior_comparison.png)','',
    '*Rows are Control and ASD. The left two columns compare delay posterior means and 95% variational intervals '
    'at ASL 20/40; gray open circles are the 300 ms fit, group-colored marks the 500 ms fit. '
    'Short gray marks give the condition-specific causal limits. The right panels pool all stimulus differences '
    'using actual trial counts: blue/orange are ASL 20/40, dashed curves use the 300 ms prior and solid curves '
    'the 500 ms prior. Each signed density is normalized on 0–1.8 s before mixing.*','',
    f'[Full-size comparison]({base}/human_gamma_omega_delay_prior_comparison.png) · '
    f'[PDF]({base}/human_gamma_omega_delay_prior_comparison.pdf) · '
    f'[All parameter changes]({base}/delay_bound_parameter_comparison.csv) · '
    f'[Delay shifts and probability above 300 ms]({base}/delay_bound_delay_comparison.csv)','',
    'Comparison source: `ASD_Human_Data/compare_human_gamma_omega_delay_bounds.py` '
    f'([source]({base}/source/compare_human_gamma_omega_delay_bounds.py)).','',
    '### Delay posteriors after widening','',
    '| Group | Delay means above 300 ms | Largest delay mean | Largest upper 95% endpoint | Closest sampled delay to a causal cap |',
    '|---|---:|---:|---:|---:|']
for group in ['Control','ASD']:
    samples=np.load(out/group.lower()/'posterior_samples.npz')['t_E_aff']
    conditions=pd.read_csv(out/group.lower()/'conditions.csv',float_precision='round_trip')
    lines.append(f'| {group} | {int((samples.mean(0)>.3).sum())} / 20 | {samples.mean(0).max()*1000:.2f} ms | '
                 f'{np.quantile(samples,.975,axis=0).max()*1000:.2f} ms | '
                 f'{np.min(conditions.guide_delay_upper_s.to_numpy()-samples)*1000:.2f} ms below |')
lines += ['',
    'No new posterior sample is within 5 ms of its condition’s causal limit. The 500 ms hard prior '
    'is above every condition’s causal support. For ASD at ASL 40, ΔdB = +0.3, the delay moves '
    'from a 290.05 ms mean to **348.27 ms [320.10, 367.53]**, with 99.93% of draws above 300 ms. '
    'These are equal-tailed variational credible intervals.','',
    '### Data and mean-parameter fits with the 500 ms prior','']
for group in ['Control','ASD']:
    stem=f'human_{group.lower()}_gamma_omega_rtd_psychometric'
    lines += [f'![{group} RTDs by ASL and absolute difference, with fitted psychometric]({base}/{stem}.png)','',
        f'*{group}: ASL 20/40 occupy the two RTD rows; columns are |ΔdB| = 0.1, 0.3, 0.65, 1.3, 2. '
        'Steps are data in 20 ms bins; curves use arithmetic posterior means of physical Gamma, Omega, delay, and bias. '
        'Each RTD has unit area on the full 0–1.8 s window. The right panel shows empirical P(right) as dots without error bars '
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
    old=bench.loc[bench.Group.eq(group)&bench.model.str.contains('300 ms prior')].iloc[0]
    new=bench.loc[bench.Group.eq(group)&bench.model.str.contains('500 ms prior')].iloc[0]
    changes.append(f'**{group}: {old.mean_condition_cdf_gap:.4f} → {new.mean_condition_cdf_gap:.4f}**')
lines += ['', 'Widening the delay prior changes the mean signed-condition RT CDF gap: '+ '; '.join(changes)+'. '
          'Control RTDs are essentially unchanged; ASD improves modestly on this average CDF metric, '
          'and its psychometric RMSE decreases. Maximum CDF gaps do not improve. '
          'The sharp empirical RTD peaks remain underfit: relaxing the delay prior does not resolve the main shape mismatch.']
lines += ['',
    'Likelihoods use the same rows and normalization. More flexible fits have more parameters; these in-sample '
    'scores do not establish predictive superiority. The 5/23/42-parameter rows are maximum-likelihood diagnostics; '
    'both 61-parameter SVI rows are evaluated at arithmetic physical posterior means.','',
    f'[Fit-quality table]({base}/fit_quality.csv) · [Benchmark comparisons]({base}/benchmark_comparison.csv) · '
    f'[All signed-condition CDF gaps and probabilities]({base}/condition_fit_quality.csv) · '
    f'[300 versus 500 ms fit metrics]({base}/delay_bound_fit_comparison.csv)','',
    '### Priors, initialization and convergence','',
    '| Parameter | Hard interval | Flat interval of trapezoidal prior |','|---|---:|---:|']
for name,p in PRIORS.items():lines.append(f'| `{name}` | {p["hard"]} | {p["plausible"]} |')
lines += ['',
    'Linear tails connect each flat region to its hard limits. Gamma favors the stimulus-consistent sign but permits '
    'small reversals. The true delay prior remains 0–500 ms; each guide transform alone is confined below its '
    'condition’s earliest RT. All 40 causal limits are exported. Initialization uses the same-trial joint '
    'Gamma/Omega MLE with its shared delay copied into all conditions (Control 102.8 ms, ASD 173.4 ms).','',
    'NumPyro AutoMultivariateNormal; float64 CPU, full batch, one ELBO particle, seed 0, one initialization per group. '
    'ClippedAdam learning rate 0.0002, gradient clip 1, guide initial scale 0.1. Minimum 50,000 and maximum '
    '1,000,000 updates. Any lower 1,000-update window mean resets patience; stopping requires 25 later windows '
    'and posterior-mean ranges below 0.1 posterior SD over the last three checkpoints for all 61 parameters. '
    'Fresh optimizer/RNG runs use the same physical initialization as the 300 ms fit. '
    'The selected best checkpoint is restored; 10,000 joint posterior samples are retained. '
    'Control and ASD ran in parallel using separate CPU processes and checkpoints.','',
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
page.write_text(old_page+'\n'.join(lines)+suffix)
assert page.read_text().startswith(old_page)
index=ROOT/'docs/results/index.md'
old_index=index.read_text();new_index=[]
for line in old_index.splitlines():
    if line.startswith('- [2026-10-21]'):
        if 'human Gamma–Omega delay-prior sensitivity: 300 versus 500 ms' not in line:
            line+='; human Gamma–Omega delay-prior sensitivity: 300 versus 500 ms'
    new_index.append(line)
index.write_text('\n'.join(new_index)+'\n')
assert 'results/2026-10-21.md' in (ROOT/'mkdocs.yml').read_text()
manifest=dict(result_page=str(page.relative_to(ROOT)),assets=str(assets.relative_to(ROOT)),
    result_date=date,page_sha256=sha256(page),previous_page_sha256=sha256(out/'result_book_page_before.md'),
    prior_page_preserved_as_prefix=True,navigation='Existing 2026-10-21 entry verified',
    sources={p:sha256(ROOT/p) for p in source_files},archive_sha256=sha256(archive),
    figures={name:sha256(assets/name) for name in plot['figures']+comparison['figures']})
(out/'publication.json').write_text(json.dumps(manifest,indent=2))
print('Published',page)
