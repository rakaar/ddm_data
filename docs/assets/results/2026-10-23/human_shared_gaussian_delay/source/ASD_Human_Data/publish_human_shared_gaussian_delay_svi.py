# %%
"""Append verified shared-Omega human results, leaving earlier fit entries intact."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import zipfile
import pandas as pd
from human_shared_gaussian_delay_svi_utils import ROOT, OUTPUT, PRIORS, sha256

# %% Editable publication settings.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
parser.add_argument('--date',default='2026-10-23')
args=parser.parse_args();out=args.output_dir;date=args.date
assets=ROOT/f'docs/assets/results/{date}/human_shared_gaussian_delay'
base=f'../assets/results/{date}/human_shared_gaussian_delay'
page=ROOT/f'docs/results/{date}.md'
start='<!-- human-shared-gaussian-delay:start -->';end='<!-- human-shared-gaussian-delay:end -->'
config=json.loads((out/'run_config.json').read_text())
audit=json.loads((out/'independent_validation.json').read_text())
visual=json.loads((out/'visual_inspection.json').read_text())
plot=json.loads((out/'plot_manifest.json').read_text())
assert audit['status']=='passed' and audit['stage']=='posterior' and visual['status']=='passed'
assert audit['verifier_sha256']==sha256(ROOT/'ASD_Human_Data/verify_human_shared_gaussian_delay_svi.py')
assert plot['source_sha256']==sha256(ROOT/'ASD_Human_Data/plot_human_shared_gaussian_delay_svi.py')
for name,digest in config['source_hashes'].items():assert sha256(ROOT/name)==digest,name
for name,digest in plot['files'].items():assert sha256(out/name)==digest,name
inspected={r['file']:r['sha256'] for r in visual['figures']}
for name in plot['figures']:
    if name.endswith('.png'):assert inspected[name]==sha256(out/name)
reports={g:json.loads((out/g.lower()/'convergence_summary.json').read_text()) for g in ['Control','ASD']}
schedule=json.loads((out/'execution_schedule.json').read_text())
schedule.update(completed_utc=datetime.now(timezone.utc).isoformat(),group_completion=reports,
    status='completed_with_flags' if any(r['flagged'] for r in reports.values()) else 'completed_and_verified')
(out/'execution_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
sources=list(config['source_hashes'])+[
    'ASD_Human_Data/plot_human_shared_gaussian_delay_svi.py',
    'ASD_Human_Data/verify_human_shared_gaussian_delay_svi.py',
    'ASD_Human_Data/publish_human_shared_gaussian_delay_svi.py',
    'ASD_Human_Data/prepare_simultaneous_rt_trials.py',
    'fit_each_condn/svi_gamma_omega_likelihood_utils.py']
for name in sources:
    target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
quality=pd.read_csv(out/'baseline_vs_shared_gaussian_delay_fit_quality.csv')
summary=pd.read_csv(out/'parameter_summary.csv')
tails=pd.read_csv(out/'posterior_boundary_audit.csv')
near=(tails[['lower_tail_distance_fraction','upper_tail_distance_fraction']].min(axis=1)<.01)
accum=(tails[['fraction_within_1pct_lower','fraction_within_1pct_upper']].max(axis=1)>=.05)
correlations=pd.read_csv(out/'strongest_correlations.csv')

# %% Append both figures inline, with exact model and stopping definitions.
lines=[start,'## Human RT: one shared Gaussian delay per group','',
    '**Control and ASD are fitted separately with 16 parameters per group.** Gamma has ten values, one per signed ΔdB, '
    'shared across ASL 20 and 40 cd/m². **One Gaussian delay mean μ and one delay SD σ apply to every stimulus within '
    'each group.** Omega, starting bias and the two lapse probabilities are also scalar within each group. Every trial '
    'draws an independent delay from its group’s Gaussian, truncated below zero with no upper cutoff. No symmetry is '
    'imposed between negative and positive Gamma values. ASL invariance and a common delay distribution are restrictions imposed in this fit.','',
    '**Selection:** simultaneous visual `Condition == "RT"`, `Valid == 1`, inclusive **0 ≤ RT_s ≤ 1.8 s**. '
    '**6,574 Control trials from 11 participants** and **4,770 ASD trials from 8 participants**; both choices and '
    'all four RTs below 200 ms are included. The 56 longer RTs are excluded. Raw trials are pooled across participants, '
    'sessions and ASLs during fitting. These are the same 11,344 retained identities as the preceding 34-parameter model.','',
    '### RTDs, psychometric and chronometric','',
    f'![Human ASL-pooled fits: RTDs, psychometric and chronometric]({base}/human_shared_gaussian_delay_pooled_comparison.png)','',
    '*Control blue; ASD red. RTD histograms use 20 ms bins; psychometric and chronometric data are dots without error bars. '
    'Model curves use arithmetic physical posterior means, include uniform lapses and retain 0–1.8 s normalization. '
    'The five RTDs and chronometrics pool opposite signs using actual retained-trial frequencies. Model choice and mean-RT '
    'points are connected only at measured differences. Gamma is shared across matching ASLs, and the same delay Gaussian is used at every signed difference.*','',
    f'[Full-size PNG]({base}/human_shared_gaussian_delay_pooled_comparison.png) · [PDF]({base}/human_shared_gaussian_delay_pooled_comparison.pdf)','',
    '### Parameter comparison','',
    f'![ASL-pooled human parameter comparison]({base}/human_shared_gaussian_delay_parameter_comparison.png)','',
    '*Gamma versus signed ΔdB, with scalar panels for Omega, Gaussian delay mean μ, delay SD σ, starting bias, lapse '
    'probability and lapse-right probability. Control blue; ASD red. Points are posterior means; bars are equal-tailed '
    '**95% variational credible intervals**. Uncertainty is a **pooled-trial variational approximation** and does not '
    'represent participant-to-participant variability. Delays are Gaussian draws truncated below zero, with no upper cutoff.*','',
    f'[Full-size PNG]({base}/human_shared_gaussian_delay_parameter_comparison.png) · [PDF]({base}/human_shared_gaussian_delay_parameter_comparison.pdf) · '
    f'[All parameter summaries]({base}/parameter_summary.csv) · [Truncated-delay moments]({base}/actual_truncated_delay_moments.csv)','',
    '### Fit quality against condition-specific delays','',
    '| Group | Parameters | Joint log likelihood | Marginal RT log likelihood | Psychometric RMSE | Mean CDF gap | Maximum CDF gap | Mean-RT RMSE (s) |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in quality.itertuples():
    lines.append(f'| {r.Group} | {r.free_parameters} | {r.joint_log_likelihood:.2f} | {r.marginal_rt_log_likelihood:.2f} | '
        f'{r.psychometric_rmse:.4f} | {r.mean_condition_cdf_gap:.4f} | {r.max_condition_cdf_gap:.4f} | {r.pooled_chronometric_rmse_s:.4f} |')
lines+=['','34 parameters: Gaussian μ/σ per signed difference. 16 parameters: one Gaussian μ/σ per group. Both models '
    'pool ASLs during fitting and share Omega, bias and lapse parameters. All scores use identical retained trials and '
    'physical posterior means. Psychometric and CDF errors use the same ten signed differences, and chronometric errors '
    'use the five absolute differences. These are descriptive in-sample comparisons; no global optimum or model-selection claim is made.','',
    f'[Complete comparison]({base}/baseline_vs_shared_gaussian_delay_fit_quality.csv) · [Condition metrics]({base}/condition_fit_quality.csv)','',
    '### Likelihood, priors and initialization','',
    r'The observed-choice likelihood is $(1-p)h_y(t)/Z_D + p q_y/1.8$. The density $h_y$ convolves the reactive '
    r'first-passage density with a Gaussian normalized by $\Phi(\mu_D/\sigma_D)$ for truncation below zero. '
    '`Z_D` is the either-bound mass over 0–1.8 s. Both normalizations are recomputed at every likelihood evaluation. '
    'Lapse RT is Uniform(0,1.8 s) in stored RT, without a delay shift; p is the lapse fraction among retained trials.','',
    '| Parameter | Hard bounds | Flat region |','|---|---|---|']
for name,prior in PRIORS.items():lines.append(f'| `{name}` | {prior["hard"]} | {prior["plausible"]} |')
lines+=['','Independent trapezoidal priors with linear tails, applied once per free parameter. Omega has one prior contribution '
    'per group. Delay units are seconds; Omega units are s⁻¹. The priors match the preceding 34-parameter Gaussian-delay fit. Each shared μ and σ prior contributes once per group.','',
    '**Original starting values were selected by the user.** Original Gamma and shared scalar centers are copied from '
    'the 34-parameter run’s saved initialization files. The ten original μ values are averaged to initialize one μ, '
    'and likewise for σ. Omega starts at **0.716870754 Control / 0.811822111 ASD**; delay μ at **102.787 / 173.426 ms**; '
    'σ at **50 ms**, p at **0.02** and q at **0.5**. All 16 parameters are re-estimated.','',
    '### Loss-only stopping and posterior diagnostics','',
    'Both groups run concurrently on the GPU with bounded CPU allocations and memory preallocation disabled. '
    'Float64, full-batch NumPyro AutoMultivariateNormal, one ELBO particle, seed 0, guide scale 0.1, '
    'ClippedAdam learning rate 0.0002 and gradient clip 1.','',
    '**Stopping uses loss alone:** minimum 50,000 and maximum 1,000,000 updates; nonoverlapping 1,000-update '
    'average-loss windows. Every strictly lower average saves a new best checkpoint and resets patience. '
    'Require 25 subsequent windows without improvement after the minimum update count, then restore the lowest '
    'recorded window-average-loss checkpoint. **There is no posterior-mean/SD stability gate.** A cap stop without '
    'confirmation is flagged. Each group saves 10,000 aligned joint variational posterior draws.','',
    '| Group | Final updates | Restored best | Best average negative ELBO | Subsequent windows | Status |',
    '|---|---:|---:|---:|---:|---|']
for g,c in reports.items():
    status='**Step cap: loss confirmation incomplete**' if c['flagged'] else 'Loss confirmation passed'
    lines.append(f'| {g} | {c["completed_steps"]:,} | {c["restored_best_step"]:,} | {c["best_window_mean_negative_elbo"]:.4f} | {c["confirmation_windows"]} | {status} |')
lines+=['','Loss confirmation does not establish a global optimum.','',
    f'[Loss traces and correlations]({base}/posterior_diagnostics.png) · [Control stopping record]({base}/control/convergence_summary.json) · '
    f'[ASD stopping record]({base}/asd/convergence_summary.json)','',
    f'Both the **0.5th and 99.5th percentiles** were checked for all 32 parameters. **{int(near.sum())}** have an endpoint '
    f'in the outer 1% of their hard range; **{int(accum.sum())}** have at least 5% of draws in an outer 1% region. '
    'This relative-range screen describes proximity; it is not a stopping gate or proof that a bound constrains the fit. '
    'Bounds were not automatically changed.','',f'[Complete boundary audit]({base}/posterior_boundary_audit.csv)','']
for g in ['Control','ASD']:
    r=correlations.loc[correlations.Group.eq(g)].iloc[0]
    lines += [f'{g}: strongest reported correlation is `{r.parameter_1}` with `{r.parameter_2}`, **r={r.correlation:.3f}**.','']
lines+=['### Verification and source audit','',
    'Independent checks reconstruct every retained/excluded source identity, the ten-condition mapping, original initialization, '
    '16 free parameters and single prior contributions. The new likelihood and chain gradients match the existing model '
    'when the ten delay Gaussians are equal; scalar μ/σ gradients match the corresponding summed condition gradients. Checks cover Gaussian truncation, retained mass, density/CDF agreement, finite gradients, '
    'K=8/40/80, finer grids, posterior summaries/draws, lowest-loss restoration, patience reset, minimum/cap stopping, '
    'histograms, probabilities, sign weights and chronometric means. Prior fit files are hash-checked.','',
    f'Source: `ASD_Human_Data/plot_human_shared_gaussian_delay_svi.py`. Native outputs: `{out.relative_to(ROOT)}/`.','',
    f'- [Fitter]({base}/source/ASD_Human_Data/fit_human_shared_gaussian_delay_svi.py) · '
    f'[Likelihood adapter]({base}/source/ASD_Human_Data/human_shared_gaussian_delay_svi_utils.py) · '
    f'[Plotter]({base}/source/ASD_Human_Data/plot_human_shared_gaussian_delay_svi.py) · '
    f'[Verifier]({base}/source/ASD_Human_Data/verify_human_shared_gaussian_delay_svi.py).',
    f'- [Configuration and provenance]({base}/run_config.json) · [Numerical verification]({base}/independent_validation.json) · '
    f'[Visual inspection]({base}/visual_inspection.json) · [Trial counts]({base}/selection_counts.csv).',
    f'- [Psychometrics]({base}/pooled_psychometric.csv) · [Chronometrics]({base}/chronometric_points.csv) · [Mixture weights]({base}/mixture_weights.csv).',
    f'- [Posterior draws, checkpoints, calculations and source ZIP]({base}/human_shared_gaussian_delay_audits.zip).','',end,'']
entry='\n'.join(lines)
before=page.read_text() if page.exists() else f'# Results: {date}\n'
if start in before:
    a=before.index(start);b=before.index(end,a)+len(end)
    after=before[:a]+entry.rstrip()+before[b:]
else:after=before.rstrip()+'\n\n'+entry
page.parent.mkdir(parents=True,exist_ok=True);page.write_text(after)
index=ROOT/'docs/results/index.md';text=index.read_text();link=f'- [{date}]({date}.md)'
if link not in text:index.write_text(text.rstrip()+'\n'+link+'\n')
section_link=f'- [Human RT: one shared Gaussian delay per group]({date}.md#human-rt-one-shared-gaussian-delay-per-group)'
text=index.read_text()
if section_link not in text:index.write_text(text.rstrip()+'\n'+section_link+'\n')
nav=ROOT/'mkdocs.yml';text=nav.read_text();line=f'      - "{date}": results/{date}.md'
if line not in text:
    anchor='      - Overview: results/index.md'
    assert anchor in text;nav.write_text(text.replace(anchor,anchor+'\n'+line))

# %% Keep large calculation tables/posteriors in the downloadable audit bundle.
assets.mkdir(parents=True,exist_ok=True)
direct=plot['figures']+['parameter_summary.csv','actual_truncated_delay_moments.csv',
    'baseline_vs_shared_gaussian_delay_fit_quality.csv','condition_fit_quality.csv','posterior_boundary_audit.csv',
    'strongest_correlations.csv','posterior_diagnostics.png','pooled_psychometric.csv','chronometric_points.csv',
    'mixture_weights.csv','run_config.json','independent_validation.json','visual_inspection.json',
    'selection_counts.csv','plot_manifest.json','execution_schedule.json']
for g in ['control','asd']:direct.append(f'{g}/convergence_summary.json')
for name in direct:
    dest=assets/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(out/name,dest)
shutil.copytree(out/'source',assets/'source',dirs_exist_ok=True)
bundle=out/'human_shared_gaussian_delay_audits.zip'
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix not in ['.zip','.png','.pdf'] and path.name not in ['publication_manifest.json','result_book_verification.json']:
            archive.write(path,path.relative_to(out))
shutil.copy2(bundle,assets/bundle.name)
publication=dict(date=date,page=str(page.relative_to(ROOT)),assets=str(assets.relative_to(ROOT)),
    original_page_prefix_preserved=after.startswith(before.rstrip()) if start not in before else True,
    files={str(p.relative_to(assets)):sha256(p) for p in assets.rglob('*') if p.is_file()})
(out/'publication_manifest.json').write_text(json.dumps(publication,indent=2)+'\n')
print(json.dumps(dict(page=str(page),assets=str(assets),files=len(publication['files'])),indent=2))
