# %%
"""Publish the audited Gaussian-delay human fits and all three primary figures."""
from pathlib import Path
import argparse
import json
import shutil
import zipfile
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from human_gaussian_delay_svi_utils_wide import ROOT, OUTPUT, PRIORS, sha256

# %% Publication settings. Keep the previous result-book entries and fitted models.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
args=parser.parse_args();out=args.output_dir
date='2026-10-22'
assets=ROOT/f'docs/assets/results/{date}/human_gaussian_delay_svi_wide'
base=f'../assets/results/{date}/human_gaussian_delay_svi_wide'
page=ROOT/f'docs/results/{date}.md'
start='<!-- human-gaussian-delay-svi-wide:start -->'
end='<!-- human-gaussian-delay-svi-wide:end -->'
config=json.loads((out/'run_config.json').read_text())
audit=json.loads((out/'independent_validation.json').read_text())
visual=json.loads((out/'visual_inspection.json').read_text())
plot=json.loads((out/'plot_manifest.json').read_text())
assert audit['status']=='passed' and audit['stage']=='posterior' and visual['status']=='passed'
for name,digest in config['source_hashes'].items():assert sha256(ROOT/name)==digest,name
assert plot['source_sha256']==sha256(ROOT/'ASD_Human_Data/plot_human_gaussian_delay_svi_wide.py')
assert audit['verifier_sha256']==sha256(ROOT/'ASD_Human_Data/verify_human_gaussian_delay_svi_wide.py')
inspected={r['file']:r['sha256'] for r in visual['figures']}
for name in plot['figures']:
    if name.endswith('.png'):assert inspected[name]==sha256(out/name),name
schedule=json.loads((out/'execution_schedule.json').read_text())
schedule['completed_utc']=datetime.now(timezone.utc).isoformat()
schedule['group_completion']={group:json.loads((out/group.lower()/'convergence_summary.json').read_text())
                              for group in ['Control','ASD']}
schedule['status']='completed_with_flags' if any(r['flagged'] for r in schedule['group_completion'].values()) else 'completed_and_verified'
(out/'execution_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
sources=list(config['source_hashes'])+[
    'ASD_Human_Data/plot_human_gaussian_delay_svi_wide.py',
    'ASD_Human_Data/verify_human_gaussian_delay_svi_wide.py',
    'ASD_Human_Data/publish_human_gaussian_delay_svi_wide.py',
    'ASD_Human_Data/prepare_simultaneous_rt_trials.py',
    'ASD_Human_Data/audit_human_gaussian_delay_bounds.py',
    'ASD_Human_Data/validate_gaussian_delay_likelihood.py',
    'fit_each_condn/svi_gamma_omega_likelihood_utils.py',
]
assets.mkdir(parents=True,exist_ok=True)
for name in plot['figures']+plot['tables']+[
    'run_config.json','independent_validation.json','independent_preflight.json',
    'visual_inspection.json','plot_manifest.json','selection_counts.csv','execution_schedule.json',
    'posterior_99pct_boundary_audit.csv','posterior_99pct_boundary_summary.json',
    'original_vs_wide_parameter_comparison.csv','narrow_sigma_path_validation.json']:
    shutil.copy2(out/name,assets/name)
for group in ['control','asd']:
    (assets/group).mkdir(exist_ok=True)
    for name in ['posterior_summary.csv','posterior_covariance.csv','posterior_correlation.csv',
                 'convergence_summary.json','convergence.csv','posterior_numerical_checks.csv',
                 'independent_curve_checks.csv','data_manifest.json','conditions.csv']:
        shutil.copy2(out/group/name,assets/group/name)
for name in sources:
    dest=assets/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)

# %% Inline scientific result entry, with the three requested images shown directly.
quality=pd.read_csv(out/'fixed_vs_gaussian_fit_quality.csv')
summary=pd.read_csv(out/'parameter_summary.csv')
boundaries=pd.read_csv(out/'posterior_boundary_audit.csv')
near=boundaries[['fraction_within_1pct_lower','fraction_within_1pct_upper']].max(axis=1)>=.05
correlations=pd.read_csv(out/'strongest_correlations.csv')
lines=[start,'## Human RT: wider Gaussian-delay bounds, choice and uniform lapses','',
    '**Control and ASD are fitted separately.** Retain `Condition == "RT"`, `Valid == 1`, '
    'and inclusive **0 ≤ RT_s ≤ 1.8 s**: **6,574 Control trials from 11 participants** and '
    '**4,770 ASD trials from 8 participants**. These are simultaneous presentations. '
    'Pool raw trials across participants and sessions, including correct and incorrect choices '
    'and all four RTs below 200 ms; exclude the 56 RT trials above 1.8 s, all SD and all sequential CB trials. '
    'The retained identities are identical to the previous fixed-delay lapse fit.','',
    'Each group has **83 free parameters**: independent Gamma, Omega, Gaussian μ and Gaussian σ for each '
    'of 20 signed-ΔdB × ASL conditions, plus shared starting bias `w`, lapse fraction `p`, and '
    'lapse-right probability `q`. Gamma/Omega remain descriptive condition parameters, without an auditory '
    'stimulus-to-drift mapping or symmetry constraint.','',
    r'For each non-lapse trial, delay is drawn independently from $N(\mu,\sigma^2)$ '
    '**truncated below zero, with no upper cutoff**. This is trial variability within a condition. '
    'The fitted μ and σ describe the underlying Gaussian; its mean and SD after truncation are different '
    'and are provided in a separate table. Those delay-distribution moments are calculated before '
    'the additional RT retention filter.','',
    r'The joint likelihood is $(1-p) f_{G,y}(t)/Z + p q_y/1.8$, where $f_G$ convolves the '
    'reactive first-passage density with the truncated-Gaussian delay, `Z` is its either-choice mass '
    'inside 0–1.8 s, and `q_right=q`, `q_left=1-q`. The lapse fraction refers to **retained trials**. '
    '**Lapse RT is Uniform(0,1.8 s)** in stored RT, without a delay shift.','']
for group in ['Control','ASD']:
    stem=f'human_{group.lower()}_gaussian_delay_rtd_psychometric'
    lines += [f'### {group}: RT distributions and psychometric','',
        f'![{group}: Gaussian-delay joint-with-lapse RTDs and psychometric]({base}/{stem}.png)','',
        '*ASL 20 and 40 cd/m² occupy the two RTD rows, with |ΔdB| = 0.1, 0.3, 0.65, 1.3 and 2 '
        'across columns. Empirical histograms use 20 ms bins. Analytic curves use arithmetic posterior '
        'means of the physical parameters, include lapses, and normalize each signed prediction on '
        '0–1.8 s before combining signs with actual retained trial frequencies. Psychometric data are '
        'dots without error bars; model probabilities at the ten measured differences are connected '
        'only as visual guides. Both groups share RTD axis limits.*','',
        f'[Full-size PNG]({base}/{stem}.png) · [PDF]({base}/{stem}.pdf)','']
lines += ['### Parameter comparison','',
    f'![Gaussian-delay posterior comparison: Gamma, Omega, Gaussian mu and sigma, and shared bias/lapse scalars]({base}/human_gaussian_delay_parameter_comparison.png)','',
    '*Four rows show Gamma, Omega, Gaussian μ and Gaussian σ; columns separate ASL 20/40. '
    'Control is blue and ASD red. Points are posterior means with equal-tailed **95% variational '
    'credible intervals**. The side panels show shared `w`, `p`, and `q`. These intervals describe '
    'a pooled-trial variational approximation and do not measure participant-to-participant variability.*','',
    f'[Full-size PNG]({base}/human_gaussian_delay_parameter_comparison.png) · '
    f'[PDF]({base}/human_gaussian_delay_parameter_comparison.pdf) · '
    f'[Parameter means, SDs and intervals]({base}/parameter_summary.csv) · '
    f'[Actual truncated-delay mean and SD]({base}/actual_truncated_delay_moments.csv)','',
    '### Comparison with the fixed-delay lapse fit','',
    '| Group | Delay model | Parameters | Joint log likelihood | Marginal RT log likelihood | Psychometric RMSE | Mean signed-condition CDF gap | Maximum CDF gap |',
    '|---|---|---:|---:|---:|---:|---:|---:|']
for row in quality.itertuples():
    label='Gaussian' if row.free_parameters==83 else 'Fixed'
    lines.append(f'| {row.Group} | {label} | {row.free_parameters} | {row.joint_log_likelihood:.2f} | '
        f'{row.marginal_rt_log_likelihood:.2f} | {row.psychometric_rmse:.4f} | '
        f'{row.mean_condition_cdf_gap:.4f} | {row.max_condition_cdf_gap:.4f} |')
lines += ['', 'All scores use the same retained trials, window normalization and arithmetic physical '
    'posterior-mean convention. CDF gaps are maximum empirical-versus-model differences within each '
    'signed condition. These are in-sample comparisons; the Gaussian-delay model has 20 additional '
    'parameters. ELBO differences between model families are not used as a fit-quality score.','',
    f'[Comparison table]({base}/fixed_vs_gaussian_fit_quality.csv) · '
    f'[Condition-specific quality]({base}/condition_fit_quality.csv) · '
    f'[Psychometric calculations]({base}/psychometric_comparison.csv) · '
    f'[Histogram counts]({base}/rtd_histograms.csv) · [RTD mixture curves and weights]({base}/rtd_model_curves.csv)','',
    '### Priors and initialization','',
    '| Parameter | Hard bounds | Flat region |','|---|---:|---:|']
for name,prior in PRIORS.items():lines.append(f'| `{name}` | {prior["hard"]} | {prior["plausible"]} |')
lines += ['', 'Priors are independent trapezoids with linear tails. Delay quantities are in seconds, '
    'Omega is in s⁻¹, and the Gaussian μ support is not restricted by the fastest observed RT. '
    'A 1000 ms bound on μ is not a maximum delay draw; truncated-Gaussian draws have no upper cap.','',
    'Initialization uses the **original saved Gamma/Omega/bias centers**, not the posterior means '
    'of the completed fixed-delay fit. Each Gaussian μ starts at the original shared delay '
    '(Control 102.787 ms; ASD 173.426 ms); each σ starts at 50 ms. '
    'Shared lapse centers are `p=0.02`, `q=0.5`.','',
    '### Computation, convergence and limitations','',
    'NumPyro AutoMultivariateNormal, float64 CPU, full-batch likelihood, one ELBO particle, seed 0 '
    'and one initialization per group; guide scale 0.1, ClippedAdam learning rate 0.0002, gradient '
    'clip 1. Control and ASD ran concurrently on separate eight-core CPU allocations. '
    'The run permits 50,000–1,000,000 updates. Any lower 1,000-update mean loss resets patience; '
    'stopping requires 25 subsequent windows plus posterior-mean ranges below 0.1 posterior SD '
    'over the last three checkpoints. The selected best checkpoint is restored for 10,000 '
    'constrained posterior draws.','',
    '| Group | Final updates | Restored best | Subsequent windows | Terminal max change (SD) | Best locally stable | Status |',
    '|---|---:|---:|---:|---:|---|---|']
for group in ['Control','ASD']:
    c=json.loads((out/group.lower()/'convergence_summary.json').read_text())
    lines.append(f'| {group} | {c["completed_steps"]:,} | {c["restored_best_step"]:,} | '
        f'{c["confirmation_windows"]} | {c["maximum_terminal_mean_change_sd"]:.3f} | '
        f'{c["restored_checkpoint_parameter_stable"]} | '
        + ('**PROVISIONAL: checks not satisfied**' if c['flagged'] else 'Confirmation and terminal stability passed')+' |')
lines += ['', 'The stability gate applies to the terminal checkpoints; local stability at the restored '
    'checkpoint is reported separately. These checks do not establish a global optimum.','',
    f'![Gaussian-delay convergence traces and physical posterior correlations]({base}/human_gaussian_delay_convergence_correlations.png)','',
    f'**{int(near.sum())} of 166 parameters** have at least 5% of their posterior draws in the outer '
    '1% of a hard prior interval. The boundary audit lists both sides explicitly.','']
if near.any():
    near_table=boundaries.loc[near].merge(summary[['Group','parameter','ASL_cd','signed_delta']],on=['Group','parameter'])
    lines += ['| Group | Parameter | ASL, signed ΔdB | Near lower bound | Near upper bound |',
              '|---|---|---|---:|---:|']
    for r in near_table.itertuples():
        stimulus='shared' if pd.isna(r.ASL_cd) else f'{r.ASL_cd:g}, {r.signed_delta:+g}'
        lines.append(f'| {r.Group} | `{r.parameter}` | {stimulus} | '
            f'{100*r.fraction_within_1pct_lower:.1f}% | {100*r.fraction_within_1pct_upper:.1f}% |')
    lines += ['']
for group in ['Control','ASD']:
    r=correlations.loc[correlations.Group.eq(group)].iloc[0]
    lines += [f'{group} strongest reported physical-parameter correlation: '
              f'`{r.parameter_1}` with `{r.parameter_2}`, **r={r.correlation:.3f}**.','']
lines += [f'[Boundary audit]({base}/posterior_boundary_audit.csv) · '
    f'[Strongest correlations]({base}/strongest_correlations.csv) · '
    f'[Control convergence]({base}/control/convergence_summary.json) · '
    f'[ASD convergence]({base}/asd/convergence_summary.json)','',
    '### Likelihood accuracy and speed','',
    'The fast path evaluates the two first-passage densities on a shared **0.5 ms time grid**, '
    'then convolves each with its delay density using zero-padded FFTs. The convolution is computed '
    'once per stimulus and interpolated at the individual observed RTs. JAX compiles and '
    'differentiates this batched calculation. It is a trial-level likelihood, without histogram fitting.','',
    'Positive quadrature on log decision time supplies the retention normalizer and handles '
    'conditions whose very fast first-passage mass cannot be resolved on the grid. '
    'Negligible tails may use the FFT result when its numerical error is dominated by the uniform '
    'lapse floor. The image/eigen series uses K=8, checked against K=40/80. Independent adaptive '
    'SciPy integration, half-step grids, higher quadrature orders, finite gradients, unit conditional '
    'mass and 100 posterior draws per group are audited.','',
    'This rerun raises the μ upper bound from 500 to 1000 ms and lowers the σ bound '
    'from 5 to 0.5 ms. All other prior limits, flat regions, initialization, data and SVI settings '
    'remain unchanged. Direct quadrature also handles Gaussians narrower than four grid steps. '
    'The earlier results and their source files are preserved. Widening trapezoidal support '
    'changes the tail taper and prior normalization even though the flat regions stay fixed.','',
    '### Sources and audit files','',
    'Native output: `ASD_Human_Data/gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms/`.','',
    f'- [Fitter]({base}/source/ASD_Human_Data/fit_human_gaussian_delay_svi_wide.py) · '
    f'[Human likelihood adapter]({base}/source/ASD_Human_Data/human_gaussian_delay_svi_utils_wide.py) · '
    f'[Gaussian-delay likelihood]({base}/source/ASD_Human_Data/gaussian_delay_likelihood_wide.py).',
    f'- [Plotter]({base}/source/ASD_Human_Data/plot_human_gaussian_delay_svi_wide.py) · '
    f'[Independent verifier]({base}/source/ASD_Human_Data/verify_human_gaussian_delay_svi_wide.py) · '
    f'[Publisher]({base}/source/ASD_Human_Data/publish_human_gaussian_delay_svi_wide.py).',
    f'- [Configuration and provenance]({base}/run_config.json) · '
    f'[Independent verification]({base}/independent_validation.json) · '
    f'[Visual inspection]({base}/visual_inspection.json) · '
    f'[Selection counts]({base}/selection_counts.csv).',
    f'- [Full posterior, checkpoint, calculation and source audit ZIP]({base}/human_gaussian_delay_svi_audits.zip).','',end,'']
# Both posterior tails, with explicit screening and actual distances.
tails=pd.read_csv(out/'posterior_99pct_boundary_audit.csv')
extra=['### Wider versus original Gaussian priors','',
    'Only the Gaussian μ hard upper bound (500 → 1000 ms) and σ hard lower bound (5 → 0.5 ms) changed. '
    'The data, original initialization, flat prior regions and SVI settings match. '
    'The wider supports also change the trapezoidal tail slopes and normalizing constants.','',
    '| Group | Prior bounds | Joint log likelihood | Marginal RT log likelihood | Psychometric RMSE | Mean CDF gap |',
    '|---|---|---:|---:|---:|---:|']
compare=pd.read_csv(out/'original_vs_wide_fit_quality.csv')
for r in compare.itertuples():
    label='Original' if 'original bounds' in r.model else 'Wider'
    extra.append(f'| {r.Group} | {label} | {r.joint_log_likelihood:.2f} | {r.marginal_rt_log_likelihood:.2f} | {r.psychometric_rmse:.4f} | {r.mean_condition_cdf_gap:.4f} |')
extra += ['',f'[Fit-quality comparison]({base}/original_vs_wide_fit_quality.csv) · '
    f'[Parameter shifts and both tail endpoints]({base}/original_vs_wide_parameter_comparison.csv)','',
    '### Both posterior tails versus hard bounds','',
    'The audit checks the **0.5th and 99.5th percentiles** of all 166 parameters using 10,000 draws per group '
    '(central 99% variational intervals). For a reproducible screen, proximity is an endpoint in the outer '
    '1% of its hard range; accumulation is at least 5% of draws in that region. '
    'Raw distances and distances in posterior SD are also saved. A broad hard range can make the '
    'relative-distance screen flag parameters well separated in posterior SD; it is not proof of a constraint.','',
    f'**{int(tails.tail_within_1pct.sum())} parameters** meet the proximity screen; '
    f'**{int(tails.accumulation_5pct.sum())}** meet the accumulation screen.','',
    '| Group | Parameter | ASL / signed Δ | 0.5th percentile | 99.5th percentile | Hard interval |',
    '|---|---|---|---:|---:|---|']
# Show the nearest five endpoints on either side; CSV contains every parameter.
nearest=pd.concat([tails.nsmallest(5,'lower_gap_fraction'),tails.nsmallest(5,'upper_gap_fraction')]).drop_duplicates(['Group','parameter'])
for r in nearest.itertuples():
    stimulus='shared' if pd.isna(r.ASL_cd) else f'{r.ASL_cd:g} / {r.signed_delta:+g}'
    extra.append(f'| {r.Group} | `{r.parameter}` | {stimulus} | {r.q005:.6g} | {r.q995:.6g} | [{r.hard_lower:g}, {r.hard_upper:g}] |')
expanded=pd.read_csv(out/'original_vs_wide_parameter_comparison.csv')
max_upper=expanded.loc[expanded.fraction_above_previous_upper.idxmax()]
max_lower=expanded.loc[expanded.fraction_below_previous_lower.idxmax()]
extra += ['', f'The expanded μ range is used: **{100*max_upper.fraction_above_previous_upper:.2f}%** of '
    f'{max_upper.Group} `{max_upper.parameter}` (ASL {max_upper.ASL_cd:g}, Δ {max_upper.signed_delta:+g}) '
    'draws exceed the previous 500 ms cap. The largest posterior fraction below the previous 5 ms '
    f'σ floor is **{100*max_lower.fraction_below_previous_lower:.2f}%** '
    f'({max_lower.Group} `{max_lower.parameter}`).','',
    '| Group | Smallest μ 0.5th percentile | Largest μ 99.5th percentile | Smallest σ 0.5th percentile | Largest σ 99.5th percentile |',
    '|---|---:|---:|---:|---:|']
for group in ['Control','ASD']:
    mu=tails.loc[tails.Group.eq(group)&tails.block.eq('delay_mu')]
    sd=tails.loc[tails.Group.eq(group)&tails.block.eq('delay_sigma')]
    extra.append(f'| {group} | {mu.q005.min()*1000:.2f} ms | {mu.q995.max()*1000:.2f} ms | '
        f'{sd.q005.min()*1000:.2f} ms | {sd.q995.max()*1000:.2f} ms |')
extra += ['', 'Against μ bounds [0, 1000] ms and σ bounds [0.5, 300] ms, neither side triggers '
    'the predeclared tail-proximity screen. No further bound widening was indicated by this audit.','']
extra += ['', 'The parameter-by-parameter table uses seconds for delays; the group summary uses milliseconds. Numerical proximity, prior sensitivity and '
    'variational uncertainty are distinct: this audit cannot prove that all bounds or priors are irrelevant.','',
    f'[Complete both-tail boundary audit]({base}/posterior_99pct_boundary_audit.csv) · '
    f'[Audit summary]({base}/posterior_99pct_boundary_summary.json) · '
    f'[0.5 ms likelihood check]({base}/narrow_sigma_path_validation.json)','']
lines[lines.index(end):lines.index(end)]=extra
entry='\n'.join(lines)
old=page.read_text() if page.exists() else f'# Results: {date}\n\n'
if start in old:
    a=old.index(start);b=old.index(end,a)+len(end)
    updated=old[:a]+entry.rstrip()+old[b:]
else:
    updated=old.rstrip()+'\n\n'+entry
page.write_text(updated)
index=ROOT/'docs/results/index.md'
index_text=index.read_text()
description='Wider Gaussian delay bounds (μ ≤1 s, σ ≥0.5 ms): additional Control/ASD fits and posterior tail audit.'
if f'({date}.md)' not in index_text:
    index_text=index_text.rstrip()+f'\n- [{date}]({date}.md) — {description}\n'
elif description not in index_text:
    line=next(line for line in index_text.splitlines() if f'({date}.md)' in line)
    index_text=index_text.replace(line,line+' '+description)
index.write_text(index_text)
nav=ROOT/'mkdocs.yml'
if f'results/{date}.md' not in nav.read_text():
    text=nav.read_text();needle='      - "2026-10-21": results/2026-10-21.md'
    assert needle in text
    nav.write_text(text.replace(needle,needle+f'\n      - "{date}": results/{date}.md'))

# %% README summary and a complete local audit bundle, without external upload.
native_readme=(out/'README.md')
native_readme.write_text('# Human Gaussian-delay Gamma/Omega SVI\n\n'
    'Joint RT + choice + uniform lapse; 83 parameters per group. Saved output and methodology: '
    '[22 October result-book page](../../docs/results/2026-10-22.md#human-rt-wider-gaussian-delay-bounds-choice-and-uniform-lapses).\n\n'
    'Reproduction sources are the fit_human_gaussian_delay_svi_wide.py, plot_human_gaussian_delay_svi_wide.py, '
    'verify_human_gaussian_delay_svi_wide.py and publish_human_gaussian_delay_svi_wide.py scripts one directory above. '
    'Use a separate --output-dir for a new fit; --resume-existing continues saved checkpoints. '
    'run_config.json freezes likelihood source hashes, priors, original initialization and numerical settings.\n\n'
    'Earlier Gaussian fit and simulation outputs remain unchanged. Current independent verification '
    'includes extreme-bias and narrow-delay cases and posterior numerical refinement checks.\n')
readme=ROOT/'ASD_Human_Data/README.md';text=readme.read_text()
text=text.replace('the Gaussian does not describe a variational posterior. No human fit is run.',
    'the Gaussian does not describe a variational posterior. That synthetic stage did not fit human data; '
    'the later human-fit section below uses the validated likelihood with additional numerical safeguards.')
heading='## Wider Gaussian delay bounds: μ up to 1 s and σ down to 0.5 ms'
block='\n\n'+heading+'\n\n'+'Implemented under `gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms/`. '
block+='The same 11,344 retained simultaneous RT trials are fitted with independent condition-wise Gamma, '
block+='Omega, zero-truncated Gaussian μ and σ, plus shared bias and lapse probabilities (83 parameters per group). '
block+='Lapse RT remains Uniform(0,1.8 s). Original saved initial centers are retained; σ starts at 50 ms. '
block+='Both groups ran in parallel. See the [22 October result-book entry](../docs/results/2026-10-22.md#human-rt-wider-gaussian-delay-bounds-choice-and-uniform-lapses) '
block+='for the three inline figures, priors, convergence status, fixed-delay comparison, source links and full posterior audit.\n'
if heading not in text:text=text.rstrip()+block
readme.write_text(text)
frozen=json.loads((out/'preservation_baseline.json').read_text())
changed=[name for name,digest in frozen.items() if not (ROOT/name).exists() or sha256(ROOT/name)!=digest]
assert not set(changed)-{'ASD_Human_Data/README.md'},changed
preservation=dict(checked_files=len(frozen),unchanged_files=len(frozen)-len(changed),
                  intended_changes=changed,checked_after_readme_update=True)
(out/'publication_preservation.json').write_text(json.dumps(preservation,indent=2)+'\n')
shutil.copy2(out/'publication_preservation.json',assets/'publication_preservation.json')
with zipfile.ZipFile(assets/'human_gaussian_delay_svi_audits.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(out.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name not in ['publication_manifest.json','worktree_before.txt']:
            z.write(path,'outputs/'+str(path.relative_to(out)))
    for name in sources:z.write(ROOT/name,'source/'+name)
    for group in ['control','asd']:
        record=json.loads((out/group/'data_manifest.json').read_text())
        name=record['initialization_source']
        assert sha256(ROOT/name)==record['initialization_sha256']
        z.write(ROOT/name,'original_initialization/'+group+'.json')
    for name in ['validation_report.json','README.md']:
        z.write(ROOT/'ASD_Human_Data/gaussian_delay_validation'/name,'earlier_synthetic_validation/'+name)
manifest={str(p.relative_to(assets)):sha256(p) for p in sorted(assets.rglob('*')) if p.is_file()}
(out/'publication_manifest.json').write_text(json.dumps(dict(date=date,assets=manifest,
    page=str(page.relative_to(ROOT)),page_sha256=sha256(page)),indent=2)+'\n')
print('Published all three primary figures inline to',page)
