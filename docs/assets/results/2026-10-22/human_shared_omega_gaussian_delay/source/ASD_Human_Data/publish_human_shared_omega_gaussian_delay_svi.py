# %%
"""Append verified shared-Omega human results, leaving earlier fit entries intact."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import zipfile
import pandas as pd
from human_shared_omega_gaussian_delay_svi_utils import ROOT, OUTPUT, PRIORS, sha256

# %% Editable publication settings.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=OUTPUT)
parser.add_argument('--date',default=datetime.now().astimezone().date().isoformat())
args=parser.parse_args();out=args.output_dir;date=args.date
assets=ROOT/f'docs/assets/results/{date}/human_shared_omega_gaussian_delay'
base=f'../assets/results/{date}/human_shared_omega_gaussian_delay'
page=ROOT/f'docs/results/{date}.md'
start='<!-- human-shared-omega-gaussian-delay:start -->';end='<!-- human-shared-omega-gaussian-delay:end -->'
config=json.loads((out/'run_config.json').read_text())
audit=json.loads((out/'independent_validation.json').read_text())
visual=json.loads((out/'visual_inspection.json').read_text())
plot=json.loads((out/'plot_manifest.json').read_text())
assert audit['status']=='passed' and audit['stage']=='posterior' and visual['status']=='passed'
assert audit['verifier_sha256']==sha256(ROOT/'ASD_Human_Data/verify_human_shared_omega_gaussian_delay_svi.py')
assert plot['source_sha256']==sha256(ROOT/'ASD_Human_Data/plot_human_shared_omega_gaussian_delay_svi.py')
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
    'ASD_Human_Data/finalize_human_shared_omega_saved_best.py',
    'ASD_Human_Data/plot_human_shared_omega_gaussian_delay_svi.py',
    'ASD_Human_Data/verify_human_shared_omega_gaussian_delay_svi.py',
    'ASD_Human_Data/publish_human_shared_omega_gaussian_delay_svi.py',
    'ASD_Human_Data/prepare_simultaneous_rt_trials.py',
    'fit_each_condn/svi_gamma_omega_likelihood_utils.py']
for name in sources:
    target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
quality=pd.read_csv(out/'baseline_vs_shared_omega_fit_quality.csv')
summary=pd.read_csv(out/'parameter_summary.csv')
tails=pd.read_csv(out/'posterior_boundary_audit.csv')
near=(tails[['lower_tail_distance_fraction','upper_tail_distance_fraction']].min(axis=1)<.01)
accum=(tails[['fraction_within_1pct_lower','fraction_within_1pct_upper']].max(axis=1)>=.05)
correlations=pd.read_csv(out/'strongest_correlations.csv')

# %% Publication requires completed numerical and visual checks; both main PNGs are inline.
lines=[start,'## Human RT: one shared Omega per group','',
    '**Control and ASD are refitted separately, with one Omega shared across both ASLs and all ten signed differences within each group.** '
    'All parameters are re-estimated. Each group has **64 parameters**: 20 Gamma, one Omega, 20 Gaussian delay μ, 20 Gaussian delay σ, '
    'one starting bias, one lapse probability and one lapse-right probability. Gamma and delay parameters remain condition-specific.','',
    '**Selection:** `Condition == "RT"`, `Valid == 1`, inclusive **0 ≤ RT_s ≤ 1.8 s**. '
    '**6,574 Control trials from 11 participants** and **4,770 ASD trials from 8 participants**; the same 11,344 retained identities '
    'as the previous 83-parameter fit. Both choices and all four RTs below 200 ms are included; 56 RTs above 1.8 s are excluded. '
    'These are simultaneous visual trials. Raw trials are pooled across participants and sessions.','',
    '### Pooled RTDs, psychometric and chronometric','',
    f'![Shared-Omega Control and ASD: pooled RTDs, psychometric and chronometric]({base}/human_shared_omega_pooled_comparison.png)','',
    '*Control blue; ASD red. Both ASLs are pooled for display. Five RTD panels show |ΔdB| = 0.1, 0.3, 0.65, 1.3 and 2; '
    'data histograms use 20 ms bins. Choice proportions and raw-trial mean RTs are dots without error bars. Model curves use arithmetic '
    'posterior means, include lapses and retain 0–1.8 s normalization. RTDs/chronometrics pool ASLs and signs using retained-trial '
    'frequencies; psychometrics pool ASLs at each signed difference. Model points are connected only at measured differences.*','',
    f'[Full-size PNG]({base}/human_shared_omega_pooled_comparison.png) · [PDF]({base}/human_shared_omega_pooled_comparison.pdf)','',
    '### Parameter comparison','',
    f'![Shared-Omega human parameter comparison]({base}/human_shared_omega_parameter_comparison.png)','',
    '*Condition-specific Gamma, Gaussian μ and Gaussian σ are shown separately for ASL 20/40 cd/m². Scalar panels show Omega, '
    'starting bias and both lapse probabilities, each shared across stimuli within its own group. Points are posterior means with '
    'equal-tailed **95% variational credible intervals**. This is a **pooled-trial variational approximation**; intervals do not '
    'represent participant-to-participant variability. μ/σ refer to the underlying Gaussian, not its post-truncation moments.*','',
    f'[Full-size PNG]({base}/human_shared_omega_parameter_comparison.png) · [PDF]({base}/human_shared_omega_parameter_comparison.pdf) · '
    f'[Parameter table]({base}/parameter_summary.csv) · [Actual truncated-delay moments]({base}/actual_truncated_delay_moments.csv)','',
    '### Comparison with condition-specific Omega','',
    '| Group | Parameters | Joint log likelihood | Marginal RT log likelihood | Psychometric RMSE | Mean CDF gap | Maximum CDF gap | Pooled mean-RT RMSE (s) |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in quality.itertuples():
    lines.append(f'| {r.Group} | {r.free_parameters} | {r.joint_log_likelihood:.2f} | {r.marginal_rt_log_likelihood:.2f} | '
        f'{r.psychometric_rmse:.4f} | {r.mean_condition_cdf_gap:.4f} | {r.max_condition_cdf_gap:.4f} | {r.pooled_chronometric_rmse_s:.4f} |')
lines+=['','83 parameters: condition-specific Omega. 64 parameters: one Omega per group. Scores use identical trials, '
    'normalization and arithmetic physical posterior means. Psychometric errors and CDF discrepancies assess the 20 signed '
    'ASL conditions; chronometric errors assess the five ASL-pooled absolute differences. These are descriptive **in-sample** '
    'comparisons; ELBO differences are not interpreted as fit-quality differences.','',
    f'[Complete comparison]({base}/baseline_vs_shared_omega_fit_quality.csv) · [Condition quality]({base}/condition_fit_quality.csv)','',
    '### Likelihood, unchanged priors and initialization','',
    r'The joint likelihood is $(1-p)h_y(t)/Z_c + p q_y/1.8$, where $h_y$ is the reactive first-passage density convolved with '
    r'$N(\mu_c,\sigma_c^2)$ truncated below zero. The Gaussian density is divided by $\Phi(\mu_c/\sigma_c)$. '
    '`Z_c` is the convolved either-bound mass over 0–1.8 s for that condition. Both normalizations are recomputed for '
    'each parameter evaluation. Delay draws have no upper cutoff. Lapse RT is **Uniform(0,1.8 s)** in stored RT, '
    'without a delay shift; `p` is the lapse fraction among retained trials.','',
    '| Parameter | Hard bounds | Flat region |','|---|---|---|']
for name,prior in PRIORS.items():lines.append(f'| `{name}` | {prior["hard"]} | {prior["plausible"]} |')
lines+=['','Independent trapezoidal priors with linear tails. **The Omega prior is applied once per group.** Delay units are '
    'seconds and Omega units are s⁻¹. All prior bounds and flat regions match the latest widened-bound 83-parameter fit.','',
    'Original saved initialization is preserved for Gamma, μ, σ, starting bias and lapse parameters. Omega starts at the '
    'arithmetic mean of the 20 original initial Omegas: **0.716870754 Control**, **0.811822111 ASD**. '
    'Each μ starts at 102.787 ms (Control) or 173.426 ms (ASD); each σ starts at 50 ms; `p=0.02`, `q=0.5`.','',
    '### Convergence, posterior boundaries and correlations','',
    'Both groups began concurrently on separate eight-core CPU allocations and continued concurrently on an RTX 4090 GPU '
    'from their complete checkpoints, following a user-approved backend change. Float64, full-batch NumPyro '
    'AutoMultivariateNormal, one ELBO particle, seed 0 and one initialization per group; guide scale 0.1, '
    'ClippedAdam learning rate 0.0002 and gradient clip 1. The original 50,000–1,000,000-update rule uses 1,000-update windows: '
    'every lower mean loss resets patience, then stopping requires 25 subsequent windows plus last-three-checkpoint '
    'posterior-mean ranges below 0.1 posterior SD. Restore the selected best checkpoint and draw 10,000 joint samples.','',
    'A persistent JAX compilation cache was enabled by resuming the saved optimizer/RNG states at 7,000 Control '
    'and 9,000 ASD updates. A separate 100-update benchmark produced exactly matching losses and optimizer states '
    'with the cache. The fitter source, model, priors, seeds, settings and completed optimization history were unchanged; '
    'the resume record and pre-resume checkpoint copies are in the audit bundle.','',
    'The CPU-to-GPU continuation resumed Control at **25,000** updates and ASD at **32,000**, retaining optimizer state, '
    'random-number state and the complete loss/checkpoint history. No mathematical settings or likelihood sources changed. '
    'A 100-update comparison from an identical saved checkpoint found maximum loss difference **5.23×10⁻¹²** and '
    'optimizer-state difference **3.32×10⁻¹⁴**. Warm timings were 19.04 s CPU and 0.474 s GPU (about 40× in this benchmark; '
    'CPU was under concurrent fit load). Compilation and full-run overhead are excluded from this speed ratio. '
    'Backend records, source history and CPU checkpoint copies are preserved in the audit bundle.','',
    'A further execution-only continuation at **43,000 Control / 56,000 ASD updates** reuses the compiled 1,000-update '
    '`stable_update` loop. On identical copied GPU checkpoints, its 1,000 losses and final optimizer/RNG states matched '
    '`SVI.run` within 2.51×10⁻¹² (loss) and 8.20×10⁻¹⁴ (state). The same stopping rules, seed, priors and optimizer '
    'settings were preserved during execution changes; all previous loss history is checked against the archived checkpoints.','',
    '**User-requested Control stopping change:** after the loss plateaued, the user chose to stop Control and restore '
    'the lowest saved **1,000-update average loss**, without requiring the remaining parameter-stability gate. '
    f'Control stopped at **{reports["Control"]["completed_steps"]:,} updates** and restores **{reports["Control"]["restored_best_step"]:,}**, '
    f'with minimum average negative ELBO **{reports["Control"]["best_window_mean_negative_elbo"]:.6f}**. '
    'It is explicitly flagged as stability-unconfirmed; the later, higher-loss checkpoint is not used for figures or posterior summaries. '
    'ASD satisfied the original terminal stopping rule. No additional optimizer updates were performed during Control finalization.','',
    f'[Control stop/restore record]({base}/user_requested_finalize.json)','',
    '| Group | Final updates | Restored best | Follow-up windows | Terminal max change (SD) | Restored checkpoint locally stable | Status |',
    '|---|---:|---:|---:|---:|---|---|']
for g,c in reports.items():
    status='**User-stopped at loss plateau; parameter stability unconfirmed**' if c['flagged'] else 'Confirmation and terminal stability passed'
    lines.append(f'| {g} | {c["completed_steps"]:,} | {c["restored_best_step"]:,} | {c["confirmation_windows"]} | '
        f'{c["maximum_terminal_mean_change_sd"]:.3f} | {c["restored_checkpoint_parameter_stable"]} | {status} |')
lines+=['','Terminal and restored-checkpoint stability are distinct checks. Neither establishes a global optimum.','',
    f'[Loss traces and posterior correlations]({base}/posterior_diagnostics.png) · '
    f'[Control convergence]({base}/control/convergence_summary.json) · [ASD convergence]({base}/asd/convergence_summary.json)','',
    f'Both the **0.5th and 99.5th percentiles** were checked for all 128 parameters. **{int(near.sum())}** have an endpoint '
    f'in the outer 1% of the hard range; **{int(accum.sum())}** have at least 5% of draws in an outer 1% region. '
    'This relative-range screen identifies proximity, not proof that a bound constrains the fit. Bounds were not widened in this experiment.','',
    '| Group | Parameter | 0.5th percentile | 99.5th percentile | Hard range |','|---|---|---:|---:|---|']
nearest=pd.concat([tails.nsmallest(5,'lower_tail_distance_fraction'),tails.nsmallest(5,'upper_tail_distance_fraction')]).drop_duplicates(['Group','parameter'])
for r in nearest.itertuples():lines.append(f'| {r.Group} | `{r.parameter}` | {r.q005:.6g} | {r.q995:.6g} | [{r.hard_low:g}, {r.hard_high:g}] |')
lines+=['',f'[All parameter tails and boundary accumulation]({base}/posterior_boundary_audit.csv)','']
for g in ['Control','ASD']:
    r=correlations.loc[correlations.Group.eq(g)].iloc[0]
    lines+=[f'{g}: strongest reported correlation is `{r.parameter_1}` with `{r.parameter_2}`, **r={r.correlation:.3f}**.','']
lines+=['### Verification and audit','',
    'The independent verifier reconstructs every retained/excluded trial from the MAT source, checks the 64-dimensional '
    'guide and single Omega prior, and compares likelihoods and gradients with the existing model after setting all Omegas equal. '
    'It independently checks quadrature, density/CDF agreement, unit mass, integrated right-bound probability, finite gradients, '
    'K=8/40/80 agreement, finer time grids, 100 posterior draws per group, posterior summaries, restored checkpoints, '
    'empirical histograms, pooling weights and density/CDF chronometric means. Earlier fit artifacts are hash-checked.','',
    f'Source: `ASD_Human_Data/plot_human_shared_omega_gaussian_delay_svi.py`. Native outputs: `{out.relative_to(ROOT)}/`.','',
    f'- [Fitter]({base}/source/ASD_Human_Data/fit_human_shared_omega_gaussian_delay_svi.py) · '
    f'[Likelihood adapter]({base}/source/ASD_Human_Data/human_shared_omega_gaussian_delay_svi_utils.py) · '
    f'[Plotter]({base}/source/ASD_Human_Data/plot_human_shared_omega_gaussian_delay_svi.py) · '
    f'[Verifier]({base}/source/ASD_Human_Data/verify_human_shared_omega_gaussian_delay_svi.py).',
    f'- [Configuration/provenance]({base}/run_config.json) · [Independent verification]({base}/independent_validation.json) · '
    f'[Visual inspection]({base}/visual_inspection.json) · [Trial counts]({base}/selection_counts.csv).',
    f'- [Pooled psychometrics]({base}/pooled_psychometric.csv) · [Chronometrics]({base}/chronometric_points.csv) · '
    f'[Mixture weights]({base}/mixture_weights.csv).',
    f'- [Posterior, checkpoint, calculation and source audit ZIP]({base}/human_shared_omega_audits.zip).','',end,'']
entry='\n'.join(lines)
before=page.read_text() if page.exists() else f'# Results: {date}\n'
if start in before:
    a=before.index(start);b=before.index(end,a)+len(end)
    after=before[:a]+entry.rstrip()+before[b:]
else:after=before.rstrip()+'\n\n'+entry
page.parent.mkdir(parents=True,exist_ok=True);page.write_text(after)
index=ROOT/'docs/results/index.md';text=index.read_text();link=f'- [{date}]({date}.md)'
if link not in text:index.write_text(text.rstrip()+'\n'+link+'\n')
section_link=f'- [Human RT: one shared Omega per group]({date}.md#human-rt-one-shared-omega-per-group)'
text=index.read_text()
if section_link not in text:index.write_text(text.rstrip()+'\n'+section_link+'\n')
nav=ROOT/'mkdocs.yml';text=nav.read_text();line=f'      - "{date}": results/{date}.md'
if line not in text:
    anchor='      - Overview: results/index.md'
    assert anchor in text;nav.write_text(text.replace(anchor,anchor+'\n'+line))

# %% Keep large calculation tables/posteriors in the downloadable audit bundle.
assets.mkdir(parents=True,exist_ok=True)
direct=plot['figures']+['parameter_summary.csv','actual_truncated_delay_moments.csv',
    'baseline_vs_shared_omega_fit_quality.csv','condition_fit_quality.csv','posterior_boundary_audit.csv',
    'strongest_correlations.csv','posterior_diagnostics.png','pooled_psychometric.csv','chronometric_points.csv',
    'mixture_weights.csv','run_config.json','independent_validation.json','visual_inspection.json',
    'selection_counts.csv','plot_manifest.json','execution_schedule.json','user_requested_finalize.json']
for g in ['control','asd']:direct.append(f'{g}/convergence_summary.json')
for name in direct:
    dest=assets/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(out/name,dest)
shutil.copytree(out/'source',assets/'source',dirs_exist_ok=True)
bundle=out/'human_shared_omega_audits.zip'
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
