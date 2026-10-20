# %%
"""Publish verified RT-only fits, the 200-ms delay bound and common-RT comparisons."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import zipfile
import pandas as pd
from human_reactive_tied_rt_only_utils import ROOT, PARAM_NAMES, sha256

OUT=ROOT/'ASD_Human_Data/reactive_tied_rt_only_svi_0p2_1p5s_delay200ms'
ASSETS=ROOT/'docs/assets/results/2026-10-20/human_tied_RT_only_delay200ms'
PAGE=ROOT/'docs/results/2026-10-20.md'
validation=json.loads((OUT/'independent_validation.json').read_text())
visual=json.loads((OUT/'visual_inspection.json').read_text())
assert validation['status']==visual['status']=='passed'
quality=pd.read_csv(OUT/'common_RT_fit_quality.csv')
parameters=pd.read_csv(OUT/'parameter_summary.csv')
boundary=pd.read_csv(OUT/'delay_boundary_diagnostics.csv').set_index('Group')
asset='../assets/results/2026-10-20/human_tied_RT_only_delay200ms/'
parts=['## Human RT-only TIED: 200 ms delay bound and 0.2–1.5 s trials','',
    '**Question:** can the reactive model match RT distributions when observed choices are ignored '
    'and the shared nondecision delay may reach 200 ms?','',
    '**Selection:** `Condition == "RT"` and `0.2 <= RT_s <= 1.5`. Pool raw trials across '
    'participants and sessions, including correct and incorrect choices. **Observed response '
    'side does not enter the fit.** SD and sequential CB are excluded.','',
    '| Group | Participants | All RT trials | Prior ≤1.5 s analysis | New 0.2–1.5 s analysis |',
    '|---|---:|---:|---:|---:|',
    '| Control | 11 | 6,600 | 6,531 | 6,528 |',
    '| ASD | 8 | 4,800 | 4,704 | 4,703 |','',
    'The user approved removing **four responses below 200 ms** (3 Control, 1 ASD) in addition '
    'to the 165 responses above 1.5 s already removed. Those early trials otherwise force a '
    'causal shared delay below 111/182 ms despite a wider prior. The fastest retained responses '
    'are 226.246 ms Control and 277.840 ms ASD; both groups now use '
    '**`t_E_aff ~ Uniform(0, 0.200 s)`**. There is no imputation or change to recorded RTs.','',
    '### RT-only objective and model','',
    'For each trial, retain its `Mean_dB` brightness and signed `Delta_dB` stimulus coordinates, '
    'and marginalize the unobserved response side by adding the existing left/right first-passage '
    'densities. This is a trial-level RT likelihood, conditional on the stimulus and retained '
    'RT window. It does not fit a histogram-error score.','',
    '```text\n'
    'p(RT | B,D,retained) = [rho_right(RT) + rho_left(RT)] / Z(B,D)\n'
    'Z(B,D) = F_right(1.5) + F_left(1.5) - F_right(0.2) - F_left(0.2)\n'
    'first-passage time = RT_s - t_E_aff\n'
    '```','',
    'Keep five parameters shared across both ASLs and all ten signed differences within each '
    'group: `rate_lambda`, `T_0`, `theta_E`, `w`, `t_E_aff`. The IPL/vanilla equations and other '
    'four priors are unchanged: lognormal λ/T₀/θ with natural-log means log(.2)/log(.001)/log(30) '
    'and log SDs 1.5/2/1.5; `w ~ Uniform(.05,.95)`. Plot ASL labels are 20/40 cd/m²; model '
    'levels are ≈10.892/15.089 visual dB.','',
    '**This run changes three things relative to the previous joint fit:** choices are '
    'marginalized, the four early trials are removed, and the delay prior is widened. A change '
    'in fit quality cannot be attributed to ignoring choices alone. All numerical comparisons '
    'below evaluate both parameter sets on the same new rows, with the same marginal RT '
    'likelihood and 0.2–1.5 s normalization.','',
    '### Optimization and saved posterior','',
    'One full-rank variational initialization per group transfers the previous joint-fit guide. '
    'Map the delay’s latent center and local covariance to its new 0–200 ms support, preserving '
    'the physical starting center; reset Adam, RNG (seed 0), losses and patience. CPU float64; '
    'AutoMultivariateNormal; full batches; one ELBO particle; ClippedAdam rate .0002, clip 1. '
    'Check K=40 against K=80.','',
    'At least 50,000 and at most 1,000,000 new updates. Every lower 1,000-step mean loss resets '
    'patience; require 25 subsequent windows and terminal/restored mean stability (<0.1 '
    'posterior SD over three checkpoints). Restore the selected best checkpoint. This checks '
    'recent windows and does not prove a unique/global optimum.','',
    '| Group | Selected best | Final checked | Later windows | Status |',
    '|---|---:|---:|---:|---|']
for group in ['Control','ASD']:
    r=json.loads((OUT/group.lower()/'convergence_summary.json').read_text())
    status='25-window/local stability checks passed' if not r['flagged'] else '**flagged / unconfirmed**'
    parts.append(f"| {group} | {r['restored_best_step']:,} | {r['completed_steps']:,} | {r['confirmation_windows']} | {status} |")
    if r['flagged']:
        reasons=[]
        if not r['restored_checkpoint_parameter_stable']: reasons.append('selected-checkpoint mean stability')
        if not r['terminal_parameter_stable']: reasons.append('terminal mean stability')
        parts.extend(['',f"**{group} is provisional after {r['completed_steps']:,} updates:** "+', '.join(reasons)+' did not pass. Do not describe this posterior as converged.',''])
        history=pd.read_csv(OUT/group.lower()/'convergence.csv')
        selected=history.loc[history.step.eq(r['restored_best_step'])].iloc[0]
        changes=selected[[name+'_change_sd' for name in PARAM_NAMES]]
        parameter=changes.astype(float).idxmax().removesuffix('_change_sd')
        parts.extend([f"At the selected checkpoint the largest three-checkpoint mean movement was "
            f"{float(changes.astype(float).max()):.3f} posterior SD ({parameter}), versus the 0.1 SD criterion. "
            'The post-best loss-window counter and this parameter check are separate.',''])
parts+=['','### RTDs and CDFs','',
    f'![RT-only TIED versus previous joint-fit predictions]({asset}human_RT_only_rtd_cdf_comparison_full_0p2_1p5s.png)','',
    '*Control above ASD; pooled RTD left and RT CDF right. Blue/orange: ASL 20/40 cd/m². '
    'Solid curves: RT-only fit at arithmetic physical posterior means; dotted curves: previous '
    'joint-fit parameters, recomputed on the same selected trials and RT window. Thin steps: '
    'data. Sum both response bounds and mix signed conditions with actual retained trial counts. '
    'No psychometric/choice score is used.*','',
    f'[Full-size PNG]({asset}human_RT_only_rtd_cdf_comparison_full_0p2_1p5s.png) · '
    f'[PDF]({asset}human_RT_only_rtd_cdf_comparison_full_0p2_1p5s.pdf) · '
    f'[0.2–1 s display zoom]({asset}human_RT_only_rtd_cdf_comparison_zoom_0p2_1s.png)','',
    '| Group | Common RT metric | Previous joint-fit parameters | RT-only fit | Change |',
    '|---|---|---:|---:|---:|']
metrics={'mean_parameter_RT_log_likelihood':'Marginal RT log likelihood (higher better)',
         'mean_condition_cdf_error':'Mean signed-condition CDF gap',
         'max_condition_cdf_error':'Largest signed-condition CDF gap'}
for group in ['Control','ASD']:
    old=quality.loc[quality.Group.eq(group)&quality.fit.eq('previous_joint')].iloc[0]
    new=quality.loc[quality.Group.eq(group)&quality.fit.eq('rt_only')].iloc[0]
    for key,label in metrics.items():
        parts.append(f'| {group} | {label} | {old[key]:.6g} | {new[key]:.6g} | {new[key]-old[key]:+.6g} |')
parts+=['','The mean CDF gap is an equal-weight average over the 20 ASL × signed-difference cells. '
    'The likelihood gives each retained trial one contribution; RTD/CDF mixtures use actual '
    'retained stimulus frequencies.','']
parts+=['','### RTDs by absolute difference','',
    f'![RT-only fits by absolute difference]({asset}human_RT_only_rtd_by_delta_full_0p2_1p5s.png)','',
    '*Control above ASD; columns |ΔdB| = 0.1, 0.3, 0.65, 1.3, 2. Dashed steps are '
    '20 ms empirical histograms; solid curves are the RT-only model on a 1 ms grid. '
    'Blue/orange identify ASL 20/40 cd/m². Pool ±ΔdB with actual retained counts; '
    'normalization remains 0.2–1.5 s in the display zoom. Matching axes include all peaks.*','',
    f'[Full-size PNG]({asset}human_RT_only_rtd_by_delta_full_0p2_1p5s.png) · '
    f'[PDF]({asset}human_RT_only_rtd_by_delta_full_0p2_1p5s.pdf) · '
    f'[0.2–1 s display zoom]({asset}human_RT_only_rtd_by_delta_zoom_0p2_1s.png)','',
    '### Delay posterior and the requested 200 ms bound','',
    '| Group | Delay mean (ms) | SD (ms) | Central 95% (ms) | Draws above 199 ms | Draws above 195 ms |',
    '|---|---:|---:|---|---:|---:|']
for group in ['Control','ASD']:
    r=boundary.loc[group]
    parts.append(f'| {group} | {r.delay_mean_s*1000:.5f} | {r.delay_sd_s*1000:.5f} | '
        f'[{r.delay_q025_s*1000:.5f}, {r.delay_q975_s*1000:.5f}] | '
        f'{r.fraction_within_1ms_of_upper:.2%} | {r.fraction_within_5ms_of_upper:.2%} |')
parts+=['','A posterior concentrated near 200 ms is constrained by the requested prior cap. '
    'It should not be described as an unconstrained estimate of the nondecision delay. '
    'The cap is marked on the corner and parameter-trace figures.','',
    '### All five posterior parameters','',
    '| Group | Parameter | Mean | SD | Central 95% |', '|---|---|---:|---:|---|']
for r in parameters.itertuples():
    factor=1000 if r.parameter in ['T_0','t_E_aff'] else 1
    name=r.parameter+' (ms)' if factor==1000 else r.parameter
    parts.append(f'| {r.Group} | {name} | {r.mean*factor:.6g} | {r.sd*factor:.5g} | '
                 f'[{r.q025*factor:.5g}, {r.q975*factor:.5g}] |')
parts+=['','These are **variational approximations** from 10,000 constrained draws per group. '
    'Pooled-trial posterior uncertainty does not represent participant-to-participant variability.','',
    '| Group | Parameter pair | Posterior correlation |', '|---|---|---:|']
for group in ['Control','ASD']:
    corr=pd.read_csv(OUT/group.lower()/'posterior_correlation.csv',index_col=0)
    pairs=sorted([(abs(corr.iloc[i,j]),PARAM_NAMES[i],PARAM_NAMES[j],corr.iloc[i,j])
                  for i in range(5) for j in range(i)],reverse=True)
    for _,a,b,r in pairs[:3]: parts.append(f'| {group} | {a}, {b} | {r:+.4f} |')
parts+=['',f'[Control corner]({asset}control/parameter_corner.png) · '
    f'[ASD corner]({asset}asd/parameter_corner.png) · '
    f'[Control loss/parameter traces]({asset}control/convergence.png) · '
    f'[ASD loss/parameter traces]({asset}asd/convergence.png)','',
    'The strong λ/T₀/θ correlations and gradual movement in their traces remain. Passing '
    'the recent-window checks does not establish unique parameter estimates or global optimality.','',
    '### Sources and independent checks','',
    'Fit root: `ASD_Human_Data/reactive_tied_rt_only_svi_0p2_1p5s_delay200ms/`. '
    'Previous joint fit: `ASD_Human_Data/reactive_tied_rt_svi_rt_le_1p5s/`.','',
    f'[Selected/excluded counts]({asset}selection_counts.csv) · '
    f'[Excluded trial identities]({asset}excluded_trials.csv) · '
    f'[Participant counts]({asset}participant_selection_counts.csv) · '
    f'[Condition counts]({asset}condition_selection_counts.csv) · '
    f'[Common RT likelihood/CDF metrics]({asset}common_RT_fit_quality.csv) · '
    f'[Delay-boundary diagnostics]({asset}delay_boundary_diagnostics.csv) · '
    f'[Posterior summaries]({asset}parameter_summary.csv) · '
    f'[Independent validation]({asset}independent_validation.json) · '
    f'[Full posterior/source/audit ZIP]({asset}human_RT_only_delay200ms_audits.zip)','']
sources=['human_reactive_tied_rt_only_utils.py','fit_human_reactive_tied_rt_only_svi.py',
         'plot_human_reactive_tied_rt_only_svi.py','verify_human_reactive_tied_rt_only_svi.py',
         'publish_human_reactive_tied_rt_only_svi.py','human_tied_independent_fpt.py',
         'human_reactive_tied_svi_utils.py']
for name in sources: parts.append(f'- `ASD_Human_Data/{name}` ([source]({asset}source/{name})).')
parts+=['',f"Independent checks matched all 11,231 retained and 169 excluded source identities; "
    f"flipping/adding response labels left the RT likelihood and gradients unchanged. "
    f"Checks verified the five free parameters and 200 ms support, posterior summaries/covariances, "
    f"warm-start coordinate mapping, each patience reset, finite gradients, K=40/80 convergence, "
    f"SciPy density/CDF agreement and unit conditional mass. Reconstructed "
    f"{validation['condition_CDF_checks']} condition-wise CDF checks, {validation['histogram_bins_verified']} "
    f"histogram bins and {validation['mixture_rows_verified']} mixture weights. All "
    f"{validation['prior_artifacts_hash_verified']} prior fit/asset files are byte-identical. "
    f"PNGs were visually inspected and PDFs passed text-bound checks.",'']
control_old=quality.loc[quality.Group.eq('Control')&quality.fit.eq('previous_joint')].iloc[0]
control_new=quality.loc[quality.Group.eq('Control')&quality.fit.eq('rt_only')].iloc[0]
asd_old=quality.loc[quality.Group.eq('ASD')&quality.fit.eq('previous_joint')].iloc[0]
asd_new=quality.loc[quality.Group.eq('ASD')&quality.fit.eq('rt_only')].iloc[0]
asdr=json.loads((OUT/'asd/convergence_summary.json').read_text())
outcome=(f"**Outcome:** Control's average condition-wise CDF gap changes from "
    f"{control_old.mean_condition_cdf_error:.3f} to {control_new.mean_condition_cdf_error:.3f}; "
    f"ASD's changes from {asd_old.mean_condition_cdf_error:.3f} to {asd_new.mean_condition_cdf_error:.3f}. "
    'At arithmetic physical posterior means, a substantial RTD shape mismatch remains. '
    'The delay posteriors are close to the requested 200 ms cap. ')
if asdr['flagged']: outcome+='**ASD remains provisional because its convergence checks did not all pass.**'
parts[4:4]=[outcome,'']
section='\n'.join(parts)
(OUT/'RT_only_result.md').write_text(section)
ASSETS.mkdir(parents=True,exist_ok=True)
for path in OUT.rglob('*'):
    if path.is_file() and path.suffix in ['.png','.pdf','.csv','.json']:
        target=ASSETS/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
(ASSETS/'source').mkdir(exist_ok=True)
for name in sources: shutil.copy2(ROOT/'ASD_Human_Data'/name,ASSETS/'source'/name)
kernel=ROOT/'fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py'
shutil.copy2(kernel,ASSETS/'source'/kernel.name)
archive=ASSETS/'human_RT_only_delay200ms_audits.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix not in ['.png','.pdf']: z.write(path,'fit/'+str(path.relative_to(OUT)))
    for name in sources: z.write(ROOT/'ASD_Human_Data'/name,'source/'+name)
    z.write(kernel,'source/'+kernel.name)
    previous=ROOT/'ASD_Human_Data/reactive_tied_rt_svi_rt_le_1p5s'
    for group in ['control','asd']:
        for name in ['guide_parameters.npz','posterior_samples.npz','data_manifest.json']:
            z.write(previous/group/name,f'input_previous_joint/{group}/{name}')
text=PAGE.read_text();heading='# Results — 2026-10-20\n'
assert heading in text and parts[0] not in text
(OUT/'result_book_previous_page.md').write_text(text)
PAGE.write_text(text.replace(heading,heading+'\n'+section+'\n\n---\n',1))
index=ROOT/'docs/results/index.md';text=index.read_text();link='- [2026-10-20](2026-10-20.md)'
assert link in text
lines=text.splitlines()
lines=[line+'; human RT-only fits with 200 ms delay' if line.startswith(link) else line for line in lines]
index.write_text('\n'.join(lines)+'\n')
(OUT/'publication.json').write_text(json.dumps(dict(page=str(PAGE.relative_to(ROOT)),
    published_at_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=sha256(archive),source_sha256=sha256(__file__)),indent=2))
print('Published',PAGE)
