# %%
"""Publish verified human DDM debugging results on 21 October 2026."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import shutil
import zipfile
import pandas as pd

# %% User's client date; earlier result pages and baseline fits are preserved.
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'ASD_Human_Data/reactive_ddm_debug_0_1p8s_delay300ms')
args=parser.parse_args();out=args.output_dir
verification=json.loads((out/'verification.json').read_text())
visual=json.loads((out/'visual_inspection.json').read_text())
assert verification['status']==visual['status']=='passed'
assert verification['figures']==visual['figures_sha256']
assets=ROOT/'docs/assets/results/2026-10-21/human_ddm_debug'
assets.mkdir(parents=True,exist_ok=True)
page=ROOT/'docs/results/2026-10-21.md'
title='Human RT debugging: plain DDM versus TIED'
quality=pd.read_csv(out/'fit_quality.csv')
low=pd.read_csv(out/'low_delta_choice_check.csv')
prefix='../assets/results/2026-10-21/human_ddm_debug/'

sources=[
 'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py',
 'ASD_Human_Data/debug_human_reactive_ddm.py',
 'ASD_Human_Data/debug_human_ddm_joint.py',
 'ASD_Human_Data/debug_human_ddm_shared_scale.py',
 'ASD_Human_Data/analyze_human_reactive_ddm_debug.py',
 'ASD_Human_Data/verify_human_reactive_ddm_debug.py',
 'ASD_Human_Data/publish_human_reactive_ddm_debug.py',
 'ASD_Human_Data/human_reactive_tied_rt_only_utils.py',
 'ASD_Human_Data/human_reactive_tied_svi_utils.py',
 'fit_each_condn/svi_gamma_omega_likelihood_utils.py',
 'fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py',
 'fit_animal_by_animal/time_vary_norm_utils.py',
]
source_hashes={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in sources}
(out/'source_audit.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
for source,target in [('human_ddm_debug_run.log','run_rt_only.log'),('human_ddm_joint_debug_run.log','run_joint.log'),
                      ('human_ddm_shared_debug_run.log','run_shared_scale.log'),('human_ddm_debug_analysis.log','analysis.log'),
                      ('human_ddm_debug_verify.log','verification.log')]:
    shutil.copy2(Path('/tmp')/source,out/target)
for p in out.rglob('*'):
    if p.is_file() and p.suffix in ['.png','.pdf','.csv','.json']:
        target=assets/p.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for rel in sources:
    target=assets/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)

# %% Package calculations, all optimizer starts, source, original rows and baseline draws.
archive=assets/'human_ddm_debug_audits.zip'
baseline=Path(json.loads((out/'protocol.json').read_text())['baseline_dir'])
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file() and p.suffix not in ['.png','.pdf','.zip'] and p.name not in ['publication.json','completion.json','publication_validation.json','mkdocs_build.log']:
            z.write(p,'diagnostic/'+str(p.relative_to(out)))
    for rel in sources:z.write(ROOT/rel,'source/'+rel)
    for rel in ['simultaneous_rt_trials.csv','simultaneous_rt_extraction.json','table_visualTask_15nov2026.mat']:
        z.write(ROOT/'ASD_Human_Data'/rel,'inputs/'+rel)
    for group in ['control','asd']:
        for name in ['posterior_samples.npz','guide_parameters.npz','posterior_summary.csv','data_manifest.json','convergence_summary.json']:
            z.write(baseline/group/name,'baseline/'+group+'/'+name)
    z.write(baseline/'run_config.json','baseline/run_config.json')

def figure(name,caption):
    return (f'![{caption.split(".")[0]}]({prefix}{name}.png)\n\n*{caption}*\n\n'
            f'[Full-size PNG]({prefix}{name}.png) · [PDF]({prefix}{name}.pdf)\n\n')

rows=[]
for label,display in [('SVI_mean','TIED: original SVI physical means'),('SVI_predictive','TIED: posterior predictive'),
    ('TIED_RT','TIED: RT-only MLE'),('DDM_shared_RT','Plain DDM: shared scale, RT only'),
    ('DDM_free_RT','Plain DDM: free scales, RT only'),('TIED_joint','TIED: RT + choice MLE'),
    ('DDM_shared_joint','Plain DDM: shared scale, RT + choice'),('DDM_free_joint','Plain DDM: free scales, RT + choice')]:
    q=quality.loc[quality.model.eq(label)].set_index('Group')
    rows.append(f'| {display} | {int(q.n_parameters.iloc[0])} | {q.loc["Control","mean_condition_cdf_error"]:.4f} | {q.loc["ASD","mean_condition_cdf_error"]:.4f} | {q.loc["Control","psychometric_RMSE"]:.4f} | {q.loc["ASD","psychometric_RMSE"]:.4f} |')
table='\n'.join(rows)
accuracy=[]
for group in ['Control','ASD']:
    q=low.loc[low.Group.eq(group)&low.model.eq('DDM_shared_RT')].iloc[0]
    accuracy.append(f'| {group} | {int(q.n_trials):,} | {q.observed_accuracy*100:.2f}% | {q.predicted_accuracy*100:.5f}% |')
section=f'''## {title}

**Finding:** a standard plain DDM fits RT shape much better after freeing the stimulus drifts, but its RT-only fit implies nearly deterministic choices at the smallest difference. Including observed choices brings back broad RT distributions. The present evidence points to restrictions in the pooled, pure-reactive, shared-delay model; it does not establish a general failure of DDMs or a first-passage likelihood bug.

### Selection, normalization and fitting method

Use `Condition == "RT"` and **`0 <= RT_s <= 1.8`**: **6,574 Control trials from 11 participants** and **4,770 ASD trials from 8 participants**. Retain correct and incorrect trials, both simultaneous sessions, and **all four RTs below 200 ms**. Pool raw trials within group; exclude SD and sequential CB. No extra exclusions, shifts, or JND eligibility filter. The same 56 RTs above 1.8 s remain excluded.

Stimulus coordinates remain `B = Mean_dB` and signed `D = Delta_dB` (right minus left). ASL 20/40 cd/m² are plot labels. RT-only fits use `f_right(RT) + f_left(RT)`; joint fits use the density of the observed side. Both divide by the **either-bound mass on 0–1.8 s**, separately for each stimulus condition. Pooled curves use actual retained signed-condition frequencies.

**Interpretation of “pure DDM”:** the standard diagnostic fits 20 free stimulus drifts with one common boundary/noise time scale, one starting bias, and one delay per group. An additional diagnostic frees the time scale per condition. These are **direct maximum-likelihood fits without priors**, using multi-start L-BFGS-B and exact float64 gradients, to separate likelihood/model restrictions from SVI and prior effects. They are not new variational posteriors. The original SVI posterior and requested priors are preserved for comparison.

| Model | Free parameters per group | Shared quantities |
|---|---:|---|
| Original IPL/vanilla TIED | 5 | `rate_lambda`, `T_0`, `theta_E`, `w`, `t_E_aff` |
| Standard plain DDM | 23 | 20 free Γ; one common Ω, `w`, `t_E_aff` |
| DDM with free condition scales | 42 | 20 free Γ, 20 free Ω; common `w`, `t_E_aff` |

In unit-diffusion coordinates, the common-scale DDM has boundary magnitude `a = 1/sqrt(Ω)`, drift `v = Γ*sqrt(Ω)`, and start `(2w-1)*a`. Thus it has a common boundary and a free drift for every ASL × signed-difference condition. Γ direction is not identified by RT alone; the RT-only choice illustration uses the stimulus-consistent branch. Joint DDM fits allow either drift sign.

Delay bounds remain **0–300 ms**. Causality still requires a shared delay below the fastest retained RT: **111.1898 ms Control**, **181.6110 ms ASD**. The optimizer uses a 0.1 μs margin below that causal limit. Widening the nominal upper bound cannot remove this data-imposed restriction.

### RT-only fit comparison

'''+figure('human_rt_only_ddm_debug','Control above ASD; ASL 20 and 40 in separate columns. Steps are data in 20-ms bins. Gray dashed: original SVI physical means; purple: directly re-optimized five-parameter TIED; teal: standard plain DDM with a shared time scale; green: DDM with free condition scales. Every density integrates to one on the retained 0–1.8 s window. Plain DDMs improve shape substantially but still underestimate some sharp peaks.')+f'''### Quantitative diagnostics

CDF error is the largest absolute empirical/model CDF difference within each condition, averaged **equally across the 20 signed stimulus conditions**. The fitted likelihood itself weights every retained trial. Psychometric RMSE averages the 20 retained-window probabilities. These are training-fit diagnostics with different parameter counts, not an out-of-sample model-selection result.

| Model | Parameters | Control mean CDF error | ASD mean CDF error | Control choice RMSE | ASD choice RMSE |
|---|---:|---:|---:|---:|---:|
{table}

The five-parameter TIED fit improves only modestly after direct likelihood optimization. Five starts were tried per group; all five Control starts and four ASD starts reached the best TIED basin. The other ASD start reached a poorer local result. Free-scale DDM RT fits also have multiple local results, all retained in the audit. No global-optimum claim is made.

Control's TIED MLE approaches the small-λ limit: λ reaches the diagnostic computational lower limit `1e-4`. Reducing λ to `1e-6` while preserving the effective drift slope and reference time scale changes log likelihood by only **{verification['lambda_lower_bound_sensitivity'][0]['log_likelihood_change']:.6f}** and the largest CDF value by **{verification['lambda_lower_bound_sensitivity'][0]['max_cdf_change']:.3g}**. This numerical limit does not explain the large fit discrepancy. These are MLE computational bounds, not new priors; see the protocol for all ranges.

### Why RT-only success is insufficient

At **|ΔdB| = 0.1**, the standard RT-only DDM uses strong drift magnitudes to obtain narrow RT distributions. Interpreting those drifts in the natural stimulus direction gives:

| Group | Trials at |ΔdB| = 0.1 | Observed accuracy | RT-only DDM predicted accuracy |
|---|---:|---:|---:|
'''+ '\n'.join(accuracy)+'''\n
The observed choices are much less deterministic. This is a timing–choice trade-off: a strong reactive drift narrows RTs but predicts high accuracy. With weak drift, decision times are broader; a larger common nondecision delay could shorten the decision-time component, but the earliest responses constrain that delay. This is a model-based explanation of these fits, not proof that all human responses follow one mechanism.

'''+figure('human_ddm_choice_debug','Data show retained-window P(right) with binomial sample SE. Dotted teal is the RT-only standard DDM, which did not fit choices. Other curves are joint RT-and-choice fits. Conditional probabilities are integrated right-bound mass divided by either-bound retained mass. Joint fits recover graded choices; RT-only drift direction is not identifiable from RT observations alone.')+'''### Adding choice to the likelihood

'''+figure('human_joint_ddm_debug','Same rows, window and group pooling, now optimizing the joint RT-and-observed-choice likelihood. Purple: TIED; teal: standard DDM with a shared scale; green: free condition scales. Accommodating the observed error rates broadens the fitted RTDs again. Free condition scales help but do not recover the sharp data peaks completely.')+'''### Stimulus mapping versus distribution shape

'''+figure('human_ddm_rt_by_delta_debug','Two rows: Control then ASD; columns are the five absolute differences. Blue/orange are ASL 20/40. Steps show pooled ±D data, solid curves show the standard RT-only DDM and dashed curves the RT-only TIED MLE. Fits themselves used all 20 signed conditions; only display mixtures pool signs.')+figure('human_ddm_parameter_constraints_debug','Derived |Γ| and Ω averaged across the two signed conditions at each absolute difference. TIED links drift strength tightly to stimulus difference, while successful RT-only plain-DDM fits need strong drift even at small differences. The plotted averages are parameter summaries, not an additional fitting objective.')+'''### Posterior-mean plotting check

'''+figure('human_svi_predictive_ddm_debug','Gray dashed curves use arithmetic means of the original five physical parameters. Black dotted curves average complete model predictions across 128 reproducibly selected joint variational posterior draws, preserving parameter correlations. Averaging predictions does not rescue the original RT shape. Variational uncertainty in this pooled model does not describe between-participant variability.')+'''### Paper-code comparison and remaining hypotheses

The human adapter, the direct Γ/Ω primitives, an independent SciPy image/eigenfunction calculation, and the original paper-code evidence kernels agree at corresponding parameter values. The original paper's `time_vary_norm_utils.py` also contains the proactive/reactive race. After summing choices, its RT density has the form

```text
f_RT(t) = f_A(t) * [1 - F_E(t)] + f_E(t) * [1 - F_A(t)]
```

with the appropriate timing offsets. This adds proactive responses and suppresses late reactive responses, so removing it can materially change RT shape. The local condition-delay animal fitter also has a separate `t_E_aff` for each stimulus condition; the current human baseline has one scalar delay per group. Therefore the successful animal model and the present five-parameter pure-reactive baseline impose different timing restrictions. The mechanistic constraint and race interpretation are described in the [2019 TIED paper](https://www.nature.com/articles/s41593-019-0439-7) and [2021 proactive/reactive paper](https://www.nature.com/articles/s41467-021-27302-8).

The results justify a controlled proactive-plus-reactive fit as the next model check. Alternative timing explanations include a small fast-response component, variable nondecision time, and participant/session heterogeneity. These alternatives were not fitted here; no new trials were removed. A better free-DDM fit alone does not imply a likelihood bug, because freeing the stimulus parameters removes TIED's cross-condition constraints.

### Sources and audit

Sources: `ASD_Human_Data/debug_human_reactive_ddm.py`, `debug_human_ddm_shared_scale.py`, `debug_human_ddm_joint.py`, `human_plain_ddm_diagnostics_utils.py`, `analyze_human_reactive_ddm_debug.py`, `verify_human_reactive_ddm_debug.py` and `publish_human_reactive_ddm_debug.py` under `ASD_Human_Data/`.

'''+f'''[RT-only fitter]({prefix}source/ASD_Human_Data/debug_human_reactive_ddm.py) · [Standard DDM control]({prefix}source/ASD_Human_Data/debug_human_ddm_shared_scale.py) · [Joint fitter]({prefix}source/ASD_Human_Data/debug_human_ddm_joint.py) · [Diagnostic kernels]({prefix}source/ASD_Human_Data/human_plain_ddm_diagnostics_utils.py) · [Analysis and plots]({prefix}source/ASD_Human_Data/analyze_human_reactive_ddm_debug.py) · [Verifier]({prefix}source/ASD_Human_Data/verify_human_reactive_ddm_debug.py) · [Paper-code kernels]({prefix}source/fit_animal_by_animal/time_vary_norm_utils.py)

[Fit-quality table]({prefix}fit_quality.csv) · [Condition parameters]({prefix}condition_parameters.csv) · [Condition diagnostics]({prefix}condition_fit_quality.csv) · [Small-difference choice check]({prefix}low_delta_choice_check.csv) · [Original trial identities]({prefix}selected_trials.csv) · [Protocol and bounds]({prefix}protocol.json) · [Independent verification]({prefix}verification.json) · [Visual inspection]({prefix}visual_inspection.json) · [Source hashes]({prefix}source_audit.json) · [Complete audit bundle]({prefix}human_ddm_debug_audits.zip)

The audit bundle includes every optimization start and trace, fitted vectors, derived curves and metrics, source scripts, original MAT/CSV provenance, and the baseline variational draws used for posterior prediction. Verification checks all 11,344 original MAT identities, group membership, preserved early trials, column mapping, finite gradients, K=40/80 agreement, CDF/PDF consistency, conditional unit mass, old paper kernels, empirical histograms and probabilities, likelihood scores, six PNG/PDF pairs, PDF text bounds, and unchanged baseline fit hashes.

'''
# Escape the vertical bars inside a Markdown table cell.
section=section.replace('Trials at |ΔdB| = 0.1','Trials at absolute ΔdB = 0.1')
heading='# Results — 2026-10-21\n\n'
if page.exists():
    old=page.read_text()
    if f'## {title}' in old:
        before,rest=old.split(f'## {title}',1)
        tail='\n## '+rest.split('\n## ',1)[1] if '\n## ' in rest else ''
        page.write_text(before+section+tail)
    else:page.write_text(old.rstrip()+'\n\n'+section)
else:page.write_text(heading+section)

index=ROOT/'docs/results/index.md';text=index.read_text()
entry='- [2026-10-21](2026-10-21.md) — Human TIED/DDM debugging: likelihood checks, RT-only and joint fits, timing–choice trade-off.'
if '(2026-10-21.md)' not in text:index.write_text(text.rstrip()+'\n'+entry+'\n')
nav=ROOT/'mkdocs.yml';text=nav.read_text();entry='      - "2026-10-21": results/2026-10-21.md\n'
if 'results/2026-10-21.md' not in text:
    anchor='      - "2026-10-20": results/2026-10-20.md\n';assert anchor in text
    nav.write_text(text.replace(anchor,anchor+entry))
readme=ROOT/'ASD_Human_Data/README.md';text=readme.read_text()
if '### Human TIED/DDM debugging on unchanged 0--1.8 s trials' not in text:
    readme.write_text(text.rstrip()+'''\n\n### Human TIED/DDM debugging on unchanged 0--1.8 s trials

`reactive_ddm_debug_0_1p8s_delay300ms/` contains direct-likelihood diagnostic
fits, not new SVI posteriors: the same five-parameter TIED model, a standard
23-parameter DDM (20 stimulus drifts plus shared Omega/start/delay), and a
42-parameter DDM (20 Gamma, 20 Omega, shared start/delay). Run RT-only and
joint-choice controls on exactly the existing 11,344 rows, keep all four RTs
below 200 ms, retain the requested 0--300 ms delay bounds, and normalize each
stimulus condition on 0--1.8 s. The actual causal delay support remains below
the minimum group RT. Multiple L-BFGS-B starts and float64 gradients test
optimization separately from the old variational approximation and priors.

Run `debug_human_reactive_ddm.py`, `debug_human_ddm_joint.py`, then
`debug_human_ddm_shared_scale.py` with `.venv/bin/python`. Analyze with
`analyze_human_reactive_ddm_debug.py` and independently verify with
`verify_human_reactive_ddm_debug.py`. All scripts live in `ASD_Human_Data/`.
The stable diagnostic Wiener kernels are in `human_plain_ddm_diagnostics_utils.py`;
check them against the paper/current primitives and independent SciPy before
interpreting a score. Publish verified six-panel-type PNG/PDF outputs with
`publish_human_reactive_ddm_debug.py` on the 21 October result page. The bundle
includes every start/trace, exact rows, parameters, predictions and baseline
posterior samples used for a joint-draw posterior-predictive check.

Plain DDM RT-only fits improve shape but imply nearly deterministic choices
at small differences; joint fits recover graded choices and broader RTDs.
This isolates a timing/choice and cross-stimulus-constraint issue, without
claiming global optimization, predictive superiority, or failure of all DDMs.
The original baseline and earlier analyses remain reference results.
''')
publication=dict(page=str(page.relative_to(ROOT)),assets=str(assets.relative_to(ROOT)),
                 archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 published_at_utc=datetime.now(timezone.utc).isoformat(),sources=source_hashes)
(out/'publication.json').write_text(json.dumps(publication,indent=2)+'\n')
print('Published',page,flush=True)
print('Archive SHA256',publication['archive_sha256'],flush=True)
