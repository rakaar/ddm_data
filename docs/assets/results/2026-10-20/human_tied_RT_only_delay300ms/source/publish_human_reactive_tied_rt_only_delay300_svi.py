# %%
"""Publish the 0--1.8 s RT-only fit with the requested Uniform(0,.3) delay prior."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import zipfile
import pandas as pd
from human_reactive_tied_rt_only_utils import ROOT, PARAM_NAMES, sha256

# %% Separate output and publication roots preserve the previous analyses.
OUT = ROOT/'ASD_Human_Data/reactive_tied_rt_only_svi_0_1p8s_delay300ms'
ASSETS = ROOT/'docs/assets/results/2026-10-20/human_tied_RT_only_delay300ms'
PAGE = ROOT/'docs/results/2026-10-20.md'
asset = '../assets/results/2026-10-20/human_tied_RT_only_delay300ms/'
validation = json.loads((OUT/'independent_validation.json').read_text())
visual = json.loads((OUT/'visual_inspection.json').read_text())
config = json.loads((OUT/'run_config.json').read_text())
repair = json.loads((OUT/'checkpoint_repair.json').read_text())
assert validation['status'] == visual['status'] == repair['status'] == 'passed'
quality = pd.read_csv(OUT/'RT_fit_quality.csv')
parameters = pd.read_csv(OUT/'parameter_summary.csv')
boundary = pd.read_csv(OUT/'delay_boundary_diagnostics.csv').set_index('Group')
parts = ['## Human RT-only TIED: 0–1.8 s trials and a 0–300 ms delay prior', '',
    '**Selection:** `Condition == "RT"` and `0 <= RT_s <= 1.8`. Pool raw trials across '
    'participants and sessions within each group; keep correct and incorrect trials. '
    'Observed response side is ignored in fitting. SD and sequential CB are excluded.', '',
    '| Group | Participants | All RT trials | Retained 0–1.8 s | Excluded >1.8 s |',
    '|---|---:|---:|---:|---:|',
    '| Control | 11 | 6,600 | 6,574 | 26 |',
    '| ASD | 8 | 4,800 | 4,770 | 30 |', '',
    '**All four RTs below 200 ms are retained.** The user rounded the pooled 99.5th percentile '
    '(1.792625 s) to **1.8 s**. This selection keeps 11,344 trials and excludes 56, rather '
    'than the 57 excluded by the exact percentile. Relative to the preceding 250-ms-prior '
    'analysis, the retained window changes from 0.25–1.5 to 0–1.8 s, adding eight responses '
    'below 250 ms and 109 responses between 1.5 and 1.8 s. Recorded RTs are not shifted.', '',
    '### Requested prior and causal support', '',
    'The fitted model uses **`t_E_aff ~ Uniform(0, 0.300 s)`**. The other four priors remain '
    'LogNormal(log(.2), 1.5) for `rate_lambda`, LogNormal(log(.001), 2) for `T_0` (seconds), '
    'LogNormal(log(30), 1.5) for `theta_E`, and Uniform(.05,.95) for `w`. LogNormal arguments '
    'use natural logarithms; the three positive parameters have no finite upper hard bound.', '',
    'The likelihood is zero for a delay at or above any retained RT. The earliest retained '
    'RT is **111.1898 ms Control** and **181.6110 ms ASD**. These are data-imposed posterior '
    'support limits, distinct from the requested 300 ms prior ceiling and the RT data cutoff.', '',
    'For numerical SVI, a separate guide-only prototype gives AutoMultivariateNormal a '
    'delay transform confined to the positive-likelihood interval. The actual SVI model '
    'still evaluates the Uniform(0,.300) prior with its original density. The prototype’s '
    'prior is used only to construct the guide transform. Independent checks verify the '
    'model/prototype log-density difference is exactly log(causal_upper/.300) at valid '
    'parameters, and that noncausal delays have zero likelihood despite lying inside the '
    'requested prior. No extra RTs are discarded to enlarge delay support.', '',
    '### Five shared parameters and RT-only likelihood', '',
    'Each group has five parameters shared across both ASLs and all ten signed differences: '
    '`rate_lambda`, `T_0`, `theta_E`, `w`, `t_E_aff`. Use the unchanged IPL/vanilla density '
    'and CDF primitives with `B = Mean_dB` and `D = Delta_dB` (right minus left). '
    'ASL plot labels are 20/40 cd/m²; model brightness coordinates are ≈10.892/15.089 visual dB.', '',
    '```text\n'
    'gamma = theta_E * tanh(rate_lambda * D / 17.37)\n'
    'omega = 10**(rate_lambda * B / 20) * cosh(rate_lambda * D / 17.37) / (T_0 * theta_E**2)\n'
    'Z_E = (w - 0.5) * 2 * theta_E\n'
    'first-passage time = RT_s - t_E_aff\n'
    'p(RT | B,D,retained) = [rho_right(RT) + rho_left(RT)] / Z(B,D)\n'
    'Z(B,D) = F_right(1.8) + F_left(1.8) - F_right(0) - F_left(0)\n'
    '```', '',
    'This fits the trial-level RT likelihood conditional on stimulus and retention, '
    'marginalizing response side. No histogram-error or psychometric objective is fitted.', '',
    '### Data and mean-parameter fit curves', '',
    f'![Human 0–1.8 s RT-only data and fitted curves]({asset}human_RT_only_rtd_cdf_comparison_full_0_1p8s.png)', '',
    '*Control above ASD; pooled RTD left and CDF right. Blue/orange denote ASL 20/40 cd/m². '
    'Steps show data; solid curves evaluate the model at arithmetic posterior means of the '
    'five physical parameters. Sum both response bounds and mix signed conditions using '
    'actual retained trial counts. Both data and predictions are conditional on 0–1.8 s. '
    'RTD bins are 20 ms; the model grid is 1 ms. Each panel retains its group’s recorded '
    'early RTs.*', '',
    f'[Full-size PNG]({asset}human_RT_only_rtd_cdf_comparison_full_0_1p8s.png) · '
    f'[PDF]({asset}human_RT_only_rtd_cdf_comparison_full_0_1p8s.pdf) · '
    f'[0–1.5 s display zoom]({asset}human_RT_only_rtd_cdf_comparison_zoom_0_1p5s.png)', '',
    '**Visual assessment:** '+visual['fit_assessment'], '',
    '| Group | RT log likelihood at physical means | Mean condition CDF gap | Largest condition CDF gap |',
    '|---|---:|---:|---:|']
for row in quality.itertuples():
    parts.append(f'| {row.Group} | {row.mean_parameter_RT_log_likelihood:.6g} | '
                 f'{row.mean_condition_cdf_error:.6g} | {row.max_condition_cdf_error:.6g} |')
parts += ['', 'The likelihood weights individual trials equally. The mean CDF gap is an '
    'equal-weight average over the 20 ASL × signed-difference cells. Pooled curves use actual '
    'retained stimulus frequencies. These scores use the current selection and are not '
    'directly comparable with training scores from the earlier truncated datasets.', '',
    f'![RT-only fits by absolute difference]({asset}human_RT_only_rtd_by_delta_full_0_1p8s.png)', '',
    '*Control above ASD; columns |ΔdB| = 0.1, 0.3, 0.65, 1.3, 2. Blue/orange identify ASL '
    '20/40 cd/m². Dashed steps show empirical RTDs; solid lines show the model at the physical '
    'posterior means. Pool ±ΔdB with their actual retained frequencies. Full and zoomed '
    'views retain 0–1.8 s normalization, with common axes including all peaks.*', '',
    f'[Full-size PNG]({asset}human_RT_only_rtd_by_delta_full_0_1p8s.png) · '
    f'[PDF]({asset}human_RT_only_rtd_by_delta_full_0_1p8s.pdf) · '
    f'[0–1.5 s display zoom]({asset}human_RT_only_rtd_by_delta_zoom_0_1p5s.png)', '',
    '### Posterior parameters', '',
    '| Group | Parameter | Mean | SD | Central 95% interval |', '|---|---|---:|---:|---|']
for r in parameters.itertuples():
    factor = 1000 if r.parameter in ['T_0','t_E_aff'] else 1
    name = r.parameter+(' (ms)' if factor == 1000 else '')
    parts.append(f'| {r.Group} | {name} | {r.mean*factor:.6g} | {r.sd*factor:.5g} | '
                 f'[{r.q025*factor:.5g}, {r.q975*factor:.5g}] |')
parts += ['', 'These are **variational approximations** from 10,000 constrained draws per '
    'group. Pooled-trial posterior uncertainty does not represent participant-to-participant '
    'variability. The plots use arithmetic physical means rather than posterior-averaged predictions.', '',
    '| Group | Causal delay limit (ms) | Draws within 1 ms of causal limit | Draws >299 ms |',
    '|---|---:|---:|---:|']
for group in ['Control','ASD']:
    r = boundary.loc[group]
    parts.append(f'| {group} | {r.causal_upper_s*1000:.5f} | '
        f'{r.fraction_within_1ms_of_causal_upper:.2%} | {r.fraction_within_1ms_of_upper:.2%} |')
parts += ['', '| Group | Parameter pair | Posterior correlation |', '|---|---|---:|']
for group in ['Control','ASD']:
    corr = pd.read_csv(OUT/group.lower()/'posterior_correlation.csv', index_col=0)
    pairs = sorted([(abs(corr.iloc[i,j]),PARAM_NAMES[i],PARAM_NAMES[j],corr.iloc[i,j])
                    for i in range(5) for j in range(i)], reverse=True)
    for _, a, b, r in pairs[:3]: parts.append(f'| {group} | {a}, {b} | {r:+.4f} |')
parts += ['', '### Initialization and stopping checks', '',
    'One warm start per group uses the confirmed earlier joint fit '
    '`ASD_Human_Data/reactive_tied_rt_svi_rt_le_1p5s/`, which retained the same earliest '
    'responses. Later 200/250-ms RT-only fits have incompatible delay starting values. '
    'Map the delay’s latent center and local covariance to the causal guide interval, '
    'preserving its physical starting center. Reset Adam, RNG (seed 0), losses and patience '
    'for the new objective. CPU float64; full-batch AutoMultivariateNormal; one ELBO particle; '
    'ClippedAdam learning rate .0002, clip 1; minimum 50k and maximum 1M new updates.', '',
    '**Stopping correction:** the earlier scripts additionally required backward-looking '
    'stability at the selected minimum itself. That extra gate can prevent stopping even '
    'after many post-minimum windows. This run follows the original requested rule: every '
    'lower 1,000-step mean loss resets patience; require at least 25 subsequent windows and '
    'terminal mean stability below 0.1 posterior SD over three checkpoints, then restore '
    'the minimum. The selected checkpoint’s own movement is recorded as a diagnostic. '
    'The Control optimizer/RNG/checkpoint trajectory was preserved when applying this '
    'correction; no additional statistical initialization was introduced. Earlier runs '
    'and their reported stopping rules are preserved.', '',
    '**Checkpoint repair:** the interrupted original Control worker continued writing after '
    'the resumed fit stopped. That leftover worker was terminated, and its extra iterations '
    'were archived. The original saved initialization and RNG trajectory were replayed '
    'through the accepted stop. Every loss and checkpoint mean was checked, all 10,000 '
    'posterior draws were reproduced, and the best/latest checkpoints were restored. '
    'The accepted posterior and guide files are byte-identical to their pre-repair values. '
    'The repair record and archived traces are included in the audit bundle.', '',
    '| Group | Selected best | Final checked | Later windows | Terminal stable | Selected checkpoint stable (diagnostic) | Status |',
    '|---|---:|---:|---:|---|---|---|']
for group in ['Control','ASD']:
    r = json.loads((OUT/group.lower()/'convergence_summary.json').read_text())
    status = '**flagged / unconfirmed**' if r['flagged'] else 'Required checks passed'
    parts.append(f"| {group} | {r['restored_best_step']:,} | {r['completed_steps']:,} | "
        f"{r['confirmation_windows']} | {r['terminal_parameter_stable']} | "
        f"{r['restored_checkpoint_parameter_stable']} | {status} |")
parts += ['', 'Passing these observed-window checks does not establish a global optimum '
    'or unique parameter estimates. Strong parameter correlations are shown above.', '',
    f'[Control corner]({asset}control/parameter_corner.png) · '
    f'[ASD corner]({asset}asd/parameter_corner.png) · '
    f'[Control convergence]({asset}control/convergence.png) · '
    f'[ASD convergence]({asset}asd/convergence.png)', '',
    '### Sources and audits', '',
    'Fit root: `ASD_Human_Data/reactive_tied_rt_only_svi_0_1p8s_delay300ms/`.', '',
    f'[Configuration]({asset}run_config.json) · [Protocol]({asset}analysis_protocol.json) · '
    f'[Prior/guide preflight]({asset}prior_guide_preflight.json) · '
    f'[Stopping correction]({asset}stopping_rule_correction.json) · '
    f'[Checkpoint repair]({asset}checkpoint_repair.json) · '
    f'[Selection counts]({asset}selection_counts.csv) · [Excluded rows]({asset}excluded_trials.csv) · '
    f'[Participant counts]({asset}participant_selection_counts.csv) · '
    f'[Condition counts]({asset}condition_selection_counts.csv) · '
    f'[RT fit metrics]({asset}RT_fit_quality.csv) · '
    f'[Posterior summaries]({asset}parameter_summary.csv) · '
    f'[Delay support diagnostics]({asset}delay_boundary_diagnostics.csv) · '
    f'[Independent verification]({asset}independent_validation.json) · '
    f'[Full posterior/source/audit ZIP]({asset}human_RT_only_delay300ms_audits.zip)', '']
sources = ['human_reactive_tied_rt_only_utils.py','human_reactive_tied_rt_only_support_utils.py',
           'fit_human_reactive_tied_rt_only_delay300_svi.py',
           'plot_human_reactive_tied_rt_only_delay300_svi.py',
           'verify_human_reactive_tied_rt_only_delay300_svi.py',
           'publish_human_reactive_tied_rt_only_delay300_svi.py',
           'human_tied_independent_fpt.py','human_reactive_tied_svi_utils.py']
for name in sources: parts.append(f'- `ASD_Human_Data/{name}` ([source]({asset}source/{name})).')
parts += ['', f"Independent checks verified all {validation['selected_source_rows']:,} retained "
    f"and {validation['excluded_source_rows']} excluded source identities, all four sub-200-ms "
    'responses, the actual 300-ms prior and causal guide supports, prior normalization, five '
    'free parameters, response-label invariance, finite gradients, K=40/80 series convergence, '
    'independent SciPy density/CDF agreement, conditional unit mass, posterior summaries, '
    'warm-start mapping, and every patience reset. Reconstructed '
    f"{validation['condition_CDF_checks']} condition CDF comparisons, "
    f"{validation['histogram_bins_verified']} histogram bins and "
    f"{validation['mixture_rows_verified']} mixture weights. Earlier fit/figure files were "
    'hash checked, prior documentation preserved, PNGs visually inspected, and PDF text '
    'bounds checked.', '']
section = '\n'.join(parts)
(OUT/'RT_only_delay300_result.md').write_text(section)
ASSETS.mkdir(parents=True, exist_ok=True)
for path in OUT.rglob('*'):
    if path.is_file() and path.suffix in ['.png','.pdf','.csv','.json']:
        target = ASSETS/path.relative_to(OUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,target)
(ASSETS/'source').mkdir(exist_ok=True)
for name in sources: shutil.copy2(ROOT/'ASD_Human_Data'/name,ASSETS/'source'/name)
kernel = ROOT/'fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py'
shutil.copy2(kernel,ASSETS/'source'/kernel.name)
archive = ASSETS/'human_RT_only_delay300ms_audits.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix not in ['.png','.pdf']:
            z.write(path,'fit/'+str(path.relative_to(OUT)))
    for name in sources: z.write(ROOT/'ASD_Human_Data'/name,'source/'+name)
    z.write(kernel,'source/'+kernel.name)
    for group in ['control','asd']:
        for name in ['guide_parameters.npz','posterior_samples.npz','data_manifest.json']:
            z.write(Path(config['source_fit'])/group/name,f'input_initialization/{group}/{name}')
text = PAGE.read_text();heading = '# Results — 2026-10-20\n'
assert heading in text and parts[0] not in text
assert text == (OUT/'previous_result_page.md').read_text()
PAGE.write_text(text.replace(heading,heading+'\n'+section+'\n\n---\n',1))
index = ROOT/'docs/results/index.md';lines = index.read_text().splitlines()
link = '- [2026-10-20](2026-10-20.md)'
assert any(line.startswith(link) for line in lines)
index.write_text('\n'.join(line+'; human RT-only 0–1.8 s with 300 ms delay prior'
    if line.startswith(link) else line for line in lines)+'\n')
(OUT/'publication.json').write_text(json.dumps(dict(page=str(PAGE.relative_to(ROOT)),
    published_at_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=sha256(archive),
    source_sha256=sha256(__file__)),indent=2))
print('Published',PAGE)
