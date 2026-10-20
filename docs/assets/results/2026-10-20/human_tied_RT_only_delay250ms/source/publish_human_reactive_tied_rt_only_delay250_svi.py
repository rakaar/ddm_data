# %%
"""Publish the verified 250-ms RT-only sensitivity fit alongside the 200-ms fit."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import zipfile
import pandas as pd
from human_reactive_tied_rt_only_utils import ROOT, PARAM_NAMES, sha256

# %% Fixed analysis and publication roots; preserve all earlier result entries.
OUT = ROOT/'ASD_Human_Data/reactive_tied_rt_only_svi_0p25_1p5s_delay250ms'
ASSETS = ROOT/'docs/assets/results/2026-10-20/human_tied_RT_only_delay250ms'
PAGE = ROOT/'docs/results/2026-10-20.md'
asset = '../assets/results/2026-10-20/human_tied_RT_only_delay250ms/'
validation = json.loads((OUT/'independent_validation.json').read_text())
visual = json.loads((OUT/'visual_inspection.json').read_text())
config = json.loads((OUT/'run_config.json').read_text())
assert validation['status'] == visual['status'] == 'passed'
quality = pd.read_csv(OUT/'common_RT_fit_quality.csv')
parameters = pd.read_csv(OUT/'parameter_summary.csv')
boundary = pd.read_csv(OUT/'delay_boundary_diagnostics.csv').set_index('Group')

parts = ['## Human RT-only TIED: 250 ms delay bound and 0.25–1.5 s trials', '',
    '**Question:** does allowing a shared delay up to 250 ms improve the reactive model’s '
    'RT-distribution fit, with observed choices still ignored?', '',
    '**Selection:** `Condition == "RT"` and `0.25 <= RT_s <= 1.5`. Pool raw trials across '
    'participants and sessions within each group. Retain correct and incorrect responses; '
    'observed response side does not enter the objective. SD and sequential CB are excluded.', '',
    '| Group | Participants | Previous 0.2–1.5 s fit | New 0.25–1.5 s fit | Additional removals |',
    '|---|---:|---:|---:|---:|',
    '| Control | 11 | 6,528 | 6,524 | 4 |',
    '| ASD | 8 | 4,703 | 4,703 | 0 |', '',
    'The user approved the 250 ms cap after reviewing the current parameter means and the '
    'selection issue. Four additional Control responses (Participant 36, source rows '
    '6418/6492/6987/6992; 226–243 ms) are excluded so that the full delay range is available. '
    'Relative to all 11,400 RT trials, 8 responses below 250 ms and 165 above 1.5 s are '
    'excluded. The fastest retained responses are 257.898 ms Control and 277.840 ms ASD. '
    'Recorded RTs are not shifted or imputed.', '',
    '**Two changes from the previous RT-only fit:** the delay prior rises from 0–200 to '
    '0–250 ms, and the retained lower RT increases from 200 to 250 ms. Comparisons below '
    'evaluate both saved parameter sets on the same new rows and with the same '
    '**0.25–1.5 s conditional normalization**.', '',
    '### Model and RT-only objective', '',
    'Keep exactly five free parameters shared across both ASLs and all ten signed differences '
    'within each group: `rate_lambda`, `T_0`, `theta_E`, `w`, `t_E_aff`. Use the unchanged '
    'vanilla/IPL TIED first-passage primitives. `B = Mean_dB` (≈10.892/15.089 visual dB), '
    '`D = Delta_dB` (right minus left); retain ASL 20/40 cd/m² as plot labels.', '',
    '```text\n'
    'gamma = theta_E * tanh(rate_lambda * D / 17.37)\n'
    'omega = 10**(rate_lambda * B / 20) * cosh(rate_lambda * D / 17.37) / (T_0 * theta_E**2)\n'
    'Z_E = (w - 0.5) * 2 * theta_E\n'
    'first-passage time = RT_s - t_E_aff\n'
    'p(RT | B,D,retained) = [rho_right(RT) + rho_left(RT)] / Z(B,D)\n'
    'Z(B,D) = F_right(1.5) + F_left(1.5) - F_right(0.25) - F_left(0.25)\n'
    '```', '',
    'This is a trial-level RT likelihood conditional on the stimulus and retained window. '
    'Summing both response densities ignores observed choice; the objective does not fit '
    'histogram heights or psychometric probabilities.', '',
    '| Parameter | Prior | Hard support |', '|---|---|---|',
    '| `rate_lambda` | LogNormal(log(0.2), 1.5) | >0; no finite upper bound |',
    '| `T_0` (s) | LogNormal(log(0.001), 2) | >0; no finite upper bound |',
    '| `theta_E` | LogNormal(log(30), 1.5) | >0; no finite upper bound |',
    '| `w` | Uniform(0.05, 0.95) | 0.05–0.95 |',
    '| `t_E_aff` (s) | Uniform(0, 0.250) | 0–250 ms |', '',
    'LogNormal arguments use natural logarithms. All priors except the delay cap are unchanged.', '',
    '### RTDs and CDFs', '',
    f'![250-ms versus 200-ms RT-only fits on common trials]({asset}human_RT_only_rtd_cdf_comparison_full_0p25_1p5s.png)', '',
    '*Control above ASD; pooled RTD left and CDF right. Blue/orange denote ASL 20/40 cd/m². '
    'Solid curves use the new 250-ms-cap fit’s arithmetic physical posterior means; dotted '
    'curves use the previous 200-ms-cap RT-only means, recomputed with the new retention '
    'window. Thin steps show data. Signed conditions are mixed with their actual retained '
    'trial frequencies. Histograms use 20 ms bins with a final 10 ms bin from 1.49 to 1.5 s. '
    'Choices are not fitted.*', '',
    f'[Full-size PNG]({asset}human_RT_only_rtd_cdf_comparison_full_0p25_1p5s.png) · '
    f'[PDF]({asset}human_RT_only_rtd_cdf_comparison_full_0p25_1p5s.pdf) · '
    f'[0.25–1 s zoom]({asset}human_RT_only_rtd_cdf_comparison_zoom_0p25_1s.png)', '',
    '| Group | Common RT metric | Previous 200 ms cap | New 250 ms cap | Change |',
    '|---|---|---:|---:|---:|']
for group in ['Control', 'ASD']:
    old = quality.loc[quality.Group.eq(group) & quality.fit.eq('previous_rt_only_200ms')].iloc[0]
    new = quality.loc[quality.Group.eq(group) & quality.fit.eq('rt_only')].iloc[0]
    for key, label in [('mean_parameter_RT_log_likelihood', 'RT log likelihood (higher better)'),
                       ('mean_condition_cdf_error', 'Mean signed-condition CDF gap'),
                       ('max_condition_cdf_error', 'Largest signed-condition CDF gap')]:
        parts.append(f'| {group} | {label} | {old[key]:.6g} | {new[key]:.6g} | {new[key]-old[key]:+.6g} |')
parts += ['', 'The mean CDF gap weights each of the 20 ASL × signed-difference cells equally. '
    'The likelihood weights individual trials equally; pooled curves use actual trial frequencies. '
    'All model curves and scores use arithmetic means of the five physical parameters, '
    'rather than averaging predictions over the posterior.', '',
    '**Visual assessment:** ' + visual['fit_assessment'], '',
    '### RTDs by absolute difference', '',
    f'![250-ms RT-only fit by absolute difference]({asset}human_RT_only_rtd_by_delta_full_0p25_1p5s.png)', '',
    '*Control above ASD; columns |ΔdB| = 0.1, 0.3, 0.65, 1.3, 2. Dashed steps are empirical '
    'RTDs; solid curves are the 250-ms-cap model at the physical posterior means on a 1 ms '
    'grid. Blue/orange identify ASL 20/40 cd/m². Pool ±ΔdB using actual retained counts. '
    'The zoom retains the full 0.25–1.5 s normalization. Matching axes include all peaks.*', '',
    f'[Full-size PNG]({asset}human_RT_only_rtd_by_delta_full_0p25_1p5s.png) · '
    f'[PDF]({asset}human_RT_only_rtd_by_delta_full_0p25_1p5s.pdf) · '
    f'[0.25–1 s zoom]({asset}human_RT_only_rtd_by_delta_zoom_0p25_1s.png)', '',
    '### Posterior parameters and delay boundary', '',
    '| Group | Parameter | Mean | SD | Central 95% interval |', '|---|---|---:|---:|---|']
for r in parameters.itertuples():
    factor = 1000 if r.parameter in ['T_0', 't_E_aff'] else 1
    label = r.parameter + (' (ms)' if factor == 1000 else '')
    parts.append(f'| {r.Group} | {label} | {r.mean*factor:.6g} | {r.sd*factor:.5g} | '
                 f'[{r.q025*factor:.5g}, {r.q975*factor:.5g}] |')
parts += ['', '| Group | Delay draws >249 ms | Delay draws >245 ms |', '|---|---:|---:|']
for group in ['Control', 'ASD']:
    r = boundary.loc[group]
    parts.append(f'| {group} | {r.fraction_within_1ms_of_upper:.2%} | {r.fraction_within_5ms_of_upper:.2%} |')
parts += ['', 'These are **variational approximations**, using 10,000 constrained draws per '
    'group. Uncertainty from pooled trials does not represent participant-to-participant '
    'variability. A delay posterior near 250 ms remains sensitive to the imposed cap.', '',
    '| Group | Parameter pair | Posterior correlation |', '|---|---|---:|']
for group in ['Control', 'ASD']:
    corr = pd.read_csv(OUT/group.lower()/'posterior_correlation.csv', index_col=0)
    pairs = sorted([(abs(corr.iloc[i,j]), PARAM_NAMES[i], PARAM_NAMES[j], corr.iloc[i,j])
                    for i in range(5) for j in range(i)], reverse=True)
    for _, a, b, r in pairs[:3]:
        parts.append(f'| {group} | {a}, {b} | {r:+.4f} |')
parts += ['', '### Optimization and confirmation', '',
    'One warm start per group transfers the saved 200-ms RT-only full-rank guide, mapping '
    'the delay’s latent center and local covariance to 0–250 ms while preserving its physical '
    'center. Reset Adam, RNG (seed 0), losses and patience. CPU float64; '
    'AutoMultivariateNormal; full batches; one ELBO particle; ClippedAdam rate 0.0002, clip 1. '
    'Minimum 50,000 and maximum 1,000,000 new updates.', '',
    'Every lower 1,000-step mean loss resets patience. Require at least 25 later windows '
    'and both terminal and selected-checkpoint local mean stability (<0.1 posterior SD over '
    'three checkpoints). Restore the selected checkpoint. These checks do not establish '
    'global optimality or unique parameter estimates.', '',
    '| Group | Selected best step | Final checked step | Later windows | Status |',
    '|---|---:|---:|---:|---|']
for group in ['Control', 'ASD']:
    r = json.loads((OUT/group.lower()/'convergence_summary.json').read_text())
    status = '**flagged / unconfirmed**' if r['flagged'] else 'Window and local-stability checks passed'
    parts.append(f"| {group} | {r['restored_best_step']:,} | {r['completed_steps']:,} | {r['confirmation_windows']} | {status} |")
for group in ['Control', 'ASD']:
    r = json.loads((OUT/group.lower()/'convergence_summary.json').read_text())
    if r['flagged']:
        history = pd.read_csv(OUT/group.lower()/'convergence.csv').set_index('step')
        parts += ['', f"**{group} remains provisional after {r['completed_steps']:,} updates.**"]
        for step, passed, label in [(r['restored_best_step'], r['restored_checkpoint_parameter_stable'], 'Selected checkpoint'),
                                    (r['completed_steps'], r['terminal_parameter_stable'], 'Terminal checkpoint')]:
            if not passed:
                movement = history.loc[step, [name+'_change_sd' for name in PARAM_NAMES]].astype(float)
                parameter = movement.idxmax().removesuffix('_change_sd')
                parts.append(f'{label} at step {step:,}: the largest three-checkpoint mean '
                    f'change was {movement.max():.4f} posterior SD in `{parameter}`, above '
                    'the 0.1 criterion. The number of post-minimum windows and this '
                    'parameter-stability test are separate requirements.')
parts += ['', f'[Control corner]({asset}control/parameter_corner.png) · '
    f'[ASD corner]({asset}asd/parameter_corner.png) · '
    f'[Control convergence]({asset}control/convergence.png) · '
    f'[ASD convergence]({asset}asd/convergence.png)', '',
    '### Sources and independent verification', '',
    'Fit root: `ASD_Human_Data/reactive_tied_rt_only_svi_0p25_1p5s_delay250ms/`. '
    'Previous RT-only fit: `ASD_Human_Data/reactive_tied_rt_only_svi_0p2_1p5s_delay200ms/`.', '',
    f'[Protocol]({asset}analysis_protocol.json) · [Configuration]({asset}run_config.json) · '
    f'[Selection counts]({asset}selection_counts.csv) · [Excluded rows]({asset}excluded_trials.csv) · '
    f'[Participant counts]({asset}participant_selection_counts.csv) · '
    f'[Condition counts]({asset}condition_selection_counts.csv) · '
    f'[Common-window metrics]({asset}common_RT_fit_quality.csv) · '
    f'[Posterior summaries]({asset}parameter_summary.csv) · '
    f'[Delay-boundary diagnostics]({asset}delay_boundary_diagnostics.csv) · '
    f'[Independent validation]({asset}independent_validation.json) · '
    f'[Full posterior/source/audit ZIP]({asset}human_RT_only_delay250ms_audits.zip)', '']
sources = ['human_reactive_tied_rt_only_utils.py', 'fit_human_reactive_tied_rt_only_svi.py',
           'plot_human_reactive_tied_rt_only_delay250_svi.py',
           'verify_human_reactive_tied_rt_only_delay250_svi.py',
           'publish_human_reactive_tied_rt_only_delay250_svi.py',
           'human_tied_independent_fpt.py', 'human_reactive_tied_svi_utils.py']
for name in sources:
    parts.append(f'- `ASD_Human_Data/{name}` ([source]({asset}source/{name})).')
parts += ['', f"Independent checks verified all {validation['selected_source_rows']:,} retained "
    f"and {validation['excluded_source_rows']} excluded source identities, the newly removed "
    'four Control rows, choice-label invariance, exactly five free parameters, the 250 ms '
    'support, posterior summaries/covariances, warm-start mapping, strict patience resets, '
    'finite gradients, K=40/80 convergence, independent SciPy PDF/CDF agreement, and unit '
    f"conditional mass. Reconstructed {validation['condition_CDF_checks']} condition CDF "
    f"comparisons, {validation['histogram_bins_verified']} histogram bins and "
    f"{validation['mixture_rows_verified']} mixture weights. Earlier fit/figure files were "
    'hash checked, prior documentation preserved, PNGs visually inspected, and PDF text '
    'bounds checked.', '']
section = '\n'.join(parts)
(OUT/'RT_only_delay250_result.md').write_text(section)
ASSETS.mkdir(parents=True, exist_ok=True)
for path in OUT.rglob('*'):
    if path.is_file() and path.suffix in ['.png', '.pdf', '.csv', '.json']:
        target = ASSETS/path.relative_to(OUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
(ASSETS/'source').mkdir(exist_ok=True)
for name in sources:
    shutil.copy2(ROOT/'ASD_Human_Data'/name, ASSETS/'source'/name)
kernel = ROOT/'fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py'
shutil.copy2(kernel, ASSETS/'source'/kernel.name)
archive = ASSETS/'human_RT_only_delay250ms_audits.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix not in ['.png', '.pdf']:
            z.write(path, 'fit/'+str(path.relative_to(OUT)))
    for name in sources:
        z.write(ROOT/'ASD_Human_Data'/name, 'source/'+name)
    z.write(kernel, 'source/'+kernel.name)
    for group in ['control', 'asd']:
        for name in ['guide_parameters.npz', 'posterior_samples.npz', 'data_manifest.json']:
            z.write(Path(config['source_fit'])/group/name, f'input_previous_rt_only_200ms/{group}/{name}')
text = PAGE.read_text()
heading = '# Results — 2026-10-20\n'
assert heading in text and parts[0] not in text
assert text == (OUT/'previous_result_page.md').read_text()
PAGE.write_text(text.replace(heading, heading+'\n'+section+'\n\n---\n', 1))
index = ROOT/'docs/results/index.md'
lines = index.read_text().splitlines()
link = '- [2026-10-20](2026-10-20.md)'
assert any(line.startswith(link) for line in lines)
index.write_text('\n'.join(line+'; human RT-only 250 ms delay sensitivity' if line.startswith(link)
                           else line for line in lines)+'\n')
(OUT/'publication.json').write_text(json.dumps(dict(page=str(PAGE.relative_to(ROOT)),
    published_at_utc=datetime.now(timezone.utc).isoformat(), archive_sha256=sha256(archive),
    source_sha256=sha256(__file__)), indent=2))
print('Published', PAGE)
