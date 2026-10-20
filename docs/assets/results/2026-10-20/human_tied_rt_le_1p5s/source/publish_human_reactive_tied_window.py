# %%
"""Publish the verified 1.5-second human TIED sensitivity refit beside earlier runs."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import zipfile
import numpy as np
import pandas as pd
from human_reactive_tied_window_utils import ROOT, PARAM_NAMES, sha256

# %% Distinct fit and asset roots preserve both previous analyses.
OUT = ROOT/'ASD_Human_Data/reactive_tied_rt_svi_rt_le_1p5s'
ASSETS = ROOT/'docs/assets/results/2026-10-20/human_tied_rt_le_1p5s'
PAGE = ROOT/'docs/results/2026-10-20.md'
validation = json.loads((OUT/'independent_validation.json').read_text())
visual = json.loads((OUT/'visual_inspection.json').read_text())
assert validation['status'] == visual['status'] == 'passed'
ASSETS.mkdir(parents=True, exist_ok=True)
counts = pd.read_csv(OUT/'selection_counts.csv').set_index('Group')
quality = pd.read_csv(OUT/'common_window_fit_quality.csv')
parameters = pd.read_csv(OUT/'parameter_summary.csv')
asset = '../assets/results/2026-10-20/human_tied_rt_le_1p5s/'
parts = ['## Human RT: refit after removing responses above 1.5 s', '',
    '**Question:** do the long responses explain the poor reactive-only TIED fits?', '',
    '**Outcome:** removing the long tail does **not resolve the RTD shape mismatch** in '
    'this five-parameter pooled fit. Both groups improve their common-window joint '
    'likelihood and largest signed-condition CDF gap, but their average condition-wise '
    'CDF gaps increase. The model RTDs remain much broader and less peaked than the data.', '',
    '**Selection:** `Condition == "RT"` and `RT_s <= 1.5`. This is a new trial filter, '
    'with likelihood and predictions conditional on **0–1.5 s**. Earlier figures used all '
    '11,400 recorded RT trials and 0–3 s normalization; their 1.5 s view was only a display zoom. '
    'Correct and incorrect choices remain included, with raw trials pooled across participants '
    'and sessions. SD and sequential CB are excluded.', '',
    '| Group | Participants | Original RT trials | Retained ≤1.5 s | Removed >1.5 s |',
    '|---|---:|---:|---:|---:|']
for group in ['Control', 'ASD']:
    c = counts.loc[group]
    parts.append(f'| {group} | {int(c.participants)} | {int(c.original_trials):,} | '
                 f'{int(c.selected_trials):,} | {int(c.removed_trials):,} |')
parts += ['', 'Only **165/11,400 trials (1.45%)** are removed. All 19 participants, both ASLs '
    'and all ten signed differences remain. Cell counts are now unequal, so model mixtures '
    'use actual retained trial frequencies. The fastest trials are retained: delay support '
    'remains `0 < t_E_aff < 0.111188799848944 s` for Control and '
    '`0 < t_E_aff < 0.181610000094563 s` for ASD.', '',
    '### Model and refitting', '',
    'The same five parameters (`rate_lambda`, `T_0`, `theta_E`, `w`, `t_E_aff`) are shared '
    'across both ASLs and all signed differences within each group. The IPL/vanilla equations '
    'use `B = Mean_dB`, `D = Delta_dB`, `Choice = +1/-1` for right/left, and first-passage '
    'time `RT_s - t_E_aff`. Divide each choice density by the **either-bound mass** in '
    '0–1.5 s. ASL 20/40 cd/m² are plot labels, with actual model levels ≈10.892/15.089 visual dB.', '',
    'Priors are unchanged: lognormal λ/T₀/θ with natural-log means '
    'log(.2)/log(.001)/log(30) and log SDs 1.5/2/1.5; uniform w in (.05,.95) '
    'and delay below the group minimum RT minus 1 μs. '
    'One warm start per group uses the previous confirmed **full-rank guide**, then resets '
    'Adam, RNG (seed 0), loss history and patience because the target likelihood changed. '
    'The actual initial guide covariance is the saved covariance. This is a new fit, '
    'rather than an optimizer-state continuation.', '',
    'CPU float64; NumPyro AutoMultivariateNormal; full batches, one ELBO particle; '
    'ClippedAdam learning rate .0002 and clip 1; K=40 checked against K=80. '
    'Check 1,000-step means, minimum 50,000 and cap 1,000,000 new updates. Every lower mean '
    'resets patience; require 25 later windows and terminal/restored mean stability '
    '(<0.1 posterior SD over three checkpoints), then restore the best checkpoint. '
    'This is observed-window confirmation, not proof of a global optimum.', '',
    '| Group | Selected best step | Final checked step | Later windows | Status |',
    '|---|---:|---:|---:|---|']
for group in ['Control', 'ASD']:
    report = json.loads((OUT/group.lower()/'convergence_summary.json').read_text())
    status = 'confirmed; means stable' if not report['flagged'] else '**flagged / incomplete confirmation**'
    parts.append(f"| {group} | {report['restored_best_step']:,} | {report['completed_steps']:,} | "
                 f"{report['confirmation_windows']} | {status} |")
parts += ['', '### Previous versus refitted model on the same selected trials', '',
    f'![Human TIED refit after the 1.5 s cutoff]({asset}human_tied_rt_cutoff_comparison.png)', '',
    '*Control above ASD; psychometric left and pooled RTD right. Solid curves: new fit. '
    'Dotted curves: previous confirmed full-RT fit, recomputed and reconditioned on 0–1.5 s. '
    'Blue/orange identify ASL 20/40 cd/m²; points/steps show the same retained data. '
    'Both predictions use arithmetic means of physical posterior parameters and actual '
    'retained condition frequencies.*', '',
    f'[Full-size PNG]({asset}human_tied_rt_cutoff_comparison.png) · '
    f'[PDF]({asset}human_tied_rt_cutoff_comparison.pdf)', '',
    '**Common-window comparison:** both models are evaluated on exactly the same selected '
    'rows with 0–1.5 s normalization. The original training losses on all trials cannot be '
    'compared directly with the new training losses.', '',
    '| Group | Metric | Previous parameters, reconditioned | Refit | Change |',
    '|---|---|---:|---:|---:|']
metrics = {'mean_parameter_joint_log_likelihood': 'Joint log likelihood at means (higher better)',
           'psychometric_rmse': 'Psychometric RMSE', 'mean_condition_cdf_error': 'Mean signed-condition RT CDF gap',
           'max_condition_cdf_error': 'Largest signed-condition RT CDF gap'}
for group in ['Control', 'ASD']:
    old = quality.loc[quality.Group.eq(group)&quality.fit.eq('previous_reconditioned')].iloc[0]
    new = quality.loc[quality.Group.eq(group)&quality.fit.eq('refit')].iloc[0]
    for key, label in metrics.items():
        parts.append(f'| {group} | {label} | {old[key]:.6g} | {new[key]:.6g} | {new[key]-old[key]:+.5g} |')
parts += ['', '### RTDs for each absolute difference', '',
    f'![Refitted RTDs by absolute difference]({asset}human_reactive_tied_rtd_by_delta_full_0_1p5s.png)', '',
    '*Control above ASD; columns |ΔdB| = 0.1, 0.3, 0.65, 1.3, 2. Data: 20 ms histogram '
    'steps; model: 1 ms curves at arithmetic posterior means. Pool ±ΔdB with observed retained '
    'counts and sum both choices. Matching axes include all histogram and model peaks; '
    'histograms and model densities have unit area over the fitted 0–1.5 s window.*', '',
    f'[Full-size PNG]({asset}human_reactive_tied_rtd_by_delta_full_0_1p5s.png) · '
    f'[PDF]({asset}human_reactive_tied_rtd_by_delta_full_0_1p5s.pdf) · '
    f'[Refit psychometric/RTD PNG]({asset}human_reactive_tied_psychometric_rtd_full_0_1p5s.png) · '
    f'[0–1 s display zoom]({asset}human_reactive_tied_psychometric_rtd_zoom_0_1s.png)', '',
    '### Posterior parameters', '',
    '| Group | Parameter | Mean | SD | Central 95% interval |',
    '|---|---|---:|---:|---|']
for row in parameters.itertuples():
    factor = 1000 if row.parameter in ['T_0', 't_E_aff'] else 1
    label = row.parameter+' (ms)' if factor == 1000 else row.parameter
    parts.append(f'| {row.Group} | {label} | {row.mean*factor:.6g} | {row.sd*factor:.5g} | '
                 f'[{row.q025*factor:.5g}, {row.q975*factor:.5g}] |')
parts += ['', 'These are **variational approximations**, from 10,000 constrained draws per group. '
    'Pooled-trial posterior uncertainty does not measure participant-to-participant variability.', '',
    '| Group | Parameter pair | Posterior correlation |', '|---|---|---:|']
for group in ['Control', 'ASD']:
    corr = pd.read_csv(OUT/group.lower()/'posterior_correlation.csv', index_col=0)
    pairs = sorted([(abs(corr.iloc[i,j]), PARAM_NAMES[i], PARAM_NAMES[j], corr.iloc[i,j])
                    for i in range(5) for j in range(i)], reverse=True)
    for _, a, b, r in pairs[:3]: parts.append(f'| {group} | {a}, {b} | {r:+.4f} |')
parts += ['', f'[Control corner]({asset}control/parameter_corner.png) · '
    f'[ASD corner]({asset}asd/parameter_corner.png) · '
    f'[Control loss/parameter traces]({asset}control/convergence.png) · '
    f'[ASD loss/parameter traces]({asset}asd/convergence.png)', '',
    'λ, T₀ and θ remain strongly correlated; the traces show slow movement along this '
    'parameter tradeoff. The stopping checks assess recent windows and do not establish '
    'a unique/global minimum. The fastest retained responses still constrain the shared delay.', '',
    '### Sources, selected rows and verification', '',
    'Previous fit root: `ASD_Human_Data/reactive_tied_rt_svi_confirm25/`. '
    'New fit root: `ASD_Human_Data/reactive_tied_rt_svi_rt_le_1p5s/`.', '',
    f'[Selection counts]({asset}selection_counts.csv) · '
    f'[Condition counts]({asset}condition_selection_counts.csv) · '
    f'[Participant counts]({asset}participant_selection_counts.csv) · '
    f'[Excluded trial identities]({asset}excluded_long_trials.csv) · '
    f'[Common-window fit quality]({asset}common_window_fit_quality.csv) · '
    f'[Parameter summaries]({asset}parameter_summary.csv) · '
    f'[Independent validation]({asset}independent_validation.json) · '
    f'[Full posterior/source/audit ZIP]({asset}human_tied_rt_le_1p5s_audits.zip)', '']
sources = ['human_reactive_tied_window_utils.py', 'fit_human_reactive_tied_window_svi.py',
           'plot_human_reactive_tied_window_svi.py', 'compare_human_reactive_tied_window.py',
           'verify_human_reactive_tied_window_svi.py', 'publish_human_reactive_tied_window.py',
           'human_reactive_tied_svi_utils.py']
for name in sources:
    parts.append(f'- `ASD_Human_Data/{name}` ([source]({asset}source/{name})).')
parts += ['', f"Independent verification matched all 11,235 retained and 165 excluded source identities, "
    f"verified both groups' posterior summaries/covariances and strict patience resets, checked finite "
    f"gradients, K=40/80 series, SciPy density/CDF agreement and unit conditional mass. It reconstructed "
    f"{validation['probability_cells_verified']} probabilities, {validation['histogram_bins_verified']} histogram bins "
    f"and both parameter sets' common-window comparisons. All {validation['prior_artifacts_hash_verified']} "
    f"previous fit/asset files are byte-identical. Figures were visually inspected and PDFs passed text-bound checks.", '']
section = '\n'.join(parts)
(OUT/'cutoff_result.md').write_text(section)
for path in OUT.rglob('*'):
    if path.is_file() and path.suffix in ['.png', '.pdf', '.csv', '.json']:
        destination = ASSETS/path.relative_to(OUT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
(ASSETS/'source').mkdir(exist_ok=True)
for name in sources: shutil.copy2(ROOT/'ASD_Human_Data'/name, ASSETS/'source'/name)
kernel = ROOT/'fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py'
shutil.copy2(kernel, ASSETS/'source'/kernel.name)
archive = ASSETS/'human_tied_rt_le_1p5s_audits.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix not in ['.png', '.pdf']:
            z.write(path, 'fit/'+str(path.relative_to(OUT)))
    for name in sources: z.write(ROOT/'ASD_Human_Data'/name, 'source/'+name)
    z.write(kernel, 'source/'+kernel.name)
page = PAGE.read_text()
heading = '# Results — 2026-10-20\n'
assert heading in page and parts[0] not in page
(OUT/'result_book_previous_page.md').write_text(page)
PAGE.write_text(page.replace(heading, heading+'\n'+section+'\n\n---\n', 1))
index = ROOT/'docs/results/index.md'
old_link = '- [2026-10-20](2026-10-20.md)'
text = index.read_text()
assert old_link in text
index.write_text(text.replace(old_link, old_link+' — human TIED RT ≤1.5 s sensitivity refits and full-RT comparisons', 1))
(OUT/'publication.json').write_text(json.dumps(dict(page=str(PAGE.relative_to(ROOT)),
    published_at_utc=datetime.now(timezone.utc).isoformat(), archive_sha256=sha256(archive),
    source_sha256=sha256(__file__)), indent=2))
print('Published', PAGE)
