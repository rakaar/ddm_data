# %% Publish the two model comparisons beside the new Omega-upper-25 fits
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-11-08'
OUT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl_omega25'
PAGE = ROOT/'docs/results'/f'{DATE}.md'
ASSETS = ROOT/'docs/assets/results'/DATE/'shank3_gamma_omega_gaussian_abl_delay_omega25'
COMPARISONS = [
    ('Gaussian delay comparison: condition-specific versus NPL+alpha',
     'shank3_cohort1_wt_hom_gaussian_delay_model_comparison_omega25',
     'SHANK3_cohort1_WT_HOM_gaussian_ABL_delay_model_comparison_omega25_2x2',
     'shank3_gaussian_abl_delay_model_comparison_omega25',
     'WT top, HOM bottom; underlying Gaussian mean μ versus ABL left, SD σ right. '
     'Black circles: new condition-specific Gamma/Omega fits with Omega upper bound 25 s⁻¹; '
     'blue squares: the unchanged 13-parameter NPL+alpha fits. Both share one Gaussian delay '
     'distribution across all signed ILDs within ABL. Dots and capped bars show posterior means '
     'and central 95% variational credible intervals from 10,000 draws; times are milliseconds. '
     'These are underlying Gaussian parameters before zero truncation, not animal SEMs.'),
    ('Gamma/Omega comparison: condition-specific versus continuous NPL+alpha',
     'shank3_cohort1_wt_hom_gamma_omega_model_comparison_omega25',
     'SHANK3_cohort1_WT_HOM_Gamma_Omega_direct_vs_NPL_alpha_omega25_2x2',
     'shank3_gamma_omega_model_comparison_omega25',
     'WT top, HOM bottom; Gamma left, Omega right. ABL 20 blue, 40 orange, 60 green. '
     'Dots and capped central 95% credible intervals are the new condition-specific posterior '
     'estimates with Omega upper bound 25 s⁻¹. Lines use the unchanged NPL+alpha mapping '
     'at physical posterior-mean parameters, on signed ILD −16 to +16 in 0.1 dB steps. '
     'NPL+alpha Gamma is independent of ABL, so its common curve is black; Omega has three '
     'ABL-colored curves. Dot offsets ±0.18 dB are display-only. No curve uncertainty bands.')]

assert json.loads((OUT/'publication_validation.json').read_text())['status'] == 'passed'
assert json.loads((OUT/'sensitivity_validation.json').read_text())['status'] == 'passed'
lines = ['<!-- shank3-omega25-comparisons:start -->',
    '### Comparisons with the unchanged NPL+alpha fits', '',
    'Only the direct-fit Omega hard upper bound changed, from **15 to 25 s⁻¹**. '
    'Its flat prior interval remains **0.1–12 s⁻¹**. Trial identities, physical seed-0 '
    'initialization, all other priors, frozen proactive/lapse values, likelihood conventions '
    'and loss-confirmation stopping settings match the original direct fits. '
    'The NPL+alpha fits are reused without refitting. The 25-bound direct fits are the current '
    'result-book view; superseded 15-bound figures are not displayed.', '']
published = []
for title, directory, stem, source, caption in COMPARISONS:
    folder = ROOT/'asd_analysis'/directory
    manifest = json.loads((folder/'manifest.json').read_text())
    assert json.loads((folder/'validation.json').read_text())['status'] == 'passed'
    visual = json.loads((folder/'visual_inspection.json').read_text())
    assert visual['status'] == 'passed' and visual['artifacts'] == manifest['artifacts']
    destination = ASSETS/directory
    destination.mkdir(exist_ok=True)
    for suffix in ['png', 'pdf']:
        shutil.copy2(folder/f'{stem}.{suffix}', destination/f'{stem}.{suffix}')
    scripts = [ROOT/'asd_analysis'/f'{prefix}_{source}.py' for prefix in ['plot', 'verify']]
    for path in scripts:
        shutil.copy2(path, destination/path.name)
    archive = destination/(directory+'_audits.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in folder.iterdir():
            if path.is_file() and path.suffix in ['.csv', '.json']:
                zipped.write(path, path.name)
        for path in scripts:
            zipped.write(path, 'source/'+path.name)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
    base = f'../assets/results/{DATE}/shank3_gamma_omega_gaussian_abl_delay_omega25/{directory}'
    lines += ['#### '+title, '', f'![{title}]({base}/{stem}.png)', '', '*'+caption+'*', '',
        f'[PDF]({base}/{stem}.pdf) · [Plotting source]({base}/{scripts[0].name}) · '
        f'[Independent verifier]({base}/{scripts[1].name}) · [Numerical/provenance audits]({base}/{archive.name})', '']
    published.append(str((destination/f'{stem}.png').relative_to(ROOT)))

metrics = pd.read_csv(OUT/'omega_bound_in_sample_comparison.csv')
metrics = metrics.loc[metrics.model.ne('direct_omega15')].copy()
metrics.to_csv(OUT/'current_in_sample_comparison.csv', index=False, float_format='%.17g')
lines += ['### Current-fit checks and interpretation', '',
    '| Group | Model | Joint log likelihood | Psychometric RMSE | Mean max RT-CDF gap |',
    '|---|---|---:|---:|---:|']
labels = {'direct_omega25': 'Direct Γ/Ω; Ω ≤ 25', 'npl_alpha': 'NPL+alpha; unchanged'}
for row in metrics.itertuples():
    lines.append(f'| {row.group.upper()} | {labels[row.model]} | {row.joint_log_likelihood:,.2f} | '
                 f'{row.psychometric_rmse:.5f} | {row.mean_max_rt_cdf_gap:.5f} |')
pressure = pd.read_csv(OUT/'omega_bound_pressure_by_ABL.csv')
pressure = pressure.loc[pressure.omega_hard_upper.eq(25)].copy()
pressure.to_csv(OUT/'current_omega_boundary_by_ABL.csv', index=False, float_format='%.17g')
lines += ['', '**ABL-60 Omega boundary pressure:**', '',
    '| Group | Omega ceiling | Conditions with mean within 1 s⁻¹ of ceiling | Largest condition mean | Highest upper 95% limit |',
    '|---|---:|---:|---:|---:|']
for row in pressure.loc[pressure.ABL.eq(60)].itertuples():
    lines.append(f'| {row.group.upper()} | {row.omega_hard_upper:g} | '
                 f'{row.means_within_1_of_upper}/10 | {row.maximum_condition_mean:.3f} | {row.maximum_upper95:.3f} |')
remaining = pressure.loc[pressure.omega_hard_upper.eq(25) & pressure.ABL.eq(60)]
if remaining.means_within_1_of_upper.gt(0).any() or remaining.maximum_upper95.gt(24.75).any():
    lines += ['', 'Some ABL-60 condition posteriors still approach the new ceiling. Raising the '
        'bound therefore reduces the old restriction but does not establish that every estimate '
        'is free of boundary influence. See the draw-fraction audit for the degree of pressure.']
sensitivity = pd.read_csv(OUT/'omega_bound_parameter_sensitivity.csv')
lines += ['', '**Selected posterior means, current 25-bound fits:**', '',
    '| Group | ABL-60 Gaussian μ (ms) | ABL-60 Gaussian σ (ms) | Shared del_go (ms) |',
    '|---|---:|---:|---:|']
for group in ['wt', 'hom']:
    values = []
    for block in ['delay_mu', 'delay_sigma', 'del_go']:
        z = sensitivity.loc[sensitivity.group.eq(group) & sensitivity.block.eq(block)]
        if block != 'del_go':
            z = z.loc[z.ABL.eq(60)]
        row = z.iloc[0]
        values.append(f'{1000*row.new_mean:.2f}')
    lines.append('| '+group.upper()+' | '+' | '.join(values)+' |')
lines += ['', 'The readout interval del_go is separate from the '
    'Gaussian sensory delay. Both fits use one initialization, so parameter tradeoffs and '
    'alternative solutions remain possible. At ABL 60 and +16 dB, Gamma remains high and '
    'broadly uncertain in both groups; increasing Omega’s ceiling does not eliminate every '
    'condition-specific mismatch. The parameter-estimate download retains all interval endpoints.']
base = f'../assets/results/{DATE}/shank3_gamma_omega_gaussian_abl_delay_omega25'
names = ['current_omega_boundary_by_ABL.csv', 'current_in_sample_comparison.csv',
         'bound_change_validation.json', 'sensitivity_validation.json',
         'verify_shank3_gamma_omega_omega25_sensitivity.py', 'verify_shank3_gamma_omega_omega25_resume.py',
         Path(__file__).name]
for name in names:
    path = ROOT/'asd_analysis'/name if name.endswith('.py') else OUT/name
    shutil.copy2(path, ASSETS/name)
lines += ['', 'These are in-sample posterior-mean comparisons on identical trials; they do not establish '
    'held-out performance or initialization robustness. The changed hard bound also changes the '
    'trapezoidal prior taper above 12 s⁻¹. Posterior intervals remain conditional on frozen '
    'proactive/exponential-lapse parameters.', '',
    f'[Current Omega boundary pressure by ABL]({base}/current_omega_boundary_by_ABL.csv) · '
    f'[Current in-sample metrics]({base}/current_in_sample_comparison.csv) · '
    f'[Preservation/settings verification]({base}/sensitivity_validation.json)',
    '<!-- shank3-omega25-comparisons:end -->', '']
start, end = '<!-- shank3-omega25-comparisons:start -->', '<!-- shank3-omega25-comparisons:end -->'
text = PAGE.read_text()
block = '\n'.join(lines)
if start in text:
    first, rest = text.split(start, 1)
    _, last = rest.split(end, 1)
    PAGE.write_text(first+block+last)
else:
    PAGE.write_text(text.rstrip()+'\n\n'+block)

readme = ROOT/'asd_analysis/README.md'
title = '## Omega-upper-25 sensitivity refits (2026-11-08)'
if title not in readme.read_text():
    readme.write_text(readme.read_text().rstrip()+'\n\n'+title+'\n\n'
        'Separate 68-parameter WT/HOM fits in `shank3_cohort1_wt_hom_gamma_omega_gaussian_delay_by_abl_omega25/` '
        'raise only Omega hard upper 15→25 s⁻¹; flat 0.1–12 and all other settings, exact trials, '
        'frozen values and physical seed-0 starts match the prior fits. Sources carry `_omega25`. '
        'Seven PNGs are directly embedded in the November 8 result book: two RTD grids, two '
        'psychometrics, black/red WT/HOM parameter intervals, and the two 2×2 comparisons against '
        'unchanged NPL+alpha delays/evidence curves. The comparison folders end in `_omega25/`. '
        'Posterior draws, resumable checkpoints, independent numerical/resume/figure checks and '
        'before/after boundary-pressure and common-trial metric tables are saved. Existing fits, '
        'figures, sources and CSVs remain byte-identical. Parameter stability is diagnostic only.\n')
index = ROOT/'docs/results/index.md'
text = index.read_text()
entry = f'- [{DATE}]({DATE}.md) — SHANK3 WT/HOM current Gamma/Omega fits (Omega upper 25 s⁻¹): RTDs, psychometrics, posterior parameters and both NPL+alpha comparisons. Superseded 15-bound figures are retired.'
index.write_text(re.sub(r'(?m)^- \['+re.escape(DATE)+r'\]\([^\n]+$', entry, text))
for link in re.findall(r'\]\(([^)]+)\)', PAGE.read_text()):
    if '://' not in link:
        assert (PAGE.parent/link.split('#')[0]).exists(), link
subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'mkdocs', 'build', '--strict'], cwd=ROOT, check=True)
record = dict(status='published', png_embeds=7, page=str(PAGE.relative_to(ROOT)),
    title='SHANK3 cohort 1 WT/HOM: Gamma/Omega refits with Omega upper bound 25',
    comparisons=published, strict_mkdocs_build='passed', previous_fits_preserved=True)
(OUT/'completion.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2), flush=True)
