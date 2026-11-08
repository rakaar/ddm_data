# %%
"""Publish the human RT-only and joint-with-lapse comparisons on 21 October 2026."""
from pathlib import Path
import argparse
import json
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
from human_gamma_omega_variants_svi_utils import ROOT, BASELINE_ROOT, OUTPUTS, MODES, get_priors, sha256

# %% Publication controls. Scientific verification and visual inspection precede copying.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=[*MODES, 'all'], default='all')
parser.add_argument('--output-dir', type=Path, help='Override one variant root; requires a single --mode')
parser.add_argument('--rt-only-dir', type=Path, default=OUTPUTS['rt_only'])
parser.add_argument('--lapse-dir', type=Path, default=OUTPUTS['rt_choice_lapse'])
parser.add_argument('--update-existing', action='store_true', help='Replace only these result sections and assets')
args = parser.parse_args()
assert not (args.output_dir is not None and args.mode == 'all')
date = '2026-10-21'
page = ROOT / f'docs/results/{date}.md'
paths = {'rt_only': args.rt_only_dir, 'rt_choice_lapse': args.lapse_dir}
if args.output_dir is not None:
    paths[args.mode] = args.output_dir
modes = list(MODES) if args.mode == 'all' else [args.mode]
labels = {'rt_only': 'RT-only', 'rt_choice_lapse': 'RT + choice + lapses'}
asset_names = {'rt_only': 'human_gamma_omega_rt_only_svi',
               'rt_choice_lapse': 'human_gamma_omega_rt_choice_lapse_svi'}
headings = {'rt_only': '## Human RT: Gamma–Omega SVI fitting RT shape without choice',
            'rt_choice_lapse': '## Human RT: Gamma–Omega joint RT and choice SVI with lapses'}
comparison_heading = '## Human RT: baseline, RT-only and lapse-model comparison'
source_files = [
    'ASD_Human_Data/fit_human_gamma_omega_variants_svi.py',
    'ASD_Human_Data/human_gamma_omega_variants_svi_utils.py',
    'ASD_Human_Data/plot_human_gamma_omega_variants_svi.py',
    'ASD_Human_Data/verify_human_gamma_omega_variants_svi.py',
    'ASD_Human_Data/publish_human_gamma_omega_variants_svi.py',
    'ASD_Human_Data/human_gamma_omega_delay500_svi_utils.py',
    'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py',
    'ASD_Human_Data/human_reactive_tied_svi_utils.py',
    'ASD_Human_Data/prepare_simultaneous_rt_trials.py',
    'fit_each_condn/svi_gamma_omega_likelihood_utils.py',
    'fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py',
]
current_page = page.read_text()
original_page = current_page
records = {}
for mode in modes:
    out = paths[mode]
    config = json.loads((out / 'run_config.json').read_text())
    assert config['mode'] == mode
    for name, digest in config['source_hashes'].items():
        assert sha256(ROOT / name) == digest, name
    assert json.loads((out / 'independent_validation.json').read_text())['status'] == 'passed'
    visual = json.loads((out / 'visual_inspection.json').read_text())
    assert visual['status'] == 'passed'
    plot = json.loads((out / 'plot_manifest.json').read_text())
    assert plot['source_sha256'] == sha256(ROOT / source_files[2])
    inspected = {row['file']: row['sha256'] for row in visual['figures']}
    required_figures = list(plot['figures'])
    comparison = None
    if mode == 'rt_choice_lapse':
        comparison = json.loads((out / 'variant_comparison_manifest.json').read_text())
        assert comparison['source_sha256'] == sha256(ROOT / source_files[2])
        required_figures += comparison['figures']
    for name in required_figures:
        if name.endswith('.png'):
            assert inspected[name] == sha256(out / name), f'Visual inspection does not match {name}'
    if headings[mode] in current_page:
        assert args.update_existing, 'Already published; use --update-existing for a revision'
    records[mode] = dict(config=config, plot=plot, comparison=comparison, visual=visual)
if 'rt_choice_lapse' in modes and comparison_heading in current_page:
    assert args.update_existing, 'Comparison already published; use --update-existing for a revision'


# %% Reusable section replacement preserves preceding and following analyses.
def replace_section(text, heading, lines):
    block = '\n'.join(['', heading, '', *lines, ''])
    if heading not in text:
        return text + '\n' + block
    start = text.index(heading)
    following = re.search(r'\n## ', text[start + len(heading):])
    end = start + len(heading) + following.start() if following else len(text)
    return text[:start].rstrip() + '\n' + block + text[end:]


# %% Export sources, audits and model-specific entries without replacing old figures.
for mode in modes:
    out = paths[mode]
    info = records[mode]
    config, plot, comparison = info['config'], info['plot'], info['comparison']
    assets = ROOT / f'docs/assets/results/{date}/{asset_names[mode]}'
    assets.mkdir(parents=True, exist_ok=True)
    before = out / 'result_book_page_before.md'
    if not before.exists():
        before.write_text(original_page)
    copy_names = plot['figures'] + plot['tables'] + [
        'run_config.json', 'environment.json', 'independent_validation.json', 'independent_preflight_validation.json',
        'visual_inspection.json', 'selection_counts.csv', 'plot_manifest.json', 'execution_schedule.json',
    ]
    if comparison is not None:
        copy_names += comparison['figures'] + comparison['tables'] + ['variant_comparison_manifest.json']
    for name in dict.fromkeys(copy_names):
        shutil.copy2(out / name, assets / name)
    for group in ['control', 'asd']:
        dest = assets / group
        dest.mkdir(exist_ok=True)
        for name in ['posterior_summary.csv', 'convergence_summary.json', 'convergence.csv',
                     'conditions.csv', 'data_manifest.json', 'initialization.json', 'preflight.json']:
            shutil.copy2(out / group / name, dest / name)
    (assets / 'source').mkdir(exist_ok=True)
    for name in source_files:
        shutil.copy2(ROOT / name, assets / 'source' / Path(name).name)
    archive = assets / 'human_gamma_omega_variants_audits.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.name != 'result_book_page_before.md':
                if path.parent == out and path.name in ['publication.json', 'completion_audit.json']:
                    continue
                z.write(path, 'analysis/' + str(path.relative_to(out)))
        for name in source_files:
            z.write(ROOT / name, 'source/' + name)
        for name in ['run_config.json', 'selected_trials.csv', 'excluded_trials.csv',
                     'parameter_summary.csv', 'fit_quality.csv', 'condition_model_curves.csv']:
            z.write(BASELINE_ROOT / name, 'baseline_joint_500ms/' + name)
        for group in ['control', 'asd']:
            for name in ['initialization.json', 'posterior_samples.npz', 'conditions.csv', 'convergence_summary.json']:
                z.write(BASELINE_ROOT / group / name, f'baseline_joint_500ms/{group}/{name}')
        if comparison is not None:
            other = paths['rt_only']
            for name in ['run_config.json', 'fit_quality.csv']:
                z.write(other / name, 'rt_only_comparison/' + name)
            for group in ['control', 'asd']:
                for name in ['posterior_samples.npz', 'conditions.csv', 'initialization.json', 'convergence_summary.json']:
                    z.write(other / group / name, f'rt_only_comparison/{group}/{name}')
    base = f'../assets/results/{date}/{asset_names[mode]}'
    n_params = 61 if mode == 'rt_only' else 63
    lines = [
        '**Selection:** simultaneous `Condition == "RT"`, `Valid == 1`, and **0 ≤ RT_s ≤ 1.8 s**. '
        '**Control: 6,574 trials from 11 participants; ASD: 4,770 trials from 8 participants.** '
        'Correct and incorrect choices are retained, including all four RTs below 200 ms. The 56 RTs above 1.8 s '
        'are excluded. SD and sequential CB are excluded. Statistics and likelihoods pool raw trials across '
        'participants and sessions; they are not equal-participant averages.', '',
        '**Model:** reactive-only descriptive two-bound DDM, with 20 independent Gamma, 20 Omega and '
        f'20 delays, plus one shared starting bias `w` ({n_params} parameters per group'
        + (', including two shared lapse probabilities).' if mode == 'rt_choice_lapse' else ').')
        + ' Each signed ΔdB × ASL condition remains separate during fitting, without sign symmetry. '
        '`Delta_dB` is right minus left visual dB, `ASL_cd` labels 20/40 cd/m², and `Choice=1/-1` means right/left. '
        'There is no proactive process or `del_go`.', '',
    ]
    if mode == 'rt_only':
        lines += [
            'The fitted likelihood is **`(f_right(t) + f_left(t)) / Z`**, where `Z` is the condition’s either-bound '
            'mass within 0–1.8 s. Choice is marginalized at every observed RT; choice labels do not enter the objective. '
            'This fits trial-level RT densities rather than histogram errors. Each guide delay remains below its '
            'condition’s fastest retained RT; the target trapezoidal delay prior remains 0–500 ms.', '',
            '**Psychometric curves are diagnostic predictions:** empirical choice dots are shown for comparison, '
            'but these choices were not used in the RT-only objective.', '',
        ]
    else:
        lines += [
            'The fitted joint density is **`(1-p) f_y(t)/Z + p q_y/1.8`**, with '
            '`q_right = lapse_prob_right` and `q_left = 1-lapse_prob_right`. '
            '**`p` is the lapse fraction among retained trials.** The DDM component is normalized within '
            '0–1.8 s before mixing; lapse RTs are uniform on the original stored RT coordinate, without a delay shift. '
            'Both lapse probabilities are shared across all 20 stimuli within a group.', '',
            '**Lapse-model delays have the full 0–500 ms support.** Early observations may arise from the lapse '
            'component, so the guide delay transform is not restricted by the fastest observed RT. The RT-only and '
            'earlier no-lapse joint fits retain that causal restriction.', '',
            '| Group | Lapse fraction p: mean [95% interval] | P(right given lapse): mean [95% interval] |',
            '|---|---:|---:|',
        ]
        summary = pd.read_csv(out / 'parameter_summary.csv')
        for group in ['Control', 'ASD']:
            p = summary.loc[summary.Group.eq(group) & summary.parameter.eq('lapse_prob')].iloc[0]
            q = summary.loc[summary.Group.eq(group) & summary.parameter.eq('lapse_prob_right')].iloc[0]
            lines.append(f'| {group} | {p["mean"]:.4f} [{p.q025:.4f}, {p.q975:.4f}] | '
                         f'{q["mean"]:.4f} [{q.q025:.4f}, {q.q975:.4f}] |')
        lines += ['']
    lines += ['### Data and mean-parameter predictions', '']
    for group in ['Control', 'ASD']:
        stem = f'human_{group.lower()}_gamma_omega_rtd_psychometric'
        lines += [
            f'![{group} {labels[mode]} RTDs and psychometric]({base}/{stem}.png)', '',
            f'*{group}: ASL 20/40 occupy the two RTD rows; columns are |ΔdB| = 0.1, 0.3, 0.65, 1.3 and 2. '
            'Data use 20 ms histogram bins; analytic curves are evaluated at arithmetic means of physical posterior '
            'parameters. Each signed density has unit mass on 0–1.8 s before ±Δ mixing with actual retained counts. '
            'Psychometric data are dots without error bars; model probabilities are evaluated only at the ten '
            'measured differences and joined as visual guides. No additional sigmoid or stimulus interpolation is fitted.'
            + (' Choices are not part of this variant’s fitted objective.' if mode == 'rt_only' else ' Model curves include the lapse mixture.')
            + '*', '',
            f'[Full-size PNG]({base}/{stem}.png) · [PDF]({base}/{stem}.pdf)', '',
        ]
    lines += [
        '### Parameter comparison', '',
        f'![{labels[mode]} Gamma, Omega, delay and shared scalar posteriors]({base}/human_gamma_omega_parameter_comparison.png)', '',
        '*Gamma, Omega and delay form three rows, with ASL 20/40 in two columns. Control uses blue circles and ASD '
        'red stars. Points are posterior means and bars are equal-tailed **95% variational credible intervals**. '
        'The right column shows the shared bias'
        + (' and the two shared lapse probabilities.' if mode == 'rt_choice_lapse' else '.')
        + ' These are pooled-trial variational approximations; their uncertainty does not measure '
        'participant-to-participant variability.*', '',
        f'[Full-size PNG]({base}/human_gamma_omega_parameter_comparison.png) · '
        f'[PDF]({base}/human_gamma_omega_parameter_comparison.pdf) · '
        f'[Parameter means, SDs and intervals]({base}/parameter_summary.csv)', '',
        '### Fit quality on the retained trials', '',
        '| Group | Common marginal RT log likelihood | Joint log likelihood | Psychometric RMSE | Mean CDF gap | Maximum CDF gap |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in pd.read_csv(out / 'fit_quality.csv').itertuples():
        joint = '— (not fitted)' if np.isnan(row.joint_log_likelihood) else f'{row.joint_log_likelihood:.2f}'
        lines.append(f'| {row.Group} | {row.marginal_rt_log_likelihood:.2f} | {joint} | '
                     f'{row.psychometric_rmse:.4f} | {row.mean_condition_cdf_gap:.4f} | {row.max_condition_cdf_gap:.4f} |')
    lines += [
        '', 'Scores use arithmetic posterior means and identical trial identities/window normalization. '
        'The CDF discrepancy is the largest empirical-versus-model CDF gap within each signed condition. '
        'More flexible in-sample scores do not establish predictive superiority.', '',
        f'[Fit-quality table]({base}/fit_quality.csv) · [Condition-specific scores]({base}/condition_fit_quality.csv) · '
        f'[Psychometric calculations]({base}/psychometric_comparison.csv) · '
        f'[RTD histogram calculations]({base}/rtd_histograms.csv) · [Analytic RTD mixtures]({base}/rtd_model_curves.csv)', '',
        '### Priors, initialization and convergence', '',
        '| Parameter | Hard bounds | Flat region of trapezoidal prior |', '|---|---:|---:|',
    ]
    for name, prior in get_priors(mode).items():
        lines.append(f'| `{name}` | {prior["hard"]} | {prior["plausible"]} |')
    lines += [
        '', 'Prior tails decrease linearly from the flat region to zero at the hard bounds. Gamma favors the '
        'stimulus-consistent sign while permitting small reversals. Delays are in seconds and Omega in s⁻¹. '
        'Initialization reuses the original physical centers saved with the completed 500 ms joint fit, '
        'which came from the same-trial joint Gamma/Omega MLE: its shared delay is copied to all conditions '
        '(Control ≈102.8 ms, ASD ≈173.4 ms).'
        + (' The added centers are `lapse_prob=0.02` and `lapse_prob_right=0.5`.' if mode == 'rt_choice_lapse' else ''), '',
        'NumPyro AutoMultivariateNormal, full-batch float64 CPU, one ELBO particle, seed 0, one initialization per '
        'group, guide scale 0.1; ClippedAdam learning rate 0.0002 and gradient clip 1. '
        f'Run limits are {config["min_steps"]:,}–{config["max_steps"]:,} updates. Every lower 1,000-update mean '
        f'resets patience; stopping requires {config["confirmation_windows"]} later windows and posterior-mean '
        'ranges below 0.1 posterior SD across the final three checkpoints. The selected best checkpoint is restored '
        'and 10,000 constrained joint draws are saved. All four group/variant jobs use separate parallel processes, '
        'checkpoints and bounded CPU allocations.', '',
        '| Group | Final updates | Restored best | Later windows | Terminal max shift (SD) | Best locally stable | Status |',
        '|---|---:|---:|---:|---:|---|---|',
    ]
    for group in ['Control', 'ASD']:
        r = json.loads((out / group.lower() / 'convergence_summary.json').read_text())
        status = '**PROVISIONAL: unconfirmed**' if r['flagged'] else 'Confirmation and terminal stability passed'
        lines.append(f'| {group} | {r["completed_steps"]:,} | {r["restored_best_step"]:,} | '
                     f'{r["confirmation_windows"]} | {r["maximum_terminal_mean_change_sd"]:.3f} | '
                     f'{r["restored_checkpoint_parameter_stable"]} | {status} |')
    boundaries = pd.read_csv(out / 'posterior_boundary_audit.csv')
    near = np.maximum(boundaries.fraction_within_1pct_lower_hard,
                      boundaries.fraction_within_1pct_upper_hard) >= .05
    lines += [
        '', 'The mean-stability stopping gate applies to the terminal checkpoints. The restored-best checkpoint’s '
        'own local stability is recorded separately. These checks do not establish a global optimum. '
        '**ELBO values from different objectives are not used as model-comparison scores.**', '',
        f'![{labels[mode]} convergence and physical posterior dependence]({base}/human_gamma_omega_convergence_correlations.png)', '',
        f'{int(near.sum())} of {2 * n_params} parameters have at least 5% of posterior draws in the outer 1% of a '
        'hard prior interval. Boundary and correlation tables below identify these parameters directly.', '',
    ]
    if near.any():
        boundary_stimuli = boundaries.loc[near].merge(
            pd.read_csv(out / 'parameter_summary.csv')[['Group', 'parameter', 'ASL_cd', 'signed_delta']],
            on=['Group', 'parameter'], validate='one_to_one')
        lines += [
            '| Group | Parameter | ASL (cd/m²), ΔdB | Hard interval | Draws near lower bound | Draws near upper bound |',
            '|---|---|---|---:|---:|---:|',
        ]
        for row in boundary_stimuli.itertuples():
            stimulus = 'Shared' if pd.isna(row.ASL_cd) else f'{row.ASL_cd:g}, {row.signed_delta:+g}'
            lines.append(f'| {row.Group} | `{row.parameter}` | {stimulus} | [{row.hard_low:g}, {row.hard_high:g}] | '
                         f'{100 * row.fraction_within_1pct_lower_hard:.1f}% | '
                         f'{100 * row.fraction_within_1pct_upper_hard:.1f}% |')
        lines += ['', 'Near-bound fractions use the outer 1% of the complete hard interval, '
                  'not exact contact with the bound.', '']
    if mode == 'rt_only':
        lines += ['Marginalizing choice weakens information about drift direction. The RT-only objective '
                  'does not require its psychometric predictions to match observed choices; drift reversals '
                  'and boundary pressure should therefore be interpreted alongside the choice predictions.', '']
    lines += ['| Group | Strongest physical posterior pair | Correlation |', '|---|---|---:|']
    correlations = pd.read_csv(out / 'strongest_correlations.csv')
    for group in ['Control', 'ASD']:
        for row in correlations.loc[correlations.Group.eq(group)].head(3).itertuples():
            lines.append(f'| {group} | `{row.parameter_1}` / `{row.parameter_2}` | {row.correlation:.3f} |')
    lines += [
        '', f'[Convergence/correlation PDF]({base}/human_gamma_omega_convergence_correlations.pdf) · '
        f'[Boundary audit]({base}/posterior_boundary_audit.csv) · '
        f'[Strongest correlations]({base}/strongest_correlations.csv)', '',
        f'[Control conditions and delay supports]({base}/control/conditions.csv) · '
        f'[ASD conditions and delay supports]({base}/asd/conditions.csv) · '
        f'[Control convergence]({base}/control/convergence_summary.json) · '
        f'[ASD convergence]({base}/asd/convergence_summary.json) · '
        f'[Parallel execution record]({base}/execution_schedule.json)', '',
        '### Sources and audit', '', f'Output root: `ASD_Human_Data/{out.name}/`.', '',
    ]
    for name in source_files[:5]:
        lines.append(f'- `{name}` ([source]({base}/source/{Path(name).name})).')
    lines += [
        '', f'[Complete samples/checkpoints/calculations/source audit ZIP]({base}/{archive.name}) · '
        f'[Run configuration and source hashes]({base}/run_config.json) · '
        f'[Independent verification]({base}/independent_validation.json) · '
        f'[Independent preflight checks]({base}/independent_preflight_validation.json) · '
        f'[Visual inspection]({base}/visual_inspection.json)', '',
        'Verification checks trial identities, stimulus/choice coding, parameter counts and priors, delay support, '
        'RT-only choice-label invariance, zero-/pure-lapse limits, density/CDF normalization, finite gradients '
        'and K=40 versus K=80 accuracy. Empirical probabilities/histograms, mixture weights, posterior summaries '
        'and the convergence rule are independently reconstructed. The original joint fits and earlier analyses '
        'are preserved.',
    ]
    current_page = replace_section(current_page, headings[mode], lines)
    records[mode].update(assets=assets, archive=archive, copy_names=list(dict.fromkeys(copy_names)))

# %% Common comparison lives with the lapse variant and retains baseline source provenance.
if 'rt_choice_lapse' in modes:
    out = paths['rt_choice_lapse']
    base = f'../assets/results/{date}/{asset_names["rt_choice_lapse"]}'
    lines = [
        f'![Baseline joint, RT-only and joint-with-lapse model comparisons]({base}/human_gamma_omega_variant_comparison.png)', '',
        '*Control is above ASD; psychometric panels are left and RTD panels right. Blue/orange label ASL 20/40. '
        'Dashed curves are the completed joint no-lapse fit, dotted curves the RT-only fit, and solid curves the '
        'joint-with-lapse fit. Data are the same empirical choice dots and 20 ms RTD steps. Each RTD pools all '
        'signed stimulus conditions using actual retained frequencies, after normalizing each signed condition '
        'on 0–1.8 s. All curves use arithmetic posterior means. RT-only psychometrics are predictions whose choices '
        'were not fitted.*', '',
        f'[Full-size PNG]({base}/human_gamma_omega_variant_comparison.png) · '
        f'[PDF]({base}/human_gamma_omega_variant_comparison.pdf)', '',
        '| Group / model | Parameters | Common marginal RT log likelihood | Joint log likelihood | Psychometric RMSE | Mean CDF gap | Maximum CDF gap |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in pd.read_csv(out / 'variant_comparison_metrics.csv').itertuples():
        joint = '—' if np.isnan(row.joint_log_likelihood) else f'{row.joint_log_likelihood:.2f}'
        lines.append(f'| {row.Group}: {row.model} | {int(row.free_parameters)} | '
                     f'{row.marginal_rt_log_likelihood:.2f} | {joint} | {row.psychometric_rmse:.4f} | '
                     f'{row.mean_condition_cdf_gap:.4f} | {row.max_condition_cdf_gap:.4f} |')
    lines += [
        '', 'The marginal RT likelihood is evaluated on the same retained rows for all models. Joint RT-and-choice '
        'likelihood is reported only for the two joint models. Likelihood units/objectives differ; compare values '
        'within the same column. Objective-specific ELBOs are convergence diagnostics. These are in-sample '
        'comparisons at physical posterior means and do not establish generalization or a global optimum.', '',
        f'[Comparison metrics]({base}/variant_comparison_metrics.csv) · '
        f'[Signed-condition probabilities and CDF gaps]({base}/variant_comparison_condition_fit_quality.csv) · '
        f'[Pooled RTD curves]({base}/variant_comparison_pooled_rtd_curves.csv) · '
        f'[Pooled histograms]({base}/variant_comparison_pooled_rtd_histograms.csv) · '
        f'[Actual condition mixture weights]({base}/variant_comparison_mixture_weights.csv) · '
        f'[Comparison provenance]({base}/variant_comparison_manifest.json)', '',
        'Source: `ASD_Human_Data/plot_human_gamma_omega_variants_svi.py` '
        f'([source]({base}/source/plot_human_gamma_omega_variants_svi.py)). '
        f'The baseline is `{BASELINE_ROOT.relative_to(ROOT)}`; new fit roots and posterior source hashes are in '
        'the comparison manifest.',
    ]
    current_page = replace_section(current_page, comparison_heading, lines)
page.write_text(current_page)
if not args.update_existing:
    assert page.read_text().startswith(original_page)
index = ROOT / 'docs/results/index.md'
index_lines = index.read_text().splitlines()
addition = 'human Gamma–Omega RT-only and joint-with-lapse comparisons'
found = False
for i, line in enumerate(index_lines):
    if line.startswith(f'- [{date}]'):
        found = True
        if addition not in line:
            index_lines[i] = line + '; ' + addition
if not found:
    first_date = next((i for i, line in enumerate(index_lines) if line.startswith('- [20')), len(index_lines))
    index_lines.insert(first_date, f'- [{date}]({date}.md): {addition}')
index.write_text('\n'.join(index_lines) + '\n')
assert f'results/{date}.md' in (ROOT / 'mkdocs.yml').read_text(), 'Date navigation entry missing'
for mode in modes:
    info = records[mode]
    assets = info['assets']
    manifest = dict(mode=mode, result_page=str(page.relative_to(ROOT)), assets=str(assets.relative_to(ROOT)),
        result_date=date, page_sha256=sha256(page),
        previous_page_sha256=sha256(paths[mode] / 'result_book_page_before.md'),
        prior_page_preserved_as_prefix=not args.update_existing,
        navigation=f'Existing {date} entry verified',
        sources={name: sha256(ROOT / name) for name in source_files},
        archive_sha256=sha256(info['archive']),
        figures={name: sha256(assets / name) for name in info['copy_names'] if name.endswith(('.png', '.pdf'))})
    (paths[mode] / 'publication.json').write_text(json.dumps(manifest, indent=2))
print('Published', ', '.join(modes), 'on', page)
