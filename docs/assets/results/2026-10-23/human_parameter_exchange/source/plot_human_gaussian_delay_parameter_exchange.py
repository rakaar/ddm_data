# %%
"""Model-only Omega and Gaussian-delay exchanges between Control and ASD."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/human_gaussian_delay_mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# %% Editable settings and scenario definitions.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIT = ROOT / 'ASD_Human_Data/gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms'
GROUPS = ['Control', 'ASD']
COLORS = {'Control': '#1976d2', 'ASD': '#d62728'}
DELTAS = [.1, .3, .65, 1.3, 2.]
BLOCKS = ['gamma', 'omega', 'delay_mu', 'delay_sigma', 'w', 'lapse_prob', 'lapse_prob_right']
MODELS = {
    'control': ('Control', None, []),
    'asd': ('ASD', None, []),
    'asd_control_omega': ('ASD', 'Control', ['omega']),
    'control_asd_omega': ('Control', 'ASD', ['omega']),
    'asd_control_delay': ('ASD', 'Control', ['delay_mu', 'delay_sigma']),
    'control_asd_delay': ('Control', 'ASD', ['delay_mu', 'delay_sigma']),
}
LABELS = {
    'control': 'Control: all Control parameters',
    'asd': 'ASD: all ASD parameters',
    'asd_control_omega': 'ASD: all ASD parameters except Omega from Control',
    'control_asd_omega': 'Control: all Control parameters except Omega from ASD',
    'asd_control_delay': 'ASD: all ASD parameters except delay μ and σ from Control',
    'control_asd_delay': 'Control: all Control parameters except delay μ and σ from ASD',
}
SCENARIOS = {
    '1': ['control', 'asd_control_omega'],
    '2': ['control_asd_omega', 'asd'],
    '3': ['control', 'asd_control_delay'],
    '4': ['control_asd_delay', 'asd'],
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fit-dir', type=Path, default=DEFAULT_FIT)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--scenario', choices=['all', *SCENARIOS], default='all')
parser.add_argument('--threads', type=int, default=8)
args = parser.parse_args()
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.threads])
import jax
from human_gaussian_delay_svi_utils_wide import prediction_components

fit = args.fit_dir.resolve()
out = args.output_dir.resolve() if args.output_dir else fit / 'parameter_exchange'
assert out != fit
out.mkdir(parents=True, exist_ok=True)
times = np.linspace(0., 1.8, 3601)
fine_times = np.linspace(0., 1.8, 7201)
chosen = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]
print(f'Source: {fit}\nOutput: {out}\nScenarios: {chosen}', flush=True)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


# %% Freeze existing artifacts; only new derived outputs are written.
baseline_path = out / 'preservation_baseline.json'
if not baseline_path.exists():
    existing = [p for p in fit.rglob('*') if p.is_file() and not p.is_relative_to(out)]
    existing += [p for p in (ROOT / 'docs/assets/results/2026-10-23').rglob('*') if p.is_file()]
    page = ROOT / 'docs/results/2026-10-23.md'
    baseline_path.write_text(json.dumps(dict(files={str(p): sha256(p) for p in existing},
        page=str(page), page_bytes=page.stat().st_size, page_prefix_sha256=sha256(page)), indent=2)+'\n')
config = json.loads((fit / 'run_config.json').read_text())
assert config['mode'] == 'gaussian_delay_lapse' and config['rt_window_s'] == [0., 1.8]
for path, digest in config['source_hashes'].items():
    assert sha256(ROOT / path) == digest, path
conditions, original = {}, {}
source_files = [fit / 'run_config.json', fit / 'condition_model_curves.csv', fit / 'psychometric_comparison.csv']
for group in GROUPS:
    folder = fit / group.lower()
    conditions[group] = pd.read_csv(folder / 'conditions.csv', float_precision='round_trip').sort_values('condition_index')
    with np.load(folder / 'posterior_samples.npz') as samples:
        original[group] = {k: np.asarray(samples[k].mean(axis=0)) for k in BLOCKS}
    source_files += [folder / 'conditions.csv', folder / 'posterior_samples.npz']
mapping = ['condition_index', 'ASL_cd', 'signed_delta']
pd.testing.assert_frame_equal(conditions['Control'][mapping], conditions['ASD'][mapping])
assert conditions['Control'].n.sum() == 6574 and conditions['ASD'].n.sum() == 4770
previous = pd.read_csv(fit / 'condition_model_curves.csv', float_precision='round_trip')
previous_psych = pd.read_csv(fit / 'psychometric_comparison.csv', float_precision='round_trip')

# %% Six distinct models: two baselines plus four exchanges, evaluated without refitting.
parameters, predictions, numerical, parameter_rows, baseline_checks = {}, {}, [], [], []
for model, (recipient, donor, swapped) in MODELS.items():
    params = {k: original[donor if k in swapped else recipient][k].copy() for k in BLOCKS}
    parameters[model] = params
    for block in BLOCKS:
        origin = donor if block in swapped else recipient
        np.testing.assert_array_equal(params[block], original[origin][block])
        for i, value in enumerate(np.atleast_1d(params[block])):
            condition = conditions[recipient].iloc[i] if block in BLOCKS[:4] else None
            parameter_rows.append(dict(model=model, recipient=recipient, parameter=block,
                condition_index=i if condition is not None else -1,
                ASL_cd=condition.ASL_cd if condition is not None else np.nan,
                signed_delta=condition.signed_delta if condition is not None else np.nan,
                source_group=origin, value=float(value)))
    print(f'Evaluating {model}', flush=True)
    pred = prediction_components(params, times)
    predictions[model] = pred
    np.savez_compressed(out / f'{model}_predictions.npz', time_s=times, **pred)
    mass = np.trapz(pred['density'], times, axis=0)
    right_mass = np.trapz(pred['right_density'], times, axis=0)
    assert all(np.isfinite(v).all() for v in pred.values())
    assert pred['density'].min() >= 0 and pred['left_density'].min() >= 0 and pred['right_density'].min() >= 0
    assert np.all((pred['p_right'] >= 0) & (pred['p_right'] <= 1))
    np.testing.assert_allclose(mass, 1., atol=1e-5, rtol=0)
    np.testing.assert_allclose(right_mass, pred['p_right'], atol=1e-5, rtol=0)
    np.testing.assert_allclose(pred['left_density']+pred['right_density'], pred['density'], atol=1e-12)
    numerical.append(dict(model=model, check='unit_mass_and_choice', max_mass_error=float(abs(mass-1).max()),
        max_choice_error=float(abs(right_mass-pred['p_right']).max())))
    if donor is None:
        saved = previous.loc[previous.Group.eq(recipient)]
        maximum = 0.
        for key in ['density', 'left_density', 'right_density', 'cdf']:
            values = saved.pivot(index='time_s', columns='condition_index', values=key).to_numpy()
            error = float(abs(pred[key]-values).max()); maximum = max(maximum, error)
            np.testing.assert_allclose(pred[key], values, atol=1e-9, rtol=1e-9)
        probs = previous_psych.loc[previous_psych.Group.eq(recipient)].sort_values('condition_index').model_p_right
        np.testing.assert_allclose(pred['p_right'], probs, atol=1e-10)
        baseline_checks.append(dict(model=model, max_saved_curve_error=maximum))
    else:
        for name, K, dt, grid in [('K40', 40, .0005, times), ('K80', 80, .0005, times), ('fine_grid', 40, .00025, fine_times)]:
            comparison = prediction_components(params, grid, K=K, dt=dt)
            aligned = {k: v[::2] if name == 'fine_grid' and v.ndim == 2 else v for k, v in comparison.items()}
            row = dict(model=model, check=name,
                max_density_error=float(abs(aligned['density']-pred['density']).max()),
                max_cdf_error=float(abs(aligned['cdf']-pred['cdf']).max()),
                max_choice_error=float(abs(aligned['p_right']-pred['p_right']).max()),
                max_retained_mass_error=float(abs(aligned['retained_mass']-pred['retained_mass']).max()),
                max_mean_error_s=float(abs(np.trapz(comparison['density']*grid[:, None], grid, axis=0)
                                          -np.trapz(pred['density']*times[:, None], times, axis=0)).max()))
            assert row['max_density_error'] < 1e-3, row
            assert row['max_cdf_error'] < 1e-5 and row['max_choice_error'] < 1e-5, row
            assert row['max_mean_error_s'] < 1e-5 and row['max_retained_mass_error'] < 1e-5, row
            numerical.append(row)
            print(f'  {name}: max density difference {row["max_density_error"]:.3g}', flush=True)
            del comparison, aligned
    jax.clear_caches()

# %% Aggregate each model using only its recipient's observed condition frequencies.
condition_curves, condition_points, weights, pooled_curves, psych, chrono = [], [], [], [], [], []
for model, (recipient, donor, swapped) in MODELS.items():
    pred = predictions[model]; ct = conditions[recipient]
    for c in ct.itertuples():
        i = c.condition_index
        condition_points.append(dict(model=model, recipient=recipient, condition_index=i, ASL_cd=c.ASL_cd,
            signed_delta=c.signed_delta, n=c.n, p_right=pred['p_right'][i], retained_mass=pred['retained_mass'][i],
            mean_RT_s=np.trapz(times*pred['density'][:, i], times)))
        condition_curves.append(pd.DataFrame(dict(model=model, recipient=recipient, condition_index=i,
            ASL_cd=c.ASL_cd, signed_delta=c.signed_delta, time_s=times,
            **{k: pred[k][:, i] for k in ['left_density', 'right_density', 'density', 'cdf']})))
    for delta in DELTAS:
        pair = ct.loc[np.isclose(ct.signed_delta.abs(), delta)].sort_values('condition_index')
        ids = pair.condition_index.to_numpy(int); w = pair.n.to_numpy()/pair.n.sum()
        density = pred['density'][:, ids] @ w; cdf = pred['cdf'][:, ids] @ w
        mean = float(np.trapz(times*density, times)); cdf_mean = float(np.trapz(1-cdf, times))
        assert abs(mean-cdf_mean) < 1e-6
        pooled_curves.append(pd.DataFrame(dict(model=model, recipient=recipient, abs_delta=delta,
            time_s=times, density=density, cdf=cdf)))
        chrono.append(dict(model=model, recipient=recipient, abs_delta=delta, n=pair.n.sum(),
            mean_RT_s=mean, cdf_mean_RT_s=cdf_mean))
        for c, weight in zip(pair.itertuples(), w):
            weights.append(dict(model=model, recipient=recipient, panel='rtd_chronometric', delta=delta,
                condition_index=c.condition_index, n=c.n, total_n=pair.n.sum(), weight=weight))
    for delta, pair in ct.groupby('signed_delta', sort=True):
        ids = pair.condition_index.to_numpy(int); w = pair.n.to_numpy()/pair.n.sum()
        psych.append(dict(model=model, recipient=recipient, signed_delta=delta, n=pair.n.sum(), p_right=pred['p_right'][ids] @ w))
        for c, weight in zip(pair.itertuples(), w):
            weights.append(dict(model=model, recipient=recipient, panel='psychometric', delta=delta,
                condition_index=c.condition_index, n=c.n, total_n=pair.n.sum(), weight=weight))
pooled_curves = pd.concat(pooled_curves, ignore_index=True)
psych, chrono = pd.DataFrame(psych), pd.DataFrame(chrono)
tables = {'model_parameters.csv': pd.DataFrame(parameter_rows), 'condition_predictions.csv': pd.DataFrame(condition_points),
    'condition_model_curves.csv': pd.concat(condition_curves, ignore_index=True), 'mixture_weights.csv': pd.DataFrame(weights),
    'pooled_rtd_curves.csv': pooled_curves, 'psychometric_predictions.csv': psych, 'chronometric_predictions.csv': chrono,
    'numerical_checks.csv': pd.DataFrame(numerical), 'baseline_checks.csv': pd.DataFrame(baseline_checks)}
for name, table in tables.items():
    table.to_csv(out / name, index=False, float_format='%.17g')

# %% All four figures share axes; no empirical histograms, scatter or error bars.
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'})
ymax = float(np.ceil(pooled_curves.density.max()*1.08*2)/2)
chrono_limits = [float(np.floor((chrono.mean_RT_s.min()-.035)*20)/20),
                 float(np.ceil((chrono.mean_RT_s.max()+.035)*20)/20)]
figures = []
for scenario in chosen:
    ids = SCENARIOS[scenario]
    fig = plt.figure(figsize=(26, 5.8))
    grid = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1.6, 1.4],
                           left=.035, right=.99, bottom=.24, top=.66, wspace=.38)
    for c, delta in enumerate(DELTAS):
        ax = fig.add_subplot(grid[0, c])
        for model in ids:
            group = MODELS[model][0]
            p = pooled_curves.loc[pooled_curves.model.eq(model) & pooled_curves.abs_delta.eq(delta)]
            ax.plot(p.time_s, p.density, color=COLORS[group], lw=2.2)
        ax.set(title=f'|ΔdB| = {delta:g}', xlabel='RT (s)', xlim=(0, 1.8), ylim=(0, ymax), xticks=[0, .6, 1.2, 1.8])
        ax.grid(alpha=.16)
        if c == 0: ax.set_ylabel('Density (s⁻¹)')
        else: ax.tick_params(labelleft=False)
    ax = fig.add_subplot(grid[0, 5])
    for model in ids:
        p = psych.loc[psych.model.eq(model)].sort_values('signed_delta')
        ax.plot(p.signed_delta, p.p_right, color=COLORS[MODELS[model][0]], lw=2.2)
    ax.set(title='Psychometric\nBoth ASLs pooled', xlabel='Signed ΔdB (right − left)', ylabel='P(right)',
           xlim=(-2.15, 2.15), ylim=(-.015, 1.015), xticks=[-2, -1, 0, 1, 2]); ax.grid(alpha=.18)
    ax = fig.add_subplot(grid[0, 6])
    for model in ids:
        p = chrono.loc[chrono.model.eq(model)].sort_values('abs_delta')
        ax.plot(p.abs_delta, p.mean_RT_s, color=COLORS[MODELS[model][0]], lw=2.2)
    ax.set(title='Chronometric\nBoth ASLs pooled', xlabel='|ΔdB|', ylabel='Mean RT (s)', xlim=(0, 2.1),
           ylim=chrono_limits, xticks=DELTAS, xticklabels=['0.1', '0.3', '0.65', '1.3', '2']); ax.grid(alpha=.18)
    fig.suptitle(f'Parameter exchange {scenario} | model only | both ASLs pooled', fontsize=18, y=.98)
    fig.text(.5, .908, LABELS[ids[0]], ha='center', color=COLORS['Control'], fontsize=15)
    fig.text(.5, .846, LABELS[ids[1]], ha='center', color=COLORS['ASD'], fontsize=15)
    fig.legend(handles=[Line2D([], [], color=COLORS[MODELS[m][0]], lw=2.2, label=LABELS[m]) for m in ids],
               loc='upper center', bbox_to_anchor=(.5, .80), ncol=2, frameon=False, fontsize=11)
    fig.text(.5, .12, 'RT-fitted arithmetic posterior means | Each signed condition renormalized over 0–1.8 s after exchange | No refitting', ha='center', fontsize=10)
    fig.text(.5, .075, 'ASLs 20 + 40 cd/m² pooled using the receiving group’s retained-trial weights; RTDs and chronometrics also pool ±Δ', ha='center', fontsize=10)
    fig.text(.5, .03, 'Gaussian delay truncated at zero; lapse RT uniform on 0–1.8 s | Model points joined at measured differences | Parameter sensitivity, not causal evidence', ha='center', fontsize=10)
    stem = f'human_parameter_exchange_{scenario}'
    for suffix in ['png', 'pdf']:
        name = f'{stem}.{suffix}'; fig.savefig(out / name, dpi=180); figures.append(name)
    plt.close(fig)

# %% Provenance and prior-artifact preservation.
baseline = json.loads(baseline_path.read_text())
for path, digest in baseline['files'].items():
    assert sha256(path) == digest, path
assert hashlib.sha256(Path(baseline['page']).read_bytes()[:baseline['page_bytes']]).hexdigest() == baseline['page_prefix_sha256']
source_files += [ROOT / p for p in config['source_hashes']] + [Path(__file__).resolve()]
manifest = dict(fit_dir=str(fit), scenarios={s: dict(models=SCENARIOS[s], labels=[LABELS[m] for m in SCENARIOS[s]]) for s in chosen},
    source_hashes={str(p): sha256(p) for p in source_files}, figures=figures, tables=list(tables),
    parameter_convention='arithmetic physical posterior means', refitted=False,
    weights='recipient retained condition counts; 6574 Control and 4770 ASD RT trials',
    rt_window_s=[0., 1.8], delay_exchange=['delay_mu', 'delay_sigma'],
    axes=dict(rtd_ylim=[0., ymax], rt_xlim=[0., 1.8], psychometric_ylim=[-.015, 1.015], chronometric_ylim=chrono_limits),
    numerical_checks_passed=True, previous_files_preserved=len(baseline['files']))
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Finished all selected figures and numerical checks.', flush=True)
