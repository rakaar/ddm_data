# %% ASD0047 full-batch joint GPU SVI: proactive + reactive + contaminants
import argparse
import json
import os
import pickle
from pathlib import Path
from time import perf_counter

# %% Editable run settings are imported from the new joint-only utility module
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--seed', type=int, choices=[0, 1], default=0)
parser.add_argument('--benchmark-only', action='store_true')
parser.add_argument('--benchmark-steps', type=int, default=30)
parser.add_argument('--smoke-test', action='store_true')
parser.add_argument('--smoke-folder', default='validation_smoke')
parser.add_argument('--checkpoint-only-after', type=int)
args = parser.parse_args()
os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.35')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
if os.environ.get('ASDFIT_CPUS'):
    os.sched_setaffinity(0, {int(x) for x in os.environ['ASDFIT_CPUS'].split(',')})
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
from shank3_asd0047_joint_utils import (PARAMETER_COUNT, VARIANT, FIXED_PARAMETERS, ROOT, OUT, BLOCKS, make_model, log_likelihood,
    parameter_names, sample_matrix, sha256, loss_window_decision, WINDOW, MIN_STEPS,
    CONFIRMATION, MAX_STEPS, LEARNING_RATE, CLIP_NORM, INIT_SCALE, DRAWS)

round_folder = OUT / f'round_{args.round:02d}'
policy = json.loads((OUT / 'likelihood_policy.json').read_text())
assert policy['status'] == 'confirmed'
assert policy['exclude_valid_RT_above_1'] and policy['joint_retained_window_normalization'] and policy['condition_on_survival_to_fixation']
priors = json.loads((round_folder / 'bounds.json').read_text())
initial = {k: np.asarray(v) for k, v in json.loads((round_folder / 'initialization.json').read_text())[str(args.seed)].items()}
grid_end = max(1.5, 1+priors['del_go'][-1])
model = make_model(priors, grid_end)
prep = json.loads((OUT / 'preparation.json').read_text())
for name, expected in prep['input_hashes'].items():
    assert sha256(OUT / name) == expected, name
assert json.loads((round_folder / 'prefit_validation.json').read_text())['status'] == 'passed'
data = {k: jnp.asarray(v) for k, v in np.load(OUT / 'observations.npz').items()}
names = parameter_names()
assert len(names) == PARAMETER_COUNT
DIAGNOSTIC_DRAWS = 2048
folder = round_folder / (args.smoke_folder if args.smoke_test else f'seed_{args.seed}')
folder.mkdir(exist_ok=True)
if args.smoke_test:
    ids = np.array([np.flatnonzero(np.asarray(data['condition']) == i)[0] for i in range(30)])
    for name in ['rt', 'onset', 'choice', 'condition']:
        data[name] = data[name][ids]
    for name in ['abort_times', 'abort_counts', 'late_onsets', 'late_condition', 'parent_onsets', 'parent_condition']:
        data[name] = data[name][:30]
    data['parent_count'] = len(ids)+data['abort_counts'].sum()+data['late_onsets'].size
    WINDOW, MIN_STEPS, CONFIRMATION, MAX_STEPS, DRAWS, DIAGNOSTIC_DRAWS = 5, 10, 2, 15, 128, 128
else:
    assert jax.default_backend() == 'gpu'

# %% Pin exact inputs and every reused kernel before any optimizer state is saved
source_paths = [Path(__file__), ROOT / 'asd_analysis/shank3_asd0047_joint_utils.py', ROOT / 'asd_analysis/shank3_wt_joint_contaminant_utils.py',
    ROOT / 'asd_analysis/shank3_individual_adaptive_bounds_utils.py',
    ROOT / 'asd_analysis/shank3_npl_alpha_gaussian_abl_delay_utils.py',
    ROOT / 'asd_analysis/shank3_npl_alpha_gaussian_utils.py',
    ROOT / 'asd_analysis/shank3_wt_npl_alpha_utils.py', ROOT / 'asd_analysis/shank3_wt_gamma_omega_utils.py',
    ROOT / 'asd_analysis/proactive_lapse_no_led_utils.py', ROOT / 'ASD_Human_Data/gaussian_delay_likelihood.py',
    ROOT / 'fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py', OUT / 'preparation.json',
    OUT / 'likelihood_policy.json', OUT / 'observations.npz', round_folder / 'bounds.json', round_folder / 'initialization.json']
config = dict(model='ASD0047 joint proactive/NPL+alpha/Gaussian-ABL/'+VARIANT,
    seed=args.seed, bounds_round=args.round, parameters=PARAMETER_COUNT, parameter_order=names, blocks=BLOCKS,
    priors=priors, initial={k: v.tolist() for k, v in initial.items()},
    frozen_parameters=list(FIXED_PARAMETERS), fixed_parameters=FIXED_PARAMETERS, policy=policy, backend=jax.default_backend(), device=jax.devices()[0].device_kind,
    float64=True, full_batch=True, guide='AutoMultivariateNormal', learning_rate=LEARNING_RATE,
    clip_norm=CLIP_NORM, guide_scale=INIT_SCALE, window=WINDOW, min_steps=MIN_STEPS,
    confirmation_windows=CONFIRMATION, max_steps=MAX_STEPS, parameter_stability_required=False,
    posterior_draws=DRAWS, diagnostic_draws=DIAGNOSTIC_DRAWS, grid_end=grid_end,
    jax=jax.__version__, numpyro=numpyro.__version__, smoke_test=args.smoke_test,
    numerical_kernel='Unmodified corrected adaptive-bound Gaussian race,0.5ms grid with5x/25x/direct safeguards.',
    source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in source_paths})
config = json.loads(json.dumps(config))
if not args.benchmark_only:
    assert not (folder / 'run_summary.json').exists(), 'Completed fit remains preserved.'
    if (folder / 'configuration.json').exists():
        assert json.loads((folder / 'configuration.json').read_text()) == config, 'Changed configuration cannot resume saved state.'
    else:
        (folder / 'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
        import shutil
        for source in source_paths:
            path = OUT / 'source_history' / sha256(source) / source.name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                shutil.copy2(source, path)

guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values=initial), init_scale=INIT_SCALE)
svi = SVI(model, guide, numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM), Trace_ELBO())
state = svi.init(jax.random.PRNGKey(args.seed), data)


def checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(jax.device_get(payload), stream, pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


@jax.jit
def run_window(current):
    return jax.lax.scan(lambda s, _: svi.stable_update(s, data), current, None, length=WINDOW)


@jax.jit
def diagnostic_samples(params):
    return guide.sample_posterior(jax.random.PRNGKey(99173), params, sample_shape=(DIAGNOSTIC_DRAWS,))


# %% Disposable complete-model benchmark; never reuse benchmark optimizer states
if args.benchmark_only:
    timed_run = jax.jit(lambda s: jax.lax.scan(lambda x, _: svi.stable_update(x, data), s, None, length=args.benchmark_steps))
    tick = perf_counter()
    state, loss = timed_run(state)
    jax.block_until_ready(loss)
    compilation = perf_counter()-tick
    tick = perf_counter()
    state, loss = timed_run(state)
    loss = np.asarray(loss)
    assert np.isfinite(loss).all()
    stats = jax.devices()[0].memory_stats()
    import ctypes
    cuda = ctypes.CDLL('libcuda.so.1')
    assert cuda.cuInit(0) == 0
    device, physical = ctypes.c_int(), ctypes.c_size_t()
    assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0
    assert cuda.cuDeviceTotalMem_v2(ctypes.byref(physical), device) == 0
    record = dict(status='passed', compile_seconds=compilation,
        seconds_per_update=(perf_counter()-tick)/args.benchmark_steps,
        physical_memory_bytes=physical.value, memory_stats=stats,
        initial_loss=float(loss[0]), full_batch=True, states_discarded=True)
    (round_folder / f'gpu_benchmark_seed{args.seed}.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(0)

# %% Commit complete1000-update windows atomically; replay exact state on resume
step, patience, best_loss, best_step = 0, 0, np.inf, 0
losses, rows, history, sd_history = [], [], [], []
best = None
if (folder / 'latest_checkpoint.pkl').exists():
    latest = pickle.loads((folder / 'latest_checkpoint.pkl').read_bytes())
    state, step, patience, best = latest['state'], latest['step'], latest['patience'], latest['best']
    best_loss, best_step = best['loss'], best['step']
    losses, rows = latest['losses'].tolist(), latest['rows']
    history = [[r[n+'_mean'] for n in names] for r in rows]
    sd_history = [[r[n+'_sd'] for n in names] for r in rows]
    assert len(losses) == step == rows[-1]['step']
    checkpoint(folder / 'best_checkpoint.pkl', best)
started = perf_counter()
while step < MAX_STEPS and not (step >= MIN_STEPS and patience >= CONFIRMATION):
    tick = perf_counter()
    state, window_loss = run_window(state)
    window_loss = np.asarray(window_loss)
    assert np.isfinite(window_loss).all(), 'Nonfinite loss; last complete checkpoint is retained.'
    step += WINDOW
    losses.extend(window_loss.tolist())
    mean = float(window_loss.mean())
    params = svi.get_params(state)
    improved, patience, stop = loss_window_decision(mean, best_loss, step, patience, MIN_STEPS, CONFIRMATION)
    if improved:
        best_loss, best_step = mean, step
        best = dict(state=state, params=params, loss=mean, step=step)
        checkpoint(folder / 'best_checkpoint.pkl', best)
    matrix = sample_matrix(diagnostic_samples(params))
    means, sds = matrix.mean(0), matrix.std(0, ddof=1)
    history.append(means.tolist())
    sd_history.append(sds.tolist())
    change = np.ptp(history[-3:], axis=0)/np.maximum(np.min(sd_history[-3:], axis=0), 1e-300) if len(history) >= 3 else np.full(PARAMETER_COUNT, np.inf)
    row = dict(step=step, mean_negative_elbo=mean, best_loss=best_loss, best_step=best_step,
        patience=patience, updated_best=improved, max_mean_change_sd=float(change.max()),
        parameter_stable=bool(np.all(change < .1)), window_seconds=perf_counter()-tick)
    for i, name in enumerate(names):
        row[name+'_mean'], row[name+'_sd'] = means[i], sds[i]
    rows.append(row)
    checkpoint(folder / 'latest_checkpoint.pkl', dict(state=state, step=step, patience=patience,
        best=best, losses=np.asarray(losses), rows=rows))
    np.save(folder / 'losses.npy', np.asarray(losses))
    pd.DataFrame(rows).to_csv(folder / 'convergence.csv', index=False, float_format='%.17g')
    print(f'ASD0047 joint seed{args.seed} round{args.round} {step:,}: loss={mean:.5f} best={best_step:,} tail={patience}/{CONFIRMATION} seconds={row["window_seconds"]:.1f}', flush=True)
    if args.checkpoint_only_after and step >= args.checkpoint_only_after:
        print('Stopped at requested complete checkpoint; rerun without this option to resume.', flush=True)
        raise SystemExit(0)
    if stop:
        break

# %% Restore best full-rank guide and optimizer; preserve latest separately
samples = {k: np.asarray(v) for k, v in guide.sample_posterior(jax.random.PRNGKey(args.seed+100000),
    best['params'], sample_shape=(DRAWS,)).items()}
assert all(np.isfinite(x).all() for x in samples.values())
matrix = sample_matrix(samples)
np.savez_compressed(folder / 'posterior_samples.npz', **samples)
pd.DataFrame(dict(parameter=names, mean=matrix.mean(0), sd=matrix.std(0, ddof=1),
    q025=np.quantile(matrix, .025, axis=0), median=np.median(matrix, axis=0),
    q975=np.quantile(matrix, .975, axis=0))).to_csv(folder / 'posterior_summary.csv', index=False, float_format='%.17g')
np.save(folder / 'posterior_covariance.npy', np.cov(matrix, rowvar=False))
checkpoint(folder / 'restored_guide_optimizer.pkl', best)
mean_params = {k: jnp.asarray(v.mean(0)) for k, v in samples.items()}
confirmed = step >= MIN_STEPS and patience >= CONFIRMATION
summary = dict(status='converged' if confirmed else 'inconclusive', seed=args.seed,
    bounds_round=args.round, parameter_count=PARAMETER_COUNT, parent_trials=prep['parent'],
    valid_trials=prep['valid'], abort_events=prep['abort_events'], animals=1,
    final_checked_step=step, restored_best_step=best_step, best_window_negative_elbo=best_loss,
    confirmation_windows=patience, parameter_stability_required=False,
    stop_reason='confirmed_minimum_and_loss_patience' if confirmed else 'unconfirmed_limit_hit',
    posterior_draws=DRAWS, posterior_mean_joint_log_likelihood=float(log_likelihood(mean_params, data, grid_end)),
    elapsed_seconds_this_invocation=perf_counter()-started)
(folder / 'run_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2), flush=True)
