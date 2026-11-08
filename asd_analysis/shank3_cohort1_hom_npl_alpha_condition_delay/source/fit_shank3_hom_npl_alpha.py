# %% Editable full-batch GPU SVI settings and resumable seed-0 initialization
"""NPL+alpha, 30 condition delays, and fixed HOM proactive/exponential lapses."""
import argparse
import json
import os
import pickle
from pathlib import Path
from time import perf_counter
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--benchmark-only', action='store_true')
parser.add_argument('--benchmark-steps', type=int, default=30)
parser.add_argument('--smoke-test', action='store_true')
parser.add_argument('--checkpoint-only-after', type=int)
args = parser.parse_args()
os.environ['JAX_PLATFORMS'] = 'cpu' if args.smoke_test else 'cuda'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
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
from shank3_hom_npl_alpha_utils import (ROOT, OUT, K_DEFAULT, INITIAL, PRIORS, model, log_likelihood,
    parameter_names, sample_matrix, loss_window_decision, sha256)

SEED, LEARNING_RATE, CLIP_NORM, INIT_SCALE = 0, .0002, 1., .1
WINDOW, MIN_STEPS, MAX_STEPS, CONFIRMATION = 1000, 50000, 1000000, 25
DRAWS, DIAGNOSTIC_DRAWS = 10000, 2048
assert json.loads((OUT/'prefit_validation.json').read_text())['status'] == 'passed'
prep = json.loads((OUT/'preparation.json').read_text())
for name, digest in prep['input_hashes'].items():
    assert sha256(OUT/name) == digest, name
data = {k: jnp.asarray(v) for k,v in np.load(OUT/'likelihood_data.npz').items() if k != 'onset'}
initial = {k: np.asarray(v) for k,v in INITIAL.items()}
folder = OUT/'validation_smoke' if args.smoke_test else OUT/'seed_0'
folder.mkdir(exist_ok=True)
if args.smoke_test:
    ids = np.array([np.flatnonzero(np.asarray(data['condition']) == i)[0] for i in range(30)])
    data = {k: v[ids] if v.ndim else v for k,v in data.items()}
    WINDOW, MIN_STEPS, MAX_STEPS, CONFIRMATION, DRAWS, DIAGNOSTIC_DRAWS = 5, 10, 15, 2, 128, 128
else:
    assert jax.default_backend() == 'gpu', 'GPU required; CPU fallback is not authorized.'
names = parameter_names()
assert len(names) == 37
config = dict(seed=SEED, parameter_count=37, parameter_order=names, priors=PRIORS,
    initial={k:v.tolist() for k,v in initial.items()}, backend=jax.default_backend(),
    device=jax.devices()[0].device_kind, float64=True, jax=jax.__version__, numpyro=numpyro.__version__,
    learning_rate=LEARNING_RATE, clip_norm=CLIP_NORM, guide_scale=INIT_SCALE, window=WINDOW,
    min_steps=MIN_STEPS, max_steps=MAX_STEPS, confirmation_windows=CONFIRMATION,
    parameter_stability_required=False, draws=DRAWS, diagnostic_draws=DIAGNOSTIC_DRAWS, series_terms=K_DEFAULT,
    source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),
        ROOT/'asd_analysis/shank3_hom_npl_alpha_utils.py', ROOT/'asd_analysis/shank3_wt_npl_alpha_utils.py',
        ROOT/'asd_analysis/shank3_wt_gamma_omega_utils.py',
        ROOT/'asd_analysis/proactive_lapse_no_led_utils.py', ROOT/'fit_animal_by_animal/numpyro_npl_alpha_svi_utils.py',
        OUT/'preparation.json', OUT/'likelihood_data.npz', OUT/'fixed_proactive.json']})
if not args.benchmark_only:
    assert not (folder/'run_summary.json').exists(), 'Completed fit is preserved.'
    path = folder/'configuration.json'
    if path.exists():
        assert json.loads(path.read_text()) == config, 'Incompatible resume configuration.'
    else:
        path.write_text(json.dumps(config, indent=2)+'\n')
guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values=initial), init_scale=INIT_SCALE)
svi = SVI(model, guide, numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM), Trace_ELBO())
state = svi.init(jax.random.PRNGKey(SEED), data)


def checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(jax.device_get(payload), stream, pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


@jax.jit
def run_window(current):
    return jax.lax.scan(lambda s,_: svi.stable_update(s, data), current, None, length=WINDOW)


@jax.jit
def diagnostic_samples(params):
    return guide.sample_posterior(jax.random.PRNGKey(99173), params, sample_shape=(DIAGNOSTIC_DRAWS,))


# %% Disposable timing states are never used to initialize the actual fit
if args.benchmark_only:
    run_benchmark = jax.jit(lambda s: jax.lax.scan(lambda x,_:svi.stable_update(x,data), s, None, length=args.benchmark_steps))
    tick = perf_counter()
    state, loss = run_benchmark(state)
    jax.block_until_ready(loss)
    compile_seconds = perf_counter()-tick
    tick = perf_counter()
    state, loss = run_benchmark(state)
    loss = np.asarray(loss)
    assert np.isfinite(loss).all()
    stats = jax.devices()[0].memory_stats() or {}
    record = dict(device=jax.devices()[0].device_kind, backend=jax.default_backend(), compile_seconds=compile_seconds,
        seconds_per_update=(perf_counter()-tick)/args.benchmark_steps, timed_updates=args.benchmark_steps,
        peak_bytes=stats.get('peak_bytes_in_use'), states_discarded=True, initial_loss=float(loss[0]))
    (OUT/'gpu_benchmark.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(0)

# %% The atomic latest checkpoint is the complete window's commit record
step, best_step, best_loss, patience = 0, 0, np.inf, 0
losses, rows, history, sd_history = [], [], [], []
best = None
if (folder/'latest_checkpoint.pkl').exists():
    latest = pickle.loads((folder/'latest_checkpoint.pkl').read_bytes())
    state, step, patience, best = latest['state'], latest['step'], latest['patience'], latest['best']
    best_step, best_loss = best['step'], best['loss']
    losses, rows = latest['losses'].tolist(), latest['rows']
    history = [[row[n+'_mean'] for n in names] for row in rows]
    sd_history = [[row[n+'_sd'] for n in names] for row in rows]
    assert len(losses) == step == rows[-1]['step']
    checkpoint(folder/'best_checkpoint.pkl', best)
started = perf_counter()
while step < MAX_STEPS and not (step >= MIN_STEPS and patience >= CONFIRMATION):
    tick = perf_counter()
    state, window = run_window(state)
    window = np.asarray(window)
    assert np.isfinite(window).all(), f'Nonfinite loss after {step}; last complete checkpoint retained.'
    step += WINDOW
    losses.extend(window.tolist())
    mean = float(window.mean())
    params = svi.get_params(state)
    improved, patience, stop = loss_window_decision(mean, best_loss, step, patience, MIN_STEPS, CONFIRMATION)
    if improved:
        best_step, best_loss = step, mean
        best = dict(state=state, params=params, step=step, loss=mean)
        checkpoint(folder/'best_checkpoint.pkl', best)
    matrix = sample_matrix(diagnostic_samples(params))
    means, sds = matrix.mean(0), matrix.std(0, ddof=1)
    history.append(means.tolist()); sd_history.append(sds.tolist())
    change = np.ptp(history[-3:], axis=0)/np.maximum(np.min(sd_history[-3:], axis=0), 1e-300) if len(history)>=3 else np.full(37,np.inf)
    row = dict(step=step, mean_negative_elbo=mean, best_loss=best_loss, best_step=best_step, patience=patience,
        updated_best=improved, parameter_stable=bool(np.all(change < .1)), max_mean_change_sd=float(change.max()),
        window_seconds=perf_counter()-tick)
    for i,name in enumerate(names):
        row[name+'_mean'], row[name+'_sd'], row[name+'_change_sd'] = means[i], sds[i], change[i]
    rows.append(row)
    checkpoint(folder/'latest_checkpoint.pkl', dict(state=state, step=step, patience=patience, best=best,
        losses=np.asarray(losses), rows=rows))
    np.save(folder/'losses.npy', np.asarray(losses))
    pd.DataFrame(rows).to_csv(folder/'convergence.csv', index=False, float_format='%.17g')
    print(f'NPL+alpha {step:,}: loss={mean:.5f} best={best_step:,} tail={patience}/{CONFIRMATION} seconds={row["window_seconds"]:.1f}', flush=True)
    if args.checkpoint_only_after and step >= args.checkpoint_only_after:
        print('Paused at requested complete-window checkpoint; resume without this option.', flush=True)
        raise SystemExit(0)
    if stop:
        break

# %% Restore absolute-best state; retain latest state separately
samples = {k:np.asarray(v) for k,v in guide.sample_posterior(jax.random.PRNGKey(SEED+100000),
    best['params'], sample_shape=(DRAWS,)).items()}
assert all(np.isfinite(x).all() for x in samples.values())
matrix = sample_matrix(samples)
np.savez_compressed(folder/'posterior_samples.npz', **samples)
pd.DataFrame(dict(parameter=names, mean=matrix.mean(0), sd=matrix.std(0,ddof=1), q025=np.quantile(matrix,.025,axis=0),
    median=np.median(matrix,axis=0), q975=np.quantile(matrix,.975,axis=0))).to_csv(folder/'posterior_summary.csv',index=False)
np.save(folder/'posterior_covariance.npy', np.cov(matrix,rowvar=False))
checkpoint(folder/'restored_guide_optimizer.pkl', best)
mean_params = {k:jnp.asarray(x.mean(0)) for k,x in samples.items()}
confirmed = step >= MIN_STEPS and patience >= CONFIRMATION
summary = dict(status='converged' if confirmed else 'inconclusive', parameter_count=37, seed=SEED,
    stop_reason='confirmed_minimum_and_loss_patience' if confirmed else 'maximum_steps_without_confirmation',
    final_checked_step=step, restored_best_step=best_step, best_window_negative_elbo=best_loss,
    confirmation_windows=patience, parameter_stability_required=False, terminal_parameter_stable=rows[-1]['parameter_stable'],
    restored_parameter_stable=next(row['parameter_stable'] for row in rows if row['step']==best_step),
    valid_trials=30 if args.smoke_test else 67715, animals=6, posterior_draws=DRAWS,
    posterior_mean_joint_log_likelihood=float(log_likelihood(mean_params,data)),
    posterior_mean_rt_log_likelihood=float(log_likelihood(mean_params,data,marginal=True)),
    elapsed_seconds_this_invocation=perf_counter()-started, initializations=1)
(folder/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
