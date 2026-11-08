# %% Editable fit settings; CPU is allowed only for an isolated tiny smoke test
import argparse
import json
import os
import pickle
from pathlib import Path
from time import perf_counter
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--group', choices=['wt','hom'], required=True)
parser.add_argument('--variant', choices=['gamma_by_ild'], default='gamma_by_ild')
parser.add_argument('--benchmark-only', action='store_true')
parser.add_argument('--benchmark-steps', type=int, default=30)
parser.add_argument('--checkpoint-only-after', type=int)
parser.add_argument('--smoke-test', action='store_true')
args = parser.parse_args()
os.environ['JAX_PLATFORMS'] = 'cpu' if args.smoke_test else 'cuda'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
if os.environ.get('ASDFIT_CPUS'):
    os.sched_setaffinity(0, {int(x) for x in os.environ['ASDFIT_CPUS'].split(',')})
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
from shank3_gamma_by_ild_utils import (ROOT, OUT, PARAM_BLOCKS, K_DEFAULT, EXPECTED, sha256,
    model, expand_parameters, expanded_log_likelihood, loss_window_decision, parameter_names)
from shank3_wt_gamma_omega_utils import sample_matrix

SEED, LEARNING_RATE, CLIP_NORM, INIT_SCALE = 0, .0002, 1., .1
WINDOW, MIN_STEPS, MAX_STEPS, CONFIRMATION = 1000, 50000, 1000000, 25
DRAWS, DIAGNOSTIC_DRAWS = 10000, 2048
prepared = OUT / args.group / args.variant
config = json.loads((prepared / 'prepared_configuration.json').read_text())
assert json.loads((OUT / 'prefit_validation.json').read_text())['status'] == 'passed'
initial = {k:np.asarray(v) for k,v in config['initial'].items()}
data = {k:jnp.asarray(v) for k,v in np.load(OUT / args.group / 'likelihood_data.npz').items() if k != 'onset'}
if args.smoke_test:
    folder = OUT / 'validation_smoke' / args.group / args.variant
    ids = np.array([np.flatnonzero(np.asarray(data['condition']) == i)[0] for i in range(30)])
    data = {k:(v[ids] if v.ndim else v) for k,v in data.items()}
    WINDOW, MIN_STEPS, MAX_STEPS, CONFIRMATION, DRAWS, DIAGNOSTIC_DRAWS = 5, 10, 15, 2, 128, 128
else:
    folder = prepared
    assert jax.default_backend() == 'gpu', 'GPU required; CPU fallback is not authorized.'
folder.mkdir(parents=True, exist_ok=True)
if not args.benchmark_only:
    assert not (folder / 'run_summary.json').exists(), 'Completed run is preserved.'
names = parameter_names(args.variant)
assert len(names) == config['parameter_count']
for name, expected in config['input_hashes'].items():
    assert sha256(OUT / args.group / name) == expected, name
runtime = dict(prepared_configuration_sha256=sha256(prepared / 'prepared_configuration.json'),
    sources={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),
        ROOT / 'asd_analysis/shank3_gamma_by_ild_utils.py', ROOT / 'asd_analysis/shank3_wt_gamma_omega_utils.py',
        ROOT / 'asd_analysis/shank3_tied_omega_utils.py', ROOT / 'asd_analysis/proactive_lapse_no_led_utils.py']},
    backend=jax.default_backend(), device=jax.devices()[0].device_kind,
    jax=jax.__version__, numpyro=numpyro.__version__, series_terms=K_DEFAULT,
    smoke_test=args.smoke_test, window=WINDOW, min_steps=MIN_STEPS, max_steps=MAX_STEPS,
    checkpoint_format='atomic_state_best_and_history_v2')
if not args.benchmark_only:
    path = folder / 'runtime_configuration.json'
    if path.exists():
        assert json.loads(path.read_text()) == runtime, 'Incompatible resume configuration.'
    else:
        path.write_text(json.dumps(runtime, indent=2) + '\n')
guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values=initial), init_scale=INIT_SCALE)
svi = SVI(model, guide, numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM), Trace_ELBO())
state = svi.init(jax.random.PRNGKey(SEED), data, args.variant)


def checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(jax.device_get(payload), stream, pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


@jax.jit
def run_window(current):
    return jax.lax.scan(lambda s,_:svi.stable_update(s,data,args.variant), current, None, length=WINDOW)


# %% Independent timing run: no optimizer state is reused for the real fit
if args.benchmark_only:
    tick = perf_counter()
    benchmark_window = jax.jit(lambda s:jax.lax.scan(
        lambda x,_:svi.stable_update(x,data,args.variant), s, None, length=args.benchmark_steps))
    state, losses = benchmark_window(state)
    jax.block_until_ready(losses)
    compile_seconds = perf_counter() - tick
    tick = perf_counter()
    state, losses = benchmark_window(state)
    losses = np.asarray(losses)
    assert np.isfinite(losses).all()
    stats = jax.devices()[0].memory_stats() or {}
    report = dict(group=args.group, variant=args.variant, backend=jax.default_backend(),
        device=jax.devices()[0].device_kind, compile_seconds=compile_seconds,
        timed_steps=args.benchmark_steps, seconds_per_step=(perf_counter()-tick)/args.benchmark_steps,
        peak_bytes=int(stats.get('peak_bytes_in_use',stats.get('bytes_in_use',0))),
        timing_includes_scan_compilation=False, states_discarded=True, fresh_seed_for_real_fit=True)
    (prepared / 'gpu_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0)

# %% Resume complete windows; strict lower loss resets the confirmation count
step, best_step, best_loss, patience = 0, 0, np.inf, 0
losses, rows, history, sd_history = [], [], [], []
best_state = best_params = None
if (folder / 'latest_checkpoint.pkl').exists():
    latest = pickle.loads((folder / 'latest_checkpoint.pkl').read_bytes())
    state, step, patience = latest['state'], latest['step'], latest['patience']
    best = latest['best']
    best_state, best_params, best_loss, best_step = best['state'], best['params'], best['loss'], best['step']
    losses = latest['losses'].tolist()
    rows = latest['rows']
    history = [[row[n+'_mean'] for n in names] for row in rows]
    sd_history = [[row[n+'_sd'] for n in names] for row in rows]
    assert len(losses) == step == rows[-1]['step']
    # The atomic latest file is the commit record even if power failed between
    # writing a new best file and completing its window's other exports.
    checkpoint(folder / 'best_checkpoint.pkl', best)
    np.save(folder / 'losses.npy', np.asarray(losses))
    pd.DataFrame(rows).to_csv(folder / 'convergence.csv',index=False,float_format='%.17g')
started = perf_counter()
while step < MAX_STEPS and not (step >= MIN_STEPS and patience >= CONFIRMATION):
    tick = perf_counter()
    state, window = run_window(state)
    window = np.asarray(window)
    assert np.isfinite(window).all(), 'Nonfinite updates: retain last complete checkpoints.'
    step += WINDOW
    losses.extend(window.tolist())
    decision = loss_window_decision(float(window.mean()), step, best_loss, best_step, patience,
                                    MIN_STEPS, MAX_STEPS, CONFIRMATION)
    best_loss, best_step, patience = decision['best_loss'], decision['best_step'], decision['patience']
    params = svi.get_params(state)
    if decision['improved']:
        best_state, best_params = state, params
        checkpoint(folder / 'best_checkpoint.pkl', dict(state=state, params=params, step=step, loss=best_loss))
    diagnostic = guide.sample_posterior(jax.random.PRNGKey(99173), params, sample_shape=(DIAGNOSTIC_DRAWS,))
    matrix = sample_matrix(diagnostic)
    means, sds = matrix.mean(0), matrix.std(0,ddof=1)
    history.append(means.tolist()); sd_history.append(sds.tolist())
    movement = np.ptp(history[-3:],axis=0)/np.min(sd_history[-3:],axis=0) if len(history)>=3 else np.full(len(names),np.inf)
    row = dict(step=step, mean_negative_elbo=float(window.mean()), best_loss=best_loss, best_step=best_step,
        patience=patience, updated_best=decision['improved'], parameter_stable=bool(np.all(movement<.1)),
        max_mean_change_sd=float(movement.max()), window_seconds=perf_counter()-tick)
    for i,name in enumerate(names):
        row[name+'_mean'], row[name+'_sd'], row[name+'_change_sd'] = means[i], sds[i], movement[i]
    rows.append(row)
    checkpoint(folder / 'latest_checkpoint.pkl', dict(state=state,params=params,step=step,patience=patience,
        losses=np.asarray(losses),rows=rows,
        best=dict(state=best_state,params=best_params,step=best_step,loss=best_loss)))
    np.save(folder / 'losses.npy', np.asarray(losses))
    pd.DataFrame(rows).to_csv(folder / 'convergence.csv',index=False,float_format='%.17g')
    print(f'{args.group}/{args.variant}: {step:,}; best={best_step:,}; tail={patience}/{CONFIRMATION}; '
          f'window={row["window_seconds"]:.1f}s',flush=True)
    if args.checkpoint_only_after is not None and step >= args.checkpoint_only_after:
        raise SystemExit(0)

# %% Restore the absolute best; save physical posterior draws and summaries
samples = {k:np.asarray(v) for k,v in guide.sample_posterior(
    jax.random.PRNGKey(100000), best_params, sample_shape=(DRAWS,)).items()}
assert all(np.isfinite(v).all() for v in samples.values())
matrix = sample_matrix(samples)
assert matrix.shape == (DRAWS,len(names))
np.savez_compressed(folder / 'posterior_samples.npz', **samples)
pd.DataFrame(dict(parameter=names,mean=matrix.mean(0),sd=matrix.std(0,ddof=1),
    q025=np.quantile(matrix,.025,axis=0),median=np.median(matrix,axis=0),q975=np.quantile(matrix,.975,axis=0))).to_csv(
        folder / 'posterior_summary.csv',index=False,float_format='%.17g')
np.save(folder / 'posterior_covariance.npy',np.cov(matrix,rowvar=False))
checkpoint(folder / 'restored_guide_optimizer.pkl', dict(state=best_state,params=best_params,step=best_step))
physical = {k:jnp.asarray(v.mean(0)) for k,v in samples.items()}
expanded = expand_parameters(physical,args.variant)
best_row = next(row for row in rows if row['step']==best_step)
confirmed = step >= MIN_STEPS and patience >= CONFIRMATION
summary = dict(group=args.group,variant=args.variant,status='converged' if confirmed else 'inconclusive',
    smoke_test=args.smoke_test,final_checked_step=step,restored_best_step=best_step,
    best_window_negative_elbo=best_loss,confirmation_windows=patience,
    parameter_stability_required=False,terminal_parameter_stable=rows[-1]['parameter_stable'],
    restored_parameter_stable=best_row['parameter_stable'],posterior_draws=DRAWS,parameter_count=len(names),
    valid_trials=int(len(data['rt'])),initializations=1,
    posterior_mean_joint_log_likelihood=float(expanded_log_likelihood(expanded,data,'condition_delay')),
    posterior_mean_rt_log_likelihood=float(expanded_log_likelihood(expanded,data,'condition_delay',marginal=True)),
    elapsed_seconds_this_invocation=perf_counter()-started,
    stop_reason='confirmed_minimum_and_loss_patience' if confirmed else 'maximum_steps_without_confirmation')
(folder / 'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
