# %%
"""Fit ASL-pooled human RT+choice with 16 parameters and loss-only restore-best stopping."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import argparse
import json
import pickle
import time
import platform
from functools import partial
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
from human_shared_gaussian_delay_svi_utils import (
    ROOT, OUTPUTS, BASELINE_ROOT, MODES, RT_WINDOW, sha256, load_data, prepare_group,
    get_priors, parameter_names, parameter_blocks, gamma_omega_model, guide_support_model,
    sample_matrix, vector_params, numerical_check, GRID_DT, QUAD_ORDER, loss_stop_status,
)

# %% Editable defaults and CLI; previous runs are never overwritten.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=MODES, default=MODES[0])
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--initial-root', type=Path, default=BASELINE_ROOT)
parser.add_argument('--group', choices=['all', 'Control', 'ASD'], default='all')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--min-steps', type=int, default=50000)
parser.add_argument('--max-steps', type=int, default=1000000)
parser.add_argument('--confirmation-windows', type=int, default=25)
parser.add_argument('--resume-existing', action='store_true')
parser.add_argument('--preflight-only', action='store_true')
args = parser.parse_args()
assert args.max_steps >= args.min_steps >= 3000
assert args.max_steps % 1000 == args.min_steps % 1000 == 0
assert args.confirmation_windows >= 1
mode = args.mode
PRIORS, PARAMETER_NAMES, PARAMETER_BLOCKS = get_priors(mode), parameter_names(mode), parameter_blocks(mode)
N_PARAMETERS = len(PARAMETER_NAMES)
model = partial(gamma_omega_model, mode=mode)
support_model = partial(guide_support_model, mode=mode)
out = args.output_dir or OUTPUTS[mode]
out.mkdir(parents=True, exist_ok=True)
sources = ['ASD_Human_Data/human_gaussian_delay_svi_utils_wide.py',
           'ASD_Human_Data/human_shared_gaussian_delay_svi_utils.py',
           'ASD_Human_Data/fit_human_shared_gaussian_delay_svi.py',
           'ASD_Human_Data/gaussian_delay_likelihood_wide.py',
           'ASD_Human_Data/human_gamma_omega_variants_svi_utils.py',
           'ASD_Human_Data/human_gamma_omega_delay500_svi_utils.py',
           'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py',
           'ASD_Human_Data/human_reactive_tied_svi_utils.py',
           'fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py']

# Freeze prior human artifacts before this analysis creates its own outputs.
preservation = out / 'preservation_baseline.json'
if not preservation.exists():
    frozen = {}
    for root in [BASELINE_ROOT]:
        for path in root.rglob('*'):
            if path.is_file() and not any(p.resolve() in path.resolve().parents for p in OUTPUTS.values()) and '__pycache__' not in path.parts:
                if 'gamma_omega_variants' in path.name or path.name == 'README.md':
                    continue
                frozen[str(path.relative_to(ROOT))] = sha256(path)
    preservation.write_text(json.dumps(frozen, indent=2))
full, selected, excluded, provenance = load_data(mode)
config = dict(model='ASLs pooled in fitting; Gamma per signed difference, one shared Gaussian mu/sigma and Omega/w/lapses per group', mode=mode,
              free_parameters_per_group=N_PARAMETERS, parameter_names=PARAMETER_NAMES, parameter_blocks=PARAMETER_BLOCKS,
              priors=PRIORS, rt_window_s=RT_WINDOW, group_order=['Control', 'ASD'],
              seed=args.seed, min_steps=args.min_steps, max_steps=args.max_steps,
              checkpoint_steps=1000, confirmation_windows=args.confirmation_windows,
              parameter_stability_gate=False,
              stopping='Any strictly lower 1000-step mean resets patience; require configured follow-up windows and minimum steps; no parameter-stability gate; restore best',
              guide='AutoMultivariateNormal', guide_init_scale=.1, learning_rate=.0002, gradient_clip=1.,
              elbo_particles=1, full_batch=True, dtype='float64', platform=jax.default_backend(), K_max=8, grid_dt_s=GRID_DT, quadrature_order=QUAD_ORDER,
              posterior_draws=10000, initial_root=str(args.initial_root.resolve()),
              initializations_per_group=1, initialization='Original 34-parameter initialization: copy Gamma/shared scalars; arithmetic means of original ten mu and sigma centers',
              likelihood='(1-p)*Gaussian-delay observed-choice density / retained mass + p*choice_prob/1.8',
              lapse_probability_semantics='fraction among retained trials',
              delay_guide='full mu and sigma priors; delay draws truncated at zero and unbounded above',
              source_hashes={p: sha256(ROOT/p) for p in sources}, data=provenance)
config_path = out / 'run_config.json'
if config_path.exists():
    previous_config = json.loads(config_path.read_text())
    current_config = json.loads(json.dumps(config))
    assert previous_config == current_config, 'Run configuration changed; use a separate output directory'
else:
    temporary_config = out/f'run_config.{os.getpid()}.json'
    temporary_config.write_text(json.dumps(config, indent=2))
    temporary_config.replace(config_path)
if not (out/'selected_trials.csv').exists():
    selected.to_csv(out/'selected_trials.csv', index=False, float_format='%.17g')
if not (out/'excluded_trials.csv').exists():
    excluded.to_csv(out/'excluded_trials.csv', index=False, float_format='%.17g')
counts = full.groupby('Group').agg(original_trials=('Choice', 'size'), participants=('Participant', 'nunique'))
counts['retained_trials'] = selected.groupby('Group').size()
counts['excluded_trials'] = counts.original_trials - counts.retained_trials
if not (out/'selection_counts.csv').exists():
    counts.to_csv(out/'selection_counts.csv')
(out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(), jax=jax.__version__,
    numpyro=numpyro.__version__, numpy=np.__version__, devices=[str(x) for x in jax.devices()],
    cpu_count=os.cpu_count(), backend=jax.default_backend(), device_kinds=[x.device_kind for x in jax.devices()],
    float64_enabled=bool(jax.config.x64_enabled)), indent=2))
print(counts.to_string(), flush=True)

# %% Group-local fitting; launch separate group processes for parallel execution.
for group in (['Control', 'ASD'] if args.group == 'all' else [args.group]):
    folder = out / group.lower()
    folder.mkdir(exist_ok=True)
    if (folder/'posterior_samples.npz').exists():
        assert args.resume_existing or args.preflight_only, 'Completed outputs exist; use --resume-existing'
        if not args.preflight_only:
            print(group, 'already completed', flush=True)
            continue
    sub, conditions, data, initial, initial_source = prepare_group(selected, group, mode, args.initial_root)
    sub.to_csv(folder/'selected_trials.csv', index=False, float_format='%.17g')
    conditions.to_csv(folder/'conditions.csv', index=False, float_format='%.17g')
    (folder/'data_manifest.json').write_text(json.dumps(dict(group=group, trials=len(sub),
        participants=int(sub.Participant.nunique()), conditions=10, free_parameters=N_PARAMETERS,
        selected_trials_sha256=sha256(folder/'selected_trials.csv'),
        condition_table_sha256=sha256(folder/'conditions.csv'),
        initialization_source=str(initial_source.relative_to(ROOT)), initialization_sha256=sha256(initial_source),
        condition_minimum_RT_s=conditions.minimum_RT_s.tolist(), cpu_affinity=sorted(os.sched_getaffinity(0)), gaussian_mu_bounds_s=PRIORS['delay_mu']['hard'], gaussian_sigma_bounds_s=PRIORS['delay_sigma']['hard']), indent=2))
    (folder/'initialization.json').write_text(json.dumps({n: np.asarray(v).tolist() for n,v in initial.items()}, indent=2))
    check = numerical_check(initial, data, mode)
    trace = handlers.trace(handlers.seed(handlers.substitute(model, data=initial), rng_seed=args.seed)).get_trace(data)
    n_parameters = sum(np.size(site['value']) for site in trace.values()
                       if site['type']=='sample' and not site.get('is_observed', False))
    assert n_parameters == N_PARAMETERS
    check['free_parameters'] = n_parameters
    guide = AutoMultivariateNormal(support_model, init_loc_fn=init_to_value(values=initial), init_scale=.1)
    svi = SVI(model, guide, ClippedAdam(.0002, clip_norm=1.), Trace_ELBO(num_particles=1))
    state = svi.init(random.PRNGKey(args.seed), data)
    assert np.isfinite(svi.evaluate(state, data))
    check['guide_latent_dimension'] = int(guide.latent_dim)
    assert guide.latent_dim == N_PARAMETERS
    (folder/'preflight.json').write_text(json.dumps(check, indent=2))
    print(group, 'preflight', check, flush=True)
    if args.preflight_only:
        continue
    (folder/'execution.json').write_text(json.dumps(dict(group=group, mode=mode,
        started_utc=datetime.now(timezone.utc).isoformat(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        argv=__import__('sys').argv, seed=args.seed, initialization_count=1,
        backend=jax.default_backend(), device_kinds=[x.device_kind for x in jax.devices()]),indent=2))
    losses, rows, best_loss, best_step, patience, start_step = [], [], np.inf, 0, 0, 0
    if (folder/'latest_checkpoint.pkl').exists():
        assert args.resume_existing, 'Checkpoint exists; use --resume-existing'
        with (folder/'latest_checkpoint.pkl').open('rb') as f: latest = pickle.load(f)
        with (folder/'best_checkpoint.pkl').open('rb') as f: best = pickle.load(f)
        state = jax.tree.map(jax.device_put, latest['state'])
        start_step = int(latest['step'])
        best_params, best_step = best['params'], int(best['step'])
        rows = pd.read_csv(folder/'convergence.csv').to_dict('records')
        losses = np.load(folder/'losses.npy').tolist()
        best_loss, patience = min(r['mean_negative_elbo'] for r in rows), int(rows[-1]['patience'])
        assert len(losses) == start_step == int(state.optim_state[0])
    else:
        np.savez_compressed(folder/'initial_guide_parameters.npz', **{n: np.asarray(v) for n,v in svi.get_params(state).items()})
    # Reuse the same compiled scan at each checkpoint. This is the stable_update
    # scan used by SVI.run(progress_bar=False), with unchanged optimizer/RNG flow.
    @jax.jit
    def update_window(window_state):
        def update_one(current_state, unused):
            return svi.stable_update(current_state, data)
        return jax.lax.scan(update_one, window_state, None, length=1000)
    started = time.monotonic()
    old_elapsed = rows[-1]['elapsed_s'] if rows else 0.
    stop_reason = 'maximum_steps_without_confirmation'
    completed = start_step
    stop, confirmed = loss_stop_status(completed,best_step,args.min_steps,args.max_steps,args.confirmation_windows)
    if confirmed: stop_reason = 'confirmed_loss_minimum_restore_best'
    for completed in (() if stop else range(start_step+1000, args.max_steps+1, 1000)):
        state, window_losses = update_window(state)
        params = svi.get_params(state)
        chunk = np.asarray(window_losses)
        assert np.isfinite(chunk).all(), f'{group}: nonfinite ELBO at {completed}'
        losses.extend(chunk.tolist())
        loss = float(chunk.mean())
        improved = loss < best_loss
        if improved:
            best_loss, best_step, best_params, patience = loss, completed, params, 0
            with (folder/'best_checkpoint.pkl').open('wb') as f:
                pickle.dump(jax.device_get(dict(state=state, params=best_params, step=completed)), f)
        else:
            patience += 1
        row = dict(step=completed, mean_negative_elbo=loss, best_loss=best_loss, best_step=best_step,
                   patience=patience, updated_best=improved, elapsed_s=old_elapsed+time.monotonic()-started)
        rows.append(row)
        pd.DataFrame(rows).to_csv(folder/'convergence.csv', index=False, float_format='%.17g')
        np.save(folder/'losses.npy', np.asarray(losses))
        with (folder/'latest_checkpoint.pkl').open('wb') as f:
            pickle.dump(jax.device_get(dict(state=state, params=params, step=completed)), f)
        print(f'{group} {completed:,}: loss={loss:.3f}, best={best_step:,}, follow-up={patience}/{args.confirmation_windows}, '
              f'elapsed={old_elapsed+time.monotonic()-started:.1f}s', flush=True)
        stop, confirmed = loss_stop_status(completed,best_step,args.min_steps,args.max_steps,args.confirmation_windows)
        if stop:
            if confirmed: stop_reason = 'confirmed_loss_minimum_restore_best'
            break
    samples = guide.sample_posterior(random.PRNGKey(args.seed+100000), best_params, sample_shape=(10000,))
    matrix = sample_matrix(samples, mode)
    assert matrix.shape == (10000,N_PARAMETERS) and np.isfinite(matrix).all()
    np.savez_compressed(folder/'posterior_samples.npz', **{n: np.asarray(samples[n]) for n in PARAMETER_BLOCKS})
    np.savez_compressed(folder/'guide_parameters.npz', **{n: np.asarray(v) for n,v in best_params.items()})
    summary = pd.DataFrame(dict(parameter=PARAMETER_NAMES, mean=matrix.mean(0), sd=matrix.std(0,ddof=1),
        q025=np.quantile(matrix,.025,axis=0), median=np.median(matrix,axis=0), q975=np.quantile(matrix,.975,axis=0)))
    summary.to_csv(folder/'posterior_summary.csv', index=False, float_format='%.17g')
    for name, values in [('covariance', np.cov(matrix,rowvar=False)), ('correlation', np.corrcoef(matrix,rowvar=False))]:
        pd.DataFrame(values,index=PARAMETER_NAMES,columns=PARAMETER_NAMES).to_csv(folder/f'posterior_{name}.csv',float_format='%.17g')
    numerical = numerical_check(vector_params(matrix.mean(0), mode), data, mode)
    report = dict(mode=mode, free_parameters=N_PARAMETERS, stop_reason=stop_reason, completed_steps=completed, restored_best_step=best_step,
        best_window_mean_negative_elbo=best_loss, parameter_stability_gate=False,
        confirmation_windows=patience, required_confirmation_windows=args.confirmation_windows,
        updates_after_restored_best=completed-best_step, flagged=not confirmed,
        wall_seconds=old_elapsed+time.monotonic()-started,
        numerical_check_at_posterior_means=numerical,
        posterior_kind='full-rank Gaussian variational approximation in unconstrained coordinates',
        warning='Confirmation does not prove a global optimum; pooled-trial posterior uncertainty excludes participant variability')
    (folder/'convergence_summary.json').write_text(json.dumps(report, indent=2))
    print(group, 'FINISHED', json.dumps(report), flush=True)
