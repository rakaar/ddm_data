# %%
"""Fit pooled human Gamma/Omega with trial-varying Gaussian delays and uniform lapses."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
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
from human_gaussian_delay_svi_utils import (
    ROOT, OUTPUTS, BASELINE_ROOT, MODES, RT_WINDOW, sha256, load_data, prepare_group,
    get_priors, parameter_names, parameter_blocks, gamma_omega_model, guide_support_model,
    sample_matrix, vector_params, numerical_check, GRID_DT, QUAD_ORDER,
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
sources = ['ASD_Human_Data/human_gaussian_delay_svi_utils.py',
           'ASD_Human_Data/fit_human_gaussian_delay_svi.py',
           'ASD_Human_Data/gaussian_delay_likelihood.py',
           'ASD_Human_Data/human_gamma_omega_variants_svi_utils.py',
           'ASD_Human_Data/human_gamma_omega_delay500_svi_utils.py',
           'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py',
           'ASD_Human_Data/human_reactive_tied_svi_utils.py',
           'fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py']

# Freeze prior human artifacts before this analysis creates its own outputs.
preservation = out / 'preservation_baseline.json'
if not preservation.exists():
    frozen = {}
    for root in [ROOT/'ASD_Human_Data', ROOT/'docs/assets/results/2026-10-20', ROOT/'docs/assets/results/2026-10-21']:
        for path in root.rglob('*'):
            if path.is_file() and not any(p.resolve() in path.resolve().parents for p in OUTPUTS.values()) and '__pycache__' not in path.parts:
                if 'gamma_omega_variants' in path.name or path.name == 'README.md':
                    continue
                frozen[str(path.relative_to(ROOT))] = sha256(path)
    preservation.write_text(json.dumps(frozen, indent=2))
full, selected, excluded, provenance = load_data(mode)
config = dict(model='reactive-only condition-wise Gamma/Omega/Gaussian mu/sigma, shared w and uniform lapse', mode=mode,
              free_parameters_per_group=N_PARAMETERS, parameter_names=PARAMETER_NAMES, parameter_blocks=PARAMETER_BLOCKS,
              priors=PRIORS, rt_window_s=RT_WINDOW, group_order=['Control', 'ASD'],
              seed=args.seed, min_steps=args.min_steps, max_steps=args.max_steps,
              checkpoint_steps=1000, confirmation_windows=args.confirmation_windows,
              stability='last three posterior-mean ranges / minimum SD across those checkpoints < 0.1 for all free parameters',
              stopping='Any strictly lower 1000-step mean resets patience; require follow-up windows AND terminal parameter stability; restore best',
              guide='AutoMultivariateNormal', guide_init_scale=.1, learning_rate=.0002, gradient_clip=1.,
              elbo_particles=1, full_batch=True, dtype='float64', platform='cpu', K_max=8, grid_dt_s=GRID_DT, quadrature_order=QUAD_ORDER,
              posterior_draws=10000, initial_root=str(args.initial_root.resolve()),
              initializations_per_group=1, initialization='Original saved Gamma/Omega/w; mu from original delay; sigma=.05 s; p=.02 and q=.5',
              likelihood='(1-p)*Gaussian-delay observed-choice density / retained mass + p*choice_prob/1.8',
              lapse_probability_semantics='fraction among retained trials',
              delay_guide='full mu and sigma priors; delay draws truncated at zero and unbounded above',
              source_hashes={p: sha256(ROOT/p) for p in sources}, data=provenance)
config_path = out / 'run_config.json'
if config_path.exists():
    assert json.loads(config_path.read_text()) == json.loads(json.dumps(config)), 'Run configuration changed; use a separate output directory'
else:
    config_path.write_text(json.dumps(config, indent=2))
if not (out/'selected_trials.csv').exists():
    selected.to_csv(out/'selected_trials.csv', index=False, float_format='%.17g')
if not (out/'excluded_trials.csv').exists():
    excluded.to_csv(out/'excluded_trials.csv', index=False, float_format='%.17g')
counts = full.groupby('Group').agg(original_trials=('Choice', 'size'), participants=('Participant', 'nunique'))
counts['retained_trials'] = selected.groupby('Group').size()
counts['excluded_trials'] = counts.original_trials - counts.retained_trials
counts.to_csv(out/'selection_counts.csv')
(out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(), jax=jax.__version__,
    numpyro=numpyro.__version__, numpy=np.__version__, devices=[str(x) for x in jax.devices()],
    cpu_count=os.cpu_count()), indent=2))
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
        participants=int(sub.Participant.nunique()), conditions=20, free_parameters=N_PARAMETERS,
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
        argv=__import__('sys').argv, seed=args.seed, initialization_count=1),indent=2))
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
    diagnostics = jax.jit(lambda p: guide.sample_posterior(random.PRNGKey(1729), p, sample_shape=(4096,)))
    history = [[r[n+'_mean'] for n in PARAMETER_NAMES] for r in rows]
    sd_history = [[r[n+'_sd'] for n in PARAMETER_NAMES] for r in rows]
    started = time.monotonic()
    old_elapsed = rows[-1]['elapsed_s'] if rows else 0.
    stop_reason = 'maximum_steps_without_confirmation'
    for completed in range(start_step+1000, args.max_steps+1, 1000):
        result = svi.run(random.PRNGKey(args.seed), 1000, data, init_state=state,
                         progress_bar=False, stable_update=True)
        state = result.state
        chunk = np.asarray(result.losses)
        assert np.isfinite(chunk).all(), f'{group}: nonfinite ELBO at {completed}'
        losses.extend(chunk.tolist())
        loss = float(chunk.mean())
        improved = loss < best_loss
        if improved:
            best_loss, best_step, best_params, patience = loss, completed, result.params, 0
            with (folder/'best_checkpoint.pkl').open('wb') as f:
                pickle.dump(jax.device_get(dict(state=state, params=best_params, step=completed)), f)
        else:
            patience += 1
        matrix = sample_matrix(diagnostics(result.params), mode)
        means, sds = matrix.mean(0), matrix.std(0, ddof=1)
        history.append(means.tolist()); sd_history.append(sds.tolist())
        changes = (np.ptp(history[-3:], axis=0)/np.min(sd_history[-3:], axis=0)
                   if len(history)>=3 else np.full(N_PARAMETERS, np.inf))
        stable = bool(np.all(changes < .1))
        row = dict(step=completed, mean_negative_elbo=loss, best_loss=best_loss, best_step=best_step,
                   patience=patience, updated_best=improved, parameter_stable=stable,
                   maximum_mean_change_sd=float(changes.max()), elapsed_s=old_elapsed+time.monotonic()-started)
        for i, name in enumerate(PARAMETER_NAMES):
            row[name+'_mean'], row[name+'_sd'], row[name+'_change_sd'] = means[i], sds[i], changes[i]
        rows.append(row)
        pd.DataFrame(rows).to_csv(folder/'convergence.csv', index=False, float_format='%.17g')
        np.save(folder/'losses.npy', np.asarray(losses))
        with (folder/'latest_checkpoint.pkl').open('wb') as f:
            pickle.dump(jax.device_get(dict(state=state, params=result.params, step=completed)), f)
        print(f'{group} {completed:,}: loss={loss:.3f}, best={best_step:,}, follow-up={patience}/{args.confirmation_windows}, '
              f'max parameter change={changes.max():.3f} SD, elapsed={old_elapsed+time.monotonic()-started:.1f}s', flush=True)
        if completed >= args.min_steps and patience >= args.confirmation_windows and stable:
            stop_reason = 'confirmed_minimum_and_terminal_parameter_stability'
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
        best_window_mean_negative_elbo=best_loss, terminal_parameter_stable=stable,
        restored_checkpoint_parameter_stable=bool(rows[best_step//1000-1]['parameter_stable']),
        confirmation_windows=patience, required_confirmation_windows=args.confirmation_windows,
        updates_after_restored_best=completed-best_step, flagged=not(patience>=args.confirmation_windows and stable),
        maximum_terminal_mean_change_sd=float(changes.max()), wall_seconds=old_elapsed+time.monotonic()-started,
        numerical_check_at_posterior_means=numerical,
        posterior_kind='full-rank Gaussian variational approximation in unconstrained coordinates',
        warning='Confirmation does not prove a global optimum; pooled-trial posterior uncertainty excludes participant variability')
    (folder/'convergence_summary.json').write_text(json.dumps(report, indent=2))
    print(group, 'FINISHED', json.dumps(report), flush=True)
