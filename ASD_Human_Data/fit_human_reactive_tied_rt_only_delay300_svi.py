# %%
"""Fit pooled 0--1.8 s human RTs with a true 0--300 ms delay prior and a causal SVI guide."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
from pathlib import Path
import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd
import jax
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
from human_reactive_tied_rt_only_support_utils import (
    ROOT, PARAM_NAMES, INITIAL, RT_LOWER, RT_UPPER, sha256, load_trials, load_all_trials,
    group_data, reactive_model, guide_support_model, log_likelihood,
)

# %% One warm start per group; new data, optimizer, RNG and loss history.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path, default=ROOT/'ASD_Human_Data/reactive_tied_rt_svi_rt_le_1p5s')
parser.add_argument('--output-dir', type=Path, default=ROOT/'ASD_Human_Data/reactive_tied_rt_only_svi_0_1p8s_delay300ms')
parser.add_argument('--group', choices=['all', 'Control', 'ASD'], default='all')
parser.add_argument('--rt-lower', type=float, default=RT_LOWER)
parser.add_argument('--rt-upper', type=float, default=RT_UPPER)
parser.add_argument('--delay-upper', type=float, default=.3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--min-steps', type=int, default=50000)
parser.add_argument('--max-steps', type=int, default=1000000)
parser.add_argument('--confirmation-windows', type=int, default=25)
parser.add_argument('--resume-existing', action='store_true')
args = parser.parse_args()
assert args.source_dir.resolve() != args.output_dir.resolve()
assert args.max_steps >= args.min_steps >= 3000
assert args.min_steps % 1000 == args.max_steps % 1000 == 0
args.output_dir.mkdir(parents=True, exist_ok=True)
source_config = json.loads((args.source_dir/'run_config.json').read_text())
for path, digest in source_config['source_hashes'].items():
    assert sha256(ROOT/path) == digest, path
frame, provenance = load_trials(args.rt_lower, args.rt_upper)
full, _ = load_all_trials()
sources = ['ASD_Human_Data/human_reactive_tied_rt_only_utils.py',
           'ASD_Human_Data/fit_human_reactive_tied_rt_only_delay300_svi.py',
           'ASD_Human_Data/human_reactive_tied_rt_only_support_utils.py',
           'ASD_Human_Data/human_reactive_tied_svi_utils.py',
           'fit_animal_by_animal/numpyro_vanilla_condition_delay_svi_utils.py']
config = dict(source_fit=str(args.source_dir.resolve()), seed=args.seed,
              min_steps=args.min_steps, max_steps=args.max_steps, checkpoint_steps=1000,
              confirmation_windows=args.confirmation_windows, minimum_relative_improvement=0.,
              stopping_rule='Any lower 1000-step mean resets patience; >=25 later windows and terminal parameter stability; selected-checkpoint movement is diagnostic',
              restored_checkpoint_stability_required=False,
              retention_window_s=[args.rt_lower, args.rt_upper], data=provenance, parameters=PARAM_NAMES,
              priors=dict(source_config['priors'], t_E_aff=f'Uniform(0, {args.delay_upper} s)'), initial_values=INITIAL,
              likelihood='RT only: sum both choice densities / either-bound retained mass',
              response_side_used_in_likelihood=False, delay_prior_upper_s=args.delay_upper,
              initialization='Earlier joint guide retaining the fastest RTs; map delay center/Jacobian to causal guide interval; reset Adam, RNG and history',
              source_guide_is_initialization=True, optimizer_state_preserved=False,
              statistical_initializations_per_group=1,
              learning_rate=.0002, gradient_clip=1., guide='AutoMultivariateNormal',
              guide_delay_support='(0, min(requested prior upper, minimum retained group RT)); actual model prior remains requested Uniform',
              guide_init_scale_note='Source covariance with a first-order delay-coordinate map to the new bound',
              elbo_particles=1, full_batch=True, dtype='float64', platform='cpu', K_max=40,
              posterior_draws=10000, source_hashes={p: sha256(ROOT/p) for p in sources})
if args.resume_existing and (args.output_dir/'run_config.json').exists():
    assert json.loads((args.output_dir/'run_config.json').read_text()) == config
else:
    (args.output_dir/'run_config.json').write_text(json.dumps(config, indent=2))
frame.to_csv(args.output_dir/'selected_trials.csv', index=False, float_format='%.17g')
excluded=full.loc[~full.RT_s.between(args.rt_lower,args.rt_upper)].copy()
excluded['exclusion_reason']=np.where(excluded.RT_s.lt(args.rt_lower),f'RT below {args.rt_lower:g} s',f'RT above {args.rt_upper:g} s')
excluded.to_csv(args.output_dir/'excluded_trials.csv', index=False, float_format='%.17g')
counts = full.groupby('Group').agg(original_trials=('Choice', 'size'), participants=('Participant', 'nunique'))
counts['selected_trials'] = frame.groupby('Group').size()
counts['removed_trials'] = counts.original_trials-counts.selected_trials
counts.to_csv(args.output_dir/'selection_counts.csv')
print('CPU', jax.devices(), 'output', args.output_dir, flush=True)
print(counts.to_string(), flush=True)

# %% Sequential CPU fits with strict post-best confirmation.
for group in (['Control', 'ASD'] if args.group == 'all' else [args.group]):
    folder = args.output_dir/group.lower()
    folder.mkdir(exist_ok=True)
    if (folder/'posterior_samples.npz').exists():
        if args.resume_existing:
            print(group, 'already completed', flush=True)
            continue
        raise FileExistsError(folder)
    sub, conditions, data, feasible_upper = group_data(frame, group)
    upper=args.delay_upper
    assert 0 <= args.rt_lower < args.rt_upper and upper > 0 and float(sub.RT_s.min()) > 0
    guide_upper=min(upper, float(sub.RT_s.min()))
    data.pop('choice')
    data.update(rt_lower=args.rt_lower, rt_upper=args.rt_upper, guide_delay_upper_s=guide_upper)
    sub.to_csv(folder/'selected_trials.csv', index=False, float_format='%.17g')
    conditions.to_csv(folder/'conditions.csv', index=False, float_format='%.17g')
    (folder/'data_manifest.json').write_text(json.dumps(dict(
        group=group, trials=len(sub), participants=sub.Participant.nunique(), delay_upper_s=upper,
        guide_delay_upper_s=guide_upper, minimum_retained_RT_s=float(sub.RT_s.min()),
        model_prior='Uniform(0, requested delay upper); guide interval is a likelihood-implied support restriction',
        retention_window_s=[args.rt_lower, args.rt_upper], selected_source_rows_sha256=sha256(folder/'selected_trials.csv'),
        source_provenance=provenance), indent=2))
    guide = AutoMultivariateNormal(guide_support_model, init_loc_fn=init_to_value(values=INITIAL), init_scale=.1)
    svi = SVI(reactive_model, guide, ClippedAdam(.0002, clip_norm=1.), Trace_ELBO(num_particles=1))
    warm_path = args.source_dir/group.lower()/'guide_parameters.npz'
    warm = {n: jax.device_put(v) for n, v in np.load(warm_path).items()}
    old_upper=json.loads((args.source_dir/group.lower()/'data_manifest.json').read_text())['delay_upper_s']
    old_center=float(old_upper*jax.nn.sigmoid(warm['auto_loc'][4]))
    assert 0 < old_center < guide_upper
    new_center=np.log(old_center/(guide_upper-old_center))
    jacobian=(1-old_center/old_upper)/(1-old_center/guide_upper)
    warm['auto_loc']=warm['auto_loc'].at[4].set(new_center)
    warm['auto_scale_tril']=warm['auto_scale_tril'].at[4,:].multiply(jacobian)
    state = svi.init(random.PRNGKey(args.seed), data, upper, init_params=warm)
    for name, value in svi.get_params(state).items():
        # NumPyro applies inverse/forward Cholesky transforms on initialization.
        np.testing.assert_allclose(value, warm[name], rtol=1e-14, atol=1e-15)
    assert int(state.optim_state[0]) == 0
    starting_step, losses, rows = 0, [], []
    best_loss, best_step, patience = np.inf, 0, 0
    if args.resume_existing and (folder/'latest_checkpoint.pkl').exists():
        with (folder/'latest_checkpoint.pkl').open('rb') as stream: current = pickle.load(stream)
        with (folder/'best_checkpoint.pkl').open('rb') as stream: best = pickle.load(stream)
        state = jax.tree.map(jax.device_put, current['state'])
        starting_step = int(current['step'])
        best_params, best_step = best['params'], int(best['step'])
        rows = pd.read_csv(folder/'convergence.csv').to_dict('records')
        losses = np.load(folder/'losses.npy').tolist()
        best_loss = float(min(r['mean_negative_elbo'] for r in rows))
        patience = int(rows[-1]['patience'])
        assert len(losses) == starting_step == int(state.optim_state[0])
    else:
        np.savez_compressed(folder/'initial_guide_parameters.npz', **{n: np.asarray(v) for n, v in warm.items()})
        (folder/'initialization.json').write_text(json.dumps(dict(
            source_guide=str(warm_path.resolve()), source_guide_sha256=sha256(warm_path),
            optimizer_iteration=0, fresh_loss_history=True, delay_upper_s=upper, seed=args.seed,
            source_delay_upper_s=old_upper, source_delay_center_s=old_center,
            mapped_delay_center_s=float(guide_upper*jax.nn.sigmoid(new_center)),
            guide_delay_upper_s=guide_upper, model_prior_upper_s=upper,
            delay_unconstrained_jacobian=jacobian,
            transfer='Identity for first four latent coordinates; center/Jacobian map for delay'), indent=2))
    mean_history = [[r[n+'_mean'] for n in PARAM_NAMES] for r in rows]
    sd_history = [[r[n+'_sd'] for n in PARAM_NAMES] for r in rows]
    diagnostics = jax.jit(lambda p: guide.sample_posterior(random.PRNGKey(1729), p, sample_shape=(4096,)))
    # Validate the actual warm-start physical center before launching updates.
    init_draws = diagnostics(svi.get_params(state))
    initial_mean = np.array([np.asarray(init_draws[n]).mean() for n in PARAM_NAMES])
    ll40 = log_likelihood(initial_mean, data, 40)
    ll80 = log_likelihood(initial_mean, data, 80)
    gradient = jax.grad(log_likelihood)(initial_mean, data)
    assert np.isfinite(ll40) and np.isfinite(gradient).all()
    np.testing.assert_allclose(ll40, ll80, rtol=1e-11, atol=1e-7)
    print(group, 'warm-start validation: LL', float(ll40), 'prior upper', upper, 'causal guide upper', guide_upper, flush=True)
    started = time.monotonic()
    prior_elapsed = rows[-1]['elapsed_s'] if rows else 0.
    stop_reason = 'maximum_steps_without_confirmation'
    for completed in range(starting_step+1000, args.max_steps+1, 1000):
        result = svi.run(random.PRNGKey(args.seed), 1000, data, upper, progress_bar=False,
                         init_state=state, stable_update=True)
        state = result.state
        chunk = np.asarray(result.losses)
        assert np.isfinite(chunk).all(), completed
        losses.extend(chunk.tolist())
        loss = float(chunk.mean())
        improved = loss < best_loss
        if improved:
            best_loss, best_step, best_params = loss, completed, result.params
            patience = 0
            with (folder/'best_checkpoint.pkl').open('wb') as stream:
                pickle.dump(jax.device_get(dict(state=state, params=result.params, step=completed)), stream)
        else:
            patience += 1
        draws = diagnostics(result.params)
        means = np.array([np.asarray(draws[n]).mean() for n in PARAM_NAMES])
        sds = np.array([np.asarray(draws[n]).std(ddof=1) for n in PARAM_NAMES])
        mean_history.append(means.tolist()); sd_history.append(sds.tolist())
        changes = (np.ptp(mean_history[-3:], axis=0)/np.min(sd_history[-3:], axis=0)
                   if len(mean_history) >= 3 else np.full(5, np.inf))
        stable = bool(np.all(changes < .1))
        row = dict(step=completed, mean_negative_elbo=loss, best_loss=best_loss, best_step=best_step,
                   patience=patience, updated_best=improved, parameter_stable=stable,
                   maximum_mean_change_sd=float(changes.max()), elapsed_s=prior_elapsed+time.monotonic()-started)
        for i, name in enumerate(PARAM_NAMES):
            row[name+'_mean'] = means[i]; row[name+'_sd'] = sds[i]; row[name+'_change_sd'] = changes[i]
        rows.append(row)
        pd.DataFrame(rows).to_csv(folder/'convergence.csv', index=False)
        np.save(folder/'losses.npy', np.asarray(losses))
        with (folder/'latest_checkpoint.pkl').open('wb') as stream:
            pickle.dump(jax.device_get(dict(state=state, params=result.params, step=completed)), stream)
        restored_stable = bool(rows[best_step//1000-1]['parameter_stable'])
        print(f'{group} {completed:7,d}: loss {loss:.4f}, best {best_step:,} ({best_loss:.4f}), '
              f'post-best {patience}/{args.confirmation_windows}, drift {changes.max():.3f} SD, '
              f'elapsed {time.monotonic()-started:.1f}s', flush=True)
        if completed >= args.min_steps and patience >= args.confirmation_windows and stable:
            stop_reason = 'confirmed_minimum_25_windows_and_terminal_parameter_stability'
            break
    samples = guide.sample_posterior(random.PRNGKey(args.seed+100000), best_params, sample_shape=(10000,))
    matrix = np.column_stack([np.asarray(samples[n]) for n in PARAM_NAMES])
    assert np.isfinite(matrix).all()
    np.savez_compressed(folder/'posterior_samples.npz', **{n: np.asarray(samples[n]) for n in PARAM_NAMES})
    np.savez_compressed(folder/'guide_parameters.npz', **{n: np.asarray(v) for n, v in best_params.items()})
    pd.DataFrame(dict(parameter=PARAM_NAMES, mean=matrix.mean(0), sd=matrix.std(0, ddof=1),
                      q025=np.quantile(matrix, .025, axis=0), median=np.median(matrix, axis=0),
                      q975=np.quantile(matrix, .975, axis=0))).to_csv(folder/'posterior_summary.csv', index=False, float_format='%.17g')
    pd.DataFrame(np.cov(matrix, rowvar=False), index=PARAM_NAMES, columns=PARAM_NAMES).to_csv(folder/'posterior_covariance.csv')
    pd.DataFrame(np.corrcoef(matrix, rowvar=False), index=PARAM_NAMES, columns=PARAM_NAMES).to_csv(folder/'posterior_correlation.csv')
    confirmed = bool(patience >= args.confirmation_windows)
    report = dict(stop_reason=stop_reason, completed_steps=completed, restored_best_step=best_step,
                  best_window_mean_negative_elbo=best_loss, terminal_parameter_stable=stable,
                  restored_checkpoint_parameter_stable=restored_stable, minimum_followup_confirmed=confirmed,
                  confirmation_windows=patience, required_confirmation_windows=args.confirmation_windows,
                  updates_after_restored_best=completed-best_step, flagged=not(confirmed and stable),
                  restored_checkpoint_stability_required=False,
                  wall_seconds=time.monotonic()-started, new_updates=completed,
                  posterior_kind='full-rank Gaussian variational approximation in unconstrained coordinates',
                  warning='Observed-window confirmation does not prove a global optimum; pooled-trial uncertainty excludes participant variability')
    (folder/'convergence_summary.json').write_text(json.dumps(report, indent=2))
    print(group, 'FINISHED', report, flush=True)
