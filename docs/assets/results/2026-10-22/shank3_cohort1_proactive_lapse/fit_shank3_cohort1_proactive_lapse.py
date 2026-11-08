# %% Imports and editable parameters
"""Pool SHANK3 cohort 1 trials by genotype and fit two full-rank SVI starts.

Run --prepare-only first, verify with --pre-fit, then run this script normally.
For concurrent runs use --genotype wt --seed 0 (etc.); --select-only combines
completed starts. ASDFIT_CPUS may restrict a worker to comma-separated CPU IDs.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from time import perf_counter

if os.environ.get('ASDFIT_CPUS'):
    os.sched_setaffinity(0, {int(x) for x in os.environ['ASDFIT_CPUS'].split(',')})
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
from proactive_lapse_no_led_utils import PARAM_NAMES, PRIOR_BOUNDS, model, log_likelihood

REPO = Path(__file__).resolve().parents[1]
INPUT = REPO/'asd_analysis/session_type_csvs/ASD_session_type_1.csv'
OUTPUT = REPO/'asd_analysis/shank3_cohort1_proactive_lapse'
GENOTYPES = ['wt', 'hom', 'het']
INITIALS = {0: [1.6, 2.5, .09, .05, 5.], 1: [3., 1., .04, .15, 2.]}
LEARNING_RATE, CLIP_NORM, INIT_SCALE = .0002, 1., .1
CHECK_EVERY, MIN_STEPS, INITIAL_MAX_STEPS, EXTENDED_MAX_STEPS = 1000, 50000, 150000, 250000
PATIENCE, REL_IMPROVEMENT, TIE_RELATIVE = 12, .001, .001
POSTERIOR_DRAWS = 10000
EXPECTED = {'wt': [5, 83921, 14205, 69716], 'hom': [6, 81721, 13580, 68141],
            'het': [3, 59603, 10439, 49164]}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepare-only', action='store_true')
parser.add_argument('--select-only', action='store_true')
parser.add_argument('--genotype', choices=GENOTYPES)
parser.add_argument('--seed', type=int, choices=[0, 1])
args = parser.parse_args()
OUTPUT.mkdir(exist_ok=True, parents=True)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_checkpoint(path, bundle):
    # Atomic replacement makes an interrupted run resumable at a complete window.
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        pickle.dump(jax.device_get(bundle), stream, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


# %% Prepare exact trial identities, clocks and outcome-authoritative classifications
if args.prepare_only or not (OUTPUT/'preparation.json').exists():
    input_sha = sha256(INPUT)
    cols = ['filename', 'source_file', 'source_date', 'animal', 'genotype', 'session', 'trial',
            'trial_start', 'session_type', 'training_level', 'ABL', 'ILD', 'abort_type', 'success',
            'fix_time', 'intended_fix_time', 'timed_rt', 'trial_is_repeat', 'is_short_sound']
    raw = pd.read_csv(INPUT, usecols=cols, float_precision='round_trip')
    assert raw.session_type.eq(1).all()
    raw['plot_ABL'] = raw.ABL.replace({59: 60})
    selected = raw.loc[raw.filename.eq('SHANK3_cohort1.csv') & raw.training_level.eq(16)
        & raw.plot_ABL.isin([20, 40, 60]) & raw.ILD.abs().isin([1, 2, 4, 8, 16])].copy()
    assert selected.trial_is_repeat.eq(False).all() and selected.is_short_sound.eq(0).all()
    assert selected.genotype.isin(GENOTYPES).all()
    selected['prepared_csv_line'] = selected.index+2
    selected['fixation_s'] = selected.fix_time/1000
    selected['censor_s'] = selected.intended_fix_time/1000
    assert np.isfinite(selected[['fixation_s', 'censor_s']]).all().all()
    assert selected.fixation_s.ge(0).all() and selected.censor_s.gt(0).all()
    valid = selected.success.isin([1, -1])
    fix = selected.abort_type.eq('Fixation')
    assert (valid | fix).all() and not (valid & fix).any()
    selected['is_event'] = fix & selected.fixation_s.lt(selected.censor_s)
    selected['classification'] = np.select([selected.is_event, valid],
        ['pre_onset_fixation_abort', 'valid_choice_censored'], default='late_fixation_abort_censored')
    selected['logged_fix_minus_onset_s'] = selected.fixation_s-selected.censor_s
    selected['timed_rt_minus_fix_difference_s'] = selected.timed_rt-selected.logged_fix_minus_onset_s
    selected.to_csv(OUTPUT/'trial_classifications.csv.gz', index=False, compression='gzip')
    selected.loc[(fix & ~selected.is_event) | (valid & selected.fixation_s.lt(selected.censor_s))].to_csv(
        OUTPUT/'timing_discrepancies.csv', index=False)
    membership = selected.groupby(['genotype', 'animal'], as_index=False).agg(
        trials=('is_event', 'size'), pre_onset_aborts=('is_event', 'sum'))
    membership['censored'] = membership.trials-membership.pre_onset_aborts
    membership.to_csv(OUTPUT/'animal_membership.csv', index=False)
    table = []
    for genotype in GENOTYPES:
        rows = selected.loc[selected.genotype.eq(genotype)]
        actual = [rows.animal.nunique(), len(rows), int(rows.is_event.sum()), int((~rows.is_event).sum())]
        assert actual == EXPECTED[genotype], (genotype, actual)
        event_times, event_counts = np.unique(rows.loc[rows.is_event, 'fixation_s'], return_counts=True)
        np.savez(OUTPUT/(genotype+'_likelihood_data.npz'), event_times=event_times,
            event_counts=event_counts.astype(float), censor_times=rows.loc[~rows.is_event, 'censor_s'].to_numpy(),
            all_censor_times=rows.censor_s.to_numpy())
        table.append(dict(zip(['genotype', 'animals', 'total_trials', 'pre_onset_aborts', 'censored_trials'], [genotype]+actual)))
    pd.DataFrame(table).to_csv(OUTPUT/'selection_counts.csv', index=False)
    sources = [INPUT, REPO/'SHANK3_cohort1.csv', REPO/'animal_structure_columns.txt',
               REPO/'asd_analysis/session_type_csvs/ASD_session_type_2.csv',
               REPO/'asd_analysis/session_type_csvs/ASD_session_type_23.csv']
    prep = {'source_sha256': input_sha, 'source_hashes': {str(p.relative_to(REPO)): sha256(p) for p in sources},
        'selected_trials': len(selected), 'animals': selected.animal.nunique(), 'counts': table,
        'late_fixation_censored': int((fix & ~selected.is_event).sum()),
        'valid_logged_before_onset_censored': int((valid & selected.fixation_s.lt(selected.censor_s)).sum()),
        'no_measured_rt_filter': True, 'repeat_filter_reapplied': False,
        'timing': 'fix_time/1000 and intended_fix_time/1000 seconds; outcome labels authoritative',
        'pooling': 'Actual trials across animals, sessions, ABL and signed ILD; no equal-animal weights',
        'session_type_definition': 'animal_structure_columns.txt:16; 1=standard RT, 2=short-duration, 23=short-duration biased blocks',
        'parameter_order': PARAM_NAMES, 'prior_bounds_lo_plateau_lo_plateau_hi_hi': PRIOR_BOUNDS}
    (OUTPUT/'preparation.json').write_text(json.dumps(prep, indent=2)+'\n')
    print(pd.DataFrame(table).to_string(index=False), flush=True)
if args.prepare_only:
    raise SystemExit(0)
prep = json.loads((OUTPUT/'preparation.json').read_text())
assert sha256(INPUT) == prep['source_sha256']
assert json.loads((OUTPUT/'prefit_validation.json').read_text())['status'] == 'passed'

# %% Full-rank SVI, exact full-data likelihood, saved/restored absolute-best window
if not args.select_only:
    for genotype in ([args.genotype] if args.genotype else GENOTYPES):
        source = np.load(OUTPUT/(genotype+'_likelihood_data.npz'))
        data = {k: jnp.asarray(source[k]) for k in ['event_times', 'event_counts', 'censor_times']}
        for seed in ([args.seed] if args.seed is not None else [0, 1]):
            run_dir = OUTPUT/genotype/('seed_'+str(seed))
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir/'run_summary.json').exists():
                print('Already completed:', run_dir, flush=True)
                continue
            config = {'genotype': genotype, 'seed': seed, 'init': INITIALS[seed], 'learning_rate': LEARNING_RATE,
                'clip_norm': CLIP_NORM, 'init_scale': INIT_SCALE, 'check_every': CHECK_EVERY,
                'min_steps': MIN_STEPS, 'initial_max_steps': INITIAL_MAX_STEPS, 'extended_max_steps': EXTENDED_MAX_STEPS,
                'patience': PATIENCE, 'relative_improvement': REL_IMPROVEMENT, 'posterior_draws': POSTERIOR_DRAWS,
                'source_sha256': prep['source_sha256'], 'jax_backend': jax.default_backend(),
                'jax_version': jax.__version__, 'numpyro_version': numpyro.__version__, 'float64': True,
                'parameter_order': PARAM_NAMES, 'prior_bounds': PRIOR_BOUNDS,
                'source_hashes': {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('proactive_lapse_no_led_utils.py')]}}
            (run_dir/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
            guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values=dict(zip(PARAM_NAMES, INITIALS[seed]))), init_scale=INIT_SCALE)
            svi = SVI(model, guide, numpyro.optim.ClippedAdam(LEARNING_RATE, clip_norm=CLIP_NORM), Trace_ELBO())
            state = svi.init(jax.random.PRNGKey(seed), data)
            best_state, best_mean, best_step, no_improve = None, np.inf, 0, 0
            losses, windows, step, budget = [], [], 0, INITIAL_MAX_STEPS
            checkpoint = run_dir/'checkpoint.pkl'
            if checkpoint.exists():
                saved = pickle.loads(checkpoint.read_bytes())
                state, best_state = saved['state'], saved['best_state']
                best_mean, best_step, no_improve = saved['best_mean'], saved['best_step'], saved['no_improve']
                losses, windows, step, budget = saved['losses'], saved['windows'], saved['step'], saved['budget']
                print(f'Resuming {genotype} seed {seed} at {step}', flush=True)

            # Cache a single compiled 1000-step scan; no minibatching or time binning.
            @jax.jit
            def run_window(current):
                return jax.lax.scan(lambda s, _: svi.stable_update(s, data), current, None, length=CHECK_EVERY)

            started = perf_counter()
            stop_reason = 'step_limit_inconclusive'
            while step < budget:
                tick = perf_counter()
                state, window = run_window(state)
                window = np.asarray(window)
                step += CHECK_EVERY
                losses.extend(window.tolist())
                finite = bool(np.isfinite(window).all())
                mean = float(window.mean())
                improved = finite and mean < best_mean
                significant = improved and (not np.isfinite(best_mean) or (best_mean-mean)/max(1, abs(best_mean)) >= REL_IMPROVEMENT)
                no_improve = 0 if significant else no_improve+1
                if improved:
                    best_state, best_mean, best_step = state, mean, step
                since_best = (step-best_step)//CHECK_EVERY
                can_stop = finite and step >= MIN_STEPS and no_improve >= PATIENCE and since_best >= PATIENCE
                windows.append({'end_step': step, 'mean_negative_elbo': mean, 'min_loss': float(window.min()),
                    'max_loss': float(window.max()), 'seconds': perf_counter()-tick, 'best_step': best_step,
                    'best_mean_negative_elbo': best_mean, 'updated_best': improved, 'significant_improvement': significant,
                    'no_improve_windows': no_improve, 'windows_since_best': since_best, 'can_stop': can_stop,
                    'nonfinite_losses': int((~np.isfinite(window)).sum())})
                if step >= budget and not can_stop and budget < EXTENDED_MAX_STEPS:
                    budget = EXTENDED_MAX_STEPS
                save_checkpoint(checkpoint, {'state': state, 'best_state': best_state, 'best_mean': best_mean,
                    'best_step': best_step, 'no_improve': no_improve, 'losses': losses, 'windows': windows, 'step': step, 'budget': budget})
                pd.DataFrame(windows).to_csv(run_dir/'convergence_windows.csv', index=False)
                print(f'{genotype} seed={seed} step={step} window={mean:.6f} best={best_step} tail={since_best} seconds={windows[-1]["seconds"]:.2f}', flush=True)
                if not finite:
                    stop_reason = 'nonfinite_loss_inconclusive'
                    break
                if can_stop:
                    stop_reason = 'patience12_restore_best'
                    break
            assert best_state is not None
            params = svi.get_params(best_state)
            samples = guide.sample_posterior(jax.random.PRNGKey(seed+1000), params, sample_shape=(POSTERIOR_DRAWS,))
            samples = {name: np.asarray(samples[name]) for name in PARAM_NAMES}
            assert all(np.isfinite(x).all() for x in samples.values())
            np.savez_compressed(run_dir/'posterior_samples.npz', **samples)
            estimates = pd.DataFrame([{'parameter': name, 'mean': x.mean(), 'sd': x.std(ddof=1),
                'median': np.median(x), 'q025': np.quantile(x, .025), 'q975': np.quantile(x, .975)} for name, x in samples.items()])
            estimates.to_csv(run_dir/'posterior_summary.csv', index=False)
            pd.DataFrame({'step': np.arange(1, len(losses)+1), 'negative_elbo': losses}).to_csv(run_dir/'losses.csv.gz', index=False, compression='gzip')
            save_checkpoint(run_dir/'restored_guide_and_optimizer.pkl', {'params': params, 'state': best_state, 'step': best_step})
            mean_params = jnp.array([samples[name].mean() for name in PARAM_NAMES])
            summary = {'genotype': genotype, 'seed': seed, 'status': 'converged' if stop_reason == 'patience12_restore_best' else 'inconclusive',
                'stop_reason': stop_reason, 'final_checked_step': step, 'restored_best_step': best_step,
                'restored_window_negative_elbo': best_mean, 'posterior_mean_log_likelihood': float(log_likelihood(mean_params, data)),
                'posterior_draws': POSTERIOR_DRAWS, 'wall_seconds_this_invocation': perf_counter()-started,
                'extended_after_150k': step > INITIAL_MAX_STEPS, 'source_sha256': prep['source_sha256']}
            (run_dir/'run_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(summary, flush=True)

# %% Reproducible start selection; a near-tie uses posterior-mean likelihood
if args.genotype is None or args.select_only:
    chosen, comparisons = [], []
    for genotype in GENOTYPES:
        runs = [json.loads((OUTPUT/genotype/f'seed_{seed}/run_summary.json').read_text()) for seed in [0, 1]]
        finite = [r for r in runs if np.isfinite(r['restored_window_negative_elbo']) and np.isfinite(r['posterior_mean_log_likelihood'])]
        assert finite
        best = min(finite, key=lambda r: (r['restored_window_negative_elbo'], r['seed']))
        tolerance = TIE_RELATIVE*max(1, abs(best['restored_window_negative_elbo']))
        tied = [r for r in finite if r['restored_window_negative_elbo']-best['restored_window_negative_elbo'] <= tolerance]
        winner = max(tied, key=lambda r: (r['posterior_mean_log_likelihood'], -r['seed']))
        chosen.append({**winner, 'selection_rule': 'within 0.1% window negative-ELBO tie: highest posterior-mean log likelihood' if len(tied)>1 else 'lowest restored window negative ELBO',
                       'tie_tolerance': tolerance})
        comparisons.extend([{**r, 'selected': r['seed'] == winner['seed'], 'in_tie_set': r in tied} for r in runs])
    pd.DataFrame(comparisons).to_csv(OUTPUT/'start_comparison.csv', index=False)
    (OUTPUT/'selected_starts.json').write_text(json.dumps(chosen, indent=2)+'\n')
    print(pd.DataFrame(chosen).to_string(index=False), flush=True)
