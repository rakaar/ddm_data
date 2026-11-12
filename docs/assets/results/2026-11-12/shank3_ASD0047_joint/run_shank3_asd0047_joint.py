# %% ASD0047 joint fits, adaptive bounds, and authorized exponential simplification
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import fcntl
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from shank3_asd0047_joint_utils import ROOT, BASE, OLD, sha256
import shank3_individual_bounds_policy as boundary

# %% Editable workflow policy
PYTHON = str(ROOT / '.venv/bin/python')
MAX_ROUNDS = 10
EXPONENTIAL_DOMINANCE_MEAN = .95
boundary.ABSOLUTE_STRIPS.update(lapse_prob_right=.02, lapse_exp_weight=.02)
boundary.DOMAINS.update(lapse_prob_right=(0., 1.), lapse_exp_weight=(0., 1.))
cores = sorted(os.sched_getaffinity(0))
lock = (BASE / 'pipeline.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
status = dict(status='running', pid=os.getpid(), animal='ASD0047',
              started_utc=datetime.now(timezone.utc).isoformat())


def update(stage, **values):
    status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
    (BASE / 'completion.json').write_text(json.dumps(status, indent=2) + '\n')


def command(script, extra, variant, label, seed=0):
    selected_cores = cores[seed * 8:(seed + 1) * 8] or cores[:8]
    env = {**os.environ, 'ASD0047_JOINT_VARIANT': variant,
           'JAX_PLATFORMS': 'cuda' if script.startswith('fit_') else 'cpu',
           'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'XLA_PYTHON_CLIENT_MEM_FRACTION': '.35',
           'ASDFIT_CPUS': ','.join(map(str, selected_cores))}
    with (BASE / variant / f'{label}.log').open('a') as stream:
        subprocess.run([PYTHON, 'asd_analysis/' + script, *extra], cwd=ROOT,
                       env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def exponential_tendency(row):
    # A bounded, left-skewed posterior may peak at the exponential edge while
    # its mean is pulled below .95. This authorizes a separate nested fit only;
    # it never converts the original physical-bound flag into a passing audit.
    return bool(row['mean'] >= EXPONENTIAL_DOMINANCE_MEAN or
        (row['upper_hard_pressure'] and not row['lower_hard_pressure'] and
         row['histogram_mode'] >= .95 and row['median'] >= .90))


# %% Fit both starts within a common prior round; preserve every completed round
for variant in ['mixture', 'exponential_only']:
    out = BASE / variant
    if variant == 'exponential_only':
        decision = json.loads((BASE / 'contaminant_model_decision.json').read_text())
        if decision['next_model'] != 'exponential_only':
            break
        if not (out / 'preparation.json').exists():
            import shutil
            out.mkdir(exist_ok=True)
            mix = BASE / 'mixture'
            prep = json.loads((mix / 'preparation.json').read_text())
            for name in prep['input_hashes']:
                shutil.copy2(mix / name, out / name)
            prep.update(status='prepared_ASD0047_joint_pure_exponential', free_parameters=19,
                        fixed_parameters={'lapse_exp_weight': 1.})
            (out / 'preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
            policy = json.loads((mix / 'likelihood_policy.json').read_text())
            policy.update(contaminant_model='Pure truncated exponential; fixed d=1',
                          fixed_parameters={'lapse_exp_weight': 1.})
            (out / 'likelihood_policy.json').write_text(json.dumps(policy, indent=2) + '\n')
            selected = ROOT / json.loads((mix / 'selected_fit.json').read_text())['selected_run']
            priors = json.loads((selected / 'configuration.json').read_text())['priors']
            del priors['lapse_exp_weight']
            initials = {}
            for seed in [0, 1]:
                with np.load(selected.parent / f'seed_{seed}/posterior_samples.npz') as draws:
                    initials[str(seed)] = {key: draws[key].mean(0).tolist() for key in draws.files if key != 'lapse_exp_weight'}
            folder = out / 'round_00'
            folder.mkdir(exist_ok=True)
            (folder / 'bounds.json').write_text(json.dumps(priors, indent=2) + '\n')
            (folder / 'initialization.json').write_text(json.dumps(initials, indent=2) + '\n')
    count = 20 if variant == 'mixture' else 19
    accepted = None
    for iteration in range(MAX_ROUNDS):
        folder = out / f'round_{iteration:02d}'
        receipt = folder / 'bound_acceptance.json'
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous['status'] == 'refit_with_expanded_bounds':
                continue
            if (variant == 'mixture' and previous['status'] == 'intrinsic_domain_review_required'
                    and set(previous['intrinsic_flags']) == {'lapse_exp_weight'}):
                audited = pd.read_csv(folder / 'selected_boundary_audit.csv')
                d_row = audited.loc[audited.parameter.eq('lapse_exp_weight')].iloc[0].to_dict()
                if exponential_tendency(d_row):
                    original_receipt = folder / 'mixture_intrinsic_flag_before_nested_refit.json'
                    if not original_receipt.exists():
                        original_receipt.write_bytes(receipt.read_bytes())
                    previous.update(status='exponential_weight_requires_nested_refit',
                        nested_refit_reason='Upper-edge posterior mode >=.95 and median >=.90; original physical-domain flag retained.')
                    receipt.write_text(json.dumps(previous, indent=2) + '\n')
            if previous['status'] in ['passed', 'exponential_weight_requires_nested_refit']:
                accepted = previous
                break
            raise RuntimeError(f'Unresolved intrinsic domain: {receipt}')
        priors = json.loads((folder / 'bounds.json').read_text())
        update('preflight', variant=variant, bounds_round=iteration)
        if not (folder / 'prefit_validation.json').exists():
            command('verify_shank3_asd0047_joint.py', ['--round', str(iteration), '--pre-fit'],
                    variant, f'round{iteration:02d}_preflight')
        update('benchmark', variant=variant, bounds_round=iteration)
        if not (folder / 'gpu_benchmark_seed0.json').exists():
            command('fit_shank3_asd0047_joint.py', ['--round', str(iteration), '--benchmark-only'],
                    variant, f'round{iteration:02d}_benchmark')
        benchmark = json.loads((folder / 'gpu_benchmark_seed0.json').read_text())
        # peak_pool_bytes denotes allocator capacity, not measured resident use.
        measured = benchmark['memory_stats']['peak_bytes_in_use']
        workers = 2 if 2 * measured < .8 * benchmark['physical_memory_bytes'] else 1
        update('fitting', variant=variant, bounds_round=iteration, concurrent_starts=workers,
               seconds_per_update=benchmark['seconds_per_update'], measured_peak_bytes=measured)
        def fit(seed):
            if not (folder / f'seed_{seed}/run_summary.json').exists():
                command('fit_shank3_asd0047_joint.py', ['--round', str(iteration), '--seed', str(seed)],
                        variant, f'round{iteration:02d}_seed{seed}', seed)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(fit, [0, 1]))
        update('independent_verification', variant=variant, bounds_round=iteration)
        command('verify_shank3_asd0047_joint.py', ['--round', str(iteration)],
                variant, f'round{iteration:02d}_verification')
        runs = [json.loads((folder / f'seed_{seed}/run_summary.json').read_text()) for seed in [0, 1]]
        assert all(run['status'] == 'converged' for run in runs), 'Unconfirmed fits cannot be accepted.'
        best = min(runs, key=lambda run: run['best_window_negative_elbo'])
        tolerance = .001 * max(1., abs(best['best_window_negative_elbo']))
        tied = [run for run in runs if run['best_window_negative_elbo'] - best['best_window_negative_elbo'] <= tolerance]
        chosen = max(tied, key=lambda run: (run['posterior_mean_joint_log_likelihood'], -run['seed']))
        all_rows = []
        for seed in [0, 1]:
            with np.load(folder / f'seed_{seed}/posterior_samples.npz') as samples:
                for key in samples.files:
                    vectors = samples[key].reshape(10000, -1)
                    for index in range(vectors.shape[1]):
                        parameter = key + f'_ABL{[20,40,60][index]}' if key.startswith('delay_') else key
                        all_rows.append(dict(seed=seed, prior_key=key, parameter=parameter,
                            **boundary.audit_column(vectors[:, index], priors[key], key)))
        pd.DataFrame(all_rows).to_csv(folder / 'all_start_boundary_audit.csv', index=False, float_format='%.17g')
        rows = [row for row in all_rows if row['seed'] == chosen['seed']]
        pd.DataFrame(rows).to_csv(folder / 'selected_boundary_audit.csv', index=False, float_format='%.17g')
        assert len(rows) == count
        artificial = [row for row in rows if row['lower_artificial_pressure'] or row['upper_artificial_pressure']]
        taper = [row for row in rows if row['lower_taper_pressure'] or row['upper_taper_pressure']]
        intrinsic = [row for row in rows if (row['lower_hard_pressure'] and row['lower_intrinsic_boundary']) or
                     (row['upper_hard_pressure'] and row['upper_intrinsic_boundary'])]
        accepted = dict(status='passed', variant=variant, parameter_count=count, bounds_round=iteration,
            selected_seed=chosen['seed'], selected_run=str((folder / f'seed_{chosen["seed"]}').relative_to(ROOT)),
            artificial_flags=[r['parameter'] for r in artificial], taper_flags=[r['parameter'] for r in taper],
            intrinsic_flags=[r['parameter'] for r in intrinsic], starts=runs)
        if artificial or taper:
            revised, changes = boundary.expand_priors(priors, rows)
            assert changes
            accepted.update(status='refit_with_expanded_bounds', changes=changes)
            next_round = out / f'round_{iteration+1:02d}'
            next_round.mkdir(exist_ok=True)
            (next_round / 'bounds.json').write_text(json.dumps(revised, indent=2) + '\n')
            (next_round / 'initialization.json').write_bytes((folder / 'initialization.json').read_bytes())
        elif intrinsic:
            if variant == 'mixture' and set(accepted['intrinsic_flags']) == {'lapse_exp_weight'}:
                d_row = next(row for row in rows if row['parameter'] == 'lapse_exp_weight')
                accepted['status'] = ('exponential_weight_requires_nested_refit' if exponential_tendency(d_row)
                                      else 'intrinsic_domain_review_required')
            else:
                accepted['status'] = 'intrinsic_domain_review_required'
        receipt.write_text(json.dumps(accepted, indent=2) + '\n')
        if accepted['status'] in ['passed', 'exponential_weight_requires_nested_refit']:
            break
        if accepted['status'] == 'intrinsic_domain_review_required':
            update('intrinsic_domain_review_required', acceptance=accepted)
            raise RuntimeError('Cannot remove a physical-domain flag by widening beyond the model domain.')
    else:
        raise RuntimeError('Adaptive prior rounds exhausted without acceptance.')
    (out / 'selected_fit.json').write_text(json.dumps(accepted, indent=2) + '\n')
    if variant == 'mixture':
        selected = ROOT / accepted['selected_run']
        with np.load(selected / 'posterior_samples.npz') as samples:
            d = samples['lapse_exp_weight']
        audited = pd.read_csv(selected.parent / 'selected_boundary_audit.csv')
        d_row = audited.loc[audited.parameter.eq('lapse_exp_weight')].iloc[0].to_dict()
        next_model = 'exponential_only' if exponential_tendency(d_row) else 'mixture'
        decision = dict(status='recorded', selected_mixture_run=accepted['selected_run'],
            d_mean=float(d.mean()), d_q025=float(np.quantile(d, .025)), d_q975=float(np.quantile(d, .975)),
            exponential_dominance_mean_threshold=EXPONENTIAL_DOMINANCE_MEAN,
            d_median=float(np.median(d)), d_histogram_mode=float(d_row['histogram_mode']),
            original_d_upper_hard_pressure=bool(d_row['upper_hard_pressure']),
            alternative_tendency_rule='Upper-bound pressure, no lower-bound pressure, histogram mode >=0.95 and median >=0.90',
            rule_refinement='Mode/median rule added after inspecting the first completed start, before either candidate was accepted; not a predeclared model-selection test.',
            next_model=next_model, user_authorized_nested_refit=True,
            note='Dominant exponential weight prompts the authorized d=1 refit; this is not a Bayes-factor test.')
        (BASE / 'contaminant_model_decision.json').write_text(json.dumps(decision, indent=2) + '\n')
        if next_model == 'mixture':
            break

# %% No output marked complete until figures and independent report checks exist
(BASE / 'selected_fit.json').write_text(json.dumps(accepted, indent=2) + '\n')
update('accepted_ready_for_reports', status='fit_accepted_reports_pending', selected_run=accepted['selected_run'],
       final_variant=accepted['variant'])
print(json.dumps(status, indent=2), flush=True)
