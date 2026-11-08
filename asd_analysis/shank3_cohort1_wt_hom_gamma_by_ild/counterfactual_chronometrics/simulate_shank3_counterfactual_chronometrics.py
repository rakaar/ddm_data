# %% Editable simulation settings; saved physical posterior means, no refitting
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import shutil
import time
import numpy as np
import pandas as pd
from shank3_gamma_by_ild_utils import ROOT,OUT,ABLS,ILDS,EXPECTED,sha256
from shank3_tied_omega_prediction_utils import VALID_GRID,ABS_ILDS,valid_predictions

FIT_ROOT = OUT
DEST = FIT_ROOT/'counterfactual_chronometrics'
GROUPS = ['wt','hom']
SCENARIOS = ['intact','delay_swap','omega_swap']
REPLICATES = 100
SEED = 0
COLORS = {'wt':'#4D4D4D','hom':'#C24A7A'}
DEST.mkdir(exist_ok=True)
print('Saved fit root:',FIT_ROOT,flush=True)
started = time.monotonic()

# %% Source hashes and exact parameter maps, keeping recipient timing and design
sources = [FIT_ROOT/'signed_condition_means.csv',FIT_ROOT/'chronometric_points.csv',
    ROOT/'asd_analysis/session_type_csvs/ASD_session_type_1.csv',
    Path(__file__),ROOT/'asd_analysis/shank3_gamma_by_ild_utils.py',
    ROOT/'asd_analysis/shank3_wt_gamma_omega_utils.py',
    ROOT/'asd_analysis/shank3_tied_omega_prediction_utils.py']
for group in GROUPS:
    sources += [FIT_ROOT/group/'fit_trials.csv.gz',FIT_ROOT/group/'conditions.csv',
        FIT_ROOT/group/'fixed_proactive.json',FIT_ROOT/group/'gamma_by_ild/posterior_samples.npz',
        FIT_ROOT/group/'gamma_by_ild/predictions/valid_cdfs_1ms.csv.gz']
sources += [p for ext in ['*.pdf','*.png'] for p in FIT_ROOT.rglob(ext) if DEST not in p.parents]
source_hashes = {str(p.resolve()):sha256(p) for p in sources}
means, fixed, populations, specifications = {}, {}, {}, {}
parameter_rows, condition_rows, point_rows, data_rows, animal_rows, membership_rows = [], [], [], [], [], []
for group in GROUPS:
    with np.load(FIT_ROOT/group/'gamma_by_ild/posterior_samples.npz') as draws:
        means[group] = {k:draws[k].mean(axis=0) for k in draws.files}
    fixed[group] = np.asarray(json.loads((FIT_ROOT/group/'fixed_proactive.json').read_text())['values'])
    valid = pd.read_csv(FIT_ROOT/group/'fit_trials.csv.gz',float_precision='round_trip')
    assert len(valid)==EXPECTED[group]['valid'] and valid.animal.nunique()==EXPECTED[group]['animals']
    assert valid.success.isin([1,-1]).all() and valid.timed_rt.between(0,1).all()
    assert valid.session_type.eq(1).all() and valid.training_level.eq(16).all()
    assert valid.ABL_mapped.isin(ABLS).all() and valid.ILD.isin(ILDS).all()
    populations[group] = valid
    for animal,sub in valid.groupby('animal'):
        membership_rows.append(dict(group=group,animal=animal,trials=len(sub)))
    for ild in ABS_ILDS:
        sub = valid.loc[valid.ILD.abs().eq(ild)]
        am = sub.groupby('animal').timed_rt.mean()
        sem = am.std(ddof=1)/np.sqrt(len(am))
        data_rows.append(dict(group=group,abs_ILD=ild,trials=len(sub),animals=len(am),
            data_mean_RT_s=sub.timed_rt.mean(),animal_SD_s=am.std(ddof=1),animal_SEM_s=sem))
        animal_rows.extend(dict(group=group,animal=a,abs_ILD=ild,mean_RT_s=t,
            trials=int(sub.animal.eq(a).sum())) for a,t in am.items())

for group in GROUPS:
    donor = 'hom' if group=='wt' else 'wt'
    specifications[group] = {}
    for scenario in SCENARIOS:
        params = {k:np.array(v,copy=True) for k,v in means[group].items()}
        swapped = 't_E_aff' if scenario=='delay_swap' else 'omega' if scenario=='omega_swap' else None
        if swapped:params[swapped] = np.array(means[donor][swapped],copy=True)
        specifications[group][scenario] = params
        for block,values in params.items():
            for i,value in enumerate(np.asarray(values).reshape(-1)):
                parameter_rows.append(dict(group=group,scenario=scenario,block=block,index=i,
                    ABL=ABLS[i] if block=='omega' else np.nan,
                    ILD=ILDS[i] if block=='gamma' else np.nan,value=float(value),
                    parameter_source=donor if block==swapped else group))
        for k in means[group]:
            np.testing.assert_array_equal(params[k],means[donor][k] if k==swapped else means[group][k])

config = dict(seed=SEED,replicates_per_condition=REPLICATES,groups=GROUPS,scenarios=SCENARIOS,
    stimulus_relative_RT_window_s=[0,1],source_hashes=source_hashes,
    sampling='Inverse of per-condition CDF, after normalizing every exact onset then averaging trials',
    common_random_numbers='Identical uniforms for intact/delay_swap/omega_swap within each group-condition',
    RNG='numpy default_rng(SeedSequence([seed, group_index, condition]))',
    fixed_proactive={g:fixed[g].tolist() for g in GROUPS},refitted=False)
(DEST/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
pd.DataFrame(parameter_rows).to_csv(DEST/'parameter_swaps.csv',index=False,float_format='%.17g')
pd.DataFrame(data_rows).to_csv(DEST/'observed_chronometrics.csv',index=False,float_format='%.17g')
pd.DataFrame(animal_rows).to_csv(DEST/'animal_chronometrics.csv',index=False,float_format='%.17g')
pd.DataFrame(membership_rows).to_csv(DEST/'memberships.csv',index=False)

# %% Six scenarios × 30 signed conditions; same uniforms, recipient-specific CDFs
for gi,group in enumerate(GROUPS):
    folder = DEST/group
    folder.mkdir(exist_ok=True)
    conditions = pd.read_csv(FIT_ROOT/group/'conditions.csv',float_precision='round_trip')
    for condition in conditions.itertuples():
        cid = condition.condition
        sub = populations[group].loc[populations[group].condition.eq(cid)]
        assert len(sub)==condition.trials
        assert sub.ABL_mapped.eq(condition.ABL).all() and sub.ILD.eq(condition.ILD).all()
        rng = np.random.default_rng(np.random.SeedSequence([SEED,gi,cid]))
        uniforms = rng.random(REPLICATES*len(sub))
        uniform_hash = hashlib.sha256(uniforms.tobytes()).hexdigest()
        predictions, simulations = {'rt_grid_s':VALID_GRID}, {}
        for scenario in SCENARIOS:
            params = specifications[group][scenario]
            g = float(params['gamma'][ILDS.index(condition.ILD)])
            o = float(params['omega'][ABLS.index(condition.ABL)])
            d,w,go = (float(params[k]) for k in ['t_E_aff','w','del_go'])
            result = valid_predictions(sub.onset_s.to_numpy(),fixed[group],g,o,d,w,go)
            cdf = result['cdf']
            assert np.diff(cdf).min()>=-1e-12
            np.testing.assert_allclose(cdf[[0,-1]],[0,1],atol=1e-12)
            # Remove only floating-point endpoint noise, after checks above.
            cdf = np.maximum.accumulate(np.clip(cdf,0,1)); cdf[0],cdf[-1] = 0.,1.
            simulated = np.interp(uniforms,cdf,VALID_GRID)
            assert np.isfinite(simulated).all() and simulated.min()>=0 and simulated.max()<=1
            predictions[scenario+'_cdf'] = cdf
            predictions[scenario+'_density'] = result['density']
            predictions[scenario+'_window_mass'] = result['window_mass']
            predictions['onset_s'] = result['onsets']
            predictions['onset_count'] = result['onset_counts']
            simulations[scenario+'_rt_s'] = simulated
            variance = simulated.var(ddof=1)
            mcsem = np.sqrt(variance/len(simulated))
            condition_rows.append(dict(group=group,scenario=scenario,condition=cid,
                ABL=condition.ABL,ILD=condition.ILD,abs_ILD=abs(condition.ILD),
                fitted_trials=len(sub),simulated_trials=len(simulated),gamma=g,omega=o,t_E_aff=d,w=w,del_go=go,
                exact_mean_RT_s=result['mean_rt_s'],CDF_mean_RT_s=result['cdf_mean_s'],
                simulated_mean_RT_s=simulated.mean(),simulated_variance_s2=variance,MC_SE_s=mcsem,
                simulation_minus_exact_s=simulated.mean()-result['mean_rt_s'],
                MC_z=(simulated.mean()-result['mean_rt_s'])/mcsem,
                minimum_window_mass=result['window_mass'].min(),maximum_window_mass=result['window_mass'].max(),
                integrated_mass=result['integrated_mass'],quadrature_difference=result['quad_order_difference'],
                uniforms_sha256=uniform_hash))
        np.savez_compressed(folder/f'conditional_distribution_{cid:02d}.npz',**predictions)
        np.savez_compressed(folder/f'simulated_condition_{cid:02d}.npz',**simulations)
        print(f'{group.upper()} signed condition {cid+1:02d}/30: all three scenarios saved; '
              f'{time.monotonic()-started:.1f} s elapsed',flush=True)
    pd.DataFrame(condition_rows).to_csv(DEST/'condition_means.csv',index=False,float_format='%.17g')

# %% Pool signed ILDs and ABLs by observed fitted-trial counts, with fixed allocation
condition_table = pd.DataFrame(condition_rows)
assert len(condition_table)==180
for (group,scenario,ild),sub in condition_table.groupby(['group','scenario','abs_ILD'],sort=False):
    n = sub.simulated_trials.sum()
    assert n==REPLICATES*sub.fitted_trials.sum()
    mean = np.average(sub.simulated_mean_RT_s,weights=sub.simulated_trials)
    exact = np.average(sub.exact_mean_RT_s,weights=sub.fitted_trials)
    # Signed-condition sample sizes are fixed, so Monte Carlo variance excludes
    # between-condition mean differences. This is not a posterior/animal SEM.
    mcsem = np.sqrt(np.sum(sub.simulated_trials*sub.simulated_variance_s2)/n**2)
    point_rows.append(dict(group=group,scenario=scenario,abs_ILD=ild,
        fitted_trials=int(sub.fitted_trials.sum()),simulated_trials=int(n),
        simulated_mean_RT_s=mean,exact_mean_RT_s=exact,
        CDF_mean_RT_s=np.average(sub.CDF_mean_RT_s,weights=sub.fitted_trials),MC_SE_s=mcsem,
        simulation_minus_exact_s=mean-exact,MC_z=(mean-exact)/mcsem))
points = pd.DataFrame(point_rows)
assert len(points)==30 and points.simulated_trials.sum()==40769700
assert condition_table.MC_z.abs().max()<6 and points.MC_z.abs().max()<6
baseline = pd.read_csv(FIT_ROOT/'chronometric_points.csv',float_precision='round_trip').query('view == "ABL_pooled"')
for row in points.loc[points.scenario.eq('intact')].itertuples():
    old = baseline.loc[baseline.group.eq(row.group)&baseline.abs_ILD.eq(row.abs_ILD)].iloc[0]
    np.testing.assert_allclose(row.exact_mean_RT_s,old.model_mean_RT_s,atol=1e-6,rtol=0)
points.to_csv(DEST/'pooled_chronometrics.csv',index=False,float_format='%.17g')
condition_table['pooled_fitted_weight'] = condition_table.fitted_trials/condition_table.groupby(
    ['group','scenario','abs_ILD']).fitted_trials.transform('sum')
condition_table.to_csv(DEST/'condition_means.csv',index=False,float_format='%.17g')
for name,expected in source_hashes.items():assert sha256(Path(name))==expected,name
for script in ['simulate_shank3_counterfactual_chronometrics.py',
               'plot_shank3_counterfactual_chronometrics.py','verify_shank3_counterfactual_chronometrics.py']:
    shutil.copy2(ROOT/'asd_analysis'/script,DEST/script)
report = dict(status='simulated_pending_independent_verification',created_utc=datetime.now(timezone.utc).isoformat(),
    scenarios=6,signed_condition_predictions=180,chronometric_points=30,
    synthetic_RTs=int(points.simulated_trials.sum()),maximum_condition_abs_MC_z=float(condition_table.MC_z.abs().max()),
    maximum_pooled_abs_MC_z=float(points.MC_z.abs().max()),source_hashes_unchanged=True,
    elapsed_seconds=time.monotonic()-started)
(DEST/'simulation_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
