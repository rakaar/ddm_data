# %% Imports and editable settings
"""One resumable full-rank HOM fit with frozen HOM abort parameters.

Prepare and independently verify first, then launch the condition-delay fit.
ASDFIT_CPUS sets comma-separated cores; use the CUDA platform for fitting.
"""
import argparse
import json
import os
import pickle
from pathlib import Path
from time import perf_counter
if os.environ.get('ASDFIT_CPUS'):
    os.sched_setaffinity(0,{int(x) for x in os.environ['ASDFIT_CPUS'].split(',')})
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
from shank3_wt_gamma_omega_utils import (ROOT,ABLS,ILDS,PRIORS,PARAM_BLOCKS,K_DEFAULT,sha256,
    fixed_at,model,log_likelihood,sample_matrix,parameter_names)

OUT = ROOT/'asd_analysis/shank3_cohort1_hom_gamma_omega'
INPUT = ROOT/'asd_analysis/session_type_csvs/ASD_session_type_1.csv'
ABORT_ROOT = ROOT/'asd_analysis/shank3_cohort1_proactive_lapse'
SEED, LEARNING_RATE, CLIP_NORM, INIT_SCALE = 0,.0002,1.,.1
WINDOW, MIN_STEPS, MAX_STEPS, PATIENCE = 1000,50000,1000000,25
POSTERIOR_DRAWS, DIAGNOSTIC_DRAWS, STABILITY_SD = 10000,2048,.1
REQUIRE_PARAMETER_STABILITY = False  # User removed this stopping requirement; keep the diagnostic.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepare-only',action='store_true')
parser.add_argument('--variant',choices=['condition_delay'],default='condition_delay')
parser.add_argument('--checkpoint-only-after',type=int,help='Save an unfinished checkpoint and exit at this step; resume without this flag.')
args = parser.parse_args()
OUT.mkdir(parents=True,exist_ok=True)


def checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(jax.device_get(payload),stream,pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


# %% Independent source selection and immutable fixed-parameter provenance
if args.prepare_only:
    assert not (OUT/'preparation.json').exists(), 'Preparation already exists; preserve it or use a new output root.'
    original_sha = sha256(INPUT)
    names = ['filename','animal','genotype','session','source_file','source_date','trial','trial_start',
        'session_type','training_level','trial_is_repeat','is_short_sound','ABL','ILD','success','response_poke',
        'timed_rt','fix_time','intended_fix_time','abort_type']
    full = pd.read_csv(INPUT,usecols=names,float_precision='round_trip')
    full['ABL_mapped'] = full.ABL.replace({59:60})
    parent = full.loc[full.filename.eq('SHANK3_cohort1.csv') & full.genotype.eq('hom') & full.training_level.eq(16)
        & full.session_type.eq(1) & full.ABL_mapped.isin(ABLS) & full.ILD.isin(ILDS)].copy()
    assert len(parent)==81721 and parent.animal.nunique()==6
    assert parent.trial_is_repeat.eq(False).all() and parent.is_short_sound.eq(0).all()
    parent['prepared_csv_line'] = parent.index+2
    parent['onset_s'] = parent.intended_fix_time/1000
    parent['abort_rt_s'] = (parent.fix_time-parent.intended_fix_time)/1000
    # Match the exact previous clock operations, including the subtraction order.
    parent['abort_rt_s'] = parent.fix_time/1000-parent.onset_s
    parent['observed_abort'] = parent.abort_type.eq('Fixation') & parent.abort_rt_s.lt(0)
    mapping = {(abl,ild):i for i,(abl,ild) in enumerate((a,d) for a in ABLS for d in ILDS)}
    parent['condition'] = [mapping[a,d] for a,d in zip(parent.ABL_mapped,parent.ILD)]
    train = parent.loc[parent.success.isin([1,-1]) & np.isfinite(parent.timed_rt) & parent.timed_rt.between(0,1)].copy()
    assert len(train)==67715 and train.response_poke.isin([-1,1]).all()
    assert train.response_poke.eq(1).sum()==35320 and train.response_poke.eq(-1).sum()==32395
    negative = parent.loc[parent.observed_abort & parent.abort_rt_s.ge(-1)].copy()
    assert len(negative)==13274
    parent['fit_valid'] = parent.index.isin(train.index)
    parent['display_abort'] = parent.index.isin(negative.index)
    parent.to_csv(OUT/'parent_trials.csv.gz',index=False,compression='gzip')
    train.to_csv(OUT/'fit_trials.csv.gz',index=False,compression='gzip')
    excluded = parent.loc[parent.success.isin([1,-1]) & ~parent.fit_valid]
    assert len(excluded)==357 and excluded.timed_rt.gt(1).all()
    excluded.to_csv(OUT/'excluded_valid_trials.csv',index=False)
    parent.groupby('animal',as_index=False).agg(parent_trials=('animal','size'),fit_trials=('fit_valid','sum'),
        display_aborts=('display_abort','sum')).to_csv(OUT/'membership.csv',index=False)
    selection = json.loads((ABORT_ROOT/'selected_starts.json').read_text())
    selected = next(row for row in selection if row['genotype']=='hom')
    assert selected['seed']==0 and selected['status']=='converged'
    summary = pd.read_csv(ABORT_ROOT/'hom/seed_0/posterior_summary.csv',float_precision='round_trip').set_index('parameter')
    fixed_names = ['V_A','theta_A','t_nd','lapse_prob','beta_lapse']
    fixed = summary.loc[fixed_names,'mean'].to_numpy()
    (OUT/'fixed_proactive.json').write_text(json.dumps({'parameter_order':fixed_names,'values':fixed.tolist(),
        'selected_seed':0,'source':'asd_analysis/shank3_cohort1_proactive_lapse/hom/seed_0/posterior_summary.csv',
        'uncertainty':'Fixed physical posterior means; no propagation of abort-fit uncertainty'},indent=2)+'\n')
    rt,c = train.timed_rt.to_numpy(),train.onset_s.to_numpy()
    cached = {k:np.asarray(v) for k,v in fixed_at(rt,c,jnp.array(fixed)).items()}
    at0,at1 = fixed_at(0.,c,jnp.array(fixed)),fixed_at(1.,c,jnp.array(fixed))
    data = dict(rt=rt,onset=c,choice=train.response_poke.to_numpy().astype(int),condition=train.condition.to_numpy().astype(int),
        lapse_prob=fixed[3],survival_onset=np.asarray(at0['proactive_survival']),survival_end=np.asarray(at1['proactive_survival']),
        lapse_window_mass=np.asarray(at0['lapse_survival']-at1['lapse_survival']),**cached)
    np.savez(OUT/'likelihood_data.npz',**data)
    conditions = []
    for (abl,ild),i in mapping.items():
        z = train.loc[train.condition.eq(i)]
        pr = float(z.response_poke.eq(1).mean())
        gamma = .5*np.log(pr/(1-pr))
        gb = PRIORS['gamma_positive' if ild>0 else 'gamma_negative']
        gamma = np.clip(gamma,gb[0]+1e-4,gb[-1]-1e-4)
        dimensionless_mean = np.tanh(gamma)/gamma if abs(gamma)>1e-7 else 1.
        omega = np.clip(dimensionless_mean/(z.timed_rt.mean()-.05),.0101,14.9999)
        conditions.append(dict(condition=i,ABL=abl,ILD=ild,parent_trials=int(parent.condition.eq(i).sum()),
            trials=len(z),right=int(z.response_poke.eq(1).sum()),left=int(z.response_poke.eq(-1).sum()),
            p_right=pr,mean_rt=z.timed_rt.mean(),minimum_rt=z.timed_rt.min(),gamma_init=gamma,omega_init=omega))
    pd.DataFrame(conditions).to_csv(OUT/'conditions.csv',index=False)
    preserved = {str(INPUT.relative_to(ROOT)):original_sha}
    for folder in [ABORT_ROOT,ROOT/'asd_analysis/shank3_cohort1_wt_gamma_omega',ROOT/'asd_analysis/shank3_cohort1_fixation_abort_durations']:
        for path in folder.rglob('*'):
            if path.is_file():
                preserved[str(path.relative_to(ROOT))] = sha256(path)
    for name in ['SHANK3_cohort1.csv','asd_analysis/session_type_csvs/ASD_session_type_2.csv','asd_analysis/session_type_csvs/ASD_session_type_23.csv']:
        preserved[name]=sha256(ROOT/name)
    for name in ['fit_shank3_wt_gamma_omega.py','shank3_wt_gamma_omega_utils.py','plot_shank3_wt_gamma_omega.py','verify_shank3_wt_gamma_omega.py','publish_shank3_wt_gamma_omega.py']:
        preserved['asd_analysis/'+name]=sha256(ROOT/'asd_analysis'/name)
    preparation = {'parent_trials':81721,'valid_trials':67715,'negative_display_aborts':13274,'display_trials':80989,
        'animals':6,'conditions':30,'right_choices':35320,'left_choices':32395,'excluded_valid_above_1':357,
        'preservation_hashes':preserved,'repeat_filter_reapplied':False,'fit_window_s':[0,1],
        'positive_clock':'recorded timed_rt; model absolute response = intended_fix_time/1000 + timed_rt',
        'negative_display_clock':'fix_time/1000 - intended_fix_time/1000 for strict pre-onset Fixation aborts',
        'condition_order':conditions,'priors':PRIORS,'fit_uncertainty':'conditional on frozen proactive/lapse means'}
    (OUT/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
    print('Prepared 67,715 valid choices, 30 conditions, 6 animals; 80,989 display rows.',flush=True)
    raise SystemExit(0)

# %% Model initialization, immutable configuration and resumable full-rank guide
assert args.variant is not None, 'Specify --variant shared_delay or condition_delay.'
assert json.loads((OUT/'prefit_validation.json').read_text())['status']=='passed'
variant = args.variant
assert jax.default_backend()=='gpu', 'This fit is configured for the GPU; set JAX_PLATFORMS=cuda.'
folder = OUT/variant
folder.mkdir(exist_ok=True)
assert not (folder/'run_summary.json').exists(), 'Completed fit is preserved; choose a new output root to rerun.'
prep = json.loads((OUT/'preparation.json').read_text())
assert sha256(INPUT)==prep['preservation_hashes'][str(INPUT.relative_to(ROOT))]
conditions = pd.read_csv(OUT/'conditions.csv',float_precision='round_trip')
data = {k:jnp.asarray(v) for k,v in np.load(OUT/'likelihood_data.npz').items() if k!='onset'}
initial = dict(gamma=conditions.gamma_init.to_numpy(),omega=conditions.omega_init.to_numpy(),
    t_E_aff=np.array(.05) if variant=='shared_delay' else np.full(30,.05),w=np.array(.5),del_go=np.array(.08))
names = parameter_names(variant)
assert len(names)==(63 if variant=='shared_delay' else 92)
config = {'variant':variant,'parameter_count':len(names),'seed':SEED,'learning_rate':LEARNING_RATE,'clip_norm':CLIP_NORM,
    'initial_scale':INIT_SCALE,'check_every':WINDOW,'min_steps':MIN_STEPS,'max_steps':MAX_STEPS,'patience_windows':PATIENCE,
    'stability_sd':STABILITY_SD,'require_parameter_stability':REQUIRE_PARAMETER_STABILITY,
    'diagnostic_draws':DIAGNOSTIC_DRAWS,'posterior_draws':POSTERIOR_DRAWS,'series_terms':K_DEFAULT,
    'parameter_order':names,'priors':PRIORS,'initial':{k:v.tolist() for k,v in initial.items()},
    'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),Path(__file__).with_name('shank3_wt_gamma_omega_utils.py'),
        Path(__file__).with_name('proactive_lapse_no_led_utils.py'),OUT/'likelihood_data.npz',OUT/'fixed_proactive.json']},
    'device_kind':jax.devices()[0].device_kind,'jax_version':jax.__version__,'numpyro_version':numpyro.__version__,'backend':jax.default_backend(),'float64':True}
if (folder/'configuration.json').exists():
    assert config==json.loads((folder/'configuration.json').read_text()), 'Configuration changed; do not resume incompatible state.'
else:
    (folder/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
guide = AutoMultivariateNormal(model,init_loc_fn=init_to_value(values=initial),init_scale=INIT_SCALE)
svi = SVI(model,guide,numpyro.optim.ClippedAdam(LEARNING_RATE,clip_norm=CLIP_NORM),Trace_ELBO())
state = svi.init(jax.random.PRNGKey(SEED),data,variant)
best_state,best_params,best_loss,best_step,patience = None,None,np.inf,0,0
losses,rows,history,sd_history,step = [],[],[],[],0
if (folder/'latest_checkpoint.pkl').exists():
    saved=pickle.loads((folder/'latest_checkpoint.pkl').read_bytes())
    state,step,patience=saved['state'],saved['step'],saved['patience']
    best=pickle.loads((folder/'best_checkpoint.pkl').read_bytes())
    best_state,best_params,best_loss,best_step=best['state'],best['params'],best['loss'],best['step']
    losses=np.load(folder/'losses.npy').tolist()
    rows=pd.read_csv(folder/'convergence.csv').to_dict('records')
    history=[[r[n+'_mean'] for n in names] for r in rows]
    sd_history=[[r[n+'_sd'] for n in names] for r in rows]
    assert len(losses)==step==rows[-1]['step']


@jax.jit
def run_window(current):
    return jax.lax.scan(lambda s,_:svi.stable_update(s,data,variant),current,None,length=WINDOW)


@jax.jit
def diagnostic_samples(params):
    return guide.sample_posterior(jax.random.PRNGKey(99173),params,sample_shape=(DIAGNOSTIC_DRAWS,))


# %% Exact full-batch updates; every lower window resets the 25-window tail
start = perf_counter()
stop_reason='maximum_steps_without_confirmation'
changes=np.ptp(history[-3:],axis=0)/np.min(sd_history[-3:],axis=0) if len(history)>=3 else np.full(len(names),np.inf)
stable=bool(np.all(changes<STABILITY_SD))
# A resumed run that already meets the authorized rule needs no more updates.
while step<MAX_STEPS and not (step>=MIN_STEPS and patience>=PATIENCE and (stable or not REQUIRE_PARAMETER_STABILITY)):
    tick=perf_counter()
    state,window=run_window(state)
    window=np.asarray(window)
    assert np.isfinite(window).all(), f'Nonfinite losses at step {step}; last complete checkpoint retained.'
    step+=WINDOW
    losses.extend(window.tolist())
    mean=float(window.mean())
    params=svi.get_params(state)
    improved=mean<best_loss
    if improved:
        best_state,best_params,best_loss,best_step,patience=state,params,mean,step,0
        checkpoint(folder/'best_checkpoint.pkl',dict(state=state,params=params,step=step,loss=mean))
    else:
        patience+=1
    matrix=sample_matrix(diagnostic_samples(params))
    means,sds=matrix.mean(0),matrix.std(0,ddof=1)
    history.append(means.tolist());sd_history.append(sds.tolist())
    changes=np.ptp(history[-3:],axis=0)/np.min(sd_history[-3:],axis=0) if len(history)>=3 else np.full(len(names),np.inf)
    stable=bool(np.all(changes<STABILITY_SD))
    row=dict(step=step,mean_negative_elbo=mean,best_loss=best_loss,best_step=best_step,patience=patience,
        updated_best=improved,parameter_stable=stable,max_mean_change_sd=float(changes.max()),window_seconds=perf_counter()-tick)
    for i,name in enumerate(names):
        row[name+'_mean'],row[name+'_sd'],row[name+'_change_sd']=means[i],sds[i],changes[i]
    rows.append(row)
    np.save(folder/'losses.npy',np.asarray(losses))
    pd.DataFrame(rows).to_csv(folder/'convergence.csv',index=False,float_format='%.17g')
    checkpoint(folder/'latest_checkpoint.pkl',dict(state=state,params=params,step=step,patience=patience))
    print(f'{variant} {step:,}: loss={mean:.5f} best={best_step:,} tail={patience}/25 change={changes.max():.3f}SD seconds={row["window_seconds"]:.1f}',flush=True)
    if args.checkpoint_only_after is not None and step>=args.checkpoint_only_after:
        print('Saved unfinished checkpoint for exact continuation.',flush=True)
        raise SystemExit(0)
    if step>=MIN_STEPS and patience>=PATIENCE and (stable or not REQUIRE_PARAMETER_STABILITY):
        break
if step>=MIN_STEPS and patience>=PATIENCE and (stable or not REQUIRE_PARAMETER_STABILITY):
    stop_reason='confirmed_minimum_and_terminal_parameter_stability' if REQUIRE_PARAMETER_STABILITY else 'confirmed_minimum_and_loss_patience'

# %% Restore the selected checkpoint and save the complete posterior
samples={k:np.asarray(v) for k,v in guide.sample_posterior(jax.random.PRNGKey(SEED+100000),best_params,sample_shape=(POSTERIOR_DRAWS,)).items()}
assert all(np.isfinite(x).all() for x in samples.values())
matrix=sample_matrix(samples)
np.savez_compressed(folder/'posterior_samples.npz',**samples)
pd.DataFrame(dict(parameter=names,mean=matrix.mean(0),sd=matrix.std(0,ddof=1),q025=np.quantile(matrix,.025,axis=0),
    median=np.median(matrix,axis=0),q975=np.quantile(matrix,.975,axis=0))).to_csv(folder/'posterior_summary.csv',index=False)
np.save(folder/'posterior_covariance.npy',np.cov(matrix,rowvar=False))
checkpoint(folder/'restored_guide_optimizer.pkl',dict(state=best_state,params=best_params,step=best_step))
mean_params={k:jnp.asarray(v.mean(axis=0)) for k,v in samples.items()}
summary={'variant':variant,'status':'converged' if stop_reason.startswith('confirmed') else 'inconclusive',
    'stop_reason':stop_reason,'final_checked_step':step,'restored_best_step':best_step,'best_window_negative_elbo':best_loss,
    'confirmation_windows':patience,'terminal_parameter_stable':stable,'max_terminal_mean_change_sd':float(changes.max()),
    'parameter_stability_required':REQUIRE_PARAMETER_STABILITY,'stopping_rule':'minimum 50000 updates and 25 windows since absolute-best mean loss',
    'parameter_count':len(names),'posterior_draws':POSTERIOR_DRAWS,'valid_trials':67715,'animals':6,
    'posterior_mean_joint_log_likelihood':float(log_likelihood(mean_params,data,variant)),
    'posterior_mean_rt_log_likelihood':float(log_likelihood(mean_params,data,variant,marginal=True)),
    'elapsed_seconds_this_invocation':perf_counter()-start,'initializations':1}
best_row=next(row for row in rows if row['step']==best_step)
summary['restored_checkpoint_parameter_stable']=bool(best_row['parameter_stable'])
summary['max_restored_mean_change_sd']=float(best_row['max_mean_change_sd'])
(folder/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
