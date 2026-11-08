# %% Independent selection, prior-count, SciPy likelihood and checkpoint checks
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import pickle
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import invgauss
from scipy.integrate import quad
import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
import numpyro
from shank3_gamma_by_ild_utils import (ROOT,OUT,ABLS,ILDS,PRIORS,PARAM_BLOCKS,VARIANTS,EXPECTED,
    sha256,expand_parameters,expanded_log_likelihood,model,initial_parameters,parameter_names,loss_window_decision)
from shank3_wt_gamma_omega_utils import choice_density, fixed_at, total_cdf, sample_matrix
sys.path.insert(0,str(ROOT/'ASD_Human_Data'))
from human_plain_ddm_diagnostics_utils import scipy_ddm

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pre-fit',action='store_true')
args=parser.parse_args()


def scipy_choice(rt,onset,side,gamma,omega,delay,w,go,fixed):
    v,a,nd,p,beta=fixed
    t=np.asarray(rt)+onset
    ap=invgauss.pdf(t-nd,mu=1/(v*a),scale=a*a)
    ass=invgauss.sf(t-nd,mu=1/(v*a),scale=a*a)
    pe=scipy_ddm(rt,gamma,omega,w,delay,side=side)
    fe=scipy_ddm(rt,gamma,omega,w,delay,cumulative=True,side=side)
    fg=scipy_ddm(np.asarray(rt)+go,gamma,omega,w,delay,cumulative=True,side=side)
    other=scipy_ddm(np.asarray(rt)+go,gamma,omega,w,delay,cumulative=True,side=-np.asarray(side))
    readout=np.maximum(.5*(1-fg-other)+fg-fe,0.)
    return (1-p)*(ap*readout+pe*ass)+.5*p*np.where(t>=0,beta*np.exp(-beta*np.maximum(t,0)),0.)


def scipy_window_mass(onset,gamma,omega,delay,w,fixed):
    v,a,nd,p,beta=fixed
    start=invgauss.sf(onset-nd,mu=1/(v*a),scale=a*a)
    end=invgauss.sf(onset+1-nd,mu=1/(v*a),scale=a*a)
    es=1-scipy_ddm(1.,gamma,omega,w,delay,cumulative=True)
    return (1-p)*(start-end*es)+p*(np.exp(-beta*onset)-np.exp(-beta*(onset+1)))


# %% Independently rebuild the input populations; original repeat decisions stand
raw=pd.read_csv(ROOT/'asd_analysis/session_type_csvs/ASD_session_type_1.csv',float_precision='round_trip',
    usecols=['filename','genotype','animal','session_type','training_level','ABL','ILD','success',
             'response_poke','timed_rt','fix_time','intended_fix_time'])
raw['mapped_ABL']=raw.ABL.replace({59:60})
checks=[];maximum_reference_error=0.;posteriors_checked=0
for group,expected in EXPECTED.items():
    mask=(raw.filename.eq('SHANK3_cohort1.csv')&raw.genotype.eq(group)&raw.session_type.eq(1)&raw.training_level.eq(16)
          &raw.mapped_ABL.isin(ABLS)&raw.ILD.isin(ILDS))
    parent=raw.loc[mask]
    valid=parent.loc[parent.success.isin([1,-1])&np.isfinite(parent.timed_rt)&parent.timed_rt.between(0,1)]
    folder=OUT/group
    saved=pd.read_csv(folder/'fit_trials.csv.gz',float_precision='round_trip')
    assert len(parent)==expected['parent'] and len(valid)==expected['valid'] and valid.animal.nunique()==expected['animals']
    assert int(valid.response_poke.eq(1).sum())==expected['right'] and int(valid.response_poke.eq(-1).sum())==expected['left']
    np.testing.assert_array_equal(saved.prepared_csv_line,valid.index+2)
    np.testing.assert_array_equal(saved.timed_rt,valid.timed_rt)
    np.testing.assert_array_equal(saved.response_poke,valid.response_poke)
    np.testing.assert_array_equal(saved.onset_s,valid.intended_fix_time/1000)
    fixed_json=json.loads((folder/'fixed_proactive.json').read_text())
    fixed=np.array(fixed_json['values'])
    abort_summary=pd.read_csv(ROOT/fixed_json['source'],float_precision='round_trip').set_index('parameter')
    np.testing.assert_array_equal(fixed,abort_summary.loc[fixed_json['parameter_order'],'mean'])
    assert fixed_json['selected_seed']==0
    conditions=pd.read_csv(folder/'conditions.csv',float_precision='round_trip')
    assert list(zip(conditions.ABL,conditions.ILD))==[(a,i) for a in ABLS for i in ILDS]
    for condition in conditions.itertuples():
        z=valid.loc[valid.mapped_ABL.eq(condition.ABL)&valid.ILD.eq(condition.ILD)]
        assert len(z)==condition.trials and int(z.response_poke.eq(1).sum())==condition.right
        np.testing.assert_allclose(condition.gamma_init,.5*np.log(condition.p_right/(1-condition.p_right)),atol=1e-12)
    full={k:jnp.asarray(v) for k,v in np.load(folder/'likelihood_data.npz').items() if k!='onset'}
    ids=np.concatenate([np.flatnonzero(saved.condition.to_numpy()==i)[np.linspace(0,int(conditions.iloc[i].trials)-1,10,dtype=int)]
                        for i in range(30)])
    data={k:(v[ids] if v.ndim else v) for k,v in full.items()}
    for variant in VARIANTS:
        run=folder/variant
        config=json.loads((run/'prepared_configuration.json').read_text())
        initial={k:jnp.asarray(v) for k,v in config['initial'].items()}
        names=parameter_names(variant)
        assert len(names)==config['parameter_count']==16
        rebuilt=initial_parameters(conditions,variant)
        for name in PARAM_BLOCKS:
            np.testing.assert_array_equal(initial[name],rebuilt[name])
        expanded=expand_parameters(initial,variant)
        om=np.asarray(expanded['omega'])
        for abl_index in range(3):
            np.testing.assert_array_equal(om[abl_index*10:(abl_index+1)*10],np.repeat(
                np.average(conditions.loc[conditions.ABL.eq(ABLS[abl_index])].omega_init,
                           weights=conditions.loc[conditions.ABL.eq(ABLS[abl_index])].trials)
                if variant=='gamma_by_ild' else np.average(conditions.omega_init,weights=conditions.trials),10))
        np.testing.assert_array_equal(expanded['t_E_aff'],np.full(30,.05))
        np.testing.assert_array_equal(expanded['gamma'], np.tile(initial['gamma'], 3))
        for i, ild in enumerate(ILDS):
            selected = valid.loc[valid.ILD.eq(ild)]
            probability = selected.response_poke.eq(1).mean()
            np.testing.assert_allclose(initial['gamma'][i], .5*np.log(probability/(1-probability)), atol=1e-12)
        from shank3_tied_omega_utils import expand_parameters as previous_expansion
        previous = previous_expansion({**initial, 'gamma': expanded['gamma']}, 'omega_by_abl')
        for name in PARAM_BLOCKS:
            np.testing.assert_array_equal(previous[name], expanded[name])
        np.testing.assert_array_equal(
            expanded_log_likelihood(previous, data, 'condition_delay'),
            expanded_log_likelihood(expanded, data, 'condition_delay'))

        trace=handlers.trace(handlers.seed(handlers.condition(model,data=initial),rng_seed=0)).get_trace(data,variant)
        assert sum(np.asarray(trace[n]['value']).size for n in PARAM_BLOCKS)==len(names)
        # Independent normalized trapezoid density, exactly once per free value.
        actual_prior=0.;reference_prior=0.
        for name in PARAM_BLOCKS:
            site=trace[name];factor=trace[name+'_trapezoid']
            actual_prior+=float(site['fn'].log_prob(site['value']).sum()+factor['fn'].log_prob(factor['value']).sum())
            values=np.asarray(initial[name]).reshape(-1)
            bounds=np.array([PRIORS['gamma_positive' if i>0 else 'gamma_negative'] for i in ILDS]) if name=='gamma' else np.tile(PRIORS[name],(len(values),1))
            lo,pl,ph,hi=bounds.T
            height=1/((pl-lo+hi-ph)/2+ph-pl)
            shape=np.minimum(np.minimum((values-lo)/(pl-lo),(hi-values)/(hi-ph)),1)
            reference_prior+=float(np.log(shape*height).sum())
        np.testing.assert_allclose(actual_prior,reference_prior,atol=1e-12)
        cond=np.asarray(data['condition'])
        g,o,d=[np.asarray(expanded[k])[cond] for k in ['gamma','omega','t_E_aff']]
        rt,onset,side=np.asarray(data['rt']),saved.iloc[ids].onset_s.to_numpy(),np.asarray(data['choice'])
        q=scipy_choice(rt,onset,side,g,o,d,.5,.08,fixed)
        mass=scipy_window_mass(onset,g,o,d,.5,fixed)
        reference=float(np.sum(np.log(q)-np.log(mass)))
        value,gradient=jax.value_and_grad(lambda p:expanded_log_likelihood(expand_parameters(p,variant),data,'condition_delay'))(initial)
        np.testing.assert_allclose(float(value),reference,atol=1e-8,rtol=1e-11)
        assert all(np.isfinite(v).all() for v in jax.tree.leaves(gradient))
        free_gradient = jax.grad(lambda p:expanded_log_likelihood(p, data, 'condition_delay'))(expanded)
        np.testing.assert_allclose(gradient['gamma'], np.asarray(free_gradient['gamma']).reshape(3,10).sum(0), atol=1e-9)
        np.testing.assert_allclose(gradient['omega'], np.asarray(free_gradient['omega']).reshape(3,10).sum(1), atol=1e-9)
        np.testing.assert_allclose(gradient['t_E_aff'], np.asarray(free_gradient['t_E_aff']).sum(), atol=1e-9)

        maximum_reference_error=max(maximum_reference_error,abs(float(value)-reference))
        checks.append(dict(group=group,variant=variant,free_parameters=len(names),
                           reference_trials=len(ids),likelihood_error=abs(float(value)-reference),prior_once_verified=True))
        if not args.pre_fit:
            samples=dict(np.load(run/'posterior_samples.npz'))
            matrix=sample_matrix(samples)
            assert matrix.shape==(10000,len(names)) and np.isfinite(matrix).all()
            posterior_mean={k:jnp.asarray(samples[k].mean(0)) for k in PARAM_BLOCKS}
            posterior_expanded=expand_parameters(posterior_mean,variant)
            pg,po,pdly=[np.asarray(posterior_expanded[k])[cond] for k in ['gamma','omega','t_E_aff']]
            pw,pgo=float(posterior_mean['w']),float(posterior_mean['del_go'])
            posterior_reference=float(np.sum(np.log(scipy_choice(rt,onset,side,pg,po,pdly,pw,pgo,fixed))
                -np.log(scipy_window_mass(onset,pg,po,pdly,pw,fixed))))
            posterior_value=float(expanded_log_likelihood(posterior_expanded,data,'condition_delay'))
            np.testing.assert_allclose(posterior_value,posterior_reference,atol=1e-8,rtol=1e-11)
            checks[-1]['posterior_mean_likelihood_error']=abs(posterior_value-posterior_reference)
            maximum_reference_error=max(maximum_reference_error,abs(posterior_value-posterior_reference))
            summary=pd.read_csv(run/'posterior_summary.csv',float_precision='round_trip')
            assert summary.parameter.tolist()==names
            np.testing.assert_allclose(summary[['mean','q025','q975']].to_numpy(),
                np.c_[matrix.mean(0),np.quantile(matrix,.025,axis=0),np.quantile(matrix,.975,axis=0)],atol=1e-12)
            losses=np.load(run/'losses.npy')
            convergence=pd.read_csv(run/'convergence.csv',float_precision='round_trip')
            np.testing.assert_allclose(losses.reshape(-1,1000).mean(1),convergence.mean_negative_elbo,atol=1e-10)
            record=json.loads((run/'run_summary.json').read_text())
            best_row=convergence.iloc[int(np.argmin(convergence.mean_negative_elbo))]
            assert record['restored_best_step']==int(best_row.step)
            decision=dict(best_loss=np.inf,best_step=0,patience=0)
            for row in convergence.itertuples():
                decision=loss_window_decision(row.mean_negative_elbo,int(row.step),decision['best_loss'],decision['best_step'],decision['patience'])
                assert decision['best_step']==row.best_step and decision['patience']==row.patience
            assert record['status']==decision['status'] and not record['parameter_stability_required']
            best=pickle.loads((run/'best_checkpoint.pkl').read_bytes())
            restored=pickle.loads((run/'restored_guide_optimizer.pkl').read_bytes())
            assert best['step']==restored['step']==record['restored_best_step']
            for x,y in zip(jax.tree.leaves(best['params']),jax.tree.leaves(restored['params'])):
                np.testing.assert_array_equal(x,y)
            guide=AutoMultivariateNormal(model,init_loc_fn=init_to_value(values=initial),init_scale=.1)
            svi=SVI(model,guide,numpyro.optim.ClippedAdam(.0002,clip_norm=1),Trace_ELBO())
            svi.init(jax.random.PRNGKey(0),data,variant)
            reproduced=guide.sample_posterior(jax.random.PRNGKey(100000),restored['params'],sample_shape=(10000,))
            for name in PARAM_BLOCKS:
                np.testing.assert_allclose(samples[name],reproduced[name],atol=1e-10,rtol=1e-10)
            posteriors_checked+=len(names)
    # Delay-boundary and full-mixture normalization checks use independent SciPy.
    for gamma,omega,delay in [(0.,4.,0.),(1.2,8.,.05),(-1.3,6.,.5)]:
        onset=.7;w=.5;go=.08
        integral=quad(lambda t:float(scipy_choice(t,onset,1,gamma,omega,delay,w,go,fixed)
                    +scipy_choice(t,onset,-1,gamma,omega,delay,w,go,fixed)),0,1,
                    points=sorted(set(t for t in [delay,delay-go] if 0<t<1)),epsabs=1e-9)[0]
        mass=float(scipy_window_mass(onset,gamma,omega,delay,w,fixed))
        np.testing.assert_allclose(integral,mass,atol=1e-9,rtol=1e-8)
        assert mass>0
    print(group.upper()+': selection, shared mappings, priors, likelihoods and finite gradients passed',flush=True)

# %% Stopping edge cases: ties, resets, minimum, resume and unconfirmed limit hit
decision=dict(best_loss=np.inf,best_step=0,patience=0)
for step in range(1000,51000,1000):
    decision=loss_window_decision(-100.,step,decision['best_loss'],decision['best_step'],decision['patience'])
    if step<50000: assert decision['status']=='running'
assert decision['status']=='converged' and decision['best_step']==1000
reset=loss_window_decision(-101.,51000,decision['best_loss'],decision['best_step'],decision['patience'])
assert reset['patience']==0 and reset['status']=='running'
tie=loss_window_decision(-101.,52000,reset['best_loss'],reset['best_step'],reset['patience'])
assert tie['patience']==1 and not tie['improved']
limit=loss_window_decision(-102.,1000000,tie['best_loss'],tie['best_step'],tie['patience'])
assert limit['status']=='inconclusive'
resumed=json.loads(json.dumps(reset))
for step in range(52000,77000,1000):
    resumed=loss_window_decision(-101.,step,resumed['best_loss'],resumed['best_step'],resumed['patience'])
assert resumed['status']=='converged' and resumed['patience']==25
try:
    loss_window_decision(np.nan,1000,np.inf,0,0)
except ValueError:
    pass
else:
    raise AssertionError('Nonfinite windows must fail.')
baseline=json.loads((OUT/'preservation_baseline.json').read_text())
for name,expected in baseline.items():
    assert sha256(ROOT/name)==expected,name
report=dict(status='passed',stage='pre_fit' if args.pre_fit else 'post_fit',populations=EXPECTED,
    runs=checks,posterior_parameters_checked=posteriors_checked,max_SciPy_log_likelihood_error=maximum_reference_error,
    exact_trial_identities_verified=True,frozen_parameters_verified=True,priors_applied_once=True,
    finite_gradients=True,delay_boundaries_and_joint_window_normalization=True,
    loss_only_stopping_minimum_reset_tie_resume_and_limit_tests=True,source_and_previous_results_unchanged=True,
    verifier_source_sha256=sha256(Path(__file__)))
name='prefit_validation.json' if args.pre_fit else 'postfit_validation.json'
(OUT/name).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
