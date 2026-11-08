# %% Independent raw-data, SciPy likelihood and posterior checks
"""--pre-fit checks data and numerical kernels before either expensive SVI run."""
import argparse
import json
import pickle
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.stats import invgauss
from scipy.integrate import quad, trapezoid
import jax
import jax.numpy as jnp
from shank3_wt_gamma_omega_utils import (ROOT,ABLS,ILDS,PRIORS,sha256,evidence_bound,fixed_at,
    choice_density,total_cdf,log_likelihood,parameter_names,sample_matrix,model)
sys.path.insert(0,str(ROOT/'ASD_Human_Data'))
from human_plain_ddm_diagnostics_utils import scipy_ddm,ddm_log_density,ddm_log_cdf
VERIFICATION_SOURCES={str(path.relative_to(ROOT)):sha256(path) for path in [
    Path(__file__),ROOT/'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py',
    ROOT/'fitting_aborts/numpyro_proactive_led_step_jump_npl_alpha_utils.py']}

OUT=ROOT/'asd_analysis/shank3_cohort1_hom_gamma_omega'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pre-fit',action='store_true')
parser.add_argument('--preview-initial',action='store_true',help='Independently verify the isolated initial-prediction dry run.')
args=parser.parse_args()
ARTIFACTS=OUT/'initial_preview' if args.preview_initial else OUT
prep=json.loads((OUT/'preparation.json').read_text())
fixed=np.array(json.loads((OUT/'fixed_proactive.json').read_text())['values'])
conditions=pd.read_csv(OUT/'conditions.csv',float_precision='round_trip')


def scipy_race(rt, onset, gamma, omega, delay, w, go, side, marginal=False):
    v,a,nd,p,beta=fixed
    evidence_rt=np.asarray(rt)
    rt,onset=np.broadcast_arrays(rt,onset)
    t=rt+onset
    age=t-nd
    apdf=invgauss.pdf(age,mu=1/(v*a),scale=a*a)
    asf=invgauss.sf(age,mu=1/(v*a),scale=a*a)
    pe=scipy_ddm(evidence_rt,gamma,omega,w,delay,side=side)
    fe=scipy_ddm(evidence_rt,gamma,omega,w,delay,cumulative=True,side=side)
    fe_go=scipy_ddm(evidence_rt+go,gamma,omega,w,delay,cumulative=True,side=side)
    fe_other_go=scipy_ddm(evidence_rt+go,gamma,omega,w,delay,cumulative=True,side=-np.asarray(side))
    ordinary=apdf*(.5*(1-fe_go-fe_other_go)+fe_go-fe)+pe*asf
    density=(1-p)*ordinary+.5*p*np.where(t>=0,beta*np.exp(-beta*t),0.)
    if marginal:
        density+=scipy_race(rt,onset,gamma,omega,delay,w,go,-np.asarray(side))
    return density


def scipy_mass(rt, onset, gamma, omega, delay, w):
    v,a,nd,p,beta=fixed
    t=np.asarray(rt)+onset
    asf=invgauss.sf(t-nd,mu=1/(v*a),scale=a*a)
    es=scipy_ddm(np.asarray(rt),gamma,omega,w,delay,cumulative=True)
    return 1-(1-p)*asf*(1-es)-p*np.exp(-beta*np.maximum(t,0))


# %% Reconstruct identities from current CSV, independent of the fit selector
raw=pd.read_csv(ROOT/'asd_analysis/session_type_csvs/ASD_session_type_1.csv',float_precision='round_trip',
    usecols=['filename','genotype','animal','session_type','training_level','ABL','ILD','success','response_poke','timed_rt','fix_time','intended_fix_time'])
mask=(raw.filename=='SHANK3_cohort1.csv')&(raw.genotype=='hom')&(raw.session_type==1)&(raw.training_level==16)
mask &= np.isin(raw.ABL,[20,40,59,60]) & np.isin(raw.ILD,ILDS)
parent=raw.loc[mask].copy()
v=parent.loc[np.isin(parent.success,[-1,1]) & (parent.timed_rt>=0) & (parent.timed_rt<=1)]
saved=pd.read_csv(OUT/'fit_trials.csv.gz',float_precision='round_trip')
np.testing.assert_array_equal(v.index+2,saved.prepared_csv_line)
np.testing.assert_array_equal(v.timed_rt,saved.timed_rt)
np.testing.assert_array_equal(v.response_poke,saved.response_poke)
np.testing.assert_array_equal(v.intended_fix_time/1000,saved.onset_s)
assert len(v)==67715 and len(parent)==81721 and v.animal.nunique()==6
assert set(v.animal)=={'ASD0023','ASD0024','ASD0043','ASD0045','ASD0046','ASD0048'}
assert v.response_poke.value_counts().to_dict()=={1.:35320,-1.:32395}
assert len(parent.loc[parent.success.isin([-1,1])&parent.timed_rt.gt(1)])==357
all_saved=pd.read_csv(OUT/'parent_trials.csv.gz',float_precision='round_trip')
neg=(parent.fix_time/1000-parent.intended_fix_time/1000)
observed=(parent.loc[:,'fix_time']<parent.loc[:,'intended_fix_time'])&~parent.success.isin([-1,1])
negative_lines=parent.index[observed & (neg>=-1)]+2
np.testing.assert_array_equal(negative_lines,all_saved.loc[all_saved.display_abort,'prepared_csv_line'])
assert len(negative_lines)==13274
abort_summary=pd.read_csv(ROOT/'asd_analysis/shank3_cohort1_proactive_lapse/hom/seed_0/posterior_summary.csv',float_precision='round_trip')
np.testing.assert_array_equal(fixed,abort_summary['mean'])
data={k:jnp.asarray(x) for k,x in np.load(OUT/'likelihood_data.npz').items() if k!='onset'}
initial={'gamma':jnp.array(conditions.gamma_init),'omega':jnp.array(conditions.omega_init),
    't_E_aff':jnp.array(.05),'w':jnp.array(.5),'del_go':jnp.array(.08)}
for row in conditions.itertuples():
    z=v.loc[v.ABL.replace({59:60}).eq(row.ABL)&v.ILD.eq(row.ILD)]
    assert len(z)==row.trials and z.response_poke.eq(1).sum()==row.right
    np.testing.assert_array_equal(saved.loc[saved.condition==row.condition,'prepared_csv_line'],z.index+2)

# %% Evidence kernels: zero drift, sign/bias extremes, short/long seam and onset
rng=np.random.default_rng(31)
age=np.r_[[-.1,0,1e-10,.01,.6/5,1.,4.],rng.uniform(.0001,1,100)]
kernel_error=0.
for gamma,w in [(0,.5),(.2,.4),(-1.8,.6),(19.,.02),(-19.,.98),(1e-8,.3)]:
    for side in [-1,1]:
        pdf,cdf,rem=[np.asarray(x) for x in evidence_bound(age,gamma,5.,w,side)]
        reference_pdf=scipy_ddm(age,gamma,5.,w,0.,side=side)
        reference_cdf=scipy_ddm(age,gamma,5.,w,0.,cumulative=True,side=side)
        np.testing.assert_allclose(pdf,reference_pdf,atol=2e-11,rtol=2e-9)
        np.testing.assert_allclose(cdf,reference_cdf,atol=2e-11,rtol=2e-9)
        for k in [40,80]:
            other=[np.asarray(x) for x in evidence_bound(age,gamma,5.,w,side,k)]
            for x,y in zip([pdf,cdf,rem],other):
                np.testing.assert_allclose(x,y,atol=2e-12,rtol=2e-10)
        kernel_error=max(kernel_error,float(np.max(np.abs(cdf-reference_cdf))))
        assert (pdf>=0).all() and (cdf>=0).all() and (rem>=0).all()

# %% Independent complete race values, integrals, finite gradients and delay limits
checks=[]
for gamma,omega,delay,w,go,onset in [(0,4,.05,.5,.08,.6),(1.2,10,.1,.42,.15,1.),(-2,12,.2,.58,0.,.4),
                                 (.4,.5,0.,.5,.2,1.1),(-.8,8,.5,.5,.2,.7)]:
    grid=np.unique(np.r_[np.linspace(-.3,1,170),delay,np.nextafter(delay,0.),delay-go])
    cached=fixed_at(jnp.array(grid),onset,jnp.array(fixed))
    total=np.zeros(len(grid))
    for side in [-1,1]:
        q=np.asarray(choice_density(jnp.array(grid),side,gamma,omega,delay,w,go,cached,fixed[3]))
        qref=scipy_race(grid,onset,gamma,omega,delay,w,go,side)
        np.testing.assert_allclose(q,qref,atol=1e-10,rtol=2e-9)
        total+=q
    mass=float(total_cdf(1.,onset,gamma,omega,delay,w,jnp.array(fixed))-total_cdf(0.,onset,gamma,omega,delay,w,jnp.array(fixed)))
    integrated=quad(lambda t:float(scipy_race(t,onset,gamma,omega,delay,w,go,1,marginal=True)),0,1,
        points=sorted(set(x for x in [delay,delay-go] if 0<x<1)),epsabs=1e-9)[0]
    np.testing.assert_allclose(integrated,mass,atol=1e-8)
    assert mass>0 and mass<1 and np.all(total>=0)
    checks.append({'gamma':gamma,'omega':omega,'delay':delay,'w':w,'go':go,'onset':onset,'window_mass':mass,'integrated_density':integrated})

# Full data are evaluated once per representation. This is also a runtime preflight.
tick=perf_counter()
value,gradient=jax.jit(jax.value_and_grad(log_likelihood),static_argnames=['variant'])(initial,data,'shared_delay')
assert np.isfinite(value) and all(np.isfinite(np.asarray(x)).all() for x in gradient.values())
expanded={**initial,'t_E_aff':jnp.full(30,.05)}
value2,gradient2=jax.jit(jax.value_and_grad(log_likelihood),static_argnames=['variant'])(expanded,data,'condition_delay')
np.testing.assert_allclose(value,value2,atol=1e-8)
np.testing.assert_allclose(gradient['t_E_aff'],gradient2['t_E_aff'].sum(),atol=1e-7)
full_ref=0.
for row in conditions.itertuples():
    z=saved.loc[saved.condition==row.condition]
    q=scipy_race(z.timed_rt.to_numpy(),z.onset_s.to_numpy(),row.gamma_init,row.omega_init,.05,.5,.08,z.response_poke.to_numpy())
    norm=scipy_mass(1.,z.onset_s.to_numpy(),row.gamma_init,row.omega_init,.05,.5)-scipy_mass(0.,z.onset_s.to_numpy(),row.gamma_init,row.omega_init,.05,.5)
    full_ref+=np.sum(np.log(q)-np.log(norm))
np.testing.assert_allclose(value,full_ref,atol=1e-6,rtol=1e-11)
report={'status':'passed','fit_trials':67715,'parent_trials':81721,'display_trials':80989,'animals':6,'conditions':30,
    'kernel_max_cdf_error':kernel_error,'race_integral_checks':checks,'initial_log_likelihood':float(value),
    'scipy_initial_log_likelihood':full_ref,'finite_gradients':True,'shared_broadcast_equivalence':True,
    'numerical_preflight_seconds':perf_counter()-tick,'parameter_counts':{'condition_delay':92},
    'verification_source_hashes':VERIFICATION_SOURCES}

# %% Verify posterior restoration and published numerical predictions
if not args.pre_fit:
    import numpyro
    from numpyro.infer import SVI,Trace_ELBO,init_to_value
    from numpyro.infer.autoguide import AutoMultivariateNormal
    summaries=[]
    plot_checks=[]
    report['posterior_reproduction']={}
    plot_manifest=json.loads((ARTIFACTS/'plot_manifest.json').read_text())
    assert plot_manifest['status'] in (['initial_preview'] if args.preview_initial else
        ['generated_pending_independent_verification','independently_verified'])
    import subprocess
    from PIL import Image
    for artifact,digest in plot_manifest['artifacts'].items():
        assert sha256(ARTIFACTS/artifact)==digest
        if artifact.endswith('.pdf'):
            info=subprocess.check_output(['pdfinfo',str(ARTIFACTS/artifact)],text=True)
            assert int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:')))==1
        elif artifact.endswith('.png'):
            with Image.open(ARTIFACTS/artifact) as im:
                assert im.width>=1400 and im.height>=800
    assert plot_manifest['ABL_colors']=={'20':'#1f77b4','40':'#ff7f0e','60':'#2ca02c'}
    assert plot_manifest['display_xlim']==[-1,1]
    assert plot_manifest['data_linewidth']==1.4 and plot_manifest['model_linewidth']==.8
    assert plot_manifest['psychometric_linewidth']==1. and plot_manifest['data_zorder']>plot_manifest['model_zorder']
    data_bins=pd.read_csv(ARTIFACTS/'rtd_data_bins.csv',float_precision='round_trip')
    shown=all_saved.loc[all_saved.fit_valid|all_saved.display_abort].copy()
    shown['rt']=np.where(shown.fit_valid,shown.timed_rt,shown.abort_rt_s)
    assert len(shown)==80989
    edges=np.linspace(-1,1,101)
    for (abl,ild),z in data_bins.groupby(['ABL','abs_ILD']):
        obs=shown.loc[shown.ABL_mapped.eq(abl)&shown.ILD.abs().eq(ild)]
        counts=np.histogram(obs.rt,edges)[0]
        np.testing.assert_array_equal(counts,z['count'])
        np.testing.assert_allclose(z.density,counts/len(obs)/.02,atol=1e-12)
        np.testing.assert_allclose(z.right-z.left,.02,atol=1e-14)
    assert data_bins['count'].sum()==80989
    for variant in ['condition_delay']:
        folder=ARTIFACTS/variant
        if args.preview_initial:
            p=expanded
            ll=float(value)
            summary={'variant':variant,'status':'initial_preview'}
        else:
            config=json.loads((folder/'configuration.json').read_text())
            summary=json.loads((folder/'run_summary.json').read_text())
            assert not config['require_parameter_stability'] and not summary['parameter_stability_required']
            for path,digest in config['source_hashes'].items():
                assert sha256(ROOT/path)==digest,path
            convergence=pd.read_csv(folder/'convergence.csv')
            losses=np.load(folder/'losses.npy')
            assert np.isfinite(losses).all()
            assert len(losses)==summary['final_checked_step']
            resume=json.loads((OUT/'resume_validation.json').read_text())
            resume_folder=OUT/'condition_delay/before_resume_check'
            for name,digest in resume['archived_files'].items():
                assert sha256(resume_folder/name)==digest,name
            np.testing.assert_array_equal(losses[:resume['checkpoint_step']],np.load(resume_folder/'losses.npy'))
            assert config==json.loads((resume_folder/'configuration.json').read_text())
            assert resume['optimizer_rng_and_guide_gpu_roundtrip_exact']
            assert convergence.loc[convergence.step>resume['checkpoint_step'],'step'].iloc[0]==resume['checkpoint_step']+1000
            report['checkpoint_resume_preserves_history_optimizer_guide_rng_and_configuration']=True
            np.testing.assert_allclose(losses.reshape(-1,1000).mean(1),convergence.mean_negative_elbo,atol=1e-8)
            best=convergence.loc[convergence.mean_negative_elbo.idxmin()]
            assert summary['restored_best_step']==int(best.step)
            # Reconstruct every strict-improvement reset and the first allowed
            # stopping window, independently of the fitting loop's booleans.
            running_best=np.inf
            tail=0
            first_allowed=None
            for row in convergence.itertuples():
                if row.mean_negative_elbo<running_best:
                    running_best=row.mean_negative_elbo
                    tail=0
                else:
                    tail+=1
                assert row.patience==tail
                np.testing.assert_allclose(row.best_loss,running_best,atol=1e-8,rtol=0)
                if first_allowed is None and row.step>=50000 and tail>=25:
                    first_allowed=int(row.step)
            if summary['status']=='converged':
                assert summary['final_checked_step']==first_allowed
            assert summary['restored_checkpoint_parameter_stable']==bool(best.parameter_stable)
            np.testing.assert_allclose(summary['max_restored_mean_change_sd'],best.max_mean_change_sd,atol=1e-10)
            report['loss_confirmation_resets_and_first_allowed_stop_verified']=True
            names=parameter_names(variant)
            means=convergence[[n+'_mean' for n in names]].to_numpy()
            sds=convergence[[n+'_sd' for n in names]].to_numpy()
            change=np.ptp(means[-3:],axis=0)/np.min(sds[-3:],axis=0)
            np.testing.assert_allclose(change,convergence.iloc[-1][[n+'_change_sd' for n in names]].astype(float),rtol=1e-8,atol=1e-9)
            if summary['status']=='converged':
                assert convergence.iloc[-1].patience>=25 and len(losses)>=50000
                if config['require_parameter_stability']:
                    assert (change<.1).all()
                else:
                    assert summary['stop_reason']=='confirmed_minimum_and_loss_patience'
                assert summary['terminal_parameter_stable']==bool((change<.1).all())
            else:
                assert summary['status']=='inconclusive' and len(losses)==1000000
            restored=pickle.loads((folder/'restored_guide_optimizer.pkl').read_bytes())
            best_state=pickle.loads((folder/'best_checkpoint.pkl').read_bytes())
            for a,b in zip(jax.tree_util.tree_leaves(restored),jax.tree_util.tree_leaves({k:best_state[k] for k in restored})):
                np.testing.assert_array_equal(a,b)
            init={k:jnp.array(x) for k,x in config['initial'].items()}
            guide=AutoMultivariateNormal(model,init_loc_fn=init_to_value(values=init),init_scale=.1)
            svi=SVI(model,guide,numpyro.optim.ClippedAdam(.0002,clip_norm=1),Trace_ELBO())
            svi.init(jax.random.PRNGKey(0),data,variant)
            repeat=guide.sample_posterior(jax.random.PRNGKey(100000),restored['params'],sample_shape=(10000,))
            samples=dict(np.load(folder/'posterior_samples.npz'))
            cross_backend=config['backend']!=jax.default_backend()
            draw_error=0.
            for name,x in samples.items():
                assert np.isfinite(x).all()
                if cross_backend:
                    # GPU and CPU floating-point transforms need not be bitwise
                    # identical; the seed, saved guide and all draws are fixed.
                    np.testing.assert_allclose(x,repeat[name],rtol=5e-12,atol=5e-12)
                else:
                    np.testing.assert_array_equal(x,repeat[name])
                draw_error=max(draw_error,float(np.max(np.abs(x-np.asarray(repeat[name])))))
            report['posterior_reproduction'][variant]={'fit_backend':config['backend'],
                'verification_backend':jax.default_backend(),'max_absolute_draw_difference':draw_error,
                'comparison':'float64 allclose (rtol=atol=5e-12)' if cross_backend else 'bitwise equal'}
            matrix=sample_matrix(samples)
            estimates=pd.read_csv(folder/'posterior_summary.csv')
            np.testing.assert_allclose(matrix.mean(0),estimates['mean'],atol=1e-12)
            np.testing.assert_allclose(matrix.std(0,ddof=1),estimates['sd'],atol=1e-12)
            np.testing.assert_allclose(np.median(matrix,axis=0),estimates['median'],atol=1e-12)
            np.testing.assert_allclose(np.quantile(matrix,[.025,.975],axis=0).T,estimates[['q025','q975']],atol=1e-12)
            np.testing.assert_allclose(np.cov(matrix,rowvar=False),np.load(folder/'posterior_covariance.npy'),atol=1e-12)
            p={k:jnp.asarray(x.mean(0)) for k,x in samples.items()}
            likelihood_value,posterior_gradient=jax.jit(jax.value_and_grad(log_likelihood),static_argnames=['variant'])(p,data,variant)
            ll=float(likelihood_value)
            assert all(np.isfinite(np.asarray(x)).all() for x in posterior_gradient.values())
            np.testing.assert_allclose(ll,summary['posterior_mean_joint_log_likelihood'],atol=1e-7)
            assert len(estimates)==(63 if variant=='shared_delay' else 92)
        # Independent SciPy likelihood and prediction reconstruction at the
        # final fitted values; input conditions/onsets are never resampled.
        model_parameters=pd.read_csv(folder/'condition_parameter_means.csv',float_precision='round_trip')
        psych=pd.read_csv(folder/'psychometric_predictions.csv',float_precision='round_trip')
        cdfs=pd.read_csv(folder/'valid_cdfs_1ms.csv.gz',float_precision='round_trip')
        parent_model=pd.read_csv(folder/'signed_parent_model_1ms.csv.gz',float_precision='round_trip')
        model_bins=pd.read_csv(folder/'pooled_rtd_model_bins.csv',float_precision='round_trip')
        panel_audit=pd.read_csv(folder/'rtd_panel_audit.csv',float_precision='round_trip')
        pooled_curves=pd.read_csv(folder/'pooled_rtd_model_1ms.csv.gz',float_precision='round_trip')
        assert len(psych)==len(model_parameters)==30 and len(panel_audit)==15
        np.testing.assert_allclose(model_parameters.gamma,p['gamma'],atol=1e-12)
        np.testing.assert_allclose(model_parameters.omega,p['omega'],atol=1e-12)
        np.testing.assert_allclose(model_parameters.t_E_aff,np.broadcast_to(p['t_E_aff'],(30,)),atol=1e-12)
        np.testing.assert_allclose(model_parameters.w,p['w'],atol=1e-12)
        np.testing.assert_allclose(model_parameters.del_go,p['del_go'],atol=1e-12)
        assert psych.model_p_right.between(0,1).all()
        assert (model_bins.window_mass>=-1e-12).all() and (pooled_curves.density>=0).all()
        ll_scipy=0.;max_psych_error=0.;independent_signed_cdfs={}
        for row in model_parameters.itertuples():
            i=row.condition
            z=saved.loc[saved.condition.eq(i)]
            design=all_saved.loc[all_saved.condition.eq(i)]
            onset=z.onset_s.to_numpy()
            g,o,d,w,go=row.gamma,row.omega,row.t_E_aff,row.w,row.del_go
            norm=scipy_mass(1.,onset,g,o,d,w)-scipy_mass(0.,onset,g,o,d,w)
            q=scipy_race(z.timed_rt.to_numpy(),onset,g,o,d,w,go,z.response_poke.to_numpy())
            ll_scipy+=np.sum(np.log(q)-np.log(norm))
            pp=psych.loc[psych.condition.eq(i)].iloc[0]
            assert pp.trials==len(z) and pp.right==z.response_poke.eq(1).sum()
            np.testing.assert_allclose(pp.data_p_right,z.response_poke.eq(1).mean(),atol=1e-15)
            # Adaptive independent quadrature, with analytic-onset split points.
            probability=quad(lambda rt:float(np.mean(scipy_race(rt,onset,g,o,d,w,go,1)/norm)),0,1,
                points=sorted(set(x for x in [d-go,d] if 0<x<1)),epsabs=2e-8,epsrel=2e-8,limit=150)[0]
            np.testing.assert_allclose(probability,pp.model_p_right,atol=2e-7,rtol=2e-7)
            max_psych_error=max(max_psych_error,abs(probability-pp.model_p_right))
            # All 1 ms CDF values and density values are reconstructed from
            # independent Wiener kernels and SciPy inverse-Gaussian survival.
            times=np.linspace(-1,1,2001)
            evidence_cdf=scipy_ddm(times,g,o,w,d,cumulative=True)
            evidence_pdf=scipy_ddm(times,g,o,w,d)
            design_onsets=design.onset_s.to_numpy()
            independent_cdf=np.empty(len(times));independent_pdf=np.empty(len(times))
            vv,aa,tt,pl,bb=fixed
            for start in range(0,len(times),100):
                sl=slice(start,start+100)
                absolute=times[sl,None]+design_onsets[None,:]
                sa=invgauss.sf(absolute-tt,mu=1/(vv*aa),scale=aa*aa).mean(1)
                fa=invgauss.pdf(absolute-tt,mu=1/(vv*aa),scale=aa*aa).mean(1)
                le=np.exp(-bb*np.maximum(absolute,0))
                independent_cdf[sl]=1-(1-pl)*sa*(1-evidence_cdf[sl])-pl*le.mean(1)
                independent_pdf[sl]=(1-pl)*(fa*(1-evidence_cdf[sl])+sa*evidence_pdf[sl])+pl*np.where(absolute>=0,bb*le,0.).mean(1)
            reference=parent_model.loc[parent_model.condition.eq(i)]
            np.testing.assert_allclose(reference.unnormalized_density,independent_pdf,atol=3e-10,rtol=2e-8)
            np.testing.assert_allclose(reference.unconditional_cdf,independent_cdf,atol=3e-11,rtol=2e-8)
            independent_signed_cdfs[i]=(independent_cdf,independent_pdf)
            ref_cdf=cdfs.loc[cdfs.condition.eq(i)]
            for start in range(0,len(ref_cdf),100):
                part=ref_cdf.iloc[start:start+100]
                absolute=part.rt_s.to_numpy()[:,None]+onset[None,:]
                surv=invgauss.sf(absolute-tt,mu=1/(vv*aa),scale=aa*aa)
                evidence_surv=1-scipy_ddm(part.rt_s.to_numpy(),g,o,w,d,cumulative=True)
                raw_cdf=1-(1-pl)*surv*evidence_surv[:,None]-pl*np.exp(-bb*absolute)
                conditional=((raw_cdf-scipy_mass(0.,onset,g,o,d,w))/norm).mean(1)
                np.testing.assert_allclose(part.model_cdf,conditional,atol=3e-11)
            empirical=np.searchsorted(np.sort(z.timed_rt),ref_cdf.rt_s,side='right')/len(z)
            np.testing.assert_allclose(empirical,ref_cdf.data_cdf,atol=1e-15)
            np.testing.assert_allclose(pp.rt_cdf_max_abs_gap_1ms,np.max(np.abs(empirical-ref_cdf.model_cdf)),atol=1e-12)
            np.testing.assert_allclose(pp.rt_cdf_integrated_abs_gap,trapezoid(np.abs(empirical-ref_cdf.model_cdf),ref_cdf.rt_s),atol=1e-12)
            print(f'Independent {variant}: condition {i+1}/30 passed',flush=True)
        np.testing.assert_allclose(ll_scipy,ll,atol=1e-6,rtol=1e-10)
        for panel in panel_audit.itertuples():
            matching=conditions.loc[conditions.ABL.eq(panel.ABL)&conditions.ILD.abs().eq(panel.abs_ILD)]
            weights=matching.parent_trials.to_numpy()
            aggregate_cdf=np.average([independent_signed_cdfs[i][0] for i in matching.condition],axis=0,weights=weights)
            aggregate_pdf=np.average([independent_signed_cdfs[i][1] for i in matching.condition],axis=0,weights=weights)
            window_mass=aggregate_cdf[-1]-aggregate_cdf[0]
            np.testing.assert_allclose(window_mass,panel.model_window_mass,atol=1e-12)
            bins=model_bins.loc[model_bins.ABL.eq(panel.ABL)&model_bins.abs_ILD.eq(panel.abs_ILD)]
            np.testing.assert_allclose(bins.unnormalized_mass,np.diff(aggregate_cdf[::20]),atol=3e-11)
            np.testing.assert_allclose(bins.window_mass,np.diff(aggregate_cdf[::20])/window_mass,atol=3e-11)
            np.testing.assert_allclose(np.sum(bins.density*.02),1,atol=1e-12)
            curve=pooled_curves.loc[pooled_curves.ABL.eq(panel.ABL)&pooled_curves.abs_ILD.eq(panel.abs_ILD)]
            np.testing.assert_allclose(curve.density,aggregate_pdf/window_mass,atol=3e-10)
            np.testing.assert_allclose(trapezoid(curve.density,curve.rt_s),1,atol=3e-4)
            assert curve.density.max()<plot_manifest['display_ylim'][1]
        plot_checks.append(dict(variant=variant,rtd_panels=15,psychometric_points=30,
            independent_log_likelihood=ll_scipy,max_psychometric_integration_error=max_psych_error,
            all_parent_weighted_model_values_verified=True,all_valid_conditional_cdf_values_verified=True))
        summaries.append(summary)
    report['fit_summaries']=summaries
    report['posterior_draws_and_checkpoint_restoration_verified']=not args.preview_initial
    report['posterior_mean_gradients_finite']=not args.preview_initial
    report['plot_checks']=plot_checks
    report['all_pdf_page_counts_and_image_dimensions_verified']=True
    report['figure_artifact_hashes']=plot_manifest['artifacts']
# %% All inputs and frozen previous analyses must remain byte-identical
for path,digest in prep['preservation_hashes'].items():
    assert sha256(ROOT/path)==digest,path
report['input_and_abort_fit_hashes_unchanged']=True
path=ARTIFACTS/('prefit_validation.json' if args.pre_fit else 'independent_validation.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
