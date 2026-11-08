# %%
"""83-parameter human Gamma/Omega/Gaussian-delay joint fit with uniform lapses."""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from human_gamma_omega_variants_svi_utils import (
    ROOT, RT_WINDOW, sha256, load_data as old_load_data, prepare_group as old_prepare_group,
    get_priors as old_priors, trapezoidal_logpdf_jax,
)
from human_gamma_omega_delay500_svi_utils import sample_trapezoid, sample_trapezoid_vector
from gaussian_delay_likelihood import (
    gaussian_delay_grid, gaussian_delay_trial_logpdf, gaussian_delay_log_integral,
    needs_time_rescaling,
)

# %% Editable analysis settings.
MODE='gaussian_delay_lapse'
MODES=(MODE,)
OUTPUT=ROOT/'ASD_Human_Data/gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s'
OUTPUTS={MODE:OUTPUT}
BASELINE_ROOT=ROOT/'ASD_Human_Data/gamma_omega_rt_choice_lapse_svi_0_1p8s_delay500ms'
GRID_DT=.0005
K_MAX=8
QUAD_ORDER=24
PARAMETER_BLOCKS=('gamma','omega','delay_mu','delay_sigma','w','lapse_prob','lapse_prob_right')
PARAMETER_NAMES=tuple(f'{n}_{i:02d}' for n in PARAMETER_BLOCKS[:4] for i in range(20))+PARAMETER_BLOCKS[4:]
PRIORS=old_priors('rt_choice_lapse')
PRIORS['delay_mu']=PRIORS.pop('t_E_aff')
PRIORS['delay_sigma']=dict(hard=[.005,.3],plausible=[.01,.15])


def get_priors(mode=MODE):
    return copy.deepcopy(PRIORS)


def parameter_names(mode=MODE):
    return PARAMETER_NAMES


def parameter_blocks(mode=MODE):
    return PARAMETER_BLOCKS


def load_data(mode=MODE):
    full,selected,excluded,provenance=old_load_data('rt_choice_lapse')
    provenance.update(objective=MODE, delay='independent Normal(mu,sigma) per trial, truncated below zero; unbounded above')
    return full,selected,excluded,provenance


def prepare_group(selected,group,mode=MODE,initial_root=BASELINE_ROOT):
    sub,conditions,data,old,source=old_prepare_group(selected,group,'rt_choice_lapse',initial_root)
    initial=dict(gamma=old['gamma'],omega=old['omega'],delay_mu=old['t_E_aff'],
                 delay_sigma=np.full(20,.05),w=old['w'],lapse_prob=old['lapse_prob'],
                 lapse_prob_right=old['lapse_prob_right'])
    conditions=conditions.drop(columns=['delay_prior_upper_s','guide_delay_upper_s'])
    conditions['delay_mu_hard_upper_s']=.5
    conditions['delay_sigma_hard_lower_s']=.005
    conditions['delay_sigma_hard_upper_s']=.3
    data.pop('guide_delay_upper',None)
    data.update(window_low=0.,window_high=1.8)
    return sub,conditions,data,initial,source


def log_likelihood(params,data,mode=MODE,K=K_MAX,marginal=False,dt=GRID_DT,order=QUAD_ORDER):
    grid=jnp.linspace(0.,1.8,round(1.8/dt)+1)
    floor=params['lapse_prob']*jnp.minimum(params['lapse_prob_right'],1-params['lapse_prob_right'])/1.8
    conditional,_=gaussian_delay_trial_logpdf(params,data,grid,K,order,lapse_floor=floor)
    if marginal:
        opposite,_=gaussian_delay_trial_logpdf(params,dict(data,choice=-data['choice']),grid,K,order,lapse_floor=floor)
        conditional=jnp.logaddexp(conditional,opposite)
        q=1.
    else:
        q=jnp.where(data['choice']==1,params['lapse_prob_right'],1-params['lapse_prob_right'])
    mixture=jnp.logaddexp(jnp.log1p(-params['lapse_prob'])+conditional,
                          jnp.log(params['lapse_prob'])+jnp.log(q)-jnp.log(1.8))
    return jnp.sum(mixture)


def gamma_omega_model(data,mode=MODE,K=K_MAX):
    p={'gamma':sample_trapezoid_vector('gamma',20,data['gamma_hard'],data['gamma_plausible'])}
    for name in PARAMETER_BLOCKS[1:4]:
        prior=PRIORS[name]
        p[name]=sample_trapezoid_vector(name,20,prior['hard'],prior['plausible'])
    for name in PARAMETER_BLOCKS[4:]:
        prior=PRIORS[name]
        p[name]=sample_trapezoid(name,prior['hard'],prior['plausible'])
    numpyro.factor('retained_gaussian_delay_choice_lapse',log_likelihood(p,data,K=K))


def guide_support_model(data,mode=MODE,K=K_MAX):
    numpyro.sample('gamma',dist.Uniform(*data['gamma_hard']).to_event(1))
    for name in PARAMETER_BLOCKS[1:4]:
        numpyro.sample(name,dist.Uniform(*PRIORS[name]['hard']).expand([20]).to_event(1))
    for name in PARAMETER_BLOCKS[4:]:
        numpyro.sample(name,dist.Uniform(*PRIORS[name]['hard']))


def sample_matrix(samples,mode=MODE):
    return np.column_stack([np.asarray(samples[n]).reshape(len(samples[n]),-1) for n in PARAMETER_BLOCKS])


def vector_params(values,mode=MODE):
    return dict(gamma=values[:20],omega=values[20:40],delay_mu=values[40:60],delay_sigma=values[60:80],
                w=values[80],lapse_prob=values[81],lapse_prob_right=values[82])


def numerical_check(params,data,mode=MODE):
    p={n:jnp.asarray(v) for n,v in params.items()}
    scores={}
    for K in [8,40,80]:
        scores[K]=float(jax.jit(lambda q:log_likelihood(q,data,K=K))(p))
    fine=float(jax.jit(lambda q:log_likelihood(q,data,dt=GRID_DT/2,order=48))(p))
    gradient=jax.jit(jax.grad(lambda q:log_likelihood(q,data)))(p)
    assert np.isfinite(list(scores.values())).all()
    assert all(np.isfinite(np.asarray(v)).all() for v in gradient.values())
    assert max(abs(v-scores[8]) for v in scores.values()) < 1e-6
    assert abs(fine-scores[8])/len(data['rt']) < 1e-4
    return dict(joint_log_likelihood=scores[8],series_log_likelihoods=scores,
                fine_grid_log_likelihood=fine,mean_log_likelihood_grid_difference=abs(fine-scores[8])/len(data['rt']),
                finite_gradients=True)


def prediction_components(params,times,mode=MODE,K=K_MAX,dt=GRID_DT):
    p={n:jnp.asarray(v) for n,v in params.items()}
    grid=jnp.linspace(0.,1.8,round(1.8/dt)+1)
    f,c=gaussian_delay_grid(p,grid,K)
    f,c=np.asarray(f).copy(),np.asarray(c).copy()
    # Rare very fast conditions need the same rescaled integral used in fitting.
    direct=np.asarray(needs_time_rescaling(p['gamma'],p['omega'],p['w'],dt))
    for i in np.flatnonzero(direct):
        for s,side in enumerate([-1,1]):
            for cumulative,dest in [(False,f),(True,c)]:
                dest[i,s]=np.exp(np.asarray(gaussian_delay_log_integral(grid,p['gamma'][i],p['omega'][i],
                    p['w'],p['delay_mu'][i],p['delay_sigma'][i],side,cumulative,K,QUAD_ORDER)))
    ends=np.exp(np.asarray(gaussian_delay_log_integral(1.8,p['gamma'][:,None],p['omega'][:,None],
        p['w'],p['delay_mu'][:,None],p['delay_sigma'][:,None],jnp.array([-1,1])[None,:],True,K,QUAD_ORDER)))
    mass=ends.sum(1)
    t=np.asarray(times);inside=(t>=0)&(t<=1.8)
    raw=np.array([[np.interp(t,np.asarray(grid),row) for row in cond] for cond in f])
    cumul=np.array([[np.interp(t,np.asarray(grid),row) for row in cond] for cond in c])
    lapse,q=float(p['lapse_prob']),float(p['lapse_prob_right'])
    joint=(1-lapse)*raw/mass[:,None,None]+lapse*np.array([1-q,q])[None,:,None]/1.8
    joint*=inside[None,None,:]
    cdf=(1-lapse)*cumul.sum(1)/mass[:,None]+lapse*np.clip(t/1.8,0,1)[None,:]
    cdf[:,t<=0]=0.;cdf[:,t>=1.8]=1.
    return dict(left_density=joint[:,0,:].T,right_density=joint[:,1,:].T,density=joint.sum(1).T,
                cdf=cdf.T,p_right=(1-lapse)*ends[:,1]/mass+lapse*q,retained_mass=mass)
