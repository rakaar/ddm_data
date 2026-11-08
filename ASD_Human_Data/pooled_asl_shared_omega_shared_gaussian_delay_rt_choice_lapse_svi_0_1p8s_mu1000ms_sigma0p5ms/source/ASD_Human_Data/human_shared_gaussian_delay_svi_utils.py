# %%
"""16-parameter human DDM: Gamma per difference, shared Gaussian delay/Omega per group."""
import copy
import json
import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import human_gaussian_delay_svi_utils_wide as wide
from human_gamma_omega_delay500_svi_utils import sample_trapezoid, sample_trapezoid_vector

# %% Shared numerical settings and explicit parameter ordering.
ROOT, RT_WINDOW, sha256 = wide.ROOT, wide.RT_WINDOW, wide.sha256
BASELINE_ROOT = ROOT/'ASD_Human_Data/pooled_asl_shared_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms'
REFERENCE_ROOT = BASELINE_ROOT
MODE = 'pooled_asl_shared_omega_shared_gaussian_delay_lapse'
MODES = (MODE,)
OUTPUT = ROOT/'ASD_Human_Data/pooled_asl_shared_omega_shared_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms'
OUTPUTS = {MODE: OUTPUT}
SIGNED_DIFFERENCES = np.array([-2.,-1.3,-.65,-.3,-.1,.1,.3,.65,1.3,2.])
GRID_DT, K_MAX, QUAD_ORDER = wide.GRID_DT, wide.K_MAX, wide.QUAD_ORDER
PARAMETER_BLOCKS = ('gamma','omega','delay_mu','delay_sigma','w','lapse_prob','lapse_prob_right')
CONDITION_BLOCKS = ('gamma',)
SHARED_BROADCAST_BLOCKS = ('omega','delay_mu','delay_sigma')
PARAMETER_NAMES = tuple(f'{n}_{i:02d}' if n in CONDITION_BLOCKS else n
    for n in PARAMETER_BLOCKS for i in (range(10) if n in CONDITION_BLOCKS else [0]))
PRIORS = wide.get_priors()


def get_priors(mode=MODE):
    return copy.deepcopy(PRIORS)


def parameter_names(mode=MODE):
    return PARAMETER_NAMES


def parameter_blocks(mode=MODE):
    return PARAMETER_BLOCKS


def load_data(mode=MODE):
    full, selected, excluded, provenance = wide.load_data()
    provenance.update(objective=MODE, pooling='Raw trials; Gamma per signed difference, pooled across ASLs',
                      delay='one group Gaussian mu/sigma across all stimuli; delay draws independent per trial, truncated below zero without upper cutoff',
                      omega='one scalar per group, shared across ten signed differences')
    return full, selected, excluded, provenance


def prepare_group(selected, group, mode=MODE, initial_root=BASELINE_ROOT):
    sub = selected.loc[selected.Group.eq(group)].copy()
    ct = sub.groupby('signed_delta',sort=True).agg(n=('Choice','size'),right=('ChoseRight','sum'),
        minimum_RT_s=('RT_s','min')).reset_index()
    np.testing.assert_array_equal(ct.signed_delta,SIGNED_DIFFERENCES)
    assert (len(sub),sub.Participant.nunique()) == ((6574,11) if group=='Control' else (4770,8))
    ct['condition_index'] = np.arange(10)
    sub = sub.merge(ct[['signed_delta','condition_index']],on='signed_delta',how='left',validate='many_to_one',sort=False)
    data = dict(rt=jnp.asarray(sub.RT_s.to_numpy()),choice=jnp.asarray(sub.Choice.to_numpy()),
        condition=jnp.asarray(sub.condition_index.to_numpy()),window_low=0.,window_high=1.8)
    for field in ['hard','plausible']:
        pair = np.where((SIGNED_DIFFERENCES>0)[:,None],PRIORS['gamma_positive'][field],PRIORS['gamma_negative'][field])
        data['gamma_'+field] = (jnp.asarray(pair[:,0]),jnp.asarray(pair[:,1]))
        ct['gamma_'+field+'_low'],ct['gamma_'+field+'_high'] = pair[:,0],pair[:,1]
    source = initial_root/group.lower()/'initialization.json'
    saved = json.loads(source.read_text())
    initial = {n:np.asarray(saved[n]) for n in PARAMETER_BLOCKS}
    assert initial['gamma'].shape == (10,)
    for n in ['delay_mu','delay_sigma']:
        assert initial[n].shape == (10,)
        initial[n] = np.mean(initial[n])
    assert initial['omega'].shape == ()
    return sub,ct,data,initial,source


def expand_shared(params):
    # Broadcasting evaluates ten condition densities without adding parameters or priors.
    return {n:(jnp.broadcast_to(jnp.asarray(v),(10,)) if n in SHARED_BROADCAST_BLOCKS else v)
            for n,v in params.items()}


def log_likelihood(params,data,mode=MODE,K=K_MAX,marginal=False,dt=GRID_DT,order=QUAD_ORDER):
    return wide.log_likelihood(expand_shared(params),data,K=K,marginal=marginal,dt=dt,order=order)


def gamma_omega_model(data,mode=MODE,K=K_MAX):
    params = {'gamma':sample_trapezoid_vector('gamma',10,data['gamma_hard'],data['gamma_plausible'])}
    for n in PARAMETER_BLOCKS[1:]:
        prior = PRIORS[n]
        params[n] = (sample_trapezoid_vector(n,10,prior['hard'],prior['plausible']) if n in CONDITION_BLOCKS
                     else sample_trapezoid(n,prior['hard'],prior['plausible']))
    numpyro.factor('retained_gaussian_delay_choice_lapse',log_likelihood(params,data,K=K))


def guide_support_model(data,mode=MODE,K=K_MAX):
    numpyro.sample('gamma',dist.Uniform(*data['gamma_hard']).to_event(1))
    for n in PARAMETER_BLOCKS[1:]:
        prior = dist.Uniform(*PRIORS[n]['hard'])
        if n in CONDITION_BLOCKS: prior = prior.expand([10]).to_event(1)
        numpyro.sample(n,prior)


def sample_matrix(samples,mode=MODE):
    return np.column_stack([np.asarray(samples[n]).reshape(len(samples[n]),-1) for n in PARAMETER_BLOCKS])


def vector_params(values,mode=MODE):
    return dict(gamma=values[:10],omega=values[10],delay_mu=values[11],delay_sigma=values[12],
                w=values[13],lapse_prob=values[14],lapse_prob_right=values[15])


def numerical_check(params,data,mode=MODE):
    return wide.numerical_check(expand_shared(params),data)


def prediction_components(params,times,mode=MODE,K=K_MAX,dt=GRID_DT):
    return wide.prediction_components(expand_shared(params),times,K=K,dt=dt)


def loss_stop_status(completed,best_step,min_steps,max_steps,confirmation_windows):
    """Loss-only stopping: count complete windows since the strict running minimum."""
    confirmed = completed>=min_steps and (completed-best_step)//1000>=confirmation_windows
    return confirmed or completed>=max_steps, confirmed
