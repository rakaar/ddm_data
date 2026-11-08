# %%
"""Human reactive Gamma/Omega fits: marginal RTs or joint choices with lapses.

The lapse rate is the fraction among retained 0--1.8 s trials. The DDM
component is conditioned first; lapse RTs use the original RT coordinate.
"""
from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from human_gamma_omega_delay500_svi_utils import (
    ROOT, PRIORS as BASE_PRIORS, RT_WINDOW, sha256, ddm_log_density, ddm_log_cdf,
    trapezoidal_logpdf_jax, sample_trapezoid, sample_trapezoid_vector,
    CDF_E_gamma_omega_with_w_jax, load_data as baseline_load_data,
)
from human_reactive_tied_svi_utils import group_data

# %% Editable analysis settings, shared by the driver and diagnostics.
BASELINE_ROOT = ROOT/'ASD_Human_Data/gamma_omega_rt_choice_svi_0_1p8s_delay500ms'
MODES = ('rt_only', 'rt_choice_lapse')
OUTPUTS = {
    mode: ROOT/f'ASD_Human_Data/gamma_omega_{mode}_svi_0_1p8s_delay500ms'
    for mode in MODES
}
BASE_BLOCKS = ('gamma', 'omega', 't_E_aff', 'w')
BASE_NAMES = [f'{name}_{i:02d}' for name in BASE_BLOCKS[:3] for i in range(20)]+['w']
LAPSE_PRIORS = {
    'lapse_prob': dict(hard=[.0001, .4], plausible=[.001, .1]),
    'lapse_prob_right': dict(hard=[.001, .999], plausible=[.4, .6]),
}


def get_priors(mode):
    priors = copy.deepcopy(BASE_PRIORS)
    if mode == 'rt_choice_lapse':
        priors.update(copy.deepcopy(LAPSE_PRIORS))
    return priors


def parameter_blocks(mode):
    return BASE_BLOCKS+tuple(LAPSE_PRIORS) if mode == 'rt_choice_lapse' else BASE_BLOCKS


def parameter_names(mode):
    return BASE_NAMES+list(LAPSE_PRIORS) if mode == 'rt_choice_lapse' else BASE_NAMES.copy()


def load_data(mode):
    full, selected, excluded, provenance = baseline_load_data()
    provenance = dict(provenance, objective=mode,
                      selection='Condition == RT, Valid == 1, 0 <= RT_s <= 1.8',
                      choice_use='marginalized' if mode == 'rt_only' else 'observed choice with lapse mixture')
    return full, selected, excluded, provenance


def prepare_group(selected, group, mode, initial_root=BASELINE_ROOT):
    sub, conditions, data, _ = group_data(selected, group)
    expected = (6574, 11) if group == 'Control' else (4770, 8)
    assert (len(sub), sub.Participant.nunique()) == expected
    minima = sub.groupby('condition_index').RT_s.min().reindex(range(20)).to_numpy()
    conditions['minimum_RT_s'] = minima
    conditions['delay_prior_upper_s'] = .5
    conditions['guide_delay_upper_s'] = (np.minimum(.5, minima-1e-6)
                                        if mode == 'rt_only' else np.full(20, .5))
    positive = conditions.signed_delta.to_numpy() > 0
    for field in ['hard', 'plausible']:
        pair = np.where(positive[:, None], BASE_PRIORS['gamma_positive'][field],
                        BASE_PRIORS['gamma_negative'][field])
        data['gamma_'+field] = (jnp.asarray(pair[:, 0]), jnp.asarray(pair[:, 1]))
        conditions['gamma_'+field+'_low'] = pair[:, 0]
        conditions['gamma_'+field+'_high'] = pair[:, 1]
    data['guide_delay_upper'] = jnp.asarray(conditions.guide_delay_upper_s.to_numpy())
    source = Path(initial_root)/group.lower()/'initialization.json'
    saved = json.loads(source.read_text())
    initial = {name: np.asarray(saved[name]) for name in BASE_BLOCKS}
    if mode == 'rt_choice_lapse':
        initial.update(lapse_prob=np.asarray(.02), lapse_prob_right=np.asarray(.5))
    assert np.all(initial['t_E_aff'] < np.asarray(data['guide_delay_upper']))
    return sub, conditions, data, initial, source


def log_likelihood(params, data, mode, K=40, marginal=False):
    ids = data['condition']
    gamma, omega, delay = (params[n] for n in BASE_BLOCKS[:3])
    density = ddm_log_density(data['rt'], gamma[ids], omega[ids], params['w'], delay[ids], K,
                              side=None if (mode == 'rt_only' or marginal) else data['choice'])
    mass = ddm_log_cdf(RT_WINDOW[1], gamma, omega, params['w'], delay, K)
    conditional = density-mass[ids]
    if mode == 'rt_choice_lapse':
        q = (jnp.ones_like(data['rt']) if marginal else
             jnp.where(data['choice'] == 1, params['lapse_prob_right'], 1-params['lapse_prob_right']))
        conditional = jnp.logaddexp(jnp.log1p(-params['lapse_prob'])+conditional,
                                   jnp.log(params['lapse_prob'])+jnp.log(q)-jnp.log(RT_WINDOW[1]))
    return jnp.sum(conditional)


def gamma_omega_model(data, mode, K=40):
    priors = get_priors(mode)
    params = {'gamma': sample_trapezoid_vector('gamma', 20, data['gamma_hard'], data['gamma_plausible'])}
    for name in ['omega', 't_E_aff']:
        p = priors[name]
        params[name] = sample_trapezoid_vector(name, 20, p['hard'], p['plausible'])
    for name in parameter_blocks(mode)[3:]:
        p = priors[name]
        params[name] = sample_trapezoid(name, p['hard'], p['plausible'])
    numpyro.factor('retained_'+mode, log_likelihood(params, data, mode, K))


def guide_support_model(data, mode, K=40):
    priors = get_priors(mode)
    numpyro.sample('gamma', dist.Uniform(*data['gamma_hard']).to_event(1))
    numpyro.sample('omega', dist.Uniform(*priors['omega']['hard']).expand([20]).to_event(1))
    numpyro.sample('t_E_aff', dist.Uniform(jnp.zeros(20), data['guide_delay_upper']).to_event(1))
    for name in parameter_blocks(mode)[3:]:
        numpyro.sample(name, dist.Uniform(*priors[name]['hard']))


def sample_matrix(samples, mode):
    return np.column_stack([np.asarray(samples[name]).reshape(len(samples[name]), -1)
                            for name in parameter_blocks(mode)])


def vector_params(values, mode):
    params = dict(gamma=values[:20], omega=values[20:40], t_E_aff=values[40:60], w=values[60])
    if mode == 'rt_choice_lapse':
        params.update(lapse_prob=values[61], lapse_prob_right=values[62])
    return params


def numerical_check(params, data, mode):
    p = {n: jnp.asarray(v) for n, v in params.items()}
    ll40, ll80 = (float(log_likelihood(p, data, mode, K)) for K in [40, 80])
    gradient = jax.grad(lambda q: log_likelihood(q, data, mode))(p)
    assert np.isfinite(ll40) and all(np.isfinite(np.asarray(x)).all() for x in gradient.values())
    np.testing.assert_allclose(ll40, ll80, rtol=1e-11, atol=1e-6)
    return dict(objective=mode, log_likelihood_K40=ll40, log_likelihood_K80=ll80,
                absolute_series_difference=abs(ll40-ll80), finite_gradients=True)


def prediction_components(params, times, mode, K=40):
    g, o, tau, w = (jnp.asarray(params[n]) for n in BASE_BLOCKS)
    times = jnp.asarray(times)
    mass = jnp.exp(ddm_log_cdf(RT_WINDOW[1], g, o, w, tau, K))
    right = jnp.exp(ddm_log_density(times[:, None], g, o, w, tau, K, side=1)-jnp.log(mass))
    left = jnp.exp(ddm_log_density(times[:, None], g, o, w, tau, K, side=-1)-jnp.log(mass))
    cdf = jnp.exp(ddm_log_cdf(times[:, None], g, o, w, tau, K)-jnp.log(mass))
    p_right = CDF_E_gamma_omega_with_w_jax(RT_WINDOW[1]-tau, g, o, 1, w, K)/mass
    if mode == 'rt_choice_lapse':
        p, q = params['lapse_prob'], params['lapse_prob_right']
        inside = ((times >= 0) & (times <= RT_WINDOW[1]))[:, None]
        right = (1-p)*right+jnp.where(inside, p*q/RT_WINDOW[1], 0.)
        left = (1-p)*left+jnp.where(inside, p*(1-q)/RT_WINDOW[1], 0.)
        cdf = (1-p)*cdf+p*jnp.clip(times[:, None]/RT_WINDOW[1], 0., 1.)
        p_right = (1-p)*p_right+p*q
    inside = ((times >= 0) & (times <= RT_WINDOW[1]))[:, None]
    right, left = jnp.where(inside, right, 0.), jnp.where(inside, left, 0.)
    cdf = jnp.where(times[:, None] >= RT_WINDOW[1], 1.,
                    jnp.where(times[:, None] <= 0, 0., cdf))
    return {name: np.asarray(value) for name, value in dict(right_density=right,
        left_density=left, density=right+left, cdf=cdf, p_right=p_right, retained_mass=mass).items()}
