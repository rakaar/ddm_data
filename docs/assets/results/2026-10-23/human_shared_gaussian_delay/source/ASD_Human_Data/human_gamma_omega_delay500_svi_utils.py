# %%
"""Reactive Gamma/Omega/delay per human stimulus; joint retained RT and choice.

Gamma is dimensionless, Omega has units s^-1, and w is the fractional starting
position between the lower and upper boundaries. There is no proactive race.
The fitted delay prior is always 0--500 ms. A separate guide prototype confines
each delay to its likelihood-implied causal interval, without replacing that
prior or its trapezoidal density.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from human_reactive_tied_svi_utils import ROOT, sha256, load_trials as load_original_trials, group_data
from human_plain_ddm_diagnostics_utils import ddm_log_density, ddm_log_cdf
from numpyro_vanilla_condition_delay_svi_utils import (
    trapezoidal_logpdf_jax, sample_trapezoid, sample_trapezoid_vector,
)
sys.path.insert(0, str(ROOT / 'fit_each_condn'))
from svi_gamma_omega_likelihood_utils import CDF_E_gamma_omega_with_w_jax

# %% Analysis settings: independent priors in the physical coordinates.
OUTPUT = ROOT / 'ASD_Human_Data/gamma_omega_rt_choice_svi_0_1p8s_delay500ms'
INITIAL_ROOT = ROOT / 'ASD_Human_Data/reactive_ddm_debug_0_1p8s_delay300ms'
RT_WINDOW = (0., 1.8)
DELAY_UPPER_S = .5
PARAMETER_BLOCKS = ('gamma', 'omega', 't_E_aff', 'w')
PARAMETER_NAMES = ([f'{name}_{i:02d}' for name in PARAMETER_BLOCKS[:3] for i in range(20)] + ['w'])
PRIORS = {
    'gamma_positive': dict(hard=[-1., 20.], plausible=[0., 12.]),
    'gamma_negative': dict(hard=[-20., 1.], plausible=[-12., 0.]),
    'omega': dict(hard=[.01, 15.], plausible=[.1, 12.]),
    't_E_aff': dict(hard=[0., DELAY_UPPER_S], plausible=[.01, .2]),
    'w': dict(hard=[0., 1.], plausible=[.4, .6]),
}


def load_data():
    full, provenance = load_original_trials()
    selected = full.loc[full.RT_s.between(*RT_WINDOW)].copy()
    excluded = full.loc[~full.RT_s.between(*RT_WINDOW)].copy()
    excluded['exclusion_reason'] = 'RT_s > 1.8 s'
    assert len(selected) == 11344 and len(excluded) == 56
    assert int(selected.RT_s.lt(.2).sum()) == 4
    provenance = dict(provenance, selection='Condition == RT and 0 <= RT_s <= 1.8; joint observed RT and choice',
                      pooling='raw trials within group across participants and sessions',
                      retention_window_s=list(RT_WINDOW), original_trials=len(full),
                      retained_trials=len(selected), excluded_trials=len(excluded),
                      retained_source_rows_sha256=hashlib.sha256(
                          np.asarray(selected.source_row, dtype='<i8').tobytes()).hexdigest())
    return full, selected, excluded, provenance


def prepare_group(selected, group, initial_root=INITIAL_ROOT):
    sub, conditions, data, _ = group_data(selected, group)
    expected = (6574, 11) if group == 'Control' else (4770, 8)
    assert (len(sub), sub.Participant.nunique()) == expected
    minima = sub.groupby('condition_index').RT_s.min().reindex(range(20)).to_numpy()
    conditions['minimum_RT_s'] = minima
    conditions['delay_prior_upper_s'] = DELAY_UPPER_S
    conditions['guide_delay_upper_s'] = np.minimum(DELAY_UPPER_S, minima - 1e-6)
    positive = conditions.signed_delta.to_numpy() > 0
    bounds = {}
    for field in ['hard', 'plausible']:
        pair = np.where(positive[:, None], PRIORS['gamma_positive'][field], PRIORS['gamma_negative'][field])
        bounds['gamma_' + field] = (jnp.asarray(pair[:, 0]), jnp.asarray(pair[:, 1]))
        conditions['gamma_' + field + '_low'] = pair[:, 0]
        conditions['gamma_' + field + '_high'] = pair[:, 1]
    data.update(bounds, guide_delay_upper=jnp.asarray(conditions.guide_delay_upper_s.to_numpy()))
    source = initial_root / group.lower() / 'joint_fits.json'
    previous = json.loads(source.read_text())
    assert previous['n_trials'] == len(sub) and 'joint' in previous['likelihood'].lower()
    x = np.asarray(previous['free_ddm']['x'])
    assert len(x) == 42
    initial = dict(gamma=np.sinh(x[:20]), omega=np.exp(x[20:40]),
                   t_E_aff=np.repeat(x[41], 20), w=x[40])
    assert np.all(initial['t_E_aff'] < np.asarray(data['guide_delay_upper']))
    assert np.all(initial['gamma'] > np.asarray(bounds['gamma_hard'][0]))
    assert np.all(initial['gamma'] < np.asarray(bounds['gamma_hard'][1]))
    return sub, conditions, data, initial, source


def log_likelihood(params, data, K=40):
    ids = data['condition']
    gamma, omega, delay = (params[n] for n in PARAMETER_BLOCKS[:3])
    density = ddm_log_density(data['rt'], gamma[ids], omega[ids], params['w'], delay[ids], K,
                              side=data['choice'])
    mass = ddm_log_cdf(RT_WINDOW[1], gamma, omega, params['w'], delay, K)
    return jnp.sum(density - mass[ids])


def gamma_omega_model(data, K=40):
    params = {'gamma': sample_trapezoid_vector('gamma', 20, data['gamma_hard'], data['gamma_plausible'])}
    for name in ['omega', 't_E_aff']:
        p = PRIORS[name]
        params[name] = sample_trapezoid_vector(name, 20, p['hard'], p['plausible'])
    params['w'] = sample_trapezoid('w', PRIORS['w']['hard'], PRIORS['w']['plausible'])
    numpyro.factor('joint_retained_rt_choice', log_likelihood(params, data, K))


def guide_support_model(data, K=40):
    """Only supplies support transforms; SVI evaluates gamma_omega_model."""
    numpyro.sample('gamma', dist.Uniform(*data['gamma_hard']).to_event(1))
    numpyro.sample('omega', dist.Uniform(*PRIORS['omega']['hard']).expand([20]).to_event(1))
    numpyro.sample('t_E_aff', dist.Uniform(jnp.zeros(20), data['guide_delay_upper']).to_event(1))
    numpyro.sample('w', dist.Uniform(*PRIORS['w']['hard']))


def sample_matrix(samples):
    return np.column_stack([np.asarray(samples[name]).reshape(len(samples[name]), -1) for name in PARAMETER_BLOCKS])


def vector_params(values):
    return dict(gamma=values[:20], omega=values[20:40], t_E_aff=values[40:60], w=values[60])


def numerical_check(params, data):
    """Gate long fitting on series agreement and finite physical gradients."""
    p = {n: jnp.asarray(v) for n, v in params.items()}
    ll40, ll80 = float(log_likelihood(p, data, 40)), float(log_likelihood(p, data, 80))
    gradients = jax.grad(log_likelihood)(p, data)
    assert np.isfinite(ll40) and all(np.isfinite(np.asarray(x)).all() for x in gradients.values())
    np.testing.assert_allclose(ll40, ll80, rtol=1e-11, atol=1e-6)
    return dict(joint_log_likelihood_K40=ll40, joint_log_likelihood_K80=ll80,
                absolute_series_difference=abs(ll40-ll80), finite_gradients=True)
