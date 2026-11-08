# %%
"""64-parameter human model: one Omega per group, condition-wise Gamma/mu/sigma."""
import json
import copy
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import human_gaussian_delay_svi_utils_wide as wide
from human_gamma_omega_delay500_svi_utils import sample_trapezoid, sample_trapezoid_vector

# %% Editable settings. The validated convolution and all priors are unchanged.
ROOT, RT_WINDOW, sha256 = wide.ROOT, wide.RT_WINDOW, wide.sha256
REFERENCE_ROOT = wide.OUTPUT
BASELINE_ROOT = REFERENCE_ROOT
MODE = 'shared_omega_gaussian_delay_lapse'
MODES = (MODE,)
OUTPUT = ROOT / 'ASD_Human_Data/shared_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms'
OUTPUTS = {MODE: OUTPUT}
GRID_DT, K_MAX, QUAD_ORDER = wide.GRID_DT, wide.K_MAX, wide.QUAD_ORDER
PARAMETER_BLOCKS = ('gamma', 'omega', 'delay_mu', 'delay_sigma', 'w', 'lapse_prob', 'lapse_prob_right')
CONDITION_BLOCKS = ('gamma', 'delay_mu', 'delay_sigma')
PARAMETER_NAMES = tuple(name if name not in CONDITION_BLOCKS else f'{name}_{i:02d}'
                        for name in PARAMETER_BLOCKS
                        for i in (range(20) if name in CONDITION_BLOCKS else [0]))
PRIORS = wide.get_priors()


def get_priors(mode=MODE):
    return copy.deepcopy(PRIORS)


def parameter_names(mode=MODE):
    return PARAMETER_NAMES


def parameter_blocks(mode=MODE):
    return PARAMETER_BLOCKS


def load_data(mode=MODE):
    full, selected, excluded, provenance = wide.load_data()
    provenance.update(objective=MODE, omega='one scalar per group, shared across all 20 stimulus conditions')
    return full, selected, excluded, provenance


def prepare_group(selected, group, mode=MODE, initial_root=BASELINE_ROOT):
    sub, conditions, data, _, _ = wide.prepare_group(selected, group)
    source = initial_root / group.lower() / 'initialization.json'
    saved = json.loads(source.read_text())
    initial = {name: np.asarray(saved[name]) for name in PARAMETER_BLOCKS}
    assert initial['omega'].shape == (20,)
    initial['omega'] = np.mean(initial['omega'])
    return sub, conditions, data, initial, source


def expand_omega(params):
    """Expand only at the existing 20-condition likelihood/prediction interface."""
    return dict(params, omega=jnp.broadcast_to(jnp.asarray(params['omega']), (20,)))


def log_likelihood(params, data, mode=MODE, K=K_MAX, marginal=False, dt=GRID_DT, order=QUAD_ORDER):
    return wide.log_likelihood(expand_omega(params), data, K=K, marginal=marginal, dt=dt, order=order)


def gamma_omega_model(data, mode=MODE, K=K_MAX):
    params = {'gamma': sample_trapezoid_vector('gamma', 20, data['gamma_hard'], data['gamma_plausible'])}
    for name in PARAMETER_BLOCKS[1:]:
        prior = PRIORS[name]
        if name in CONDITION_BLOCKS:
            params[name] = sample_trapezoid_vector(name, 20, prior['hard'], prior['plausible'])
        else:
            # In particular, the Omega prior contributes ONCE, not 20 times.
            params[name] = sample_trapezoid(name, prior['hard'], prior['plausible'])
    numpyro.factor('retained_gaussian_delay_choice_lapse', log_likelihood(params, data, K=K))


def guide_support_model(data, mode=MODE, K=K_MAX):
    numpyro.sample('gamma', dist.Uniform(*data['gamma_hard']).to_event(1))
    for name in PARAMETER_BLOCKS[1:]:
        prior = dist.Uniform(*PRIORS[name]['hard'])
        if name in CONDITION_BLOCKS:
            prior = prior.expand([20]).to_event(1)
        numpyro.sample(name, prior)


def sample_matrix(samples, mode=MODE):
    return np.column_stack([np.asarray(samples[n]).reshape(len(samples[n]), -1) for n in PARAMETER_BLOCKS])


def vector_params(values, mode=MODE):
    return dict(gamma=values[:20], omega=values[20], delay_mu=values[21:41], delay_sigma=values[41:61],
                w=values[61], lapse_prob=values[62], lapse_prob_right=values[63])


def numerical_check(params, data, mode=MODE):
    return wide.numerical_check(expand_omega(params), data)


def prediction_components(params, times, mode=MODE, K=K_MAX, dt=GRID_DT):
    return wide.prediction_components(expand_omega(params), times, K=K, dt=dt)
