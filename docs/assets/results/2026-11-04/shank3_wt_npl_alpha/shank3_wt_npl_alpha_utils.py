# %% NPL+alpha mapping, priors, and the existing complete-mixture WT likelihood
"""No LED, no proactive truncation, and no extra reactive lapse distribution."""
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from shank3_wt_gamma_omega_utils import ROOT, ABLS, ILDS, K_DEFAULT, sha256, log_likelihood as race_log_likelihood
from proactive_lapse_no_led_utils import trapezoid_logpdf

sys.path.insert(0, str(ROOT/'fit_animal_by_animal'))
from numpyro_npl_alpha_svi_utils import gamma_omega_alpha_jax, GLOBAL_BOUNDS

OUT = ROOT/'asd_analysis/shank3_cohort1_wt_npl_alpha_condition_delay'
REFERENCE = ROOT/'asd_analysis/shank3_cohort1_wt_gamma_omega'
GLOBALS = ['rate_lambda', 'T_0', 'theta_E', 'rate_norm_l', 'alpha', 'w', 'del_go']
BLOCKS = GLOBALS + ['t_E_aff']
PRIORS = {name: [*GLOBAL_BOUNDS[name]['hard'][:1], *GLOBAL_BOUNDS[name]['plausible'], GLOBAL_BOUNDS[name]['hard'][1]] for name in GLOBALS}
PRIORS['t_E_aff'] = [0., .01, .20, .50]
INITIAL = dict(rate_lambda=1.8, T_0=.15, theta_E=5., rate_norm_l=.9, alpha=1., w=.5, del_go=.08,
               t_E_aff=np.full(30, .05))
ABL_ARRAY = jnp.repeat(jnp.asarray(ABLS, dtype=float), 10)
ILD_ARRAY = jnp.tile(jnp.asarray(ILDS, dtype=float), 3)


def expand_parameters(params):
    gamma, omega = gamma_omega_alpha_jax(ABL_ARRAY, ILD_ARRAY, params['rate_lambda'], params['T_0'],
        params['theta_E'], params['rate_norm_l'], params['alpha'])
    return dict(gamma=gamma, omega=omega, t_E_aff=params['t_E_aff'], w=params['w'], del_go=params['del_go'])


def log_likelihood(params, data, marginal=False):
    return race_log_likelihood(expand_parameters(params), data, 'condition_delay', marginal=marginal)


def model(data):
    params = {}
    for name in BLOCKS:
        lo, pl, ph, hi = PRIORS[name]
        prior = dist.Uniform(lo, hi)
        if name == 't_E_aff':
            prior = prior.expand((30,)).to_event(1)
        value = numpyro.sample(name, prior)
        numpyro.factor(name+'_trapezoid', jnp.sum(trapezoid_logpdf(value, (lo, pl, ph, hi))+jnp.log(hi-lo)))
        params[name] = value
    numpyro.factor('valid_rt_choice', log_likelihood(params, data))


def sample_matrix(samples):
    return np.column_stack([np.asarray(samples[n]).reshape(len(samples[n]), -1) for n in BLOCKS])


def parameter_names():
    return GLOBALS + [f't_E_aff_{i:02d}' for i in range(30)]


def loss_window_decision(loss, best_loss, step, patience, minimum=50000, confirmation=25):
    improved = loss < best_loss
    patience = 0 if improved else patience+1
    return improved, patience, bool(step >= minimum and patience >= confirmation)
