# %% Joint WT super-animal race with exponential/uniform contaminant times
"""All 20 physical parameters are free in one likelihood; no frozen abort fit."""
from pathlib import Path
import hashlib
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from proactive_lapse_no_led_utils import proactive_logpdf_logsf, trapezoid_logpdf
from shank3_individual_adaptive_bounds_utils import evidence_grid, loss_window_decision
from shank3_npl_alpha_gaussian_utils import interpolate

# %% Editable numerical/fitting settings and broad physical priors
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'asd_analysis/shank3_cohort1_wt_joint_npl_alpha_gaussian_abl_contaminants'
ANIMALS = ['ASD0040', 'ASD0042', 'ASD0044', 'ASD0047', 'ASD0049']
ABLS, ILDS = [20, 40, 60], [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
PROACTIVE = ['V_A', 'theta_A', 't_nd']
REACTIVE = ['rate_lambda', 'T_0', 'theta_E', 'rate_norm_l', 'alpha', 'w', 'del_go', 'delay_mu', 'delay_sigma']
CONTAMINANTS = ['lapse_prob', 'lapse_prob_right', 'lapse_exp_weight', 'beta_lapse']
BLOCKS = PROACTIVE + REACTIVE + CONTAMINANTS
PRIORS = dict(V_A=[.001, .01, 6., 8.], theta_A=[.1, .5, 10., 15.],
    t_nd=[-.3, -.2, .2, .5], rate_lambda=[.5, 1., 3., 5.],
    T_0=[.001, .005, .4, .8], theta_E=[.1, .5, 10., 15.],
    rate_norm_l=[0., .5, 1.05, 2.], alpha=[0., .1, 8., 12.],
    w=[0., .1, .9, 1.], del_go=[0., .01, .4, .5],
    delay_mu=[0., .01, .2, .5], delay_sigma=[.0005, .002, .15, .3],
    lapse_prob=[0., 1e-6, 1-1e-6, 1.],
    lapse_prob_right=[0., 1e-6, 1-1e-6, 1.],
    lapse_exp_weight=[0., 1e-6, 1-1e-6, 1.], beta_lapse=[.0001, .05, 20., 50.])
WINDOW, MIN_STEPS, CONFIRMATION, MAX_STEPS = 1000, 50000, 25, 1000000
LEARNING_RATE, CLIP_NORM, INIT_SCALE, DRAWS = .0002, 1., .1, 10000
GRID_DT, K, QUAD_ORDER = .0005, 12, 24


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parameter_names():
    return PROACTIVE + REACTIVE[:7] + [f'{key}_ABL{abl}' for key in REACTIVE[7:] for abl in ABLS] + CONTAMINANTS


def sample_matrix(samples):
    return np.column_stack([np.asarray(samples[n]).reshape(len(samples[n]), -1) for n in BLOCKS])


def initialization(seed):
    # Two distinct physical starts; every parameter subsequently updates jointly.
    if seed == 0:
        values = [1.6, 2.5, .05, 1.8, .15, 5., .9, 1., .5, .08,
            [.08, .09, .10], [.02, .02, .02], .05, .5, .5, 5.]
    else:
        values = [3., 1.5, -.02, 1.5, .08, 2., .85, 2., .52, .10,
            [.09, .10, .11], [.025, .025, .025], .08, .55, .8, 2.]
    return {n: np.asarray(x, dtype=float) for n, x in zip(BLOCKS, values)}


def exp_integral_ratio(x):
    """(1-exp(-x))/x, including the uniform beta=0 limit and finite gradients."""
    small = jnp.abs(x) < 1e-4
    safe = jnp.where(small, 1., x)
    return jnp.where(small, 1-x/2+x*x/6-x*x*x/24+x**4/120, -jnp.expm1(-safe)/safe)


def contaminant(t, params, cutoff):
    """Normalized PDF/CDF/SF from fixation onset, with common finite cutoff."""
    beta, d = params['beta_lapse'], params['lapse_exp_weight']
    u = jnp.clip(jnp.asarray(t), 0., cutoff)
    normalizer = exp_integral_ratio(beta*cutoff)
    exponential_pdf = jnp.exp(-beta*u)/(cutoff*normalizer)
    exponential_cdf = u/cutoff*exp_integral_ratio(beta*u)/normalizer
    exponential_sf = jnp.exp(-beta*u)*(cutoff-u)/cutoff*exp_integral_ratio(beta*(cutoff-u))/normalizer
    pdf = jnp.where((t >= 0) & (t <= cutoff), d*exponential_pdf+(1-d)/cutoff, 0.)
    return pdf, d*exponential_cdf+(1-d)*u/cutoff, d*exponential_sf+(1-d)*(1-u/cutoff)


def _joint_log_likelihood(params, data, grid_end, dt):
    grid, pdf, cdf = evidence_grid(params, dt=dt, K=K, order=QUAD_ORDER, grid_end=grid_end)
    ids, rt, onset = data['condition'], data['rt'], data['onset']
    f_e = interpolate(pdf, grid, rt, ids)
    f_c = interpolate(cdf, grid, rt, ids)
    f_go = interpolate(cdf, grid, rt+params['del_go'], ids)
    readout = .5*(1-f_go.sum(-1))[..., None]+f_go-f_c
    lp_a, ls_a = proactive_logpdf_logsf(onset+rt, params['V_A'], params['theta_A'], params['t_nd'])
    ordinary = jnp.exp(lp_a)[..., None]*readout+jnp.exp(ls_a)[..., None]*f_e
    q, p = params['lapse_prob_right'], params['lapse_prob']
    choice_weights = jnp.array([1-q, q])
    h, _, _ = contaminant(onset+rt, params, data['cutoff'])
    joint = (1-p)*ordinary+p*h[..., None]*choice_weights
    observed = joint[jnp.arange(rt.size), (data['choice'] == 1).astype(int)]
    valid_ll = jnp.sum(jnp.log(jnp.maximum(observed, 1e-300)))

    # Abort choices are unobserved: integrate/sum them out, rather than inventing choices.
    abort_t = data['abort_times']
    abort_lp, _ = proactive_logpdf_logsf(abort_t, params['V_A'], params['theta_A'], params['t_nd'])
    abort_h, _, _ = contaminant(abort_t, params, data['cutoff'])
    abort_density = (1-p)*jnp.exp(abort_lp)+p*abort_h
    abort_ll = jnp.dot(jnp.log(jnp.maximum(abort_density, 1e-300)), data['abort_counts'])

    # End-of-window survival for every condition: the complete race and contaminant
    # are truncated together, after marginalizing the Gaussian reactive delay.
    end_e = interpolate(cdf, grid, jnp.ones(30), jnp.arange(30)).sum(-1)
    def end_survival(onsets, conditions):
        _, ls = proactive_logpdf_logsf(onsets+1, params['V_A'], params['theta_A'], params['t_nd'])
        _, _, hsf = contaminant(onsets+1, params, data['cutoff'])
        return (1-p)*jnp.exp(ls)*(1-end_e[conditions])+p*hsf

    # Late fixation-labelled trials retain censor-at-onset semantics, within the
    # selected population T <= onset+1. They contribute P(onset <= T <= onset+1).
    _, late_ls = proactive_logpdf_logsf(data['late_onsets'], params['V_A'], params['theta_A'], params['t_nd'])
    _, _, late_hsf = contaminant(data['late_onsets'], params, data['cutoff'])
    late_sf = (1-p)*jnp.exp(late_ls)+p*late_hsf
    late_mass = late_sf-end_survival(data['late_onsets'], data['late_condition'])
    censor_ll = jnp.sum(jnp.log(jnp.maximum(late_mass, 1e-300)))

    _, entry_ls = proactive_logpdf_logsf(jnp.asarray(0.), params['V_A'], params['theta_A'], params['t_nd'])
    entry_survival = (1-p)*jnp.exp(entry_ls)+p
    retained_mass = entry_survival-end_survival(data['parent_onsets'], data['parent_condition'])
    selection_correction = -jnp.sum(jnp.log(jnp.maximum(retained_mass, 1e-300)))
    return valid_ll+abort_ll+censor_ll+selection_correction


def log_likelihood(params, data, grid_end=1.5):
    return jax.lax.cond(jnp.min(params['delay_sigma']) < .0025,
        lambda _: _joint_log_likelihood(params, data, grid_end, .0001),
        lambda _: _joint_log_likelihood(params, data, grid_end, GRID_DT), None)


def make_model(priors, grid_end):
    def model(data):
        params = {}
        for key in BLOCKS:
            lo, pl, ph, hi = priors[key]
            base = dist.Uniform(lo, hi)
            if key in ['delay_mu', 'delay_sigma']:
                base = base.expand((3,)).to_event(1)
            value = numpyro.sample(key, base)
            numpyro.factor(key+'_trapezoid', jnp.sum(trapezoid_logpdf(value, (lo, pl, ph, hi))+jnp.log(hi-lo)))
            params[key] = value
        if data is not None:
            numpyro.factor('all_joint_observations', log_likelihood(params, data, grid_end))
    return model
