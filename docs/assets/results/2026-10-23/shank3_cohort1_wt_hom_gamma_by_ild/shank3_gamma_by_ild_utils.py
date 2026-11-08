# %% Editable model: 10 signed-ILD Gammas, three ABL Omegas and three scalars
"""Share Gamma across ABLs; preserve the established joint RT/choice race."""
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from shank3_wt_gamma_omega_utils import (
    ROOT, ABLS, ILDS, PRIORS, PARAM_BLOCKS, K_DEFAULT, sha256,
    log_likelihood as expanded_log_likelihood, trapezoid_logpdf)
from shank3_tied_omega_utils import EXPECTED, loss_window_decision

OUT = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_gamma_by_ild'
PREVIOUS = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_tied_omega'
VARIANTS = ['gamma_by_ild']
LABELS = {'gamma_by_ild': 'Gamma shared across ABLs; Omega by ABL; one reactive delay'}
ILD_INDEX = jnp.tile(jnp.arange(10), 3)
ABL_INDEX = jnp.repeat(jnp.arange(3), 10)


def expand_parameters(params, variant):
    """Map the 16 free values to the existing 30 signed stimulus conditions."""
    assert variant == 'gamma_by_ild'
    return {**params, 'gamma': params['gamma'][ILD_INDEX],
            'omega': params['omega'][ABL_INDEX],
            't_E_aff': jnp.broadcast_to(params['t_E_aff'], (30,))}


def model(data, variant):
    params = {}
    for name in PARAM_BLOCKS:
        if name == 'gamma':
            bounds = jnp.asarray([PRIORS['gamma_positive' if ild > 0 else 'gamma_negative']
                                  for ild in ILDS]).T
        elif name == 'omega':
            bounds = jnp.broadcast_to(jnp.asarray(PRIORS[name])[:, None], (4, 3))
        else:
            bounds = jnp.asarray(PRIORS[name])
        lo, plausible_lo, plausible_hi, hi = bounds
        prior = dist.Uniform(lo, hi)
        if jnp.ndim(lo):
            prior = prior.to_event(1)
        value = numpyro.sample(name, prior)
        numpyro.factor(name + '_trapezoid', jnp.sum(
            trapezoid_logpdf(value, (lo, plausible_lo, plausible_hi, hi)) + jnp.log(hi - lo)))
        params[name] = value
    numpyro.factor('valid_rt_choice', expanded_log_likelihood(
        expand_parameters(params, variant), data, 'condition_delay'))


def parameter_names(variant):
    assert variant == 'gamma_by_ild'
    return ([f'gamma_{i:02d}' for i in range(10)]
            + [f'omega_ABL_{abl}' for abl in ABLS] + ['t_E_aff', 'w', 'del_go'])


def initial_parameters(conditions, variant):
    assert variant == 'gamma_by_ild'
    assert list(zip(conditions.ABL, conditions.ILD)) == [(a, i) for a in ABLS for i in ILDS]
    pooled = conditions.groupby('ILD')[['right', 'trials']].sum().loc[ILDS]
    probability = (pooled.right / pooled.trials).to_numpy()
    assert np.all((probability > 0) & (probability < 1))
    bounds = np.asarray([PRIORS['gamma_positive' if i > 0 else 'gamma_negative'] for i in ILDS])
    gamma = np.clip(.5 * np.log(probability / (1 - probability)),
                    bounds[:, 0] + .0001, bounds[:, 3] - .0001)
    omega = np.array([np.average(z.omega_init, weights=z.trials)
                      for abl in ABLS for z in [conditions.loc[conditions.ABL.eq(abl)]]])
    return dict(gamma=gamma, omega=np.clip(omega, .0101, 14.9999),
                t_E_aff=np.array(.05), w=np.array(.5), del_go=np.array(.08))
