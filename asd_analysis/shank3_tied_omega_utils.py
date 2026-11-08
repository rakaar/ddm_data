# %% Shared model definitions; all scientific race kernels remain in the WT utility
from pathlib import Path
import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from shank3_wt_gamma_omega_utils import (
    ROOT, ABLS, ILDS, PRIORS, PARAM_BLOCKS, K_DEFAULT, sha256,
    log_likelihood as expanded_log_likelihood, trapezoid_logpdf)

OUT = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_tied_omega'
VARIANTS = ['omega_by_abl', 'omega_global']
LABELS = {'omega_by_abl': 'Omega shared within ABL; one reactive delay',
          'omega_global': 'One global Omega; one reactive delay'}
EXPECTED = {'wt': dict(animals=5, parent=83921, valid=68184, right=37482, left=30702, aborts=13750),
            'hom': dict(animals=6, parent=81721, valid=67715, right=35320, left=32395, aborts=13274)}
ABL_INDEX = jnp.repeat(jnp.arange(3), 10)


def expand_parameters(params, variant):
    """Deterministic expansion, not 30 extra samples or repeated priors."""
    omega = params['omega'][ABL_INDEX] if variant == 'omega_by_abl' else jnp.broadcast_to(params['omega'], (30,))
    return {**params, 'omega': omega, 't_E_aff': jnp.broadcast_to(params['t_E_aff'], (30,))}


def model(data, variant):
    params = {}
    for name in PARAM_BLOCKS:
        if name == 'gamma':
            bounds = jnp.asarray([PRIORS['gamma_positive' if ild > 0 else 'gamma_negative']
                                  for abl in ABLS for ild in ILDS]).T
        else:
            bounds = jnp.asarray(PRIORS[name])
            if name == 'omega' and variant == 'omega_by_abl':
                bounds = jnp.broadcast_to(bounds[:, None], (4, 3))
        lo, plausible_lo, plausible_hi, hi = bounds
        prior = dist.Uniform(lo, hi)
        if jnp.ndim(lo):
            prior = prior.to_event(1)
        value = numpyro.sample(name, prior)
        numpyro.factor(name + '_trapezoid', jnp.sum(
            trapezoid_logpdf(value, (lo, plausible_lo, plausible_hi, hi)) + jnp.log(hi - lo)))
        params[name] = value
    numpyro.factor('valid_rt_choice', expanded_log_likelihood(expand_parameters(params, variant), data, 'condition_delay'))


def parameter_names(variant):
    omega = [f'omega_ABL_{abl}' for abl in ABLS] if variant == 'omega_by_abl' else ['omega']
    return [f'gamma_{i:02d}' for i in range(30)] + omega + ['t_E_aff', 'w', 'del_go']


def initial_parameters(conditions, variant):
    assert list(zip(conditions.ABL, conditions.ILD)) == [(a, i) for a in ABLS for i in ILDS]
    if variant == 'omega_by_abl':
        omega = np.array([np.average(z.omega_init, weights=z.trials)
                          for abl in ABLS for z in [conditions.loc[conditions.ABL.eq(abl)]]])
    else:
        omega = np.array(np.average(conditions.omega_init, weights=conditions.trials))
    return dict(gamma=conditions.gamma_init.to_numpy(dtype=float), omega=np.clip(omega, .0101, 14.9999),
                t_E_aff=np.array(.05), w=np.array(.5), del_go=np.array(.08))


def loss_window_decision(mean_loss, step, best_loss, best_step, patience,
                         min_steps=50000, max_steps=1000000, confirmation_windows=25):
    """Pure resumable stopping decision; parameter stability is never a gate."""
    if not np.isfinite(mean_loss):
        raise ValueError('Nonfinite window loss; retain the last complete checkpoint.')
    improved = mean_loss < best_loss
    if improved:
        best_loss, best_step, patience = mean_loss, step, 0
    else:
        patience += 1
    confirmed = step >= min_steps and patience >= confirmation_windows
    status = 'converged' if confirmed else ('inconclusive' if step >= max_steps else 'running')
    return dict(improved=bool(improved), best_loss=float(best_loss), best_step=int(best_step),
                patience=int(patience), status=status)
