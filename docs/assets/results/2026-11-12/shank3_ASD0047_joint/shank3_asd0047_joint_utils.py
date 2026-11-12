# %% ASD0047 single-stage NPL+alpha race and contaminant variants
"""Reuse the verified joint likelihood; all animal parameters update together."""
import os
import json
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import shank3_wt_joint_contaminant_utils as reference
from shank3_wt_joint_contaminant_utils import (ROOT, ABLS, ILDS, PROACTIVE, REACTIVE,
    sha256, evidence_grid, loss_window_decision, WINDOW, MIN_STEPS, CONFIRMATION,
    MAX_STEPS, LEARNING_RATE, CLIP_NORM, INIT_SCALE, DRAWS)
from proactive_lapse_no_led_utils import trapezoid_logpdf

# %% Editable animal and variant; previous sources and fit folders remain immutable
ANIMALS = ['ASD0047']
BASE = ROOT / 'asd_analysis/shank3_cohort1_ASD0047_joint_npl_alpha_gaussian_abl_contaminants'
VARIANT = os.environ.get('ASD0047_JOINT_VARIANT', 'mixture')
assert VARIANT in ['mixture', 'exponential_only']
OUT = BASE / VARIANT
OLD = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_individual_adaptive_bounds/wt/ASD0047'
OLD_FIXED = json.loads((OLD / 'fixed_proactive.json').read_text())
OLD_ABORT = ROOT / OLD_FIXED['selected_run']
OLD_REACTIVE = ROOT / json.loads((OLD / 'reactive_selected_run.json').read_text())['selected_run']
CONTAMINANTS = ['lapse_prob', 'lapse_prob_right'] + ([] if VARIANT == 'exponential_only' else ['lapse_exp_weight']) + ['beta_lapse']
BLOCKS = PROACTIVE + REACTIVE + CONTAMINANTS
FIXED_PARAMETERS = {'lapse_exp_weight': 1.} if VARIANT == 'exponential_only' else {}
PARAMETER_COUNT = 19 if FIXED_PARAMETERS else 20
PRIORS = {**json.loads((OLD_ABORT / 'configuration.json').read_text())['priors'],
          **json.loads((OLD_REACTIVE / 'configuration.json').read_text())['priors']}
PRIORS.update({key: reference.PRIORS[key].copy() for key in ['lapse_prob_right', 'lapse_exp_weight']})
PRIORS = {key: PRIORS[key] for key in BLOCKS}


def parameter_names():
    return PROACTIVE + REACTIVE[:7] + [f'{key}_ABL{abl}' for key in REACTIVE[7:] for abl in ABLS] + CONTAMINANTS


def sample_matrix(samples):
    return np.column_stack([np.asarray(samples[key]).reshape(len(samples[key]), -1) for key in BLOCKS])


def initialization(seed):
    if VARIANT == 'exponential_only':
        chosen = json.loads((BASE / 'mixture/selected_fit.json').read_text())
        source = (ROOT / chosen['selected_run']).parent / f'seed_{seed}'
        with np.load(source / 'posterior_samples.npz') as samples:
            return {key: np.asarray(samples[key].mean(0)) for key in BLOCKS}
    if seed == 0:
        with np.load(OLD_ABORT / 'posterior_samples.npz') as samples:
            p = {key: np.asarray(samples[key].mean(0)) for key in samples.files}
        with np.load(OLD_REACTIVE / 'posterior_samples.npz') as samples:
            p.update({key: np.asarray(samples[key].mean(0)) for key in samples.files})
        p.update(lapse_prob_right=np.array(.5), lapse_exp_weight=np.array(.8))
        return p
    return reference.initialization(1)


def contaminant(t, params, cutoff):
    return reference.contaminant(t, {**params, **FIXED_PARAMETERS}, cutoff)


def log_likelihood(params, data, grid_end=1.5):
    return reference.log_likelihood({**params, **FIXED_PARAMETERS}, data, grid_end)


def make_model(priors, grid_end):
    def model(data):
        params = {}
        for key in BLOCKS:
            lo, pl, ph, hi = priors[key]
            base = dist.Uniform(lo, hi)
            if key in ['delay_mu', 'delay_sigma']:
                base = base.expand((3,)).to_event(1)
            value = numpyro.sample(key, base)
            numpyro.factor(key + '_trapezoid', jnp.sum(trapezoid_logpdf(value, (lo, pl, ph, hi)) + jnp.log(hi - lo)))
            params[key] = value
        if data is not None:
            numpyro.factor('all_joint_observations', log_likelihood(params, data, grid_end))
    return model
