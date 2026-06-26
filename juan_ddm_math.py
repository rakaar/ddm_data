"""Small math helpers for the 2AFC DDM teaching script.

The main workflow is in fit_teaching_2afc_ddm.py. This file keeps the
analytical two-bound first-passage-time likelihood in one place.

The first-passage density uses the usual small-time and large-time analytical
series for the Wiener diffusion model; see Navarro and Fuss (2009) and the
faster truncation discussion in Gondan, Blurton, and Kesselmeier (2014).
"""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp
import numpyro


LARGE_TIME_TERMS = 80
SMALL_TIME_TERMS = 20
SMALL_TIME_SWITCH = 0.12


def _series_indices(n_terms, ndim, dtype):
    shape = (int(n_terms),) + (1,) * int(ndim)
    return jnp.arange(1, int(n_terms) + 1, dtype=dtype).reshape(shape)


def _image_indices(n_terms, ndim, dtype):
    shape = (2 * int(n_terms) + 1,) + (1,) * int(ndim)
    return jnp.arange(-int(n_terms), int(n_terms) + 1, dtype=dtype).reshape(shape)


def _lower_bound_large_time_pdf(decision_time, drift, bound, start_fraction, n_terms):
    safe_time = jnp.maximum(decision_time, 1e-10)
    safe_bound = jnp.maximum(bound, 1e-4)
    start = start_fraction * safe_bound

    ndim = max(safe_time.ndim, jnp.ndim(drift), jnp.ndim(safe_bound), jnp.ndim(start_fraction))
    k = _series_indices(n_terms, ndim, safe_time.dtype)
    pi = jnp.asarray(jnp.pi, dtype=safe_time.dtype)

    series_terms = (
        k
        * jnp.sin(k * pi * start_fraction)
        * jnp.exp(-((k * pi) ** 2) * safe_time / (2.0 * safe_bound**2))
    )
    series_sum = jnp.sum(series_terms, axis=0)

    log_prefactor = (
        jnp.log(pi)
        - 2.0 * jnp.log(safe_bound)
        - drift * start
        - 0.5 * drift**2 * safe_time
    )
    return jnp.exp(log_prefactor) * series_sum


def _lower_bound_small_time_pdf(decision_time, drift, bound, start_fraction, n_terms):
    safe_time = jnp.maximum(decision_time, 1e-10)
    safe_bound = jnp.maximum(bound, 1e-4)
    start = start_fraction * safe_bound

    ndim = max(safe_time.ndim, jnp.ndim(drift), jnp.ndim(safe_bound), jnp.ndim(start_fraction))
    k = _image_indices(n_terms, ndim, safe_time.dtype)
    image_distance = start + 2.0 * k * safe_bound
    series_terms = image_distance * jnp.exp(-(image_distance**2) / (2.0 * safe_time))
    series_sum = jnp.sum(series_terms, axis=0)

    log_prefactor = (
        -0.5 * jnp.log(2.0 * jnp.pi * safe_time**3)
        - drift * start
        - 0.5 * drift**2 * safe_time
    )
    return jnp.exp(log_prefactor) * series_sum


def two_bound_survival_probability(
    decision_time,
    drift,
    bound,
    start_fraction=0.5,
    n_terms=LARGE_TIME_TERMS,
):
    """Probability that neither bound has been hit by `decision_time`.

    This is the total survival probability needed for right-censored RTs:

        P(no response by t) = 1 - P(upper hit by t) - P(lower hit by t)

    Instead of computing two separate CDFs and subtracting, this uses the
    closed-form sine series obtained by analytically integrating the
    killed-diffusion transition density over the interval between the bounds.
    No numerical quadrature is used during fitting.
    """

    safe_time = jnp.maximum(decision_time, 1e-10)
    safe_bound = jnp.maximum(bound, 1e-4)
    start_fraction = jnp.clip(start_fraction, 1e-4, 1.0 - 1e-4)
    start = start_fraction * safe_bound

    ndim = max(safe_time.ndim, jnp.ndim(drift), jnp.ndim(safe_bound), jnp.ndim(start_fraction))
    k_index = _series_indices(n_terms, ndim, safe_time.dtype)
    pi = jnp.asarray(jnp.pi, dtype=safe_time.dtype)
    wave_number = k_index * pi / safe_bound
    parity = jnp.where((k_index.astype(jnp.int32) % 2) == 0, 1.0, -1.0)

    # Closed-form integral of exp(drift * x) * sin(k*pi*x/bound), from
    # x=0 to x=bound. This is an analytical series term, not quadrature.
    # This turns the transition-density series into a no-bound-hit survival.
    integral = (
        wave_number
        * (1.0 - parity * jnp.exp(drift * safe_bound))
        / (drift**2 + wave_number**2)
    )
    series_terms = (
        jnp.sin(k_index * pi * start_fraction)
        * jnp.exp(-(wave_number**2) * safe_time / 2.0)
        * integral
    )
    survival = (
        (2.0 / safe_bound)
        * jnp.exp(-drift * start - 0.5 * drift**2 * safe_time)
        * jnp.sum(series_terms, axis=0)
    )
    return jnp.where(decision_time > 0, jnp.clip(survival, 1e-10, 1.0), 1.0)


def lower_bound_log_pdf(
    decision_time,
    drift,
    bound,
    start_fraction=0.5,
    large_time_terms=LARGE_TIME_TERMS,
    small_time_terms=SMALL_TIME_TERMS,
):
    """Log density for hitting the lower bound of a two-bound DDM.

    The process has unit diffusion noise, lower bound 0, upper bound `bound`,
    starting point `start_fraction * bound`, and constant drift `drift`.

    Two analytical series are used: an image series for small times and a
    sine series for larger times. This follows the practical idea used by
    Navarro and Fuss (2009) and refined by Gondan, Blurton, and Kesselmeier
    (2014): use the representation that is numerically well behaved for the
    part of the time range being evaluated.
    """

    safe_time = jnp.maximum(decision_time, 1e-10)
    safe_bound = jnp.maximum(bound, 1e-4)
    start_fraction = jnp.clip(start_fraction, 1e-4, 1.0 - 1e-4)
    scaled_time = safe_time / (safe_bound**2)

    small_time_pdf = _lower_bound_small_time_pdf(
        safe_time,
        drift,
        safe_bound,
        start_fraction,
        n_terms=small_time_terms,
    )
    large_time_pdf = _lower_bound_large_time_pdf(
        safe_time,
        drift,
        safe_bound,
        start_fraction,
        n_terms=large_time_terms,
    )
    density = jnp.where(scaled_time < SMALL_TIME_SWITCH, small_time_pdf, large_time_pdf)

    return jnp.where(
        decision_time > 0,
        jnp.log(jnp.maximum(density, 1e-30)),
        -1e10,
    )


def choice_rt_log_pdf(decision_time, choice, drift, bound, start_fraction=0.5):
    """Log density for a choice and RT under an unbiased two-bound DDM.

    `choice == 1` means the upper/right bound was hit.
    `choice == -1` means the lower/left bound was hit.

    The upper-bound density is computed by reflecting the process: hitting the
    upper bound with drift `v` is equivalent to hitting the lower bound with
    drift `-v` and starting point `1 - start_fraction`.
    """

    is_upper_choice = choice > 0
    reflected_drift = jnp.where(is_upper_choice, -drift, drift)
    reflected_start = jnp.where(is_upper_choice, 1.0 - start_fraction, start_fraction)
    return lower_bound_log_pdf(
        decision_time,
        reflected_drift,
        bound,
        reflected_start,
    )


def upper_choice_probability(drift, bound):
    """Probability of hitting the upper bound for an unbiased DDM."""

    return jnn.sigmoid(drift * bound)


def add_2afc_ddm_log_likelihood(rts, choices, drift, bound, nondecision_time, t_min, t_max):
    n_trials = rts.shape[0]

    with numpyro.plate("trials", n_trials):
        is_censored = (rts < 0) | (rts >= t_max)
        decision_time = rts - nondecision_time
        trunc_after_ndt = jnp.maximum(t_min - nondecision_time, 1e-10)
        max_after_ndt = jnp.maximum(t_max - nondecision_time, 1e-10)

        survival_after_trunc = two_bound_survival_probability(trunc_after_ndt, drift, bound)
        survival_after_max = two_bound_survival_probability(max_after_ndt, drift, bound)

        log_observed = (
            choice_rt_log_pdf(decision_time, choices, drift, bound)
            - jnp.log(survival_after_trunc)
        )
        log_censored = (
            jnp.log(survival_after_max)
            - jnp.log(survival_after_trunc)
        )

        numpyro.factor("choice_rt", jnp.where(is_censored, log_censored, log_observed))
