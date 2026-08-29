# %%
"""Independent simulator for the LED step-jump and NPL+alpha race."""

# %%
from pathlib import Path
import sys

from numba import njit
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ANIMAL_FIT_DIR = REPO_DIR / "fit_animal_by_animal"
if str(ANIMAL_FIT_DIR) not in sys.path:
    sys.path.insert(0, str(ANIMAL_FIT_DIR))

from time_vary_norm_alpha_utils import gamma_omega_alpha_fn


# %%
@njit(cache=True)
def _sample_inverse_gaussian_first_passage(drift, bound):
    """Exact first-passage draw for Brownian motion with constant drift."""
    mean = bound / drift
    shape = bound * bound
    normal_draw = np.random.normal()
    squared_normal = normal_draw * normal_draw
    candidate = (
        mean
        + mean * mean * squared_normal / (2.0 * shape)
        - mean
        / (2.0 * shape)
        * np.sqrt(
            4.0 * mean * shape * squared_normal
            + mean * mean * squared_normal * squared_normal
        )
    )
    if np.random.random() <= mean / (mean + candidate):
        return candidate
    return mean * mean / candidate


@njit(cache=True)
def _sample_step_jump_action_hit(
    led_on,
    t_LED,
    V_A_base,
    V_A_post_LED,
    theta_A,
    del_a_minus_del_LED,
    del_m_plus_del_LED,
    dt,
    max_steps,
):
    """Return the observed fixation-time proactive response."""
    change_time = t_LED - del_a_minus_del_LED
    if not led_on or abs(V_A_post_LED - V_A_base) < 1e-14:
        latent_hit = _sample_inverse_gaussian_first_passage(
            V_A_base, theta_A
        )
        return latent_hit + del_a_minus_del_LED + del_m_plus_del_LED
    if change_time <= 0.0:
        latent_hit = _sample_inverse_gaussian_first_passage(
            V_A_post_LED, theta_A
        )
        return latent_hit + del_a_minus_del_LED + del_m_plus_del_LED

    accumulator = 0.0
    elapsed = 0.0
    sqrt_dt = np.sqrt(dt)
    for _ in range(max_steps):
        drift = V_A_base if elapsed < change_time else V_A_post_LED
        accumulator += drift * dt + sqrt_dt * np.random.normal()
        elapsed += dt
        if accumulator >= theta_A:
            return elapsed + del_a_minus_del_LED + del_m_plus_del_LED
    return np.nan


@njit(cache=True)
def _simulate_trials_numba(
    t_stim,
    t_LED,
    led_on,
    V_A_base,
    V_A_post_LED,
    theta_A,
    del_a_minus_del_LED,
    del_m_plus_del_LED,
    rate_lambda,
    T_0,
    theta_E,
    w,
    del_go,
    rate_norm_l,
    alpha,
    ABL,
    ILD,
    t_E_aff,
    lapse_prob,
    beta_lapse,
    include_lapse,
    dt,
    max_time,
    seed,
):
    np.random.seed(seed)
    n_trials = len(t_stim)
    choices = np.empty(n_trials, dtype=np.int8)
    total_fix = np.empty(n_trials, dtype=np.float64)
    source = np.empty(n_trials, dtype=np.int8)

    chi = 17.37
    abl_term = 10.0 ** (
        rate_lambda * (1.0 - rate_norm_l) * ABL / 20.0
    )
    ild_arg = rate_lambda * ILD / chi
    norm_ild_arg = rate_lambda * rate_norm_l * ILD / chi
    r_r = abl_term * np.exp(ild_arg) / (
        np.exp(norm_ild_arg) + alpha * np.exp(-norm_ild_arg)
    )
    r_l = abl_term * np.exp(-ild_arg) / (
        np.exp(-norm_ild_arg) + alpha * np.exp(norm_ild_arg)
    )
    r_sum = r_r + r_l
    gamma = theta_E * (r_r - r_l) / r_sum
    omega = r_sum / (T_0 * theta_E * theta_E)
    evidence_mu = gamma * omega * theta_E
    evidence_sigma = np.sqrt(omega) * theta_E
    z_e = (w - 0.5) * 2.0 * theta_E
    sqrt_dt = np.sqrt(dt)
    max_steps = int(np.ceil(max_time / dt))

    for trial_index in range(n_trials):
        if include_lapse and np.random.random() < lapse_prob:
            total_fix[trial_index] = np.random.exponential(
                1.0 / beta_lapse
            )
            choices[trial_index] = (
                1 if np.random.random() >= 0.5 else -1
            )
            source[trial_index] = 3
            continue

        action_time = _sample_step_jump_action_hit(
            led_on,
            t_LED[trial_index],
            V_A_base,
            V_A_post_LED,
            theta_A,
            del_a_minus_del_LED,
            del_m_plus_del_LED,
            dt,
            max_steps,
        )
        if not np.isfinite(action_time):
            total_fix[trial_index] = np.nan
            choices[trial_index] = 0
            source[trial_index] = -1
            continue

        evidence_value = z_e
        evidence_time = t_stim[trial_index] + t_E_aff
        evidence_deadline = action_time + del_go
        evidence_hit = 0
        hit_time = np.nan

        if evidence_time < evidence_deadline:
            for _ in range(max_steps):
                evidence_value += (
                    evidence_mu * dt
                    + evidence_sigma * sqrt_dt * np.random.normal()
                )
                evidence_time += dt
                if evidence_value >= theta_E:
                    evidence_hit = 1
                    hit_time = evidence_time
                    break
                if evidence_value <= -theta_E:
                    evidence_hit = -1
                    hit_time = evidence_time
                    break
                if evidence_time >= evidence_deadline:
                    break

        if evidence_hit != 0 and hit_time <= action_time:
            choices[trial_index] = evidence_hit
            total_fix[trial_index] = hit_time
            source[trial_index] = 0
        elif evidence_hit != 0:
            choices[trial_index] = evidence_hit
            total_fix[trial_index] = action_time
            source[trial_index] = 2
        else:
            choices[trial_index] = (
                1 if np.random.random() >= 0.5 else -1
            )
            total_fix[trial_index] = action_time
            source[trial_index] = 1

    return choices, total_fix, source


# %%
def simulate_led_category(
    params,
    ABL,
    ILD,
    t_E_aff,
    t_stim,
    t_LED,
    led_on,
    n_trials,
    include_lapse,
    dt=0.0001,
    max_time=60.0,
    seed=20260828,
):
    """Simulate one LED category while keeping each timing row intact."""
    t_stim = np.asarray(t_stim, dtype=np.float64)
    t_LED = np.asarray(t_LED, dtype=np.float64)
    if t_stim.ndim == 0:
        t_stim = np.full(n_trials, float(t_stim), dtype=np.float64)
    if t_LED.ndim == 0:
        t_LED = np.full(n_trials, float(t_LED), dtype=np.float64)
    if len(t_stim) != n_trials or len(t_LED) != n_trials:
        raise ValueError("t_stim and t_LED must have one value per trial.")

    choices, total_fix, source = _simulate_trials_numba(
        t_stim,
        t_LED,
        bool(led_on),
        float(params["V_A_base"]),
        float(params["V_A_post_LED"]),
        float(params["theta_A"]),
        float(params["del_a_minus_del_LED"]),
        float(params["del_m_plus_del_LED"]),
        float(params["rate_lambda"]),
        float(params["T_0"]),
        float(params["theta_E"]),
        float(params["w"]),
        float(params["del_go"]),
        float(params["rate_norm_l"]),
        float(params["alpha"]),
        float(ABL),
        float(ILD),
        float(t_E_aff),
        float(params.get("lapse_prob", 0.0)),
        float(params.get("beta_lapse", 1.0)),
        bool(include_lapse),
        float(dt),
        float(max_time),
        int(seed),
    )
    if not np.all(np.isfinite(total_fix)):
        raise RuntimeError("Simulator failed to produce a finite response time.")

    gamma, omega = gamma_omega_alpha_fn(
        ABL,
        ILD,
        params["rate_lambda"],
        params["T_0"],
        params["theta_E"],
        params["rate_norm_l"],
        params["alpha"],
        True,
    )
    return {
        "choice": choices,
        "total_fix": total_fix,
        "rt_stim": total_fix - t_stim,
        "source": source,
        "t_stim": t_stim,
        "t_LED": t_LED,
        "gamma": float(gamma),
        "omega": float(omega),
        "seed": int(seed),
        "dt": float(dt),
    }
