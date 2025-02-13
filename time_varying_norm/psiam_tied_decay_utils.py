import numpy as np
import random
import math
from scipy.optimize import brentq
from scipy.special import erf, erfcx



def decay_sigmoid_integral(t, gamma, mu_d, sigma_d, alpha):
    part1 = gamma * (1 - np.exp(-t / gamma))

    log_term_t = np.log(1 + np.exp((t - mu_d) / sigma_d))
    log_term_0 = np.log(1 + np.exp(- mu_d / sigma_d))
    part2 = alpha * sigma_d * (log_term_t - log_term_0)

    return part1 + part2

def compute_tau_from_t(t, omega, decay_params):
    # return omega *( ( 1 - np.exp(-t / c) ) + c1 * t)
    gamma, mu_d, sigma_d, alpha = (decay_params[k] for k in ["gamma", "mu_d", "sigma_d", "alpha"])
    return omega * decay_sigmoid_integral(t, gamma, mu_d, sigma_d, alpha)


def decay_sigmoid(t, gamma, mu_d, sigma_d, alpha):
    return np.exp(-t/gamma) +  (alpha / (1 + np.exp(-(t - mu_d) / sigma_d )))

def tied_abs_units_decay(ILD_arr, ABL_arr, rate_lambda, theta_E, T_0, t_non_decision, dt, decay_params, max_time=100):
    ILD = np.random.choice(ILD_arr)
    ABL = np.random.choice(ABL_arr)


    max_steps = int(np.ceil(max_time / dt))
    t = np.arange(0, max_steps * dt, dt)
    dB = np.sqrt(dt)
    chi = 17.37

    common = (2 / T_0) * (10 ** (rate_lambda * ABL / 20))
    
    # decay = (1/c)*np.exp(-t / c) + c1
    gamma, mu_d, sigma_d, alpha = (decay_params[k] for k in ["gamma", "mu_d", "sigma_d", "alpha"])
    decay = decay_sigmoid(t, gamma, mu_d, sigma_d, alpha)

    mu = common * (rate_lambda * ILD / chi) * decay
    sigma = np.sqrt(common * decay)
    
    noise = np.random.normal(0, dB, size=max_steps)
    
    increments = mu * dt + sigma * noise
    DV = np.cumsum(increments)
    
    crossing_indices = np.where((DV >= theta_E) | (DV <= -theta_E))[0]
    if crossing_indices.size > 0:
        t_cross = t[crossing_indices[0]]
        choice = +1 if DV[crossing_indices[0]] > theta_E else -1
        return {'choice': choice, 'rt': t_cross + t_non_decision, 'ILD': ILD,  'ABL': ABL}
    else:
        return {'choice': None, 'rt': np.nan, 'DV': DV}


def prob_of_hitting_down_in_norm_units(t, ILD, rate_lambda, theta_E, t_non_decision, omega, decay_params, dtau_by_dt):
    chi = 17.37
    v = theta_E * rate_lambda * ILD / chi
    w = 0.5
    a = 2
    t -= t_non_decision

    # t_theta = T_0 * (theta_E**2) * (10**(-rate_lambda*ABL/20)) * (1/(2*np.cosh(rate_lambda*ILD/chi)))
    # t /= t_theta

    # t in normalized units t - > tau
    t = compute_tau_from_t(t, omega, decay_params)

    if t <= 0:
        return 0
    
    K_max = 10
    non_sum_term = (1/a**2)*(a**3/np.sqrt(2*np.pi*t**3))*np.exp(-v*a*w - (v**2 * t)/2)
    K_max = int(K_max/2)
    k_vals = np.linspace(-K_max, K_max, 2*K_max + 1)
    sum_w_term = w + 2*k_vals
    sum_exp_term = np.exp(-(a**2 * (w + 2*k_vals)**2)/(2*t))
    sum_result = np.sum(sum_w_term*sum_exp_term)

    
    density =  non_sum_term * sum_result
    if density <= 0:
        density = 1e-16

    return density * dtau_by_dt

def tied_data_gen_wrapper(ILD_arr, ABL_arr, rate_lambda, theta_E, T_0, t_non_decision, dt):
    ABL = random.choice(ABL_arr)
    ILD = random.choice(ILD_arr)
    
    choice, rt = simulated_tied_ddm_norm(ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, dt)
    return {'choice': choice, 'rt': rt, 'ABL': ABL, 'ILD': ILD}

def simulated_tied_ddm_norm(ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, dt):
    DV = 0; tau = 0; 

    chi = 17.37
    t_theta = T_0 * (theta_E**2) * (10**(-rate_lambda*ABL/20)) * (1/(2*np.cosh(rate_lambda*ILD/chi)))
    d_tau = dt/t_theta
    dB_tau = d_tau**0.5

    drift = theta_E * np.tanh(rate_lambda * ILD / chi)
    
    while True:
        DV += drift*d_tau + np.random.normal(0, dB_tau)
        tau += 1

        # bounds are +1 and -1 as simulation is done in normalized units
        if DV >= 1:
            return 1, (tau*d_tau)*t_theta + t_non_decision
        elif DV <= -1:
            return -1, (tau*d_tau)*t_theta + t_non_decision
        

def rho_E_minus_small_t_NORM_TIED_fn(t, ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, K_max=10):
    """
    in normalized time, PDF of hitting the lower bound
    """
    chi = 17.37
    v = theta_E * np.tanh(rate_lambda * ILD / chi)
    w = 0.5
    a = 2
    t -= t_non_decision

    t_theta = T_0 * (theta_E**2) * (10**(-rate_lambda*ABL/20)) * (1/(2*np.cosh(rate_lambda*ILD/chi)))
    if t <= 0:
        return 0

    t /= t_theta

    
    non_sum_term = (1/a**2)*(a**3/np.sqrt(2*np.pi*t**3))*np.exp(-v*a*w - (v**2 * t)/2)
    K_max = int(K_max/2)
    k_vals = np.linspace(-K_max, K_max, 2*K_max + 1)
    sum_w_term = w + 2*k_vals
    sum_exp_term = np.exp(-(a**2 * (w + 2*k_vals)**2)/(2*t))
    sum_result = np.sum(sum_w_term*sum_exp_term)

    
    density =  non_sum_term * sum_result
    if density <= 0:
        density = 1e-16

    return density/t_theta

def phi(x):
    """Standard Gaussian function."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def M(x):
    """Mills ratio."""
    return np.sqrt(np.pi / 2) * erfcx(x / np.sqrt(2))

def CDF_E_minus_small_t_NORM_fn(t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max):
    """
    In normalized time, CDF of hitting the lower bound.
    """
    if t <= 0:
        return 0
    
    q_e = 1
    theta = theta_E*q_e

    chi = 17.37
    v = theta_E * np.tanh(rate_lambda * ILD / chi)
    w = (Z_E + theta)/(2*theta)
    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    
    t_theta = T_0 * (theta_E**2) * (10**(-rate_lambda*ABL/20)) * (1/(2*np.cosh(rate_lambda*ILD/chi)))
    t /= t_theta


    result = np.exp(-v * a * w - (((v**2) * t) / 2))

    summation = 0
    for k in range(K_max + 1):
        if k % 2 == 0:  # even k
            r_k = k * a + a * w
        else:  # odd k
            r_k = k * a + a * (1 - w)
        
        term1 = phi((r_k) / np.sqrt(t))
        term2 = M((r_k - v * t) / np.sqrt(t)) + M((r_k + v * t) / np.sqrt(t))
        
        summation += ((-1)**k) * term1 * term2

    return (result*summation)
