import numpy as np
import random
import math
from scipy.optimize import brentq

# def compute_t_scalar(tau, omega, c, c1):
#     def f(t):
#         return (omega * ((1 - np.exp(-t / c)) + c1 * t )) - tau
 
#     # We know that f(0) = -tau. If tau > 0, f(0) is negative.
#     # Since the derivative f'(t)= (omega/c)*exp(-t/c) + c1 is positive,
#     # f is monotonic increasing and f(t) will eventually become positive.
#     # We now find a bracket [t_low, t_high] such that f(t_low) < 0 and f(t_high) > 0.
    
#     t_low = 0.0
#     # A first guess for t_high: if c1 is nonzero, start with tau/c1,
#     # otherwise use tau (or a small positive value) to ensure a positive guess.
#     t_high = tau / c1 if c1 != 0 else tau
#     if t_high <= t_low:
#         t_high = t_low + 1.0

#     # If f(t_high) is not positive, increase t_high until it is.
#     while f(t_high) <= 0:
#         t_high *= 2

#     # Now use Brent's method to find the root in [t_low, t_high].
#     t_sol = brentq(f, t_low, t_high)
#     return t_sol

# def compute_t_from_tau(tau, omega, c, c1):
#     """
#     Numerically solve for t in the equation:
#         tau = omega*(1 - exp(-t/c)) + c1*t
#     for tau. The input tau can be a scalar or a NumPy array.
    
#     Returns:
#         t_sol : scalar or NumPy array of solutions.
#     """
#     # Ensure tau is treated as an array.
#     tau_array = np.atleast_1d(tau)
    
#     # Compute solution for each element.
#     t_solutions = np.array([compute_t_scalar(t_i, omega, c, c1) for t_i in tau_array])
    
#     # If the original tau was a scalar, return a scalar.
#     return t_solutions[0] if t_solutions.size == 1 else t_solutions




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
