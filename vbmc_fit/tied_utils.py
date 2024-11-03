import numpy as np
from numba import jit
import random

def tied_data_gen_wrapper(ILD_arr, ABL_arr, rate_lambda, theta_E, T_0, t_non_decision, t_stim_0, t_stim_scale, num_iter, N_print, dt):
    ABL = random.choice(ABL_arr)
    ILD = random.choice(ILD_arr)

    t_stim = np.random.exponential(t_stim_scale) + t_stim_0

    if num_iter % N_print == 0:
        print(f"Iter # {num_iter}, ILD: {ILD}, ABL: {ABL}, t_stim: {t_stim}")
    
    choice, rt = simulated_tied_ddm_norm(ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, t_stim, dt)
    return {'choice': choice, 'rt': rt, 'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}

@jit
def simulate_tied_no_norm(ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, t_stim, dt):
    DV = 0; t = 0

    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    mu = (2*q_e/T_0) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
    sigma = np.sqrt( (2*(q_e**2)/T_0) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )
    dB = dt**0.5

    while True:
        if t > t_stim:
            DV += mu*dt + sigma*np.random.normal(0, dB)
        t += 1

        if DV >= theta:
            return 1, t*dt + t_non_decision
        elif DV <= -theta:
            return -1, t*dt + t_non_decision
       

def tied_data_gen_wrapper(ILD_arr, ABL_arr, rate_lambda, theta_E, T_0, t_non_decision, dt):
    ABL = random.choice(ABL_arr)
    ILD = random.choice(ILD_arr)
    
    choice, rt = simulated_tied_ddm_norm(ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, dt)
    return {'choice': choice, 'rt': rt, 'ABL': ABL, 'ILD': ILD}

@jit
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
        

def rho_E_minus_small_t_NORM_TIED_fn(t, ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, K_max):
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

def rho_E_plus_small_t_NORM_TIED_fn(t, ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, K_max):
    """
    in normalized time, PDF of hitting the upper bound
    """
    chi = 17.37
    v = theta_E * np.tanh(rate_lambda * ILD / chi)
    w = 0.5
    a = 2
    t -= t_non_decision

    # for hitting upper bound
    v = -v
    w = 1-w

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

def rho_E_small_t_NORM_TIED_fn(t, ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, K_max):
    """
    in normalized time, PDF of any bound
    """
    return rho_E_minus_small_t_NORM_TIED_fn(t, ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, K_max) \
            + rho_E_plus_small_t_NORM_TIED_fn(t, ILD, ABL, rate_lambda, theta_E, T_0, t_non_decision, K_max) 