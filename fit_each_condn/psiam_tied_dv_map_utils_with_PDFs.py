import numpy as np
from scipy.special import erf, erfcx
from psiam_tied_no_dv_map_utils import rho_A_t_fn, cum_A_t_fn, rho_E_minus_small_t_NORM_fn,\
                 CDF_E_minus_small_t_NORM_fn, P_small_t_btn_x1_x2

import random
from scipy.integrate import quad
from scipy.integrate import trapezoid as trapz

# simulation part
import math

def all_RTs_fit_OPTIM_omega_gamma_PA_CA_wrt_stim_fn(t, P_A, C_A, gamma, omega, t_E_aff, K_max):
    """
    PDF of all RTs array irrespective of choice
    t is wrt stim
    """

    C_E = CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_E_aff, gamma, omega, 1, K_max) \
           + CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_E_aff, gamma, omega, -1, K_max)
    

    P_E = rho_E_minus_small_t_NORM_omega_gamma_fn(t-t_E_aff, gamma, omega, 1, K_max) \
           + rho_E_minus_small_t_NORM_omega_gamma_fn(t-t_E_aff, gamma, omega, -1, K_max)
          

    P_A = np.array(P_A); C_E = np.array(C_E); P_E = np.array(P_E); C_A = np.array(C_A)
    P_all = P_A*(1-C_E) + P_E*(1-C_A)

    return P_all


def d_A_RT(a, t):
    """
    Calculate the standard PA probability density function.

    Parameters:
        a (float): Scalar parameter.
        t (float): Time value (must be > 0).

    Returns:
        float: The computed pdf value (0 if t <= 0).
    """
    if t <= 0:
        return 0.0
    p = (1.0 / math.sqrt(2 * math.pi * (t**3)) )* math.exp(-((1 - a * t)**2) / (2 * t))
    return p

def stupid_f_integral(v, vON, theta, t, tp):
    """
    Calculate the PA pdf after the v_A change via an integral expression.

    Parameters:
        v (float): Scalar parameter.
        vON (float): Scalar parameter.
        theta (float): Scalar parameter.
        t (float): Time value.
        tp (float): A shifted time value.

    Returns:
        float: The evaluated integral expression.
    """
    a1 = 0.5 * (1 / t + 1 / tp)
    b1 = theta / t + (v - vON)
    c1 = -0.5 * (vON**2 * t - 2 * theta * vON + theta**2 / t + v**2 * tp)

    a2 = a1
    b2 = theta * (1 / t + 2 / tp) + (v - vON)
    c2 = -0.5 * (vON**2 * t - 2 * theta * vON + theta**2 / t + v**2 * tp + 4 * theta * v + 4 * theta**2 / tp) + 2 * v * theta

    F01 = 1.0 / (4 * math.pi * a1 * math.sqrt(tp * t**3))
    F02 = 1.0 / (4 * math.pi * a2 * math.sqrt(tp * t**3))

    T11 = b1**2 / (4 * a1)
    T12 = (2 * a1 * theta - b1) / (2 * math.sqrt(a1))
    T13 = theta * (b1 - theta * a1)

    T21 = b2**2 / (4 * a2)
    T22 = (2 * a2 * theta - b2) / (2 * math.sqrt(a2))
    T23 = theta * (b2 - theta * a2)

    I1 = F01 * (T12 * math.sqrt(math.pi) * math.exp(T11 + c1) * (math.erf(T12) + 1) + math.exp(T13 + c1))
    I2 = F02 * (T22 * math.sqrt(math.pi) * math.exp(T21 + c2) * (math.erf(T22) + 1) + math.exp(T23 + c2))

    STF = I1 - I2
    return STF

def PA_with_LEDON_2(t, v, vON, a, tfix, tled, delta_A):
    """
    Compute the PA pdf by combining contributions before and after LED onset.

    Parameters:
        t (float): Time value.
        v (float): Drift parameter before LED.
        vON (float): Drift parameter after LED onset.
        a (float): Decision bound.
        tfix (float): Fixation time.
        tled (float): LED time.
        delta_A (float): Delta parameter.

    Returns:
        float: The combined PA pdf value.
    """
    # For a scalar, we choose one branch based on the condition.
    if (t + tfix) <= tled + 1e-6:
        # Before LED onset:
        return d_A_RT(v * a, (t - delta_A + tfix) / (a**2)) / (a**2)
    else:
        # After LED onset:
        return stupid_f_integral(v, vON, a, t + tfix - tled, tled - delta_A + tfix) 


def psiam_tied_data_gen_wrapper_V2(V_A, theta_A, ABL_arr, ILD_arr, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, t_motor, L, \
                                t_stim, iter_num, N_print, dt):
    ABL = random.choice(ABL_arr)
    ILD = random.choice(ILD_arr)
    
    # print after every N_print iterations
    # if iter_num % N_print == 0:
    #     print(f'In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')


    choice, rt, is_act = simulate_psiam_tied(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, dt)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}


def simulate_psiam_tied(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, dt):
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5
    
    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    mu = (2*q_e/T_0) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
    sigma = np.sqrt( (2*(q_e**2)/T_0) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )
    
    is_act = 0
    while True:
        if t > t_stim + t_E_aff:
            DV += mu*dt + sigma*np.random.normal(0, dB)
        
        AI += V_A*dt + np.random.normal(0, dB)
        
        t += dt
        
        if DV >= theta:
            choice = +1; RT = t + t_motor
            break
        elif DV <= -theta:
            choice = -1; RT = t + t_motor
            break
        
        if AI >= theta_A:
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + t_E_aff + t_motor):#  u can process evidence till stim plays
                if t > t_stim + t_E_aff: # Evid accum wil begin only after stim starts and afferent delay
                    DV += mu*dt + sigma*np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        break
                    elif DV <= -theta:
                        DV = -theta
                        break
                t += dt
            
            break
        
        
    if is_act == 1:
        RT = AI_hit_time + t_motor
        # if DV != 0:
        if DV >= (1 + (L/2) - 1)*theta:
            choice = 1
        elif DV <= (1 - (L/2) - 1)*theta:
            choice = -1
        else:
            prob_hit_up = (1/L)*((DV/theta) + 1) + (0.5 - (1/L))            
            if np.random.rand() <= prob_hit_up:
                choice = 1
            else:
                choice = -1
             
    
    return choice, RT, is_act

def psiam_tied_data_gen_wrapper_V4(V_A, theta_A, ABL_arr, ILD_arr, rate_lambda, T_0, T_0_tau, theta_E, Z_E, t_A_aff, t_E_aff, t_motor, L, \
                                t_stim, iter_num, N_print, dt):
    ABL = random.choice(ABL_arr)
    ILD = random.choice(ILD_arr)
    
    # print after every N_print iterations
    if iter_num % N_print == 0:
        print(f'In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')


    choice, rt, is_act = simulate_psiam_tied_4(V_A, theta_A, ABL, ILD, rate_lambda, T_0, T_0_tau, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, dt)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}

def calc_T_0_t(t, t_stim, t_E_aff, T_0, T_0_tau):
    Nr0 = 1 / T_0
    Nr_t = (Nr0*np.exp(-(t - t_stim - t_E_aff)/T_0_tau))
    # Nr_t = max(Nr_t, 1e-10)
    if Nr_t == 0:
        Nr_t = 1e-10
    T_0_t = 1/Nr_t
    return T_0_t


def simulate_psiam_tied_4(V_A, theta_A, ABL, ILD, rate_lambda, T_0, T_0_tau, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, dt):
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5
    
    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    
    
    is_act = 0
    while True:
        if t > t_stim + t_E_aff:
            T_0_t = calc_T_0_t(t, t_stim, t_E_aff, T_0, T_0_tau)
            
            mu = (2*q_e/T_0_t) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
            sigma = np.sqrt( (2*(q_e**2)/T_0_t) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )

            DV += mu * dt + sigma * np.random.normal(0, dB)

        
        AI += V_A*dt + np.random.normal(0, dB)
        
        t += dt
        
        if DV >= theta:
            choice = +1; RT = t + t_motor
            break
        elif DV <= -theta:
            choice = -1; RT = t + t_motor
            break
        
        if AI >= theta_A:
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + t_E_aff + t_motor):#  u can process evidence till stim plays
                if t > t_stim + t_E_aff: # Evid accum wil begin only after stim starts and afferent delay
                    T_0_t = calc_T_0_t(t, t_stim, t_E_aff, T_0, T_0_tau)
                    
                    mu = (2*q_e/T_0_t) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
                    sigma = np.sqrt( (2*(q_e**2)/T_0_t) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )

                    DV += mu*dt + sigma*np.random.normal(0, dB)

                    if DV >= theta:
                        DV = theta
                        break
                    elif DV <= -theta:
                        DV = -theta
                        break
                t += dt
            
            break
        
        
    if is_act == 1:
        RT = AI_hit_time + t_motor
        # if DV != 0:
        if DV >= (1 + (L/2) - 1)*theta:
            choice = 1
        elif DV <= (1 - (L/2) - 1)*theta:
            choice = -1
        else:
            prob_hit_up = (1/L)*((DV/theta) + 1) + (0.5 - (1/L))            
            if np.random.rand() <= prob_hit_up:
                choice = 1
            else:
                choice = -1
             
    
    return choice, RT, is_act


def psiam_tied_data_gen_wrapper_V3(V_A, theta_A, ABL_arr, ILD_arr, rate_lambda, T_0_old, T_0_new, theta_E, Z_E, t_A_aff, t_E_aff, t_motor, L, \
                                t_stim, iter_num, N_print, dt):
    ABL = random.choice(ABL_arr)
    ILD = random.choice(ILD_arr)
    
    # print after every N_print iterations
    if iter_num % N_print == 0:
        print(f'In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')


    choice, rt, is_act = simulate_psiam_tied_T0_change(V_A, theta_A, ABL, ILD, rate_lambda, T_0_old, T_0_new, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, dt)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}


def simulate_psiam_tied_T0_change(V_A, theta_A, ABL, ILD, rate_lambda, T_0_old, T_0_new, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, dt):
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5
    
    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    
    
    is_act = 0
    while True:
        if t - t_stim < 0.21:
            T_0 = T_0_old
        else:
            T_0 = T_0_new
        
        mu = (2*q_e/T_0) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
        sigma = np.sqrt( (2*(q_e**2)/T_0) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )

        if t > t_stim + t_E_aff:
            DV += mu*dt + sigma*np.random.normal(0, dB)
        
        AI += V_A*dt + np.random.normal(0, dB)
        
        t += dt
        
        if DV >= theta:
            choice = +1; RT = t + t_motor
            break
        elif DV <= -theta:
            choice = -1; RT = t + t_motor
            break
        
        if AI >= theta_A:
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + t_E_aff + t_motor):#  u can process evidence till stim plays
                if t > t_stim + t_E_aff: # Evid accum wil begin only after stim starts and afferent delay
                    DV += mu*dt + sigma*np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        break
                    elif DV <= -theta:
                        DV = -theta
                        break
                t += dt
            
            break
        
        
    if is_act == 1:
        RT = AI_hit_time + t_motor
        # if DV != 0:
        if DV >= (1 + (L/2) - 1)*theta:
            choice = 1
        elif DV <= (1 - (L/2) - 1)*theta:
            choice = -1
        else:
            prob_hit_up = (1/L)*((DV/theta) + 1) + (0.5 - (1/L))            
            if np.random.rand() <= prob_hit_up:
                choice = 1
            else:
                choice = -1
             
    
    return choice, RT, is_act



def psiam_tied_data_gen_wrapper(V_A, theta_A, ABL_arr, ILD_arr, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, t_motor, L, \
                                t_stim_0, t_stim_tau, iter_num, N_print, dt):
    ABL = random.choice(ABL_arr)
    ILD = random.choice(ILD_arr)
    
    # t_stim is picked from a distribution
    t_stim = np.random.exponential(t_stim_tau) + t_stim_0

    # print after every N_print iterations
    if iter_num % N_print == 0:
        print(f'In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')


    choice, rt, is_act = simulate_psiam_tied(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, dt)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}

def simulate_psiam_tied(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, dt):
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5
    
    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    mu = (2*q_e/T_0) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
    sigma = np.sqrt( (2*(q_e**2)/T_0) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )
    
    is_act = 0
    while True:
        if t > t_stim + t_E_aff:
            DV += mu*dt + sigma*np.random.normal(0, dB)
        
        AI += V_A*dt + np.random.normal(0, dB)
        
        t += dt
        
        if DV >= theta:
            choice = +1; RT = t + t_motor
            break
        elif DV <= -theta:
            choice = -1; RT = t + t_motor
            break
        
        if AI >= theta_A:
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + t_E_aff + t_motor):#  u can process evidence till stim plays
                if t > t_stim + t_E_aff: # Evid accum wil begin only after stim starts and afferent delay
                    DV += mu*dt + sigma*np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        break
                    elif DV <= -theta:
                        DV = -theta
                        break
                t += dt
            
            break
        
        
    if is_act == 1:
        RT = AI_hit_time + t_motor
        # if DV != 0:
        if DV >= (1 + (L/2) - 1)*theta:
            choice = 1
        elif DV <= (1 - (L/2) - 1)*theta:
            choice = -1
        else:
            prob_hit_up = (1/L)*((DV/theta) + 1) + (0.5 - (1/L))            
            if np.random.rand() <= prob_hit_up:
                choice = 1
            else:
                choice = -1
             
    
    return choice, RT, is_act



# PDF funcs
def integral_of_exp_term_times_x(x, A, B):
    """
    Integral of exp(-(x+A)^2 / B) * x * dx 
    """
    return (-B/2)*np.exp( -((x + A)**2)/B ) - (A*np.sqrt(np.pi*B)/2)*erf( (x + A)/np.sqrt(B) )

def integral_of_exp_term(x, A, B):
    """
    Integral of exp(-(x+A)^2 / B) * dx
    """
    return (np.sqrt(np.pi*B)/2) * erf( (x + A)/np.sqrt(B) )

def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))



def prob_x_t_and_hit_up_or_down_analytic(t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, L, N_max, bound):
    """
    Given x and t, return the probability of hitting the upper bound
    """

    if t <= 0:
        return 0
    
    q_e = 1
    theta = theta_E*q_e

    chi = 17.37
    mu = theta_E * np.tanh(rate_lambda * ILD / chi)
    z = (Z_E/theta) + 1.0
    
    t_theta = T_0 * (theta_E**2) * (10**(-rate_lambda*ABL/20)) * (1/(2*np.cosh(rate_lambda*ILD/chi)))
    t /= t_theta


    term1 = 1/np.sqrt(2*np.pi*t)
    n_terms = np.linspace(-N_max, N_max, 2*N_max+1)
    
    if bound == 1:
        m = 1/L
        c = 0.5 - 1/L
    elif bound == -1:
        m = -1/L
        c = 0.5 + 1/L
        
    sum_term = 0
    B = 2*t

    for n in n_terms:
        exp1 = np.exp(4*mu*n)
        A1 = -z - 4*n - mu*t

        exp2 = np.exp(2*mu*(2 - 2*n - z))
        A2 = z - 4*(1-n) - mu*t


        # exp_term * exp((x + A)**2 / B ) * mx +  exp_term * exp((x + A)**2 / B ) * c
        sum_term += exp1*m*integral_of_exp_term_times_x(1+(L/2), A1, B) + exp1*c*integral_of_exp_term(1+(L/2), A1, B) \
                    - exp2*m*integral_of_exp_term_times_x(1+(L/2), A2, B) - exp2*c*integral_of_exp_term(1+(L/2), A2, B) \
                    - (
                        exp1*m*integral_of_exp_term_times_x(1-(L/2), A1, B) + exp1*c*integral_of_exp_term(1-(L/2), A1, B) \
                        - exp2*m*integral_of_exp_term_times_x(1-(L/2), A2, B) - exp2*c*integral_of_exp_term(1-(L/2), A2, B)

                        )
        
    
    return term1*sum_term

def CDF_RT_fn(t, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, K_max):
    """
    CDF of RT, no choice
    """
    C_A = cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) 
    C_E = CDF_E_minus_small_t_NORM_fn(t - t_motor - t_stim - t_E_aff, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
           + CDF_E_minus_small_t_NORM_fn(t - t_motor - t_stim - t_E_aff, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max)

    return C_A + C_E - C_A*C_E
    

def all_RTs_fit_OPTIM_fn(t, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, K_max):
    """
    PDF of all RTs array irrespective of choice
    """


    P_A = rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) 
    C_E = CDF_E_minus_small_t_NORM_fn(t - t_motor - t_stim - t_E_aff, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
           + CDF_E_minus_small_t_NORM_fn(t - t_motor - t_stim - t_E_aff, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max)
    

    P_E = rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
           + rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max) 
          
    C_A = cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) 

    P_A = np.array(P_A); C_E = np.array(C_E); P_E = np.array(P_E); C_A = np.array(C_A)
    P_all = P_A*(1-C_E) + P_E*(1-C_A)

    return P_all

#### rho E gamma and omega and w - starting point ###########
def rho_E_minus_small_t_NORM_omega_gamma_w_fn(t, gamma, omega, bound, w, K_max):
    """
    in normalized time, added noise to variance of firing rates to PDF of hitting the lower bound
    """
    if t <= 0:
        return 0

    # evidence v
    v = gamma

    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    t_theta = 1 / omega
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

def CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t, gamma, omega, bound, w, K_max):
    """
    In normalized time, CDF of hitting the lower bound.
    """
    if t <= 0:
        return 0

    # evidence v
    v = gamma

    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    t_theta = 1 / omega
    t /= t_theta

    # Compute the exponent argument separately
    exponent_arg = -v * a * w - (((v**2) * t) / 2)

    # # Define safe thresholds for float64
    # max_safe_exp = np.log(np.finfo(np.float64).max)  # ~709
    # min_safe_exp = np.log(np.finfo(np.float64).tiny)   # very negative number

    # # if exponent_arg not within range print each value
    # if exponent_arg > max_safe_exp or exponent_arg < min_safe_exp:
    #     print(f'lambda = {rate_lambda}, T0 = {T_0}, theta_E = {theta_E}, Z_E = {Z_E}, ABL = {ABL}, ILD = {ILD}')
    #     print(f'v = {v}, a = {a}, w = {w}, t = {t}, exponent_arg = {exponent_arg}')


    # # Clip the exponent argument between the safe minimum and maximum
    # exponent_arg_clipped = np.clip(exponent_arg, min_safe_exp, max_safe_exp)

    # Now compute the result using the clipped exponent
    result = np.exp(exponent_arg)


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

def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_w_fn(t, t_LED, V_A, V_A_post_LED, theta_A, gamma, omega, t_stim, t_A_aff, t_E_aff, del_go, w, bound, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    P_A = PA_with_LEDON_2(t, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff)
    #CDF_E_minus_small_t_NORM_omega_gamma_fn
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, 1, w, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, -1, w, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t2, gamma, omega, bound, w, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t1, gamma, omega, bound, w, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_w_fn(t-t_E_aff-t_stim, gamma, omega, bound, w, K_max)
    
    t_pts = np.arange(0, t, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    C_A = trapz(P_A_LED_change, t_pts)

    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up
def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_w_custom_m3_fn(t, t_LED, V_A, V_A_post_LED, theta_A, gamma, omega, t_stim, t_A_aff, t_E_aff, del_go, w, bound, m3_up_prob, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    P_A = PA_with_LEDON_2(t, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff)
    #CDF_E_minus_small_t_NORM_omega_gamma_fn
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, 1, w, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, -1, w, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    if bound == 1:
        random_choice_prob = m3_up_prob
    else:
        random_choice_prob = 1 - m3_up_prob
    random_readout_if_EA_surives =  random_choice_prob * prob_EA_survives
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t2, gamma, omega, bound, w, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t1, gamma, omega, bound, w, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_w_fn(t-t_E_aff-t_stim, gamma, omega, bound, w, K_max)
    
    t_pts = np.arange(0, t, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    C_A = trapz(P_A_LED_change, t_pts)

    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up


### rho E with gamma and omega ###
def rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max):
    """
    in normalized time, added noise to variance of firing rates to PDF of hitting the lower bound
    """
    if t <= 0:
        return 0

    # evidence v
    v = gamma

    w = 0.5
    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    t_theta = 1 / omega
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

# time varying 

def rho_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t, gamma, omega, bound, phi, int_phi, w, K_max):
    """
    in normalized time, PDF of hitting the lower bound with gamma and omega, but time varying evidence
    """
    if t <= 0:
        return 0

    # evidence v
    v = gamma

    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    # t_theta = 1 / omega
    # t /= t_theta

    t = omega * int_phi

    non_sum_term = (1/a**2)*(a**3/np.sqrt(2*np.pi*t**3))*np.exp(-v*a*w - (v**2 * t)/2)
    K_max = int(K_max/2)
    k_vals = np.linspace(-K_max, K_max, 2*K_max + 1)
    sum_w_term = w + 2*k_vals
    sum_exp_term = np.exp(-(a**2 * (w + 2*k_vals)**2)/(2*t))
    sum_result = np.sum(sum_w_term*sum_exp_term)

    
    density =  non_sum_term * sum_result
    if density <= 0:
        density = 1e-16

    return density * (omega * phi)


def CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t, gamma, omega, bound, integ_phi, w, K_max):
    """
    In normalized time, CDF of hitting the lower bound with gamma and omega, but time varying evidence
    """
    if t <= 0:
        return 0

    # evidence v
    v = gamma

    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    # t_theta = 1 / omega
    # t /= t_theta

    # time normalization
    t = omega * integ_phi

    # Compute the exponent argument separately
    exponent_arg = -v * a * w - (((v**2) * t) / 2)

    # # Define safe thresholds for float64
    # max_safe_exp = np.log(np.finfo(np.float64).max)  # ~709
    # min_safe_exp = np.log(np.finfo(np.float64).tiny)   # very negative number

    # # if exponent_arg not within range print each value
    # if exponent_arg > max_safe_exp or exponent_arg < min_safe_exp:
    #     print(f'lambda = {rate_lambda}, T0 = {T_0}, theta_E = {theta_E}, Z_E = {Z_E}, ABL = {ABL}, ILD = {ILD}')
    #     print(f'v = {v}, a = {a}, w = {w}, t = {t}, exponent_arg = {exponent_arg}')


    # # Clip the exponent argument between the safe minimum and maximum
    # exponent_arg_clipped = np.clip(exponent_arg, min_safe_exp, max_safe_exp)

    # Now compute the result using the clipped exponent
    result = np.exp(exponent_arg)


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

def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_w_wrt_stim_fn(t, P_A, C_A, gamma, omega, t_E_aff, del_go, w, bound, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_E_aff + del_go
    t1 = t - t_E_aff

    #CDF_E_minus_small_t_NORM_omega_gamma_fn
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, 1, w, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, -1, w, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t2, gamma, omega, bound, w, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t1, gamma, omega, bound, w, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_w_fn(t-t_E_aff, gamma, omega, bound, w, K_max)


    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up

def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_w_wrt_stim_m3_probfn(t, P_A, C_A, gamma, omega, t_E_aff, del_go, w, bound, m3_up_prob, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_E_aff + del_go
    t1 = t - t_E_aff

    #CDF_E_minus_small_t_NORM_omega_gamma_fn
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, 1, w, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, -1, w, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    if bound == 1:
        random_choice_prob = m3_up_prob
    else:
        random_choice_prob = 1 - m3_up_prob
    random_readout_if_EA_surives = random_choice_prob * prob_EA_survives
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t2, gamma, omega, bound, w, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t1, gamma, omega, bound, w, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_w_fn(t-t_E_aff, gamma, omega, bound, w, K_max)


    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up

### Up or down, but not M3- paper model ############
# P_small_t_btn_x1_x2_omega_gamma_fn(x1, x2, t, omega, gamma, w, K_max)
def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_w_wrt_stim_NOT_m3_probfn(t, P_A, C_A, gamma, omega, t_E_aff, del_go, w, bound, m3_up_prob, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_E_aff + del_go
    t1 = t - t_E_aff

   
    if bound == 1:
        x1 = 1; x2 = 2
    else:
        x1 = 0; x2 = 1

    P_E_btn = P_small_t_btn_x1_x2_omega_gamma_fn(x1, x2, t + (del_go - t_E_aff), omega, gamma, w, K_max)
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t2, gamma, omega, bound, w, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t1, gamma, omega, bound, w, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_w_fn(t-t_E_aff, gamma, omega, bound, w, K_max)


    P_up = (P_A*(P_E_btn + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up

def all_RTs_fit_OPTIM_omega_gamma_PA_CA_wrt_stim_w_fn(t, P_A, C_A, gamma, omega, t_E_aff, w, K_max):
    """
    PDF of all RTs array irrespective of choice
    t is wrt stim
    """

    C_E = CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t - t_E_aff, gamma, omega, 1, w, K_max) \
           + CDF_E_minus_small_t_NORM_omega_gamma_w_fn(t - t_E_aff, gamma, omega, -1, w, K_max)
    

    P_E = rho_E_minus_small_t_NORM_omega_gamma_w_fn(t-t_E_aff, gamma, omega, 1, w, K_max) \
           + rho_E_minus_small_t_NORM_omega_gamma_w_fn(t-t_E_aff, gamma, omega, -1, w, K_max)
          

    P_A = np.array(P_A); C_E = np.array(C_E); P_E = np.array(P_E); C_A = np.array(C_A)
    P_all = P_A*(1-C_E) + P_E*(1-C_A)

    return P_all

## CDF E with gamma and omega
def CDF_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max):
    """
    In normalized time, CDF of hitting the lower bound.
    """
    if t <= 0:
        return 0

    # evidence v
    v = gamma

    w = 0.5
    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    t_theta = 1 / omega
    t /= t_theta

    # Compute the exponent argument separately
    exponent_arg = -v * a * w - (((v**2) * t) / 2)

    # # Define safe thresholds for float64
    # max_safe_exp = np.log(np.finfo(np.float64).max)  # ~709
    # min_safe_exp = np.log(np.finfo(np.float64).tiny)   # very negative number

    # # if exponent_arg not within range print each value
    # if exponent_arg > max_safe_exp or exponent_arg < min_safe_exp:
    #     print(f'lambda = {rate_lambda}, T0 = {T_0}, theta_E = {theta_E}, Z_E = {Z_E}, ABL = {ABL}, ILD = {ILD}')
    #     print(f'v = {v}, a = {a}, w = {w}, t = {t}, exponent_arg = {exponent_arg}')


    # # Clip the exponent argument between the safe minimum and maximum
    # exponent_arg_clipped = np.clip(exponent_arg, min_safe_exp, max_safe_exp)

    # Now compute the result using the clipped exponent
    result = np.exp(exponent_arg)


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



#### Noise change to sigma^2 ######
def rho_E_minus_small_t_NORM_added_noise_fn(t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max):
    """
    in normalized time, added noise to variance of firing rates to PDF of hitting the lower bound
    """
    if t <= 0:
        return 0
    chi = 17.37
    

    omega = (2/T_0) * (10**(rate_lambda*ABL/20))
    sigma_sq = omega

    q_e = 1
    theta = theta_E*q_e 

    # evidence v
    arg = rate_lambda * ILD / chi
    v = ( theta * omega * np.sinh(arg) ) / ( (omega * np.cosh(arg)) + (noise**2) )

    w = (Z_E + theta)/(2*theta)
    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    t_theta = (theta**2) / (sigma_sq + noise**2)
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

def CDF_E_minus_small_t_NORM_added_noise_fn(t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max):
    """
    In normalized time, CDF of hitting the lower bound.
    """
    if t <= 0:
        return 0

    chi = 17.37
    omega = (2/T_0) * (10**(rate_lambda*ABL/20))
    sigma_sq = omega 
    
    q_e = 1
    theta = theta_E*q_e

    
    arg = rate_lambda * ILD / chi
    v = ( theta * omega * np.sinh(arg) ) / ( (omega * np.cosh(arg)) + (noise**2) )


    w = (Z_E + theta)/(2*theta)
    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    
    t_theta = (theta**2) / (sigma_sq + noise**2)
    t /= t_theta


    # Compute the exponent argument separately
    exponent_arg = -v * a * w - (((v**2) * t) / 2)

    # # Define safe thresholds for float64
    # max_safe_exp = np.log(np.finfo(np.float64).max)  # ~709
    # min_safe_exp = np.log(np.finfo(np.float64).tiny)   # very negative number

    # # if exponent_arg not within range print each value
    # if exponent_arg > max_safe_exp or exponent_arg < min_safe_exp:
    #     print(f'lambda = {rate_lambda}, T0 = {T_0}, theta_E = {theta_E}, Z_E = {Z_E}, ABL = {ABL}, ILD = {ILD}')
    #     print(f'v = {v}, a = {a}, w = {w}, t = {t}, exponent_arg = {exponent_arg}')


    # # Clip the exponent argument between the safe minimum and maximum
    # exponent_arg_clipped = np.clip(exponent_arg, min_safe_exp, max_safe_exp)

    # Now compute the result using the clipped exponent
    result = np.exp(exponent_arg)


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


def all_RTs_fit_OPTIM_V_A_change_added_noise_fn(t, t_LED, V_A, V_A_post_LED, theta_A, ABL, ILD, rate_lambda, T_0, noise, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, K_max):
    """
    PDF of all RTs array irrespective of choice
    """

    P_A = PA_with_LEDON_2(t, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff)
    C_E = CDF_E_minus_small_t_NORM_added_noise_fn(t  - t_stim - t_E_aff, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, noise, K_max) \
           + CDF_E_minus_small_t_NORM_added_noise_fn(t  - t_stim - t_E_aff, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, noise, K_max)
    

    P_E = rho_E_minus_small_t_NORM_added_noise_fn(t-t_E_aff-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, noise, K_max) \
           + rho_E_minus_small_t_NORM_added_noise_fn(t-t_E_aff-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, noise, K_max) 
    
    t_pts = np.arange(0, t, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    C_A = trapz(P_A_LED_change, t_pts)

    P_A = np.array(P_A); C_E = np.array(C_E); P_E = np.array(P_E); C_A = np.array(C_A)
    P_all = P_A*(1-C_E) + P_E*(1-C_A)

    return P_all

def P_small_t_btn_x1_x2_omega_gamma_fn(x1, x2, t, omega, gamma, w, K_max):
    """
    Integration of P_small(x,t) btn x1 and x2
    """
    if t <= 0:
        return 0
    
    
    mu = gamma
    

    # z = (Z_E/theta) + 1.0 # 1 is middle point
    z = w * 2.0 # 0.5 x 2.0 = 1

    
    t_theta = 1 / omega
    t /= t_theta

    result = 0
    
    sqrt_t = np.sqrt(t)
    
    for n in range(-K_max, K_max + 1):
        term1 = np.exp(4 * mu * n) * (
            Phi((x2 - (z + 4 * n + mu * t)) / sqrt_t) -
            Phi((x1 - (z + 4 * n + mu * t)) / sqrt_t)
        )
        
        term2 = np.exp(2 * mu * (2 * (1 - n) - z)) * (
            Phi((x2 - (-z + 4 * (1 - n) + mu * t)) / sqrt_t) -
            Phi((x1 - (-z + 4 * (1 - n) + mu * t)) / sqrt_t)
        )
        
        result += term1 - term2
    
    return result


def P_small_t_btn_x1_x2_added_noise_fn(x1, x2, t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, noise, K_max):
    """
    Integration of P_small(x,t) btn x1 and x2
    """
    if t <= 0:
        return 0
    
    chi = 17.37
    omega = (2/T_0) * (10**(rate_lambda*ABL/20))
    sigma_sq = omega

    q_e = 1
    theta = theta_E*q_e

    
    arg = rate_lambda * ILD / chi
    mu = ( theta * omega * np.sinh(arg) ) / ( (omega * np.cosh(arg)) + (noise**2) )
    

    z = (Z_E/theta) + 1.0

    
    t_theta = (theta**2) / (sigma_sq + noise**2)
    t /= t_theta

    result = 0
    
    sqrt_t = np.sqrt(t)
    
    for n in range(-K_max, K_max + 1):
        term1 = np.exp(4 * mu * n) * (
            Phi((x2 - (z + 4 * n + mu * t)) / sqrt_t) -
            Phi((x1 - (z + 4 * n + mu * t)) / sqrt_t)
        )
        
        term2 = np.exp(2 * mu * (2 * (1 - n) - z)) * (
            Phi((x2 - (-z + 4 * (1 - n) + mu * t)) / sqrt_t) -
            Phi((x1 - (-z + 4 * (1 - n) + mu * t)) / sqrt_t)
        )
        
        result += term1 - term2
    
    return result


def up_RTs_fit_OPTIM_V_A_change_added_noise_fn(t, t_LED, V_A, V_A_post_LED, theta_A, ABL, ILD, rate_lambda, T_0, noise, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    bound = 1
    x1 = 1; x2 = 2
    t1 = t - t_stim - t_E_aff
    t2 = t - t_stim
    
    P_A = PA_with_LEDON_2(t, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff)
    P_EA_btn_1_2 = P_small_t_btn_x1_x2_added_noise_fn(x1, x2, t - t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, noise, K_max)
    P_E_plus_cum = CDF_E_minus_small_t_NORM_added_noise_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max) \
                    - CDF_E_minus_small_t_NORM_added_noise_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max)
    
    

    P_E_plus = rho_E_minus_small_t_NORM_added_noise_fn(t-t_E_aff-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max)
    
    t_pts = np.arange(0, t, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    C_A = trapz(P_A_LED_change, t_pts)

    P_up = (P_A*(P_EA_btn_1_2 + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up

def up_RTs_fit_OPTIM_V_A_change_added_noise_M3_delGO_fn(t, t_LED, V_A, V_A_post_LED, theta_A, ABL, ILD, rate_lambda, T_0, noise, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, del_go, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    bound = 1
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    P_A = PA_with_LEDON_2(t, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff)
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_added_noise_fn(t - t_stim - t_E_aff + del_go, ABL, ILD,\
                                                                         rate_lambda, T_0, theta_E, Z_E, 1, noise, K_max) \
                             + CDF_E_minus_small_t_NORM_added_noise_fn(t - t_stim - t_E_aff + del_go, ABL, ILD,\
                                                                         rate_lambda, T_0, theta_E, Z_E, -1, noise, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_cum = CDF_E_minus_small_t_NORM_added_noise_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max) \
                    - CDF_E_minus_small_t_NORM_added_noise_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max)
    
    

    P_E_plus = rho_E_minus_small_t_NORM_added_noise_fn(t-t_E_aff-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max)
    
    t_pts = np.arange(0, t, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    C_A = trapz(P_A_LED_change, t_pts)

    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up

def up_RTs_fit_OPTIM_V_A_change_gamma_omega_fn(t, t_LED, V_A, V_A_post_LED, theta_A, gamma, omega, t_stim, t_A_aff, t_E_aff, del_go, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    bound = 1
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    P_A = PA_with_LEDON_2(t, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff)
    #CDF_E_minus_small_t_NORM_omega_gamma_fn
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, 1, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, -1, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_fn(t2, gamma, omega, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_fn(t1, gamma, omega, bound, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_fn(t-t_E_aff-t_stim, gamma, omega, bound, K_max)
    
    t_pts = np.arange(0, t, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    C_A = trapz(P_A_LED_change, t_pts)

    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up
##### Functions related to phi(t) - decaying evid over time ############
def K0(y, h, a, b):
    # a*sin(2*pi/h*(y-b)) where (y-b)>=0 and (y-b)<h/4, else 0
    return a * np.sin(2 * np.pi/h * (y - b)) * ((y - b) >= 0) * ((y - b) < h/4)

def K1(y, h, a, b):
    # (1+a*sin(2*pi/h*(y-b))) for (y-b) in [h/4, h/2)
    return (1 + a * np.sin(2 * np.pi/h * (y - b))) * ((y - b) >= h/4) * ((y - b) < h/2)

def K2(y, h, a, b):
    # (1 - a*(y-b)/h * exp(1 - (y-b)/h)) for (y-b)>=0
    return (1 - a * (y - b) / h * np.exp(1 - (y - b)/h)) * ((y - b) >= 0)

def phi_t_fn(y, h1, a1, b1, h2, a2):
    # Combination of K0, K1, and K2 functions
    return K0(y, h1, 1 + a1, b1) + K1(y, h1, a1, b1) + K2(y, h2, a2, b1 + h1/2)

def I0(y, h, a, b):
    # a*h/pi * sin(pi/h*(y-b))^2 for (y-b)>=0 and (y-b)<=h/4
    return a * h/np.pi * (np.sin(np.pi/h * (y - b))**2) * ((y - b) >= 0) * ((y - b) <= h/4)

def I1(y, h, a, b):
    # I0 evaluated at b+h/4 plus extra terms for y in (b+h/4, b+h/2]
    term1 = I0(b + h/4, h, 1 + a, b)
    term2 = y - (b + h/4)
    term3 = a * h/(2 * np.pi) * np.cos(2 * np.pi/h * (y - b))
    return (term1 + term2 - term3) * ((y - b) > h/4) * ((y - b) <= h/2)

def I2(y, h1, a1, b1, h2, a2, b2):
    # I1 evaluated at b2 plus extra terms for y > b2
    term1 = I1(b2, h1, a1, b1)
    term2 = y - b2
    # The expression: -a2*h2*(-exp(1-(y-b2)/h2).*((y-b2)/h2+1)+exp(1))
    term3 = a2 * h2 * (-np.exp(1 - (y - b2)/h2) * ((y - b2)/h2 + 1) + np.exp(1))
    return (term1 + term2 - term3) * ((y - b2) > 0)

def int_phi_fn(y, h1, a1, b1, h2, a2):
    # Combination of I0, I1, and I2 functions with b2 = b1+h1/2
    b2 = b1 + h1/2
    return I0(y, h1, 1 + a1, b1) + I1(y, h1, a1, b1) + I2(y, h1, a1, b1, h2, a2, b2)
#############################################################################

#################### Time varying - Up and down  ###################################
def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_time_varying_fn(t, P_A, C_A, gamma, omega, t_E_aff, del_go, phi_params, w, bound, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_E_aff + del_go
    t1 = t - t_E_aff

    int_phi_t_E_g = int_phi_fn(t - t_E_aff + del_go, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    # CDF_E_minus_small_t_NORM_omega_gamma_fn
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, 1, int_phi_t_E_g, w, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, -1, int_phi_t_E_g, w, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives

    int_phi_t2 = int_phi_fn(t2, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    int_phi_t1 = int_phi_fn(t1, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t2, gamma, omega, bound, int_phi_t2, w, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t1, gamma, omega, bound, int_phi_t1, w, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    phi_t_e = phi_t_fn(t - t_E_aff, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    int_phi_t_e = int_phi_fn(t - t_E_aff, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t-t_E_aff, gamma, omega, bound, phi_t_e, int_phi_t_e, w, K_max)


    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up

def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_w_time_varying_led_off_fn(t, V_A, theta_A, gamma, omega, t_stim, \
                                            t_A_aff, t_E_aff, del_go, phi_params, w, bound, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff
    t_LED = 100 # so that stupid integral is not used

    P_A = PA_with_LEDON_2(t, V_A, np.nan, theta_A, 0, t_LED, t_A_aff)
    int_phi_t_E_g = int_phi_fn(t - t_stim - t_E_aff + del_go, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, 1, int_phi_t_E_g, w, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, -1, int_phi_t_E_g, w, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    
    # P_E_plus_cum
    int_phi_t2 = int_phi_fn(t2, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    int_phi_t1 = int_phi_fn(t1, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t2, gamma, omega, bound, int_phi_t2, w, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t1, gamma, omega, bound, int_phi_t1, w, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    phi_t_e = phi_t_fn(t - t_E_aff - t_stim, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    int_phi_t_e = int_phi_fn(t - t_E_aff - t_stim, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t-t_E_aff-t_stim, gamma, omega, bound, phi_t_e, int_phi_t_e, w, K_max)
    
    t_pts = np.arange(0, t, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, np.nan, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    C_A = trapz(P_A_LED_change, t_pts)

    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up

###########################################################################################################

def up_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_fn(t, P_A, C_A, gamma, omega, t_stim, t_E_aff, del_go, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    bound = 1
    t2 = t - t_E_aff + del_go
    t1 = t - t_E_aff

    #CDF_E_minus_small_t_NORM_omega_gamma_fn
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, 1, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, -1, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_fn(t2, gamma, omega, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_fn(t1, gamma, omega, bound, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_fn(t-t_E_aff, gamma, omega, bound, K_max)


    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up


def down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_fn(t, P_A, C_A, gamma, omega, t_stim, t_E_aff, del_go, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    bound = -1
    t2 = t - t_E_aff + del_go
    t1 = t - t_E_aff

    #CDF_E_minus_small_t_NORM_omega_gamma_fn
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, 1, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, -1, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_fn(t2, gamma, omega, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_fn(t1, gamma, omega, bound, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_fn(t-t_E_aff, gamma, omega, bound, K_max)


    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up


def down_RTs_fit_OPTIM_V_A_change_gamma_omega_fn(t, t_LED, V_A, V_A_post_LED, theta_A, gamma, omega, t_stim, t_A_aff, t_E_aff, del_go, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    bound = -1
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    P_A = PA_with_LEDON_2(t, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff)
    #CDF_E_minus_small_t_NORM_omega_gamma_fn
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, 1, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, -1, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_fn(t2, gamma, omega, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_fn(t1, gamma, omega, bound, K_max)
    
    
    # rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max)
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_fn(t-t_E_aff-t_stim, gamma, omega, bound, K_max)
    
    t_pts = np.arange(0, t, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    C_A = trapz(P_A_LED_change, t_pts)

    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up



def down_RTs_fit_OPTIM_V_A_change_added_noise_M3_delGO_fn(t, t_LED, V_A, V_A_post_LED, theta_A, ABL, ILD, rate_lambda, T_0, noise, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, del_go, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    bound = -1
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    P_A = PA_with_LEDON_2(t, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff)
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_added_noise_fn(t - t_stim - t_E_aff + del_go, ABL, ILD,\
                                                                         rate_lambda, T_0, theta_E, Z_E, 1, noise, K_max) \
                             + CDF_E_minus_small_t_NORM_added_noise_fn(t - t_stim - t_E_aff + del_go, ABL, ILD,\
                                                                         rate_lambda, T_0, theta_E, Z_E, -1, noise, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_minus_cum = CDF_E_minus_small_t_NORM_added_noise_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max) \
                    - CDF_E_minus_small_t_NORM_added_noise_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max)
    
    

    P_E_minus = rho_E_minus_small_t_NORM_added_noise_fn(t-t_E_aff-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max)
    
    t_pts = np.arange(0, t, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    C_A = trapz(P_A_LED_change, t_pts)

    P_down = (P_A*(random_readout_if_EA_surives + P_E_minus_cum) + P_E_minus*(1-C_A))

    return P_down


def down_RTs_fit_OPTIM_V_A_change_added_noise_fn(t, t_LED, V_A, V_A_post_LED, theta_A, ABL, ILD, rate_lambda, T_0, noise, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    bound = -1
    x1 = 0; x2 = 1
    t1 = t - t_stim - t_E_aff
    t2 = t - t_stim
    
    P_A = PA_with_LEDON_2(t, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff)
    P_EA_btn_0_1 = P_small_t_btn_x1_x2_added_noise_fn(x1, x2, t - t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, noise, K_max)
    P_E_minus_cum = CDF_E_minus_small_t_NORM_added_noise_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max) \
                    - CDF_E_minus_small_t_NORM_added_noise_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max)
    
    

    P_E_minus = rho_E_minus_small_t_NORM_added_noise_fn(t-t_E_aff-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, noise, K_max)
    
    t_pts = np.arange(0, t, 0.001)
    P_A_LED_change = np.array([PA_with_LEDON_2(i, V_A, V_A_post_LED, theta_A, 0, t_LED, t_A_aff) for i in t_pts])
    C_A = trapz(P_A_LED_change, t_pts)

    P_down = (P_A*(P_EA_btn_0_1 + P_E_minus_cum) + P_E_minus*(1-C_A))

    return P_down



def up_RTs_fit_TRUNC_fn(t_pts, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, K_max, T_trunc):
    """
    PDF of up RTs array
    """
    trunc_factor = 1 - cum_A_t_fn(T_trunc-t_A_aff, V_A, theta_A)
    bound = 1

    P_A = [0 if t < T_trunc else rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A)/trunc_factor for t in t_pts]
    P_EA_btn_1_2 = [prob_x_t_and_hit_up_or_down_analytic(t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, L, K_max, bound) + P_small_t_btn_x1_x2(1+L/2, 2, t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max) for t in t_pts]
    P_E_plus_cum = np.zeros(len(t_pts))
    for i,t in enumerate(t_pts):
        t1 = t - t_motor - t_stim - t_E_aff
        t2 = t - t_stim
        if t1 < 0:
            t1 = 0
        P_E_plus_cum[i] = CDF_E_minus_small_t_NORM_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max)


    P_E_plus = [rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) for t in t_pts]
    C_A = [0 if t < T_trunc else cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A)/trunc_factor for t in t_pts]

    P_A = np.array(P_A); P_EA_btn_1_2 = np.array(P_EA_btn_1_2); P_E_plus = np.array(P_E_plus); C_A = np.array(C_A)
    P_correct_unnorm = (P_A*(P_EA_btn_1_2 + P_E_plus_cum) + P_E_plus*(1-C_A))
    return P_correct_unnorm


def up_RTs_fit_fn(t_pts, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, K_max):
    """
    PDF of up RTs array
    """
    bound = 1

    P_A = [rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]
    P_EA_btn_1_2 = [prob_x_t_and_hit_up_or_down_analytic(t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, L, K_max, bound) + P_small_t_btn_x1_x2(1+L/2, 2, t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max) for t in t_pts]
    P_E_plus_cum = np.zeros(len(t_pts))
    for i,t in enumerate(t_pts):
        t1 = t - t_motor - t_stim - t_E_aff
        t2 = t - t_stim
        if t1 < 0: 
            t1 = 0
        P_E_plus_cum[i] = CDF_E_minus_small_t_NORM_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max)


    P_E_plus = [rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) for t in t_pts]
    C_A = [cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]

    P_A = np.array(P_A); P_EA_btn_1_2 = np.array(P_EA_btn_1_2); P_E_plus = np.array(P_E_plus); C_A = np.array(C_A)
    P_correct_unnorm = (P_A*(P_EA_btn_1_2 + P_E_plus_cum) + P_E_plus*(1-C_A))
    return P_correct_unnorm


def up_RTs_fit_single_t_fn(t, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, K_max):
    """
    PDF of up RTs array
    """
    bound = 1

    P_A = rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) 
    P_EA_btn_1_2 = prob_x_t_and_hit_up_or_down_analytic(t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, L, K_max, bound) \
          + P_small_t_btn_x1_x2(1+L/2, 2, t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max) 
    t1 = t - t_motor - t_stim - t_E_aff
    t2 = t - t_stim
    if t1 < 0:
        t1 = 0
    P_E_plus_cum = CDF_E_minus_small_t_NORM_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) \
                - CDF_E_minus_small_t_NORM_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max)


    P_E_plus = rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) 
    C_A = cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) 

    P_correct_unnorm = (P_A*(P_EA_btn_1_2 + P_E_plus_cum) + P_E_plus*(1-C_A))
    return P_correct_unnorm

def down_RTs_fit_single_t_fn(t, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, K_max):
    """
    PDF of down RTs array
    """
    bound = -1
        
    P_A = rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A)
    P_EA_btn_0_1 = prob_x_t_and_hit_up_or_down_analytic(t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, L, K_max, bound) + P_small_t_btn_x1_x2(0, 1 - L/2, t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max)
    t1 = t - t_motor - t_stim - t_E_aff
    t2 = t - t_stim
    P_E_minus_cum = CDF_E_minus_small_t_NORM_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) \
                - CDF_E_minus_small_t_NORM_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max)


    P_E_minus = rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max)
    C_A = cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A)

    P_wrong_unnorm = (P_A*(P_EA_btn_0_1+P_E_minus_cum) + P_E_minus*(1-C_A))
    return P_wrong_unnorm

def down_RTs_fit_TRUNC_fn(t_pts, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, K_max, T_trunc):
    """
    PDF of down RTs array
    """
    trunc_factor = 1 - cum_A_t_fn(T_trunc-t_A_aff, V_A, theta_A)
    bound = -1
        
    P_A = [0 if t < T_trunc else rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A)/trunc_factor for t in t_pts]
    P_EA_btn_0_1 = [prob_x_t_and_hit_up_or_down_analytic(t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, L, K_max, bound) + P_small_t_btn_x1_x2(0, 1 - L/2, t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max)  for t in t_pts]
    P_E_minus_cum = np.zeros(len(t_pts))
    for i,t in enumerate(t_pts):
        t1 = t - t_motor - t_stim - t_E_aff
        t2 = t - t_stim
        # if t1 < 0:
        #     t1 = 0
        P_E_minus_cum[i] = CDF_E_minus_small_t_NORM_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max)


    P_E_minus = [rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) for t in t_pts]
    C_A = [0 if t < T_trunc else cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A)/trunc_factor for t in t_pts]

    P_A = np.array(P_A); P_EA_btn_0_1 = np.array(P_EA_btn_0_1); P_E_minus = np.array(P_E_minus); C_A = np.array(C_A)
    P_wrong_unnorm = (P_A*(P_EA_btn_0_1+P_E_minus_cum) + P_E_minus*(1-C_A))
    return P_wrong_unnorm


def down_RTs_fit_fn(t_pts, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, L, K_max):
    """
    PDF of down RTs array
    """
    bound = -1
        
    P_A = [rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]
    P_EA_btn_0_1 = [prob_x_t_and_hit_up_or_down_analytic(t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, L, K_max, bound) + P_small_t_btn_x1_x2(0, 1 - L/2, t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max)  for t in t_pts]
    P_E_minus_cum = np.zeros(len(t_pts))
    for i,t in enumerate(t_pts):
        t1 = t - t_motor - t_stim - t_E_aff
        t2 = t - t_stim
        # if t1 < 0:
        #     t1 = 0
        P_E_minus_cum[i] = CDF_E_minus_small_t_NORM_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max)


    P_E_minus = [rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) for t in t_pts]
    C_A = [cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]

    P_A = np.array(P_A); P_EA_btn_0_1 = np.array(P_EA_btn_0_1); P_E_minus = np.array(P_E_minus); C_A = np.array(C_A)
    P_wrong_unnorm = (P_A*(P_EA_btn_0_1+P_E_minus_cum) + P_E_minus*(1-C_A))
    return P_wrong_unnorm

def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def cum_A_t_fn(t, V_A, theta_A):
    """
    For AI, calculate cummulative distrn of a time t given V_A, theta_A
    """
    if t <= 0:
        return 0

    term1 = Phi(V_A * ((t) - (theta_A/V_A)) / np.sqrt(t))
    term2 = np.exp(2 * V_A * theta_A) * Phi(-V_A * ((t) + (theta_A / V_A)) / np.sqrt(t))
    
    return term1 + term2

# Helper functions for PDF and CDF
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

def cum_E_t_fn(t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max):
    return CDF_E_minus_small_t_NORM_fn(t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) + \
    CDF_E_minus_small_t_NORM_fn(t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max)

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
