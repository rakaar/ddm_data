import numpy as np
import random
from scipy.special import erf, erfcx
from numba import jit

def psiam_tied_data_gen_wrapper(V_A, theta_A, ABL_arr, ILD_arr, rate_lambda, T_0, theta_E, Z_E, t_stim_arr, t_A_aff, t_E_aff, t_motor, dt):
    ABL = random.choice(ABL_arr)
    ILD = random.choice(ILD_arr)
    t_stim = random.choice(t_stim_arr)
    
    
    choice, rt, is_act = simulate_psiam_tied(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, dt)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD}

@jit
def simulate_psiam_tied(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, dt):
    AI = 0; DV = Z_E; t = 0; dB = dt**0.5
    
    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    mu = (2*q_e/T_0) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
    sigma = np.sqrt( (2*(q_e**2)/T_0) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )
    
    is_act = 0
    while True:
        if t*dt > t_stim + t_E_aff:
            DV += mu*dt + sigma*np.random.normal(0, dB)
        
        if t*dt > t_A_aff:
            AI += V_A*dt + np.random.normal(0, dB)
        
        t += 1
        
        if DV >= theta:
            choice = +1; RT = t*dt + t_motor
            break
        elif DV <= -theta:
            choice = -1; RT = t*dt + t_motor
            break
        
        if AI >= theta_A:
            is_act = 1
            AI_hit_time = t*dt
            # if t*dt > t_stim - t_motor:
            while t*dt <= (AI_hit_time + t_E_aff + t_motor):#  u can process evidence till stim plays
                if t*dt > t_stim + t_E_aff: # Evid accum wil begin only after stim starts and afferent delay
                    DV += mu*dt + sigma*np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        break
                    elif DV <= -theta:
                        DV = -theta
                        break
                t += 1
            
            break
        
        
    if is_act == 1:
        RT = AI_hit_time + t_motor
        if DV > 0:
            choice = 1
        elif DV < 0:
            choice = -1
        else: # if DV is 0 because stim has not yet been played, then choose right/left randomly
            randomly_choose_up = np.random.rand() >= 0.5
            if randomly_choose_up:
                choice = 1
            else:
                choice = -1       
    
    return choice, RT, is_act


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

# AI
def rho_A_t_fn(t, V_A, theta_A):
    """
    For AI,prob density of t given V_A, theta_A
    """
    if t <= 0:
        return 0
    return (theta_A*1/np.sqrt(2*np.pi*(t)**3))*np.exp(-0.5 * (V_A**2) * (((t) - (theta_A/V_A))**2)/(t))


def cum_A_t_fn(t, V_A, theta_A):
    """
    For AI, calculate cummulative distrn of a time t given V_A, theta_A
    """
    if t <= 0:
        return 0

    term1 = Phi(V_A * ((t) - (theta_A/V_A)) / np.sqrt(t))
    term2 = np.exp(2 * V_A * theta_A) * Phi(-V_A * ((t) + (theta_A / V_A)) / np.sqrt(t))
    
    return term1 + term2

# EA 
def rho_E_minus_small_t_NORM_fn(t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max):
    """
    in normalized time, PDF of hitting the lower bound
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


def P_small_t_btn_x1_x2(x1, x2, t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max):
    """
    Integration of P_small(x,t) btn x1 and x2
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

# PDF for RTs
def all_RTs_fit_fn(t_pts, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, K_max):
    """
    PDF of all RTs array irrespective of choice
    """


    P_A = [rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]# if AI hit
    C_E = [CDF_E_minus_small_t_NORM_fn(t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
           + CDF_E_minus_small_t_NORM_fn(t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max) for t in t_pts]
    P_E_cum = np.zeros(len(t_pts))
    for i,t in enumerate(t_pts):
        t1 = t - t_motor - t_stim - t_E_aff
        t2 = t - t_stim
        if t1 < 0:
            t1 = 0
        P_E_cum[i] = CDF_E_minus_small_t_NORM_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
                    + CDF_E_minus_small_t_NORM_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max) \
                    - CDF_E_minus_small_t_NORM_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
                    - CDF_E_minus_small_t_NORM_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max)


    P_E = [rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
           + rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max) \
            for t in t_pts]
    C_A = [cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]

    P_A = np.array(P_A); C_E = np.array(C_E); P_E = np.array(P_E); C_A = np.array(C_A)
    P_all = P_A*((1-C_E)+P_E_cum) + P_E*(1-C_A)

    return P_all

def all_RTs_fit_OPTIM_fn(t_pts, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, K_max):
    """
    PDF of all RTs array irrespective of choice
    """


    P_A = [rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]# if AI hit
    C_E = [CDF_E_minus_small_t_NORM_fn(t - t_motor - t_stim - t_E_aff, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
           + CDF_E_minus_small_t_NORM_fn(t - t_motor - t_stim - t_E_aff, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max) for t in t_pts]
    

    P_E = [rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
           + rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max) \
            for t in t_pts]
    C_A = [cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]

    P_A = np.array(P_A); C_E = np.array(C_E); P_E = np.array(P_E); C_A = np.array(C_A)
    P_all = P_A*(1-C_E) + P_E*(1-C_A)

    return P_all

def all_RTs_fit_single_t_fn(t, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, K_max):
    """
    PDF of all RTs array irrespective of choice
    """


    P_A = rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A)
    C_E = CDF_E_minus_small_t_NORM_fn(t - t_motor - t_stim - t_E_aff, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
           + CDF_E_minus_small_t_NORM_fn(t - t_motor - t_stim - t_E_aff, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max)
    

    P_E = rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, 1, K_max) \
           + rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, -1, K_max) \
            
    C_A = cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A)

    P_all = P_A*(1-C_E) + P_E*(1-C_A)

    return P_all



def up_RTs_fit_fn(t_pts, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, K_max):
    """
    PDF of up RTs array
    """
    bound = 1

    P_A = [rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]
    P_EA_btn_1_2 = [P_small_t_btn_x1_x2(1, 2, t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max) for t in t_pts]
    P_E_plus_cum = np.zeros(len(t_pts))
    for i,t in enumerate(t_pts):
        t1 = t - t_motor - t_stim - t_E_aff
        t2 = t - t_stim
        # if t1 < 0:
        #     t1 = 0
        P_E_plus_cum[i] = CDF_E_minus_small_t_NORM_fn(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_fn(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max)


    P_E_plus = [rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) for t in t_pts]
    C_A = [cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]

    P_A = np.array(P_A); P_EA_btn_1_2 = np.array(P_EA_btn_1_2); P_E_plus = np.array(P_E_plus); C_A = np.array(C_A)
    P_correct_unnorm = (P_A*(P_EA_btn_1_2 + P_E_plus_cum) + P_E_plus*(1-C_A))
    return P_correct_unnorm



def down_RTs_fit_fn(t_pts, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, K_max):
    """
    PDF of down RTs array
    """
    bound = -1
        
    P_A = [rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]
    P_EA_btn_0_1 = [P_small_t_btn_x1_x2(0, 1, t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max) for t in t_pts]
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


# VECTORIZED funcs
def P_small_t_btn_x1_x2_vectorized(x1, x2, t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max):
    """
    Vectorized version of the P_small_t_btn_x1_x2 function.
    Computes the integration of P_small(x,t) between x1 and x2.
    """
    if t <= 0:
        return 0
    
    q_e = 1
    theta = theta_E * q_e

    chi = 17.37
    mu = theta_E * np.tanh(rate_lambda * ILD / chi)
    z = (Z_E / theta) + 1.0

    # Compute t_theta and normalize t
    t_theta = T_0 * (theta_E ** 2) * (10 ** (-rate_lambda * ABL / 20)) * (1 / (2 * np.cosh(rate_lambda * ILD / chi)))
    t_normalized = t / t_theta

    # Compute sqrt(t_normalized)
    sqrt_t = np.sqrt(t_normalized)
    
    # Handle potential division by zero
    sqrt_t = np.where(sqrt_t == 0, 1e-10, sqrt_t)

    # Create an array of n values from -K_max to K_max inclusive
    n = np.arange(-K_max, K_max + 1)

    # Compute exponentials for term1 and term2
    exp_term1 = np.exp(4 * mu * n)
    exp_term2 = np.exp(2 * mu * (2 * (1 - n) - z))

    # Compute arguments for Phi functions in term1
    phi1_upper = (x2 - (z + 4 * n + mu * t_normalized)) / sqrt_t
    phi1_lower = (x1 - (z + 4 * n + mu * t_normalized)) / sqrt_t

    # Compute Phi for term1
    Phi_term1 = Phi(phi1_upper) - Phi(phi1_lower)

    # Compute arguments for Phi functions in term2
    phi2_upper = (x2 - (-z + 4 * (1 - n) + mu * t_normalized)) / sqrt_t
    phi2_lower = (x1 - (-z + 4 * (1 - n) + mu * t_normalized)) / sqrt_t

    # Compute Phi for term2
    Phi_term2 = Phi(phi2_upper) - Phi(phi2_lower)

    # Compute term1 and term2
    term1 = exp_term1 * Phi_term1
    term2 = exp_term2 * Phi_term2

    # Compute the result by summing over all n
    result = np.sum(term1 - term2)

    return result



def CDF_E_minus_small_t_NORM_fn_vectorized(t, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max):
    """
    Vectorized version of the CDF of hitting the lower bound in normalized time.
    Utilizes custom phi and M functions.
    """
    if t <= 0:
        return 0
    
    q_e = 1
    theta = theta_E * q_e

    chi = 17.37
    v = theta_E * np.tanh(rate_lambda * ILD / chi)
    w = (Z_E + theta) / (2 * theta)
    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    t_theta = T_0 * (theta_E ** 2) * (10 ** (-rate_lambda * ABL / 20)) * (1 / (2 * np.cosh(rate_lambda * ILD / chi)))
    t_normalized = t / t_theta

    result = np.exp(-v * a * w - ((v ** 2) * t_normalized) / 2)

    k = np.arange(K_max + 1)

    is_even = (k % 2 == 0).astype(float)
    r_k = k * a + a * np.where(is_even, w, 1 - w)
    sqrt_t = np.sqrt(t_normalized)
    sqrt_t = np.where(sqrt_t == 0, 1e-10, sqrt_t)

    phi_args = r_k / sqrt_t
    M_args_positive = (r_k - v * t_normalized) / sqrt_t
    M_args_negative = (r_k + v * t_normalized) / sqrt_t

    phi_vals = phi(phi_args)
    M_vals = M(M_args_positive) + M(M_args_negative)

    sign = (-1) ** k

    summation = np.sum(sign * phi_vals * M_vals)

    return result * summation

def up_RTs_fit_VEC_fn(t_pts, V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, t_motor, K_max):
    """
    PDF of up RTs array
    """
    bound = 1

    P_A = [rho_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]
    P_EA_btn_1_2 = [P_small_t_btn_x1_x2_vectorized(1, 2, t-t_stim, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, K_max) for t in t_pts]
    P_E_plus_cum = np.zeros(len(t_pts))
    for i,t in enumerate(t_pts):
        t1 = t - t_motor - t_stim - t_E_aff
        t2 = t - t_stim
        # if t1 < 0:
        #     t1 = 0
        P_E_plus_cum[i] = CDF_E_minus_small_t_NORM_fn_vectorized(t2, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_fn_vectorized(t1, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max)


    P_E_plus = [rho_E_minus_small_t_NORM_fn(t-t_E_aff-t_stim-t_motor, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, bound, K_max) for t in t_pts]
    C_A = [cum_A_t_fn(t-t_A_aff-t_motor, V_A, theta_A) for t in t_pts]

    P_A = np.array(P_A); P_EA_btn_1_2 = np.array(P_EA_btn_1_2); P_E_plus = np.array(P_E_plus); C_A = np.array(C_A)
    P_correct_unnorm = (P_A*(P_EA_btn_1_2 + P_E_plus_cum) + P_E_plus*(1-C_A))
    return P_correct_unnorm