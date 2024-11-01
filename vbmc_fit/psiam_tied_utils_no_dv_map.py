import numpy as np
import random
from scipy.special import erf, erfcx




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