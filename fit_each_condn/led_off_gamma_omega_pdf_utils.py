import numpy as np
from scipy.special import erf, erfcx


###########################################
############ Proactive PDFs ##############
###########################################
def rho_A_t_fn(t, V_A, theta_A):
    """
    Proactive PDF, takes input as scalar, delays should be already subtracted before calling func
    """
    if t <= 0:
        return 0
    return (theta_A*1/np.sqrt(2*np.pi*(t)**3))*np.exp(-0.5 * (V_A**2) * (((t) - (theta_A/V_A))**2)/(t))


def rho_A_t_VEC_fn(t_pts, V_A, theta_A):
    """
    Proactive PDF, takes input as vector, delays should be already subtracted before calling func
    """
    t_pts = np.asarray(t_pts)
    rho = np.zeros_like(t_pts, dtype=float)

    # Apply the formula only where t > 0
    valid_idx = t_pts > 0
    t_valid = t_pts[valid_idx]
    
    rho[valid_idx] = (theta_A / np.sqrt(2 * np.pi * t_valid**3)) * \
        np.exp(-0.5 * (V_A**2) * ((t_valid - theta_A / V_A)**2) / t_valid)
    
    return rho

def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def cum_A_t_fn(t, V_A, theta_A):
    """
    Proactive cdf, input time scalar, delays should be already subtracted before calling func
    """
    if t <= 0:
        return 0

    term1 = Phi(V_A * ((t) - (theta_A/V_A)) / np.sqrt(t))
    term2 = np.exp(2 * V_A * theta_A) * Phi(-V_A * ((t) + (theta_A / V_A)) / np.sqrt(t))
    
    return term1 + term2

###########################################
############ Reactive PDFs ##############
###########################################

def rho_E_minus_small_t_NORM_omega_gamma_fn(t, gamma, omega, bound, K_max):
    """
    Reactive PDF, time scalar, delays should be already subtracted before calling func
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

def phi(x):
    """Standard Gaussian function."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

def M(x):
    """Mills ratio."""
    return np.sqrt(np.pi / 2) * erfcx(x / np.sqrt(2))


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

###########################################
############ Pro + Re - active PDFs #######
###########################################
def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_fn(t, V_A, theta_A, gamma, omega, t_stim, t_A_aff, t_E_aff, del_go, bound, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    P_A = rho_A_t_fn(t - t_A_aff, V_A, theta_A)
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, 1, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, -1, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_or_minus_cum = CDF_E_minus_small_t_NORM_omega_gamma_fn(t2, gamma, omega, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_fn(t1, gamma, omega, bound, K_max)
    
    
    P_E_plus_or_minus = rho_E_minus_small_t_NORM_omega_gamma_fn(t-t_E_aff-t_stim, gamma, omega, bound, K_max)
    
    C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
    return (P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum) + P_E_plus_or_minus*(1-C_A))

    
def cum_pro_and_reactive(t, V_A, theta_A, t_A_aff, gamma, omega, t_stim, t_E_aff, K_max = 10):

    c_A = cum_A_t_fn(t-t_A_aff, V_A, theta_A)
    c_E = CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_stim - t_E_aff, gamma, omega, 1, K_max) + \
        CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_stim - t_E_aff, gamma, omega, -1, K_max)
    
    return c_A + c_E - c_A * c_E


def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_P_A_C_A_wrt_stim_fn(t, P_A, C_A, gamma, omega, t_E_aff, del_go, bound, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_E_aff + del_go
    t1 = t - t_E_aff

    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, 1, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_fn(t - t_E_aff + del_go,\
                                                                         gamma, omega, -1, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_cum = CDF_E_minus_small_t_NORM_omega_gamma_fn(t2, gamma, omega, bound, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_fn(t1, gamma, omega, bound, K_max)
    
    
    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_fn(t-t_E_aff, gamma, omega, bound, K_max)


    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up