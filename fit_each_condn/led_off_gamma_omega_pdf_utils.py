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

def rho_E_minus_small_t_NORM_omega_gamma_with_w_fn(t, gamma, omega, bound, w, K_max):
    """
    Reactive PDF, time scalar, delays should be already subtracted before calling func
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

def CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t, gamma, omega, bound, w, K_max):
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

    if np.isnan(summation) or np.isnan(result):
        raise ValueError(f"summation or result is nan or inf for t={t}, gamma={gamma}, omega={omega}, bound={bound}, summation={summation}, result={result}, w={w}, K_max={K_max}")

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


def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_fn(t, V_A, theta_A, gamma, omega, t_stim, t_A_aff, t_E_aff, del_go, bound, w, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    P_A = rho_A_t_fn(t - t_A_aff, V_A, theta_A)
    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, 1, w, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, -1, w, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_or_minus_cum = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t2, gamma, omega, bound, w, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t1, gamma, omega, bound, w, K_max)
    
    
    P_E_plus_or_minus = rho_E_minus_small_t_NORM_omega_gamma_with_w_fn(t-t_E_aff-t_stim, gamma, omega, bound, w, K_max)
    
    C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
    return (P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum) + P_E_plus_or_minus*(1-C_A))

def up_or_down_RTs_fit_OPTIM_V_A_change_gamma_omega_with_w_PA_CA_fn(t, P_A, C_A, gamma, omega, t_stim, t_E_aff, del_go, bound, w, K_max):
    """
    PDF of all RTs array irrespective of choice
    """
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, 1, w, K_max) \
                             + CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t - t_stim - t_E_aff + del_go,\
                                                                         gamma, omega, -1, w, K_max)
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    P_E_plus_or_minus_cum = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t2, gamma, omega, bound, w, K_max) \
                    - CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t1, gamma, omega, bound, w, K_max)
    
    
    P_E_plus_or_minus = rho_E_minus_small_t_NORM_omega_gamma_with_w_fn(t-t_E_aff-t_stim, gamma, omega, bound, w, K_max)
    
    return (P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum) + P_E_plus_or_minus*(1-C_A))


def cum_pro_and_reactive_trunc_fn(
        t, c_A_trunc_time,
        V_A, theta_A, t_A_aff,
        t_stim, t_E_aff, gamma, omega, w, K_max):
    c_A = cum_A_t_fn(t-t_A_aff, V_A, theta_A)
    if c_A_trunc_time is not None:
        if t < c_A_trunc_time:
            c_A = 0
        else:
            c_A  /= (1 - cum_A_t_fn(c_A_trunc_time - t_A_aff, V_A, theta_A))

    c_E = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t - t_stim - t_E_aff, gamma, omega, 1, w, K_max) + \
        CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t - t_stim - t_E_aff, gamma, omega, -1, w, K_max)
    
    return c_A + c_E - c_A * c_E


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


##################################################
############### time varying evidence ############
#################################################
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


################################################
########### PDF and CDF time vary ##############
###############################################
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


#################################################
######## Pro + reactive time vary PDF ###########
#################################################

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

    P_A = rho_A_t_fn(t - t_A_aff, V_A, theta_A)
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
    
    
    phi_t_e = phi_t_fn(t - t_E_aff - t_stim, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    int_phi_t_e = int_phi_fn(t - t_E_aff - t_stim, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

    P_E_plus = rho_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t-t_E_aff-t_stim, gamma, omega, bound, phi_t_e, int_phi_t_e, w, K_max)
    
    C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
    

    P_up = (P_A*(random_readout_if_EA_surives + P_E_plus_cum) + P_E_plus*(1-C_A))

    return P_up

###

def cum_pro_and_reactive_time_vary_fn(t, V_A, theta_A, t_A_aff, gamma, omega, t_stim, t_E_aff, w, phi_params, K_max = 10):

    c_A = cum_A_t_fn(t-t_A_aff, V_A, theta_A)

    int_phi_t_E_g = int_phi_fn(t - t_stim - t_E_aff, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    c_E = CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t - t_stim - t_E_aff,gamma,omega,1,int_phi_t_E_g,w,K_max) + \
        CDF_E_minus_small_t_NORM_omega_gamma_time_varying_fn(t - t_stim - t_E_aff,gamma, omega,-1,int_phi_t_E_g,w,K_max)
    
    return c_A + c_E - c_A * c_E