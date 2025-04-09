import numpy as np
from scipy.special import erf, erfcx
from torch import Value


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
def M(x):
    """Mills ratio."""
    # clip
    x = np.clip(x, -37, 37)
    return np.sqrt(np.pi / 2) * erfcx(x / np.sqrt(2))

def phi(x):
    """Standard Gaussian function."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

def rho_E_minus_small_t_NORM_rate_norm_time_varying_fn(
        t, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, phi_t, int_phi_t, 
        rate_norm_l, is_norm, is_time_vary, K_max):
    """
    in normalized time, PDF of hitting the lower bound with gamma and omega, but time varying evidence
    """
    if t <= 0:
        return 0

    # evidence v
    chi = 17.37
    # v = rate_lambda * theta_E * ILD / chi # LINEAR
    # NON linear
    v = theta_E * np.tanh(rate_lambda * ILD / chi)

    w = 0.5 + ( Z_E / (2.0 * theta_E) )
    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    if not is_norm:
        rate_norm_l = 0
    # omega = ( 1/(T0 * (theta_E**2)) ) * (10 ** ( (rate_lambda * (1 - rate_norm_l) * ABL) / 20 ) )
    # non linear
    cosh_ratio = np.cosh(rate_lambda * ILD / chi)/np.cosh(rate_lambda * rate_norm_l * ILD / chi)
    omega = ( 1/(T0 * (theta_E**2)) ) * (10 ** ( (rate_lambda * (1 - rate_norm_l) * ABL) / 20 ) ) * cosh_ratio
    
    if not is_time_vary:
        int_phi_t = t
        phi_t = 1

    t = omega * int_phi_t

    if t == 0:
        raise ValueError(f't = {t}, for int_phi_t = {int_phi_t} or omega = {omega}')

    non_sum_term = (1/a**2)*(a**3/np.sqrt(2*np.pi*t**3))*np.exp(-v*a*w - (v**2 * t)/2)
    K_max = int(K_max/2)
    k_vals = np.linspace(-K_max, K_max, 2*K_max + 1)
    sum_w_term = w + 2*k_vals
    sum_exp_term = np.exp(-(a**2 * (w + 2*k_vals)**2)/(2*t))
    sum_result = np.sum(sum_w_term*sum_exp_term)

    
    density =  non_sum_term * sum_result
    if density <= 0:
        density = 1e-16

    return density * (omega * phi_t)

def CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
        t, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t, 
        rate_norm_l, is_norm, is_time_vary, K_max):
    """
    In normalized time, CDF of hitting the lower bound with gamma and omega, but time varying evidence
    """
    if t <= 0:
        return 0

    # evidence v
    chi = 17.37
    # v = rate_lambda * theta_E * ILD / chi # LINEAR
    # NON linear
    v = theta_E * np.tanh(rate_lambda * ILD / chi)
    

    w = 0.5 + ( Z_E / (2.0 * theta_E) )
    a = 2
    if bound == 1:
        v = -v
        w = 1 - w

    if not is_norm:
        rate_norm_l = 0
    # omega = ( 1/(T0 * (theta_E**2)) ) * (10 ** ( (rate_lambda * (1 - rate_norm_l) * ABL) / 20 ) ) # LINEAR
    # non linear
    cosh_ratio = np.cosh(rate_lambda * ILD / chi)/np.cosh(rate_lambda * rate_norm_l * ILD / chi)
    omega = ( 1/(T0 * (theta_E**2)) ) * (10 ** ( (rate_lambda * (1 - rate_norm_l) * ABL) / 20 ) ) * cosh_ratio

    if not is_time_vary:
        int_phi_t = t

    t = omega * int_phi_t

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
        
        if np.isnan(term2) or np.isinf(term2):
            print(f'omega = {omega}, T0 = {T0}, theta_E = {theta_E}, rate_lambda = {rate_lambda}')
            raise ValueError(f'term2 = {term2}, v = {v}, t = {t}, r_k = {r_k}, M(r_k - v*t) = {M((r_k - v * t) / np.sqrt(t))}, M(r_k + v*t) = {M((r_k + v * t) / np.sqrt(t))}')
        
        summation += ((-1)**k) * term1 * term2

    return (result*summation)


def up_or_down_RTs_fit_fn(
        t, bound,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff, del_go,
        phi_params, rate_norm_l, 
        is_norm, is_time_vary, K_max):
    
    # t1, t2 - if proactive wins, time range in which EA can hit bound and confirm a choice
    t1 = max(t - t_stim - t_E_aff, 1e-6)
    t2 = max(t - t_stim - t_E_aff + del_go, 1e-6)

    # phi(t) and its integral for different times
    if is_time_vary:
        int_phi_t_E_g = int_phi_fn(max(t - t_stim - t_E_aff + del_go, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        phi_t_e = phi_t_fn(max(t - t_stim - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
        int_phi_t_e = int_phi_fn(max(t - t_stim - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        int_phi_t2 = int_phi_fn(t2, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
        int_phi_t1 = int_phi_fn(t1, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        if int_phi_t_E_g * int_phi_t_e * int_phi_t2 * int_phi_t1 == 0:
            raise ValueError(
                f'''
                t = {t}, t_stim = {t_stim}, t_E_aff = {t_E_aff}
                t1 = {t1}
                one of them is zero
                int_phi_t_E_g = {int_phi_t_E_g}
                int_phi_t_e = {int_phi_t_e}
                int_phi_t2 = {int_phi_t2}
                int_phi_t1 = {int_phi_t1}

                params  = {phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2}
                '''
                    
                )
    else:
        int_phi_t_E_g = np.nan
        
        phi_t_e = np.nan
        int_phi_t_e = np.nan

        int_phi_t2 = np.nan
        int_phi_t1 = np.nan

    # PA wins and random choice due to EA survival
    P_A = rho_A_t_fn(t - t_A_aff, V_A, theta_A)
    P_EA_hits_either_bound = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
                                t - t_stim - t_E_aff + del_go, 1, 
                                ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E_g, rate_norm_l, 
                                is_norm, is_time_vary, K_max)  \
                                + \
                                CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
                                t - t_stim - t_E_aff + del_go, -1, 
                                ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E_g, rate_norm_l,
                                is_norm, is_time_vary, K_max)
    
    P_EA_survives = 1 - P_EA_hits_either_bound
    random_readout_if_EA_survives = 0.5 * P_EA_survives
    
    # PA wins and EA hits later
    P_E_plus_cum = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
                            t2, bound, 
                            ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t2, rate_norm_l, 
                            is_norm, is_time_vary, K_max)  \
                            - \
                            CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
                            t1, bound, 
                            ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t1, rate_norm_l,
                            is_norm, is_time_vary, K_max)
    

    # EA wins
    P_E_plus = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn(
        t - t_stim - t_E_aff, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, phi_t_e, int_phi_t_e, 
        rate_norm_l, is_norm, is_time_vary, K_max)
    

    
    C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
    return (P_A*(random_readout_if_EA_survives + P_E_plus_cum) + P_E_plus*(1-C_A))


def up_or_down_RTs_fit_wrt_stim_fn(
        t, bound,
        P_A, C_A,
        t_stim, ABL, ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff, del_go,
        phi_params, rate_norm_l, 
        is_norm, is_time_vary, K_max):
    
    # t1, t2 - if proactive wins, time range in which EA can hit bound and confirm a choice
    t1 = max(t - t_E_aff, 1e-6)
    t2 = max(t - t_E_aff + del_go, 1e-6)

    # phi(t) and its integral for different times
    if is_time_vary:
        int_phi_t_E_g = int_phi_fn(max(t - t_E_aff + del_go, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        phi_t_e = phi_t_fn(max(t - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
        int_phi_t_e = int_phi_fn(max(t - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        int_phi_t2 = int_phi_fn(t2, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
        int_phi_t1 = int_phi_fn(t1, phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)

        if int_phi_t_E_g * int_phi_t_e * int_phi_t2 * int_phi_t1 == 0:
            raise ValueError(
                f'''
                t = {t}, t_stim = {t_stim}, t_E_aff = {t_E_aff}
                t1 = {t1}
                one of them is zero
                int_phi_t_E_g = {int_phi_t_E_g}
                int_phi_t_e = {int_phi_t_e}
                int_phi_t2 = {int_phi_t2}
                int_phi_t1 = {int_phi_t1}

                params  = {phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2}
                '''
                    
                )
    else:
        int_phi_t_E_g = np.nan
        
        phi_t_e = np.nan
        int_phi_t_e = np.nan

        int_phi_t2 = np.nan
        int_phi_t1 = np.nan

    # PA wins and random choice due to EA survival
    P_EA_hits_either_bound = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
                                t - t_E_aff + del_go, 1, 
                                ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E_g, rate_norm_l, 
                                is_norm, is_time_vary, K_max)  \
                                + \
                                CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
                                t - t_E_aff + del_go, -1, 
                                ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E_g, rate_norm_l,
                                is_norm, is_time_vary, K_max)
    
    P_EA_survives = 1 - P_EA_hits_either_bound
    random_readout_if_EA_survives = 0.5 * P_EA_survives
    
    # PA wins and EA hits later
    P_E_plus_cum = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
                            t2, bound, 
                            ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t2, rate_norm_l, 
                            is_norm, is_time_vary, K_max)  \
                            - \
                            CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
                            t1, bound, 
                            ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t1, rate_norm_l,
                            is_norm, is_time_vary, K_max)
    

    # EA wins
    P_E_plus = rho_E_minus_small_t_NORM_rate_norm_time_varying_fn(
        t - t_E_aff, bound, ABL, ILD, rate_lambda, T0, theta_E, Z_E, phi_t_e, int_phi_t_e, 
        rate_norm_l, is_norm, is_time_vary, K_max)
    

    return (P_A*(random_readout_if_EA_survives + P_E_plus_cum) + P_E_plus*(1-C_A))

def cum_pro_and_reactive_time_vary_fn(
        t,
        V_A, theta_A, t_A_aff,
        t_stim, ABL, ILD, rate_lambda, T0, theta_E, Z_E, t_E_aff,
        phi_params, rate_norm_l, 
        is_norm, is_time_vary, K_max):

    c_A = cum_A_t_fn(t-t_A_aff, V_A, theta_A)

    if is_time_vary:
        int_phi_t_E = int_phi_fn(t - t_stim - t_E_aff, \
                        phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
    else:
        int_phi_t_E = np.nan
    c_E = CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
                                t - t_stim - t_E_aff, 1, 
                                ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E, rate_norm_l, 
                                is_norm, is_time_vary, K_max)  \
                                + \
                                CDF_E_minus_small_t_NORM_rate_norm_l_time_varying_fn(
                                t - t_stim - t_E_aff, -1, 
                                ABL, ILD, rate_lambda, T0, theta_E, Z_E, int_phi_t_E, rate_norm_l,
                                is_norm, is_time_vary, K_max)
    
    return c_A + c_E - c_A * c_E