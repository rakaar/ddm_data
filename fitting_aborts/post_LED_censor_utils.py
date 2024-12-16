import numpy as np
from scipy.special import erf
from scipy.integrate import trapezoid as trapz
from scipy.integrate import quad

def rho_A_t_fn(t, V_A, theta_A):
    """
    For AI,prob density of t given V_A, theta_A
    """
    if t <= 0:
        return 0
    return (theta_A*1/np.sqrt(2*np.pi*(t)**3))*np.exp(-0.5 * (V_A**2) * (((t) - (theta_A/V_A))**2)/(t))


def P_t_x(x, t, v, a):
    """
    Prob that DV = x at time t given v, a 
    """
    if t <= 0:
        return 0
    return (1/np.sqrt(2 * (np.pi) * t)) * \
        ( np.exp(-((x - v*t)**2)/(2*t)) - np.exp( 2*v*a - ((x - 2*a - v*t)**2)/(2*t) ) )


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



    

def PDF_t_v_change(t, t_led, base_v, new_v, theta, aff):
    """
    Prob that proactive hits bound at time 't' with drift  change to new_v at t_led
    """
    if t <= 0:
        return 0
    else:
        if t < t_led:
            return rho_A_t_fn(t-aff, base_v, theta)
        else:
            # dx = 1e-4
            # x_pts = np.arange(-10, theta, dx)
            # y =  [P_t_x(x, t_led-aff, base_v, theta) * rho_A_t_fn(t - t_led, new_v, theta - x) for x in x_pts]
            # return trapz(y, x_pts)
            return quad(lambda x: P_t_x(x, t_led-aff, base_v, theta) * rho_A_t_fn(t - t_led, new_v, theta - x), -30, theta)[0]


def PDF_t_v_change_trunc_adj_fn(t, t_led, base_v, new_v, theta, aff, T_trunc, trunc_factor):
    if t <= T_trunc:
        return 0
    else:
        return PDF_t_v_change(t, t_led, base_v, new_v, theta, aff) / trunc_factor


def CDF_v_change_till_stim_trunc_adj_fn(t_stim, t_led, base_v, new_v, theta, aff, T_trunc, trunc_factor):
    """
    CDF till t of proactive process till t_stim with drift change at t_led
    """
    # dt = 1e-4
    # t_pts = np.arange(0, t_stim - aff, dt)
    # y = [PDF_t_v_change_trunc_adj_fn(t, t_led, base_v, new_v, theta, aff, T_trunc, trunc_factor) for t in t_pts]
    # return trapz(y, t_pts)
    return quad(PDF_t_v_change_trunc_adj_fn, 0, t_stim - aff, args=(t_led, base_v, new_v, theta, aff, T_trunc, trunc_factor))[0]

def CDF_v_change_till_trunc_fn(T_trunc, t_led, base_v, new_v, theta, aff):
    # dt = 0.01
    # t_pts = np.arange(0, T_trunc, dt)
    # y = [PDF_t_v_change(t, t_led, base_v, new_v, theta, aff) for t in t_pts]
    # return trapz(y, t_pts)    

    return quad(PDF_t_v_change, 0, T_trunc, args=(t_led, base_v, new_v, theta, aff))[0]
