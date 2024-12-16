import numpy as np
from scipy.special import erf
from scipy.integrate import trapezoid as trapz
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
    if t < t_led:
        return rho_A_t_fn(t-aff, base_v, theta)
    else:
        dx = 0.01
        x_pts = np.arange(-10, theta, dx)
        y =  [P_t_x(x, t_led-aff, base_v, theta) * rho_A_t_fn(t - t_led, new_v, theta - x) for x in x_pts]
        return trapz(y, x_pts)


    
def CDF_proactive_till_stim_fn(t_stim, t_led, base_v, new_v, theta, aff):
    """
    CDF till t of proactive process till t_stim with drift change at t_led
    """
    dt = 0.01
    t_pts = np.arange(0, t_stim - aff, dt)
    y = [PDF_t_v_change(t, t_led, base_v, new_v, theta, aff) for t in t_pts]
    return trapz(y, t_pts)

