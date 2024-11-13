import numpy as np
from scipy.integrate import quad

# --- PDF --- #
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
    return (1/np.sqrt(2 * (np.pi) * t)) * \
        ( np.exp(-((x - v*t)**2)/(2*t)) - np.exp( 2*v*a - ((x - 2*a - v*t)**2)/(2*t) ) )


def P_old_at_x_times_P_new_hit(x, t, V_A_old, V_A_new, a, t_LED):
    """
    Prob that DV is at x at t_LED and new V_A hits "a-x" bound at t - t_LED
    """
    return P_t_x(x, t_LED, V_A_old, a) * rho_A_t_fn(t-t_LED, V_A_new, a - x)



def PDF_hit_V_A_change(t, V_A_old, V_A_new, a, t_LED):
    """
    PDF of RT of hitting single bound with V_A change at t_LED
    """
    if t <= t_LED:
        p = rho_A_t_fn(t, V_A_old, a)
    else:
        p = quad(P_old_at_x_times_P_new_hit, -np.inf, a, args=(t, V_A_old, V_A_new, a, t_LED))[0]
    
    return p


# --- CDF ----
from scipy.special import erf
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

def P_old_at_x_times_CDF_new_hit(x, t, V_A_old, V_A_new, a, t_LED):
    """
    Prob that DV is at x at t_LED times CDF new V_A hits "a-x" bound at t - t_LED
    A helper func for CDF_hit_V_A_change
    """
    return P_t_x(x, t_LED, V_A_old, a) * cum_A_t_fn(t - t_LED, V_A_new, a - x)

def CDF_hit_V_A_change(t, V_A_old, V_A_new, a, t_LED):
    """
    CDF of hitting bound with V_A change at t_LED
    """
    if t <= t_LED:
        return cum_A_t_fn(t, V_A_old, a)
    else:
        return cum_A_t_fn(t_LED, V_A_old, a) + quad(P_old_at_x_times_CDF_new_hit, -10, a, args=(t, V_A_old, V_A_new, a, t_LED))[0]

