import numpy as np
from scipy.special import erf

def prob_fix_survive_and_hit(t, t_stim_0, t_stim_tau, V_A, theta_A, t_A_aff):
    if t < t_stim_0:
        fix_survive = 1
    else:
        fix_survive = np.exp(- ( (t - t_stim_0) / t_stim_tau ))

    return fix_survive * rho_A_t_fn(t-t_A_aff, V_A, theta_A)

def prob_fix_not_survive_an_censored_pdf(t, t_stim_0, t_stim_tau, V_A, theta_A, t_A_aff, t_stim):
    return (1 - cum_A_t_fn(t_stim-t_A_aff, V_A, theta_A))


def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

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