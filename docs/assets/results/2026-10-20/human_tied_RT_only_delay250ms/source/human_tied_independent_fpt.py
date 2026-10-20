# %%
"""Independent SciPy image-CDF and image/eigenfunction-PDF calculations."""
import numpy as np
from scipy.special import log_ndtr

def scipy_first_passage(rt, choice, B, D, params, cumulative=False):
    rate,t0,theta,w,delay = params
    t = np.asarray(rt)-delay
    gamma = theta*np.tanh(rate*np.asarray(D)/17.37)
    omega = 10**(rate*np.asarray(B)/20)*np.cosh(rate*np.asarray(D)/17.37)/(t0*theta**2)
    v = np.where(np.asarray(choice)==1, -gamma, gamma)
    wb = np.where(np.asarray(choice)==1, 1-w, w)
    tt, v, wb, om = np.broadcast_arrays(omega*np.maximum(t,1e-14),v,wb,omega)
    result = np.zeros_like(tt)
    if cumulative:
        k = np.arange(81)
        r = np.where(k%2==0,2*k+2*wb[...,None],2*k+2*(1-wb[...,None]))
        root = np.sqrt(tt)[...,None]
        vf = v[...,None]
        pref = -2*vf*wb[...,None]
        terms = np.exp(pref-vf*r+log_ndtr((vf*tt[...,None]-r)/root))
        terms += np.exp(pref+vf*r+log_ndtr((-vf*tt[...,None]-r)/root))
        result = np.sum((-1.)**k*terms,axis=-1)
    else:
        small = tt < .6
        if small.any():
            ts,vs,ws = tt[small],v[small],wb[small]
            k = np.arange(-40,41)
            paths = ws[:,None]+2*k
            image_sum = np.sum(paths*np.exp(-2*paths**2/ts[:,None]),axis=1)
            result[small] = 2/np.sqrt(2*np.pi*ts**3)*np.exp(-2*vs*ws-vs**2*ts/2)*image_sum
        if (~small).any():
            tl,vl,wl = tt[~small],v[~small],wb[~small]
            n = np.arange(1,81)
            spectral_sum = np.sum(n*np.sin(n*np.pi*wl[:,None])*np.exp(-n*n*np.pi**2*tl[:,None]/8),axis=1)
            result[~small] = np.pi/4*np.exp(-2*vl*wl-vl**2*tl/2)*spectral_sum
        result *= om
    return np.where(t>0,result,0.)
