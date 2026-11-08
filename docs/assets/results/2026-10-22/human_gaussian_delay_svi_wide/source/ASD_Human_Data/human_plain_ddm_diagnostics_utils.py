# %%
"""Stable, choice-marginalized Wiener calculations for human fit diagnostics.

Dimensionless bounds are -1/+1, start is 2*w-1, drift is gamma, and
dimensionless time is omega*(RT-delay). These are the same Gamma/Omega
coordinates as the repository's IPL and condition-wise DDM primitives.
Image sums are used at short times and eigenfunction sums at long times.
Log calculations avoid the legacy density floor during optimization.
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp, log_ndtr
from scipy.special import log_ndtr as scipy_log_ndtr


def ddm_log_density(rt, gamma, omega, w, delay, K=40, side=None):
    t = jnp.asarray(rt)-delay
    tt = jnp.where(t > 0, omega*t, 1.)
    small = tt < .6
    ts = jnp.where(small, tt, .4)
    tl = jnp.where(small, 1., tt)
    pieces = []
    for bound in ([-1, 1] if side is None else [side]):
        v = jnp.where(jnp.asarray(bound) == 1, -gamma, gamma)
        wb = jnp.where(jnp.asarray(bound) == 1, 1.-w, w)
        ts, tl, v, wb, om = jnp.broadcast_arrays(ts, tl, v, wb, omega)
        paths = wb[..., None]+2*jnp.arange(-K//2, K//2+1)
        image, image_sign = logsumexp(
            jnp.log(jnp.abs(paths))-2*paths**2/ts[..., None],
            b=jnp.sign(paths), axis=-1, return_sign=True)
        n = jnp.arange(1, K+1)
        coefficients = n*jnp.sin(n*jnp.pi*wb[..., None])
        eigen, eigen_sign = logsumexp(
            jnp.log(jnp.abs(coefficients))-n**2*jnp.pi**2*tl[..., None]/8,
            b=jnp.sign(coefficients), axis=-1, return_sign=True)
        short_log = (jnp.log(2*om)-.5*jnp.log(2*jnp.pi)-1.5*jnp.log(ts)
                     -2*v*wb-v**2*ts/2+image)
        long_log = jnp.log(jnp.pi*om/4)-2*v*wb-v**2*tl/2+eigen
        value = jnp.where(small, short_log, long_log)
        sign = jnp.where(small, image_sign, eigen_sign)
        pieces.append(jnp.where((t > 0) & (sign > 0), value, -jnp.inf))
    return logsumexp(jnp.stack(pieces), axis=0)


def ddm_log_cdf(rt, gamma, omega, w, delay, K=40):
    """Log mass at either bound by RT; stable even for tiny retained mass."""
    t = jnp.asarray(rt)-delay
    tt = jnp.where(t > 0, omega*t, 1.)
    small = tt < .6
    ts = jnp.where(small, tt, .4)
    tl = jnp.where(small, 1., tt)
    cdf_pieces, survival_pieces = [], []
    k = jnp.arange(K+1)
    n = jnp.arange(1, K+1)
    for bound in [-1, 1]:
        v = jnp.where(bound == 1, -gamma, gamma)
        wb = jnp.where(bound == 1, 1.-w, w)
        ts, tl, v, wb = jnp.broadcast_arrays(ts, tl, v, wb)
        r = jnp.where(k % 2 == 0, 2*k+2*wb[..., None], 2*k+2*(1-wb[..., None]))
        pref = -2*v*wb
        term1 = pref[..., None]-v[..., None]*r+log_ndtr(
            (v[..., None]*ts[..., None]-r)/jnp.sqrt(ts)[..., None])
        term2 = pref[..., None]+v[..., None]*r+log_ndtr(
            (-v[..., None]*ts[..., None]-r)/jnp.sqrt(ts)[..., None])
        pair = jnp.logaddexp(term1, term2)
        image, sign = logsumexp(pair, b=(-1.)**k, axis=-1, return_sign=True)
        cdf_pieces.append(jnp.where(sign > 0, image, -jnp.inf))
        coefficients = n*jnp.sin(n*jnp.pi*wb[..., None])
        rate = v[..., None]**2/2+n**2*jnp.pi**2/8
        eigen, sign = logsumexp(
            jnp.log(jnp.abs(coefficients))-rate*tl[..., None]-jnp.log(rate),
            b=jnp.sign(coefficients), axis=-1, return_sign=True)
        survival_pieces.append(jnp.where(sign > 0, jnp.log(jnp.pi/4)+pref+eigen, -jnp.inf))
    image_cdf = logsumexp(jnp.stack(cdf_pieces), axis=0)
    log_survival = logsumexp(jnp.stack(survival_pieces), axis=0)
    eigen_cdf = jnp.log1p(-jnp.exp(log_survival))
    return jnp.where(t > 0, jnp.where(small, image_cdf, eigen_cdf), -jnp.inf)


def tied_coordinates(x, B, D, reference_B):
    """Invertible reparameterization: lambda, small-D Gamma slope, reference Omega."""
    rate, slope, reference_omega = jnp.exp(x[:3])
    theta = slope*17.37/rate
    arg = rate*D/17.37
    gamma = theta*jnp.tanh(arg)
    omega = reference_omega*jnp.exp(rate*jnp.log(10.)*(B-reference_B)/20)*jnp.cosh(arg)
    return gamma, omega, x[3], x[4]


def tied_physical(x, reference_B):
    rate, slope, reference_omega = np.exp(np.asarray(x)[:3])
    theta = slope*17.37/rate
    t0 = 10**(rate*reference_B/20)/(reference_omega*theta**2)
    return np.array([rate, t0, theta, x[3], x[4]])


def physical_to_tied(params, reference_B):
    rate, t0, theta, w, delay = params
    slope = theta*rate/17.37
    reference_omega = 10**(rate*reference_B/20)/(t0*theta**2)
    return np.array([np.log(rate), np.log(slope), np.log(reference_omega), w, delay])


def scipy_ddm(rt, gamma, omega, w, delay, cumulative=False, side=None, K=80):
    """Independent NumPy/SciPy calculation, using the image CDF directly."""
    rt, gamma, omega, w = np.broadcast_arrays(rt, gamma, omega, w)
    t = rt-delay
    tt = omega*np.where(t > 0, t, 1.)
    answer = np.zeros_like(tt)
    for bound in ([-1, 1] if side is None else [side]):
        v = np.where(bound == 1, -gamma, gamma)
        wb = np.where(bound == 1, 1-w, w)
        if cumulative:
            k = np.arange(K+1)
            r = np.where(k % 2 == 0, 2*k+2*wb[..., None], 2*k+2*(1-wb[..., None]))
            pref = -2*v[..., None]*wb[..., None]
            first = np.exp(pref-v[..., None]*r+scipy_log_ndtr(
                (v[..., None]*tt[..., None]-r)/np.sqrt(tt)[..., None]))
            second = np.exp(pref+v[..., None]*r+scipy_log_ndtr(
                (-v[..., None]*tt[..., None]-r)/np.sqrt(tt)[..., None]))
            answer += np.sum((-1.)**k*(first+second), axis=-1)
        else:
            small = tt < .6
            short_t = np.where(small, tt, .4)
            long_t = np.where(small, 1., tt)
            k = np.arange(-K//2, K//2+1)
            paths = wb[..., None]+2*k
            image = np.sum(paths*np.exp(-2*paths**2/short_t[..., None]), axis=-1)
            short = 2/np.sqrt(2*np.pi*short_t**3)*np.exp(-2*v*wb-v**2*short_t/2)*image
            n = np.arange(1, K+1)
            eigen = np.sum(n*np.sin(n*np.pi*wb[..., None])*np.exp(
                -n**2*np.pi**2*long_t[..., None]/8), axis=-1)
            long = np.pi/4*np.exp(-2*v*wb-v**2*long_t/2)*eigen
            answer += omega*np.where(small, short, long)
    return np.where(t > 0, answer, 0.)
