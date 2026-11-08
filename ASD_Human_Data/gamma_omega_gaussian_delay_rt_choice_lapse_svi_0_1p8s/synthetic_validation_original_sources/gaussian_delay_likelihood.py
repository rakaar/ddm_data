# %%
"""Reactive Gamma/Omega DDM with an independent random delay on every trial.

Delay ~ Normal(mu, sigma) conditioned on delay >= 0, with no upper cutoff.
mu/sigma describe the underlying Gaussian, not its truncated moments and not
posterior uncertainty. The response time is delay + first-passage time.

The fast method uses a zero-padded FFT for the convolution on a uniform grid.
It returns joint (RT, choice) densities, in choice order [-1, +1], and CDFs.
Conditioning on an RT window happens AFTER integrating over random delays.
The default K=8 is validated against K=40/80 and the original log-space kernel.
This is a numerical approximation: validate grid refinement for new regimes,
especially very narrow delays, extreme bias, or very fast decision times.
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp

from human_plain_ddm_diagnostics_utils import ddm_log_density


def ddm_grid_density(time, gamma, omega, w, K=8):
    """Image/eigen series from ddm_log_density, evaluated in density space.

    FFT convolution needs densities, so exponentiating signed log-sums only
    adds work. The same t*omega=0.6 switch keeps direct summation well behaved.
    Validation compares this kernel to the existing log-space implementation.
    Inputs have a condition axis; output axes are condition, choice, time.
    """
    tau = omega[:, None, None]*time[None, None, :]
    short = tau < .6
    ts, tl = jnp.where(short, tau, .4), jnp.where(short, 1., tau)
    v = jnp.stack([gamma, -gamma], axis=1)[:, :, None]
    wb = jnp.stack([w, 1-w], axis=1)[:, :, None]
    paths = wb[..., None]+2*jnp.arange(-K//2, K//2+1)
    image_exponent = (-2*paths**2/ts[..., None]-2*v[..., None]*wb[..., None]
                      - v[..., None]**2*ts[..., None]/2)
    image = jnp.sum(paths*jnp.exp(image_exponent), axis=-1)
    small_pdf = 2*omega[:, None, None]/jnp.sqrt(2*jnp.pi*ts**3)*image
    n = jnp.arange(1, K+1)
    eigen_exponent = (-n**2*jnp.pi**2*tl[..., None]/8
                      -2*v[..., None]*wb[..., None]-v[..., None]**2*tl[..., None]/2)
    eigen = jnp.sum(n*jnp.sin(n*jnp.pi*wb[..., None])*jnp.exp(eigen_exponent), axis=-1)
    large_pdf = jnp.pi*omega[:, None, None]/4*eigen
    return jnp.where(short, small_pdf, large_pdf)


def gaussian_delay_grid(params, time, K=8):
    """Convolve each condition once; time is equally spaced and starts at zero.

    params: vectors gamma, omega, delay_mu, delay_sigma; scalar or vector w.
    Returns arrays (condition, choice, time). Unconditional density is retained:
    mass beyond time[-1] is NOT silently renormalized away.
    """
    gamma, omega, w, mu, sigma = jnp.broadcast_arrays(*[
        jnp.atleast_1d(params[name])
        for name in ('gamma', 'omega', 'w', 'delay_mu', 'delay_sigma')])
    dt = time[1] - time[0]
    # Avoid differentiating log(0) at the zero-time grid node.
    safe_time = jnp.where(time > 0, time, dt)
    f = ddm_grid_density(safe_time, gamma, omega, w, K)
    f = f.at[:, :, 0].set(0.)
    delay_logpdf = (-.5 * ((time[None, :] - mu[:, None]) / sigma[:, None])**2
                    - jnp.log(sigma[:, None]) - .5*jnp.log(2*jnp.pi)
                    - log_ndtr(mu[:, None] / sigma[:, None]))
    g = jnp.exp(delay_logpdf)
    # Trapezoid endpoint at delay=0; the other endpoint multiplies f(0)=0.
    g = g.at[:, 0].multiply(.5)
    n = time.shape[0]
    fft_size = 1 << (2*n-2).bit_length()
    raw = jnp.fft.irfft(jnp.fft.rfft(f, n=fft_size, axis=-1)
                       * jnp.fft.rfft(g[:, None, :], n=fft_size, axis=-1),
                       n=fft_size, axis=-1)[..., :n] * dt
    # FFT roundoff can create ~1e-15 negative values in almost-empty tails.
    # The likelihood below uses positive log-quadrature in that regime.
    density = jnp.maximum(raw, 0.).at[:, :, 0].set(0.)
    cumulative = jnp.concatenate([
        jnp.zeros_like(density[..., :1]),
        jnp.cumsum(.5*dt*(density[..., :-1]+density[..., 1:]), axis=-1)], axis=-1)
    return density, cumulative


def gaussian_delay_logpdf_quadrature(rt, gamma, omega, w, mu, sigma, side,
                                     K=8, order=96):
    """Positive log-space quadrature for an independent accuracy/tail fallback.

    Integrate over delay in [0, RT] using fixed Gauss-Legendre nodes; sigma > 0.
    This remains differentiable. Narrow distributions may need higher order.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    t, gamma, omega, w, mu, sigma, side = jnp.broadcast_arrays(
        rt, gamma, omega, w, mu, sigma, side)
    t_safe = jnp.where(t > 0, t, 1.)
    delay = t_safe[..., None] * jnp.asarray((nodes+1)/2)
    log_f = ddm_log_density(
        t_safe[..., None]-delay, gamma[..., None], omega[..., None],
        w[..., None], 0., K, side=side[..., None])
    log_g = (-.5*((delay-mu[..., None])/sigma[..., None])**2
             - jnp.log(sigma[..., None]) - .5*jnp.log(2*jnp.pi)
             - log_ndtr(mu[..., None]/sigma[..., None]))
    value = (logsumexp(log_f+log_g+jnp.log(jnp.asarray(weights)), axis=-1)
             + jnp.log(t_safe/2))
    return jnp.where(t > 0, value, -jnp.inf)


def gaussian_delay_log_likelihood(params, data, time, K=8, marginal=False):
    """Retained-window RT+choice or RT-only objective, with one Z per condition.

    data contains rt, choice (+/-1), condition, window_low, window_high.
    The time grid must cover [0, window_high]. No prior or fit is prescribed.
    Rare, very small FFT densities use direct positive quadrature so that a
    numerical density floor cannot reward an impossible early response.
    """
    density, cumulative = gaussian_delay_grid(params, time, K)
    lo, hi = data['window_low'], data['window_high']
    mass = jax.vmap(lambda c: jnp.interp(hi, time, c)-jnp.interp(lo, time, c))(
        cumulative.sum(axis=1))
    ids = data['condition']
    position = jnp.clip((data['rt']-time[0])/(time[1]-time[0]), 0., time.size-1.)
    index = jnp.minimum(jnp.floor(position).astype(int), time.size-2)
    fraction = position-index
    if marginal:
        curves = density.sum(axis=1)
        first, second = curves[ids, index], curves[ids, index+1]
    else:
        choices = (data['choice'] == 1).astype(int)
        first, second = density[ids, choices, index], density[ids, choices, index+1]
    observed = first*(1-fraction)+second*fraction
    scale = jnp.max(density, axis=(1, 2))[ids]
    fallback = observed < jnp.maximum(1e-12, scale*1e-10)

    def with_fallback(values):
        # lax.map avoids allocating (trials x quadrature x series) for all rows.
        p = {name: jnp.broadcast_to(jnp.atleast_1d(params[name]), mass.shape)[ids]
             for name in ('gamma', 'omega', 'w', 'delay_mu', 'delay_sigma')}
        def row(args):
            t, y, need, val, ga, om, w, mu, sd = args
            def direct(_):
                if marginal:
                    lr = gaussian_delay_logpdf_quadrature(t, ga, om, w, mu, sd, 1, K)
                    ll = gaussian_delay_logpdf_quadrature(t, ga, om, w, mu, sd, -1, K)
                    return jnp.logaddexp(lr, ll)
                return gaussian_delay_logpdf_quadrature(t, ga, om, w, mu, sd, y, K)
            return jax.lax.cond(need, direct, lambda _: jnp.log(jnp.maximum(val, 1e-300)), None)
        return jax.lax.map(row, (data['rt'], data['choice'], fallback, values,
            p['gamma'], p['omega'], p['w'], p['delay_mu'], p['delay_sigma']))

    logpdf = jax.lax.cond(jnp.any(fallback), with_fallback,
                         lambda v: jnp.log(jnp.maximum(v, 1e-300)), observed)
    terms = logpdf-jnp.log(mass[ids])
    inside = (data['rt'] >= lo) & (data['rt'] <= hi)
    return jnp.sum(jnp.where(inside, terms, -jnp.inf))
