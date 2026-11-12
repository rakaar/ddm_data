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


def gaussian_delay_log_integral(rt, gamma, omega, w, mu, sigma, side,
                                cumulative=False, K=8, order=16):
    """Positive composite quadrature on log(decision time).

    Breakpoints resolve BOTH the DDM's short-time scale and the Gaussian peak.
    This handles boundary starts where a uniform FFT grid misses first-passage
    mass. The omitted interval has image exponent below -1024. For a CDF,
    integrate the same DDM density against the truncated delay CDF.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    t, gamma, omega, w, mu, sigma, side = jnp.broadcast_arrays(
        *[jnp.asarray(x) for x in (rt,gamma,omega,w,mu,sigma,side)])
    ts = jnp.where(t>0,t,1.)
    distance = 2*jnp.where(side==1,1-w,w)
    mode = 2*distance**2/(omega*(3+jnp.sqrt(9+4*(gamma*distance)**2)))
    low = jnp.minimum(distance**2/(2048*omega),ts*1e-8)
    low = jnp.maximum(low,1e-300)
    candidates = [low,ts,.6/omega, mode*.01,mode,mode*100]
    for z in [-10.,-3.,0.,3.,10.]:
        candidates.append(ts-mu-z*sigma)
    breaks = jnp.sort(jnp.log(jnp.clip(jnp.stack(candidates,axis=-1),low[...,None],ts[...,None])),axis=-1)
    half = (breaks[...,1:]-breaks[...,:-1])/2
    middle = (breaks[...,1:]+breaks[...,:-1])/2
    z = middle[...,None]+half[...,None]*jnp.asarray(nodes)
    decision = jnp.exp(z)
    delay = jnp.maximum(ts[...,None,None]-decision,0.)
    lf = ddm_log_density(decision,gamma[...,None,None],omega[...,None,None],w[...,None,None],0.,K,
                         side=side[...,None,None])
    if cumulative:
        upper=log_ndtr((delay-mu[...,None,None])/sigma[...,None,None])
        lower=log_ndtr(-mu[...,None,None]/sigma[...,None,None])
        # The finite lower cap avoids a 0/0 derivative at coincident endpoints;
        # its absolute probability is at most 5e-16, below the integration gate.
        lg=upper+jnp.log(-jnp.expm1(jnp.minimum(lower-upper,-1e-15)))
        lg-=log_ndtr(mu[...,None,None]/sigma[...,None,None])
    else:
        lg=(-.5*((delay-mu[...,None,None])/sigma[...,None,None])**2-jnp.log(sigma[...,None,None])
            -.5*jnp.log(2*jnp.pi)-log_ndtr(mu[...,None,None]/sigma[...,None,None]))
    logweight=jnp.log(jnp.maximum(half,1e-300))[...,None]+jnp.log(jnp.asarray(weights))+z
    terms=jnp.where(half[...,None]>0,lf+lg+logweight,-jnp.inf)
    answer=logsumexp(terms,axis=(-2,-1))
    return jnp.where(t>0,answer,-jnp.inf)


def needs_time_rescaling(gamma,omega,w,dt):
    distance=2*jnp.minimum(w,1-w)
    shortest_mode=2*distance**2/(omega*(3+jnp.sqrt(9+4*(gamma*distance)**2)))
    return shortest_mode < 10*dt


def gaussian_delay_trial_logpdf(params,data,time,K=8,order=24,lapse_floor=None):
    """Conditional joint per-trial log densities, before any lapse mixture.

    Normalization always uses positive rescaled quadrature, independently of
    the FFT. Conditions with unresolved first-passage times use quadrature for
    trial densities as well. Small FFT tails use the same positive fallback.
    """
    names=('gamma','omega','w','delay_mu','delay_sigma')
    p=dict(zip(names,jnp.broadcast_arrays(*[jnp.atleast_1d(params[n]) for n in names])))
    ga,om,w,mu,sd=[p[n] for n in names]
    bounds=jnp.array([-1,1])[None,:]
    hi=gaussian_delay_log_integral(data['window_high'],ga[:,None],om[:,None],w[:,None],
                                  mu[:,None],sd[:,None],bounds,True,K,order)
    # Current analysis uses lower endpoint zero, but retain a proper window API.
    lo=gaussian_delay_log_integral(data['window_low'],ga[:,None],om[:,None],w[:,None],
                                  mu[:,None],sd[:,None],bounds,True,K,order)
    logmass=logsumexp(hi+jnp.log1p(-jnp.exp(jnp.minimum(lo-hi,-1e-15))),axis=1)
    density,_=gaussian_delay_grid(params,time,K)
    ids=data['condition']; side=(data['choice']==1).astype(int)
    position=jnp.clip(data['rt']/(time[1]-time[0]),0.,time.size-1.)
    index=jnp.minimum(jnp.floor(position).astype(int),time.size-2)
    frac=position-index
    # Cubic Hermite interpolation resolves narrow Gaussian delays without
    # doubling the FFT grid. At endpoints use the adjacent one-sided slope.
    y0=density[ids,side,index];y1=density[ids,side,index+1]
    ym=density[ids,side,jnp.maximum(index-1,0)]
    yp=density[ids,side,jnp.minimum(index+2,time.size-1)]
    m0=jnp.where(index>0,(y1-ym)/2,y1-y0)
    m1=jnp.where(index<time.size-2,(yp-y0)/2,y1-y0)
    observed=jnp.maximum((2*frac**3-3*frac**2+1)*y0+(frac**3-2*frac**2+frac)*m0
                         +(-2*frac**3+3*frac**2)*y1+(frac**3-frac**2)*m1,0.)
    use_direct=needs_time_rescaling(ga,om,w,time[1]-time[0])[ids]
    scale=jnp.max(density,axis=(1,2))[ids]
    small_tail=observed < jnp.maximum(1e-12,scale*1e-10)
    if lapse_floor is not None:
        # A uniform lapse already dominates numerical FFT tails. Only skip the
        # tail integral when a conservative roundoff bound is <1e-4 of that
        # mixture floor. Fast-decision fallback is never skipped by this rule.
        small_tail &= lapse_floor < scale*jnp.exp(-logmass[ids])*1e-9
    use_direct|=small_tail
    def direct_rows(_):
        def row(args):
            t,y,need,f,g,o,b,m,s=args
            return jax.lax.cond(need,
                lambda _:gaussian_delay_log_integral(t,g,o,b,m,s,y,False,K,order),
                lambda _:jnp.log(jnp.maximum(f,1e-300)),None)
        return jax.lax.map(row,(data['rt'],data['choice'],use_direct,observed,ga[ids],om[ids],
                               w[ids],mu[ids],sd[ids]))
    logpdf=jax.lax.cond(jnp.any(use_direct),direct_rows,
                       lambda _:jnp.log(jnp.maximum(observed,1e-300)),None)
    return logpdf-logmass[ids],logmass


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
    # Cumulative Simpson integration prevents an O(dt^2) CDF error for narrow
    # delay distributions. All fitting grids have an even number of intervals.
    if (n-1)%2==0:
        a,b,c=density[...,0:-2:2],density[...,1:-1:2],density[...,2::2]
        first=dt*(5*a+8*b-c)/12
        second=dt*(-a+8*b+5*c)/12
        increments=jnp.stack([first,second],axis=-1).reshape(density.shape[:-1]+(n-1,))
    else:
        increments=.5*dt*(density[...,:-1]+density[...,1:])
    cumulative=jnp.concatenate([jnp.zeros_like(density[...,:1]),jnp.cumsum(increments,axis=-1)],axis=-1)
    cumulative=jnp.maximum(cumulative,0.)
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
