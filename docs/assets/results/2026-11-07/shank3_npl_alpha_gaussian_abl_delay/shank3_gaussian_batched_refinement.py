# %% Numerically equivalent batched FFT refinement for the Gaussian rat race
"""Use the accepted grid/series/quadrature, batching only the FFT refinements.

Inputs have the reference's 30 expanded delay means/SDs. Direct rescaled
quadrature remains lazy and condition-specific for unresolved fast decisions.
"""
import jax
import jax.numpy as jnp
import shank3_npl_alpha_gaussian_utils as reference


def evidence_grid(params,dt=reference.GRID_DT,K=reference.K_DEFAULT,order=reference.QUAD_ORDER,direct=False):
    if direct:
        return reference.evidence_grid(params,dt,K,order,direct=True)
    time=jnp.linspace(0.,reference.GRID_END,round(reference.GRID_END/dt)+1)
    p=reference.expanded(params)
    pdf,cdf=reference.gaussian_delay_grid(p,time,K)
    fast=reference.needs_time_rescaling(p['gamma'],p['omega'],p['w'],dt)
    def refine(_):
        fine_time=jnp.linspace(0.,reference.GRID_END,round(reference.GRID_END/dt)*5+1)
        pf,cf=reference.gaussian_delay_grid(p,fine_time,K)
        f=jnp.where(fast[:,None,None],pf[:,:,::5],pdf)
        c=jnp.where(fast[:,None,None],cf[:,:,::5],cdf)
        finer=fast & reference.needs_time_rescaling(p['gamma'],p['omega'],p['w'],dt/5)
        def at25(_):
            t25=jnp.linspace(0.,reference.GRID_END,round(reference.GRID_END/dt)*25+1)
            p25,c25=reference.gaussian_delay_grid(p,t25,K)
            return jnp.where(finer[:,None,None],p25[:,:,::25],f),jnp.where(finer[:,None,None],c25[:,:,::25],c)
        f,c=jax.lax.cond(jnp.any(finer),at25,lambda _:(f,c),None)
        unresolved=finer & reference.needs_time_rescaling(p['gamma'],p['omega'],p['w'],dt/25)
        def quadrature(_):
            def condition(args):
                g,o,mu,sd,ff,cc,need=args
                def integrate(_):
                    def at(side):
                        def integrals(t):
                            return jnp.exp(reference.gaussian_delay_log_integral(t,g,o,p['w'],mu,sd,side,False,K,order)), \
                                   jnp.exp(reference.gaussian_delay_log_integral(t,g,o,p['w'],mu,sd,side,True,K,order))
                        return jax.checkpoint(integrals)(time)
                    fm,cm=at(-1);fp,cp=at(1)
                    return jnp.stack([fm,fp]),jnp.stack([cm,cp])
                return jax.lax.cond(need,integrate,lambda _:(ff,cc),None)
            return jax.lax.map(condition,(p['gamma'],p['omega'],p['delay_mu'],p['delay_sigma'],f,c,unresolved))
        return jax.lax.cond(jnp.any(unresolved),quadrature,lambda _:(f,c),None)
    pdf,cdf=jax.lax.cond(jnp.any(fast),refine,lambda _:(pdf,cdf),None)
    return time,pdf,cdf


def log_likelihood(params,data,marginal=False,dt=reference.GRID_DT,K=reference.K_DEFAULT,order=reference.QUAD_ORDER):
    time,pdf,cdf=evidence_grid(params,dt,K,order)
    ids,rt=data['condition'],data['rt']
    ef=reference.interpolate(pdf,time,rt,ids)
    ec=reference.interpolate(cdf,time,rt,ids)
    eg=reference.interpolate(cdf,time,rt+params['del_go'],ids)
    readout=.5*(1-eg.sum(-1))[...,None]+eg-ec
    ordinary=data['proactive_pdf'][...,None]*readout+data['proactive_survival'][...,None]*ef
    joint=(1-data['lapse_prob'])*ordinary+.5*data['lapse_prob']*data['lapse_pdf'][...,None]
    observed=joint.sum(-1) if marginal else joint[jnp.arange(rt.size),(data['choice']==1).astype(int)]
    p=reference.expanded(params)
    endpoint=jnp.exp(reference.gaussian_delay_log_integral(1.,p['gamma'][:,None],p['omega'][:,None],p['w'],
        p['delay_mu'][:,None],p['delay_sigma'][:,None],jnp.array([-1,1])[None,:],True,K,order)).sum(-1)
    mass=(1-data['lapse_prob'])*(data['survival_onset']-data['survival_end']*(1-endpoint[ids]))+data['lapse_prob']*data['lapse_window_mass']
    return jnp.sum(jnp.log(jnp.maximum(observed,1e-300))-jnp.log(jnp.maximum(mass,1e-300)))
