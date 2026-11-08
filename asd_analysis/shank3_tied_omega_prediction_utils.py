# %% Exact-onset prediction primitives, reusable by fitted and isolated initial previews
import numpy as np
from scipy.stats import invgauss
from scipy.integrate import trapezoid
from numpy.polynomial.legendre import leggauss
import jax
import jax.numpy as jnp
from shank3_wt_gamma_omega_utils import evidence_bound

GRID = np.linspace(-1,1,2001)
VALID_GRID = np.linspace(0,1,1001)
EDGES = np.linspace(-1,1,101)
ABS_ILDS = [1,2,4,8,16]


@jax.jit
def evidence_terms(times,gamma,omega,delay,w,go):
    pr,fr,sr=evidence_bound(times-delay,gamma,omega,w,1)
    pl,fl,sl=evidence_bound(times-delay,gamma,omega,w,-1)
    _,fgr,sgr=evidence_bound(times-delay+go,gamma,omega,w,1)
    _,_,sgl=evidence_bound(times-delay+go,gamma,omega,w,-1)
    return pr+pl,sr+sl,pr,.5*(sgr+sgl)+fgr-fr


def average_fixed(times,onsets,fixed,weights,chunk=128):
    """Exact onset values and multiplicities; no rounding, resampling or animal means."""
    v,a,nd,p,beta=fixed
    result={k:np.empty(len(times)) for k in ['proactive_pdf','proactive_survival','lapse_pdf','lapse_survival']}
    for start in range(0,len(times),chunk):
        sl=slice(start,start+chunk)
        t=np.asarray(times)[sl,None]+onsets[None,:]
        quantities=[invgauss.pdf(t-nd,mu=1/(v*a),scale=a*a),invgauss.sf(t-nd,mu=1/(v*a),scale=a*a),
                    np.where(t>=0,beta*np.exp(-beta*np.maximum(t,0)),0.),np.exp(-beta*np.maximum(t,0))]
        for key,quantity in zip(result,quantities):
            result[key][sl]=quantity@weights
    return result


def valid_predictions(onsets,fixed,gamma,omega,delay,w,go):
    """Normalize each exact onset before trial-count averaging the complete mixture."""
    v,a,nd,p,beta=fixed
    onset,count=np.unique(onsets,return_counts=True)
    fraction=count/len(onsets)
    sa0=invgauss.sf(onset-nd,mu=1/(v*a),scale=a*a)
    sa1=invgauss.sf(onset+1-nd,mu=1/(v*a),scale=a*a)
    se1=float(evidence_terms(jnp.array([1.]),gamma,omega,delay,w,go)[1][0])
    mass=(1-p)*(sa0-sa1*se1)+p*(np.exp(-beta*onset)-np.exp(-beta*(onset+1)))
    assert np.all((mass>0)&(mass<=1))
    weights=fraction/mass
    cuts=sorted(set([0.,1.]+[t for t in [delay-go,delay] if 0<t<1]))
    results=[]
    for order in [128,256]:
        x,qw=leggauss(order)
        times=np.concatenate([(hi-lo)/2*x+(hi+lo)/2 for lo,hi in zip(cuts[:-1],cuts[1:])])
        quadrature=np.concatenate([(hi-lo)/2*qw for lo,hi in zip(cuts[:-1],cuts[1:])])
        avg=average_fixed(times,onset,fixed,weights)
        ep,es,pr,readout=map(np.asarray,evidence_terms(jnp.asarray(times),gamma,omega,delay,w,go))
        qr=(1-p)*(avg['proactive_pdf']*readout+avg['proactive_survival']*pr)+.5*p*avg['lapse_pdf']
        qt=(1-p)*(avg['proactive_pdf']*es+avg['proactive_survival']*ep)+p*avg['lapse_pdf']
        results.append(np.array([quadrature@qr,quadrature@qt,quadrature@(times*qt)]))
    np.testing.assert_allclose(results[0],results[1],atol=2e-8,rtol=2e-8)
    np.testing.assert_allclose(results[-1][1],1.,atol=2e-8)
    avg=average_fixed(VALID_GRID,onset,fixed,weights)
    ep,es,_,_=map(np.asarray,evidence_terms(jnp.asarray(VALID_GRID),gamma,omega,delay,w,go))
    cdf=(1-p)*((sa0*weights).sum()-avg['proactive_survival']*es)+p*((np.exp(-beta*onset)*weights).sum()-avg['lapse_survival'])
    density=(1-p)*(avg['proactive_pdf']*es+avg['proactive_survival']*ep)+p*avg['lapse_pdf']
    cdf_mean=trapezoid(1-cdf,VALID_GRID)
    np.testing.assert_allclose(cdf_mean,results[-1][2],atol=1e-6,rtol=0)
    np.testing.assert_allclose(cdf[[0,-1]],[0,1],atol=1e-12)
    return dict(p_right=results[-1][0],integrated_mass=results[-1][1],mean_rt_s=results[-1][2],
                cdf_mean_s=cdf_mean,quad_order_difference=float(np.max(np.abs(results[0]-results[1]))),
                cdf=cdf,density=density,onsets=onset,onset_counts=count,window_mass=mass)
