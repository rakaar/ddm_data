# %%
"""RT+choice diagnostic on the same human rows as the RT-only DDM comparison."""
import os
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from human_reactive_tied_rt_only_utils import ROOT,load_trials,group_data
from human_plain_ddm_diagnostics_utils import ddm_log_density,ddm_log_cdf,tied_coordinates,tied_physical

# %% Configuration and identical retained trials.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'ASD_Human_Data/reactive_ddm_debug_0_1p8s_delay300ms')
parser.add_argument('--maxiter',type=int,default=1600)
args=parser.parse_args()
frame,_=load_trials(0.,1.8)

for group in ['Control','ASD']:
    folder=args.output_dir/group.lower()
    if (folder/'joint_fits.json').exists():
        print(group,'joint fits already saved',flush=True)
        continue
    previous=json.loads((folder/'fits.json').read_text())
    sub,conditions,data,_=group_data(frame,group)
    B,D=data['condition_B'],data['condition_D']
    rt,choice,ids=data['rt'],data['choice'],data['condition']
    reference_B=previous['reference_B']
    upper=previous['causal_upper_s']
    rows=[];fits={}
    for family in ['tied','free_ddm']:
        free=family=='free_ddm'
        def objective(x):
            if free:
                gamma=jnp.sinh(x[:20]);omega=jnp.exp(x[20:40]);w,delay=x[40:42]
            else:
                gamma,omega,w,delay=tied_coordinates(x,B,D,reference_B)
            logpdf=ddm_log_density(rt,gamma[ids],omega[ids],w,delay,side=choice)
            mass=ddm_log_cdf(1.8,gamma,omega,w,delay)
            return -jnp.sum(logpdf-mass[ids])
        evaluate=jax.jit(jax.value_and_grad(objective))
        if not free:
            bounds=[(np.log(1e-4),np.log(10.)),(np.log(1e-4),np.log(1000.)),
                    (np.log(1e-4),np.log(1000.)),(.05,.95),(0.,upper)]
            starts=[np.array(previous['tied']['x']),
                    np.array([np.log(.2),np.log(.8),np.log(1.5),.5,.03])]
        else:
            bounds=[(-np.arcsinh(100.),np.arcsinh(100.))]*20+[(np.log(1e-4),np.log(100.))]*20+[(.05,.95),(0.,upper)]
            g,o,w,delay=tied_coordinates(jnp.array(fits['tied']['x']),B,D,reference_B)
            starts=[np.r_[np.arcsinh(np.array(g)),np.log(np.array(o)),float(w),float(delay)]]
            prob=(conditions.right.to_numpy()+.5)/(conditions.n.to_numpy()+1)
            g=.5*np.log(prob/(1-prob))
            moments=sub.groupby('condition_index').RT_s.mean().to_numpy()
            for delay in [0.,.7*upper]:
                ratio=np.where(abs(g)>1e-6,np.tanh(g)/np.where(abs(g)>1e-6,g,1.),1.)
                omega=ratio/(moments-delay)
                starts.append(np.r_[np.arcsinh(g),np.log(omega),.5,delay])
            old=np.array(previous['free_ddm']['x'])
            starts.append(np.r_[np.arcsinh(np.sign(np.array(D))*np.exp(old[:20])),old[20:]])
        best=None
        for i,start in enumerate(starts):
            started=time.monotonic();trace=[]
            start=np.clip(start,np.array(bounds)[:,0]+1e-10,np.array(bounds)[:,1]-1e-10)
            def fun(x):
                loss,grad=evaluate(jnp.asarray(x));loss=float(loss);grad=np.asarray(grad)
                assert np.isfinite(loss) and np.isfinite(grad).all(),(family,loss,x,grad)
                return loss,grad
            def checkpoint(x):trace.append(dict(iteration=len(trace)+1,negative_log_likelihood=fun(x)[0]))
            initial=fun(start)[0]
            print(group,'joint',family,'start',i,'NLL',initial,flush=True)
            result=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=bounds,callback=checkpoint,
                            options=dict(maxiter=args.maxiter,ftol=1e-12,gtol=1e-6,maxls=45,maxcor=20))
            row=dict(Group=group,family=family,start_index=i,initial_nll=initial,final_nll=float(result.fun),
                     iterations=int(result.nit),success=bool(result.success),message=str(result.message),
                     seconds=time.monotonic()-started)
            rows.append(row)
            pd.DataFrame(trace).to_csv(folder/f'joint_{family}_start{i}_trace.csv',index=False)
            np.savez_compressed(folder/f'joint_{family}_start{i}.npz',x=result.x,gradient=result.jac,start=start)
            print(group,'joint',family,'final',result.fun,'iterations',result.nit,result.message,flush=True)
            if best is None or result.fun<best['nll']:
                best=dict(x=result.x.tolist(),nll=float(result.fun),start=i,iterations=int(result.nit),
                          success=bool(result.success),message=str(result.message))
        fits[family]=best
        pd.DataFrame(rows).to_csv(folder/'joint_optimizer_runs.csv',index=False,float_format='%.17g')
    fits.update(group=group,n_trials=len(sub),tied_physical=tied_physical(fits['tied']['x'],reference_B).tolist(),
                likelihood='Joint RT and observed choice; either-bound normalization on 0--1.8 s',
                method='Direct maximum likelihood, no priors or variational approximation')
    (folder/'joint_fits.json').write_text(json.dumps(fits,indent=2)+'\n')
    print(group,'joint diagnostics saved',flush=True)
