# %%
"""Standard DDM control: one common time scale and 20 free stimulus drifts."""
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
from human_plain_ddm_diagnostics_utils import ddm_log_density,ddm_log_cdf,tied_coordinates

# %% Unit diffusion, a common boundary/noise time scale, shared start and delay.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'ASD_Human_Data/reactive_ddm_debug_0_1p8s_delay300ms')
args=parser.parse_args()
frame,_=load_trials(0.,1.8)
for group in ['Control','ASD']:
    folder=args.output_dir/group.lower()
    if (folder/'shared_scale_fits.json').exists():
        print(group,'shared-scale controls already saved',flush=True)
        continue
    previous=json.loads((folder/'fits.json').read_text())
    joint=json.loads((folder/'joint_fits.json').read_text())
    sub,conditions,data,_=group_data(frame,group)
    rt,choice,ids=data['rt'],data['choice'],data['condition']
    B,D=data['condition_B'],data['condition_D']
    rows=[];fits={}
    for mode in ['rt_only','joint']:
        def objective(x):
            gamma=jnp.sinh(x[:20]);omega=jnp.exp(x[20]);w,delay=x[21:23]
            lp=ddm_log_density(rt,gamma[ids],omega,w,delay,side=choice if mode=='joint' else None)
            mass=ddm_log_cdf(1.8,gamma,omega,w,delay)
            return -jnp.sum(lp-mass[ids])
        evaluate=jax.jit(jax.value_and_grad(objective))
        source=previous if mode=='rt_only' else joint
        g,o,w,delay=tied_coordinates(jnp.array(source['tied']['x']),B,D,previous['reference_B'])
        starts=[np.r_[np.arcsinh(np.array(g)),float(np.log(np.array(o)).mean()),float(w),float(delay)]]
        prob=(conditions.right.to_numpy()+.5)/(conditions.n.to_numpy()+1)
        g=.5*np.log(prob/(1-prob))
        if mode=='rt_only':g=np.sign(np.array(D))*10.
        starts.append(np.r_[np.arcsinh(g),np.log(.15 if mode=='rt_only' else 1.5),.5,.02])
        if mode=='rt_only':
            old=np.array(previous['free_ddm']['x'])
            g=np.sign(np.array(D))*np.exp(old[:20])
        else:
            old=np.array(joint['free_ddm']['x']);g=np.sinh(old[:20])
        starts.append(np.r_[np.arcsinh(g),old[20:40].mean(),old[40:42]])
        bounds=[(-np.arcsinh(100.),np.arcsinh(100.))]*20+[(np.log(1e-4),np.log(100.)),(.05,.95),(0.,previous['causal_upper_s'])]
        best=None
        for i,start in enumerate(starts):
            started=time.monotonic();trace=[]
            def fun(x):
                value,grad=evaluate(jnp.asarray(x));value=float(value);grad=np.asarray(grad)
                assert np.isfinite(value) and np.isfinite(grad).all()
                return value,grad
            def checkpoint(x):trace.append(dict(iteration=len(trace)+1,negative_log_likelihood=fun(x)[0]))
            initial=fun(start)[0]
            print(group,'shared-scale',mode,'start',i,'NLL',initial,flush=True)
            result=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=bounds,callback=checkpoint,
                            options=dict(maxiter=1800,ftol=1e-12,gtol=1e-6,maxls=45,maxcor=20))
            rows.append(dict(Group=group,mode=mode,start_index=i,initial_nll=initial,final_nll=float(result.fun),
                             iterations=int(result.nit),success=bool(result.success),message=str(result.message),
                             seconds=time.monotonic()-started))
            pd.DataFrame(trace).to_csv(folder/f'shared_{mode}_start{i}_trace.csv',index=False)
            np.savez_compressed(folder/f'shared_{mode}_start{i}.npz',x=result.x,gradient=result.jac,start=start)
            print(group,'shared-scale',mode,'final',result.fun,'iterations',result.nit,result.message,flush=True)
            if best is None or result.fun<best['nll']:
                best=dict(x=result.x.tolist(),nll=float(result.fun),start=i,success=bool(result.success),
                          iterations=int(result.nit),message=str(result.message))
        fits[mode]=best
        pd.DataFrame(rows).to_csv(folder/'shared_optimizer_runs.csv',index=False,float_format='%.17g')
    fits.update(group=group,n_parameters=23,meaning='20 free Gamma, 1 shared Omega, shared w and delay; fixed unit diffusion',
                method='Direct maximum likelihood on the identical retained rows; priors are not used')
    (folder/'shared_scale_fits.json').write_text(json.dumps(fits,indent=2)+'\n')
