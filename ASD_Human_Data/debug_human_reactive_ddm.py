# %%
"""Diagnose pooled human TIED fits against free condition-wise Wiener DDMs.

Direct maximum likelihood is a diagnostic, not a variational posterior rerun.
Every model uses the same RT-only, stimulus-conditional objective and rows.
"""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import argparse
import json
import time
import sys
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from human_reactive_tied_rt_only_utils import (
    ROOT, PARAM_NAMES, load_trials, group_data, log_likelihood, sha256,
)
from human_plain_ddm_diagnostics_utils import (
    ddm_log_density, ddm_log_cdf, tied_coordinates, tied_physical, physical_to_tied,
)

# %% Configuration: one new diagnostic root; original fits are read-only.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-dir', type=Path, default=ROOT/'ASD_Human_Data/reactive_tied_rt_only_svi_0_1p8s_delay300ms')
parser.add_argument('--output-dir', type=Path, default=ROOT/'ASD_Human_Data/reactive_ddm_debug_0_1p8s_delay300ms')
parser.add_argument('--group', choices=['all', 'Control', 'ASD'], default='all')
parser.add_argument('--maxiter', type=int, default=1600)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
frame, provenance = load_trials(0., 1.8)
baseline_config = json.loads((args.baseline_dir/'run_config.json').read_text())
for path, digest in baseline_config['source_hashes'].items():
    assert sha256(ROOT/path) == digest, path
if not (out/'baseline_sha256.json').exists():
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in args.baseline_dir.rglob('*') if p.is_file()}
    (out/'baseline_sha256.json').write_text(json.dumps(protected, indent=2)+'\n')
protocol = dict(
    selection='Condition == RT and 0 <= RT_s <= 1.8; all correct/error choices; observed response side ignored',
    rt_window_s=[0., 1.8], delay_hard_bound_s=[0., .3],
    causal_delay_support='Delay strictly below minimum retained group RT; same support as the existing likelihood',
    likelihood='log[f_right(RT)+f_left(RT)] - log[P(either bound by 1.8 s)] per retained trial',
    pooling='Raw trials across participants/sessions within Control/ASD; 20 signed stimulus conditions remain distinct',
    fit_kind='Direct maximum likelihood, without priors; diagnostic for optimizer/variational/model restrictions',
    tied='Same five parameters, invertibly reparameterized by log(lambda), log(Gamma small-D slope), log(reference Omega), w, delay',
    free_ddm='20 free signed Gamma and 20 free Omega, shared w and delay: 42 parameters per group',
    drift_sign='Gamma sign follows signed Delta_dB; RT-only drift directions are not identified by response observations',
    computational_bounds=dict(rate_lambda=[1e-4,10.], gamma_slope=[1e-4,1000.], reference_omega=[1e-4,1000.],
                              abs_gamma=[1e-4,100.], omega=[1e-4,100.], w=[.05,.95]),
    optimizer='SciPy L-BFGS-B, JAX float64 exact gradients, multi-start',
    K_max=40, maxiter=args.maxiter, seed=args.seed,
    baseline_dir=str(args.baseline_dir.resolve()), data=provenance,
    posterior_curves='Keep original arithmetic physical posterior means; separately compare posterior predictive curves',
)
(out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
frame.to_csv(out/'selected_trials.csv', index=False, float_format='%.17g')
frame.groupby('Group').agg(n=('RT_s','size'),participants=('Participant','nunique'),
                         minimum_RT_s=('RT_s','min')).to_csv(out/'selection_counts.csv')
print('Diagnostic output:',out,flush=True)

# %% Keep the conditional trial likelihood identical; remove TIED constraints only in the free DDM.
for group in (['Control','ASD'] if args.group == 'all' else [args.group]):
    folder = out/group.lower()
    folder.mkdir(exist_ok=True)
    if (folder/'fits.json').exists():
        print(group,'already completed; preserving saved fit',flush=True)
        continue
    sub, conditions, data, _ = group_data(frame, group)
    data.pop('choice')
    data.update(rt_lower=0., rt_upper=1.8)
    conditions.to_csv(folder/'conditions.csv', index=False, float_format='%.17g')
    rt, ids = data['rt'], data['condition']
    B, D = data['condition_B'], data['condition_D']
    reference_B = float(np.mean(np.asarray(B)))
    causal_upper = min(.3,float(sub.RT_s.min())-1e-7)
    signs = jnp.sign(D)
    baseline_samples = np.load(args.baseline_dir/group.lower()/'posterior_samples.npz')
    baseline_mean = np.array([baseline_samples[p].mean() for p in PARAM_NAMES])
    baseline_x = physical_to_tied(baseline_mean,reference_B)
    rng = np.random.default_rng(args.seed)

    def objective(x, free=False):
        if free:
            gamma = signs*jnp.exp(x[:20])
            omega = jnp.exp(x[20:40])
            w, delay = x[40],x[41]
        else:
            gamma, omega, w, delay = tied_coordinates(x,B,D,reference_B)
        logpdf = ddm_log_density(rt,gamma[ids],omega[ids],w,delay)
        logmass = ddm_log_cdf(1.8,gamma,omega,w,delay)
        return -jnp.sum(logpdf-logmass[ids])

    fits, rows = {}, []
    for family in ['tied','free_ddm']:
        free = family == 'free_ddm'
        evaluate = jax.jit(jax.value_and_grad(lambda x:objective(x,free)))
        if not free:
            bounds = [(np.log(1e-4),np.log(10)),(np.log(1e-4),np.log(1000)),
                      (np.log(1e-4),np.log(1000)),(.05,.95),(0.,causal_upper)]
            starts = [baseline_x.copy()]
            for slope,omega,w,delay in [(.8,1.6,.5,.03),(2.,1.,.48,.08),(4.,.5,.52,.02),(.2,2.,.5,.01)]:
                starts.append(np.array([np.log(.2),np.log(slope),np.log(omega),w,min(delay,.8*causal_upper)]))
        else:
            bounds = [(np.log(1e-4),np.log(100.))]*20+[(np.log(1e-4),np.log(100.))]*20+[(.05,.95),(0.,causal_upper)]
            tied_best = fits['tied']['x']
            g,o,w,d = tied_coordinates(jnp.asarray(tied_best),B,D,reference_B)
            starts = [np.r_[np.log(np.maximum(abs(np.asarray(g)),1e-4)),np.log(np.asarray(o)),float(w),float(d)]]
            # Inverse-Gaussian moment estimates initialize high-drift RT-shape fits.
            # They are starting values only; all fits use the two-bound Wiener likelihood.
            moments = sub.groupby('condition_index').RT_s.agg(['mean','var'])
            for delay,w in [(0.,.5),(.05,.5),(.8*causal_upper,.5)]:
                decision_mean = moments['mean'].to_numpy()-delay
                g = np.clip(decision_mean**2/moments['var'].to_numpy(),.2,40.)
                o = 1/(g*decision_mean)
                starts.append(np.r_[np.log(g),np.log(o),w,delay])
        best = None
        for start_index,start in enumerate(starts):
            start = np.clip(start,np.array(bounds)[:,0]+1e-10,np.array(bounds)[:,1]-1e-10)
            record = []
            started = time.monotonic()
            def fun(x):
                value, gradient = evaluate(jnp.asarray(x))
                value,gradient=float(value),np.asarray(gradient,dtype=float)
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError((group,family,start_index,value,x,gradient))
                return value,gradient
            def checkpoint(x):
                value,_ = fun(x)
                record.append(dict(iteration=len(record)+1,negative_log_likelihood=value))
            initial_value = fun(start)[0]
            print(group,family,'start',start_index,'initial NLL',initial_value,flush=True)
            result = minimize(fun,start,jac=True,method='L-BFGS-B',bounds=bounds,callback=checkpoint,
                              options=dict(maxiter=args.maxiter,ftol=1e-12,gtol=1e-6,maxls=45,maxcor=20))
            value,gradient = fun(result.x)
            elapsed = time.monotonic()-started
            row=dict(Group=group,family=family,start_index=start_index,initial_nll=initial_value,
                     final_nll=value,iterations=int(result.nit),evaluations=int(result.nfev),
                     success=bool(result.success),message=str(result.message),seconds=elapsed)
            rows.append(row)
            pd.DataFrame(record).to_csv(folder/f'{family}_start{start_index}_trace.csv',index=False)
            np.savez_compressed(folder/f'{family}_start{start_index}.npz',x=result.x,gradient=gradient,start=start)
            print(group,family,'start',start_index,'final NLL',value,'iterations',result.nit,result.message,flush=True)
            if best is None or value < best['nll']:
                best=dict(x=result.x,nll=value,start=start_index,success=bool(result.success),
                          message=str(result.message),iterations=int(result.nit),gradient=gradient)
        fits[family]=best
        pd.DataFrame(rows).to_csv(folder/'optimizer_runs.csv',index=False,float_format='%.17g')
        if family == 'tied':
            p=tied_physical(best['x'],reference_B)
            legacy_ll=float(log_likelihood(jnp.asarray(p),data))
            print(group,'TIED MLE physical parameters',dict(zip(PARAM_NAMES,p)),flush=True)
            print('Stable vs legacy log likelihood:',-best['nll'],legacy_ll,flush=True)
            assert abs(legacy_ll+best['nll']) < 1e-5
    serial = {}
    for family,result in fits.items():
        serial[family]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items()}
    serial.update(group=group,n_trials=len(sub),reference_B=reference_B,causal_upper_s=causal_upper,
                  baseline_mean_parameters=baseline_mean.tolist(),
                  baseline_mean_legacy_log_likelihood=float(log_likelihood(jnp.asarray(baseline_mean),data)),
                  tied_physical=tied_physical(fits['tied']['x'],reference_B).tolist())
    (folder/'fits.json').write_text(json.dumps(serial,indent=2)+'\n')
    print(group,'all likelihood fits saved',flush=True)

# %% Verify that the baseline posterior/results have not changed.
for path,digest in json.loads((out/'baseline_sha256.json').read_text()).items():
    assert sha256(ROOT/path)==digest,path
print('All baseline files preserved.',flush=True)
