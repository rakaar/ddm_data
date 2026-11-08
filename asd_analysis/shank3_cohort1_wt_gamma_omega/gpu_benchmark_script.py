# %% Benchmark copies of existing states; never write to either fit directory.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['JAX_PLATFORMS']='cuda,cpu'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{12,13,14,15,28,29,30,31})
import sys,json,pickle,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'asd_analysis'))
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI,Trace_ELBO,init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
from shank3_wt_gamma_omega_utils import OUT,model,log_likelihood

STEPS=20
REPEATS=2
cpu=jax.devices('cpu')[0]
gpu=jax.devices('gpu')[0]
print('GPU:',gpu.device_kind,flush=True)
print('GPU memory:',gpu.memory_stats(),flush=True)
raw=dict(np.load(OUT/'likelihood_data.npz'))
raw.pop('onset')
report={'gpu':gpu.device_kind,'float64':True,'timed_steps_per_repeat':STEPS,'repeats':REPEATS,'runs':[]}
for variant in ['shared_delay','condition_delay']:
    folder=OUT/variant
    config=json.loads((folder/'configuration.json').read_text())
    saved=pickle.loads((folder/'latest_checkpoint.pkl').read_bytes())
    row={'variant':variant,'checkpoint_step':saved['step']}
    for backend,device in [('gpu',gpu),('cpu',cpu)]:
        with jax.default_device(device):
            data={k:jax.device_put(x,device) for k,x in raw.items()}
            initial={k:jax.device_put(np.array(x),device) for k,x in config['initial'].items()}
            guide=AutoMultivariateNormal(model,init_loc_fn=init_to_value(values=initial),init_scale=.1)
            svi=SVI(model,guide,numpyro.optim.ClippedAdam(.0002,clip_norm=1),Trace_ELBO())
            svi.init(jax.random.PRNGKey(0),data,variant)
            state=jax.tree_util.tree_map(lambda x:jax.device_put(x,device) if isinstance(x,(np.ndarray,jax.Array)) else x,saved['state'])
            # The same fixed sample sequence and physical posterior means.
            samples=guide.sample_posterior(jax.random.PRNGKey(99173),svi.get_params(state),sample_shape=(2048,))
            parameters={k:x.mean(0) for k,x in samples.items()}
            vg=jax.jit(jax.value_and_grad(lambda p:log_likelihood(p,data,variant)))
            val,grad=vg(parameters)
            jax.block_until_ready((val,grad))
            row[backend+'_log_likelihood']=float(val)
            row[backend+'_gradient']={k:np.asarray(x).tolist() for k,x in grad.items()}
            row[backend+'_means']={k:np.asarray(x).tolist() for k,x in parameters.items()}
            assert np.isfinite(val) and all(np.isfinite(np.asarray(x)).all() for x in grad.values())
            def scan(s):
                return jax.lax.scan(lambda current,_:svi.stable_update(current,data,variant),s,None,length=STEPS)
            runner=jax.jit(scan)
            tick=time.perf_counter()
            warm=runner(state)
            jax.block_until_ready(warm)
            row[backend+'_compile_and_warmup_seconds']=time.perf_counter()-tick
            print(variant, backend, 'compiled and warmed up', row[backend+'_compile_and_warmup_seconds'], flush=True)
            times=[];averages=[]
            for _ in range(REPEATS):
                tick=time.perf_counter()
                candidate,losses=runner(state)
                jax.block_until_ready((candidate,losses))
                times.append((time.perf_counter()-tick)/STEPS)
                assert np.isfinite(np.asarray(losses)).all()
                averages.append(float(jnp.mean(losses)))
            row[backend+'_seconds_per_step']=times
            row[backend+'_mean_loss']=averages
            print(variant,backend,'seconds/update',times,'mean loss',averages,flush=True)
    np.testing.assert_allclose(row['cpu_log_likelihood'],row['gpu_log_likelihood'],atol=1e-6,rtol=1e-10)
    for name in row['cpu_gradient']:
        np.testing.assert_allclose(row['cpu_gradient'][name],row['gpu_gradient'][name],atol=1e-5,rtol=1e-7)
    row['speedup']=float(np.median(row['cpu_seconds_per_step'])/np.median(row['gpu_seconds_per_step']))
    row['likelihood_and_gradient_agree']=True
    report['runs'].append(row)
    (OUT/'gpu_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2),flush=True)
print('Measured speedups:',[(r['variant'],r['speedup']) for r in report['runs']],flush=True)
