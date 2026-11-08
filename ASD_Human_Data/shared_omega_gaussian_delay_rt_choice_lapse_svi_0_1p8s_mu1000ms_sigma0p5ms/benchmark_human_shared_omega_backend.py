import os,sys,time,pickle,json
backend=sys.argv[1]
os.environ['JAX_PLATFORM_NAME']=backend
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
sys.path.insert(0,'ASD_Human_Data')
import numpy as np
import jax
from jax import random,lax
from numpyro.infer import SVI,Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
import human_shared_omega_gaussian_delay_svi_utils as u
print('device',jax.devices()[0].device_kind,'x64',jax.config.x64_enabled,flush=True)
print('compute probe',float(jax.numpy.arange(10,dtype=jax.numpy.float64).sum()),flush=True)
_,selected,_,_=u.load_data()
_,_,data,initial,_=u.prepare_group(selected,'Control')
guide=AutoMultivariateNormal(u.guide_support_model,init_loc_fn=init_to_value(values=initial),init_scale=.1)
svi=SVI(u.gamma_omega_model,guide,ClippedAdam(.0002,clip_norm=1.),Trace_ELBO(num_particles=1))
svi.init(random.PRNGKey(0),data)
checkpoint=u.OUTPUT/'control/before_compilation_cache/latest_checkpoint.pkl'
with checkpoint.open('rb') as stream:saved=pickle.load(stream)
state=jax.tree.map(jax.device_put,saved['state'])
@jax.jit
def run_chunk(s):
 def body(s,_):return svi.stable_update(s,data)
 return lax.scan(body,s,None,length=100)
report=dict(backend=backend,device=jax.devices()[0].device_kind,x64=jax.config.x64_enabled,
 checkpoint=str(checkpoint),checkpoint_sha256=u.sha256(checkpoint),checkpoint_step=saved['step'],timings=[])
for repeat in range(2):
 t=time.monotonic();after,loss=run_chunk(state);loss=np.asarray(loss)
 elapsed=time.monotonic()-t
 report['timings'].append(elapsed)
 print('100 updates',repeat,'seconds',elapsed,'loss',float(loss.mean()),flush=True)
arrays={f'state_{i}':np.asarray(a) for i,a in enumerate(jax.tree.leaves(after))}
np.savez_compressed(f'/tmp/human_shared_omega_backend_{backend}.npz',loss=loss,**arrays)
from pathlib import Path
Path(f'/tmp/human_shared_omega_backend_{backend}.json').write_text(json.dumps(report,indent=2))
print('FINISHED',json.dumps(report),flush=True)
