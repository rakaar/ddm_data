import os,sys,time,pickle,json
os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'ASD_Human_Data')
import numpy as np
import jax
from jax import random,lax
from numpyro.infer import SVI,Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import ClippedAdam
import human_shared_omega_gaussian_delay_svi_utils as u
_,selected,_,_=u.load_data()
_,_,data,initial,_=u.prepare_group(selected,'Control')
guide=AutoMultivariateNormal(u.guide_support_model,init_loc_fn=init_to_value(values=initial),init_scale=.1)
svi=SVI(u.gamma_omega_model,guide,ClippedAdam(.0002,clip_norm=1.),Trace_ELBO(num_particles=1))
svi.init(random.PRNGKey(0),data)
with (u.OUTPUT/'control/latest_checkpoint.pkl').open('rb') as f:saved=pickle.load(f)
state=jax.tree.map(jax.device_put,saved['state'])
@jax.jit
def run_chunk(s):
 def body(s,_):return svi.stable_update(s,data)
 return lax.scan(body,s,None,length=100)
results={}
for name in ['old_first','old_second']:
 t=time.monotonic()
 if name.startswith('old'):
  result=svi.run(random.PRNGKey(0),100,data,init_state=state,progress_bar=False,stable_update=True)
  after,loss=result.state,result.losses
 else:after,loss=run_chunk(state)
 loss=np.asarray(loss)
 results[name]=dict(elapsed_s=time.monotonic()-t,loss=loss,after=jax.device_get(after))
 print(name,results[name]['elapsed_s'],flush=True)
for name in results:
 np.testing.assert_array_equal(results['old_first']['loss'],results[name]['loss'])
 for a,b in zip(jax.tree.leaves(results['old_first']['after']),jax.tree.leaves(results[name]['after'])):np.testing.assert_array_equal(a,b)
print('EXACT state/loss match for 100 updates; saved checkpoint',saved['step'],flush=True)
print(json.dumps({k:v['elapsed_s'] for k,v in results.items()}),flush=True)
