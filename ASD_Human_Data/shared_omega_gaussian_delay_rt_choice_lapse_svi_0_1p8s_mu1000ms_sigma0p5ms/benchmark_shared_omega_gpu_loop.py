import os,sys,time,pickle,json
os.environ['JAX_PLATFORM_NAME']='gpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false';os.environ['OPENBLAS_NUM_THREADS']='1'
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
reports={}
for group in ['Control','ASD']:
 _,_,data,initial,_=u.prepare_group(selected,group)
 guide=AutoMultivariateNormal(u.guide_support_model,init_loc_fn=init_to_value(values=initial),init_scale=.1)
 svi=SVI(u.gamma_omega_model,guide,ClippedAdam(.0002,clip_norm=1.),Trace_ELBO(num_particles=1))
 svi.init(random.PRNGKey(0),data)
 path=u.OUTPUT/group.lower()/'before_gpu/latest_checkpoint.pkl'
 saved=pickle.loads(path.read_bytes());state=jax.tree.map(jax.device_put,saved['state'])
 @jax.jit
 def fixed(s):
  def body(s,_):return svi.stable_update(s,data)
  return lax.scan(body,s,None,length=1000)
 t=time.monotonic();result=svi.run(random.PRNGKey(0),1000,data,init_state=state,progress_bar=False,stable_update=True);expected=np.asarray(result.losses);ordinary=time.monotonic()-t
 t=time.monotonic();after,loss=fixed(state);loss=np.asarray(loss);first=time.monotonic()-t
 t=time.monotonic();after,loss=fixed(state);loss=np.asarray(loss);warm=time.monotonic()-t
 np.testing.assert_allclose(loss,expected,rtol=1e-10,atol=1e-9)
 errors=[]
 for a,b in zip(jax.tree.leaves(after),jax.tree.leaves(result.state)):
  a,b=np.asarray(a),np.asarray(b);np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-9);errors.append(float(np.max(np.abs(a.astype(float)-b.astype(float)))))
 reports[group]=dict(checkpoint_step=saved['step'],checkpoint_sha256=u.sha256(path),updates=1000,ordinary_seconds=ordinary,first_compiled_seconds=first,warm_seconds=warm,maximum_loss_difference=float(abs(loss-expected).max()),maximum_state_difference=max(errors),passed=True)
 print(group,reports[group],flush=True)
(u.OUTPUT/'gpu_compiled_loop_benchmark.json').write_text(json.dumps(reports,indent=2)+'\n')
