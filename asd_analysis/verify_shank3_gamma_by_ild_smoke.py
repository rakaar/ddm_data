# %% Isolated 30-trial resume test; this is never a fitted scientific result
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI,Trace_ELBO,init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal
from shank3_gamma_by_ild_utils import OUT,VARIANTS,EXPECTED,model,parameter_names

# Run the driver to a checkpoint, corrupt only the derived test export, and resume.
# The independent uninterrupted calculation below checks the committed state.
import subprocess
from pathlib import Path
for group in EXPECTED:
    for variant in VARIANTS:
        folder=OUT/'validation_smoke'/group/variant
        if not (folder/'run_summary.json').exists():
            command=[str(Path(__file__).resolve().parents[1]/'.venv/bin/python'),
                     str(Path(__file__).with_name('fit_shank3_gamma_by_ild.py')),
                     '--group',group,'--variant',variant,'--smoke-test']
            subprocess.run(command+['--checkpoint-only-after','5'],check=True)
            np.save(folder/'losses.npy',np.array([np.nan]))
            subprocess.run(command,check=True)

reports=[]
for group in EXPECTED:
    full={k:jnp.asarray(v) for k,v in np.load(OUT/group/'likelihood_data.npz').items() if k!='onset'}
    ids=np.array([np.flatnonzero(np.asarray(full['condition'])==i)[0] for i in range(30)])
    data={k:(v[ids] if v.ndim else v) for k,v in full.items()}
    for variant in VARIANTS:
        folder=OUT/'validation_smoke'/group/variant
        config=json.loads((OUT/group/variant/'prepared_configuration.json').read_text())
        initial={k:jnp.asarray(v) for k,v in config['initial'].items()}
        guide=AutoMultivariateNormal(model,init_loc_fn=init_to_value(values=initial),init_scale=.1)
        svi=SVI(model,guide,numpyro.optim.ClippedAdam(.0002,clip_norm=1),Trace_ELBO())
        state=svi.init(jax.random.PRNGKey(0),data,variant)
        advance=jax.jit(lambda s:jax.lax.scan(lambda s,_:svi.stable_update(s,data,variant),s,None,length=5))
        losses=[]
        for _ in range(3):
            state,window=advance(state)
            losses.extend(np.asarray(window).tolist())
        saved_losses=np.load(folder/'losses.npy')
        np.testing.assert_allclose(saved_losses,losses,atol=1e-11,rtol=1e-13)
        latest=pickle.loads((folder/'latest_checkpoint.pkl').read_bytes())
        assert latest['step']==15 and len(latest['losses'])==15 and len(latest['rows'])==3
        for x,y in zip(jax.tree.leaves(state),jax.tree.leaves(latest['state'])):
            np.testing.assert_allclose(x,y,atol=1e-11,rtol=1e-13)
        samples=dict(np.load(folder/'posterior_samples.npz'))
        assert all(np.isfinite(v).all() for v in samples.values())
        assert sum(np.asarray(v).reshape(128,-1).shape[1] for v in samples.values())==len(parameter_names(variant))
        restored=pickle.loads((folder/'restored_guide_optimizer.pkl').read_bytes())
        best=latest['best']
        assert restored['step']==best['step']
        for x,y in zip(jax.tree.leaves(restored['params']),jax.tree.leaves(best['params'])):
            np.testing.assert_array_equal(x,y)
        reports.append(dict(group=group,variant=variant,resumed_updates=15,
                            losses_and_optimizer_match_uninterrupted_run=True,restored_best_verified=True))
        print(group+'/'+variant+': uninterrupted and interrupted/resumed states agree',flush=True)
(OUT/'smoke_validation.json').write_text(json.dumps(dict(status='passed',scientific_fit=False,
    trials_per_test=30,posterior_draws_per_test=128,runs=reports),indent=2)+'\n')
