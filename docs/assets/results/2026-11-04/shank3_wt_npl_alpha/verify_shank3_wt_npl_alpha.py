# %% Independent selection, NumPy/SciPy likelihood, and saved-fit verification
"""Run --pre-fit before GPU fitting; default also verifies the posterior and plots."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad, trapezoid
from scipy.stats import invgauss
import jax
import jax.numpy as jnp
from shank3_wt_npl_alpha_utils import (ROOT, OUT, REFERENCE, ABLS, ILDS, INITIAL, PRIORS,
    expand_parameters, log_likelihood, parameter_names, sample_matrix, model, loss_window_decision, sha256)
from shank3_wt_gamma_omega_utils import choice_density, fixed_at, evidence_bound, log_likelihood as race_log_likelihood
sys.path.insert(0,str(ROOT/'ASD_Human_Data'))
from human_plain_ddm_diagnostics_utils import scipy_ddm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pre-fit', action='store_true')
args = parser.parse_args()
prep = json.loads((OUT/'preparation.json').read_text())
fixed = np.array(json.loads((OUT/'fixed_proactive.json').read_text())['values'])
conditions = pd.read_csv(OUT/'conditions.csv', float_precision='round_trip')
saved = pd.read_csv(OUT/'fit_trials.csv.gz', float_precision='round_trip')
parent = pd.read_csv(OUT/'parent_trials.csv.gz', float_precision='round_trip')
data = {k:jnp.asarray(v) for k,v in np.load(OUT/'likelihood_data.npz').items() if k != 'onset'}


def numpy_mapping(params):
    abl = np.repeat(ABLS,10)
    ild = np.tile(ILDS,3)
    lam, ell, alpha = float(params['rate_lambda']), float(params['rate_norm_l']), float(params['alpha'])
    abl_gain = np.power(10., lam*(1-ell)*abl/20)
    right = abl_gain*np.exp(lam*ild/17.37)/(np.exp(lam*ell*ild/17.37)+alpha*np.exp(-lam*ell*ild/17.37))
    left = abl_gain*np.exp(-lam*ild/17.37)/(np.exp(-lam*ell*ild/17.37)+alpha*np.exp(lam*ell*ild/17.37))
    return float(params['theta_E'])*(right-left)/(right+left), (right+left)/(float(params['T_0'])*float(params['theta_E'])**2)


def scipy_race(rt, onset, gamma, omega, delay, w, go, side):
    v,a,nd,p,beta = fixed
    evidence_rt, evidence_side = np.asarray(rt), np.asarray(side)
    rt,onset = np.broadcast_arrays(rt,onset)
    t = rt+onset
    ap = invgauss.pdf(t-nd,mu=1/(v*a),scale=a*a)
    ass = invgauss.sf(t-nd,mu=1/(v*a),scale=a*a)
    ep = scipy_ddm(evidence_rt,gamma,omega,w,delay,side=evidence_side)
    ef = scipy_ddm(evidence_rt,gamma,omega,w,delay,cumulative=True,side=evidence_side)
    eg = scipy_ddm(evidence_rt+go,gamma,omega,w,delay,cumulative=True,side=evidence_side)
    other = scipy_ddm(evidence_rt+go,gamma,omega,w,delay,cumulative=True,side=-evidence_side)
    return (1-p)*(ap*(.5*(1-eg-other)+eg-ef)+ass*ep)+.5*p*np.where(t>=0,beta*np.exp(-beta*np.maximum(t,0)),0)


def scipy_cdf(rt, onset, gamma, omega, delay, w):
    v,a,nd,p,beta = fixed
    t = np.asarray(rt)+onset
    sf = invgauss.sf(t-nd,mu=1/(v*a),scale=a*a)
    ef = scipy_ddm(np.asarray(rt),gamma,omega,w,delay,cumulative=True)
    return 1-(1-p)*sf*(1-ef)-p*np.exp(-beta*np.maximum(t,0))


def independent_likelihood(params):
    gamma,omega = numpy_mapping(params)
    result = 0.
    for row in conditions.itertuples():
        z = saved.loc[saved.condition.eq(row.condition)]
        i = row.condition
        c = z.onset_s.to_numpy()
        d = float(np.asarray(params['t_E_aff'])[i])
        q = scipy_race(z.timed_rt.to_numpy(),c,gamma[i],omega[i],d,float(params['w']),float(params['del_go']),z.response_poke.to_numpy())
        mass = scipy_cdf(1,c,gamma[i],omega[i],d,float(params['w']))-scipy_cdf(0,c,gamma[i],omega[i],d,float(params['w']))
        assert np.all(q>0) and np.all(mass>0) and np.all(mass<=1)
        result += np.sum(np.log(q)-np.log(mass))
    return float(result)


# %% Reconstruct the raw selection and preserve prior numerical/figure outputs
for name,digest in prep['input_hashes'].items():
    assert sha256(OUT/name) == digest == sha256(REFERENCE/name), name
for name,digest in prep['preservation_hashes'].items():
    assert sha256(ROOT/name) == digest, name
raw = pd.read_csv(ROOT/'asd_analysis/session_type_csvs/ASD_session_type_1.csv',float_precision='round_trip',
    usecols=['filename','genotype','animal','session_type','training_level','ABL','ILD','success','response_poke','timed_rt','fix_time','intended_fix_time','abort_type'])
mask = raw.filename.eq('SHANK3_cohort1.csv') & raw.genotype.eq('wt') & raw.session_type.eq(1) & raw.training_level.eq(16)
mask &= raw.ABL.isin([20,40,59,60]) & raw.ILD.isin(ILDS)
eligible = raw.loc[mask]
valid = eligible.loc[eligible.success.isin([-1,1]) & np.isfinite(eligible.timed_rt) & eligible.timed_rt.between(0,1)]
assert len(eligible) == 83921 and len(valid) == 68184 and valid.animal.nunique() == 5
assert valid.response_poke.value_counts().to_dict() == {1.:37482,-1.:30702}
np.testing.assert_array_equal(valid.index+2,saved.prepared_csv_line)
np.testing.assert_array_equal(valid.timed_rt,saved.timed_rt)
np.testing.assert_array_equal(valid.response_poke,saved.response_poke)
np.testing.assert_array_equal(valid.intended_fix_time/1000,saved.onset_s)
np.testing.assert_array_equal(eligible.index+2,parent.prepared_csv_line)
negative_rt = eligible.fix_time/1000-eligible.intended_fix_time/1000
negative = eligible.abort_type.eq('Fixation') & negative_rt.between(-1,0,inclusive='left')
np.testing.assert_array_equal(eligible.index[negative]+2,parent.loc[parent.display_abort,'prepared_csv_line'])
assert negative.sum() == 13750 and parent.fit_valid.sum()+parent.display_abort.sum() == 81934
assert (eligible.success.isin([-1,1]) & eligible.timed_rt.gt(1)).sum() == 373
abort_summary = pd.read_csv(ROOT/'asd_analysis/shank3_cohort1_proactive_lapse/wt/seed_0/posterior_summary.csv',float_precision='round_trip').set_index('parameter')
np.testing.assert_array_equal(fixed,abort_summary.loc[['V_A','theta_A','t_nd','lapse_prob','beta_lapse'],'mean'])
for row in conditions.itertuples():
    z = saved.loc[saved.condition.eq(row.condition)]
    assert (z.ABL_mapped == row.ABL).all() and (z.ILD == row.ILD).all() and len(z) == row.trials

# %% Independent mapping, exact expansion, prior multiplicities, and stop rules
for changes in [{},{'rate_lambda':3.,'theta_E':2.4,'T_0':.04,'rate_norm_l':1.02,'alpha':.2}]:
    params = {**INITIAL,**changes}
    expanded = expand_parameters(params)
    g,o = numpy_mapping(params)
    np.testing.assert_allclose(expanded['gamma'],g,atol=2e-14,rtol=1e-13)
    np.testing.assert_allclose(expanded['omega'],o,atol=2e-14,rtol=1e-13)
    np.testing.assert_allclose(log_likelihood(params,data),race_log_likelihood(expanded,data,'condition_delay'),atol=1e-9)
from numpyro import handlers
trace = handlers.trace(handlers.seed(handlers.substitute(model,data=INITIAL),jax.random.PRNGKey(12))).get_trace(data)
sample_sites = {n:x for n,x in trace.items() if x['type']=='sample' and not x.get('is_observed',False)}
assert set(sample_sites) == set(INITIAL) and sum(np.size(x['value']) for x in sample_sites.values()) == 37
prior_logprob = 0.
for name,site in sample_sites.items():
    lo,pl,ph,hi = PRIORS[name]
    x = np.asarray(site['value'])
    height = 1/((pl-lo+hi-ph)/2+ph-pl)
    shape = np.minimum(np.minimum((x-lo)/(pl-lo),(hi-x)/(hi-ph)),1.)
    expected = np.log(shape*height).sum()
    actual = float(site['fn'].log_prob(site['value']).sum()+trace[name+'_trapezoid']['fn'].log_prob(trace[name+'_trapezoid']['value']).sum())
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    prior_logprob += expected
assert loss_window_decision(9.,10.,50000,24) == (True,0,False)
assert loss_window_decision(10.,10.,50000,24) == (False,25,True)
assert loss_window_decision(11.,10.,49000,25) == (False,26,False)
best,patience = np.inf,0
for step,loss in [(47000,10.),(48000,11.),(49000,9.),(50000,10.)]:
    improved,patience,stop = loss_window_decision(loss,best,step,patience,50000,2)
    if improved: best = loss
    if step == 49000: resumed = (best,patience)
assert not stop and resumed == (9.,0)
assert loss_window_decision(10.,resumed[0],50000,resumed[1],50000,2) == (False,1,False)
assert loss_window_decision(10.,best,51000,patience,50000,2) == (False,2,True)

# %% Density/CDF normalization, lapse limits and initial full-data gradients
normalization = []
for g,o,d,w,go,c in [(0.,4.,.05,.5,.08,.6),(2.,12.,.10,.42,.15,1.),(-2.,6.,.2,.58,0.,.4),
                    (.4,.5,0.,.5,.2,1.1),(-.8,8.,.5,.5,.2,.7)]:
    times = np.unique(np.r_[np.linspace(-.2,1,181),d,np.nextafter(d,0.),d-go])
    cached = fixed_at(times,c,fixed)
    for side in [-1,1]:
        actual = np.asarray(choice_density(times,side,g,o,d,w,go,cached,fixed[3]))
        np.testing.assert_allclose(actual,scipy_race(times,c,g,o,d,w,go,side),atol=1e-10,rtol=3e-9)
        lapse_only = np.asarray(choice_density(times,side,g,o,d,w,go,cached,1.))
        np.testing.assert_allclose(lapse_only,.5*np.asarray(cached['lapse_pdf']),atol=1e-14)
        assert np.isfinite(np.asarray(choice_density(times,side,g,o,d,w,go,cached,0.))).all()
    cuts = sorted(set(x for x in [d,d-go] if 0<x<1))
    integral = quad(lambda t:float(scipy_race(t,c,g,o,d,w,go,1)+scipy_race(t,c,g,o,d,w,go,-1)),0,1,points=cuts,epsabs=1e-9)[0]
    mass = float(scipy_cdf(1,c,g,o,d,w)-scipy_cdf(0,c,g,o,d,w))
    np.testing.assert_allclose(integral,mass,atol=1e-8)
    normalization.append(dict(delay=d,onset=c,mass=mass,integral=integral))
initial = {k:jnp.asarray(v) for k,v in INITIAL.items()}
value,gradient = jax.jit(jax.value_and_grad(log_likelihood))(initial,data)
assert np.isfinite(value) and all(np.isfinite(np.asarray(x)).all() for x in gradient.values())
reference_value = independent_likelihood(initial)
np.testing.assert_allclose(value,reference_value,atol=1e-6,rtol=1e-11)
report = dict(status='passed',valid_trials=68184,right=37482,left=30702,animals=5,conditions=30,
    parent_trials=83921,display_trials=81934,parameter_count=37,initial_log_likelihood=float(value),
    scipy_initial_log_likelihood=reference_value,finite_gradients=True,independent_prior_logprob=prior_logprob,
    mapping_and_expansion_verified=True,normalization=normalization,stop_reset_tie_minimum_resume_tests='passed',
    verifier_sha256=sha256(Path(__file__)))
if args.pre_fit:
    (OUT/'prefit_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0)

# %% Restore posterior from the absolute-best checkpoint and audit convergence
folder = OUT/'seed_0'
config = json.loads((folder/'configuration.json').read_text())
run = json.loads((folder/'run_summary.json').read_text())
for name,digest in config['source_hashes'].items():
    assert sha256(ROOT/name) == digest, name
samples = dict(np.load(folder/'posterior_samples.npz'))
matrix = sample_matrix(samples)
assert matrix.shape == (10000,37) and np.isfinite(matrix).all()
summary = pd.read_csv(folder/'posterior_summary.csv',float_precision='round_trip')
np.testing.assert_array_equal(summary.parameter,parameter_names())
np.testing.assert_allclose(summary['mean'],matrix.mean(0),atol=2e-14)
np.testing.assert_allclose(summary.q025,np.quantile(matrix,.025,axis=0),atol=2e-14)
np.testing.assert_allclose(summary.q975,np.quantile(matrix,.975,axis=0),atol=2e-14)
convergence = pd.read_csv(folder/'convergence.csv',float_precision='round_trip')
losses = np.load(folder/'losses.npy')
assert len(losses) == run['final_checked_step'] and np.isfinite(losses).all()
latest = pickle.loads((folder/'latest_checkpoint.pkl').read_bytes())
best_checkpoint = pickle.loads((folder/'best_checkpoint.pkl').read_bytes())
restored = pickle.loads((folder/'restored_guide_optimizer.pkl').read_bytes())
assert latest['step'] == run['final_checked_step']
for a,b in zip(jax.tree_util.tree_leaves(best_checkpoint),jax.tree_util.tree_leaves(restored)):
    np.testing.assert_array_equal(a,b)
assert restored['step'] == run['restored_best_step'] == int(convergence.loc[convergence.mean_negative_elbo.idxmin(),'step'])
assert not config['parameter_stability_required'] and not run['parameter_stability_required']
best_loss,tail = np.inf,0
for row in convergence.itertuples():
    np.testing.assert_allclose(row.mean_negative_elbo,losses[row.step-1000:row.step].mean(),atol=1e-9)
    improved,tail,stop = loss_window_decision(row.mean_negative_elbo,best_loss,row.step,tail)
    assert improved == row.updated_best and tail == row.patience
    if improved: best_loss = row.mean_negative_elbo
assert (run['status']=='converged') == stop
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer import init_to_value
guide = AutoMultivariateNormal(model,init_loc_fn=init_to_value(values=initial),init_scale=.1)
# Initialize guide prototype without drawing from the likelihood a second time.
handlers.seed(guide,jax.random.PRNGKey(0))(data)
reproduced = guide.sample_posterior(jax.random.PRNGKey(100000),restored['params'],sample_shape=(10000,))
for name in samples:
    np.testing.assert_allclose(reproduced[name],samples[name],atol=2e-13,rtol=1e-12)
mean_params = {k:jnp.asarray(x.mean(0)) for k,x in samples.items()}
ll = float(log_likelihood(mean_params,data))
_,posterior_gradient = jax.jit(jax.value_and_grad(log_likelihood))(mean_params,data)
assert all(np.isfinite(np.asarray(x)).all() for x in posterior_gradient.values())
ll_reference = independent_likelihood(mean_params)
np.testing.assert_allclose(ll,ll_reference,atol=1e-6)
np.testing.assert_allclose(ll,run['posterior_mean_joint_log_likelihood'],atol=1e-7)
report.update(posterior_log_likelihood=ll,scipy_posterior_log_likelihood=ll_reference,
    posterior_restoration='passed',draws_reproduced=True,convergence_status=run['status'],posterior_gradients_finite=True)

# %% Independent histogram counts, onset weights, curves and psychometric points
manifest = json.loads((OUT/'plot_manifest.json').read_text())
for name,digest in manifest['artifacts'].items():
    assert sha256(OUT/name) == digest
assert manifest['display_xlim'] == [-.1,.6] and manifest['rtd_bin_width_s'] == .02
assert len([n for n in manifest['artifacts'] if n.endswith('.png')]) == 2
assert manifest['ABL_colors'] == {'20':'#1f77b4','40':'#ff7f0e','60':'#2ca02c'}
assert manifest['model_linewidth_pt'] == .9 and manifest['data_linewidth_pt'] == 1.4
from PIL import Image
for name in manifest['artifacts']:
    with Image.open(OUT/name) as im:
        assert im.width>=1400 and im.height>=1100
for name,digest in manifest['source_hashes'].items():
    assert sha256(ROOT/name)==digest
bins = pd.read_csv(OUT/'rtd_data_bins.csv')
old_bins = pd.read_csv(REFERENCE/'rtd_data_bins.csv')
for name in ['ABL','abs_ILD','bin','left','right','count','density','display_trials','parent_trials']:
    np.testing.assert_array_equal(bins[name],old_bins[name])
curves = pd.read_csv(OUT/'pooled_rtd_model_1ms.csv.gz')
model_bins = pd.read_csv(OUT/'pooled_rtd_model_bins.csv')
panels = pd.read_csv(OUT/'rtd_panel_audit.csv')
psych = pd.read_csv(OUT/'psychometric_predictions.csv')
assert len(panels) == 15 and len(psych) == 30
display = parent.loc[parent.fit_valid | parent.display_abort].copy()
display['rt'] = np.where(display.fit_valid,display.timed_rt,display.abort_rt_s)
edges = np.linspace(-1,1,101)
g,o = numpy_mapping(mean_params)
d = np.asarray(mean_params['t_E_aff']); w = float(mean_params['w']); go = float(mean_params['del_go'])
for row in panels.itertuples():
    obs = display.loc[display.ABL_mapped.eq(row.ABL)&display.ILD.abs().eq(row.abs_ILD)]
    z = bins.loc[bins.ABL.eq(row.ABL)&bins.abs_ILD.eq(row.abs_ILD)]
    np.testing.assert_array_equal(z['count'],np.histogram(obs.rt,edges)[0])
    np.testing.assert_allclose(z.density,z['count']/len(obs)/.02,atol=1e-12)
    assert row.display_trials == len(obs)
    curve = curves.loc[curves.ABL.eq(row.ABL)&curves.abs_ILD.eq(row.abs_ILD)]
    mb = model_bins.loc[model_bins.ABL.eq(row.ABL)&model_bins.abs_ILD.eq(row.abs_ILD)]
    np.testing.assert_allclose((mb.density*.02).sum(),1,atol=1e-12)
    np.testing.assert_allclose(trapezoid(curve.density,curve.rt_s),1,atol=3e-4)
    # Reconstruct selected times directly from every parent's exact onset.
    selected = np.arange(0,2001,37)
    t = curve.rt_s.to_numpy()[selected]
    mixed_q = np.zeros(len(t)); mixed_f = np.zeros(len(t)); total = 0
    for sign in [-1,1]:
        i = int(conditions.loc[conditions.ABL.eq(row.ABL)&conditions.ILD.eq(sign*row.abs_ILD),'condition'].iloc[0])
        group = parent.loc[parent.condition.eq(i)]
        assert len(group) == (row.negative_ILD_parent_trials if sign == -1 else row.positive_ILD_parent_trials)
        c = group.onset_s.to_numpy()
        q = scipy_race(t[:,None],c[None,:],g[i],o[i],d[i],w,go,1)+scipy_race(t[:,None],c[None,:],g[i],o[i],d[i],w,go,-1)
        f = scipy_cdf(t[:,None],c[None,:],g[i],o[i],d[i],w)
        mixed_q += q.sum(1); mixed_f += f.sum(1); total += len(group)
    assert total == row.parent_trials
    np.testing.assert_allclose(curve.unnormalized_density.to_numpy()[selected],mixed_q/total,atol=1e-9,rtol=2e-8)
    endpoint_mass = 0.
    for sign in [-1,1]:
        i = int(conditions.loc[conditions.ABL.eq(row.ABL)&conditions.ILD.eq(sign*row.abs_ILD),'condition'].iloc[0])
        group = parent.loc[parent.condition.eq(i)]
        c = group.onset_s.to_numpy()
        endpoint_mass += np.sum(scipy_cdf(1,c,g[i],o[i],d[i],w)-scipy_cdf(-1,c,g[i],o[i],d[i],w))
    np.testing.assert_allclose(endpoint_mass/total,row.model_window_mass,atol=2e-10)
    np.testing.assert_allclose(curve.window_cdf.to_numpy()[selected],(mixed_f/total-mixed_f[0]/total)/row.model_window_mass,atol=2e-9)
assert bins['count'].sum() == 81934
for row in psych.itertuples():
    z = saved.loc[saved.condition.eq(row.condition)]
    assert row.trials == len(z) and row.right == z.response_poke.eq(1).sum()
    np.testing.assert_allclose(row.data_p_right,z.response_poke.eq(1).mean(),atol=1e-14)
    i = row.condition
    c = z.onset_s.to_numpy()
    mass = scipy_cdf(1,c,g[i],o[i],d[i],w)-scipy_cdf(0,c,g[i],o[i],d[i],w)
    # The integral normalizes every onset before averaging, never afterward.
    points = sorted(set(x for x in [d[i],d[i]-go] if 0<x<1))
    probability = quad(lambda t:float(np.mean(scipy_race(t,c,g[i],o[i],d[i],w,go,1)/mass)),0,1,points=points,epsabs=2e-8)[0]
    np.testing.assert_allclose(row.model_p_right,probability,atol=8e-8)
report.update(rtd_panels=15,psychometric_predictions=30,histogram_count=81934,plot_checks='passed',
    figure_artifact_hashes=manifest['artifacts'],verification_sources={str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__)),
    'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py':sha256(ROOT/'ASD_Human_Data/human_plain_ddm_diagnostics_utils.py')})
reference_samples = dict(np.load(REFERENCE/'condition_delay/posterior_samples.npz'))
reference_means = {k:jnp.asarray(x.mean(0)) for k,x in reference_samples.items()}
reference_ll = float(race_log_likelihood(reference_means,data,'condition_delay'))
comparison = pd.read_csv(OUT/'in_sample_comparison.csv')
np.testing.assert_allclose(comparison.loc[comparison.variant.eq('condition_delay'),'joint_log_likelihood'],reference_ll,atol=1e-6)
np.testing.assert_allclose(comparison.loc[comparison.variant.eq('npl_alpha'),'joint_log_likelihood'],ll,atol=1e-6)
report['reference_common_trial_log_likelihood'] = reference_ll
report['empirical_rtd_arrays_unchanged'] = True
(OUT/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Independent fit, normalization, restoration, and all figure checks passed.',flush=True)
