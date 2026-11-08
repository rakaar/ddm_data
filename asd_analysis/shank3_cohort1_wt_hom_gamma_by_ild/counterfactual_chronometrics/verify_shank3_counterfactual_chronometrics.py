# %% Independent parameter maps, SciPy survival integrals, raw draws and plots
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
import sys
import json
import hashlib
import subprocess
import shutil
import numpy as np
import pandas as pd
from scipy.integrate import quad,trapezoid
from scipy.stats import invgauss

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_by_ild'
DEST = FIT/'counterfactual_chronometrics'
sys.path.insert(0,str(ROOT/'ASD_Human_Data'))
from human_plain_ddm_diagnostics_utils import scipy_ddm

ABLS = [20,40,60]
ILDS = [-16,-8,-4,-2,-1,1,2,4,8,16]
config = json.loads((DEST/'configuration.json').read_text())
params = pd.read_csv(DEST/'parameter_swaps.csv',float_precision='round_trip')
conditions = pd.read_csv(DEST/'condition_means.csv',float_precision='round_trip')
points = pd.read_csv(DEST/'pooled_chronometrics.csv',float_precision='round_trip')
observed = pd.read_csv(DEST/'observed_chronometrics.csv',float_precision='round_trip')
animals = pd.read_csv(DEST/'animal_chronometrics.csv',float_precision='round_trip')
plot = json.loads((DEST/'plot_manifest.json').read_text())
baseline = pd.read_csv(FIT/'signed_condition_means.csv',float_precision='round_trip')
assert len(conditions)==180 and len(points)==30 and len(params)==96 and len(animals)==55
assert points.simulated_trials.sum()==40769700 and len(plot['figures'])==2
means = {}
for group in ['wt','hom']:
    with np.load(FIT/group/'gamma_by_ild/posterior_samples.npz') as draws:
        means[group] = {k:draws[k].mean(0) for k in draws.files}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


max_mean_error,max_cdf_error = 0.,0.
independent_rows = []
for gi,group in enumerate(['wt','hom']):
    valid = pd.read_csv(FIT/group/'fit_trials.csv.gz',float_precision='round_trip')
    frozen = np.asarray(json.loads((FIT/group/'fixed_proactive.json').read_text())['values'])
    np.testing.assert_array_equal(frozen,config['fixed_proactive'][group])
    assert len(valid)==(68184 if group=='wt' else 67715)
    assert valid.animal.nunique()==(5 if group=='wt' else 6)
    assert valid.success.isin([1,-1]).all() and valid.timed_rt.between(0,1).all()
    v,a,nd,p,beta = frozen
    donor = 'hom' if group=='wt' else 'wt'
    for row in params.loc[params.group.eq(group)].itertuples():
        swapped = 't_E_aff' if row.scenario=='delay_swap' else 'omega' if row.scenario=='omega_swap' else None
        source = donor if row.block==swapped else group
        assert row.parameter_source==source
        np.testing.assert_array_equal(row.value,np.asarray(means[source][row.block]).reshape(-1)[row.index])
    for cid in range(30):
        sub = valid.loc[valid.condition.eq(cid)]
        onset = np.sort(sub.onset_s.to_numpy())
        samples = dict(np.load(DEST/group/f'simulated_condition_{cid:02d}.npz'))
        distribution = dict(np.load(DEST/group/f'conditional_distribution_{cid:02d}.npz'))
        grid = distribution['rt_grid_s']
        np.testing.assert_allclose(np.diff(grid),.001,atol=1e-14)
        np.testing.assert_array_equal(distribution['onset_s'],np.unique(onset))
        assert distribution['onset_count'].sum()==len(sub)
        rng = np.random.default_rng(np.random.SeedSequence([0,gi,cid]))
        uniforms = rng.random(100*len(sub))
        for scenario in config['scenarios']:
            row = conditions.loc[conditions.group.eq(group)&conditions.scenario.eq(scenario)&conditions.condition.eq(cid)].iloc[0]
            assert row.fitted_trials==len(sub) and row.simulated_trials==100*len(sub)
            assert row.uniforms_sha256==hashlib.sha256(uniforms.tobytes()).hexdigest()
            g,o,d,w = row.gamma,row.omega,row.t_E_aff,row.w
            original = means[group]
            omega_source = means[donor] if scenario=='omega_swap' else original
            delay_source = means[donor] if scenario=='delay_swap' else original
            np.testing.assert_array_equal([g,o,d,w,row.del_go],
                [original['gamma'][ILDS.index(row.ILD)],omega_source['omega'][ABLS.index(row.ABL)],
                 float(delay_source['t_E_aff']),float(original['w']),float(original['del_go'])])
            # Independent full-mixture survival, normalized per original onset.
            s0 = (1-p)*invgauss.sf(onset-nd,mu=1/(v*a),scale=a*a)+p*np.exp(-beta*onset)
            se1 = 1-scipy_ddm(1.,g,o,w,d,cumulative=True)
            s1 = (1-p)*invgauss.sf(onset+1-nd,mu=1/(v*a),scale=a*a)*se1+p*np.exp(-beta*(onset+1))
            mass = s0-s1
            assert (mass>0).all() and (mass<=1).all()
            def survival(t):
                es = 1-scipy_ddm(t,g,o,w,d,cumulative=True)
                st = (1-p)*invgauss.sf(onset+t-nd,mu=1/(v*a),scale=a*a)*es+p*np.exp(-beta*(onset+t))
                return float(np.mean((st-s1)/mass))
            mean,error = quad(survival,0,1,points=[d],epsabs=1e-9,epsrel=1e-9,limit=150)
            np.testing.assert_allclose(mean,row.exact_mean_RT_s,atol=1e-7,rtol=0)
            max_mean_error = max(max_mean_error,abs(mean-row.exact_mean_RT_s))
            cdf = distribution[scenario+'_cdf']
            assert np.diff(cdf).min()>=0 and cdf[0]==0 and cdf[-1]==1
            np.testing.assert_allclose(distribution[scenario+'_window_mass'],mass,atol=1e-12)
            for i in [0,1,50,79,86,100,200,400,700,1000]:
                reference = 1-survival(grid[i])
                np.testing.assert_allclose(cdf[i],reference,atol=1e-11)
                max_cdf_error = max(max_cdf_error,abs(cdf[i]-reference))
            np.testing.assert_allclose(trapezoid(1-cdf,grid),row.CDF_mean_RT_s,atol=1e-12)
            np.testing.assert_allclose(row.CDF_mean_RT_s,mean,atol=1e-6,rtol=0)
            rt = samples[scenario+'_rt_s']
            assert len(rt)==100*len(sub) and np.isfinite(rt).all() and rt.min()>=0 and rt.max()<=1
            np.testing.assert_array_equal(rt,np.interp(uniforms,cdf,grid))
            np.testing.assert_allclose([rt.mean(),rt.var(ddof=1)],
                [row.simulated_mean_RT_s,row.simulated_variance_s2],atol=1e-14)
            assert abs(rt.mean()-mean)<6*np.sqrt(rt.var(ddof=1)/len(rt))
            if scenario=='intact':
                old = baseline.loc[baseline.group.eq(group)&baseline.condition.eq(cid)].iloc[0]
                np.testing.assert_allclose(mean,old.model_mean_RT_s,atol=1e-6,rtol=0)
            independent_rows.append(dict(group=group,scenario=scenario,condition=cid,
                independent_mean_RT_s=mean,adaptive_quadrature_error=error))
        if (cid+1)%5==0:print(f'{group.upper()}: {cid+1}/30 signed conditions independently verified',flush=True)
    for row in observed.loc[observed.group.eq(group)].itertuples():
        sub = valid.loc[valid.ILD.abs().eq(row.abs_ILD)]
        am = sub.groupby('animal').timed_rt.mean()
        np.testing.assert_allclose([row.data_mean_RT_s,row.animal_SEM_s],
            [sub.timed_rt.mean(),am.std(ddof=1)/np.sqrt(len(am))],atol=1e-14)
        assert len(am)==row.animals and len(sub)==row.trials

# %% Fixed condition weights, stratified Monte Carlo SEM, and observed/model artists
for row in points.itertuples():
    sub = conditions.loc[conditions.group.eq(row.group)&conditions.scenario.eq(row.scenario)&conditions.abs_ILD.eq(row.abs_ILD)]
    n = sub.simulated_trials.sum()
    assert row.fitted_trials==sub.fitted_trials.sum() and row.simulated_trials==n
    np.testing.assert_allclose(sub.pooled_fitted_weight,sub.fitted_trials/sub.fitted_trials.sum(),atol=1e-14)
    np.testing.assert_allclose([row.simulated_mean_RT_s,row.exact_mean_RT_s,row.MC_SE_s],
        [np.average(sub.simulated_mean_RT_s,weights=sub.simulated_trials),
         np.average(sub.exact_mean_RT_s,weights=sub.fitted_trials),
         np.sqrt(np.sum(sub.simulated_trials*sub.simulated_variance_s2)/n**2)],atol=1e-14)
    assert abs(row.MC_z)<6
for trace in plot['plotted_series']:
    assert trace['color']=={'wt':'#4D4D4D','hom':'#C24A7A'}[trace['group']]
    if trace['role']=='observed':
        assert trace['panel']=='intact_comparison'
        sub = observed.loc[observed.group.eq(trace['group'])].sort_values('abs_ILD')
        np.testing.assert_allclose([trace['y'],trace['lower'],trace['upper']],
            [sub.data_mean_RT_s,sub.data_mean_RT_s-sub.animal_SEM_s,sub.data_mean_RT_s+sub.animal_SEM_s],atol=1e-14)
    else:
        sub = points.loc[points.group.eq(trace['group'])&points.scenario.eq(trace['scenario'])].sort_values('abs_ILD')
        np.testing.assert_array_equal(trace['y'],sub.simulated_mean_RT_s)
        assert trace['linestyle']==('-' if trace['scenario']=='intact' else '--')
    np.testing.assert_array_equal(trace['x'],sub.abs_ILD)
for figure in plot['figures']:
    assert len(figure['panels'])==3
    for panel in figure['panels']:
        np.testing.assert_allclose(panel['width_inches'],panel['height_inches'],atol=1e-12)
        assert not panel['grid_visible'] and panel['xlim']==[0,17] and panel['ylim']==[.1,.25]
        for trace in [t for t in plot['plotted_series'] if t['figure']==figure['swap'] and t['panel']==panel['panel']]:
            y = np.r_[trace['y'],trace.get('lower',[]),trace.get('upper',[])]
            assert y.min()>=panel['ylim'][0] and y.max()<=panel['ylim'][1]
    info = subprocess.check_output(['pdfinfo',str(DEST/(figure['stem']+'.pdf'))],text=True)
    assert int(next(s.split(':')[1] for s in info.splitlines() if s.startswith('Pages:')))==1
info = subprocess.check_output(['pdfinfo',str(DEST/plot['combined_PDF'])],text=True)
assert int(next(s.split(':')[1] for s in info.splitlines() if s.startswith('Pages:')))==2
for name,expected in plot['artifacts'].items():assert sha256(DEST/name)==expected
for name,expected in config['source_hashes'].items():assert sha256(Path(name))==expected,name
pd.DataFrame(independent_rows).to_csv(DEST/'independent_exact_means.csv',index=False,float_format='%.17g')
report = dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),
    scenarios=6,signed_conditions=180,pooled_points=30,simulated_RTs=40769700,
    maximum_independent_mean_error_s=max_mean_error,maximum_independent_CDF_error=max_cdf_error,
    maximum_condition_abs_MC_z=float(conditions.MC_z.abs().max()),maximum_pooled_abs_MC_z=float(points.MC_z.abs().max()),
    source_hashes_unchanged=True,parameter_swaps_verified=True,common_random_numbers_verified=True,
    conditional_normalization_verified=True,observed_animal_SEMs_verified=True,
    single_page_PDFs=2,combined_PDF_pages=2,ylim=[.1,.25],
    visual_inspection='pending',source_sha256=sha256(Path(__file__)))
(DEST/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(Path(__file__),DEST/Path(__file__).name)
print(json.dumps(report,indent=2),flush=True)
