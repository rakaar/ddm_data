# %% Independently reconstruct the signed-condition sensory-delay diagnostic
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import invgauss, norm
from scipy.special import ndtr
from PyPDF2 import PdfReader
from PIL import Image
from shank3_npl_alpha_gaussian_utils import ROOT, OUT, COUNTS, sha256
from human_plain_ddm_diagnostics_utils import scipy_ddm

GENOTYPE = 'wt'
N = COUNTS[GENOTYPE]
FIT = OUT/GENOTYPE
DEST = FIT/'sensory_delay_effect'
manifest = json.loads((DEST/'manifest.json').read_text())
hist = pd.read_csv(DEST/'model_histogram_bins.csv',float_precision='round_trip')
curves = pd.read_csv(DEST/'model_curves_1ms.csv.gz',float_precision='round_trip')
audit = pd.read_csv(DEST/'condition_audit.csv',float_precision='round_trip')
params = pd.read_csv(DEST/'condition_parameters.csv',float_precision='round_trip')
parent = pd.read_csv(FIT/'parent_trials.csv.gz',float_precision='round_trip')
valid = pd.read_csv(FIT/'fit_trials.csv.gz',float_precision='round_trip')
original = pd.read_csv(FIT/'signed_parent_model_1ms.csv.gz',float_precision='round_trip')
v,a,nd,p,beta = json.loads((FIT/'fixed_proactive.json').read_text())['values']
assert len(parent)==N['parent'] and len(valid)==N['valid'] and parent.animal.nunique()==N['animals']
assert valid.response_poke.eq(1).sum()==N['right'] and valid.response_poke.eq(-1).sum()==N['left']
assert valid.success.isin([1,-1]).all() and valid.timed_rt.between(0,1).all()
assert len(params)==30 and len(audit)==90 and manifest['rows']==10 and manifest['columns']==3
assert sorted(hist.ILD.unique())==[-16,-8,-4,-2,-1,1,2,4,8,16]
np.testing.assert_allclose(hist.right_s-hist.left_s,.020,atol=1e-15)
np.testing.assert_allclose(hist.density*.020,hist.probability,atol=1e-15)
np.testing.assert_allclose(audit.histogram_area,1.,atol=2e-12)
for name,digest in manifest['source_hashes'].items():
    assert sha256(ROOT/name)==digest

# %% SciPy 80-term evidence series, exact raw onset frequencies, full choices
largest_density_error = 0.
largest_cdf_error = 0.
for row in params.itertuples():
    i = row.condition
    z = parent.loc[parent.condition.eq(i)]
    onsets, counts = np.unique(z.onset_s.to_numpy(),return_counts=True)
    weights = counts/counts.sum()
    times = np.linspace(-1.,1.,101)
    absolute = times[:,None]+onsets[None,:]
    ap = invgauss.pdf(absolute-nd,mu=1/(v*a),scale=a*a)@weights
    ass = invgauss.sf(absolute-nd,mu=1/(v*a),scale=a*a)@weights
    lp = np.where(absolute>=0,beta*np.exp(-beta*np.maximum(absolute,0)),0.)@weights
    ls = np.exp(-beta*np.maximum(absolute,0))@weights
    ep = [scipy_ddm(times,row.gamma_derived,row.omega_derived,row.w,0.,side=s,K=80)
          for s in [-1,1]]
    ec = [scipy_ddm(times,row.gamma_derived,row.omega_derived,row.w,0.,cumulative=True,side=s,K=80)
          for s in [-1,1]]
    eg = [scipy_ddm(times+row.del_go_s,row.gamma_derived,row.omega_derived,row.w,0.,
          cumulative=True,side=s,K=80) for s in [-1,1]]
    # Keep proactive readout and both half-lapse terms when summing choices.
    right_and_left = [(1-p)*(ap*(.5*(1-eg[0]-eg[1])+eg[k]-ec[k])+ass*ep[k])+.5*p*lp
                      for k in [0,1]]
    pdf = right_and_left[0]+right_and_left[1]
    cdf = 1-(1-p)*ass*(1-ec[0]-ec[1])-p*ls
    saved = curves.loc[curves.condition.eq(i)&curves.distribution.eq('zero_sensory_delay_rt')].sort_values('time_s')
    check = saved.iloc[::20]
    np.testing.assert_allclose(check.time_s,times,atol=1e-15)
    np.testing.assert_allclose(check.unnormalized_density,pdf,atol=2e-10,rtol=2e-10)
    np.testing.assert_allclose(check.unconditional_cdf,cdf,atol=2e-12,rtol=2e-12)
    largest_density_error=max(largest_density_error,float(np.abs(check.unnormalized_density-pdf).max()))
    largest_cdf_error=max(largest_cdf_error,float(np.abs(check.unconditional_cdf-cdf).max()))
    mass = cdf[-1]-cdf[0]
    bins = hist.loc[hist.condition.eq(i)&hist.distribution.eq('zero_sensory_delay_rt')].sort_values('bin')
    np.testing.assert_allclose(bins.probability,np.diff(cdf)/mass,atol=3e-12)
    np.testing.assert_allclose(saved.density,saved.unnormalized_density/mass,atol=3e-12)
    old = original.loc[original.condition.eq(i)].sort_values('rt_s')
    shown = curves.loc[curves.condition.eq(i)&curves.distribution.eq('original_rt')].sort_values('time_s')
    np.testing.assert_array_equal(shown.unnormalized_density,old.unnormalized_density)
    np.testing.assert_array_equal(shown.unconditional_cdf,old.unconditional_cdf)
    np.testing.assert_allclose(saved.loc[saved.time_s.lt(0),'unnormalized_density'],
        old.loc[old.rt_s.lt(0),'unnormalized_density'],atol=2e-14)
    assert (saved.unconditional_cdf.to_numpy()-old.unconditional_cdf.to_numpy()).min()>-2e-7
    for label in ['original_rt','zero_sensory_delay_rt']:
        q=curves.loc[curves.condition.eq(i)&curves.distribution.eq(label)].sort_values('time_s')
        assert np.diff(q.unconditional_cdf).min()>-1e-8
        b=hist.loc[hist.condition.eq(i)&hist.distribution.eq(label)].sort_values('bin')
        np.testing.assert_allclose(b.probability,np.diff(q.unconditional_cdf.to_numpy()[::20])/
            (q.unconditional_cdf.iloc[-1]-q.unconditional_cdf.iloc[0]),atol=2e-12)
    ndt=curves.loc[curves.condition.eq(i)&curves.distribution.eq('sensory_delay')].sort_values('time_s')
    norm_mass=ndtr(row.delay_mu_s/row.delay_sigma_s)
    standard=(ndt.time_s.to_numpy()-row.delay_mu_s)/row.delay_sigma_s
    ndt_pdf=norm.pdf(standard)/row.delay_sigma_s/norm_mass
    ndt_cdf=(ndtr(standard)-ndtr(-row.delay_mu_s/row.delay_sigma_s))/norm_mass
    np.testing.assert_allclose(ndt.density,ndt_pdf,atol=1e-11)
    np.testing.assert_allclose(ndt.unconditional_cdf,ndt_cdf,atol=1e-14)
    b=hist.loc[hist.condition.eq(i)&hist.distribution.eq('sensory_delay')].sort_values('bin')
    np.testing.assert_allclose(b.probability,np.diff(ndt_cdf[::20]),atol=1e-14)

# %% Output appearance metadata, PDF structure and unchanged source artifacts
assert manifest['colors']=={'20':'#1f77b4','40':'#ff7f0e','60':'#2ca02c'}
assert len(manifest['geometry'])==30
assert {tuple(g['xlim']) for g in manifest['geometry'] if g['column']>0}=={(-.1,.6)}
assert {tuple(g['xlim']) for g in manifest['geometry'] if g['column']==0}=={(0.,.25)}
for name,digest in manifest['artifacts'].items():
    assert sha256(DEST/name)==digest
    if name.endswith('.pdf'): assert len(PdfReader(DEST/name).pages)==1
    else: assert Image.open(DEST/name).size==(2016,5120)
(DEST/'validation.json').write_text(json.dumps(dict(status='passed',animals=N['animals'],
    signed_conditions=30,panels=30,model_histogram_curves=90,bins=len(hist),
    parent_trials=N['parent'],fitted_valid_choices=N['valid'],
    original_model_arrays_unchanged=True,negative_raw_abort_density_unchanged=True,
    raw_parent_onset_averages_reconstructed=True,complete_choice_sum_verified=True,
    zero_delay_window_masses_recomputed=True,unit_histogram_areas_verified=True,
    maximum_independent_density_error=largest_density_error,
    maximum_independent_cdf_error=largest_cdf_error,source_hashes_unchanged=True,
    pdf_pages=1,verifier_sha256=sha256(Path(__file__))),indent=2)+'\n')
print('Passed: 30 signed conditions, 90 histograms, exact onsets and independent complete-choice densities.',flush=True)
