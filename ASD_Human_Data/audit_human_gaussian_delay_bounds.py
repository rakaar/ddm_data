# %%
"""Audit both variational posterior tails and compare the wider-prior human fit."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd

# %% Editable output roots; no likelihood imports or JAX needed.
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'ASD_Human_Data/gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s_mu1000ms_sigma0p5ms')
args=parser.parse_args();out=args.output_dir
original=ROOT/'ASD_Human_Data/gamma_omega_gaussian_delay_rt_choice_lapse_svi_0_1p8s'
config=json.loads((out/'run_config.json').read_text())
previous_config=json.loads((original/'run_config.json').read_text())
summary=pd.read_csv(out/'parameter_summary.csv')
rows=[];comparison=[]
for group in ['Control','ASD']:
    samples=dict(np.load(out/group.lower()/'posterior_samples.npz'))
    previous=dict(np.load(original/group.lower()/'posterior_samples.npz'))
    for r in summary.loc[summary.Group.eq(group)].itertuples():
        block=r.block
        x=samples[block] if samples[block].ndim==1 else samples[block][:,int(r.condition_index)]
        old=previous[block] if previous[block].ndim==1 else previous[block][:,int(r.condition_index)]
        key=('gamma_positive' if r.signed_delta>0 else 'gamma_negative') if block=='gamma' else block
        low,high=config['priors'][key]['hard'];width=high-low
        previous_low,previous_high=previous_config['priors'][key]['hard']
        q005,q995=np.quantile(x,[.005,.995]);sd=x.std(ddof=1)
        row=dict(Group=group,parameter=r.parameter,block=block,ASL_cd=r.ASL_cd,signed_delta=r.signed_delta,
            mean=x.mean(),sd=sd,hard_lower=low,hard_upper=high,q005=q005,q995=q995,
            lower_gap=q005-low,upper_gap=high-q995,
            lower_gap_fraction=(q005-low)/width,upper_gap_fraction=(high-q995)/width,
            lower_gap_sd=(q005-low)/sd,upper_gap_sd=(high-q995)/sd,
            fraction_in_lower_1pct=np.mean(x<low+.01*width),fraction_in_upper_1pct=np.mean(x>high-.01*width))
        row['tail_within_1pct']=min(row['lower_gap_fraction'],row['upper_gap_fraction'])<.01
        row['accumulation_5pct']=max(row['fraction_in_lower_1pct'],row['fraction_in_upper_1pct'])>=.05
        assert low<q005<q995<high
        rows.append(row)
        comparison.append(dict(Group=group,parameter=r.parameter,block=block,ASL_cd=r.ASL_cd,signed_delta=r.signed_delta,
            previous_mean=old.mean(),new_mean=x.mean(),mean_change=x.mean()-old.mean(),
            previous_hard_lower=previous_low,previous_hard_upper=previous_high,
            fraction_below_previous_lower=np.mean(x<previous_low),fraction_above_previous_upper=np.mean(x>previous_high),
            mean_change_previous_sd=(x.mean()-old.mean())/old.std(ddof=1),
            previous_q005=np.quantile(old,.005),previous_q995=np.quantile(old,.995),q005=q005,q995=q995))
frame=pd.DataFrame(rows);frame.to_csv(out/'posterior_99pct_boundary_audit.csv',index=False,float_format='%.17g')
pd.DataFrame(comparison).to_csv(out/'original_vs_wide_parameter_comparison.csv',index=False,float_format='%.17g')
frozen=json.loads((out/'previous_gaussian_preservation.json').read_text())
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in frozen.items())
report=dict(parameters=166,draws_per_group=10000,quantiles=[.005,.995],
    tail_within_outer_1pct=int(frame.tail_within_1pct.sum()),accumulation_at_least_5pct=int(frame.accumulation_5pct.sum()),
    screen='Endpoint in outer 1% of hard interval; also report raw gaps, gaps in posterior SD and mass. Proximity alone is not proof of constraint.',
    preserved_previous_files=len(frozen),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'posterior_99pct_boundary_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
print(frame.sort_values('upper_gap_fraction').head(5).to_string(index=False))
print(frame.sort_values('lower_gap_fraction').head(5).to_string(index=False))
