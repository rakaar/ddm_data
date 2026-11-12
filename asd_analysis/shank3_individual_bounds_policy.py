# %% Editable audit parameters: read existing fits, write a separate baseline audit
"""Independent, CPU-only bound audit for the eleven individual SHANK3 fits.

No likelihood/fit module is imported and no existing output is modified.  This
audit distinguishes a hard cutoff, a taper in a trapezoidal prior, and a model
domain boundary.  A tail entering a taper does not by itself prove biased fitting.
Range-relative diagnostics are reported but are not the acceptance rule: moving
the opposite endpoint must not make an unchanged posterior pass.
"""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'WT': ROOT / 'asd_analysis/shank3_cohort1_wt_individual_npl_alpha_gaussian_delay_by_abl_proactive_ndm50',
    'HOM': ROOT / 'asd_analysis/shank3_cohort1_hom_individual_npl_alpha_gaussian_delay_by_abl',
}
OUT = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_individual_adaptive_bounds/baseline_audit'
ABLS = [20, 40, 60]
QUANTILES = [.005, .025, .5, .975, .995]
HARD_STRIP_MASS_LIMIT = .01
TAIL_GAP_SD_LIMIT = .5
TAPER_MASS_LIMIT = .10
# These distances are physical parameter units, independent of either hard range.
ABSOLUTE_STRIPS = dict(
    V_A=.05, theta_A=.05, t_nd=.005, lapse_prob=.01, beta_lapse=.05,
    rate_lambda=.05, T_0=.005, theta_E=.05, rate_norm_l=.05, alpha=.10,
    w=.02, del_go=.005, delay_mu=.005, delay_sigma=.001,
)
# Mathematical/model-domain endpoints cannot be crossed merely to erase a flag.
# Positive floors above zero (e.g. sigma=.005) are artificial, not intrinsic.
DOMAINS = dict(
    V_A=(0., None), theta_A=(0., None), t_nd=(None, None),
    lapse_prob=(0., 1.), beta_lapse=(0., None), rate_lambda=(0., None),
    T_0=(0., None), theta_E=(0., None), rate_norm_l=(0., None),
    alpha=(0., None), w=(0., 1.), del_go=(0., None),
    delay_mu=(0., None), delay_sigma=(0., None),
)


# %% Stable reporting helpers, shared by every posterior column
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_column(values, bounds, parameter):
    """Quantify both ends without using the opposite endpoint for decisions."""
    lo, flat_lo, flat_hi, hi = map(float, bounds)
    v = np.asarray(values, dtype=np.float64)
    assert np.isfinite(v).all() and len(v) == 10000
    assert lo <= v.min() and hi >= v.max()
    q005, q025, q50, q975, q995 = np.quantile(v, QUANTILES)
    sd = v.std(ddof=1)
    # Freedman--Diaconis histogram mode; capped bins keep broad tails readable.
    width = 2 * np.subtract(*np.quantile(v, [.75, .25])) / np.cbrt(len(v))
    n_bins = int(np.clip(np.ceil(np.ptp(v) / width), 32, 256))
    hist, edges = np.histogram(v, n_bins)
    index = int(hist.argmax())
    mode = .5 * (edges[index] + edges[index + 1])
    strip = ABSOLUTE_STRIPS[parameter]
    low_mass = float(np.mean(v <= lo + strip))
    high_mass = float(np.mean(v >= hi - strip))
    low_gap_sd, high_gap_sd = (q005 - lo) / sd, (hi - q995) / sd
    low_pressure = low_mass >= HARD_STRIP_MASS_LIMIT or low_gap_sd < TAIL_GAP_SD_LIMIT
    high_pressure = high_mass >= HARD_STRIP_MASS_LIMIT or high_gap_sd < TAIL_GAP_SD_LIMIT
    intrinsic_lo, intrinsic_hi = DOMAINS[parameter]
    lower_intrinsic = intrinsic_lo is not None and lo == intrinsic_lo
    upper_intrinsic = intrinsic_hi is not None and hi == intrinsic_hi
    taper_low_mass = float(np.mean(v < flat_lo))
    taper_high_mass = float(np.mean(v > flat_hi))
    low_taper_pressure = taper_low_mass >= TAPER_MASS_LIMIT or mode < flat_lo
    high_taper_pressure = taper_high_mass >= TAPER_MASS_LIMIT or mode > flat_hi
    return dict(
        draws=len(v), mean=float(v.mean()), sd=float(sd),
        q005=float(q005), q025=float(q025), median=float(q50),
        q975=float(q975), q995=float(q995), sample_min=float(v.min()),
        sample_max=float(v.max()), histogram_mode=float(mode), mode_bins=n_bins,
        mode_bin_lower=float(edges[index]), mode_bin_upper=float(edges[index + 1]),
        hard_lower=lo, flat_lower=flat_lo, flat_upper=flat_hi, hard_upper=hi,
        exact_lower_hits=int(np.sum(v == lo)), exact_upper_hits=int(np.sum(v == hi)),
        lower_1pct_range_mass=float(np.mean(v <= lo + .01 * (hi - lo))),
        upper_1pct_range_mass=float(np.mean(v >= hi - .01 * (hi - lo))),
        lower_5pct_range_mass=float(np.mean(v <= lo + .05 * (hi - lo))),
        upper_5pct_range_mass=float(np.mean(v >= hi - .05 * (hi - lo))),
        lower_taper_mass=taper_low_mass, upper_taper_mass=taper_high_mass,
        absolute_strip=strip, lower_absolute_strip_mass=low_mass,
        upper_absolute_strip_mass=high_mass,
        lower_mean_gap_sd=float((v.mean() - lo) / sd),
        upper_mean_gap_sd=float((hi - v.mean()) / sd),
        lower_q005_gap_sd=float(low_gap_sd), upper_q995_gap_sd=float(high_gap_sd),
        lower_mode_gap_sd=float((mode - lo) / sd),
        upper_mode_gap_sd=float((hi - mode) / sd),
        lower_intrinsic_boundary=lower_intrinsic,
        upper_intrinsic_boundary=upper_intrinsic,
        lower_hard_pressure=bool(low_pressure), upper_hard_pressure=bool(high_pressure),
        lower_artificial_pressure=bool(low_pressure and not lower_intrinsic),
        upper_artificial_pressure=bool(high_pressure and not upper_intrinsic),
        lower_taper_pressure=bool(low_taper_pressure),
        upper_taper_pressure=bool(high_taper_pressure),
    )



# %% Audit an immutable completed run using its actual priors

def audit_run(run):
    config=json.loads((run/'configuration.json').read_text())
    rows=[]
    with np.load(run/'posterior_samples.npz') as samples:
        for name in samples.files:
            matrix=np.asarray(samples[name]).reshape(10000,-1)
            for j in range(matrix.shape[1]):
                label=f'{name}_ABL{ABLS[j]}' if matrix.shape[1]==3 else name
                rows.append(dict(group=config.get('group'),animal=config['animal'],stage=config['stage'],
                    seed=config['seed'],parameter=label,prior_key=name,
                    **audit_column(matrix[:,j],config['priors'][name],name)))
    return rows


def expand_priors(priors,rows):
    revised={name:list(values) for name,values in priors.items()}
    changes=[]
    for name,original in priors.items():
        proposals=[]
        triggers=[]
        for row in rows:
            if row['prior_key']!=name:
                continue
            # Every ABL proposes from the same ORIGINAL tuple.  Combining these
            # proposals once avoids multiplying a shared bound for each ABL.
            lo,pl,ph,hi=original
            sd=row['sd'];strip=row['absolute_strip']
            domain_low,domain_high=DOMAINS[name]
            if row['lower_artificial_pressure']:
                target=min(lo-strip*2,row['q005']-4*sd)
                if domain_low is not None:
                    target=max(domain_low,target)
                lo=target
                pl=max(lo+max(1e-8,abs(lo)*1e-8),min(pl,row['q005']-sd))
            elif row['lower_taper_pressure']:
                desired_flat=min(pl,row['q005']-sd)
                # A taper expansion may itself require a wider hard endpoint.
                # Keep its requested posterior-SD clearance whenever the model
                # domain permits it; do not clamp it to hard+epsilon needlessly.
                if desired_flat<=lo and (domain_low is None or lo>domain_low):
                    lo=min(lo-strip*2,row['q005']-4*sd)
                    if domain_low is not None:
                        lo=max(domain_low,lo)
                pl=max(lo+max(1e-8,abs(lo)*1e-8),desired_flat)
            if row['upper_artificial_pressure']:
                target=max(hi+strip*2,row['q995']+4*sd,hi*1.75)
                if domain_high is not None:
                    target=min(domain_high,target)
                hi=target
                ph=min(hi-max(1e-8,abs(hi)*1e-8),max(ph,row['q995']+sd))
            elif row['upper_taper_pressure']:
                desired_flat=max(ph,row['q995']+sd)
                if desired_flat>=hi and (domain_high is None or hi<domain_high):
                    hi=max(hi+strip*2,row['q995']+4*sd,hi*1.75)
                    if domain_high is not None:
                        hi=min(domain_high,hi)
                ph=min(hi-max(1e-8,abs(hi)*1e-8),desired_flat)
            proposal=[lo,pl,ph,hi]
            assert lo<pl<=ph<hi,(name,proposal)
            proposals.append(proposal)
            if list(original)!=proposal:
                triggers.append(row['parameter'])
        if proposals:
            revised[name]=[min(p[0] for p in proposals),min(p[1] for p in proposals),
                           max(p[2] for p in proposals),max(p[3] for p in proposals)]
            lo,pl,ph,hi=revised[name]
            assert lo<pl<=ph<hi,(name,revised[name])
        if list(original)!=revised[name]:
            changes.append(dict(parameter=name,before=list(original),after=revised[name],
                                trigger=', '.join(sorted(set(triggers)))))
    return revised,changes
