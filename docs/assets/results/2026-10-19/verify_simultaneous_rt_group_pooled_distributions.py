"""Check the pooled human figure directly against all selected trial rows."""

# %% Inputs and saved calculations
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import curve_fit
from scipy.special import expit

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "simultaneous_rt_group_pooled_distributions"
INPUT = HERE / "simultaneous_rt_trials.csv"
trials = pd.read_csv(INPUT, float_precision="round_trip")
trials["delta_key"] = trials.Delta_dB.round(2)
trials["abs_delta"] = trials.delta_key.abs().round(2)
assert len(trials) == 11400 and trials.Condition.eq("RT").all() and trials.Valid.eq(1).all()
assert np.isfinite(trials.RT_s).all() and trials.Choice.isin([-1, 1]).all()
summary = pd.read_csv(OUTPUT / "group_counts.csv")
assert summary.Group.tolist() == ["Control", "ASD"]
assert summary.participants.tolist() == [11, 8] and summary.trials.tolist() == [6600, 4800]
membership = pd.read_csv(OUTPUT / "group_membership.csv")
for row in membership.itertuples(index=False):
    actual = trials.loc[trials.Participant.eq(row.Participant)]
    assert actual.Group.eq(row.Group).all() and len(actual) == row.trials == 600
assert len(membership) == 19 and membership.Participant.nunique() == 19
points = pd.read_csv(OUTPUT / "psychometric_points.csv")
fits = pd.read_csv(OUTPUT / "psychometric_fits.csv")
chrono = pd.read_csv(OUTPUT / "chronometric_points.csv")
rtd = pd.read_csv(OUTPUT / "rtd_bins.csv")
cdfs = pd.read_csv(OUTPUT / "cdf_bins.csv.gz")
raw = pd.read_csv(OUTPUT / "raw_quantiles.csv")
fine = pd.read_csv(OUTPUT / "fine_quantiles.csv")
cutoffs = pd.read_csv(OUTPUT / "cdf_cutoffs.csv")
scales = pd.read_csv(OUTPUT / "quantile_scaling_fits.csv")
scaled = pd.read_csv(OUTPUT / "scaled_quantiles.csv")
assert [len(x) for x in [points, fits, chrono, raw, fine, cutoffs, scales, scaled]] == [40,4,20,100,1980,10,10,100]
assert rtd["count"].sum() == cdfs["count"].sum() == chrono.trials.sum() == 11400


# %% Choices and four new fits, reconstructed after raw-trial pooling
def sigmoid(x, upper, lower, x0, k):
    return lower + (upper-lower)*expit(k*(x-x0))


for (group, asl), condition in trials.groupby(["Group", "ASL_cd"]):
    saved = points.loc[points.Group.eq(group) & points.ASL_cd.eq(asl)].sort_values("delta_key")
    measured = condition.groupby("delta_key").Choice.agg(["size", lambda x: x.eq(1).sum()])
    np.testing.assert_array_equal(saved.trials, measured["size"])
    np.testing.assert_array_equal(saved.right_trials, measured.iloc[:,1])
    np.testing.assert_allclose(saved.p_right, measured.iloc[:,1]/measured["size"], atol=1e-12, rtol=0)
    probability = saved.p_right.to_numpy()
    parameters, _ = curve_fit(sigmoid, saved.delta_key.to_numpy(), probability,
                             p0=[probability.max(), probability.min(), 0, .1],
                             bounds=([0,0,-np.inf,0],[1,1,np.inf,np.inf]), maxfev=10000)
    fit = fits.loc[fits.Group.eq(group) & fits.ASL_cd.eq(asl)].iloc[0]
    np.testing.assert_allclose(parameters, fit[["upper","lower","x0","k"]].to_numpy(dtype=float), atol=1e-6, rtol=1e-6)
    histogram = rtd.loc[rtd.Group.eq(group) & rtd.ASL_cd.eq(asl)].sort_values("bin_left_s")
    values = np.sort(condition.RT_s.to_numpy())
    edges = np.r_[histogram.bin_left_s, histogram.bin_right_s.iloc[-1]]
    counts = np.diff(np.searchsorted(values, edges, side="left"))
    np.testing.assert_array_equal(histogram["count"], counts)
    np.testing.assert_allclose(histogram.density_per_s, counts/(len(values)*.020), atol=1e-12, rtol=0)
    np.testing.assert_allclose((histogram.density_per_s*.020).sum(), 1, atol=1e-12, rtol=0)


# %% Raw RT pools, sample SEM, CDF counts, and linearly interpolated quantiles
for row in chrono.itertuples(index=False):
    condition = trials.loc[trials.Group.eq(row.Group) & trials.ASL_cd.eq(row.ASL_cd) & trials.abs_delta.eq(row.abs_delta)]
    values = np.sort(condition.RT_s.to_numpy())
    assert len(values) == row.trials == (660 if row.Group == "Control" else 480)
    assert condition.delta_key.gt(0).sum() == row.positive_delta_trials == row.trials//2
    assert condition.delta_key.lt(0).sum() == row.negative_delta_trials == row.trials//2
    np.testing.assert_allclose(row.mean_rt_s, values.mean(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(row.sem_rt_s, values.std(ddof=1)/np.sqrt(len(values)), atol=1e-12, rtol=0)
    saved = cdfs.loc[cdfs.Group.eq(row.Group) & cdfs.ASL_cd.eq(row.ASL_cd) & cdfs.abs_delta.eq(row.abs_delta)].sort_values("bin_left_s")
    cumulative = np.searchsorted(values, saved.bin_right_s.to_numpy(), side="left")
    np.testing.assert_allclose(saved.cdf, cumulative/len(values), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(saved["count"], np.diff(np.r_[0,cumulative]))
    assert saved.cdf.iloc[-1] == 1
    for table, value_field in [(raw,"raw_rt_s"),(fine,"rt_s")]:
        subset = table.loc[table.Group.eq(row.Group) & table.ASL_cd.eq(row.ASL_cd) & table.abs_delta.eq(row.abs_delta)]
        positions = (len(values)-1)*subset["quantile"].to_numpy()
        lo = np.floor(positions).astype(int)
        hi = np.ceil(positions).astype(int)
        expected = values[lo] + (positions-lo)*(values[hi]-values[lo])
        np.testing.assert_allclose(subset[value_field], expected, atol=1e-12, rtol=0)


# %% Fresh cutoffs and scale factors from the pooled CDFs and quantiles
for row in cutoffs.itertuples(index=False):
    condition = cdfs.loc[cdfs.Group.eq(row.Group) & cdfs.abs_delta.eq(row.abs_delta)]
    c20 = condition.loc[condition.ASL_cd.eq(20)].sort_values("bin_left_s")
    c40 = condition.loc[condition.ASL_cd.eq(40)].sort_values("bin_left_s")
    difference = np.abs(c20.cdf.to_numpy()-c40.cdf.to_numpy())
    hit = next(i for i in range(10,len(difference)-10)
               if round(difference[i+1:i+11].mean()-difference[i-10:i].mean(),12) >= .005)
    expected_cutoff = (c20.bin_left_s.iloc[hit]+c20.bin_right_s.iloc[hit])/2
    np.testing.assert_allclose(row.cutoff_s,expected_cutoff,atol=1e-12,rtol=0)
    q = fine.loc[fine.Group.eq(row.Group)&fine.abs_delta.eq(row.abs_delta)]
    q20 = q.loc[q.ASL_cd.eq(20)].sort_values("quantile").rt_s.to_numpy()
    q40 = q.loc[q.ASL_cd.eq(40)].sort_values("quantile").rt_s.to_numpy()
    mask=q40>=expected_cutoff
    x=q40[mask]-expected_cutoff
    y=(q20-q40)[mask]
    factor=1+np.dot(x,y-y[0])/np.dot(x,x)
    fit=scales.loc[scales.Group.eq(row.Group)&scales.abs_delta.eq(row.abs_delta)].iloc[0]
    assert fit.fit_status=="ok" and fit.fit_points==mask.sum() and factor>0
    np.testing.assert_allclose(fit.scale_factor,factor,atol=1e-12,rtol=0)
    for point in scaled.loc[scaled.Group.eq(row.Group)&scaled.abs_delta.eq(row.abs_delta)].itertuples(index=False):
        expected=(point.raw_rt_s if point.ASL_cd==40 or point.raw_rt_s<=expected_cutoff else
                  expected_cutoff+(point.raw_rt_s-expected_cutoff)/factor)
        np.testing.assert_allclose(point.scaled_rt_s,expected,atol=1e-12,rtol=0)


# %% Saved figure and provenance
manifest=json.loads((OUTPUT/"validation.json").read_text())
assert manifest["input_sha256"]==hashlib.sha256(INPUT.read_bytes()).hexdigest()
assert manifest["cutoffs_found"]==manifest["scale_fits_ok"]==10
with Image.open(OUTPUT/manifest["png"]) as image:
    assert image.size==(3330,3204)
with fitz.open(OUTPUT/manifest["pdf"]) as pdf:
    assert len(pdf)==1
    text=pdf[0].get_text()
    assert "6,600" in text and "4,800" in text and "Control" in text and "ASD" in text
    assert text.count("Psychometric")==2 and text.splitlines().count("Chronometric")==2
    for block in pdf[0].get_text("blocks"):
        x0,y0,x1,y1=block[:4]
        assert x0>=0 and y0>=0 and x1<=pdf[0].rect.width and y1<=pdf[0].rect.height
result={"status":"passed","pooled_trials":11400,"control_participants":11,"asd_participants":8,
        "psychometric_points_checked":40,"fits_checked":4,"rtds_checked":4,"cdf_chronometric_conditions_checked":20,
        "raw_quantiles_checked":100,"fine_quantiles_checked":1980,"cutoffs_and_scales_checked":10,
        "scaled_quantiles_checked":100,"figure_pixels":[3330,3204],"pdf_pages":1}
(OUTPUT/"independent_validation.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
