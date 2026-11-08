"""Independent checks of the individual human RT distribution figures.

Run .venv/bin/python ASD_Human_Data/verify_simultaneous_rt_subject_distributions.py.
"""

# %% Imports and paths
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "simultaneous_rt_subject_distributions"
TRIALS = HERE / "simultaneous_rt_trials.csv"
LEVELS = [20, 40]
DIFFERENCES = [.1, .3, .65, 1.3, 2]
QUANTILES = [.1, .3, .5, .7, .9]


# %% Reconstruct choice, timing, and participant order from the extracted trial rows
trials = pd.read_csv(TRIALS, float_precision="round_trip")
trials["delta_key"] = trials.Delta_dB.round(2)
trials["abs_delta"] = trials.delta_key.abs().round(2)
assert len(trials) == 11400 and trials.Condition.eq("RT").all() and trials.Valid.eq(1).all()
assert trials.Choice.isin([-1, 1]).all() and trials.RT_s.between(0, 2.5, inclusive="left").all()
assert trials.groupby("Participant").size().eq(600).all()
assert trials.groupby(["Participant", "ASL_cd", "abs_delta"]).size().eq(60).all()
pages = pd.read_csv(OUTPUT / "participant_pages.csv")
expected = trials[["Participant", "Group", "GroupID"]].drop_duplicates().sort_values(["GroupID", "Participant"])
assert pages.Participant.tolist() == expected.Participant.tolist()
assert pages.Group.tolist() == expected.Group.tolist()
assert pages.page.tolist() == list(range(1, 20))
assert pages.Group.tolist() == ["Control"] * 11 + ["ASD"] * 8

points = pd.read_csv(OUTPUT / "psychometric_points.csv")
fits = pd.read_csv(HERE / "simultaneous_rt_metric_deviations/fits.csv")
assert len(points) == 380 and len(fits) == 38 and fits.fit_status.eq("ok").all()
for condition in points.itertuples(index=False):
    values = trials.loc[trials.Participant.eq(condition.Participant) &
                        trials.ASL_cd.eq(condition.ASL_cd) &
                        trials.delta_key.eq(condition.delta_key), "Choice"]
    assert len(values) == condition.trials == 30
    assert int(values.eq(1).sum()) == condition.right_trials
    np.testing.assert_allclose(condition.p_right, values.eq(1).mean(), atol=1e-13, rtol=0)


# %% Independently reconstruct RTD/CDF counts, pooled means, and order-statistic quantiles
rtd = pd.read_csv(OUTPUT / "rtd_bins.csv")
chrono = pd.read_csv(OUTPUT / "chronometric_points.csv")
cdf = pd.read_csv(OUTPUT / "cdf_bins.csv.gz")
raw = pd.read_csv(OUTPUT / "raw_quantiles.csv")
cutoffs = pd.read_csv(OUTPUT / "cdf_cutoffs.csv")
scales = pd.read_csv(OUTPUT / "quantile_scaling_fits.csv")
scaled = pd.read_csv(OUTPUT / "scaled_quantiles.csv")
assert len(chrono) == 190 and len(raw) == 950 and len(cutoffs) == len(scales) == 95 and len(scaled) == 950
assert rtd["count"].sum() == cdf["count"].sum() == chrono.trials.sum() == 11400

sorted_values = {}
for (participant, asl), group in trials.groupby(["Participant", "ASL_cd"]):
    saved = rtd.loc[rtd.Participant.eq(participant) & rtd.ASL_cd.eq(asl)].sort_values("bin_left_s")
    values = np.sort(group.RT_s.to_numpy())
    edges = np.r_[saved.bin_left_s.to_numpy(), saved.bin_right_s.iloc[-1]]
    expected_counts = np.diff(np.searchsorted(values, edges, side="left"))
    np.testing.assert_array_equal(saved["count"], expected_counts)
    np.testing.assert_allclose(saved.density_per_s, expected_counts / (len(values) * .020), atol=1e-12, rtol=0)
    np.testing.assert_allclose((saved.density_per_s * .020).sum(), 1, atol=1e-12, rtol=0)

for (participant, asl, abs_delta), group in trials.groupby(["Participant", "ASL_cd", "abs_delta"]):
    values = np.sort(group.RT_s.to_numpy())
    sorted_values[(participant, asl, abs_delta)] = values
    assert len(values) == 60
    saved_chrono = chrono.loc[chrono.Participant.eq(participant) & chrono.ASL_cd.eq(asl) &
                             chrono.abs_delta.eq(abs_delta)].iloc[0]
    np.testing.assert_allclose(saved_chrono.mean_rt_s, values.mean(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(saved_chrono.sem_rt_s, values.std(ddof=1) / np.sqrt(60), atol=1e-12, rtol=0)
    saved_cdf = cdf.loc[cdf.Participant.eq(participant) & cdf.ASL_cd.eq(asl) &
                        cdf.abs_delta.eq(abs_delta)].sort_values("bin_left_s")
    edges = saved_cdf.bin_right_s.to_numpy()
    expected_cumulative = np.searchsorted(values, edges, side="left")
    np.testing.assert_array_equal(saved_cdf["count"], np.diff(np.r_[0, expected_cumulative]))
    np.testing.assert_allclose(saved_cdf.cdf, expected_cumulative / 60, atol=1e-12, rtol=0)
    assert saved_cdf.cdf.iloc[-1] == 1
    for row in raw.loc[raw.Participant.eq(participant) & raw.ASL_cd.eq(asl) &
                       raw.abs_delta.eq(abs_delta)].itertuples(index=False):
        position = (len(values) - 1) * row.quantile
        lower = int(np.floor(position))
        fraction = position - lower
        expected_q = values[lower] + fraction * (values[min(lower + 1, len(values) - 1)] - values[lower])
        np.testing.assert_allclose(row.raw_rt_s, expected_q, atol=1e-12, rtol=0)


# %% Recalculate all 95 first-rise cutoffs and reference-based scaling factors
for saved in cutoffs.itertuples(index=False):
    condition = cdf.loc[cdf.Participant.eq(saved.Participant) & cdf.abs_delta.eq(saved.abs_delta)]
    c20 = condition.loc[condition.ASL_cd.eq(20)].sort_values("bin_left_s")
    c40 = condition.loc[condition.ASL_cd.eq(40)].sort_values("bin_left_s")
    difference = np.abs(c20.cdf.to_numpy() - c40.cdf.to_numpy())
    hit = next((i for i in range(10, len(difference) - 10)
                if round(difference[i+1:i+11].mean() - difference[i-10:i].mean(), 12) >= .005), None)
    assert hit is not None
    expected_cutoff = (c20.bin_left_s.iloc[hit] + c20.bin_right_s.iloc[hit]) / 2
    np.testing.assert_allclose(saved.cutoff_s, expected_cutoff, atol=1e-12, rtol=0)
    fit = scales.loc[scales.Participant.eq(saved.Participant) & scales.abs_delta.eq(saved.abs_delta)].iloc[0]
    assert fit.fit_status == "ok" and fit.reference_ASL_cd == 40 and fit.target_ASL_cd == 20
    q20 = np.quantile(sorted_values[(saved.Participant, 20, saved.abs_delta)], np.arange(.01, 1, .01), method="linear")
    q40 = np.quantile(sorted_values[(saved.Participant, 40, saved.abs_delta)], np.arange(.01, 1, .01), method="linear")
    mask = q40 >= expected_cutoff
    x = q40[mask] - expected_cutoff
    y = (q20 - q40)[mask]
    factor = 1 + np.dot(x, y - y[0]) / np.dot(x, x)
    assert fit.fit_points == mask.sum() and factor > 0
    np.testing.assert_allclose(fit.scale_factor, factor, atol=1e-12, rtol=0)
    for row in scaled.loc[scaled.Participant.eq(saved.Participant) & scaled.abs_delta.eq(saved.abs_delta)].itertuples(index=False):
        expected_scaled = (row.raw_rt_s if row.ASL_cd == 40 or row.raw_rt_s <= expected_cutoff else
                           expected_cutoff + (row.raw_rt_s - expected_cutoff) / factor)
        np.testing.assert_allclose(row.scaled_rt_s, expected_scaled, atol=1e-12, rtol=0)


# %% Inspect saved PDF structure and full-size page images
pdf_path = OUTPUT / "human_simultaneous_rt_subject_distributions.pdf"
with fitz.open(pdf_path) as pdf:
    assert len(pdf) == 19
    assert pdf.get_toc() == [[1, f"{row.Group} | Participant {row.Participant}", int(row.page)]
                             for row in pages.itertuples(index=False)]
    for row in pages.itertuples(index=False):
        page = pdf[row.page - 1]
        text = page.get_text()
        assert f"Participant {row.Participant}" in text and row.Group in text
        assert all(label in text for label in ["Psychometric", "RTD", "Chronometric", "CDFs", "Raw quantiles", "Scaled"])
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            assert x0 >= 0 and y0 >= 0 and x1 <= page.rect.width and y1 <= page.rect.height
        with Image.open(OUTPUT / row.png) as image:
            assert image.size == (2775, 1335)

manifest = json.loads((OUTPUT / "validation.json").read_text())
assert manifest["source_sha256"] == hashlib.sha256(TRIALS.read_bytes()).hexdigest()
assert manifest["trials"] == 11400 and manifest["cdf_conditions"] == 190
assert manifest["cutoffs_found"] == manifest["scaling_fits_ok"] == 95
result = {"status": "passed", "participants": 19, "trials": 11400,
          "psychometric_points_checked": 380, "rtd_curves_checked": 38,
          "chronometric_and_cdf_conditions_checked": 190, "quantiles_checked": 950,
          "cutoffs_and_scaling_fits_checked": 95, "scaled_points_checked": 950,
          "pdf_pages_and_pngs_checked": 19}
(OUTPUT / "independent_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
