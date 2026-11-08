# %% Imports and parameters
"""Independent histograms/statistics and actual artist checks for eight PNGs."""
import ast
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import statistics

import fitz
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "fit_animal_by_animal/fig1_tachometric_shared_negative_rts"
LEGACY = REPO / "fit_animal_by_animal/fig1_tachometric_with_negative_rts"
manifest = json.loads((OUTPUT / "run_summary.json").read_text())
WIDTHS, ABLS, ILDS = [10, 15, 20, 25], [20, 40, 60], [1, 2, 4, 8, 16]
COLORS = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c", 8: "#d62728", 16: "#9467bd"}
cohort = [(r["batch_name"], r["animal"]) for r in manifest["cohort"]]
assert len(cohort) == 30 and len(set(cohort)) == 30 and manifest["status"] == "passed"


# %% Independently read the raw numeric values using the original CSV parser
# The same pandas numeric parser preserves the established floating-point RT
# interpretation; all selection/grouping/statistics below are reconstructed.
records = []
for source in manifest["sources"]:
    rows = pd.read_csv(source["path"], usecols=["batch_name", "animal", "success", "abort_event", "RTwrtStim", "ABL", "ILD"])
    rows["source_csv"] = source["path"]
    rows["source_row_index"] = np.arange(len(rows))
    ok = np.isfinite(rows.RTwrtStim) & rows.RTwrtStim.ge(-1) & rows.RTwrtStim.le(1)
    ok &= rows.ABL.isin(ABLS) & rows.ILD.abs().isin(ILDS)
    ok &= rows.success.isin([1, -1]) | rows.abort_event.eq(3)
    records.append(rows.loc[ok])
raw = pd.concat(records, ignore_index=True)
assert len(raw) == 419558 and raw[["batch_name", "animal"]].drop_duplicates().shape[0] == 30
is_abort = ~raw.success.isin([1, -1])
assert is_abort.sum() == 71235 and (~is_abort).sum() == 348323
assert np.array_equal(raw.RTwrtStim.lt(0), is_abort)
aborts = pd.read_csv(OUTPUT / "abort_random_choices.csv", float_precision="round_trip")
expected_choices = np.random.default_rng(42).choice([-1, 1], len(aborts), p=[.5, .5])
np.testing.assert_array_equal(aborts.simulated_choice, expected_choices)
pd.testing.assert_frame_equal(raw.loc[is_abort, ["source_csv", "source_row_index", "batch_name", "animal"]].reset_index(drop=True),
    aborts[["source_csv", "source_row_index", "batch_name", "animal"]], check_dtype=False)
correct = raw.success.eq(1).to_numpy().astype(int)
correct[is_abort] = (raw.loc[is_abort, "ILD"].to_numpy()*expected_choices > 0).astype(int)
np.testing.assert_array_equal(aborts.simulated_correct, correct[is_abort])
assert correct[is_abort].sum() == 35520 and correct[~is_abort].sum() == 280680
raw["correct"] = correct
saved = pd.read_csv(OUTPUT / "selected_trials.csv.gz", float_precision="round_trip")
for name in raw.columns:
    pd.testing.assert_series_equal(raw[name].reset_index(drop=True), saved[name].reset_index(drop=True), check_dtype=False)
np.testing.assert_array_equal(saved.csv_line_number, raw.source_row_index+2)
np.testing.assert_array_equal(saved.abs_ILD, raw.ILD.abs())
np.testing.assert_array_equal(saved.choice_is_simulated, is_abort)
assert (OUTPUT / "abort_random_choices.csv").read_bytes() == (LEGACY / "abort_random_choices.csv").read_bytes()


# %% Cache independent per-animal raw samples; missing conditions retain empty arrays
negative_samples, positive_samples = {}, {}
for batch, animal in cohort:
    a = raw.loc[raw.batch_name.eq(batch) & raw.animal.eq(animal)]
    negative = a.loc[a.RTwrtStim.lt(0)]
    negative_samples[(batch, animal)] = (negative.RTwrtStim.to_numpy(), negative.correct.to_numpy())
    for abl in ABLS:
        for ild in ILDS:
            p = a.loc[a.RTwrtStim.ge(0) & a.ABL.eq(abl) & a.ILD.abs().eq(ild)]
            positive_samples[(batch, animal, abl, ild)] = (p.RTwrtStim.to_numpy(), p.correct.to_numpy())
            if batch == "SD" and ild == 16:
                assert len(p) == 0
negative_audit = pd.read_csv(OUTPUT / "negative_animal_bin_accuracy.csv.gz", float_precision="round_trip")
positive_audit = pd.read_csv(OUTPUT / "positive_animal_bin_accuracy.csv.gz", float_precision="round_trip")
negative_curves = pd.read_csv(OUTPUT / "shared_negative_accuracy.csv", float_precision="round_trip")
positive_curves = pd.read_csv(OUTPUT / "positive_condition_accuracy.csv", float_precision="round_trip")
bins = pd.read_csv(OUTPUT / "bin_definitions.csv", float_precision="round_trip")
old_summary = json.loads((LEGACY / "run_summary.json").read_text())
source = REPO / manifest["source_script"]
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "make_bin_edges")
context = {"np": np}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), context)
old_positive_edges = np.array(old_summary["binning"]["edges_s"])[50:]


# %% Scalar sample statistics, separately from the plotting script's pandas grouping
def scalar_summary(n, c):
    values = [float(correct_count/count) for count, correct_count in zip(n, c) if count > 0]
    center = statistics.fmean(values) if values else math.nan
    sd = statistics.stdev(values) if len(values) >= 2 else math.nan
    sem = sd/math.sqrt(len(values)) if values else math.nan
    return [center, sd, len(values), int(sum(n)), int(sum(c)), sem]


expected = {}
boundary_cases = 0
for width in WIDTHS:
    definition = bins.loc[bins.width_ms.eq(width)].sort_values("bin_index")
    edges = np.r_[definition.bin_start_s.to_numpy(), definition.bin_end_s.iloc[-1]]
    total = {10:200, 15:134, 20:100, 25:80}[width]
    assert len(definition) == total and edges[0] == -1 and edges[-1] == 1
    assert (edges == 0).sum() == 1 and not ((edges[:-1] < 0) & (edges[1:] > 0)).any()
    if width == 20:
        np.testing.assert_array_equal(edges, old_summary["binning"]["edges_s"])
    else:
        positive_edges = np.r_[range(0,1000,width), 1000]/1000
        np.testing.assert_array_equal(edges, np.r_[-positive_edges[:0:-1], positive_edges])
    np.testing.assert_array_equal(context["make_bin_edges"](width, old_positive_edges), edges)
    actual_widths = np.diff(edges)*1000
    np.testing.assert_allclose(definition.actual_width_ms, actual_widths, rtol=0, atol=1e-12)
    if width == 15:
        np.testing.assert_allclose(actual_widths[[0,-1]], [10,10], rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual_widths[1:-1], 15, rtol=0, atol=1e-10)
    else:
        np.testing.assert_allclose(actual_widths, width, rtol=0, atol=1e-10)
    for value in np.r_[edges, np.nextafter(edges,-np.inf), np.nextafter(edges,np.inf)]:
        if -1 <= value <= 1:
            i = min(bisect_right(edges.tolist(), value)-1, total-1)
            hist, _ = np.histogram([value], bins=edges)
            assert hist.sum() == 1 and hist[i] == 1
            assert (i < total//2) == (value < 0)
            boundary_cases += 1
    n_negative, c_negative, n_positive, c_positive = [], [], [], []
    for batch, animal in cohort:
        t, c = negative_samples[(batch, animal)]
        n_negative.append(np.histogram(t,bins=edges)[0][:total//2])
        c_negative.append(np.histogram(t,bins=edges,weights=c)[0][:total//2])
        for abl in ABLS:
            for ild in ILDS:
                t, c = positive_samples[(batch, animal, abl, ild)]
                n_positive.append(np.histogram(t,bins=edges)[0][total//2:])
                c_positive.append(np.histogram(t,bins=edges,weights=c)[0][total//2:])
    n_negative, c_negative = np.array(n_negative), np.array(c_negative)
    n_positive, c_positive = np.array(n_positive), np.array(c_positive)
    assert n_negative.sum() == 71235 and c_negative.sum() == 35520
    assert n_positive.sum() == 348323 and c_positive.sum() == 280680
    neg = negative_audit.loc[negative_audit.width_ms.eq(width)].sort_values(["batch_name","animal","bin_index"])
    pos = positive_audit.loc[positive_audit.width_ms.eq(width)].sort_values(["batch_name","animal","ABL","abs_ILD","bin_index"])
    for table,n,c in [(neg,n_negative,c_negative),(pos,n_positive,c_positive)]:
        np.testing.assert_array_equal(table.n_trials,n.ravel())
        np.testing.assert_array_equal(table.n_correct,c.ravel())
        accuracy = np.divide(c,n,out=np.full(c.shape,np.nan),where=n>0)
        np.testing.assert_allclose(table.accuracy,accuracy.ravel(),rtol=0,atol=2e-16,equal_nan=True)
        indices = table.bin_index.to_numpy()
        np.testing.assert_array_equal(table.bin_start_s,edges[indices])
        np.testing.assert_array_equal(table.bin_end_s,edges[indices+1])
    rows = negative_curves.loc[negative_curves.width_ms.eq(width)].sort_values("bin_index")
    assert len(rows) == total//2
    statistics_array = np.array([scalar_summary(n_negative[:,i],c_negative[:,i]) for i in range(total//2)])
    np.testing.assert_allclose(rows[["mean_accuracy","std_accuracy","n_animals","n_trials","n_correct","sem_accuracy"]],statistics_array,rtol=0,atol=1e-12,equal_nan=True)
    expected[(width,"shared_negative",None,None)] = (rows.bin_center_s.to_numpy()*1000,statistics_array[:,0],statistics_array[:,5])
    n_positive = n_positive.reshape(30,3,5,total//2)
    c_positive = c_positive.reshape(30,3,5,total//2)
    for ai,abl in enumerate(ABLS):
        for ii,ild in enumerate(ILDS):
            rows = positive_curves.loc[positive_curves.width_ms.eq(width) & positive_curves.ABL.eq(abl) & positive_curves.abs_ILD.eq(ild)].sort_values("bin_index")
            assert len(rows) == total//2
            summary = np.array([scalar_summary(n_positive[:,ai,ii,i],c_positive[:,ai,ii,i]) for i in range(total//2)])
            np.testing.assert_allclose(rows[["mean_accuracy","std_accuracy","n_animals","n_trials","n_correct","sem_accuracy"]],summary,rtol=0,atol=1e-12,equal_nan=True)
            assert rows.n_animals.max() <= (24 if ild == 16 else 30)
            assert rows.loc[rows.n_animals.eq(0),"mean_accuracy"].isna().all()
            assert rows.loc[rows.n_animals.lt(2),"sem_accuracy"].isna().all()
            expected[(width,"positive",abl,ild)] = (rows.bin_center_s.to_numpy()*1000,summary[:,0],summary[:,5])


# %% Original 20 ms positive accuracy and all CDF bytes/arrays must be preserved
old_positive = pd.read_csv(LEGACY / "group_bin_summary.csv",float_precision="round_trip")
old_positive = old_positive.loc[old_positive.bin_start_s.ge(0)]
columns = ["ABL","abs_ILD","bin_index","n_trials","n_correct","n_animals","mean_accuracy","std_accuracy","sem_accuracy"]
pd.testing.assert_frame_equal(positive_curves.loc[positive_curves.width_ms.eq(20),columns].reset_index(drop=True),
    old_positive[columns].reset_index(drop=True),check_exact=False,atol=1e-14,rtol=0)
for name in ["animal_cumulative_rtd.csv","group_cumulative_rtd.csv"]:
    assert (OUTPUT/name).read_bytes() == (LEGACY/name).read_bytes()
cdfs = pd.read_csv(OUTPUT/"group_cumulative_rtd.csv",float_precision="round_trip")
for abl in ABLS:
    for ild in ILDS:
        c = cdfs.loc[cdfs.ABL.eq(abl) & cdfs.abs_ILD.eq(ild)].sort_values("bin_index")
        assert c.n_animals.eq(24 if ild == 16 else 30).all()
        assert c.mean_cumulative_fraction.iloc[0] == 0 and c.mean_cumulative_fraction.iloc[-1] == 1
        assert np.all(np.diff(c.mean_cumulative_fraction) >= 0)
        expected[(None,"cdf",abl,ild)] = (c.time_s.to_numpy()*1000,c.mean_cumulative_fraction.to_numpy(),c.sem_cumulative_fraction.to_numpy())
print("Raw selections, seed-42 replay, all four histogram/statistic sets, 20 ms positive values and fixed CDFs passed",flush=True)


# %% Verify actual Matplotlib data lines and SEM polygon vertices for all eight images
artists = json.loads((OUTPUT / "plotted_artists.json").read_text())
arrays = np.load(OUTPUT / "plotted_artist_arrays.npz")
assert len(artists) == 264 and len(manifest["figures"]) == 8
assert sum(a["kind"]=="shared_negative" for a in artists) == 24
for a in artists:
    key = (None if a["kind"]=="cdf" else a["width_ms"], a["kind"], None if a["kind"]=="shared_negative" else a["ABL"], a["abs_ILD"])
    x,mean,sem = expected[key]
    np.testing.assert_array_equal(arrays[a["array_key"]+"_x"],x)
    np.testing.assert_allclose(arrays[a["array_key"]+"_y"],mean,rtol=0,atol=1e-12,equal_nan=True)
    assert a["color"] == ("#000000" if a["kind"]=="shared_negative" else COLORS[a["abs_ILD"]])
    assert a["shading_color"] == ("#666666" if a["kind"]=="shared_negative" else COLORS[a["abs_ILD"]])
    assert a["line_style"] == "-" and a["ylim"] == [0,1]
    if a["kind"] != "cdf":
        assert a["marker"] == "o"
        assert np.all(x < 0) if a["kind"]=="shared_negative" else np.all(x > 0)
    paths = [arrays[k] for k in a["shading_paths"]]
    vertices = np.concatenate(paths) if paths else np.empty((0,2))
    finite = np.isfinite(mean) & np.isfinite(sem)
    np.testing.assert_array_equal(np.unique(vertices[:,0]),x[finite])
    for xi,m,s in zip(x[finite],mean[finite],sem[finite]):
        y = vertices[vertices[:,0]==xi,1]
        np.testing.assert_allclose([y.min(),y.max()],[m-s,m+s],rtol=0,atol=1e-12)
    for path in paths:
        positions = np.searchsorted(x,np.unique(path[:,0]))
        assert np.all(np.diff(positions)==1)  # No SEM polygon bridges an empty bin.
for fig in manifest["figures"]:
    subset = [a for a in artists if a["figure"]==fig["name"]]
    assert len(subset) == 33 and sum(a["kind"]=="shared_negative" for a in subset)==3
    assert all(a["xlim_ms"]==fig["display_xlim_ms"] for a in subset)
    with Image.open(REPO/fig["png"]) as image:
        assert image.size == (3900,3750)
        image.verify()
    with fitz.open(REPO/fig["local_pdf"]) as document:
        assert len(document)==1
        page=document[0]
        text=page.get_text()
        assert f"Accuracy: {fig['width_ms']} ms" in text and "CDF: fixed 20 ms" in text
        for block in page.get_text("blocks"):
            assert 0 <= block[0] <= block[2] <= page.rect.width and 0 <= block[1] <= block[3] <= page.rect.height
for relative,digest in manifest["protected_file_sha256"].items():
    with (REPO/relative).open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest()==digest,relative
report = {"status":"passed", "source_script":str(Path(__file__).relative_to(REPO)), "cohort_animals":30,
    "selected_trials_per_width":419558,"valid_choice_trials":348323,"negative_rt_fixation_aborts":71235,
    "all_original_seed42_choices_reused":True,"bin_counts":{"10":200,"15":134,"20":100,"25":80},
    "exact_and_adjacent_boundary_cases":boundary_cases,"animal_negative_rows_verified":len(negative_audit),
    "animal_positive_rows_verified":len(positive_audit),"negative_group_rows_verified":len(negative_curves),
    "positive_group_rows_verified":len(positive_curves),"actual_line_and_SEM_artists_verified":len(artists),
    "shared_black_artists_verified":24,"PNG_count":8,"fixed_CDF_tables_byte_identical":True,
    "20ms_positive_accuracy_unchanged":True,"all_prior_outputs_preserved":len(manifest["protected_file_sha256"]),
    "empty_bins_gaps_singleton_SEM_unavailable":True,"figures_have_readable_text_bounds":True}
(OUTPUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
