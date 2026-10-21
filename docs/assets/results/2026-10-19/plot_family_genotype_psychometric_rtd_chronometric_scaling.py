# %% Imports
"""Pool current ASD trials within gene family/genotype; two pages, nine panels/group.

SHANK3 and CNTNAP2 stay separate. Top row: unrestricted valid-choice
psychometrics and all-finite-RT RTDs/chronometrics. Bottom row: [0,1]-second
CDF thresholds, raw quantiles and Figure 1 quantile scaling. Run with .venv/bin/python.
"""
import hashlib
import json
from pathlib import Path
import warnings

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expit


# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
REFERENCE = REPO / "asd_analysis/session_type_1_all_animals_psychometric_rtd_chronometric"
OUTPUT = REPO / "asd_analysis/session_type_1_family_genotype_psychometric_rtd_chronometric_scaling"
FAMILIES = ["SHANK3", "CNTNAP2"]
GENOTYPES = ["wt", "het", "hom"]
GENOTYPE_LABELS = {"wt": "WT", "het": "HET", "hom": "HOM"}
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
COLORS = {20: "tab:blue", 40: "tab:orange", 60: "tab:green"}
ABL_MARKERS = {20: "o", 40: "s", 60: "^"}
ILD_STYLES = {1: "-", 2: "--", 4: ":", 8: "-.", 16: (0, (5, 1, 1, 1, 1, 1))}
ILD_COLORS = dict(zip(ABS_ILDS, [to_hex(c) for c in plt.cm.viridis(np.linspace(.08, .86, 5))]))
PLOT_QUANTILES = np.array([.1, .3, .5, .7, .9])
FIT_QUANTILES = np.arange(.01, 1, .01)
QUANTILE_COLORS = [to_hex(c) for c in plt.cm.plasma(np.linspace(.08, .86, 5))]
QUANTILE_MARKERS = ["o", "s", "^", "D", "v"]
CDF_BIN_WIDTH = .001
RTD_BIN_WIDTH = .020
NEW_RT_WINDOW = (0, 1)
CDF_DISPLAY_XLIM = (0, .15)
CHRONOMETRIC_YLIM = (0, .45)
CHRONOMETRIC_OFFSETS = {20: -.10, 40: 0, 60: .10}
ONSET_WINDOW_BINS = 10
ONSET_RISE = .005
FIT_MAX_RT = .3
FIT_BOUNDS = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
INITIAL_K = .1
MAXFEV = 10000
PAGE_WIDTH = 18.24
GROUP_BLOCK_HEIGHT = 6.6
HEADER_HEIGHT = 1.6
FOOTER_HEIGHT = .6
UPPER_PANEL_HEIGHT = 2.2014248704663224
NEW_PANEL_SIZE = 2.2
PNG_DPI = 160
NAME = "ASD_session_type_1_family_genotype_psychometric_rtd_chronometric_scaling"
EXPECTED = {"animals": 53, "choices": 451381, "right": 231263, "left": 220118,
            "finite_rt": 449193, "nonfinite_rt": 2188, "rt_above_1": 2244, "window_rt": 446949}
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Stable mathematical steps from the existing Figure 1 workflow
def detect_onset(curve, window_bins=ONSET_WINDOW_BINS, rise=ONSET_RISE):
    """First forward-minus-backward window rise; the current bin is excluded."""
    previous = np.full(len(curve), np.nan)
    following = np.full(len(curve), np.nan)
    if len(curve) >= 2 * window_bins + 1:
        windows = np.lib.stride_tricks.sliding_window_view(curve, window_bins).mean(axis=1)
        previous[window_bins:-window_bins] = windows[:len(curve) - 2 * window_bins]
        following[window_bins:-window_bins] = windows[window_bins + 1:]
    hits = np.flatnonzero(following - previous >= rise)
    return (int(hits[0]) if len(hits) else None), previous, following


def fit_quantile_scale(reference_q, other_q, cutoff, max_rt=FIT_MAX_RT):
    """Match the original Figure 1 shifted-difference, through-origin fit."""
    result = {"fit_status": "missing_cutoff", "fit_points": 0, "slope_m": np.nan,
              "scale_factor": np.nan, "first_difference_s": np.nan, "sum_x_squared": np.nan}
    mask = np.zeros(len(reference_q), dtype=bool)
    if not np.isfinite(cutoff):
        return result, mask
    if not np.isfinite(reference_q).all() or not np.isfinite(other_q).all():
        result["fit_status"] = "nonfinite_quantiles"
        return result, mask
    mask = (reference_q >= cutoff) & (reference_q <= max_rt)
    result["fit_points"] = int(mask.sum())
    if mask.sum() < 2:
        result["fit_status"] = "insufficient_fit_points"
        return result, mask
    x = reference_q[mask] - cutoff
    y = (other_q - reference_q)[mask]
    denominator = float(x @ x)
    result.update(first_difference_s=float(y[0]), sum_x_squared=denominator)
    if denominator <= 0:
        result["fit_status"] = "zero_reference_range"
        return result, mask
    slope = float(x @ (y - y[0]) / denominator)
    factor = 1 + slope
    result.update(slope_m=slope, scale_factor=factor,
                  fit_status="ok" if np.isfinite(factor) and factor > 0 else "invalid_scale_factor")
    return result, mask


def rescale_quantiles(values, cutoff, factor):
    if not np.isfinite(cutoff) or not np.isfinite(factor) or factor <= 0:
        return np.full_like(values, np.nan, dtype=float)
    return np.where(values > cutoff, cutoff + (values - cutoff) / factor, values)


def sigmoid(x, upper, lower, x0, k):
    return lower + (upper-lower)/(1+np.exp(-k*(x-x0)))


def pdf_coordinates(ax, x, y, page_height):
    box = ax.get_position()
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    return ((box.x0 + (x-xlo)/(xhi-xlo)*box.width)*PAGE_WIDTH*72,
            (1-box.y0-(y-ylo)/(yhi-ylo)*box.height)*page_height*72)


# %% Snapshot current inputs and prior output assets; geometry is the only old value reused
old_manifest = json.loads((REFERENCE / "validation.json").read_text())
protected = set(REPO.glob("SHANK3_cohort*.csv")) | set(REPO.glob("CNTNAP2_cohort*.csv"))
protected.update(INPUT.parent.glob("*.csv"))
protected.update(INPUT.parent.glob("*.json"))
for folder in (REPO / "asd_analysis").glob("session_type_1_*"):
    if folder.is_dir() and folder != OUTPUT:
        protected.update(p for p in folder.rglob("*") if p.is_file())
for date in ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-17", "2026-10-18"]:
    protected.update(p for p in (REPO / f"docs/assets/results/{date}").glob("*") if p.is_file())
protected_hashes = {}
for path in sorted(protected):
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
cleanup = json.loads((INPUT.parent / "speaker_calibration_cleanup.json").read_text())
assert cleanup["status"] == "passed"
assert input_sha256 == cleanup["files"][str(INPUT.relative_to(REPO))]["after"]["sha256"]


# %% Select once; keep all valid choices, with separate finite/all-RT and [0,1] populations
data = pd.read_csv(INPUT, usecols=["filename", "animal", "genotype", "session", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "success", "response_poke", "ABL", "ILD", "timed_rt"], low_memory=False)
assert len(data) == 610603 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all() and data.is_short_sound.eq(0).all()
assert not (data.filename.eq("CNTNAP2_cohort4.csv") & data.session.lt(13)).any()
data["plot_ABL"] = data.ABL.replace({59: 60})
selected = data.loc[data.success.isin([1,-1]) & data.plot_ABL.isin(ABLS) & data.ILD.isin(SIGNED_ILDS)].copy()
assert selected.response_poke.isin([-1,1]).all()
selected["family"] = selected.filename.str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
selected["group_id"] = selected.family + "/" + selected.genotype
selected["abs_ILD"] = selected.ILD.abs().astype(int)
selected["plot_ABL"] = selected.plot_ABL.astype(int)
selected["right"] = selected.response_poke.eq(1)
selected["correct"] = selected.success.eq(1)
selected["finite_rt"] = np.isfinite(selected.timed_rt)
selected["window_rt"] = selected.finite_rt & selected.timed_rt.between(*NEW_RT_WINDOW)
selected["positive_ild"] = selected.ILD.gt(0)
selected["above_1"] = selected.finite_rt & selected.timed_rt.gt(1)
finite = selected.loc[selected.finite_rt].copy()
window = selected.loc[selected.window_rt].copy()
assert finite.timed_rt.ge(0).all() and selected.family.isin(FAMILIES).all()
observed = {"animals": selected.animal.nunique(), "choices": len(selected), "right": int(selected.right.sum()),
    "left": int((~selected.right).sum()), "finite_rt": len(finite), "nonfinite_rt": len(selected)-len(finite),
    "rt_above_1": int(selected.above_1.sum()), "window_rt": len(window)}
assert observed == EXPECTED
print("Current pooled population:", json.dumps(observed), flush=True)


# %% Group membership and choice/RT denominators; actual trials determine pooling weights
keys = ["group_id", "family", "genotype"]
members = selected.groupby(keys, as_index=False).agg(
    animals=("animal", "nunique"), choice_trials=("success", "size"), right_trials=("right", "sum"),
    finite_rt_trials=("finite_rt", "sum"), window_trials=("window_rt", "sum"), rt_above_1_s=("above_1", "sum"))
members["nonfinite_rt_trials"] = members.choice_trials - members.finite_rt_trials
members["left_trials"] = members.choice_trials - members.right_trials
members["pdf_page"] = members.family.map({f:i+1 for i,f in enumerate(FAMILIES)})
order = [f"{f}/{g}" for f in FAMILIES for g in GENOTYPES]
members = members.set_index("group_id").loc[order].reset_index()
assert members.animals.tolist() == [6,7,7,10,11,12]
members.to_csv(OUTPUT / "group_counts.csv", index=False)
identity = members.set_index("group_id")[["family", "genotype"]].to_dict("index")
membership = selected.groupby(keys+["filename", "animal"], as_index=False).agg(
    choice_trials=("success", "size"), right_trials=("right", "sum"), finite_rt_trials=("finite_rt", "sum"), window_trials=("window_rt", "sum"))
assert len(membership) == 53
membership.to_csv(OUTPUT / "group_membership.csv", index=False)
abl_counts = selected.groupby(keys+["plot_ABL"], as_index=False).agg(
    choice_trials=("success", "size"), finite_rt_trials=("finite_rt", "sum"), window_trials=("window_rt", "sum"), animals=("animal", "nunique"))
abl_counts["nonfinite_rt_trials"] = abl_counts.choice_trials - abl_counts.finite_rt_trials
abl_counts.to_csv(OUTPUT / "abl_counts.csv", index=False)


# %% Pool measured choice counts before fitting 18 ten-point psychometrics
psy_points = selected.groupby(keys+["plot_ABL", "ILD"], as_index=False).agg(
    trials=("success", "size"), right_trials=("right", "sum"), correct_trials=("correct", "sum"))
psy_points["left_trials"] = psy_points.trials - psy_points.right_trials
psy_points["p_right"] = psy_points.right_trials / psy_points.trials
psy_points["fitted_p_right"] = np.nan
fit_records = []
for (group_id, abl), condition in psy_points.groupby(["group_id", "plot_ABL"], sort=False):
    condition = condition.sort_values("ILD")
    x, p = condition.ILD.to_numpy(dtype=float), condition.p_right.to_numpy()
    assert condition.ILD.tolist() == SIGNED_ILDS
    initial = [float(max(p)),float(min(p)),float(np.median(x)),INITIAL_K]
    record = {"group_id":group_id, **identity[group_id], "plot_ABL":int(abl), "trials":int(condition.trials.sum()),
        "initial_parameters":json.dumps(initial), "upper":np.nan, "lower":np.nan, "x0":np.nan, "k":np.nan,
        "fit_status":"optimizer_failed", "optimizer_message":"", "warnings":""}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parameters, covariance, info, message, code = curve_fit(sigmoid,x,p,p0=initial,bounds=FIT_BOUNDS,maxfev=MAXFEV,full_output=True)
        record.update(dict(zip(["upper","lower","x0","k"],parameters)))
        record.update(optimizer_message=message,optimizer_code=int(code),function_evaluations=int(info["nfev"]),
            covariance_finite=bool(np.isfinite(covariance).all()),warnings="; ".join(str(w.message) for w in caught))
        valid = np.isfinite(parameters).all() and parameters[0]>parameters[1] and parameters[3]>0
        record["fit_status"] = "ok" if valid else "non_increasing_or_nonfinite"
        if valid:
            prediction = sigmoid(x,*parameters)
            psy_points.loc[condition.index,"fitted_p_right"] = prediction
            record["unweighted_sse"] = float(np.dot(prediction-p,prediction-p))
    except (RuntimeError,ValueError,FloatingPointError) as error:
        record["optimizer_message"] = str(error)
    fit_records.append(record)
psy_fits = pd.DataFrame(fit_records)
assert len(psy_fits) == 18 and len(psy_points) == 180
psy_points.to_csv(OUTPUT / "psychometric_points.csv", index=False)
psy_fits.to_csv(OUTPUT / "psychometric_fits.csv", index=False)
psy_fits = psy_fits.set_index(["group_id", "plot_ABL"])


# %% All-finite-RT histograms; common 20 ms grid covers the full recorded range
rtd_curves, rtd_tables = {}, []
rtd_edges = np.arange(int(np.ceil(finite.timed_rt.max()/RTD_BIN_WIDTH))+1)*RTD_BIN_WIDTH
for (group_id,abl), condition in finite.groupby(["group_id","plot_ABL"],sort=False):
    hist = np.histogram(condition.timed_rt,bins=rtd_edges)[0]
    n = len(condition)
    assert hist.sum() == n
    density = hist/(n*RTD_BIN_WIDTH)
    assert np.isclose(np.sum(density*np.diff(rtd_edges)),1)
    rtd_curves[(group_id,abl)] = (density,rtd_edges,n)
    rtd_tables.append(pd.DataFrame({"group_id":group_id,"plot_ABL":abl,"bin_index":np.arange(len(hist)),
        "bin_left_s":rtd_edges[:-1],"bin_right_s":rtd_edges[1:],"bin_count":hist,"normalization_trials":n,"density_per_s":density}))
pd.concat(rtd_tables,ignore_index=True).to_csv(OUTPUT/"rtd_histogram_bins.csv",index=False)


# %% Pool actual +ILD/-ILD RTs before computing means, sample SDs, and trial SEMs
condition_keys = keys+["plot_ABL","abs_ILD"]
counts = selected.groupby(condition_keys,as_index=False).agg(
    valid_choice_trials=("success","size"),finite_rt_trials=("finite_rt","sum"),window_trials=("window_rt","sum"),
    positive_ild_trials=("positive_ild","sum"),rt_above_1_s=("above_1","sum"))
counts["negative_ild_trials"] = counts.valid_choice_trials-counts.positive_ild_trials
counts["nonfinite_rt_trials"] = counts.valid_choice_trials-counts.finite_rt_trials
rt_stats = finite.groupby(condition_keys,as_index=False).agg(
    mean_rt_s=("timed_rt","mean"),sample_sd_rt_s=("timed_rt",lambda x:x.std(ddof=1)))
chrono = counts.merge(rt_stats,on=condition_keys,validate="one_to_one")
chrono["sem_rt_s"] = chrono.sample_sd_rt_s/np.sqrt(chrono.finite_rt_trials)
chrono["mean_minus_sem_s"] = chrono.mean_rt_s-chrono.sem_rt_s
chrono["mean_plus_sem_s"] = chrono.mean_rt_s+chrono.sem_rt_s
chrono["plot_x"] = chrono.abs_ILD.map({v:i for i,v in enumerate(ABS_ILDS)})+chrono.plot_ABL.map(CHRONOMETRIC_OFFSETS)
assert len(chrono)==90 and chrono.finite_rt_trials.sum()==449193
assert chrono.mean_minus_sem_s.ge(CHRONOMETRIC_YLIM[0]).all() and chrono.mean_plus_sem_s.le(CHRONOMETRIC_YLIM[1]).all()
chrono.to_csv(OUTPUT/"chronometric_points.csv",index=False)
window_counts = window.groupby(["group_id","plot_ABL","abs_ILD"],as_index=False).agg(
    positive_ild_window_trials=("positive_ild","sum"))
counts=counts.merge(window_counts,on=["group_id","plot_ABL","abs_ILD"],validate="one_to_one")
counts["negative_ild_window_trials"] = counts.window_trials-counts.positive_ild_window_trials
assert counts.window_trials.sum()==446949 and counts.window_trials.min()>=5
counts.to_csv(OUTPUT/"condition_counts.csv",index=False)


# %% Per-condition histograms, CDFs, five displayed quantiles, and 99 fitting quantiles
bin_edges = np.arange(0, 1 + CDF_BIN_WIDTH, CDF_BIN_WIDTH)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
cdfs, quantiles = {}, {}
bin_tables, fine_tables, raw_quantile_records = [], [], []
for (group_id, abl, ild), group in window.groupby(["group_id", "plot_ABL", "abs_ILD"]):
    values = group.timed_rt.to_numpy()
    hist = np.histogram(values, bins=bin_edges)[0]
    assert hist.sum() == len(values)
    cdf = np.cumsum(hist) / len(values)
    coarse = np.quantile(values, PLOT_QUANTILES, method="linear")
    fine = np.quantile(values, FIT_QUANTILES, method="linear")
    key = (group_id, int(abl), int(ild))
    cdfs[key] = cdf
    quantiles[key] = {"coarse": coarse, "fine": fine}
    bin_tables.append(pd.DataFrame({"group_id": group_id, "plot_ABL": int(abl), "abs_ILD": int(ild),
        "bin_index": np.arange(len(hist)), "bin_left_s": bin_edges[:-1], "bin_right_s": bin_edges[1:],
        "bin_center_s": bin_centers, "hist_count": hist, "cdf": cdf}))
    fine_tables.append(pd.DataFrame({"group_id": group_id, "plot_ABL": int(abl), "abs_ILD": int(ild),
                                    "quantile": FIT_QUANTILES, "rt_s": fine}))
    for q, value in zip(PLOT_QUANTILES, coarse):
        raw_quantile_records.append({"group_id": group_id, "plot_ABL": int(abl), "abs_ILD": int(ild),
                                    "quantile": float(q), "raw_rt_s": float(value), "window_trials": len(values)})
pd.concat(bin_tables, ignore_index=True).to_csv(OUTPUT / "histogram_cdf_bins.csv.gz", index=False, compression="gzip")
pd.concat(fine_tables, ignore_index=True).to_csv(OUTPUT / "fine_quantiles.csv.gz", index=False, compression="gzip")
raw_quantiles = pd.DataFrame(raw_quantile_records)
assert len(raw_quantiles) == 450 and len(cdfs) == 90
raw_quantiles.to_csv(OUTPUT / "raw_quantiles.csv", index=False)
del bin_tables, fine_tables


# %% Fresh mean absolute pairwise ΔCDFs and cutoffs, one per group_id/|ILD|
delta_curves, threshold_records, delta_tables = {}, [], []
for group_id in members.group_id:
    for ild in ABS_ILDS:
        c20, c40, c60 = [cdfs[(group_id, abl, ild)] for abl in ABLS]
        d2040, d4060, d2060 = abs(c20 - c40), abs(c40 - c60), abs(c20 - c60)
        delta = (d2040 + d4060 + d2060) / 3
        onset, previous, following = detect_onset(delta)
        cutoff = float(bin_centers[onset]) if onset is not None else np.nan
        threshold_records.append({"group_id": group_id, **identity[group_id], "abs_ILD": ild,
            "threshold_status": "ok" if onset is not None else "no_onset", "onset_bin_index": onset,
            "cutoff_s": cutoff, "cutoff_ms": cutoff * 1000, "window_bins": ONSET_WINDOW_BINS,
            "required_rise": ONSET_RISE, "previous_mean": previous[onset] if onset is not None else np.nan,
            "following_mean": following[onset] if onset is not None else np.nan,
            "observed_rise": following[onset] - previous[onset] if onset is not None else np.nan})
        delta_curves[(group_id, ild)] = delta
        delta_tables.append(pd.DataFrame({"group_id": group_id, "abs_ILD": ild,
            "bin_index": np.arange(len(delta)), "bin_center_s": bin_centers,
            "abs_20_minus_40": d2040, "abs_40_minus_60": d4060, "abs_20_minus_60": d2060,
            "mean_abs_delta_cdf": delta, "previous_mean": previous, "following_mean": following,
            "window_rise": following - previous}))
thresholds = pd.DataFrame(threshold_records)
thresholds.to_csv(OUTPUT / "thresholds.csv", index=False)
pd.concat(delta_tables, ignore_index=True).to_csv(OUTPUT / "delta_cdf_curves.csv.gz", index=False, compression="gzip")
del delta_tables
threshold_lookup = thresholds.set_index(["group_id", "abs_ILD"])
assert len(thresholds) == 30
print(f"Thresholds: {thresholds.threshold_status.eq('ok').sum()}/30; "
      f"range {thresholds.cutoff_ms.min():.1f}–{thresholds.cutoff_ms.max():.1f} ms", flush=True)


# %% Figure 1 scaling fits, their complete fitting inputs, and transformed quantiles
scale_records, fit_input_tables, scaled_records = [], [], []
for group_id in members.group_id:
    for ild in ABS_ILDS:
        cutoff = threshold_lookup.loc[(group_id, ild), "cutoff_s"]
        q60 = quantiles[(group_id, 60, ild)]["fine"]
        for abl in ABLS:
            raw = quantiles[(group_id, abl, ild)]["coarse"]
            if abl == 60:
                transformed, status = raw.copy(), "reference_unchanged"
            else:
                other = quantiles[(group_id, abl, ild)]["fine"]
                result, mask = fit_quantile_scale(q60, other, cutoff)
                scale_records.append({"group_id": group_id, **identity[group_id], "abs_ILD": ild, "plot_ABL": abl,
                                      "cutoff_s": cutoff, "fit_max_rt_s": FIT_MAX_RT, **result})
                fit_input_tables.append(pd.DataFrame({"group_id": group_id, "plot_ABL": abl, "abs_ILD": ild,
                    "quantile": FIT_QUANTILES, "reference_rt_s": q60, "other_rt_s": other,
                    "difference_s": other - q60, "x_s": q60 - cutoff,
                    "shifted_difference_s": other - q60 - result["first_difference_s"], "used_for_fit": mask}))
                status = result["fit_status"]
                transformed = rescale_quantiles(raw, cutoff, result["scale_factor"]) if status == "ok" else np.full_like(raw, np.nan)
            for q, before, after in zip(PLOT_QUANTILES, raw, transformed):
                scaled_records.append({"group_id": group_id, "plot_ABL": abl, "abs_ILD": ild,
                    "quantile": float(q), "raw_rt_s": float(before), "scaled_rt_s": float(after),
                    "cutoff_s": cutoff, "scaling_status": status})
scales = pd.DataFrame(scale_records)
scaled = pd.DataFrame(scaled_records)
assert len(scales) == 60 and len(scaled) == 450
scales.to_csv(OUTPUT / "scaling_fits.csv", index=False)
scaled.to_csv(OUTPUT / "scaled_quantiles.csv", index=False)
pd.concat(fit_input_tables, ignore_index=True).to_csv(OUTPUT / "scaling_fit_inputs.csv.gz", index=False, compression="gzip")
del fit_input_tables
print(f"Rescaling fits: {scales.fit_status.eq('ok').sum()}/60 valid", flush=True)


# %% Two rows per genotype, with one family on each page
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
pdf_path = OUTPUT / f"{NAME}.pdf"
page_records, layout_records, plotted_records, psy_drawn, chrono_drawn = [], [], [], [], []
old_widths = old_manifest["pages"][0]["panel_widths_in"]
left, right = .043 * PAGE_WIDTH, .98 * PAGE_WIDTH
upper_gap = (right - left - sum(old_widths)) / 2
upper_x = [left, left + old_widths[0] + upper_gap, right - old_widths[2]]
lower_gap = (right - left - 6 * NEW_PANEL_SIZE) / 5
lower_x = [left + j * (NEW_PANEL_SIZE + lower_gap) for j in range(6)]
with PdfPages(pdf_path, metadata={"Title": "ASD family/genotype pooled psychometrics, RTDs, chronometrics, CDFs and quantile scaling"}) as pdf:
    for page_number, batch in members.groupby("pdf_page", sort=True):
        height = HEADER_HEIGHT + GROUP_BLOCK_HEIGHT * len(batch) + FOOTER_HEIGHT
        fig = plt.figure(figsize=(PAGE_WIDTH, height))
        fig.text(.5, 1 - .31 / height, f"{FAMILIES[int(page_number)-1]} | WT, HET, HOM | Session type 1 | Page {page_number}/2",
                 ha="center", fontsize=14)
        fig.text(.5, 1 - .59 / height, "Raw trials pooled within family/genotype | Upper: valid choices / all finite RTs | Lower: RT in [0,1] s; fresh group cutoffs",
                 ha="center", fontsize=9)
        abl_legend = fig.legend(handles=[Line2D([], [], color=COLORS[a], label=f"ABL {a} dB") for a in ABLS],
            loc="upper center", bbox_to_anchor=(.5, 1 - .73 / height), ncol=3, frameon=False, fontsize=9)
        ild_legend = fig.legend(handles=[Line2D([], [], color=ILD_COLORS[i], linestyle=ILD_STYLES[i], label=f"|ILD| {i}") for i in ABS_ILDS],
            loc="upper center", bbox_to_anchor=(.5, 1 - .98 / height), ncol=5, frameon=False, fontsize=8,
            title="CDF: use these line styles | ΔCDF: use these colors")
        ild_legend.get_title().set_fontsize(8)
        fig.legend(handles=[Line2D([], [], color=QUANTILE_COLORS[j], marker=QUANTILE_MARKERS[j], label=f"{int(q*100)}%")
            for j, q in enumerate(PLOT_QUANTILES)], loc="upper center", bbox_to_anchor=(.5, 1 - 1.31 / height),
            ncol=5, frameon=False, fontsize=8)
        for row, member in enumerate(batch.itertuples(index=False)):
            group_id = member.group_id
            label = f"{member.family} | {GENOTYPE_LABELS[member.genotype]}"
            block_top = height - HEADER_HEIGHT - row * GROUP_BLOCK_HEIGHT
            upper_y = block_top - .45 - UPPER_PANEL_HEIGHT
            upper_axes = []
            for j in range(3):
                h = old_widths[2] if j == 2 else UPPER_PANEL_HEIGHT
                y = upper_y + (UPPER_PANEL_HEIGHT - h) / 2
                upper_axes.append(fig.add_axes([upper_x[j]/PAGE_WIDTH, y/height, old_widths[j]/PAGE_WIDTH, h/height]))
            psy_ax, rt_ax, chrono_ax = upper_axes
            for abl in ABLS:
                p = psy_points.loc[psy_points.group_id.eq(group_id) & psy_points.plot_ABL.eq(abl)].sort_values("ILD")
                estimate = psy_fits.loc[(group_id, abl)]
                psy_ax.scatter(p.ILD, p.p_right, color=COLORS[abl], marker=ABL_MARKERS[abl], s=26,
                               label=f"ABL {abl} | n={int(p.trials.sum()):,}", zorder=3)
                smooth = np.linspace(-18, 18, 600)
                if estimate.fit_status == "ok":
                    psy_ax.plot(smooth, estimate.lower + (estimate.upper - estimate.lower) * expit(estimate.k * (smooth - estimate.x0)),
                            color=COLORS[abl], linewidth=1.7)
                density, edges, n = rtd_curves[(group_id, abl)]
                rt_ax.stairs(density, edges, color=COLORS[abl], linewidth=1.6, fill=False, label=f"ABL {abl} | n={n:,}")
                c = chrono.loc[chrono.group_id.eq(group_id) & chrono.plot_ABL.eq(abl)].sort_values("abs_ILD")
                chrono_ax.errorbar(c.plot_x, c.mean_rt_s, yerr=c.sem_rt_s, fmt="o", linestyle="None", color=COLORS[abl],
                                  markersize=4.5, elinewidth=.9, capsize=2, capthick=.9, zorder=3)
            psy_ax.set(title=f"{label} | {member.animals} animals\n"
                             f"Psychometric | {int(member.choice_trials):,} valid choices",
                       xlabel="Signed ILD (dB)", ylabel="P(right)", xlim=(-18, 18), ylim=(0, 1))
            psy_ax.set_xticks([-16, -8, -4, 0, 4, 8, 16])
            rt_ax.set(title=f"{label} | RTDs, all ILDs\n{int(member.finite_rt_trials):,} finite RTs; {int(member.nonfinite_rt_trials):,} missing",
                      xlabel="Measured timed_rt (s)", ylabel="Density (s⁻¹)", xlim=(0, .6))
            rt_ax.set_xticks(np.arange(0, .601, .1))
            visible_max = max(d[(e[:-1] < .6) & (e[1:] > 0)].max() for d, e, _ in [rtd_curves[(group_id, a)] for a in ABLS])
            rt_ax.set_ylim(0, visible_max * 1.12)
            chrono_ax.set(title=f"{GENOTYPE_LABELS[member.genotype]} | Chronometric\nPooled ±ILD", xlabel="|ILD| (dB)", ylabel="Mean RT ± SEM (s)",
                          xlim=(-.45, 4.45), ylim=(0, .45))
            chrono_ax.set_xticks(np.arange(5), [str(i) for i in ABS_ILDS])
            chrono_ax.set_yticks(np.arange(0, .451, .1))
            psy_ax.legend(loc="upper left", frameon=False, fontsize=7)
            rt_ax.legend(loc="upper right", frameon=False, fontsize=7)
            new_axes = [fig.add_axes([x/PAGE_WIDTH, (block_top - 6)/height, NEW_PANEL_SIZE/PAGE_WIDTH, NEW_PANEL_SIZE/height]) for x in lower_x]
            cdf_ax, delta_ax, *q_axes = new_axes
            for ild in ABS_ILDS:
                for abl in ABLS:
                    cdf_ax.plot(bin_centers, cdfs[(group_id, abl, ild)], color=COLORS[abl], linestyle=ILD_STYLES[ild], linewidth=1)
                delta_ax.plot(bin_centers, delta_curves[(group_id, ild)], color=ILD_COLORS[ild], linewidth=1.2)
                t = threshold_lookup.loc[(group_id, ild), "cutoff_s"]
                if np.isfinite(t):
                    delta_ax.axvline(t, color=ILD_COLORS[ild], linestyle="--", linewidth=.9)
            cdf_ax.set(title="CDFs | all five |ILD|\n1 ms bins, normalized over [0,1] s", xlabel="RT (s)", ylabel="CDF", xlim=CDF_DISPLAY_XLIM, ylim=(0, 1))
            cutoff_labels = [f"{i}: {threshold_lookup.loc[(group_id,i), 'cutoff_ms']:.1f}" if np.isfinite(threshold_lookup.loc[(group_id,i), 'cutoff_s']) else f"{i}: N/A" for i in ABS_ILDS]
            delta_ax.set(title="ΔCDF | cutoffs (ms)\n" + "; ".join(cutoff_labels[:3]) + "\n" + "; ".join(cutoff_labels[3:]),
                         xlabel="RT (s)", ylabel="Mean |ΔCDF|", xlim=CDF_DISPLAY_XLIM)
            delta_ax.title.set_fontsize(7.8)
            for ax in [cdf_ax, delta_ax]:
                ax.set_xticks([0, .075, .15], ["0", ".075", ".15"])
            delta_visible = max(float(delta_curves[(group_id,i)][bin_centers <= .15].max()) for i in ABS_ILDS)
            delta_ax.set_ylim(0, max(.025, delta_visible * 1.12))
            group_id_scaled = scaled.loc[scaled.group_id.eq(group_id)]
            y_max = float(np.nanmax(group_id_scaled[["raw_rt_s", "scaled_rt_s"]]))
            q_ymax = max(.1, float(np.ceil(y_max * 1.05 / .05) * .05))
            for j, abl in enumerate(ABLS):
                ax = q_axes[j]
                abl_n = int(counts.loc[counts.group_id.eq(group_id) & counts.plot_ABL.eq(abl), "window_trials"].sum())
                for qi, q in enumerate(PLOT_QUANTILES):
                    values = [quantiles[(group_id, abl, i)]["coarse"][qi] for i in ABS_ILDS]
                    ax.plot(ABS_ILDS, values, color=QUANTILE_COLORS[qi], marker=QUANTILE_MARKERS[qi], markersize=3, linewidth=1)
                ax.set_title(f"RT quantiles | ABL {abl}\nn={abl_n:,} RTs")
                ax.set_ylabel("RT quantile (s)")
            overlay_ax = q_axes[3]
            for abl in ABLS:
                for qi, q in enumerate(PLOT_QUANTILES):
                    curve = group_id_scaled.loc[group_id_scaled.plot_ABL.eq(abl) & group_id_scaled['quantile'].eq(q)].sort_values("abs_ILD")
                    overlay_ax.plot(curve.abs_ILD, curve.scaled_rt_s, color=COLORS[abl], marker=QUANTILE_MARKERS[qi], markersize=3, linewidth=1)
            overlay_ax.set(title="Rescaled quantiles\nABL 60 is the reference", ylabel="Scaled RT quantile (s)")
            for ax in q_axes:
                ax.set(xscale="log", xlabel="|ILD| (dB)", xlim=(.8, 20), ylim=(0, q_ymax))
                ax.set_xticks(ABS_ILDS, [str(i) for i in ABS_ILDS])
                ax.xaxis.set_minor_locator(plt.NullLocator())
            for ax in upper_axes + new_axes:
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(alpha=.16)
                ax.tick_params(labelsize=8)
            new_n = int(counts.loc[counts.group_id.eq(group_id), "window_trials"].sum())
            fig.text(.043, (block_top - 6.54)/height,
                f"{label}: lower row uses {new_n:,} RTs in [0,1] s | cutoff = first 10-bin forward-minus-backward ΔCDF rise ≥ 0.005 | scaling fitted up to 0.3 s",
                fontsize=8)
            for kind, axes in [("original", upper_axes), ("new", new_axes)]:
                for col, ax in enumerate(axes):
                    b = ax.get_position()
                    layout_records.append({"group_id": group_id, "pdf_page": int(page_number), "group_id_row": row + 1,
                        "row_kind": kind, "column": col + 1, "x_in": b.x0*PAGE_WIDTH, "y_in": b.y0*height,
                        "width_in": b.width*PAGE_WIDTH, "height_in": b.height*height,
                        "xlim_low": ax.get_xlim()[0], "xlim_high": ax.get_xlim()[1],
                        "ylim_low": ax.get_ylim()[0], "ylim_high": ax.get_ylim()[1]})
            for p in psy_points.loc[psy_points.group_id.eq(group_id)].itertuples(index=False):
                px,py=pdf_coordinates(psy_ax,p.ILD,p.p_right,height)
                psy_drawn.append({"group_id":group_id,"plot_ABL":p.plot_ABL,"ILD":p.ILD,"p_right":p.p_right,
                    "pdf_page":int(page_number),"color":to_hex(COLORS[p.plot_ABL]),"pdf_x_pt":px,"pdf_y_pt":py})
            for c in chrono.loc[chrono.group_id.eq(group_id)].itertuples(index=False):
                px,py=pdf_coordinates(chrono_ax,c.plot_x,c.mean_rt_s,height)
                _,top=pdf_coordinates(chrono_ax,c.plot_x,c.mean_plus_sem_s,height)
                _,bottom=pdf_coordinates(chrono_ax,c.plot_x,c.mean_minus_sem_s,height)
                chrono_drawn.append({"group_id":group_id,"plot_ABL":c.plot_ABL,"abs_ILD":c.abs_ILD,
                    "pdf_page":int(page_number),"color":to_hex(COLORS[c.plot_ABL]),"pdf_x_pt":px,"pdf_y_pt":py,
                    "pdf_sem_top_pt":top,"pdf_sem_bottom_pt":bottom})
            # Physical coordinates allow the verifier to find every cutoff line in the PDF.
            b = delta_ax.get_position()
            for ild in ABS_ILDS:
                t = threshold_lookup.loc[(group_id, ild), "cutoff_s"]
                plotted_records.append({"group_id": group_id, "abs_ILD": ild, "pdf_page": int(page_number),
                    "cutoff_s": t, "color": ILD_COLORS[ild],
                    "pdf_x_pt": (b.x0 + t/.15*b.width)*PAGE_WIDTH*72 if np.isfinite(t) else np.nan,
                    "pdf_y_top_pt": (1-b.y1)*height*72, "pdf_y_bottom_pt": (1-b.y0)*height*72})
        fig.text(.5, .26/height, "Session 1, training 16, success in {1,−1}; prepared repeat rule already applied | Upper RTD: 20 ms bins, all finite RTs, 0–0.6 s display | Lower CDF/quantiles: [0,1] s",
                 ha="center", fontsize=8)
        pdf.savefig(fig)
        png = OUTPUT / f"ASD_family_genotype_cdf_quantile_scaling_page_{int(page_number):02d}.png"
        fig.savefig(png, dpi=PNG_DPI, facecolor="white")
        plt.close(fig)
        page_records.append({"page": int(page_number), "group_ids": batch.group_id.to_list(), "rows": len(batch),
            "page_width_in": PAGE_WIDTH, "page_height_in": height, "png": str(png.relative_to(REPO))})
        print(f"Page {page_number}/2: {', '.join(batch.group_id)}", flush=True)
pd.DataFrame(layout_records).to_csv(OUTPUT / "panel_layout.csv", index=False)
pd.DataFrame(plotted_records).to_csv(OUTPUT / "plotted_thresholds.csv", index=False)


pd.DataFrame(psy_drawn).to_csv(OUTPUT/"plotted_psychometric_points.csv",index=False)
pd.DataFrame(chrono_drawn).to_csv(OUTPUT/"plotted_chronometric_points.csv",index=False)

# %% Six genotype bookmarks and a provenance/validation manifest
with fitz.open(pdf_path) as document:
    document.set_toc([[1,f"{identity[g]['family']} | {GENOTYPE_LABELS[identity[g]['genotype']]}",1+i//3] for i,g in enumerate(order)])
    temporary=pdf_path.with_suffix(".bookmarks.pdf")
    document.save(temporary,garbage=4,deflate=True)
temporary.replace(pdf_path)
for relative,digest in protected_hashes.items():
    with (REPO/relative).open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest()==digest,relative
manifest={"status":"passed","source_script":str(Path(__file__).relative_to(REPO)),
    "input":str(INPUT.relative_to(REPO)),"input_sha256":input_sha256,"observed":observed,
    "group_order":order,"groups":members.to_dict("records"),"grouping":"family/genotype; cohorts combined within family",
    "pooling":"Raw observed trials pooled across animals and sessions within group/ABL/condition; no averaging of animal curves or parameters",
    "selection":"Prepared session1/training16; success in {1,-1}; ABL59->60 then20/40/60; ILDs±1/2/4/8/16; no repeat refilter; cohort4 session>=13",
    "repeat_filter_reapplied":False,"psychometric_RT_restriction":None,"psychometric_fits":18,"psychometric_points":180,
    "fit_procedure":"Unweighted curve_fit, upper/lower/x0/k, [maxP,minP,medianILD,.1], bounds [0,0,-inf,0]/[1,1,inf,inf], maxfev10000",
    "psychometric_fit_status_counts":psy_fits.fit_status.value_counts().to_dict(),
    "RTD_curves":18,"RTD_bin_width_s":RTD_BIN_WIDTH,"RTD_edges_range_s":[float(rtd_edges[0]),float(rtd_edges[-1])],
    "RTD_normalization":"All finite RTs per group/ABL, including >1 second; bin count/(n*.020)","RTD_display_xlim_s":[0,.6],
    "chronometric_points":90,"chronometric_sem":"Sample SD (ddof1)/sqrt(pooled finite-trial n); not between-animal uncertainty",
    "chronometric_ylim_s":list(CHRONOMETRIC_YLIM),"chronometric_offsets":CHRONOMETRIC_OFFSETS,
    "cdf_conditions":90,"cdf_bin_width_s":CDF_BIN_WIDTH,"new_rt_window_s":list(NEW_RT_WINDOW),"cdf_display_xlim_s":list(CDF_DISPLAY_XLIM),
    "threshold_conditions":30,"available_thresholds":int(thresholds.threshold_status.eq("ok").sum()),
    "cutoff_range_ms":[float(thresholds.cutoff_ms.min()),float(thresholds.cutoff_ms.max())],
    "onset_window_bins":10,"onset_rise":.005,"onset_rule":"First i with mean(D[i+1:i+11])-mean(D[i-10:i])>=.005; bin center",
    "delta_cdf":"(|F20-F40|+|F40-F60|+|F20-F60|)/3","quantile_method":"linear",
    "display_quantiles":PLOT_QUANTILES.tolist(),"fine_quantiles":FIT_QUANTILES.tolist(),"raw_quantile_points":450,"scaled_quantile_points":450,
    "scaling_fits":60,"valid_scaling_fits":int(scales.fit_status.eq("ok").sum()),"fit_max_rt_s":FIT_MAX_RT,
    "scale_factor_range":[float(scales.scale_factor.min()),float(scales.scale_factor.max())],
    "scaling_rule":"x=Q60-t0; y=Qabl-Q60; m=sum(x*(y-yfirst))/sum(x*x); factor=1+m; Q>t0 -> t0+(Q-t0)/factor; ABL60 unchanged",
    "unavailable_policy":"Missing cutoff or invalid/nonpositive factor: omit affected scaled values, keep raw data; failed psychometric: keep scatter, omit curve",
    "ABL_colors":{str(a):to_hex(c) for a,c in COLORS.items()},"ILD_colors":{str(i):c for i,c in ILD_COLORS.items()},
    "quantile_colors":QUANTILE_COLORS,"quantile_markers":QUANTILE_MARKERS,"pdf_pages":2,"page_width_in":PAGE_WIDTH,"png_dpi":PNG_DPI,
    "pages":page_records,"pdf":str(pdf_path.relative_to(REPO)),"protected_file_sha256":protected_hashes}
(OUTPUT/"validation.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("Saved two-page family/genotype PDF, two PNGs and complete calculation/plot audits",flush=True)
