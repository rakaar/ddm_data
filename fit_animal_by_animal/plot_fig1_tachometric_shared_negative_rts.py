# %% Imports
"""Figure 1: one shared negative-RT accuracy curve; four accuracy bin sizes.

The condition-specific CDFs retain their original 20 ms values. Abort choices
are copied from the verified seed-42 analysis, never redrawn per figure.
Run with .venv/bin/python from the repository root.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
LEGACY = SCRIPT_PATH.parent / "fig1_tachometric_with_negative_rts"
POSITIVE_REFERENCE = SCRIPT_PATH.parent / "fig1_tachometric_by_abl_abs_ild"
OUTPUT = SCRIPT_PATH.parent / "fig1_tachometric_shared_negative_rts"
COHORT_MANIFEST = REPO / "all_30_fit_consolidation_manifest.md"
BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
COLORS = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:red", 16: "tab:purple"}
BIN_WIDTHS_MS = [10, 15, 20, 25]
DISPLAY_LIMITS_MS = [(-1000, 600), (-100, 600)]
RT_WINDOW_S = (-1., 1.)
RANDOM_SEED = 42
PNG_DPI = 300
FIGSIZE = (13, 12.5)
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Keep zero exact and preserve the original 20 ms floating-point boundaries
def make_bin_edges(width_ms, original_positive_edges):
    if width_ms == 20:
        positive_edges = np.asarray(original_positive_edges, dtype=float)
    else:
        positive_edges = np.r_[np.arange(0, 1000, width_ms, dtype=int), 1000] / 1000
    return np.r_[-positive_edges[:0:-1], positive_edges]


# %% Protect the source CSVs and every previous Figure 1 export
legacy_summary = json.loads((LEGACY / "run_summary.json").read_text())
legacy_verified = json.loads((LEGACY / "verification.json").read_text())
original = json.loads((POSITIVE_REFERENCE / "run_summary.json").read_text())
assert legacy_verified["status"] == "passed"
protected = {COHORT_MANIFEST, REPO / "fit_animal_by_animal/plot_fig1_cdf_tachometric_with_negative_rts.py"}
for folder in [LEGACY, POSITIVE_REFERENCE]:
    protected.update(p for p in folder.iterdir() if p.is_file())
protected.update(Path(record["path"]) for record in legacy_summary["sources"])
protected_hashes = {}
for path in sorted(protected):
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
for record in legacy_summary["sources"]:
    assert protected_hashes[str(Path(record["path"]).relative_to(REPO))] == record["sha256"]
cohort = sorted((batch, int(animal)) for batch, animal in re.findall(
    r"^- `([^/`]+)/([0-9]+)`: source=", COHORT_MANIFEST.read_text(), flags=re.MULTILINE))
assert len(cohort) == 30


# %% Select the identical trials and reuse each existing simulated abort choice
frames = []
for batch in BATCHES:
    source = REPO / "raw_data/batch_csvs" / f"batch_{batch}_valid_and_aborts.csv"
    rows = pd.read_csv(source, usecols=["batch_name", "animal", "success", "abort_event", "RTwrtStim", "ABL", "ILD"])
    assert set(rows.batch_name) == {batch}
    rows["source_csv"] = str(source)
    rows["source_row_index"] = np.arange(len(rows))
    frames.append(rows)
data = pd.concat(frames, ignore_index=True)
assert set(data[["batch_name", "animal"]].itertuples(index=False, name=None)) == set(cohort)
selected = data.loc[data.RTwrtStim.between(*RT_WINDOW_S) & data.ABL.isin(ABLS) & data.ILD.abs().isin(ABS_ILDS)
    & (data.success.isin([1, -1]) | data.abort_event.eq(3))].copy()
selected["abs_ILD"] = selected.ILD.abs().astype(int)
selected["choice_is_simulated"] = ~selected.success.isin([1, -1])
selected["correct"] = selected.success.eq(1).astype(int)
abort_mask = selected.choice_is_simulated
aborts = pd.read_csv(LEGACY / "abort_random_choices.csv", float_precision="round_trip")
pd.testing.assert_frame_equal(selected.loc[abort_mask, ["source_csv", "source_row_index", "batch_name", "animal"]].reset_index(drop=True),
    aborts[["source_csv", "source_row_index", "batch_name", "animal"]], check_dtype=False)
np.testing.assert_array_equal(aborts.simulated_choice, np.random.default_rng(RANDOM_SEED).choice([-1, 1], size=len(aborts), p=[.5, .5]))
np.testing.assert_array_equal(aborts.simulated_correct, (selected.loc[abort_mask, "ILD"].to_numpy()*aborts.simulated_choice.to_numpy() > 0).astype(int))
selected.loc[abort_mask, "correct"] = aborts.simulated_correct.to_numpy()
selected["csv_line_number"] = selected.source_row_index+2
assert len(data) == 432006 and len(selected) == 419558
assert int(abort_mask.sum()) == 71235 and int((~abort_mask).sum()) == 348323
assert np.array_equal(selected.RTwrtStim.lt(0).to_numpy(), abort_mask.to_numpy())
assert selected.loc[abort_mask, "correct"].sum() == 35520 and selected.loc[~abort_mask, "correct"].sum() == 280680
selected.to_csv(OUTPUT / "selected_trials.csv.gz", index=False, compression="gzip")
shutil.copy2(LEGACY / "abort_random_choices.csv", OUTPUT / "abort_random_choices.csv")
for name in ["animal_cumulative_rtd.csv", "group_cumulative_rtd.csv"]:
    shutil.copy2(LEGACY / name, OUTPUT / name)
cdfs = pd.read_csv(LEGACY / "group_cumulative_rtd.csv", float_precision="round_trip")
print("Identical 30-animal cohort: 419,558 trials; 71,235 seed-42 abort choices reused; CDFs fixed at 20 ms", flush=True)


# %% Rebin accuracy: pooled conditions within animal before the negative-time mean
bin_tables, negative_animal_tables, negative_tables, positive_animal_tables, positive_tables = [], [], [], [], []
for width in BIN_WIDTHS_MS:
    edges = make_bin_edges(width, original["binning"]["edges_s"])
    n_bins = len(edges)-1
    assert n_bins == {10: 200, 15: 134, 20: 100, 25: 80}[width]
    assert edges[0] == -1 and edges[-1] == 1 and (edges == 0).sum() == 1
    indices = np.minimum(np.searchsorted(edges, selected.RTwrtStim.to_numpy(), side="right")-1, n_bins-1)
    trials = selected.assign(bin_index=indices)
    bins = pd.DataFrame({"width_ms": width, "bin_index": np.arange(n_bins), "bin_start_s": edges[:-1],
        "bin_end_s": edges[1:], "bin_center_s": (edges[:-1]+edges[1:])/2, "actual_width_ms": np.diff(edges)*1000})
    bins["segment"] = np.where(bins.bin_end_s.le(0), "shared_negative", "positive_condition")
    bin_tables.append(bins)
    negative_indices = bins.loc[bins.segment.eq("shared_negative"), "bin_index"].tolist()
    positive_indices = bins.loc[bins.segment.eq("positive_condition"), "bin_index"].tolist()
    negative = trials.loc[trials.RTwrtStim.lt(0)].groupby(["batch_name", "animal", "bin_index"]).agg(
        n_trials=("correct", "size"), n_correct=("correct", "sum"))
    negative_index = pd.MultiIndex.from_tuples([(b, a, i) for b, a in cohort for i in negative_indices],
        names=["batch_name", "animal", "bin_index"])
    negative = negative.reindex(negative_index, fill_value=0).reset_index()
    negative["accuracy"] = negative.n_correct/negative.n_trials.replace(0, np.nan)
    shared = negative.groupby("bin_index", as_index=False).agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
        n_animals=("accuracy", "count"), n_trials=("n_trials", "sum"), n_correct=("n_correct", "sum"))
    shared["sem_accuracy"] = shared.std_accuracy/np.sqrt(shared.n_animals)
    positive = trials.loc[trials.RTwrtStim.ge(0)].groupby(["batch_name", "animal", "ABL", "abs_ILD", "bin_index"]).agg(
        n_trials=("correct", "size"), n_correct=("correct", "sum"))
    positive_index = pd.MultiIndex.from_tuples([(b, a, abl, ild, i) for b, a in cohort for abl in ABLS for ild in ABS_ILDS for i in positive_indices],
        names=["batch_name", "animal", "ABL", "abs_ILD", "bin_index"])
    positive = positive.reindex(positive_index, fill_value=0).reset_index()
    positive["accuracy"] = positive.n_correct/positive.n_trials.replace(0, np.nan)
    separate = positive.groupby(["ABL", "abs_ILD", "bin_index"], as_index=False).agg(mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"), n_animals=("accuracy", "count"), n_trials=("n_trials", "sum"), n_correct=("n_correct", "sum"))
    separate["sem_accuracy"] = separate.std_accuracy/np.sqrt(separate.n_animals)
    for table in [negative, shared, positive, separate]:
        table.insert(0, "width_ms", width)
        table["bin_start_s"] = edges[table.bin_index.to_numpy()]
        table["bin_end_s"] = edges[table.bin_index.to_numpy()+1]
        table["bin_center_s"] = (table.bin_start_s+table.bin_end_s)/2
    assert negative.n_trials.sum() == shared.n_trials.sum() == 71235
    assert positive.n_trials.sum() == separate.n_trials.sum() == 348323
    assert shared.loc[shared.n_animals.lt(2), "sem_accuracy"].isna().all()
    assert separate.loc[separate.n_animals.lt(2), "sem_accuracy"].isna().all()
    negative_animal_tables.append(negative)
    negative_tables.append(shared)
    positive_animal_tables.append(positive)
    positive_tables.append(separate)
bins = pd.concat(bin_tables, ignore_index=True)
negative_animals = pd.concat(negative_animal_tables, ignore_index=True)
negative_curves = pd.concat(negative_tables, ignore_index=True)
positive_animals = pd.concat(positive_animal_tables, ignore_index=True)
positive_curves = pd.concat(positive_tables, ignore_index=True)
bins.to_csv(OUTPUT / "bin_definitions.csv", index=False)
negative_animals.to_csv(OUTPUT / "negative_animal_bin_accuracy.csv.gz", index=False, compression="gzip")
negative_curves.to_csv(OUTPUT / "shared_negative_accuracy.csv", index=False)
positive_animals.to_csv(OUTPUT / "positive_animal_bin_accuracy.csv.gz", index=False, compression="gzip")
positive_curves.to_csv(OUTPUT / "positive_condition_accuracy.csv", index=False)
old_positive = pd.read_csv(LEGACY / "group_bin_summary.csv", float_precision="round_trip")
old_positive = old_positive.loc[old_positive.bin_start_s.ge(0)]
columns = ["ABL", "abs_ILD", "bin_index", "n_animals", "n_trials", "n_correct", "mean_accuracy", "std_accuracy", "sem_accuracy"]
pd.testing.assert_frame_equal(positive_curves.loc[positive_curves.width_ms.eq(20), columns].reset_index(drop=True),
    old_positive[columns].reset_index(drop=True), check_exact=False, atol=1e-14, rtol=0)


# %% Eight PNGs; snapshot the actual line and shading artists for independent checks
figure_records, artist_records, artist_arrays = [], [], {}
plt.rcParams.update({"font.family": "DejaVu Sans"})
for width in BIN_WIDTHS_MS:
    for xmin, xmax in DISPLAY_LIMITS_MS:
        name = f"fig1_shared_negative_tachometric_{width}ms_xlim_minus{abs(xmin)}_to_{xmax}ms"
        fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=.075, right=.975, bottom=.20, top=.84, hspace=.29, wspace=.22)
        for row, abl in enumerate(ABLS):
            cdf_ax, accuracy_ax = axes[row]
            shared = negative_curves.loc[negative_curves.width_ms.eq(width)].sort_values("bin_index")
            x, mean, sem = [shared[column].to_numpy() for column in ["bin_center_s", "mean_accuracy", "sem_accuracy"]]
            band = accuracy_ax.fill_between(x*1000, mean-sem, mean+sem, color=".4", alpha=.22, linewidth=0)
            line, = accuracy_ax.plot(x*1000, mean, color="black", linewidth=1.8, marker="o", markersize=2)
            artists = [("shared_negative", None, line, band)]
            for ild in ABS_ILDS:
                cdf = cdfs.loc[cdfs.ABL.eq(abl) & cdfs.abs_ILD.eq(ild)].sort_values("bin_index")
                curve = positive_curves.loc[positive_curves.width_ms.eq(width) & positive_curves.ABL.eq(abl) & positive_curves.abs_ILD.eq(ild)].sort_values("bin_index")
                for kind, ax, table, xcol, ycol, semcol, marker in [
                    ("cdf", cdf_ax, cdf, "time_s", "mean_cumulative_fraction", "sem_cumulative_fraction", None),
                    ("positive", accuracy_ax, curve, "bin_center_s", "mean_accuracy", "sem_accuracy", "o")]:
                    x, mean, sem = [table[column].to_numpy() for column in [xcol, ycol, semcol]]
                    band = ax.fill_between(x*1000, mean-sem, mean+sem, color=COLORS[ild], alpha=.16, linewidth=0)
                    line, = ax.plot(x*1000, mean, color=COLORS[ild], linewidth=1.7, marker=marker, markersize=2)
                    artists.append((kind, ild, line, band))
            for kind, ild, line, band in artists:
                key = f"artist_{len(artist_records)}"
                artist_arrays[key+"_x"] = line.get_xdata()
                artist_arrays[key+"_y"] = line.get_ydata()
                path_keys = []
                for i, path in enumerate(band.get_paths()):
                    path_keys.append(key+f"_band_{i}")
                    artist_arrays[path_keys[-1]] = path.vertices
                artist_records.append({"figure": name, "width_ms": width, "ABL": abl, "kind": kind, "abs_ILD": ild,
                    "array_key": key, "color": to_hex(line.get_color()), "line_style": line.get_linestyle(), "marker": line.get_marker(),
                    "shading_color": to_hex(band.get_facecolor()[0]), "shading_paths": path_keys,
                    "xlim_ms": [xmin, xmax], "ylim": [0, 1]})
            accuracy_ax.axhline(.5, color=".6", linestyle="--", linewidth=.8, zorder=0)
            for ax in [cdf_ax, accuracy_ax]:
                ax.axvline(0, color=".45", linestyle=":", linewidth=1, zorder=0)
                ax.set(xlim=(xmin, xmax), ylim=(0, 1))
                ax.set_xticks(np.arange(xmin, xmax+1, 200 if xmin == -1000 else 100))
                ax.set_yticks([0, .5, 1])
                ax.tick_params(labelsize=11, labelleft=True)
                ax.spines[["top", "right"]].set_visible(False)
            cdf_ax.set_title(f"ABL = {abl} dB: cumulative RTD", fontsize=13, pad=10)
            accuracy_ax.set_title(f"ABL = {abl} dB: accuracy", fontsize=13, pad=10)
            cdf_ax.set_ylabel("Cumulative fraction of trials", fontsize=12)
            accuracy_ax.set_ylabel("Accuracy", fontsize=13)
        fig.suptitle("Cumulative reaction times and accuracy at each ABL", fontsize=18, y=.985)
        tail_note = " (10 ms outermost bins at ±1 s)" if width == 15 else ""
        fig.text(.5, .915, f"30-animal Figure 1 cohort | Accuracy: {width} ms{tail_note} | CDF: fixed 20 ms\n"
            "Equal-animal mean ± SEM | RT < 0: all ABLs/ILDs pooled within animal; same black curve in every row\n"
            "Fixation-abort choices simulated 50/50; original seed-42 choices reused", ha="center", fontsize=10.5, linespacing=1.5)
        fig.supxlabel("Reaction time from stimulus onset (ms)", fontsize=14, y=.145)
        handles = [Line2D([], [], color=COLORS[i], label=f"{i} dB") for i in ABS_ILDS]
        handles.append(Line2D([], [], color="black", label="Shared RT < 0"))
        fig.legend(handles=handles, title="|ILD| (dB); shared negative-time accuracy", loc="lower center",
            bbox_to_anchor=(.5, .07), ncol=6, frameon=False, fontsize=10.5, title_fontsize=10.5)
        fig.text(.5, .018, "CDFs retain separate conditions and full −1 to 1 s normalization. Accuracy uses trials in each time bin.\n"
            "RT ≥ 0: |ILD| 1–8, up to 30 animals; 16, up to 24. Empty bins are gaps; singleton means have no SEM.",
            ha="center", fontsize=9.3, linespacing=1.5)
        fig.savefig(OUTPUT / f"{name}.png", dpi=PNG_DPI, facecolor="white")
        # A local PDF permits text-bound checks; only the requested PNGs are published.
        fig.savefig(OUTPUT / f"{name}.pdf")
        figure_records.append({"name": name, "png": str((OUTPUT / f"{name}.png").relative_to(REPO)),
            "local_pdf": str((OUTPUT / f"{name}.pdf").relative_to(REPO)), "width_ms": width,
            "display_xlim_ms": [xmin, xmax], "figsize_in": list(FIGSIZE), "rows": 3, "columns": 2})
        plt.close(fig)
        print(f"Saved {name}.png", flush=True)
np.savez_compressed(OUTPUT / "plotted_artist_arrays.npz", **artist_arrays)
(OUTPUT / "plotted_artists.json").write_text(json.dumps(artist_records, indent=2)+"\n")


# %% Preserve prior outputs and record the shared-versus-condition-specific definitions
for relative, digest in protected_hashes.items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, relative
report = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(), "source_script": str(SCRIPT_PATH.relative_to(REPO)),
    "script_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(), "sources": legacy_summary["sources"],
    "cohort": [{"batch_name": b, "animal": a} for b, a in cohort], "cohort_manifest": str(COHORT_MANIFEST.relative_to(REPO)),
    "selected_trials": 419558, "valid_choice_trials": 348323, "negative_rt_fixation_aborts": 71235,
    "observed_correct": 280680, "simulated_correct": 35520, "rt_window_s": list(RT_WINDOW_S),
    "selection": "success in {1,-1} OR abort_event==3; existing RTwrtStim in inclusive [-1,1] s; ABL20/40/60; |ILD|1/2/4/8/16; original six-batch preprocessing",
    "simulation": "Copy each existing seed-42 abort choice once; valid outcomes take precedence; ILD*simulated_choice>0 defines abort correctness",
    "negative_accuracy": "Pool all ABLs and signed ILDs within animal and negative-time bin; then equal-animal mean and sample SEM; same black curve in every panel",
    "positive_accuracy": "Pool signs within animal/ABL/|ILD|/positive-time bin, then equal-animal mean and sample SEM",
    "SEM": "sample SD about equal-animal mean / sqrt(nonempty animal count); ddof=1; missing bins stay NaN; singleton SEM stays NaN",
    "cdf": "Byte-identical copies of original 20 ms condition-specific animal/group CDFs; unchanged values and SEMs in every figure",
    "binning": {"widths_ms": BIN_WIDTHS_MS, "bin_counts": {"10": 200, "15": 134, "20": 100, "25": 80},
        "15ms_edges": "10 ms outermost bins at -1/+1 s; remaining bins 15 ms", "boundary_rule": "[left,right), final bin includes +1; zero belongs to positive time",
        "20ms_edges": "Original positive edges mirrored exactly, preserving original positive assignments"},
    "condition_colors": {str(i): to_hex(COLORS[i]) for i in ABS_ILDS}, "negative_color": "#000000", "negative_SEM_color": "#666666",
    "figures": figure_records, "data_artists_saved": len(artist_records), "protected_file_sha256": protected_hashes}
(OUTPUT / "run_summary.json").write_text(json.dumps(report, indent=2)+"\n")
