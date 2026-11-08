"""Two-by-five pooled Control/ASD RTDs, one column per absolute difference.

Run .venv/bin/python ASD_Human_Data/plot_simultaneous_group_rtd_by_abs_delta.py.
The default condition is SD; --condition RT selects the separate RT export.
"""

# %% Imports and editable parameters
import argparse
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--condition", choices=["SD", "RT"], default="SD")
CONDITION = parser.parse_args().condition
INPUT = HERE / f"simultaneous_{CONDITION.lower()}_trials.csv"
EXTRACTION = HERE / f"simultaneous_{CONDITION.lower()}_extraction.json"
CDF_REFERENCE = HERE / f"simultaneous_{CONDITION.lower()}_group_pooled_distributions/cdf_bins.csv.gz"
OUTPUT = HERE / f"simultaneous_{CONDITION.lower()}_rtd_by_abs_delta"
GROUPS = ["Control", "ASD"]
ASLS = [20, 40]
ABS_DELTAS = [.1, .3, .65, 1.3, 2]
COLORS = {20: "tab:blue", 40: "tab:orange"}
GROUP_COLORS = {"Control": "tab:blue", "ASD": "tab:red"}
BIN_WIDTH_S = .020
DISPLAY_XLIM = (0, 1.5)
FIGSIZE = (22, 8)
PNG_DPI = 180
STEM = f"human_simultaneous_{CONDITION.lower()}_control_asd_rtd_by_abs_delta_2x5"
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Preserve the selected condition, both choices, and all recorded RTs
extraction = json.loads(EXTRACTION.read_text())
input_sha = hashlib.sha256(INPUT.read_bytes()).hexdigest()
assert input_sha == extraction["output_sha256"]
assert extraction["selection"] == f"Condition == '{CONDITION}'"
trials = pd.read_csv(INPUT, float_precision="round_trip")
assert trials.Condition.eq(CONDITION).all() and trials.Valid.eq(1).all()
assert trials.Choice.isin([-1, 1]).all() and np.isfinite(trials.RT_s).all()
assert trials.RT_s.ge(0).all()
if CONDITION == "SD":
    assert len(trials) == 6080 and trials.Stim_duration_s.eq(.03).all()
else:
    assert len(trials) == 11400 and trials.Stim_duration_s.isna().all()
trials["abs_delta"] = trials.Delta_dB.abs().round(2)
assert sorted(trials.abs_delta.unique()) == ABS_DELTAS
assert set(trials.ASL_cd) == set(ASLS)
summary = trials.groupby(["GroupID", "Group"], as_index=False).agg(
    participants=("Participant", "nunique"), trials=("RT_s", "size"))
assert summary.Group.tolist() == GROUPS and summary.participants.tolist() == [11, 8]


# %% Full-range, area-one RT densities; verify counts against saved 1 ms audits
last_edge = float(np.ceil(trials.RT_s.max() / BIN_WIDTH_S) * BIN_WIDTH_S)
edges = np.arange(round(last_edge / BIN_WIDTH_S) + 1) * BIN_WIDTH_S
cdf_reference = pd.read_csv(CDF_REFERENCE)
curves, records, count_rows = {}, [], []
for (group, asl, abs_delta), condition in trials.groupby(["Group", "ASL_cd", "abs_delta"]):
    values = condition.RT_s.to_numpy()
    counts = np.histogram(values, edges)[0]
    density = counts / (len(values) * BIN_WIDTH_S)
    reference = cdf_reference.loc[cdf_reference.Group.eq(group) & cdf_reference.ASL_cd.eq(asl) &
                                  cdf_reference.abs_delta.eq(abs_delta)].sort_values("bin_left_s")
    assert len(reference) == len(counts) * 20
    np.testing.assert_array_equal(counts, reference["count"].to_numpy().reshape(-1, 20).sum(axis=1))
    assert counts.sum() == len(values)
    np.testing.assert_allclose((density * BIN_WIDTH_S).sum(), 1, atol=1e-12, rtol=0)
    expected_n = (352 if group == "Control" else 256) if CONDITION == "SD" else (660 if group == "Control" else 480)
    assert len(values) == expected_n
    positive = int(condition.Delta_dB.gt(0).sum())
    negative = int(condition.Delta_dB.lt(0).sum())
    assert positive == negative == expected_n // 2
    curves[(group, asl, abs_delta)] = (density, len(values))
    count_rows.append({"Group": group, "ASL_cd": asl, "abs_delta": abs_delta,
                       "participants": condition.Participant.nunique(), "trials": len(values),
                       "positive_delta_trials": positive, "negative_delta_trials": negative})
    records.append(pd.DataFrame({"Group": group, "ASL_cd": asl, "abs_delta": abs_delta,
                                 "bin_left_s": edges[:-1], "bin_right_s": edges[1:],
                                 "count": counts, "density_per_s": density,
                                 "normalization_trials": len(values)}))
histograms = pd.concat(records, ignore_index=True)
assert len(curves) == 20 and histograms["count"].sum() == len(trials)
histograms.to_csv(OUTPUT / "rtd_bins.csv", index=False)
pd.DataFrame(count_rows).to_csv(OUTPUT / "condition_counts.csv", index=False)


# %% Control above ASD; common axes for all ten panels
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
fig, axes = plt.subplots(2, 5, figsize=FIGSIZE, sharex=True, sharey=True)
condition_label = "SD — 30 ms stimulus presentation" if CONDITION == "SD" else "RT condition"
fig.suptitle(f"Human {condition_label} | RTDs by absolute ΔdB", fontsize=19, y=.98)
fig.legend(handles=[Line2D([], [], color=COLORS[a], lw=2, label=f"ASL {a} cd/m²") for a in ASLS],
           loc="upper center", bbox_to_anchor=(.5, .935), ncol=2, frameon=False, fontsize=12)
visible = (edges[:-1] < DISPLAY_XLIM[1]) & (edges[1:] > DISPLAY_XLIM[0])
ymax = max(float(density[visible].max()) for density, _ in curves.values()) * 1.16
for row, group in enumerate(GROUPS):
    for col, abs_delta in enumerate(ABS_DELTAS):
        ax = axes[row, col]
        for asl in ASLS:
            density, n = curves[(group, asl, abs_delta)]
            ax.stairs(density, edges, color=COLORS[asl], lw=1.7)
        ax.set(title=f"|ΔdB| = {abs_delta:g}", xlim=DISPLAY_XLIM, ylim=(0, ymax), xticks=[0, .5, 1, 1.5])
        ax.text(.97, .94, f"n = {n:,} per ASL", transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        if row == 1:
            ax.set_xlabel("Stored RT_s (s)")
        if col == 0:
            ax.set_ylabel(f"{group}\nDensity (s⁻¹)", color=GROUP_COLORS[group], fontsize=13)
fig.text(.5, .05, "Pool raw +ΔdB and −ΔdB trials across participants and sessions within each group, ASL, and |ΔdB|", ha="center")
fig.text(.5, .021, "Both correct and incorrect choices | All recorded RTs | 20 ms bins | Full-range area = 1 per curve | 0–1.5 s display zoom", ha="center")
fig.subplots_adjust(left=.055, right=.99, top=.84, bottom=.14, hspace=.39, wspace=.16)
fig.savefig(OUTPUT / f"{STEM}.png", dpi=PNG_DPI, facecolor="white")
fig.savefig(OUTPUT / f"{STEM}.pdf", facecolor="white")
plt.close(fig)
with fitz.open(OUTPUT / f"{STEM}.pdf") as pdf:
    assert len(pdf) == 1
    text = pdf[0].get_text()
    assert "Control" in text and "ASD" in text and text.count("per ASL") == 10
    for block in pdf[0].get_text("blocks"):
        x0, y0, x1, y1 = block[:4]
        assert x0 >= 0 and y0 >= 0 and x1 <= pdf[0].rect.width and y1 <= pdf[0].rect.height
manifest = {"status": "passed", "condition": CONDITION, "source_csv": INPUT.name,
            "source_sha256": input_sha, "source_mat_sha256": extraction["source_sha256"],
            "groups": summary.to_dict("records"), "pooling": "raw signed trials within Group/ASL_cd/abs(Delta_dB)",
            "bin_width_s": BIN_WIDTH_S, "histogram_range_s": [0, last_edge], "display_xlim_s": list(DISPLAY_XLIM),
            "curves": 20, "panels": 10, "normalization": "count / (all finite RTs in this condition * 0.020)",
            "reference_cdf_sha256": hashlib.sha256(CDF_REFERENCE.read_bytes()).hexdigest(),
            "verification": "all 20 full-range histograms match independently saved pooled 1 ms bin counts; all areas equal one",
            "png": f"{STEM}.png", "pdf": f"{STEM}.pdf"}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
