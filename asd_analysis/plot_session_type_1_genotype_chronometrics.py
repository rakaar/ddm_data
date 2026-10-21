# %% Imports
"""Compare rat genotypes by cohort/family, using pooled means and animal SEM.

Dots pool actual [0,1]-second valid RTs. Bars use the sample SD of the
animals' condition means divided by sqrt(animal count), independently of
trial-count weighting. Single-animal conditions retain a dot and no SEM.
Run with .venv/bin/python from the repository root.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
REFERENCE = REPO / "asd_analysis/session_type_1_genotype_overlaid_rtds"
OUTPUT = REPO / "asd_analysis/session_type_1_genotype_chronometrics"
COHORTS = ["SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv"]
FAMILIES = ["SHANK3", "CNTNAP2"]
GENOTYPES = ["wt", "hom", "het"]
LABELS = {"wt": "WT", "hom": "HOM", "het": "HET"}
COLORS = {"wt": "tab:blue", "hom": "tab:green", "het": "tab:orange"}
OFFSETS = {"wt": -.12, "hom": 0, "het": .12}
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
RT_WINDOW = (0., 1.)
Y_LIM = (0., .45)
PNG_DPI = 200
PANEL_SIZE_IN = 2.45
COLUMN_GAP_IN = .60
ROW_GAP_IN = .70
LEFT_IN = 1.80
RIGHT_IN = .50
HEADER_IN = 1.60
FOOTER_IN = 1.00
TITLE = "Pooled mean RT; error bars = SEM across animal means"
VIEWS = {
    "cohort": {"column": "filename", "groups": COHORTS, "title": "Cohort-level genotype comparison", "points": 180},
    "family": {"column": "family", "groups": FAMILIES, "title": "Gene-family genotype comparison | cohorts combined", "points": 90},
}
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Preserve current input and previous genotype/family exports
protected = set(INPUT.parent.glob("*.csv"))
for folder in [REFERENCE, REPO / "asd_analysis/session_type_1_family_genotype_psychometric_rtd_chronometric_scaling"]:
    protected.update(p for p in folder.iterdir() if p.is_file())
protected_hashes = {}
for path in sorted(protected):
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
previous = json.loads((REFERENCE / "validation.json").read_text())
assert previous["status"] == "passed" and previous["input_sha256"] == input_sha256


# %% Select valid choices and the explicitly requested inclusive RT window
data = pd.read_csv(INPUT, usecols=["filename", "animal", "genotype", "session", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "success", "ABL", "ILD", "timed_rt"], low_memory=False)
assert len(data) == 610603 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all() and data.is_short_sound.eq(0).all()
assert not (data.filename.eq("CNTNAP2_cohort4.csv") & data.session.lt(13)).any()
data["plot_ABL"] = data.ABL.replace({59: 60})
choices = data.loc[data.success.isin([1, -1]) & data.plot_ABL.isin(ABLS) & data.ILD.isin(SIGNED_ILDS)].copy()
selected = choices.loc[np.isfinite(choices.timed_rt) & choices.timed_rt.between(*RT_WINDOW, inclusive="both")].copy()
selected["family"] = selected.filename.str.extract(r"^(SHANK3|CNTNAP2)_cohort")[0]
selected["abs_ILD"] = selected.ILD.abs().astype(int)
selected["plot_ABL"] = selected.plot_ABL.astype(int)
selected["positive_ild"] = selected.ILD.gt(0)
assert len(choices) == 451381 and len(selected) == 446949 and selected.animal.nunique() == 53
assert selected.filename.isin(COHORTS).all() and selected.genotype.isin(GENOTYPES).all()


# %% Each animal's mean uses its actual pooled +ILD and -ILD trials across sessions
animal_keys = ["filename", "family", "animal", "genotype", "plot_ABL", "abs_ILD"]
animal_means = selected.groupby(animal_keys, as_index=False).agg(
    window_trials=("timed_rt", "size"), rt_sum_s=("timed_rt", "sum"),
    animal_mean_rt_s=("timed_rt", "mean"), positive_ild_trials=("positive_ild", "sum"))
animal_means["negative_ild_trials"] = animal_means.window_trials - animal_means.positive_ild_trials
assert len(animal_means) == 795 and animal_means.window_trials.sum() == 446949
animal_means.to_csv(OUTPUT / "animal_condition_means.csv", index=False)


# %% Pooled point centers and equally weighted animal-mean SEMs are distinct estimators
point_tables, member_tables, group_tables = [], [], []
for view, config in VIEWS.items():
    rows = selected.assign(group=selected[config["column"]])
    animals = animal_means.assign(group=animal_means[config["column"]])
    keys = ["group", "genotype", "plot_ABL", "abs_ILD"]
    pooled = rows.groupby(keys, as_index=False).agg(
        window_trials=("timed_rt", "size"), rt_sum_s=("timed_rt", "sum"),
        pooled_mean_rt_s=("timed_rt", "mean"), positive_ild_trials=("positive_ild", "sum"))
    pooled["negative_ild_trials"] = pooled.window_trials - pooled.positive_ild_trials
    between = animals.groupby(keys, as_index=False).agg(
        animals=("animal", "nunique"), animal_mean_average_rt_s=("animal_mean_rt_s", "mean"),
        across_animal_sd_rt_s=("animal_mean_rt_s", lambda x: x.std(ddof=1)),
        animal_rt_sum_s=("rt_sum_s", "sum"), animal_trial_sum=("window_trials", "sum"))
    points = pooled.merge(between, on=keys, validate="one_to_one")
    points["animal_sem_rt_s"] = points.across_animal_sd_rt_s / np.sqrt(points.animals)
    points["sem_status"] = np.where(points.animals.gt(1), "ok", "unavailable_single_animal")
    points["mean_minus_sem_s"] = points.pooled_mean_rt_s - points.animal_sem_rt_s
    points["mean_plus_sem_s"] = points.pooled_mean_rt_s + points.animal_sem_rt_s
    points["category_position"] = points.abs_ILD.map({ild: i for i, ild in enumerate(ABS_ILDS)})
    points["plot_x"] = points.category_position + points.genotype.map(OFFSETS)
    points["color"] = points.genotype.map({g: to_hex(c) for g, c in COLORS.items()})
    points.insert(0, "view", view)
    assert len(points) == config["points"] and points.window_trials.sum() == 446949
    np.testing.assert_array_equal(points.window_trials, points.animal_trial_sum)
    np.testing.assert_allclose(points.pooled_mean_rt_s, points.animal_rt_sum_s / points.animal_trial_sum, rtol=0, atol=1e-14)
    assert points.loc[points.animals.gt(1), "mean_minus_sem_s"].ge(Y_LIM[0]).all()
    assert points.loc[points.animals.gt(1), "mean_plus_sem_s"].le(Y_LIM[1]).all()
    assert points.loc[points.animals.eq(1), "animal_sem_rt_s"].isna().all()
    point_tables.append(points)
    membership = rows.groupby(["group", "filename", "family", "genotype", "animal"], as_index=False).agg(window_trials=("timed_rt", "size"))
    membership.insert(0, "view", view)
    assert len(membership) == 53
    member_tables.append(membership)
    groups = rows.groupby(["group", "genotype"], as_index=False).agg(animals=("animal", "nunique"), window_trials=("timed_rt", "size"))
    groups.insert(0, "view", view)
    group_tables.append(groups)
points = pd.concat(point_tables, ignore_index=True)
membership = pd.concat(member_tables, ignore_index=True)
groups = pd.concat(group_tables, ignore_index=True)
unavailable = points.loc[points.sem_status.ne("ok")]
assert len(points) == 270 and len(unavailable) == 30
assert unavailable.view.eq("cohort").all() and unavailable.group.eq("SHANK3_cohort2.csv").all()
assert set(unavailable.genotype) == {"wt", "hom"}
points.to_csv(OUTPUT / "chronometric_points.csv", index=False)
membership.to_csv(OUTPUT / "group_membership.csv", index=False)
groups.to_csv(OUTPUT / "group_counts.csv", index=False)
print("Selected 446,949 RTs; 795 animal means; 270 pooled means; 240 animal SEMs; 30 single-animal conditions", flush=True)


# %% Two figures: square panels, ABL columns, genotype colors and common RT limits
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11})
figure_records, layout_records, plotted_records = [], [], []
width = LEFT_IN + 3 * PANEL_SIZE_IN + 2 * COLUMN_GAP_IN + RIGHT_IN
for view, config in VIEWS.items():
    height = HEADER_IN + len(config["groups"]) * PANEL_SIZE_IN + (len(config["groups"])-1) * ROW_GAP_IN + FOOTER_IN
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(TITLE, x=.54, y=1-.23/height, fontsize=14)
    fig.text(.54, 1-.57/height, f"{config['title']} | Valid choices; RT in [0,1] s", ha="center", fontsize=10)
    fig.legend(handles=[Line2D([], [], marker="o", color=COLORS[g], linestyle="None", label=LABELS[g]) for g in GENOTYPES],
        loc="upper center", bbox_to_anchor=(.54, 1-.78/height), ncol=3, frameon=False, fontsize=11)
    for r, group in enumerate(config["groups"]):
        top = height - HEADER_IN - r * (PANEL_SIZE_IN + ROW_GAP_IN)
        bottom = top - PANEL_SIZE_IN
        label = group.removesuffix(".csv").replace("_cohort", "\ncohort ") if view == "cohort" else group
        fig.text(.13/width, (bottom+PANEL_SIZE_IN/2+.45)/height, label, fontsize=11, weight="bold", va="center")
        group_counts = groups.loc[groups.view.eq(view) & groups.group.eq(group)].set_index("genotype")
        for i, genotype in enumerate(GENOTYPES):
            count = int(group_counts.loc[genotype, "animals"])
            fig.text(.13/width, (bottom+PANEL_SIZE_IN/2-.08-i*.20)/height,
                f"{LABELS[genotype]}: {count} animal{'s' if count != 1 else ''}", color=COLORS[genotype], fontsize=9)
        if view == "cohort" and group == "SHANK3_cohort2.csv":
            fig.text(.13/width, (bottom+PANEL_SIZE_IN/2-.80)/height, "WT, HOM: SEM N/A", fontsize=7.5, color=".3")
        for col, abl in enumerate(ABLS):
            left = LEFT_IN + col * (PANEL_SIZE_IN + COLUMN_GAP_IN)
            ax = fig.add_axes([left/width, bottom/height, PANEL_SIZE_IN/width, PANEL_SIZE_IN/height])
            condition = points.loc[points.view.eq(view) & points.group.eq(group) & points.plot_ABL.eq(abl)]
            for genotype in GENOTYPES:
                data_points = condition.loc[condition.genotype.eq(genotype)].sort_values("abs_ILD")
                available = data_points.sem_status.eq("ok")
                ax.errorbar(data_points.loc[available, "plot_x"], data_points.loc[available, "pooled_mean_rt_s"],
                    yerr=data_points.loc[available, "animal_sem_rt_s"], fmt="o", linestyle="None", color=COLORS[genotype],
                    markersize=4.2, elinewidth=1, capsize=2, capthick=1, zorder=3)
                if (~available).any():
                    ax.scatter(data_points.loc[~available, "plot_x"], data_points.loc[~available, "pooled_mean_rt_s"],
                        color=COLORS[genotype], s=4.2**2, marker="o", zorder=3)
            ax.set(title=f"ABL {abl} dB", xlabel="|ILD| (dB)", xlim=(-.5, 4.5), ylim=Y_LIM)
            ax.set_xticks(range(5), [str(i) for i in ABS_ILDS])
            ax.set_yticks(np.arange(0, .451, .1))
            if col == 0:
                ax.set_ylabel("Pooled mean RT (s)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.15)
            ax.tick_params(labelsize=9)
            layout_records.append({"view": view, "group": group, "plot_ABL": abl, "row": r+1, "column": col+1,
                "x_in": left, "y_in": bottom, "width_in": PANEL_SIZE_IN, "height_in": PANEL_SIZE_IN,
                "xlim_low": -.5, "xlim_high": 4.5, "ylim_low": Y_LIM[0], "ylim_high": Y_LIM[1]})
            for point in condition.itertuples(index=False):
                x = (left + (point.plot_x+.5)/5*PANEL_SIZE_IN)*72
                y = (height-bottom-point.pooled_mean_rt_s/.45*PANEL_SIZE_IN)*72
                upper = (height-bottom-point.mean_plus_sem_s/.45*PANEL_SIZE_IN)*72
                lower = (height-bottom-point.mean_minus_sem_s/.45*PANEL_SIZE_IN)*72
                plotted_records.append({"view": view, "group": group, "genotype": point.genotype, "plot_ABL": abl,
                    "abs_ILD": point.abs_ILD, "pooled_mean_rt_s": point.pooled_mean_rt_s, "animal_sem_rt_s": point.animal_sem_rt_s,
                    "sem_status": point.sem_status, "color": point.color, "pdf_x_pt": x, "pdf_y_pt": y,
                    "pdf_sem_top_pt": upper, "pdf_sem_bottom_pt": lower})
    fig.text(.5, .35/height, "Dots: pooled trials. Bars: sample SD(animal condition means) / √animal count. One animal: no SEM bar.", ha="center", fontsize=8)
    fig.text(.5, .13/height, "Training 16; success ∈ {1,−1}; RT [0,1] s; ABL 59→60; actual ±ILD trials pooled; prepared repeat/short-sound/calibration rules retained.",
        ha="center", fontsize=7.8)
    name = f"ASD_session_type_1_{view}_genotype_chronometrics_rt_0_to_1"
    png, pdf = OUTPUT / f"{name}.png", OUTPUT / f"{name}.pdf"
    fig.savefig(png, dpi=PNG_DPI, facecolor="white")
    fig.savefig(pdf, metadata={"Title": TITLE, "Subject": config["title"]})
    figure_records.append({"view": view, "groups": config["groups"], "rows": len(config["groups"]), "columns": 3,
        "points": config["points"], "available_SEMs": int(points.loc[points.view.eq(view), "sem_status"].eq("ok").sum()),
        "width_in": width, "height_in": height, "png": str(png.relative_to(REPO)), "pdf": str(pdf.relative_to(REPO))})
    plt.close(fig)
    print(f"Saved {view}: {len(config['groups'])}×3 panels, {config['points']} means", flush=True)
pd.DataFrame(layout_records).to_csv(OUTPUT / "panel_layout.csv", index=False)
pd.DataFrame(plotted_records).to_csv(OUTPUT / "plotted_points.csv", index=False)


# %% Preserve inputs and record the two distinct estimators in the manifest
for relative, digest in protected_hashes.items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, relative
manifest = {"status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256, "prepared_rows": 610603,
    "selection": "session1/training16; success in {1,-1}; finite timed_rt in inclusive[0,1] s; ABL59->60 then20/40/60; ILDs±1/2/4/8/16; existing prepared exclusions retained",
    "repeat_filter_reapplied": False, "RT_window_s": list(RT_WINDOW), "trials_per_view": 446949, "animals_per_view": 53,
    "animal_condition_means": len(animal_means), "pooled_mean_points": len(points), "available_SEMs": 240, "unavailable_SEMs": 30,
    "point_estimator": "Raw-trial pooled mean, equivalently trial-count-weighted animal condition means",
    "SEM_estimator": "Sample SD of animal condition means about their unweighted mean, divided by sqrt(animal count); ddof=1",
    "mean_and_SEM_weighting": "Point is trial weighted; SEM uses equally weighted animal means; bars centered on pooled point as requested",
    "single_animal_policy": "Retain mean dot, SEM/SD/endpoints NaN, no error bar; label SEM N/A; never replace missing SEM with zero",
    "unavailable_groups": [{"view": "cohort", "group": "SHANK3_cohort2.csv", "genotype": g} for g in ["wt", "hom"]],
    "genotype_order": GENOTYPES, "colors": {g: to_hex(c) for g, c in COLORS.items()}, "offsets": OFFSETS,
    "ABS_ILDS": ABS_ILDS, "ABLS": ABLS, "ylim_s": list(Y_LIM), "title": TITLE,
    "figures": figure_records, "png_dpi": PNG_DPI, "protected_file_sha256": protected_hashes}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2) + "\n")
