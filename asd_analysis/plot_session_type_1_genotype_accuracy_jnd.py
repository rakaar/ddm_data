# %% Imports
"""Genotype accuracy/JND comparisons: pooled points, SEM across animals.

Run with .venv/bin/python from the repository root. These figures use valid
choices with recorded RT in [0,1] s. Pool all selected trials for the point;
use equally weighted, available individual-animal estimates for the SEM.
"""
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# %% Editable parameters
REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "asd_analysis/session_type_csvs/ASD_session_type_1.csv"
OUTPUT = REPO / "asd_analysis/session_type_1_genotype_accuracy_jnd"
REFERENCE = REPO / "asd_analysis/session_type_1_genotype_chronometrics"
COHORTS = ["SHANK3_cohort1.csv", "SHANK3_cohort2.csv", "CNTNAP2_cohort3.csv", "CNTNAP2_cohort4.csv"]
FAMILIES = ["SHANK3", "CNTNAP2"]
ABLS = [20, 40, 60]
ABS_ILDS = [1, 2, 4, 8, 16]
SIGNED_ILDS = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
GENOTYPES = ["wt", "hom", "het"]
LABELS = {"wt": "WT", "hom": "HOM", "het": "HET"}
COLORS = {"wt": "tab:blue", "hom": "tab:green", "het": "tab:orange"}
OFFSETS = {"wt": -.12, "hom": 0., "het": .12}
RT_WINDOW = (0., 1.)
FIT_BOUNDS = ([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf])
INITIAL_K = .1
MAXFEV = 10000
PNG_DPI = 200
PANEL_IN = 2.45
COLUMN_GAP_IN, ROW_GAP_IN = .60, 1.00
LEFT_IN, RIGHT_IN, HEADER_IN, FOOTER_IN = 1.80, .50, 1.60, 1.50
VIEWS = {"cohort": {"column": "filename", "groups": COHORTS},
         "family": {"column": "family", "groups": FAMILIES}}
TITLES = {"accuracy": "Pooled accuracy; error bars = SEM across animal accuracies",
          "jnd": "JND from pooled fits; error bars = SEM across animal JNDs"}
OUTPUT.mkdir(parents=True, exist_ok=True)


# %% Protect data and the previous chronometric/JND results
protected = set(INPUT.parent.glob("*.csv"))
protected.update(REPO.glob("SHANK3_cohort*.csv"))
protected.update(REPO.glob("CNTNAP2_cohort*.csv"))
for folder in [REFERENCE, REPO / "asd_analysis/session_type_1_animal_metric_deviations_calibrated"]:
    protected.update(p for p in folder.iterdir() if p.is_file())
for pattern in ["*genotype_chronometrics*", "genotype_chronometric*", "ASD_genotype_chronometric_audits.zip"]:
    protected.update((REPO / "docs/assets/results/2026-10-21").glob(pattern))
protected_hashes = {}
for path in sorted(protected):
    with path.open("rb") as stream:
        protected_hashes[str(path.relative_to(REPO))] = hashlib.file_digest(stream, "sha256").hexdigest()
input_sha256 = protected_hashes[str(INPUT.relative_to(REPO))]
reference = json.loads((REFERENCE / "validation.json").read_text())
assert reference["status"] == "passed" and reference["input_sha256"] == input_sha256


# %% Existing paper fit and JND definition; individual undefined JNDs stay unavailable
def sigmoid(x, upper, lower, x0, k):
    return lower + (upper - lower) / (1 + np.exp(-k * (x - x0)))


def jnd_from_fit(upper, lower, x0, k):
    high, low = .75 * upper, .25 * (1 - lower)
    result = {"y_high": high, "y_low": low, "ild_high": np.nan, "ild_low": np.nan,
              "jnd": np.nan, "jnd_status": "invalid_fit"}
    if not np.isfinite([upper, lower, x0, k]).all() or upper <= lower or k <= 0:
        return result
    for side, threshold in [("high", high), ("low", low)]:
        if lower < threshold < upper:
            result[f"ild_{side}"] = x0 - np.log((upper-lower)/(threshold-lower)-1) / k
    if not lower < low < high < upper:
        result["jnd_status"] = "thresholds_outside_asymptotes_or_reversed"
        return result
    value = (result["ild_high"]-result["ild_low"]) / 2
    if np.isfinite([result["ild_high"], result["ild_low"], value]).all() and value > 0:
        result.update(jnd=float(value), jnd_status="ok")
    else:
        result["jnd_status"] = "nonfinite_or_nonpositive_jnd"
    return result


def fit_condition(x, probability):
    initial = [float(max(probability)), float(min(probability)), float(np.median(x)), INITIAL_K]
    result = dict.fromkeys(["upper", "lower", "x0", "k", "jnd", "y_high", "y_low", "ild_high", "ild_low", "rmse", "unweighted_sse"], np.nan)
    result.update(fit_status="optimizer_failed", jnd_status="fit_unavailable", initial_parameters=json.dumps(initial),
                  warnings="", optimizer_message="", function_evaluations=0, covariance_finite=False)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parameters, covariance, info, message, code = curve_fit(sigmoid, x, probability, p0=initial,
                bounds=FIT_BOUNDS, maxfev=MAXFEV, full_output=True)
        result.update(zip(["upper", "lower", "x0", "k"], parameters))
        result.update(optimizer_message=message, optimizer_code=int(code), function_evaluations=int(info["nfev"]),
                      warnings="; ".join(dict.fromkeys(str(w.message) for w in caught)), covariance_finite=bool(np.isfinite(covariance).all()))
        if not np.isfinite(parameters).all():
            result["fit_status"] = "nonfinite_parameters"
        elif parameters[0] <= parameters[1] or parameters[3] <= 0:
            result["fit_status"] = "non_increasing"
        else:
            residual = sigmoid(x, *parameters)-probability
            result.update(fit_status="ok", rmse=float(np.sqrt(np.mean(residual**2))), unweighted_sse=float(residual @ residual))
            result.update(jnd_from_fit(*parameters))
    except (RuntimeError, ValueError, FloatingPointError) as error:
        result["optimizer_message"] = str(error)
    return result


# %% Select the same RT-window population as the October 21 chronometrics
data = pd.read_csv(INPUT, usecols=["filename", "animal", "genotype", "session", "session_type", "training_level",
    "trial_is_repeat", "is_short_sound", "success", "timed_rt", "ABL", "ILD", "response_poke"], low_memory=False)
assert len(data) == 610603 and data.animal.nunique() == 53
assert data.session_type.eq(1).all() and data.training_level.eq(16).all()
assert data.trial_is_repeat.eq(False).all() and data.is_short_sound.eq(0).all()
assert not (data.filename.eq("CNTNAP2_cohort4.csv") & data.session.lt(13)).any()
data["plot_ABL"] = data.ABL.replace({59: 60})
selected = data.loc[data.success.isin([1, -1]) & np.isfinite(data.timed_rt) & data.timed_rt.between(*RT_WINDOW)
    & data.plot_ABL.isin(ABLS) & data.ILD.isin(SIGNED_ILDS)].copy()
selected["family"] = selected.filename.str.split("_").str[0]
selected["plot_ABL"] = selected.plot_ABL.astype(int)
selected["abs_ILD"] = selected.ILD.abs().astype(int)
selected["correct"] = selected.success.eq(1)
selected["right"] = selected.response_poke.eq(1)
selected["positive_ild"] = selected.ILD.gt(0)
assert len(selected) == 446949 and selected.animal.nunique() == 53
assert selected.response_poke.isin([1, -1]).all()
assert selected.correct.sum() == 343217 and selected.right.sum() == 229058
print("Selected 446,949 valid RT-window choices; all 53 animals", flush=True)


# %% Accuracy: trial-pooled dots and sample SEM across animal condition accuracies
animal_keys = ["filename", "family", "animal", "genotype", "plot_ABL", "abs_ILD"]
animal_accuracy = selected.groupby(animal_keys, as_index=False).agg(window_trials=("correct", "size"),
    correct_choices=("correct", "sum"), animal_accuracy=("correct", "mean"), positive_ild_trials=("positive_ild", "sum"))
animal_accuracy["negative_ild_trials"] = animal_accuracy.window_trials-animal_accuracy.positive_ild_trials
assert len(animal_accuracy) == 795 and animal_accuracy.window_trials.sum() == 446949
animal_accuracy.to_csv(OUTPUT / "animal_accuracy.csv", index=False)
accuracy_tables, membership_tables, group_tables = [], [], []
for view, config in VIEWS.items():
    rows = selected.assign(group=selected[config["column"]])
    animals = animal_accuracy.assign(group=animal_accuracy[config["column"]])
    keys = ["group", "genotype", "plot_ABL", "abs_ILD"]
    pooled = rows.groupby(keys, as_index=False).agg(window_trials=("correct", "size"), correct_choices=("correct", "sum"),
        pooled_value=("correct", "mean"), positive_ild_trials=("positive_ild", "sum"))
    pooled["negative_ild_trials"] = pooled.window_trials-pooled.positive_ild_trials
    between = animals.groupby(keys, as_index=False).agg(pool_animals=("animal", "nunique"),
        animal_estimate_mean=("animal_accuracy", "mean"), animal_estimate_sd=("animal_accuracy", lambda x: x.std(ddof=1)))
    points = pooled.merge(between, on=keys, validate="one_to_one")
    points["sem_animals"] = points.pool_animals
    points["sem"] = points.animal_estimate_sd / np.sqrt(points.sem_animals)
    points["sem_status"] = np.where(points.sem_animals.gt(1), "ok", "unavailable_single_animal")
    points.insert(0, "view", view)
    assert len(points) == (180 if view == "cohort" else 90) and points.window_trials.sum() == 446949
    accuracy_tables.append(points)
    membership = rows.groupby(["group", "filename", "family", "genotype", "animal"], as_index=False).agg(window_trials=("correct", "size"))
    membership.insert(0, "view", view)
    assert len(membership) == 53
    membership_tables.append(membership)
    counts = rows.groupby(["group", "genotype"], as_index=False).agg(pool_animals=("animal", "nunique"), window_trials=("correct", "size"))
    counts.insert(0, "view", view)
    group_tables.append(counts)
accuracy = pd.concat(accuracy_tables, ignore_index=True)
membership = pd.concat(membership_tables, ignore_index=True)
groups = pd.concat(group_tables, ignore_index=True)
assert accuracy["sem"].notna().sum() == 240 and accuracy["sem"].isna().sum() == 30
membership.to_csv(OUTPUT / "group_membership.csv", index=False)
groups.to_csv(OUTPUT / "group_counts.csv", index=False)


# %% Refit 159 animal and 54 pooled signed-ILD psychometrics on this RT window
condition_tables, fit_records = [], []
for view, column in [("animal", "animal"), ("cohort", "filename"), ("family", "family")]:
    rows = selected.assign(group=selected[column])
    conditions = rows.groupby(["group", "genotype", "plot_ABL", "ILD"], as_index=False).agg(
        choices=("right", "size"), right_choices=("right", "sum"), correct_choices=("correct", "sum"), pool_animals=("animal", "nunique"))
    conditions.insert(0, "view", view)
    conditions["left_choices"] = conditions.choices-conditions.right_choices
    conditions["p_right"] = conditions.right_choices/conditions.choices
    for (group, genotype, abl), condition in conditions.groupby(["group", "genotype", "plot_ABL"]):
        condition = condition.sort_values("ILD")
        assert condition.ILD.tolist() == SIGNED_ILDS
        result = fit_condition(condition.ILD.to_numpy(), condition.p_right.to_numpy())
        conditions.loc[condition.index, "fitted_p_right"] = sigmoid(condition.ILD.to_numpy(), *[result[k] for k in ["upper", "lower", "x0", "k"]])
        fit_records.append({"view": view, "group": group, "genotype": genotype, "plot_ABL": abl,
            "window_trials": int(condition.choices.sum()), "pool_animals": int(condition.pool_animals.max()), **result})
    condition_tables.append(conditions)
conditions = pd.concat(condition_tables, ignore_index=True)
fits = pd.DataFrame(fit_records)
assert len(conditions) == 2130 and len(fits) == 213 and fits.fit_status.eq("ok").all()
assert fits.loc[fits.view.eq("animal"), "jnd"].notna().sum() == 154
assert fits.loc[fits.view.ne("animal"), "jnd"].notna().sum() == 54
conditions.to_csv(OUTPUT / "condition_counts.csv", index=False)
fits.to_csv(OUTPUT / "jnd_fits.csv", index=False)
fits.groupby(["view", "fit_status", "jnd_status"]).size().rename("fits").reset_index().to_csv(OUTPUT / "fit_status_counts.csv", index=False)


# %% JND SEM uses available animal JNDs, while every animal stays in pooled fits
identities = selected[["animal", "filename", "family", "genotype"]].drop_duplicates()
animal_jnd = fits.loc[fits.view.eq("animal")].rename(columns={"group": "animal"}).merge(
    identities, on=["animal", "genotype"], validate="many_to_one")
animal_jnd.to_csv(OUTPUT / "animal_jnd.csv", index=False)
jnd_tables = []
for view, config in VIEWS.items():
    animals = animal_jnd.assign(group=animal_jnd[config["column"]])
    between = animals.groupby(["group", "genotype", "plot_ABL"], as_index=False).agg(
        sem_animals=("jnd", "count"), animal_estimate_mean=("jnd", "mean"), animal_estimate_sd=("jnd", lambda x: x.std(ddof=1)))
    points = fits.loc[fits.view.eq(view)].merge(between, on=["group", "genotype", "plot_ABL"], validate="one_to_one")
    points["pooled_value"] = points.jnd
    points["sem"] = points.animal_estimate_sd / np.sqrt(points.sem_animals)
    points["sem_status"] = np.select([points.sem_animals.gt(1), points.sem_animals.eq(1)], ["ok", "unavailable_single_animal"], default="unavailable_no_defined_animals")
    points["unavailable_animal_jnds"] = points.pool_animals-points.sem_animals
    jnd_tables.append(points)
jnd = pd.concat(jnd_tables, ignore_index=True)
assert len(jnd) == 54 and jnd["sem"].notna().sum() == 47 and jnd["sem"].isna().sum() == 7
for points in [accuracy, jnd]:
    points["lower_bar"] = points.pooled_value-points["sem"]
    points["upper_bar"] = points.pooled_value+points["sem"]
    points["color"] = points.genotype.map({g: to_hex(c) for g, c in COLORS.items()})
accuracy["plot_x"] = accuracy.abs_ILD.map({v: i for i, v in enumerate(ABS_ILDS)})+accuracy.genotype.map(OFFSETS)
jnd["plot_x"] = jnd.genotype.map({g: i for i, g in enumerate(GENOTYPES)})
accuracy.to_csv(OUTPUT / "accuracy_points.csv", index=False)
jnd.to_csv(OUTPUT / "jnd_points.csv", index=False)
endpoints = np.r_[jnd.pooled_value, jnd.lower_bar.dropna(), jnd.upper_bar.dropna()]
low, high = min(0., endpoints.min()), endpoints.max()
pad = .05*(high-low)
ylims = {"accuracy": (0., 1.), "jnd": (low-pad if low < 0 else 0., float(np.ceil((high+pad)*10)/10))}
assert accuracy.lower_bar.dropna().ge(0).all() and accuracy.upper_bar.dropna().le(1).all()
print("795 animal accuracies; 270 pooled accuracies; 213 fits; 54 pooled JNDs; 240/47 available SEMs", flush=True)


# %% Four square-panel figures, with actual PDF coordinates retained for verification
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11})
figure_records, layout_records, plotted_records = [], [], []
width = LEFT_IN+3*PANEL_IN+2*COLUMN_GAP_IN+RIGHT_IN
for metric, table in [("accuracy", accuracy), ("jnd", jnd)]:
    ylim = ylims[metric]
    xlim = (-.5, 4.5) if metric == "accuracy" else (-.5, 2.5)
    for view, config in VIEWS.items():
        height = HEADER_IN+len(config["groups"])*PANEL_IN+(len(config["groups"])-1)*ROW_GAP_IN+FOOTER_IN
        fig = plt.figure(figsize=(width, height))
        fig.suptitle(TITLES[metric], x=.54, y=1-.23/height, fontsize=13)
        fig.text(.54, 1-.57/height, f"{'Cohorts' if view == 'cohort' else 'Gene families; cohorts combined'} | 53 animals | 446,949 valid choices | RT [0,1] s", ha="center", fontsize=9.5)
        fig.legend(handles=[Line2D([], [], marker="o", color=COLORS[g], linestyle="None", label=LABELS[g]) for g in GENOTYPES],
            loc="upper center", bbox_to_anchor=(.54, 1-.78/height), ncol=3, frameon=False, fontsize=11)
        for row, group in enumerate(config["groups"]):
            bottom = height-HEADER_IN-row*(PANEL_IN+ROW_GAP_IN)-PANEL_IN
            label = group.removesuffix(".csv").replace("_cohort", "\ncohort ")
            fig.text(.13/width, (bottom+PANEL_IN/2+.45)/height, label, fontsize=11, weight="bold", va="center")
            counts = groups.loc[groups.view.eq(view) & groups.group.eq(group)].set_index("genotype")
            for i, genotype in enumerate(GENOTYPES):
                n = int(counts.loc[genotype, "pool_animals"])
                fig.text(.13/width, (bottom+PANEL_IN/2-.08-i*.20)/height, f"{LABELS[genotype]}: {n} animal{'s' if n != 1 else ''}", color=COLORS[genotype], fontsize=9)
            if metric == "jnd":
                fig.text(.13/width, (bottom+PANEL_IN/2-.82)/height, "Counts at ticks:\npooled / defined for SEM", fontsize=7.2, color=".3")
            for col, abl in enumerate(ABLS):
                left = LEFT_IN+col*(PANEL_IN+COLUMN_GAP_IN)
                ax = fig.add_axes([left/width, bottom/height, PANEL_IN/width, PANEL_IN/height])
                panel = table.loc[table.view.eq(view) & table.group.eq(group) & table.plot_ABL.eq(abl)]
                for genotype in GENOTYPES:
                    p = panel.loc[panel.genotype.eq(genotype)].sort_values("plot_x")
                    ok = p.sem_status.eq("ok")
                    if ok.any():
                        ax.errorbar(p.loc[ok, "plot_x"], p.loc[ok, "pooled_value"], yerr=p.loc[ok, "sem"], fmt="o", linestyle="None",
                            color=COLORS[genotype], markersize=4.2, elinewidth=1, capsize=2, capthick=1, zorder=3)
                    if (~ok).any():
                        ax.scatter(p.loc[~ok, "plot_x"], p.loc[~ok, "pooled_value"], s=4.2**2, color=COLORS[genotype], marker="o", zorder=3)
                if metric == "accuracy":
                    ax.set_xticks(range(5), [str(i) for i in ABS_ILDS])
                    ax.set_xlabel("|ILD| (dB)")
                else:
                    labels = []
                    for genotype in GENOTYPES:
                        p = panel.loc[panel.genotype.eq(genotype)].iloc[0]
                        labels.append(f"{LABELS[genotype]}\npool {p.pool_animals}\nSEM {p.sem_animals}" + (" (N/A)" if p.sem_animals < 2 else ""))
                    ax.set_xticks(range(3), labels, fontsize=7.5)
                    ax.set_xlabel("Genotype", fontsize=9)
                ax.set(title=f"ABL {abl} dB", xlim=xlim, ylim=ylim)
                if col == 0:
                    ax.set_ylabel("Pooled accuracy" if metric == "accuracy" else "JND (dB)")
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(alpha=.15)
                ax.tick_params(axis="y", labelsize=9)
                layout_records.append({"metric": metric, "view": view, "group": group, "plot_ABL": abl,
                    "x_in": left, "y_in": bottom, "width_in": PANEL_IN, "height_in": PANEL_IN,
                    "xlim_low": xlim[0], "xlim_high": xlim[1], "ylim_low": ylim[0], "ylim_high": ylim[1]})
                for p in panel.itertuples(index=False):
                    x = (left+(p.plot_x-xlim[0])/(xlim[1]-xlim[0])*PANEL_IN)*72
                    y = (height-bottom-(p.pooled_value-ylim[0])/(ylim[1]-ylim[0])*PANEL_IN)*72
                    top = (height-bottom-(p.upper_bar-ylim[0])/(ylim[1]-ylim[0])*PANEL_IN)*72
                    lower = (height-bottom-(p.lower_bar-ylim[0])/(ylim[1]-ylim[0])*PANEL_IN)*72
                    plotted_records.append({"metric": metric, "view": view, "group": group, "genotype": p.genotype, "plot_ABL": abl,
                        "abs_ILD": p.abs_ILD if metric == "accuracy" else np.nan, "pooled_value": p.pooled_value, "sem": p.sem,
                        "pool_animals": p.pool_animals, "sem_animals": p.sem_animals, "color": p.color,
                        "pdf_x_pt": x, "pdf_y_pt": y, "pdf_bar_top_pt": top, "pdf_bar_bottom_pt": lower})
        footer = "Dots: pooled correct / valid choices. Bars: sample SD(animal accuracies) / √animal count." if metric == "accuracy" else "Dots: JND from all pooled choices. Bars: sample SD(defined animal JNDs) / √defined-animal count."
        fig.text(.5, .42/height, footer+" Missing SEM: no bar.", ha="center", fontsize=7.7)
        fig.text(.5, .19/height, "Training 16; success ∈ {1,−1}; RT [0,1] s; ABL 59→60; prepared repeat/short-sound/calibration rules retained.", ha="center", fontsize=7.8)
        stem = f"ASD_session_type_1_{view}_genotype_{metric}_rt_0_to_1"
        fig.savefig(OUTPUT / f"{stem}.png", dpi=PNG_DPI, facecolor="white")
        fig.savefig(OUTPUT / f"{stem}.pdf", metadata={"Title": TITLES[metric]})
        figure_records.append({"metric": metric, "view": view, "rows": len(config["groups"]), "columns": 3,
            "width_in": width, "height_in": height, "points": len(table.loc[table.view.eq(view)]),
            "available_SEMs": int(table.loc[table.view.eq(view), "sem"].notna().sum()),
            "png": str((OUTPUT / f"{stem}.png").relative_to(REPO)), "pdf": str((OUTPUT / f"{stem}.pdf").relative_to(REPO))})
        plt.close(fig)
        print(f"Saved {view} {metric}", flush=True)
pd.DataFrame(layout_records).to_csv(OUTPUT / "panel_layout.csv", index=False)
pd.DataFrame(plotted_records).to_csv(OUTPUT / "plotted_points.csv", index=False)


# %% Recheck preservation and save explicit estimator and eligibility provenance
for relative, digest in protected_hashes.items():
    with (REPO / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, relative
manifest = {"status": "passed", "source_script": str(Path(__file__).relative_to(REPO)),
    "input_csv": str(INPUT.relative_to(REPO)), "input_sha256": input_sha256, "prepared_rows": len(data),
    "selection": "session1/training16; success in {1,-1}; finite recorded timed_rt in inclusive [0,1] s; ABL59->60 then20/40/60; signedILDs±1/2/4/8/16; existing preparation exclusions retained",
    "repeat_filter_reapplied": False, "trials_per_view": len(selected), "animals_per_view": 53,
    "correct_choices_per_view": int(selected.correct.sum()), "right_choices_per_view": int(selected.right.sum()),
    "animal_accuracy_points": 795, "accuracy_points": 270, "accuracy_available_SEMs": 240, "accuracy_unavailable_SEMs": 30,
    "attempted_fits": 213, "animal_fits": 159, "defined_animal_JNDs": 154, "undefined_animal_JNDs": 5,
    "pooled_JNDs": 54, "JND_available_SEMs": 47, "JND_unavailable_SEMs": 7,
    "accuracy_point": "Pooled correct choices / all selected valid choices; signed ILDs combined by actual trials",
    "JND_point": "JND from pooled right/left counts at ten signed ILDs; all animals retained even when individual JND undefined",
    "SEM": "Sample SD of available individual-animal estimates around their unweighted mean / sqrt(number of available estimates); ddof=1",
    "bar_center": "Pooled point, as requested; not the unweighted animal-estimate mean",
    "fit": {"sigmoid": "lower+(upper-lower)/(1+exp(-k*(ILD-x0)))", "parameter_order": ["upper", "lower", "x0", "k"],
        "initial": "[max(Pright),min(Pright),median(ILD),0.1]", "bounds": "asymptotes[0,1]; x0 unrestricted; k>=0", "maxfev": MAXFEV,
        "weighting": "Unweighted ten measured ILD probabilities", "JND": "half the inverse-sigmoid spacing at y_high=.75*upper and y_low=.25*(1-lower)"},
    "single_animal_policy": "SD/SEM NaN, dot retained without bar; no zero substitution",
    "genotype_order": GENOTYPES, "colors": {g: to_hex(c) for g, c in COLORS.items()}, "accuracy_offsets": OFFSETS,
    "ylims": ylims, "figures": figure_records, "protected_file_sha256": protected_hashes}
(OUTPUT / "validation.json").write_text(json.dumps(manifest, indent=2)+"\n")
