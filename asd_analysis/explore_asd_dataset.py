# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# %%
# Parameters
MERGED_SUBJECTS_REL_PATH = Path("merged_all_subjects.csv")


def find_repo_root(start_path: Path, sentinel_rel_path: Path) -> Path:
    """Walk upward from start_path until sentinel_rel_path exists."""
    for candidate in [start_path, *start_path.parents]:
        if (candidate / sentinel_rel_path).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate repository root containing '{sentinel_rel_path}'. "
        f"Started search at '{start_path}'."
    )

def prepare_data(
    df,
    #training_level_filter=16,
    session_col="session",   # <-- change to your real session column name
    trial_col="trial",       # <-- or trial_index if different
):
    """
    Full data preparation pipeline:
    1. Fix ABL values according to Mafalda's rules
    2. Filter to a specific training level
    3. Add a 'trial_is_repeat' column:
       True  = current trial is itself a repetition trial
       False = otherwise
    """

    df = df.copy()

    # ----------------------------
    # 1. ---- ABL FIXES ---------
    # ----------------------------
    mask1 = df["training_level"] < 7
    df.loc[mask1, "ABL"] = pd.to_numeric(df.loc[mask1, "ABL"], errors="coerce") * 2

    mask2 = df["ABL"] == 59
    df.loc[mask2, "ABL"] = 60

    mask3 = df["ABL"] == 58
    df.loc[mask3, "ABL"] = 60

    mask4 = (df["training_level"] == 16) & (df["ABL"] == 25)
    df.loc[mask4, "ABL"] = 50

    # ----------------------------
    # 2. ---- FILTER BY LEVEL ----
    # ----------------------------
    # if training_level_filter is not None:
    #     df = df[df["training_level"] == training_level_filter].copy()

    # ----------------------------
    # 3. ---- MARK REPEATED TRIALS ----
    #
    # repeated_trial == True flags the *failed* trial that SHOULD be repeated.
    # We want to mark the NEXT trial (within the same session)
    # as "trial_is_repeat" *if and only if* the previous trial
    #   - had repeated_trial == True
    #   - abort_type NOT in {"Fixation", "CNP"}
    # ----------------------------

    df = df.sort_values([session_col, trial_col]).copy()
    df["trial_is_repeat"] = False

    non_repeat_abort_types = {"Fixation", "CNP"}

    for session_id, df_sess in df.groupby(session_col):
        idx = df_sess.index.to_list()

        for i in range(1, len(idx)):
            prev_idx = idx[i - 1]
            this_idx = idx[i]

            prev_row = df.loc[prev_idx]

            prev_triggers_repeat = (
                bool(prev_row["repeated_trial"]) and
                str(prev_row["abort_type"]) not in non_repeat_abort_types
            )

            if prev_triggers_repeat:
                df.loc[this_idx, "trial_is_repeat"] = True

    return df




#prep data for the short durations

DUR_RULES = [(12, 16, 15), (17, 21, 60), (22, 26, 120)]
RT_MS = 6000

def add_stim_dur(df, sound_col="sound_index", session_col="session_type",
                 out_col="stim_dur", type1_value=RT_MS):  # set to pd.NA for NaN behavior
    df = df.copy()

    # default for everything (type1 -> RT_MS, or pd.NA if you prefer)
    df[out_col] = pd.Series([type1_value] * len(df), index=df.index, dtype="Int64")

    # only compute mapping for session_type == 2 (and only if column exists)
    if session_col in df.columns:
        mask = pd.to_numeric(df[session_col], errors="coerce") == 2
    else:
        mask = pd.Series(False, index=df.index)

    if mask.any():
        s = pd.to_numeric(df.loc[mask, sound_col], errors="coerce")
        conds = [s.between(lo, hi) for lo, hi, _ in DUR_RULES]
        choices = [dur for _, _, dur in DUR_RULES]
        df.loc[mask, out_col] = pd.Series(np.select(conds, choices, default=RT_MS), index=df.loc[mask].index).astype("Int64")

    # optional label column
    df["stim_dur_label"] = np.where(df[out_col] == RT_MS, "RT", df[out_col].astype(str) + "ms")
    return df
# %%
start_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
repo_root = find_repo_root(start_path, MERGED_SUBJECTS_REL_PATH)
og_df = pd.read_csv(repo_root / MERGED_SUBJECTS_REL_PATH)
og_df = prepare_data(og_df)
# %%
og_df['session'].unique()
# %%
df = og_df[
    (og_df['training_level'] == 16) &
    (og_df['session'] > 13) &
    (og_df['short_duration'] == 0) &
    (og_df['trial_is_repeat'] == False)
].copy()

# animal;cohort;line;sex;genotype
# ASD0007;2;cntnap2;male;wt
# ASD0008;2;cntnap2;male;hom
# ASD0009;2;cntnap2;male;het
# ASD0010;2;cntnap2;female;hom
# ASD0011;2;cntnap2;female;wt
# ASD0012;2;cntnap2;female;het
# ASD0013;2;cntnap2;female;hom
# ASD0014;2;cntnap2;male;wt
# ASD0015;2;cntnap2;male;wt
# ASD0016;2;cntnap2;male;het
# ASD0017;2;cntnap2;male;wt
# ASD0018;2;cntnap2;female;het
# ASD0019;2;cntnap2;female;hom
# ASD0020;2;cntnap2;female;hom
# ASD0021;2;cntnap2;female;het
# ASD0022;2;cntnap2;female;hom
# TODO
# filter only wt animals ,
# animal ID is df['animal']
# %%
WT_ANIMALS = {"ASD0007", "ASD0011", "ASD0014", "ASD0015", "ASD0017"}
HOM_ANIMALS = {"ASD0008", "ASD0010", "ASD0013", "ASD0019", "ASD0020", "ASD0022"}
HET_ANIMALS = {"ASD0009", "ASD0012", "ASD0016", "ASD0018", "ASD0021"}

animal_as_str = df["animal"].astype(str).str.strip()
animal_as_asd = animal_as_str.where(
    animal_as_str.str.startswith("ASD"),
    "ASD" + animal_as_str.str.extract(r"(\d+)")[0].str.zfill(4)
)
df = df[animal_as_asd.isin(WT_ANIMALS)].copy()
# df = df[animal_as_asd.isin(HOM_ANIMALS)].copy()


# %%
df_valid = df[df['success'].isin([1,-1] )]

df_valid.to_csv('asd_wt.csv', index=False)
print('saved')
# %%
ABL_uniq = [20, 40, 60]
ABL_color = {20: 'blue', 40: 'green', 60: 'red'}
for ABL in ABL_uniq:
    df_ABL = df_valid[df_valid['ABL'] == ABL]
    rt_abl = df_ABL['timed_rt']
    plt.hist(rt_abl, bins=np.arange(0,1,0.01), histtype='step', density=True, label=f'ABL={ABL}', color=ABL_color[ABL])

plt.legend()
plt.xlim(0, 0.5)
# %%
ABL_uniq = [20, 40, 60]
abs_ILD_uniq = [1, 2, 4, 8, 16]

df_plot = df_valid[df_valid["ABL"].isin(ABL_uniq)].copy()
df_plot["abs_ILD"] = df_plot["ILD"].abs()
df_plot = df_plot[df_plot["abs_ILD"].isin(abs_ILD_uniq)]

fig, axes = plt.subplots(3, 5, figsize=(20, 10), sharex=True, sharey=True)
bins = np.arange(0, 1, 0.01)

for i, ABL in enumerate(ABL_uniq):
    df_abl = df_plot[df_plot["ABL"] == ABL]
    for j, abs_ild in enumerate(abs_ILD_uniq):
        ax = axes[i, j]
        rt = df_abl[df_abl["abs_ILD"] == abs_ild]["timed_rt"]
        ax.hist(rt, bins=bins, histtype="step", density=True)
        ax.set_title(f"ABL={ABL}, |ILD|={abs_ild}")
        if i == 2:
            ax.set_xlabel("timed_rt")
        if j == 0:
            ax.set_ylabel("density")

for ax in axes.ravel():
    ax.set_xlim(0, 0.5)

plt.tight_layout()

# %%
# and now a 1 x 5 plot
# all 3 ABLs in one abs ILD plot
fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True)
bins = np.arange(0, 1, 0.01)

for j, abs_ild in enumerate(abs_ILD_uniq):
    ax = axes[j]
    for ABL in ABL_uniq:
        rt = df_plot[(df_plot["abs_ILD"] == abs_ild) & (df_plot["ABL"] == ABL)]["timed_rt"]
        if len(rt) > 0:
            ax.hist(
                rt,
                bins=bins,
                histtype="step",
                density=True,
                color=ABL_color[ABL],
                label=f"ABL={ABL}",
            )
    ax.set_title(f"|ILD|={abs_ild}")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("timed_rt")
    if j == 0:
        ax.set_ylabel("density")
        ax.legend(frameon=False)

plt.tight_layout()

# %%

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True, sharey=True)
bins = np.arange(0, 1, 0.01)

ild_colors = dict(zip(abs_ILD_uniq, plt.cm.viridis(np.linspace(0.1, 0.9, len(abs_ILD_uniq)))))

for i, ABL in enumerate(ABL_uniq):
    ax = axes[i]
    df_abl = df_plot[df_plot["ABL"] == ABL]
    for abs_ild in abs_ILD_uniq:
        rt = df_abl[df_abl["abs_ILD"] == abs_ild]["timed_rt"]
        if len(rt) > 0:
            ax.hist(
                rt,
                bins=bins,
                histtype="step",
                density=True,
                color=ild_colors[abs_ild],
                label=f"|ILD|={abs_ild}",
            )
    ax.set_title(f"ABL={ABL}")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("timed_rt")
    if i == 0:
        ax.set_ylabel("density")
        ax.legend(frameon=False, ncol=2)

plt.tight_layout()

# %%
def plot_ecdf(ax, values, color, label):
    vals = np.sort(np.asarray(values, dtype=float))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    y = np.arange(1, vals.size + 1) / vals.size
    ax.step(vals, y, where="post", color=color, label=label)

# %%
fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True)

for j, abs_ild in enumerate(abs_ILD_uniq):
    ax = axes[j]
    for ABL in ABL_uniq:
        rt = df_plot[(df_plot["abs_ILD"] == abs_ild) & (df_plot["ABL"] == ABL)]["timed_rt"]
        plot_ecdf(ax, rt, color=ABL_color[ABL], label=f"ABL={ABL}")
    ax.set_title(f"|ILD|={abs_ild}")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("timed_rt")
    if j == 0:
        ax.set_ylabel("CDF")
        ax.legend(frameon=False)
plt.xlim(0.04, 0.15)
plt.tight_layout()

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True, sharey=True)
ild_colors = dict(zip(abs_ILD_uniq, plt.cm.viridis(np.linspace(0.1, 0.9, len(abs_ILD_uniq)))))

for i, ABL in enumerate(ABL_uniq):
    ax = axes[i]
    df_abl = df_plot[df_plot["ABL"] == ABL]
    for abs_ild in abs_ILD_uniq:
        rt = df_abl[df_abl["abs_ILD"] == abs_ild]["timed_rt"]
        plot_ecdf(ax, rt, color=ild_colors[abs_ild], label=f"|ILD|={abs_ild}")
    ax.set_title(f"ABL={ABL}")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("timed_rt")
    if i == 0:
        ax.set_ylabel("CDF")
        ax.legend(frameon=False, ncol=2)
plt.xlim(0.04, 0.15)
plt.tight_layout()

# %%
# Aggregate CDF across all |ILD| values (one curve per ABL)
for ABL in ABL_uniq:
    rt_abl_ms = (1000.0 * df_plot[df_plot["ABL"] == ABL]["timed_rt"]).dropna().to_numpy()
    rt_abl_ms = rt_abl_ms[rt_abl_ms >= 0]
    if rt_abl_ms.size == 0:
        continue

    rt_sorted = np.sort(rt_abl_ms)
    cdf_vals = np.arange(1, rt_sorted.size + 1) / rt_sorted.size
    plt.step(
        rt_sorted,
        cdf_vals,
        where="post",
        color=ABL_color.get(ABL, "black"),
        linewidth=2,
        label=f"ABL={ABL} CDF",
    )

plt.xlim(0, 200)
plt.ylim(0, 1.01)
plt.xlabel("RT (ms)")
plt.ylabel("Cumulative probability")
plt.title("RT CDF by ABL")
plt.legend()
plt.tight_layout()

# %%
