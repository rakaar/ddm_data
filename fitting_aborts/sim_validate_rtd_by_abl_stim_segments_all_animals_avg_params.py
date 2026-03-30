# %%
SHOW_PLOT = True

DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]

MODEL_KEY = "vbmc_norm_tied_results"
ABORT_KEY = "vbmc_aborts_results"
PARAM_REDUCER = "mean"
TRIAL_POOL_MODE = "valid"  # "valid" or "valid_plus_abort3"
SEGMENT_MODE = "quantile"  # "quantile" or "fixed"
NUM_INTENDED_FIX_QUANTILE_BINS = 2
FIXED_SEGMENT_EDGES_S = (0.2, 0.4, 1.5)

ABL_VALUES = (20, 40, 60)
ILD_VALUES = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)

N_SIM = int(1e6)
N_JOBS = 30
dt = 1e-3
N_PRINT = int(N_SIM / 5)
RNG_SEED = 12345

intended_fix_min_s = 0.2
intended_fix_max_s = 1.5
rt_max_wrt_stim_s = 1.0
xlim_s = (0, 1)
bin_size_s = 0.01
png_dpi = 300

# %%
from pathlib import Path
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent
FIT_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(FIT_DIR) not in sys.path:
    sys.path.append(str(FIT_DIR))

from time_vary_and_norm_simulators import psiam_tied_data_gen_wrapper_rate_norm_fn

# %%
batch_csv_dir = FIT_DIR / "batch_csvs"
results_dir = FIT_DIR
output_dir = SCRIPT_DIR / "sim_rtd_stim_segments_all_animals_avg_params"


def reduce_param_values(values):
    values = np.asarray(values, dtype=float)
    if PARAM_REDUCER == "mean":
        return float(np.mean(values))
    if PARAM_REDUCER == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown PARAM_REDUCER: {PARAM_REDUCER}")


def get_candidate_animals():
    candidates = []
    for batch_name in DESIRED_BATCHES:
        batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
        if not batch_file.exists():
            continue
        df = pd.read_csv(batch_file)
        if "success" not in df.columns or "animal" not in df.columns:
            continue
        valid_df = df[df["success"].isin([1, -1])].copy()
        animals = pd.Series(valid_df["animal"].dropna().unique()).tolist()
        for animal in sorted(animals):
            try:
                animal_id = int(animal)
            except Exception:
                continue
            pkl_path = results_dir / f"results_{batch_name}_animal_{animal_id}.pkl"
            if not pkl_path.exists():
                continue
            try:
                with open(pkl_path, "rb") as handle:
                    results = pickle.load(handle)
            except Exception:
                continue
            if ABORT_KEY in results and MODEL_KEY in results:
                candidates.append((batch_name, animal_id))
    return candidates


def load_pooled_valid_df(candidate_pairs):
    animals_by_batch = {}
    for batch_name, animal_id in candidate_pairs:
        animals_by_batch.setdefault(str(batch_name), set()).add(int(animal_id))

    frames = []
    for batch_name in DESIRED_BATCHES:
        animal_ids = animals_by_batch.get(str(batch_name), set())
        if not animal_ids:
            continue
        batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
        if not batch_file.exists():
            continue
        df = pd.read_csv(batch_file)
        if "batch_name" not in df.columns:
            df["batch_name"] = str(batch_name)
        animal_numeric = pd.to_numeric(df["animal"], errors="coerce")
        df = df[animal_numeric.isin(animal_ids)].copy()
        if TRIAL_POOL_MODE == "valid_plus_abort3":
            df["abort_event"] = pd.to_numeric(df["abort_event"], errors="coerce")
            df = df[df["success"].isin([1, -1]) | np.isclose(df["abort_event"], 3)].copy()
        else:
            df = df[df["success"].isin([1, -1])].copy()
        df["animal"] = pd.to_numeric(df["animal"], errors="coerce")
        df["RTwrtStim"] = pd.to_numeric(df["RTwrtStim"], errors="coerce")
        df["intended_fix"] = pd.to_numeric(df["intended_fix"], errors="coerce")
        df["ABL"] = pd.to_numeric(df["ABL"], errors="coerce")
        df["ILD"] = pd.to_numeric(df["ILD"], errors="coerce")
        df = df[
            (df["intended_fix"] >= intended_fix_min_s)
            & (df["intended_fix"] <= intended_fix_max_s)
            & (df["ABL"].isin(ABL_VALUES))
            & (df["ILD"].isin(ILD_VALUES))
        ].copy()
        frames.append(df)

    if not frames:
        raise ValueError("No pooled trials found.")
    return pd.concat(frames, ignore_index=True)


def get_included_pairs_from_df(df):
    pairs_df = df[["batch_name", "animal"]].dropna().copy()
    pairs_df["batch_name"] = pairs_df["batch_name"].astype(str)
    pairs_df["animal"] = pd.to_numeric(pairs_df["animal"], errors="coerce")
    pairs_df = pairs_df.dropna().drop_duplicates(subset=["batch_name", "animal"]).copy()
    return sorted((str(row.batch_name), int(row.animal)) for row in pairs_df.itertuples(index=False))


def load_animal_params(batch_name, animal_id):
    pkl_path = results_dir / f"results_{batch_name}_animal_{animal_id}.pkl"
    with open(pkl_path, "rb") as handle:
        results = pickle.load(handle)

    abort_blob = results[ABORT_KEY]
    model_blob = results[MODEL_KEY]

    abort_params = {
        "V_A": reduce_param_values(abort_blob["V_A_samples"]),
        "theta_A": reduce_param_values(abort_blob["theta_A_samples"]),
        "t_A_aff": reduce_param_values(abort_blob["t_A_aff_samp"]),
    }
    tied_params = {
        "rate_lambda": reduce_param_values(model_blob["rate_lambda_samples"]),
        "T_0": reduce_param_values(model_blob["T_0_samples"]),
        "theta_E": reduce_param_values(model_blob["theta_E_samples"]),
        "w": reduce_param_values(model_blob["w_samples"]),
        "t_E_aff": reduce_param_values(model_blob["t_E_aff_samples"]),
        "del_go": reduce_param_values(model_blob["del_go_samples"]),
        "rate_norm_l": reduce_param_values(model_blob["rate_norm_l_samples"]),
    }
    return abort_params, tied_params


def compute_aggregate_params(included_pairs):
    unique_pairs = sorted(set((str(b), int(a)) for b, a in included_pairs))
    abort_records = []
    tied_records = []
    for batch_name, animal_id in unique_pairs:
        ap, tp = load_animal_params(batch_name, animal_id)
        abort_records.append(ap)
        tied_records.append(tp)

    abort_params = {k: reduce_param_values([r[k] for r in abort_records]) for k in abort_records[0]}
    tied_params = {k: reduce_param_values([r[k] for r in tied_records]) for k in tied_records[0]}
    return abort_params, tied_params


def add_intended_fix_segments(df):
    if SEGMENT_MODE == "quantile":
        segment_ids, segment_edges = pd.qcut(
            df["intended_fix"],
            q=NUM_INTENDED_FIX_QUANTILE_BINS,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        segmented_df = df.copy()
        segmented_df["intended_fix_segment"] = segment_ids.astype(int)
        return segmented_df, np.asarray(segment_edges, dtype=float)

    segment_edges = np.asarray(FIXED_SEGMENT_EDGES_S, dtype=float)
    cut_edges = segment_edges.copy()
    cut_edges[0] = np.nextafter(cut_edges[0], -np.inf)
    cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
    segment_ids = pd.cut(df["intended_fix"], bins=cut_edges, labels=False, include_lowest=True, right=True)
    segmented_df = df.copy()
    segmented_df["intended_fix_segment"] = segment_ids.astype(int)
    return segmented_df, segment_edges


def build_segment_specs(segment_edges):
    n_segments = len(segment_edges) - 1
    return [
        {
            "index": i,
            "name": f"Q{i + 1}/{n_segments}",
            "left": float(segment_edges[i]),
            "right": float(segment_edges[i + 1]),
        }
        for i in range(n_segments)
    ]


def run_simulations_for_subset(subset_df, abort_params, tied_params, n_sim, desc=""):
    """Sample (ABL, ILD, t_stim) from subset_df and run n_sim simulations."""
    rng = np.random.default_rng(RNG_SEED)
    indices = rng.choice(len(subset_df), size=n_sim, replace=True)
    sampled = subset_df.iloc[indices]
    ABL_samples = sampled["ABL"].values.astype(float)
    ILD_samples = sampled["ILD"].values.astype(float)
    t_stim_samples = sampled["intended_fix"].values.astype(float)

    V_A = abort_params["V_A"]
    theta_A = abort_params["theta_A"]
    t_A_aff = abort_params["t_A_aff"]
    rate_lambda = tied_params["rate_lambda"]
    T_0 = tied_params["T_0"]
    theta_E = tied_params["theta_E"]
    w = tied_params["w"]
    Z_E = (w - 0.5) * 2 * theta_E
    t_E_aff = tied_params["t_E_aff"]
    del_go = tied_params["del_go"]
    rate_norm_l = tied_params["rate_norm_l"]

    print(f"[sim] Starting {n_sim} simulations {desc}", flush=True)
    t0 = time.perf_counter()
    sim_results = Parallel(n_jobs=N_JOBS)(
        delayed(psiam_tied_data_gen_wrapper_rate_norm_fn)(
            V_A, theta_A,
            ABL_samples[i], ILD_samples[i],
            rate_lambda, T_0, theta_E, Z_E,
            t_A_aff, t_E_aff, del_go,
            t_stim_samples[i], rate_norm_l,
            i, N_PRINT, dt,
        )
        for i in tqdm(range(n_sim), desc=desc)
    )
    elapsed = time.perf_counter() - t0
    print(f"[sim] Finished {desc} in {elapsed:.1f}s", flush=True)

    sim_df = pd.DataFrame(sim_results)
    # valid trials: RT after stim onset and RTwrtStim < 1s
    sim_df["RTwrtStim"] = sim_df["rt"] - sim_df["t_stim"]
    sim_df_valid = sim_df[
        (sim_df["RTwrtStim"] > 0) & (sim_df["RTwrtStim"] < rt_max_wrt_stim_s)
    ].copy()
    return sim_df_valid


# %%
def load_data():
    total_start = time.perf_counter()
    print("[progress] load_data() started", flush=True)

    candidate_pairs = get_candidate_animals()
    print(f"[progress] Found {len(candidate_pairs)} eligible batch-animal pairs", flush=True)

    pooled_df = load_pooled_valid_df(candidate_pairs)
    print(f"[progress] Loaded pooled df with {len(pooled_df)} rows", flush=True)

    pooled_df, segment_edges = add_intended_fix_segments(pooled_df)
    segment_specs = build_segment_specs(segment_edges)
    print(f"[progress] Segment edges: {[float(e) for e in segment_edges]}", flush=True)

    included_pairs = get_included_pairs_from_df(pooled_df)
    print(f"[progress] {len(included_pairs)} included animals", flush=True)

    abort_params, tied_params = compute_aggregate_params(included_pairs)
    print("[progress] Aggregate abort params:", abort_params, flush=True)
    print("[progress] Aggregate tied params:", tied_params, flush=True)

    # Run simulations per segment
    segment_sim_results = []
    for seg_spec in segment_specs:
        seg_df = pooled_df[pooled_df["intended_fix_segment"] == seg_spec["index"]].copy()
        print(
            f"[progress] Segment {seg_spec['name']} [{seg_spec['left']:.3f}, {seg_spec['right']:.3f}]s: "
            f"{len(seg_df)} pooled trials",
            flush=True,
        )

        sim_valid = run_simulations_for_subset(
            seg_df, abort_params, tied_params, N_SIM,
            desc=f"seg {seg_spec['name']}",
        )
        print(f"[progress] Segment {seg_spec['name']}: {len(sim_valid)} valid sim trials", flush=True)

        segment_sim_results.append({
            "segment_spec": seg_spec,
            "sim_valid_df": sim_valid,
            "n_pooled_trials": len(seg_df),
        })

    print(f"[progress] load_data() done in {time.perf_counter() - total_start:.1f}s", flush=True)

    return {
        "included_pairs": included_pairs,
        "abort_params": abort_params,
        "tied_params": tied_params,
        "segment_edges": segment_edges,
        "segment_sim_results": segment_sim_results,
    }


data = load_data()

# %%
def plot_data(data):
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_sim_results = data["segment_sim_results"]
    early = segment_sim_results[0]
    late = segment_sim_results[-1]
    early_spec = early["segment_spec"]
    late_spec = late["segment_spec"]

    bins_s = np.arange(0, rt_max_wrt_stim_s + bin_size_s, bin_size_s)

    # --- 1×3 RTD (density histogram) ---
    fig_rtd, axes_rtd = plt.subplots(1, len(ABL_VALUES), figsize=(12.0, 3.8), sharex=True, sharey=True, squeeze=False)

    for col_idx, abl_value in enumerate(ABL_VALUES):
        ax = axes_rtd[0, col_idx]

        early_rt = early["sim_valid_df"].loc[np.isclose(early["sim_valid_df"]["ABL"], abl_value), "RTwrtStim"]
        late_rt = late["sim_valid_df"].loc[np.isclose(late["sim_valid_df"]["ABL"], abl_value), "RTwrtStim"]

        if len(early_rt) > 0:
            ax.hist(early_rt, bins=bins_s, density=True, histtype="step",
                    color="tab:blue", linewidth=1, label=early_spec["name"])
        if len(late_rt) > 0:
            ax.hist(late_rt, bins=bins_s, density=True, histtype="step",
                    color="tab:red", linewidth=1, label=late_spec["name"])

        ax.set_xlim(*xlim_s)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_title(
            f"ABL = {abl_value}\n"
            f"{early_spec['name']}=[{early_spec['left']:.3f}, {early_spec['right']:.3f}]s, "
            f"{late_spec['name']}=[{late_spec['left']:.3f}, {late_spec['right']:.3f}]s"
        )
        ax.set_xlabel("RT wrt stim (s)")
        if col_idx == 0:
            ax.set_ylabel("Density")

    handles, labels = axes_rtd[0, 0].get_legend_handles_labels()
    fig_rtd.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig_rtd.tight_layout(rect=(0, 0, 1, 0.90))

    rtd_path = output_dir / f"sim_rtd_by_abl_{len(data['included_pairs'])}animals"
    fig_rtd.savefig(rtd_path.with_suffix(".pdf"), bbox_inches="tight")
    fig_rtd.savefig(rtd_path.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    print(f"Saved: {rtd_path.with_suffix('.png')}")

    # --- 1×3 CDF ---
    fig_cdf, axes_cdf = plt.subplots(1, len(ABL_VALUES), figsize=(12.0, 3.8), sharex=True, sharey=True, squeeze=False)

    for col_idx, abl_value in enumerate(ABL_VALUES):
        ax = axes_cdf[0, col_idx]

        early_rt = np.sort(early["sim_valid_df"].loc[np.isclose(early["sim_valid_df"]["ABL"], abl_value), "RTwrtStim"].values)
        late_rt = np.sort(late["sim_valid_df"].loc[np.isclose(late["sim_valid_df"]["ABL"], abl_value), "RTwrtStim"].values)

        if len(early_rt) > 0:
            ecdf_y = np.arange(1, len(early_rt) + 1) / len(early_rt)
            ax.plot(early_rt, ecdf_y, color="tab:blue", linewidth=1, label=early_spec["name"])
        if len(late_rt) > 0:
            ecdf_y = np.arange(1, len(late_rt) + 1) / len(late_rt)
            ax.plot(late_rt, ecdf_y, color="tab:red", linewidth=1, label=late_spec["name"])

        ax.set_xlim(*xlim_s)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.set_title(
            f"ABL = {abl_value}\n"
            f"{early_spec['name']}=[{early_spec['left']:.3f}, {early_spec['right']:.3f}]s, "
            f"{late_spec['name']}=[{late_spec['left']:.3f}, {late_spec['right']:.3f}]s"
        )
        ax.set_xlabel("RT wrt stim (s)")
        if col_idx == 0:
            ax.set_ylabel("CDF")

    handles_cdf, labels_cdf = axes_cdf[0, 0].get_legend_handles_labels()
    fig_cdf.legend(handles_cdf, labels_cdf, loc="upper center", ncol=2, frameon=False)
    fig_cdf.tight_layout(rect=(0, 0, 1, 0.90))

    cdf_path = output_dir / f"sim_cdf_by_abl_{len(data['included_pairs'])}animals"
    fig_cdf.savefig(cdf_path.with_suffix(".pdf"), bbox_inches="tight")
    fig_cdf.savefig(cdf_path.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    print(f"Saved: {cdf_path.with_suffix('.png')}")

    # Print summary
    print(f"\nIncluded animals ({len(data['included_pairs'])}):")
    for batch_name, animal_id in data["included_pairs"]:
        print(f"  {batch_name}-{animal_id}")
    print(f"Segment edges: {[float(e) for e in data['segment_edges']]}")
    print("Aggregate abort params:", data["abort_params"])
    print("Aggregate tied params:", data["tied_params"])
    for seg_res in data["segment_sim_results"]:
        spec = seg_res["segment_spec"]
        sv = seg_res["sim_valid_df"]
        print(
            f"Segment {spec['name']} [{spec['left']:.3f}, {spec['right']:.3f}]s: "
            f"{seg_res['n_pooled_trials']} pooled data trials, {len(sv)} valid sim trials"
        )
        for abl in ABL_VALUES:
            n = int(np.isclose(sv["ABL"], abl).sum())
            print(f"  ABL={abl}: {n} valid sim trials")

    return fig_rtd, fig_cdf


fig_rtd, fig_cdf = plot_data(data)

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig_rtd)
    plt.close(fig_cdf)

# %%
