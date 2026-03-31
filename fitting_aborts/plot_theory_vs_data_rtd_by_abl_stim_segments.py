# %%
SHOW_PLOT = True

# DESIRED_BATCHES = ["LED7", "LED8"]
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]


MODEL_KEY = "vbmc_norm_tied_results"
ABORT_KEY = "vbmc_aborts_results"
PARAM_REDUCER = "mean"
TRIAL_POOL_MODE = "valid"  # "valid" or "valid_plus_abort3"
NUM_INTENDED_FIX_QUANTILE_BINS = 2
MODEL_DENSITY_MODE = "valid_conditioned"  # "raw" or "valid_conditioned"
THEORY_DATA_MODE = "avg_of_rtds"  # "avg_of_rtds" or "avg_of_params"

if TRIAL_POOL_MODE not in {"valid", "valid_plus_abort3"}:
    raise ValueError(f"Unsupported TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")
if MODEL_DENSITY_MODE not in {"raw", "valid_conditioned"}:
    raise ValueError(f"Unsupported MODEL_DENSITY_MODE: {MODEL_DENSITY_MODE}")

ABL_VALUES = (20, 40, 60)
ILD_VALUES = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)
N_MC_T_STIM_SAMPLES = 1000
RNG_SEED = 12345

intended_fix_min_s = 0.2
rt_min_s = -2.0
rt_max_s = 2.0
model_bin_size_s = 1e-3
data_bin_size_s = 5e-3
intended_fix_max_s = 1.5
xlim_s = (0, 1)
QUANTILE_PERCENTS = (10, 30, 50, 70, 90)
panel_width = 4.5
panel_height = 3.5
png_dpi = 300

# %%
from pathlib import Path
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent
FIT_DIR = REPO_ROOT / "fit_animal_by_animal"
if str(FIT_DIR) not in sys.path:
    sys.path.append(str(FIT_DIR))

from time_vary_norm_utils import (
    rho_A_t_VEC_fn,
    up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec,
)

# %%
batch_csv_dir = FIT_DIR / "batch_csvs"
results_dir = FIT_DIR
output_dir = SCRIPT_DIR / "theory_vs_data_rtd_stim_segments"

output_suffix_parts = []
if TRIAL_POOL_MODE == "valid_plus_abort3":
    output_suffix_parts.append("plus_abort3_pool")
if MODEL_DENSITY_MODE == "valid_conditioned":
    output_suffix_parts.append("valid_conditioned")
if THEORY_DATA_MODE == "avg_of_rtds":
    output_suffix_parts.append("avg_of_rtds")
output_suffix = "".join(f"_{part}" for part in output_suffix_parts)
output_base = output_dir / f"theory_vs_data_rtd_by_abl_stim_segments{output_suffix}"


# ---------------------------------------------------------------------------
# Helper: reduce parameter values
# ---------------------------------------------------------------------------
def reduce_param_values(values):
    values = np.asarray(values, dtype=float)
    if PARAM_REDUCER == "mean":
        return float(np.mean(values))
    if PARAM_REDUCER == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown PARAM_REDUCER: {PARAM_REDUCER}.")


# ---------------------------------------------------------------------------
# Discover eligible (batch, animal) pairs with both abort + model results
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Load pooled valid trials for eligible animals
# ---------------------------------------------------------------------------
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
            if "abort_event" not in df.columns:
                raise ValueError(
                    f"abort_event column is required when TRIAL_POOL_MODE='valid_plus_abort3': {batch_file}"
                )
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
            (df["RTwrtStim"] >= rt_min_s)
            & (df["RTwrtStim"] <= rt_max_s)
            & (df["intended_fix"] >= intended_fix_min_s)
            & (df["intended_fix"] <= intended_fix_max_s)
            & (df["ABL"].isin(ABL_VALUES))
            & (df["ILD"].isin(ILD_VALUES))
        ].copy()
        frames.append(df)

    if not frames:
        raise ValueError("No pooled trials found for eligible animals after filtering.")
    return pd.concat(frames, ignore_index=True)


def get_included_pairs_from_df(df):
    pairs_df = df[["batch_name", "animal"]].dropna().copy()
    pairs_df["batch_name"] = pairs_df["batch_name"].astype(str)
    pairs_df["animal"] = pd.to_numeric(pairs_df["animal"], errors="coerce")
    pairs_df = pairs_df.dropna().copy()
    pairs_df = pairs_df.drop_duplicates(subset=["batch_name", "animal"]).copy()
    return sorted((str(row.batch_name), int(row.animal)) for row in pairs_df.itertuples(index=False))


# ---------------------------------------------------------------------------
# Parameter loading / aggregation
# ---------------------------------------------------------------------------
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
    return abort_params, tied_params, pkl_path


def compute_aggregate_params(included_pairs):
    unique_included_pairs = sorted(set((str(b), int(a)) for b, a in included_pairs))
    abort_records, tied_records, pkl_paths = [], [], []

    for batch_name, animal_id in unique_included_pairs:
        abort_params, tied_params, pkl_path = load_animal_params(batch_name, animal_id)
        abort_records.append(abort_params)
        tied_records.append(tied_params)
        pkl_paths.append(pkl_path)

    abort_params = {
        key: reduce_param_values([r[key] for r in abort_records]) for key in abort_records[0]
    }
    tied_params = {
        key: reduce_param_values([r[key] for r in tied_records]) for key in tied_records[0]
    }
    return abort_params, tied_params, pkl_paths


# ---------------------------------------------------------------------------
# Quantile segmentation
# ---------------------------------------------------------------------------
def add_intended_fix_segments(df):
    if df["intended_fix"].isna().any():
        raise ValueError("Cannot build quantile segments with NaN intended_fix values.")
    if float(df["intended_fix"].nunique()) <= 1:
        raise ValueError("Cannot segment intended_fix because all filtered values are identical.")

    segment_ids, segment_edges = pd.qcut(
        df["intended_fix"],
        q=NUM_INTENDED_FIX_QUANTILE_BINS,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    if len(segment_edges) - 1 != NUM_INTENDED_FIX_QUANTILE_BINS:
        raise ValueError(
            f"Requested {NUM_INTENDED_FIX_QUANTILE_BINS} quantile bins, "
            f"but only {len(segment_edges) - 1} unique bins could be formed."
        )
    if segment_ids.isna().any():
        raise ValueError("Failed to assign intended_fix quantile segments to some trials.")

    segmented_df = df.copy()
    segmented_df["intended_fix_segment"] = segment_ids.astype(int)
    return segmented_df, np.asarray(segment_edges, dtype=float)


def build_segment_specs(segment_edges):
    n_segments = len(segment_edges) - 1
    segment_specs = []
    for idx in range(n_segments):
        segment_specs.append(
            {
                "index": int(idx),
                "name": f"Q{idx + 1}/{n_segments}",
                "left": float(segment_edges[idx]),
                "right": float(segment_edges[idx + 1]),
            }
        )
    return segment_specs


# ---------------------------------------------------------------------------
# Proactive / RTD computation (from model script)
# ---------------------------------------------------------------------------
def compute_segment_proactive_curves(t_stim_samples, abort_params):
    t_pts = np.arange(-1.0, 1.001, 0.001)
    shifted_t = t_pts[None, :] + np.asarray(t_stim_samples, dtype=float)[:, None] - abort_params["t_A_aff"]
    p_a_samples = rho_A_t_VEC_fn(shifted_t, abort_params["V_A"], abort_params["theta_A"])
    p_a_mean = np.mean(p_a_samples, axis=0)
    c_a_mean = cumulative_trapezoid(p_a_mean, t_pts, initial=0)
    return t_pts, p_a_mean, c_a_mean


def compute_segment_proactive_matrices(t_stim_samples, abort_params):
    t_pts = np.arange(-1.0, 1.001, 0.001)
    shifted_t = t_pts[None, :] + np.asarray(t_stim_samples, dtype=float)[:, None] - abort_params["t_A_aff"]
    p_a_samples = rho_A_t_VEC_fn(shifted_t, abort_params["V_A"], abort_params["theta_A"])
    c_a_samples = cumulative_trapezoid(p_a_samples, t_pts, axis=1, initial=0)
    return t_pts, p_a_samples, c_a_samples


def compute_raw_rtd_from_pa_ca(t_pts, p_a, c_a, abl_value, ild_value, tied_params):
    z_e = (tied_params["w"] - 0.5) * 2.0 * tied_params["theta_E"]
    up = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t_pts, 1, p_a, c_a, abl_value, ild_value,
        tied_params["rate_lambda"], tied_params["T_0"], tied_params["theta_E"], z_e,
        tied_params["t_E_aff"], tied_params["del_go"],
        np.nan, np.nan, np.nan, np.nan, np.nan,
        tied_params["rate_norm_l"], True, False, 10,
    )
    down = up_or_down_RTs_fit_PA_C_A_given_wrt_t_stim_fn_vec(
        t_pts, -1, p_a, c_a, abl_value, ild_value,
        tied_params["rate_lambda"], tied_params["T_0"], tied_params["theta_E"], z_e,
        tied_params["t_E_aff"], tied_params["del_go"],
        np.nan, np.nan, np.nan, np.nan, np.nan,
        tied_params["rate_norm_l"], True, False, 10,
    )
    return up + down


def compute_segment_averaged_rtd(t_pts, p_a_mean, c_a_mean, abl_value, ild_value, tied_params):
    density = compute_raw_rtd_from_pa_ca(t_pts, p_a_mean, c_a_mean, abl_value, ild_value, tied_params)
    mask = (t_pts >= rt_min_s) & (t_pts <= rt_max_s)
    return t_pts[mask], density[mask]


def curve_to_binned_density(t_pts, density, bins_s):
    cdf = cumulative_trapezoid(density, t_pts, initial=0)
    edge_cdf = np.interp(bins_s, t_pts, cdf, left=0.0, right=float(cdf[-1]))
    probs = np.diff(edge_cdf)
    widths = np.diff(bins_s)
    return probs / widths


def build_segment_mixture(segment_df, t_pts, p_a_mean, c_a_mean, abl_value, tied_params, bins_s, progress_bar=None):
    abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)].copy()
    counts = abl_df["ILD"].round().astype(int).value_counts().to_dict() if len(abl_df) else {}
    total = int(len(abl_df))
    if total == 0:
        if progress_bar is not None:
            progress_bar.update(len(ILD_VALUES))
        return np.zeros(len(bins_s) - 1, dtype=float), {}, 0.0

    mixture = np.zeros(len(bins_s) - 1, dtype=float)
    weight_map = {}
    raw_area_weighted = 0.0
    valid_mask = t_pts >= 0.0
    for ild_value in ILD_VALUES:
        count = int(counts.get(int(ild_value), 0))
        if count == 0:
            if progress_bar is not None:
                progress_bar.update(1)
            continue
        weight = count / total
        if MODEL_DENSITY_MODE == "valid_conditioned":
            raw_matrix = compute_raw_rtd_from_pa_ca(
                t_pts[None, :], p_a_mean, c_a_mean,
                abl_value, float(ild_value), tied_params,
            )
            if raw_matrix.ndim != 2:
                raise ValueError("Expected a 2D RTD matrix when MODEL_DENSITY_MODE='valid_conditioned'.")
            conditioned_matrix = np.where(valid_mask[None, :], raw_matrix, 0.0)
            valid_area_per_t_stim = np.trapz(conditioned_matrix[:, valid_mask], t_pts[valid_mask], axis=1)
            raw_area_weighted += weight * float(np.mean(valid_area_per_t_stim))
            valid_area_per_t_stim = np.clip(valid_area_per_t_stim, 1e-12, None)
            conditioned_matrix = conditioned_matrix / valid_area_per_t_stim[:, None]
            density = np.mean(conditioned_matrix, axis=0)
            mixture += weight * curve_to_binned_density(t_pts, density, bins_s)
        else:
            rt_axis, density = compute_segment_averaged_rtd(
                t_pts, p_a_mean, c_a_mean, abl_value, float(ild_value), tied_params,
            )
            raw_area_weighted += weight * float(np.trapz(density[density >= 0], rt_axis[density >= 0]))
            mixture += weight * curve_to_binned_density(rt_axis, density, bins_s)
        weight_map[int(ild_value)] = weight
        if progress_bar is not None:
            progress_bar.update(1)
    return mixture, weight_map, raw_area_weighted


# ---------------------------------------------------------------------------
# Data histogram helper
# ---------------------------------------------------------------------------
def compute_density_histogram(values, bins):
    if len(values) == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    hist, _ = np.histogram(values, bins=bins, density=True)
    return hist


# ---------------------------------------------------------------------------
# Load ALL trials (valid + aborts) for computing data valid fraction
# ---------------------------------------------------------------------------
def load_all_trials_df(candidate_pairs):
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
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main load_data
# ---------------------------------------------------------------------------
def load_data():
    total_start = time.perf_counter()
    print("[progress] load_data() started", flush=True)

    stage_start = time.perf_counter()
    candidate_pairs = get_candidate_animals()
    print(
        f"[progress] Found {len(candidate_pairs)} eligible batch-animal pairs "
        f"in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    stage_start = time.perf_counter()
    pooled_valid_df = load_pooled_valid_df(candidate_pairs)
    print(
        f"[progress] Loaded pooled dataframe with {len(pooled_valid_df)} rows "
        f"in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    stage_start = time.perf_counter()
    pooled_valid_df, segment_edges = add_intended_fix_segments(pooled_valid_df)
    segment_specs = build_segment_specs(segment_edges)
    print(
        f"[progress] Built intended_fix quantile segments with edges "
        f"{[float(e) for e in segment_edges]} in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    stage_start = time.perf_counter()
    included_pairs = get_included_pairs_from_df(pooled_valid_df)
    if not included_pairs:
        raise ValueError("No eligible batch-animal pairs remained after filtering.")
    print(
        f"[progress] Retained {len(included_pairs)} included pairs "
        f"in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    stage_start = time.perf_counter()
    abort_params, tied_params, pkl_paths = compute_aggregate_params(included_pairs)
    print(
        f"[progress] Computed aggregate params from {len(pkl_paths)} PKLs "
        f"in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    # --- Load all trials (valid + aborts) for data valid fraction ---
    stage_start = time.perf_counter()
    all_trials_df = load_all_trials_df(candidate_pairs)
    all_trials_df["intended_fix"] = pd.to_numeric(all_trials_df["intended_fix"], errors="coerce")
    # Assign segments using the same edges from valid trials
    segment_cut_edges = np.asarray(segment_edges, dtype=float).copy()
    segment_cut_edges[0] = np.nextafter(segment_cut_edges[0], -np.inf)
    segment_cut_edges[-1] = np.nextafter(segment_cut_edges[-1], np.inf)
    all_trials_df["intended_fix_segment"] = pd.cut(
        all_trials_df["intended_fix"].clip(
            lower=float(segment_edges[0]), upper=float(segment_edges[-1])
        ),
        bins=segment_cut_edges,
        include_lowest=True,
        labels=False,
    ).astype(float)
    all_trials_df = all_trials_df.dropna(subset=["intended_fix_segment"]).copy()
    all_trials_df["intended_fix_segment"] = all_trials_df["intended_fix_segment"].astype(int)
    all_trials_df["is_valid"] = all_trials_df["success"].isin([1, -1]).astype(int)
    print(
        f"[progress] Loaded {len(all_trials_df)} all trials (valid+abort) for data valid fraction "
        f"in {time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    # --- RTD computation (segments + overall) ---
    model_bins_s = np.arange(rt_min_s, rt_max_s + model_bin_size_s, model_bin_size_s)
    data_bins_s = np.arange(rt_min_s, rt_max_s + data_bin_size_s, data_bin_size_s)
    rng = np.random.default_rng(RNG_SEED)

    if THEORY_DATA_MODE == "avg_of_rtds":
        # ===============================================================
        # Per-animal approach: compute RTDs per animal, then average
        # (consistent across all 3 rows)
        # ===============================================================
        ref_t_pts = np.arange(-2.0, 2.001, 0.001)
        ref_norm_mask = (ref_t_pts >= -0.1) & (ref_t_pts <= 1.0)
        n_theory_samples = 1000

        # Accumulators per segment and overall, per ABL
        seg_theory = [{int(a): [] for a in ABL_VALUES} for _ in segment_specs]
        seg_data = [{int(a): [] for a in ABL_VALUES} for _ in segment_specs]
        seg_raw_areas = [{int(a): [] for a in ABL_VALUES} for _ in segment_specs]
        ovr_theory = {int(a): [] for a in ABL_VALUES}
        ovr_data = {int(a): [] for a in ABL_VALUES}
        ovr_raw_areas = {int(a): [] for a in ABL_VALUES}
        ovr_data_quantiles = {int(a): [] for a in ABL_VALUES}
        ovr_theory_quantiles = {int(a): [] for a in ABL_VALUES}

        for pair_idx, (batch_name, animal_id) in enumerate(included_pairs):
            animal_start = time.perf_counter()
            print(
                f"[progress] Per-animal RTDs: {pair_idx + 1}/{len(included_pairs)} "
                f"({batch_name}-{animal_id})",
                flush=True,
            )

            abort_params_i, tied_params_i, _ = load_animal_params(batch_name, animal_id)

            # Load ALL trials for this animal
            batch_file = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
            adf = pd.read_csv(batch_file)
            adf["animal"] = pd.to_numeric(adf["animal"], errors="coerce")
            adf = adf[adf["animal"] == int(animal_id)].copy()
            for col in ("ABL", "ILD", "RTwrtStim", "intended_fix"):
                adf[col] = pd.to_numeric(adf[col], errors="coerce")

            rng_a = np.random.default_rng(RNG_SEED + int(animal_id))

            def _compute_rtds_for_df(sub_df, accum_theory, accum_data, accum_areas=None):
                """Compute per-(ABL, ILD) theory+data RTDs for one animal sub-dataframe."""
                if_vals = sub_df["intended_fix"].dropna().values
                if len(if_vals) < 10:
                    return None
                t_stim_samp = rng_a.choice(if_vals, size=n_theory_samples, replace=True)
                shifted = ref_t_pts[None, :] + t_stim_samp[:, None] - abort_params_i["t_A_aff"]
                pa_s = rho_A_t_VEC_fn(shifted, abort_params_i["V_A"], abort_params_i["theta_A"])
                pa_m = np.mean(pa_s, axis=0)
                ca_m = cumulative_trapezoid(pa_m, ref_t_pts, initial=0)

                for abl_value in ABL_VALUES:
                    for ild_value in ILD_VALUES:
                        raw_rtd = compute_raw_rtd_from_pa_ca(
                            ref_t_pts, pa_m, ca_m,
                            abl_value, float(ild_value), tied_params_i,
                        )
                        area = float(np.trapz(raw_rtd[ref_norm_mask], ref_t_pts[ref_norm_mask]))
                        if accum_areas is not None:
                            accum_areas[int(abl_value)].append(area)
                        if area > 1e-12:
                            norm_rtd = raw_rtd / area
                        else:
                            norm_rtd = raw_rtd
                        accum_theory[int(abl_value)].append(
                            curve_to_binned_density(ref_t_pts, norm_rtd, model_bins_s)
                        )

                        cond = sub_df[
                            np.isclose(sub_df["ABL"], abl_value)
                            & np.isclose(sub_df["ILD"], float(ild_value))
                            & (sub_df["RTwrtStim"] >= -0.1)
                            & (sub_df["RTwrtStim"] <= 1.0)
                        ]
                        if len(cond) > 0:
                            h, _ = np.histogram(
                                cond["RTwrtStim"].values, bins=data_bins_s, density=True
                            )
                            accum_data[int(abl_value)].append(h)

                return pa_m, ca_m

            # --- Segments ---
            for seg_idx, seg_spec in enumerate(segment_specs):
                seg_adf = adf[
                    (adf["intended_fix"] >= seg_spec["left"])
                    & (adf["intended_fix"] <= seg_spec["right"])
                ]
                _compute_rtds_for_df(
                    seg_adf, seg_theory[seg_idx], seg_data[seg_idx], seg_raw_areas[seg_idx]
                )

            # --- Overall ---
            ovr_proactive = _compute_rtds_for_df(adf, ovr_theory, ovr_data, ovr_raw_areas)

            # --- Per-animal quantiles (overall) ---
            if ovr_proactive is not None:
                pa_m_ovr, ca_m_ovr = ovr_proactive
                valid_adf = adf[
                    adf["success"].isin([1, -1])
                    & (adf["RTwrtStim"] >= rt_min_s)
                    & (adf["RTwrtStim"] <= rt_max_s)
                    & (adf["intended_fix"] >= intended_fix_min_s)
                    & (adf["intended_fix"] <= intended_fix_max_s)
                    & (adf["ABL"].isin(ABL_VALUES))
                    & (adf["ILD"].isin(ILD_VALUES))
                ]
                for abl_value in ABL_VALUES:
                    a = int(abl_value)
                    rt_abl = valid_adf.loc[
                        np.isclose(valid_adf["ABL"], abl_value), "RTwrtStim"
                    ].dropna().values
                    rt_abl = rt_abl[(rt_abl >= 0) & (rt_abl <= 1.0)]
                    if len(rt_abl) >= 5:
                        ovr_data_quantiles[a].append(
                            np.percentile(rt_abl, QUANTILE_PERCENTS)
                        )
                    # Theory quantiles: ILD-weighted mixed RTD
                    ild_rtds, ild_counts = [], []
                    for ild_value in ILD_VALUES:
                        n_ild = int(np.sum(
                            np.isclose(valid_adf["ABL"], abl_value)
                            & np.isclose(valid_adf["ILD"], float(ild_value))
                        ))
                        if n_ild == 0:
                            continue
                        raw_rtd = compute_raw_rtd_from_pa_ca(
                            ref_t_pts, pa_m_ovr, ca_m_ovr,
                            abl_value, float(ild_value), tied_params_i,
                        )
                        ild_rtds.append(raw_rtd)
                        ild_counts.append(n_ild)
                    if ild_rtds:
                        wts = np.array(ild_counts, dtype=float)
                        wts /= wts.sum()
                        mixed_rtd = sum(ww * r for ww, r in zip(wts, ild_rtds))
                        area = float(np.trapz(mixed_rtd[ref_norm_mask], ref_t_pts[ref_norm_mask]))
                        if area > 1e-12:
                            mixed_rtd = mixed_rtd / area
                        cdf = cumulative_trapezoid(
                            mixed_rtd[ref_norm_mask], ref_t_pts[ref_norm_mask], initial=0
                        )
                        if cdf[-1] > 0:
                            cdf = cdf / cdf[-1]
                        ovr_theory_quantiles[a].append(
                            np.interp(
                                np.array(QUANTILE_PERCENTS) / 100.0,
                                cdf, ref_t_pts[ref_norm_mask],
                            )
                        )

            print(
                f"[progress]   done in {time.perf_counter() - animal_start:.2f}s",
                flush=True,
            )

        # --- Aggregate segment results ---
        segment_results = []
        for seg_idx, seg_spec in enumerate(segment_specs):
            theory_by_abl, data_by_abl = {}, {}
            counts_by_abl, theory_vf_by_abl, data_vf_by_abl = {}, {}, {}
            for abl_value in ABL_VALUES:
                a = int(abl_value)
                theory_by_abl[a] = (
                    np.nanmean(seg_theory[seg_idx][a], axis=0)
                    if seg_theory[seg_idx][a]
                    else np.zeros(len(model_bins_s) - 1)
                )
                data_by_abl[a] = (
                    np.nanmean(seg_data[seg_idx][a], axis=0)
                    if seg_data[seg_idx][a]
                    else np.zeros(len(data_bins_s) - 1)
                )
                theory_vf_by_abl[a] = (
                    float(np.mean(seg_raw_areas[seg_idx][a]))
                    if seg_raw_areas[seg_idx][a]
                    else 0.0
                )
                seg_df = pooled_valid_df[pooled_valid_df["intended_fix_segment"] == seg_spec["index"]]
                counts_by_abl[a] = int(len(seg_df[np.isclose(seg_df["ABL"], abl_value)]))
                all_seg_abl = all_trials_df[
                    (all_trials_df["intended_fix_segment"] == seg_spec["index"])
                    & np.isclose(all_trials_df["ABL"], abl_value)
                ]
                n_all = int(len(all_seg_abl))
                n_valid = int(all_seg_abl["is_valid"].sum())
                data_vf_by_abl[a] = n_valid / n_all if n_all > 0 else 0.0

            segment_results.append({
                "segment_spec": seg_spec,
                "theory_densities_by_abl": theory_by_abl,
                "data_densities_by_abl": data_by_abl,
                "data_counts_by_abl": counts_by_abl,
                "weights_by_abl": {},
                "theory_valid_frac_by_abl": theory_vf_by_abl,
                "data_valid_frac_by_abl": data_vf_by_abl,
                "total": int(len(pooled_valid_df[
                    pooled_valid_df["intended_fix_segment"] == seg_spec["index"]
                ])),
                "t_stim_samples": np.array([]),
            })

        # --- Aggregate overall ---
        ovr_theory_abl, ovr_data_abl, ovr_counts, ovr_data_vf = {}, {}, {}, {}
        for abl_value in ABL_VALUES:
            a = int(abl_value)
            ovr_theory_abl[a] = (
                np.nanmean(ovr_theory[a], axis=0) if ovr_theory[a]
                else np.zeros(len(model_bins_s) - 1)
            )
            ovr_data_abl[a] = (
                np.nanmean(ovr_data[a], axis=0) if ovr_data[a]
                else np.zeros(len(data_bins_s) - 1)
            )
            ovr_counts[a] = int(len(pooled_valid_df[np.isclose(pooled_valid_df["ABL"], abl_value)]))
            all_abl = all_trials_df[np.isclose(all_trials_df["ABL"], abl_value)]
            n_all, n_valid = int(len(all_abl)), int(all_abl["is_valid"].sum())
            ovr_data_vf[a] = n_valid / n_all if n_all > 0 else 0.0

        overall_result = {
            "theory_densities_by_abl": ovr_theory_abl,
            "data_densities_by_abl": ovr_data_abl,
            "data_counts_by_abl": ovr_counts,
            "weights_by_abl": {},
            "theory_valid_frac_by_abl": {
                int(a): float(np.mean(ovr_raw_areas[int(a)])) if ovr_raw_areas[int(a)] else 0.0
                for a in ABL_VALUES
            },
            "data_valid_frac_by_abl": ovr_data_vf,
            "total": int(len(pooled_valid_df)),
            "t_stim_samples": np.array([]),
            "data_quantiles_by_abl": {
                int(a): np.array(ovr_data_quantiles[int(a)])
                if ovr_data_quantiles[int(a)] else np.empty((0, len(QUANTILE_PERCENTS)))
                for a in ABL_VALUES
            },
            "theory_quantiles_by_abl": {
                int(a): np.array(ovr_theory_quantiles[int(a)])
                if ovr_theory_quantiles[int(a)] else np.empty((0, len(QUANTILE_PERCENTS)))
                for a in ABL_VALUES
            },
        }

    else:
        # ===============================================================
        # Aggregate approach: RTD from mean-of-params
        # (consistent across all 3 rows)
        # ===============================================================
        total_steps = len(segment_specs) * (1 + len(ABL_VALUES) * len(ILD_VALUES))

        if tqdm is not None:
            progress_bar = tqdm(total=total_steps, desc="Building theory RTDs", unit="step")
        else:
            progress_bar = None

        segment_results = []
        try:
            for segment_spec in segment_specs:
                segment_df = pooled_valid_df[
                    pooled_valid_df["intended_fix_segment"] == segment_spec["index"]
                ].copy()
                if len(segment_df) == 0:
                    raise ValueError(
                        f"No pooled valid trials in segment "
                        f"[{segment_spec['left']}, {segment_spec['right']}] s."
                    )

                print(
                    f"[progress] Starting segment {segment_spec['name']} "
                    f"[{segment_spec['left']:.3f}, {segment_spec['right']:.3f}] s "
                    f"with {N_MC_T_STIM_SAMPLES} MC intended_fix samples",
                    flush=True,
                )
                segment_start = time.perf_counter()
                t_stim_samples = rng.choice(
                    segment_df["intended_fix"].to_numpy(),
                    size=N_MC_T_STIM_SAMPLES, replace=True,
                )

                if MODEL_DENSITY_MODE == "valid_conditioned":
                    t_pts, p_a_mean, c_a_mean = compute_segment_proactive_matrices(
                        t_stim_samples, abort_params
                    )
                else:
                    t_pts, p_a_mean, c_a_mean = compute_segment_proactive_curves(
                        t_stim_samples, abort_params
                    )
                if progress_bar is not None:
                    progress_bar.update(1)

                theory_densities_by_abl = {}
                data_densities_by_abl = {}
                data_counts_by_abl = {}
                weights_by_abl = {}
                theory_valid_frac_by_abl = {}
                data_valid_frac_by_abl = {}

                for abl_value in ABL_VALUES:
                    theory_density, weight_map, theory_raw_area = build_segment_mixture(
                        segment_df, t_pts, p_a_mean, c_a_mean,
                        abl_value, tied_params, model_bins_s,
                        progress_bar=progress_bar,
                    )
                    theory_densities_by_abl[int(abl_value)] = theory_density
                    weights_by_abl[int(abl_value)] = weight_map
                    theory_valid_frac_by_abl[int(abl_value)] = theory_raw_area

                    abl_df = segment_df[np.isclose(segment_df["ABL"], abl_value)]
                    data_densities_by_abl[int(abl_value)] = compute_density_histogram(
                        abl_df["RTwrtStim"].to_numpy(), data_bins_s
                    )
                    data_counts_by_abl[int(abl_value)] = int(len(abl_df))

                    all_seg_abl = all_trials_df[
                        (all_trials_df["intended_fix_segment"] == segment_spec["index"])
                        & np.isclose(all_trials_df["ABL"], abl_value)
                    ]
                    n_all = int(len(all_seg_abl))
                    n_valid = int(all_seg_abl["is_valid"].sum())
                    data_valid_frac_by_abl[int(abl_value)] = (
                        n_valid / n_all if n_all > 0 else 0.0
                    )

                    print(
                        f"[progress] Finished segment {segment_spec['name']} ABL={abl_value}",
                        flush=True,
                    )

                segment_results.append({
                    "segment_spec": segment_spec,
                    "theory_densities_by_abl": theory_densities_by_abl,
                    "data_densities_by_abl": data_densities_by_abl,
                    "data_counts_by_abl": data_counts_by_abl,
                    "weights_by_abl": weights_by_abl,
                    "theory_valid_frac_by_abl": theory_valid_frac_by_abl,
                    "data_valid_frac_by_abl": data_valid_frac_by_abl,
                    "total": int(len(segment_df)),
                    "t_stim_samples": t_stim_samples,
                })
                print(
                    f"[progress] Finished segment {segment_spec['name']} total "
                    f"in {time.perf_counter() - segment_start:.2f}s",
                    flush=True,
                )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        # --- Overall (aggregate approach) ---
        print("[progress] Computing overall (aggregate approach)...", flush=True)
        overall_start = time.perf_counter()
        overall_t_stim = rng.choice(
            pooled_valid_df["intended_fix"].to_numpy(),
            size=N_MC_T_STIM_SAMPLES, replace=True,
        )
        if MODEL_DENSITY_MODE == "valid_conditioned":
            ovr_t_pts, ovr_pa, ovr_ca = compute_segment_proactive_matrices(
                overall_t_stim, abort_params
            )
        else:
            ovr_t_pts, ovr_pa, ovr_ca = compute_segment_proactive_curves(
                overall_t_stim, abort_params
            )

        ovr_theory_abl, ovr_data_abl, ovr_counts = {}, {}, {}
        ovr_weights, ovr_theory_vf, ovr_data_vf = {}, {}, {}
        for abl_value in ABL_VALUES:
            theory_density, weight_map, theory_raw_area = build_segment_mixture(
                pooled_valid_df, ovr_t_pts, ovr_pa, ovr_ca,
                abl_value, tied_params, model_bins_s,
            )
            ovr_theory_abl[int(abl_value)] = theory_density
            ovr_weights[int(abl_value)] = weight_map
            ovr_theory_vf[int(abl_value)] = theory_raw_area

            abl_df = pooled_valid_df[np.isclose(pooled_valid_df["ABL"], abl_value)]
            ovr_data_abl[int(abl_value)] = compute_density_histogram(
                abl_df["RTwrtStim"].to_numpy(), data_bins_s
            )
            ovr_counts[int(abl_value)] = int(len(abl_df))

            all_abl = all_trials_df[np.isclose(all_trials_df["ABL"], abl_value)]
            n_all, n_valid = int(len(all_abl)), int(all_abl["is_valid"].sum())
            ovr_data_vf[int(abl_value)] = n_valid / n_all if n_all > 0 else 0.0

        # --- Per-animal data quantiles (overall, avg_of_params) ---
        ovr_data_quantiles = {int(a): [] for a in ABL_VALUES}
        for batch_name_q, animal_id_q in included_pairs:
            animal_df = pooled_valid_df[
                (pooled_valid_df["batch_name"].astype(str) == str(batch_name_q))
                & (np.isclose(pooled_valid_df["animal"], int(animal_id_q)))
            ]
            for abl_value in ABL_VALUES:
                a = int(abl_value)
                rt_abl = animal_df.loc[
                    np.isclose(animal_df["ABL"], abl_value), "RTwrtStim"
                ].dropna().values
                rt_abl = rt_abl[(rt_abl >= 0) & (rt_abl <= 1.0)]
                if len(rt_abl) >= 5:
                    ovr_data_quantiles[a].append(
                        np.percentile(rt_abl, QUANTILE_PERCENTS)
                    )

        # Theory quantiles from aggregate binned density
        ovr_theory_quantiles = {int(a): [] for a in ABL_VALUES}
        for abl_value in ABL_VALUES:
            a = int(abl_value)
            td = ovr_theory_abl[a]
            bin_widths = np.diff(model_bins_s)
            bin_centers = 0.5 * (model_bins_s[:-1] + model_bins_s[1:])
            cdf_th = np.cumsum(td * bin_widths)
            if cdf_th[-1] > 0:
                cdf_th = cdf_th / cdf_th[-1]
            ovr_theory_quantiles[a].append(
                np.interp(np.array(QUANTILE_PERCENTS) / 100.0, cdf_th, bin_centers)
            )

        overall_result = {
            "theory_densities_by_abl": ovr_theory_abl,
            "data_densities_by_abl": ovr_data_abl,
            "data_counts_by_abl": ovr_counts,
            "weights_by_abl": ovr_weights,
            "theory_valid_frac_by_abl": ovr_theory_vf,
            "data_valid_frac_by_abl": ovr_data_vf,
            "total": int(len(pooled_valid_df)),
            "t_stim_samples": overall_t_stim,
            "data_quantiles_by_abl": {
                int(a): np.array(ovr_data_quantiles[int(a)])
                if ovr_data_quantiles[int(a)] else np.empty((0, len(QUANTILE_PERCENTS)))
                for a in ABL_VALUES
            },
            "theory_quantiles_by_abl": {
                int(a): np.array(ovr_theory_quantiles[int(a)])
                if ovr_theory_quantiles[int(a)] else np.empty((0, len(QUANTILE_PERCENTS)))
                for a in ABL_VALUES
            },
        }
        print(
            f"[progress] Overall (aggregate) done in "
            f"{time.perf_counter() - overall_start:.2f}s",
            flush=True,
        )

    print(f"[progress] load_data() finished in {time.perf_counter() - total_start:.2f}s", flush=True)

    return {
        "included_pairs": included_pairs,
        "valid_df": pooled_valid_df,
        "abort_params": abort_params,
        "tied_params": tied_params,
        "pkl_paths": pkl_paths,
        "model_bins_s": model_bins_s,
        "data_bins_s": data_bins_s,
        "segment_edges": segment_edges,
        "segment_results": segment_results,
        "overall_result": overall_result,
    }


data = load_data()

# %%
def save_figure(fig, output_base_path):
    fig.savefig(output_base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base_path.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def plot_data(data):
    output_dir.mkdir(parents=True, exist_ok=True)

    n_segments = len(data["segment_results"])
    n_rows = n_segments + 2  # +1 overall, +1 quantiles
    fig, axes = plt.subplots(
        n_rows,
        len(ABL_VALUES),
        figsize=(panel_width * len(ABL_VALUES), panel_height * n_rows),
        squeeze=False,
    )

    model_x_edges = data["model_bins_s"]
    data_x_edges = data["data_bins_s"]

    # Compute shared y-limits across all panels (segments + overall)
    visible_mask_model = (model_x_edges[:-1] >= xlim_s[0]) & (model_x_edges[1:] <= xlim_s[1])
    visible_mask_data = (data_x_edges[:-1] >= xlim_s[0]) & (data_x_edges[1:] <= xlim_s[1])
    all_results = list(data["segment_results"]) + [data["overall_result"]]
    global_max = 0.0
    for seg_result in all_results:
        for abl_value in ABL_VALUES:
            theory_d = seg_result["theory_densities_by_abl"][int(abl_value)]
            data_d = seg_result["data_densities_by_abl"][int(abl_value)]
            if np.any(np.isfinite(theory_d)):
                global_max = max(global_max, float(np.nanmax(theory_d[visible_mask_model])))
            if np.any(np.isfinite(data_d)):
                global_max = max(global_max, float(np.nanmax(data_d[visible_mask_data])))
    y_max = 1.05 * global_max if global_max > 0 else 1.0

    for row_idx, seg_result in enumerate(data["segment_results"]):
        segment_spec = seg_result["segment_spec"]
        for col_idx, abl_value in enumerate(ABL_VALUES):
            ax = axes[row_idx, col_idx]

            # Data histogram (step, unfilled)
            data_density = seg_result["data_densities_by_abl"][int(abl_value)]
            ax.stairs(
                data_density,
                data_x_edges,
                color="tab:blue",
                linewidth=1.2,
                label="Data",
            )

            # Theory curve
            theory_density = seg_result["theory_densities_by_abl"][int(abl_value)]
            ax.stairs(
                theory_density,
                model_x_edges,
                color="tab:red",
                linewidth=1.5,
                label="Theory",
            )

            ax.set_xlim(*xlim_s)
            ax.set_ylim(0, y_max)
            ax.grid(alpha=0.2, linewidth=0.6)

            n_data = seg_result["data_counts_by_abl"][int(abl_value)]
            data_vf = seg_result["data_valid_frac_by_abl"][int(abl_value)]
            theory_vf = seg_result["theory_valid_frac_by_abl"][int(abl_value)]
            if n_segments == 2:
                seg_name = "Early stim" if row_idx == 0 else "Late stim"
            else:
                seg_name = segment_spec["name"]
            ax.set_title(
                f"ABL = {abl_value}, {seg_name}\n"
                f"[{segment_spec['left']:.3f}, {segment_spec['right']:.3f}]s, n={n_data}\n"
                f"data valid frac={data_vf:.3f}, theory valid frac={theory_vf:.3f}"
            )

            if col_idx == 0:
                ax.set_ylabel("Density")

    # --- 3rd row: overall (all segments pooled) ---
    overall = data["overall_result"]
    for col_idx, abl_value in enumerate(ABL_VALUES):
        ax = axes[n_segments, col_idx]

        data_density = overall["data_densities_by_abl"][int(abl_value)]
        ax.stairs(data_density, data_x_edges, color="tab:blue", linewidth=1.2, label="Data")

        theory_density = overall["theory_densities_by_abl"][int(abl_value)]
        ax.stairs(theory_density, model_x_edges, color="tab:red", linewidth=1.5, label="Theory")

        ax.set_xlim(*xlim_s)
        ax.set_ylim(0, y_max)
        ax.grid(alpha=0.2, linewidth=0.6)

        n_data = overall["data_counts_by_abl"][int(abl_value)]
        data_vf = overall["data_valid_frac_by_abl"][int(abl_value)]
        theory_vf = overall["theory_valid_frac_by_abl"][int(abl_value)]
        ax.set_title(
            f"ABL = {abl_value}, All stim\nn={n_data}\n"
            f"data valid frac={data_vf:.3f}, theory valid frac={theory_vf:.3f}"
        )
        if col_idx == 0:
            ax.set_ylabel("Density")

    # --- 4th row: RT quantiles (mean ± std across animals) ---
    for col_idx, abl_value in enumerate(ABL_VALUES):
        ax = axes[n_segments + 1, col_idx]
        a = int(abl_value)
        data_q = overall["data_quantiles_by_abl"][a]
        theory_q = overall["theory_quantiles_by_abl"][a]

        if len(data_q) > 0:
            data_mean = np.mean(data_q, axis=0)
            data_std = np.std(data_q, axis=0)
            ax.errorbar(
                QUANTILE_PERCENTS, data_mean, yerr=data_std,
                color="tab:blue", marker="o", markersize=5,
                capsize=0, linewidth=1.2, label="Data",
            )
        if len(theory_q) > 0:
            theory_mean = np.mean(theory_q, axis=0)
            theory_std = (
                np.std(theory_q, axis=0) if len(theory_q) > 1
                else np.zeros_like(theory_mean)
            )
            ax.errorbar(
                QUANTILE_PERCENTS, theory_mean, yerr=theory_std,
                color="tab:red", marker="s", markersize=5,
                capsize=0, linewidth=1.5, label="Theory",
            )

        ax.set_title(f"ABL = {abl_value}, RT Quantiles")
        ax.set_xlabel("Percentile")
        ax.grid(alpha=0.2, linewidth=0.6)
        if col_idx == 0:
            ax.set_ylabel("RT wrt stim (s)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    tagged_output = output_dir / f"{output_base.name}_{len(data['included_pairs'])}animals"
    save_figure(fig, tagged_output)

    # Print summary
    print(f"Included animals ({len(data['included_pairs'])}):")
    for batch_name, animal_id in data["included_pairs"]:
        print(f"  {batch_name}-{animal_id}")
    print(f"TRIAL_POOL_MODE: {TRIAL_POOL_MODE}")
    print(f"MODEL_DENSITY_MODE: {MODEL_DENSITY_MODE}")
    print(f"THEORY_DATA_MODE: {THEORY_DATA_MODE}")
    print(f"Filtered pooled trials: {len(data['valid_df'])}")
    print(f"Segment edges (s): {[float(e) for e in data['segment_edges']]}")
    print(f"PKLs used: {len(data['pkl_paths'])}")
    print("Aggregate norm-tied parameter means:")
    for key, value in data["tied_params"].items():
        print(f"  {key}: {value}")
    print("Aggregate abort parameter means:")
    for key, value in data["abort_params"].items():
        print(f"  {key}: {value}")
    for seg_result in data["segment_results"]:
        ss = seg_result["segment_spec"]
        samples = seg_result["t_stim_samples"]
        if len(samples) > 0:
            print(
                f"Segment {ss['name']} [{ss['left']:.3f}, {ss['right']:.3f}] s, "
                f"MC intended_fix n={len(samples)}, mean={np.mean(samples):.3f}"
            )
        else:
            print(
                f"Segment {ss['name']} [{ss['left']:.3f}, {ss['right']:.3f}] s "
                f"(per-animal mode)"
            )
        for abl_value in ABL_VALUES:
            weights = seg_result["weights_by_abl"].get(int(abl_value), {})
            print(
                f"  ABL={abl_value}, data n={seg_result['data_counts_by_abl'][int(abl_value)]}"
                + (f", ILD weights={weights}" if weights else "")
            )
    print(f"Saved: {tagged_output.with_suffix('.pdf')}")
    print(f"Saved: {tagged_output.with_suffix('.png')}")

    return fig


fig = plot_data(data)

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)

# %%
