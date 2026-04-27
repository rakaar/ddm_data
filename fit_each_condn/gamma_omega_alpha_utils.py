# %%
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd


def load_batch_animal_pairs(batch_dir, desired_batches):
    batch_files = [f"batch_{batch_name}_valid_and_aborts.csv" for batch_name in desired_batches]
    dfs = []
    for fname in batch_files:
        fpath = os.path.join(batch_dir, fname)
        if os.path.exists(fpath):
            dfs.append(pd.read_csv(fpath))

    if len(dfs) == 0:
        raise FileNotFoundError(f"No batch CSVs found in {batch_dir}")

    merged_data = pd.concat(dfs, ignore_index=True)
    merged_valid = merged_data[merged_data["success"].isin([1, -1])].copy()
    return sorted(list(map(tuple, merged_valid[["batch_name", "animal"]].drop_duplicates().values)))


def print_batch_animal_table(batch_animal_pairs):
    print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")

    batch_to_animals = defaultdict(list)
    for batch, animal in batch_animal_pairs:
        animal_str = str(animal)
        if animal_str not in batch_to_animals[batch]:
            batch_to_animals[batch].append(animal_str)

    if len(batch_to_animals) == 0:
        return

    max_batch_len = max(len(b) for b in batch_to_animals.keys())
    animal_strings = {b: ", ".join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values())

    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")
    for batch in sorted(animal_strings.keys()):
        print(f"{batch:<{max_batch_len}}  {animal_strings[batch]}")


def get_param_means_by_ABL_ILD(
    batch_name,
    animal_id,
    ABLs_to_fit,
    ILDs_to_fit,
    pkl_folder,
    n_samples=int(1e5),
    param_names=None,
):
    if param_names is None:
        param_names = ["gamma", "omega"]

    param_dict = {}
    missing_files = []
    for ABL in ABLs_to_fit:
        for ILD in ILDs_to_fit:
            pkl_file = os.path.join(
                pkl_folder,
                f"vbmc_cond_by_cond_{batch_name}_{animal_id}_{ABL}_ILD_{ILD}_FIX_t_E_w_del_go_same_as_parametric.pkl",
            )
            if not os.path.exists(pkl_file):
                missing_files.append(pkl_file)
                continue

            with open(pkl_file, "rb") as f:
                vp = pickle.load(f)
            vp_samples = vp.vp.sample(n_samples)[0]
            param_dict[(ABL, ILD)] = {
                name: float(np.mean(vp_samples[:, i])) for i, name in enumerate(param_names)
            }

    return param_dict, missing_files


def build_cond_fit_arrays(batch_animal_pairs, ABLs, ILDs, pkl_folder, n_samples=int(1e5)):
    gamma_all_animals = {str(ABL): np.full((len(batch_animal_pairs), len(ILDs)), np.nan) for ABL in ABLs}
    omega_all_animals = {str(ABL): np.full((len(batch_animal_pairs), len(ILDs)), np.nan) for ABL in ABLs}
    missing_files = []

    for animal_idx, (batch_name, animal_id) in enumerate(batch_animal_pairs):
        print("##########################################")
        print(f"Batch: {batch_name}, Animal: {animal_id}")
        print("##########################################")

        param_dict, missing_for_animal = get_param_means_by_ABL_ILD(
            batch_name,
            animal_id,
            ABLs,
            ILDs,
            pkl_folder,
            n_samples=n_samples,
        )
        missing_files.extend(missing_for_animal)

        for ABL in ABLs:
            for ild_idx, ILD in enumerate(ILDs):
                if (ABL, ILD) in param_dict:
                    gamma_all_animals[str(ABL)][animal_idx, ild_idx] = param_dict[(ABL, ILD)]["gamma"]
                    omega_all_animals[str(ABL)][animal_idx, ild_idx] = param_dict[(ABL, ILD)]["omega"]

    return gamma_all_animals, omega_all_animals, missing_files


def mean_and_sem_by_abl(values_by_abl, ABLs):
    means = {}
    sems = {}
    ns = {}
    for ABL in ABLs:
        arr = values_by_abl[str(ABL)]
        n = np.sum(~np.isnan(arr), axis=0)
        means[str(ABL)] = np.nanmean(arr, axis=0)
        sems[str(ABL)] = np.nanstd(arr, axis=0) / np.sqrt(n)
        ns[str(ABL)] = n
    return means, sems, ns


def levels_from_abl_ild(abl, ild):
    R_dB = abl + ild / 2
    L_dB = abl - ild / 2
    return R_dB, L_dB


def pressure_from_db(db, P_0):
    return P_0 * (10 ** (db / 20))


def firing_rates(P_R, P_L, alpha, rate_lambda, ell, P_0):
    P_R_scaled = P_R / P_0
    P_L_scaled = P_L / P_0

    r_R = (P_R_scaled**rate_lambda) / (
        (P_R_scaled ** (rate_lambda * ell)) + alpha * (P_L_scaled ** (rate_lambda * ell))
    )
    r_L = (P_L_scaled**rate_lambda) / (
        (P_L_scaled ** (rate_lambda * ell)) + alpha * (P_R_scaled ** (rate_lambda * ell))
    )
    return r_R, r_L


def gamma_from_rates(r_R, r_L, theta):
    return theta * (r_R - r_L) / (r_R + r_L)


def omega_from_rates(r_R, r_L, T_0, theta):
    return (r_R + r_L) / (T_0 * theta**2)


def gamma_omega_alpha_model(abl, ild, rate_lambda, ell, alpha, theta, T_0, P_0):
    R_dB, L_dB = levels_from_abl_ild(abl, ild)
    P_R = pressure_from_db(R_dB, P_0)
    P_L = pressure_from_db(L_dB, P_0)
    r_R, r_L = firing_rates(P_R, P_L, alpha, rate_lambda, ell, P_0)
    gamma = gamma_from_rates(r_R, r_L, theta)
    omega = omega_from_rates(r_R, r_L, T_0, theta)
    return gamma, omega


def gamma_alpha_model(abl, ild, rate_lambda, ell, alpha, theta, P_0):
    gamma, _ = gamma_omega_alpha_model(abl, ild, rate_lambda, ell, alpha, theta, 1.0, P_0)
    return gamma


def omega_alpha_model(abl, ild, rate_lambda, ell, alpha, theta, T_0, P_0):
    _, omega = gamma_omega_alpha_model(abl, ild, rate_lambda, ell, alpha, theta, T_0, P_0)
    return omega
