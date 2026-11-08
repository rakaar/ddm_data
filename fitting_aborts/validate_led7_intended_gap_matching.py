# %%
"""Independent exhaustive and MILP checks of the interval-capacity matcher."""

from pathlib import Path
import itertools
import time
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack
from led7_intended_gap_matching_utils import match_intended_gap, paired_ks


# %%
############ Editable validation parameters ############
SCRIPT_DIR = Path(__file__).resolve().parent
STEM = SCRIPT_DIR / "led7_s7_s8_first_last_abort_intended_gap_matching"
KS_INVERSE = 5
SEED = 20261004
MILP_TIME_LIMIT_S = 60
records = []


# %%
############ Exhaustive independent subset enumeration, including ties and zeros ############
rng = np.random.default_rng(531)
examples = [(np.array([1., 2., 3.]), np.array([1., 2., 3.])),
            (np.array([1., 2., 3., 4., 5.]), np.array([0., 1., 2., 3., 4.])),
            (np.ones(8), np.zeros(8))]
for _ in range(30):
    gap = rng.integers(0, 8, 10).astype(float)
    examples.append((gap + rng.integers(0, 5, 10), gap))
for i, (intended, gap) in enumerate(examples):
    keep, audit = match_intended_gap(intended, gap, KS_INVERSE)
    best = 0
    for subset in itertools.product([False, True], repeat=len(intended)):
        selected = np.asarray(subset)
        n = int(selected.sum())
        if n <= best:
            continue
        points = np.unique(np.r_[intended[selected], gap[selected]])
        overlap = ((gap[selected, None] <= points) & (intended[selected, None] > points)).sum(axis=0).max()
        if KS_INVERSE * overlap <= n:
            best = n
    assert int(keep.sum()) == best
    records.append({"validation": "exhaustive", "group": f"example_{i}", "n": len(intended),
                    "interval_retained": int(keep.sum()), "reference_retained": best, "passed": True})
print(f"{len(examples)} exhaustive examples passed (ties, zero-length, touching and all-overlapping intervals).", flush=True)


# %%
############ Independent sparse mixed-integer formulation, not the sweep algorithm ############
def milp_maximum_count(intended, gap):
    n = len(intended)
    points = np.unique(np.r_[intended, gap])
    m = len(points)
    starts, ends = np.searchsorted(points, gap), np.searchsorted(points, intended)
    j, k = np.arange(n), np.arange(m)
    # Variables: binary kept-row flags, pointwise interval load, total selected n.
    balance = coo_matrix((np.r_[-np.ones(n), np.ones(n), np.ones(m), -np.ones(m - 1)],
                          (np.r_[starts, ends, k, k[1:]], np.r_[j, j, n + k, n + k[:-1]])),
                         shape=(m, n + m + 1)).tocsc()
    upper = coo_matrix((np.r_[np.ones(m), np.full(m, -1 / KS_INVERSE)],
                        (np.r_[k, k], np.r_[n + k, np.full(m, n + m)])), shape=(m, n + m + 1)).tocsc()
    total = coo_matrix((np.r_[np.ones(n), -1.], (np.zeros(n + 1, dtype=int), np.r_[j, n + m])),
                       shape=(1, n + m + 1)).tocsc()
    matrix = vstack([balance, upper, total], format="csc")
    objective = np.zeros(n + m + 1)
    objective[-1] = -1
    started = time.perf_counter()
    solution = milp(objective, integrality=np.r_[np.ones(n), np.zeros(m + 1)],
                    bounds=Bounds(np.zeros(n + m + 1), np.r_[np.ones(n), np.full(m + 1, n)]),
                    constraints=LinearConstraint(matrix, np.r_[np.zeros(m), np.full(m, -np.inf), 0.], np.zeros(2 * m + 1)),
                    options={"mip_rel_gap": 0., "time_limit": MILP_TIME_LIMIT_S})
    assert solution.success and solution.status == 0 and solution.mip_gap == 0, solution.message
    kept = np.rint(solution.x[:n]).astype(bool)
    assert np.isclose(solution.fun, -kept.sum(), rtol=0, atol=1e-7)
    assert np.isclose(solution.mip_dual_bound, solution.fun, rtol=0, atol=1e-7)
    assert paired_ks(intended[kept], gap[kept]) <= 1 / KS_INVERSE + 1e-12
    return int(kept.sum()), float(solution.mip_gap), float(solution.mip_dual_bound), time.perf_counter() - started


# %%
############ All 72 selected-abort groups plus the largest eligible population ############
selected = pd.read_csv(SCRIPT_DIR / "led7_s7_s8_first_last_abort_timing_stimulus_gap_selected_timing_values.csv", float_precision="round_trip")
eligible = pd.read_csv(SCRIPT_DIR / "led7_s7_s8_first_last_abort_timing_all_trials_denominator_trials.csv", float_precision="round_trip")
groups = [(f"selected_{case}_{animal}_{led}", group) for (case, animal, led), group in
          selected.groupby(["case", "animal", "LED_trial"], sort=True)]
largest = eligible.loc[eligible.session_type.eq(7) & eligible.animal.eq(92) & eligible.LED_trial.eq(0)]
groups.append(("largest_eligible_s7_92_OFF", largest))
rng = np.random.default_rng(SEED)
for label, group in groups:
    group = group.sort_values("source_csv_row_index").iloc[rng.permutation(len(group))]
    intended = group.intended_fix.to_numpy()
    gap = intended - group.LED_onset_time.to_numpy()
    keep, audit = match_intended_gap(intended, gap, KS_INVERSE)
    count, mip_gap, dual, elapsed = milp_maximum_count(intended, gap)
    assert int(keep.sum()) == count
    records.append({"validation": "MILP", "group": label, "n": len(group),
                    "interval_retained": int(keep.sum()), "reference_retained": count,
                    "mip_gap": mip_gap, "mip_dual_bound": dual, "elapsed_s": elapsed, "passed": True})
    if len(records) % 12 == 0 or label.startswith("largest"):
        print(f"Verified {label}: maximum retention {count}/{len(group)}.", flush=True)
pd.DataFrame(records).to_csv(f"{STEM}_optimizer_validation.csv", index=False)
print(f"Passed {len(examples)} exhaustive + {len(groups)} certified MILP comparisons.", flush=True)
