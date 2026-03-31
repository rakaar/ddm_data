# %%
from pathlib import Path

import numpy as np
import pandas as pd

# %%
DESIRED_BATCHES = ["SD", "LED34", "LED6", "LED8", "LED7", "LED34_even"]
PROACTIVE_TRUNC_FIX_TIME_S = {"default": 0.3, "LED34_even": 0.15}


def get_trunc_time(batch_name):
    if PROACTIVE_TRUNC_FIX_TIME_S is None:
        return None
    return PROACTIVE_TRUNC_FIX_TIME_S.get(
        str(batch_name), PROACTIVE_TRUNC_FIX_TIME_S["default"]
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
batch_csv_dir = REPO_ROOT / "fit_animal_by_animal" / "batch_csvs"

# %%
frames = []
for batch_name in DESIRED_BATCHES:
    csv_path = batch_csv_dir / f"batch_{batch_name}_valid_and_aborts.csv"
    if csv_path.exists():
        frames.append(pd.read_csv(csv_path))
    else:
        print(f"Missing: {csv_path.name}")

merged = pd.concat(frames, ignore_index=True)
for col in ["RTwrtStim", "ABL", "ILD", "intended_fix", "abort_event", "TotalFixTime"]:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

# %%
# Select valid + abort3 pool (same logic as the plotting script)
pool = merged[
    merged["success"].isin([1, -1]) | np.isclose(merged["abort_event"], 3)
].copy()

print(f"Total valid + abort3 pool: {len(pool)} rows")
print(f"  Valid trials (success in {{1,-1}}): {merged['success'].isin([1,-1]).sum()}")
print(f"  Abort event 3 trials:              {np.isclose(merged['abort_event'], 3).sum()}")
print()

# %%
# Identify rows that would be truncated
early_abort_mask = pd.Series(False, index=pool.index)
for bn in pool["batch_name"].astype(str).unique():
    tt = get_trunc_time(bn)
    if tt is not None:
        batch_mask = pool["batch_name"].astype(str) == bn
        early_abort_mask |= (
            batch_mask
            & np.isclose(pool["abort_event"], 3)
            & (pool["TotalFixTime"] < tt)
        )

truncated_rows = pool[early_abort_mask]
kept_rows = pool[~early_abort_mask]

print("=" * 70)
print("TRUNCATION SUMMARY")
print("=" * 70)
print(f"Rows BEFORE truncation: {len(pool)}")
print(f"Rows REMOVED by truncation: {len(truncated_rows)}")
print(f"Rows AFTER truncation: {len(kept_rows)}")
print(f"Fraction removed: {len(truncated_rows) / len(pool):.4f}")
print()

# %%
# Breakdown by batch
print("=" * 70)
print("BREAKDOWN BY BATCH")
print("=" * 70)
batch_summary = []
for bn in DESIRED_BATCHES:
    batch_pool = pool[pool["batch_name"].astype(str) == bn]
    batch_removed = truncated_rows[truncated_rows["batch_name"].astype(str) == bn]
    tt = get_trunc_time(bn)
    batch_summary.append({
        "batch": bn,
        "trunc_time_s": tt,
        "pool_total": len(batch_pool),
        "abort3_in_pool": int(np.isclose(batch_pool["abort_event"], 3).sum()),
        "removed": len(batch_removed),
        "kept": len(batch_pool) - len(batch_removed),
        "frac_removed": len(batch_removed) / len(batch_pool) if len(batch_pool) > 0 else 0.0,
    })

summary_df = pd.DataFrame(batch_summary)
print(summary_df.to_string(index=False))
print()

# %%
# Breakdown by batch + animal
print("=" * 70)
print("BREAKDOWN BY BATCH + ANIMAL")
print("=" * 70)
animal_rows = []
for bn in DESIRED_BATCHES:
    batch_pool = pool[pool["batch_name"].astype(str) == bn]
    batch_removed = truncated_rows[truncated_rows["batch_name"].astype(str) == bn]
    for animal in sorted(batch_pool["animal"].astype(str).unique()):
        animal_pool = batch_pool[batch_pool["animal"].astype(str) == animal]
        animal_removed = batch_removed[batch_removed["animal"].astype(str) == animal]
        animal_rows.append({
            "batch": bn,
            "animal": animal,
            "pool": len(animal_pool),
            "abort3": int(np.isclose(animal_pool["abort_event"], 3).sum()),
            "removed": len(animal_removed),
            "kept": len(animal_pool) - len(animal_removed),
            "frac_removed": len(animal_removed) / len(animal_pool) if len(animal_pool) > 0 else 0.0,
        })

animal_df = pd.DataFrame(animal_rows)
print(animal_df.to_string(index=False))
