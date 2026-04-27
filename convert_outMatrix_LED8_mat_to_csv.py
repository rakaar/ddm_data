# %%
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


# %%
MAT_PATH = Path("outMatrix_LED8.mat")
CSV_PATH = Path("outMatrix_LED8_converted.csv")
DATA_STRUCT_NAME = "data"


# %%
mat = scipy.io.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
data = mat[DATA_STRUCT_NAME]

rows = []
skipped_fields = []

for field_name in data._fieldnames:
    values = getattr(data, field_name)
    arr = np.asarray(values)

    if arr.size == 0:
        skipped_fields.append((field_name, arr.shape, str(arr.dtype), "empty"))
        continue

    if arr.dtype.names is not None or arr.dtype == object:
        skipped_fields.append((field_name, arr.shape, str(arr.dtype), "non-numeric/object"))
        continue

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    field_df = pd.DataFrame(arr, columns=[f"value_{i}" for i in range(arr.shape[1])])
    field_df.insert(0, "row_index", np.arange(len(field_df)))
    field_df.insert(0, "source_field", field_name)
    rows.append(field_df)

out = pd.concat(rows, ignore_index=True, sort=False)
out.to_csv(CSV_PATH, index=False)


# %%
print(f"Loaded: {MAT_PATH}")
print(f"Saved: {CSV_PATH}")
print(f"Rows: {len(out):,}")
print(f"Columns: {list(out.columns)}")

print("\nRows by source field:")
print(out["source_field"].value_counts(sort=False).to_string())

if skipped_fields:
    print("\nSkipped fields:")
    for field_name, shape, dtype, reason in skipped_fields:
        print(f"- {field_name}: shape={shape}, dtype={dtype}, reason={reason}")
