"""Extract simultaneous RT trials from the saved visual-task MATLAB table.

Run with .venv/bin/python ASD_Human_Data/prepare_simultaneous_rt_trials.py.
"""

# %% Paths and analysis definition
from io import BytesIO
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "table_visualTask_15nov2026.mat"
OUTPUT = HERE / "simultaneous_rt_trials.csv"
SUMMARY = HERE / "simultaneous_rt_extraction.json"

# The first analysis uses Condition == "RT" only. SD also has simultaneous
# left/right stimuli, but its 30 ms presentation is a separate condition.
CONDITION = "RT"


# %% Read the MATLAB table's MCOS workspace
# scipy.io.loadmat sees a MATLAB table as an opaque object. The table's saved
# column arrays are present in its function workspace, which has a four-byte
# version marker followed by four padding bytes and a MAT matrix. Read its
# first matrix with SciPy's low-level reader to retain the MCOS cell contents.
workspace_bytes = loadmat(SOURCE)["__function_workspace__"].tobytes()
mat_header = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\0" * 8
workspace_stream = BytesIO(
    mat_header + workspace_bytes[:4] + workspace_bytes[8:]
)
reader = MatFile5Reader(workspace_stream)
reader.initialize_read()
reader.read_file_header()
header, _ = reader.read_var_header()
wrapper = reader.read_var_array(header)["MCOS"][0, 0][0]
assert wrapper["s2"] == b"FileWrapper__"
cells = list(wrapper["arr"].flat)

variable_names = [value.item() for value in cells[10].flat]
column_arrays = list(cells[5].flat)
n_rows = int(cells[7].item())
assert len(variable_names) == len(column_arrays) == int(cells[9].item())

# Group is a MATLAB categorical; Condition is a MATLAB string array. Its
# encoded data stores one two-character label per row (RT, SD, or CB).
group_names = [value.item() for value in cells[2].flat]
group_codes = cells[3].ravel()
assert len(group_codes) == n_rows
assert np.isin(group_codes, np.arange(1, len(group_names) + 1)).all()

encoded_condition = cells[4].ravel()
assert encoded_condition[:4].tolist() == [1, 2, n_rows, 1]
assert np.all(encoded_condition[4 : 4 + n_rows] == 2)
condition_chars = encoded_condition[4 + n_rows :].tobytes().decode("utf-16le")
assert len(condition_chars) == 2 * n_rows

data = {
    name: values.ravel()
    for name, values in zip(variable_names, column_arrays)
    if name not in {"Group", "Condition"}
}
data["Group"] = [group_names[int(code) - 1] for code in group_codes]
data["Condition"] = [
    condition_chars[i : i + 2] for i in range(0, len(condition_chars), 2)
]
trials = pd.DataFrame(data, columns=variable_names)
assert len(trials) == n_rows


# %% Check stored relationships and select the simultaneous RT rows
assert trials["Valid"].eq(1).all()
assert trials["Choice"].isin([-1, 1]).all()
assert (trials["ChoseRight"].eq(1) == trials["Choice"].eq(1)).all()
assert np.allclose(
    trials["Delta_dB"], trials["Stim_right_dB"] - trials["Stim_left_dB"],
    rtol=0, atol=1e-12,
)
assert not trials.duplicated(["Participant", "Trial"]).any()

# In this file, both stored luminance columns obey the same exact numerical
# mapping to the dB columns. The file does not identify what the 7.72 cd/m²
# offset represents physically (for example, display background luminance).
observed_offsets = np.concatenate([
    trials[f"Stim_{side}_cdm2"].to_numpy()
    - 10 ** (trials[f"Stim_{side}_dB"].to_numpy() / 10)
    for side in ("left", "right")
])
luminance_offset_cdm2 = float(np.median(observed_offsets))
conversion_max_abs_error_cdm2 = float(
    np.max(np.abs(observed_offsets - luminance_offset_cdm2))
)
assert conversion_max_abs_error_cdm2 < 1e-10
assert np.isclose(luminance_offset_cdm2, 7.72, rtol=0, atol=1e-10)

chosen = trials.loc[trials["Condition"].eq(CONDITION)].copy()
assert len(chosen) > 0
assert chosen["Stim_duration_s"].isna().all()
assert chosen["FirstSide"].isna().all()
assert chosen["RT_s"].notna().all()
assert chosen.groupby(["Participant", "Session", "Block"])["ASL_cd"].nunique().eq(1).all()

chosen.insert(0, "source_row", np.flatnonzero(trials["Condition"].eq(CONDITION)) + 1)
chosen["Mean_dB"] = (chosen["Stim_left_dB"] + chosen["Stim_right_dB"]) / 2
for name in ["Participant", "GroupID", "Session", "Block", "Trial"]:
    chosen[name] = chosen[name].astype(int)

output_columns = [
    "source_row", "Participant", "GroupID", "Group", "Trial", "Session",
    "Block", "Condition", "ASL_cd", "Stim_left_dB", "Stim_right_dB",
    "Mean_dB", "Delta_dB", "Stim_left_cdm2", "Stim_right_cdm2",
    "Delta_cdm2", "Stim_duration_s", "Choice", "ChoseRight", "Correct",
    "RT_s", "Valid",
]
chosen.to_csv(OUTPUT, columns=output_columns, index=False, float_format="%.17g")

summary = {
    "source": SOURCE.name,
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "source_rows": n_rows,
    "source_columns": variable_names,
    "condition_counts": {k: int(v) for k, v in trials["Condition"].value_counts().items()},
    "selection": "Condition == 'RT'",
    "selected_rows": len(chosen),
    "selected_participants": int(chosen["Participant"].nunique()),
    "mean_brightness": "Mean_dB = (Stim_left_dB + Stim_right_dB) / 2",
    "difference": "Delta_dB is the stored Stim_right_dB - Stim_left_dB",
    "observed_luminance_mapping": "Stim_side_cdm2 = offset + 10^(Stim_side_dB / 10)",
    "observed_luminance_offset_cdm2": luminance_offset_cdm2,
    "conversion_max_abs_error_cdm2": conversion_max_abs_error_cdm2,
    "output": OUTPUT.name,
    "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    "rt_s_min": float(chosen["RT_s"].min()),
    "rt_s_max": float(chosen["RT_s"].max()),
    "rt_ge_2_49_s": int(chosen["RT_s"].ge(2.49).sum()),
    "rt_ge_2_4999_s": int(chosen["RT_s"].ge(2.4999).sum()),
    "rt_ge_2_5_s": int(chosen["RT_s"].ge(2.5).sum()),
}
SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: summary[key] for key in ["condition_counts", "selected_rows", "selected_participants", "rt_s_min", "rt_s_max", "output"]}, indent=2))
