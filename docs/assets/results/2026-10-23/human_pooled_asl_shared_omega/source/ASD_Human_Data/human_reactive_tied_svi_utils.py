# %%
"""Human RT adapter for the repository's vanilla/IPL first-passage primitives.

Five physical parameters, shared across every stimulus within a pooled group.
The likelihood is joint in RT and choice and conditional on retention in [0,3] s.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fit_animal_by_animal"))
from numpyro_vanilla_condition_delay_svi_utils import (
    CDF_E_vanilla_jax, rho_E_vanilla_jax, vanilla_gamma_omega_jax,
)

PARAM_NAMES = ["rate_lambda", "T_0", "theta_E", "w", "t_E_aff"]
INITIAL = dict(zip(PARAM_NAMES, [0.2, 0.001, 30.0, 0.5, 0.05]))
DELTAS = np.array([-2., -1.3, -.65, -.3, -.1, .1, .3, .65, 1.3, 2.])
RT_LOWER, RT_UPPER = 0.0, 3.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trials():
    folder = ROOT / "ASD_Human_Data"
    provenance = json.loads((folder / "simultaneous_rt_extraction.json").read_text())
    source = folder / provenance["source"]
    csv = folder / "simultaneous_rt_trials.csv"
    assert sha256(source) == provenance["source_sha256"]
    assert sha256(csv) == provenance["output_sha256"]
    frame = pd.read_csv(csv)
    assert len(frame) == 11400 and frame.source_row.is_unique
    assert not frame.duplicated(["Participant", "Trial"]).any()
    assert frame.Condition.eq("RT").all() and frame.Valid.eq(1).all()
    assert frame.Stim_duration_s.isna().all()
    assert frame.Choice.isin([-1, 1]).all()
    assert np.array_equal(frame.ChoseRight, frame.Choice.eq(1).astype(int))
    assert np.allclose(frame.Mean_dB, (frame.Stim_left_dB + frame.Stim_right_dB)/2, atol=1e-12, rtol=0)
    assert np.allclose(frame.Delta_dB, frame.Stim_right_dB - frame.Stim_left_dB, atol=1e-12, rtol=0)
    assert np.isfinite(frame.RT_s).all() and frame.RT_s.between(0, 3, inclusive="neither").all()
    assert np.allclose(frame.Delta_dB, frame.Delta_dB.round(2), atol=1e-12, rtol=0)
    frame["signed_delta"] = frame.Delta_dB.round(2)
    assert set(frame.signed_delta) == set(DELTAS)
    for name, gid, n, people in [("Control", 0, 6600, 11), ("ASD", 1, 4800, 8)]:
        sub = frame.loc[frame.Group.eq(name)]
        assert len(sub) == n and sub.Participant.nunique() == people
        assert sub.GroupID.eq(gid).all() and sub.groupby("Participant").size().eq(600).all()
        assert sub.groupby(["ASL_cd", "signed_delta"]).size().eq(n//20).all()
    provenance.update(csv_sha256=sha256(csv), selected_source_rows_sha256=hashlib.sha256(
        np.asarray(frame.source_row, dtype="<i8").tobytes()).hexdigest())
    return frame, provenance


def group_data(frame, group):
    sub = frame.loc[frame.Group.eq(group)].copy()
    conditions = sub.groupby(["ASL_cd", "signed_delta"], sort=True).agg(
        Mean_dB=("Mean_dB", "mean"), n=("Choice", "size"), right=("ChoseRight", "sum"),
    ).reset_index()
    assert len(conditions) == 20
    conditions["condition_index"] = np.arange(20)
    sub = sub.merge(conditions[["ASL_cd", "signed_delta", "condition_index"]],
                    on=["ASL_cd", "signed_delta"], how="left", validate="many_to_one", sort=False)
    data = {
        "rt": jnp.array(sub.RT_s.to_numpy()), "choice": jnp.array(sub.Choice.to_numpy()),
        "B": jnp.array(sub.Mean_dB.to_numpy()), "D": jnp.array(sub.Delta_dB.to_numpy()),
        "condition": jnp.array(sub.condition_index.to_numpy()),
        "condition_B": jnp.array(conditions.Mean_dB.to_numpy()),
        "condition_D": jnp.array(conditions.signed_delta.to_numpy()),
    }
    upper = float(sub.RT_s.min() - 1e-6)
    return sub, conditions, data, upper


def bound_cdf(rt, choice, B, D, params, K_max=40):
    rate, t0, theta, w, delay = params
    return CDF_E_vanilla_jax(rt-delay, choice, B, D, rate, t0, theta,
                             (w-.5)*2*theta, K_max)


def bound_density(rt, choice, B, D, params, K_max=40):
    rate, t0, theta, w, delay = params
    return rho_E_vanilla_jax(rt-delay, choice, B, D, rate, t0, theta,
                             (w-.5)*2*theta, K_max)


def retained_mass(B, D, params, K_max=40):
    return sum(bound_cdf(RT_UPPER, c, B, D, params, K_max)
               - bound_cdf(RT_LOWER, c, B, D, params, K_max) for c in [-1, 1])


def log_likelihood(params, data, K_max=40):
    density = bound_density(data["rt"], data["choice"], data["B"], data["D"], params, K_max)
    mass = retained_mass(data["condition_B"], data["condition_D"], params, K_max)
    # The primitive's existing density floor is retained; never normalize choices separately.
    return jnp.sum(jnp.log(density) - jnp.log(mass[data["condition"]]))


def reactive_model(data, delay_upper, K_max=40):
    rate = numpyro.sample("rate_lambda", dist.LogNormal(np.log(.2), 1.5))
    t0 = numpyro.sample("T_0", dist.LogNormal(np.log(.001), 2.0))
    theta = numpyro.sample("theta_E", dist.LogNormal(np.log(30.), 1.5))
    w = numpyro.sample("w", dist.Uniform(.05, .95))
    delay = numpyro.sample("t_E_aff", dist.Uniform(0., delay_upper))
    params = jnp.stack([rate, t0, theta, w, delay])
    numpyro.factor("joint_retained_rt_choice", log_likelihood(params, data, K_max))
