# %%
import numpy as np
import pandas as pd
import pyddm
from pyddm import Model, Fittable, Sample
from pyddm.models import Drift, Bound, NoiseConstant, ICPointSourceCenter
from pyddm.models import OverlayChain, OverlayNonDecision, OverlayPoissonMixture
from pyddm.models.loss import LossLikelihood, LossRobustLikelihood


class Drift_LEDLinear(Drift):
    """v(t) = v0 + opto * v_slope * max(0, t - led_onset)"""
    name = "Drift: baseline + linear change after LED"
    required_parameters = ["v0", "v_slope"]
    required_conditions = ["opto", "led_onset"]

    def get_drift(self, t, conditions, **kwargs):
        v = self.v0
        if conditions["opto"] in [1, True]:
            tau = max(0.0, t - float(conditions["led_onset"]))
            v = v + self.v_slope * tau
        return v


class Bound_LEDLinear(Bound):
    """B(t) = max(Bmin, B0 - opto * B_collapse * max(0, t - led_onset))"""
    name = "Bound: constant then linear collapse after LED"
    required_parameters = ["B0", "B_collapse", "Bmin"]
    required_conditions = ["opto", "led_onset"]

    def get_bound(self, t, conditions, **kwargs):
        B = self.B0
        if conditions["opto"] in [1, True]:
            tau = max(0.0, t - float(conditions["led_onset"]))
            B = max(self.Bmin, self.B0 - self.B_collapse * tau)
        return B


model = Model(
    drift=Drift_LEDLinear(
        v0=Fittable(minval=-5, maxval=5),
        v_slope=Fittable(minval=-20, maxval=20),  # units: drift/sec after LED
    ),
    noise=NoiseConstant(noise=1.0),
    bound=Bound_LEDLinear(
        B0=Fittable(minval=0.2, maxval=5.0),
        B_collapse=Fittable(minval=0.0, maxval=20.0),  # collapse rate (1/sec)
        Bmin=0.05,  # floor to keep bounds > 0
    ),
    IC=ICPointSourceCenter(),
    overlay=OverlayChain(overlays=[
        OverlayNonDecision(nondectime=Fittable(minval=0.0, maxval=0.8)),
        # optional but strongly recommended for likelihood robustness:
        OverlayPoissonMixture(pmixturecoef=Fittable(minval=0.0, maxval=0.05)),
    ]),
    dx=0.01,
    dt=0.001,
    T_dur=2.0,  # must be >= max RT in your data (seconds)
    choice_names=("hit", "other"),
)

# quick sanity solve
sol = model.solve(conditions={"opto": 1, "led_onset": 0.25})
print("P(hit):", sol.prob("hit"), "P(other):", sol.prob("other"))
print("pdf(hit) at rt=0.40:", sol.evaluate(rt=0.40, choice="hit"))