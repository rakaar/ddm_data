# %%
import numpy as np
import matplotlib.pyplot as plt
import pyddm
from tqdm.auto import tqdm


# %%
# Single-bound process params
V_A = 1.4730
sigma_A = 1.0
theta_A0 = 2.1596

# Bound drop schedule: theta_A(t) = theta_A0 - slope * (t - t_drop_start), clipped at theta_A_min
t_drop_start = 1.0
bound_slope = 0.5
theta_A_min = 0.01

# PyDDM 2-bound mapping params
# At t=0, upper distance must be theta_A0:
#   theta_A0 = theta_E0 * (1 - x)  ->  x = 1 - theta_A0/theta_E0
theta_E0 = 20.0

# Solver settings
dt = 1e-3
dx = 1e-3
T_dur = 6.0

# Simulator settings
sim_dt = 1e-4
N_sim = int(50e3)
sim_seed = 123


# %%
if theta_A0 <= 0:
    raise ValueError("theta_A0 must be > 0.")
if theta_A_min <= 0:
    raise ValueError("theta_A_min must be > 0.")
if theta_E0 <= theta_A0:
    raise ValueError("theta_E0 must be > theta_A0.")
if bound_slope < 0:
    raise ValueError("bound_slope must be >= 0.")

x = 1.0 - (theta_A0 / theta_E0)  # starting position ratio used by PyDDM
x_abs = x * theta_E0             # absolute start in PyDDM coordinates
theta_E_min = x_abs + theta_A_min


# %%
def theta_single_t(t, theta0, slope, t_start, theta_min):
    if t <= t_start:
        return theta0
    return max(theta_min, theta0 - slope * (t - t_start))


def simulate_single_bound_bound_drop_rt(
    V_A,
    theta_A0,
    bound_slope,
    t_drop_start,
    theta_A_min,
    sigma_A=1.0,
    dt=1e-3,
    max_t=6.0,
):
    """
    Single-bound simulator with linearly dropping bound after t_drop_start.
    """
    A = 0.0
    t_now = 0.0
    dB = sigma_A * np.sqrt(dt)

    while t_now < max_t:
        theta_t = theta_single_t(t_now, theta_A0, bound_slope, t_drop_start, theta_A_min)
        A += V_A * dt + np.random.normal(0.0, dB)
        t_now += dt
        if A >= theta_t:
            return t_now

    return np.nan


def simulate_single_trial():
    return simulate_single_bound_bound_drop_rt(
        V_A=V_A,
        theta_A0=theta_A0,
        bound_slope=bound_slope,
        t_drop_start=t_drop_start,
        theta_A_min=theta_A_min,
        sigma_A=sigma_A,
        dt=sim_dt,
        max_t=T_dur,
    )


np.random.seed(sim_seed)
sim_results = [simulate_single_trial() for _ in tqdm(range(N_sim), desc="Simulating single-bound RTs")]
sim_rts = np.array([rt for rt in sim_results if np.isfinite(rt)], dtype=float)

print(f"Single-bound sim hits: {len(sim_rts)} / {N_sim} ({len(sim_rts)/N_sim:.6f})")


# %%
def pyddm_bound_drop(t):
    """
    Symmetric PyDDM bound B(t), chosen so upper distance equals single-bound theta_A(t):
        B(t) - x_abs == theta_A(t)
    """
    B_t = theta_E0 - bound_slope * max(0.0, t - t_drop_start)
    return max(theta_E_min, B_t)


m = pyddm.gddm(
    drift=V_A,
    noise=sigma_A,
    bound=pyddm_bound_drop,
    starting_position=x,
    nondecision=0.0,
    mixture_coef=0.0,
    dt=dt,
    dx=dx,
    T_dur=T_dur,
    choice_names=("upper_hit", "lower_hit"),
)

sol = m.solve()
t = sol.t_domain
pdf_upper = sol.pdf("upper_hit")
pdf_lower = sol.pdf("lower_hit")
P_upper = sol.prob("upper_hit")
P_lower = sol.prob("lower_hit")
pdf_upper_cond = pdf_upper / P_upper if P_upper > 0 else np.zeros_like(pdf_upper)

print(f"PyDDM hit probs: upper={P_upper:.6f}, lower={P_lower:.6f}")
print(
    f"Integral checks: upper={np.trapz(pdf_upper, t):.6f}, "
    f"lower={np.trapz(pdf_lower, t):.6f}, "
    f"upper_cond={np.trapz(pdf_upper_cond, t):.6f}"
)


# %%
t_check = np.linspace(0.0, T_dur, 2000)
theta_single = np.array([theta_single_t(tt, theta_A0, bound_slope, t_drop_start, theta_A_min) for tt in t_check])
theta_from_pyddm = np.array([pyddm_bound_drop(tt) - x_abs for tt in t_check])
max_bound_diff = np.max(np.abs(theta_single - theta_from_pyddm))
print(f"Max |single-bound theta(t) - PyDDM upper-distance(t)| = {max_bound_diff:.3e}")


# %%
plt.figure(figsize=(7.5, 4.5))
bins = np.linspace(0, T_dur, 110)
plt.plot(t, pdf_upper_cond, lw=2, label="PyDDM likelihood")
if len(sim_rts) > 0:
    plt.hist(
        sim_rts,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        label=f"bound drop sim",
    )
plt.xlabel("RT (s)")
plt.ylabel("Density")
plt.title("Bound Drop: sim vs likelihood")
plt.legend()
plt.tight_layout()


# %%
plt.figure(figsize=(7.5, 3.8))
plt.plot(t_check, theta_single, lw=2, label="single-bound theta_A(t)")
# plt.plot(t_check, theta_from_pyddm, lw=1.5, ls="--", label="PyDDM upper distance(t)")
plt.xlabel("t (s)")
plt.ylabel("bound")
plt.title("Bound over time")
plt.legend()
plt.tight_layout()

