# %%
import numpy as np
import matplotlib.pyplot as plt
import pyddm
from tqdm.auto import tqdm


# %%
# Single-bound process params
V_A = 1.4730
theta_A = 2.1596
sigma_A = 1.0

# PyDDM 2-bound params
# Bounds are +/-theta_E and start is x * theta_E.
theta_E = 100
x = 1.0 - (theta_A / theta_E)  # ensures start->upper distance is theta_A

# Numerical settings
dt = 1e-3
dx = 1e-3
T_dur = 6

# Simulator settings (no LED, no t_stim)
sim_dt = 1e-3
N_sim = 3000
sim_seed = 123

if theta_A <= 0:
    raise ValueError("theta_A must be > 0.")
if theta_E <= theta_A:
    raise ValueError("theta_E must be > theta_A so x < 1 and the upper bound stays ahead.")
if not (-1.0 <= x <= 1.0):
    raise ValueError("Derived x must lie in [-1, 1].")

upper_dist = theta_E * (1.0 - x)
lower_dist = theta_E * (1.0 + x)

m_true = pyddm.gddm(
    drift=V_A,
    noise=sigma_A,
    bound=theta_E,  # PyDDM bounds are +/-theta_E
    starting_position=x,  # ratio in [-1, 1]
    nondecision=0.0,
    mixture_coef=0.0,  # pure DDM, no contaminant mixture
    dt=dt,
    dx=dx,
    T_dur=T_dur,
    choice_names=("upper_hit", "lower_hit"),
)

# %%
sol_true = m_true.solve()
t = sol_true.t_domain
pdf_upper = sol_true.pdf("upper_hit")
pdf_lower = sol_true.pdf("lower_hit")
P_upper = sol_true.prob("upper_hit")
P_lower = sol_true.prob("lower_hit")
print(f'P_upper = {P_upper}, P_lower = {P_lower}')
# %%
pdf_upper_cond = pdf_upper / P_upper if P_upper > 0 else np.zeros_like(pdf_upper)

# Numerical integral checks
int_upper = np.trapz(pdf_upper, t)
int_lower = np.trapz(pdf_lower, t)
int_upper_cond = np.trapz(pdf_upper_cond, t)
# Conditional density given upper hit ("upper-only likelihood"), integrates to 1.

print(f"V_A={V_A:.4f}, theta_A={theta_A:.4f}, theta_E={theta_E:.4f}, sigma={sigma_A:.3f}")
print(
    "PyDDM params:",
    f"bound(theta_E)={theta_E:.4f},",
    f"starting_position(x)={x:.4f}",
)
print(
    "Distance checks:",
    f"start->upper={upper_dist:.4f}",
    f"start->lower={lower_dist:.4f}",
)
print(
    f"Theoretical hit probs: upper={P_upper:.6f}, "
    f"lower={P_lower:.6f}"
)
print(
    f"Integral checks: int(pdf_upper)={int_upper:.6f}, \n"
    f"int(pdf_lower)={int_lower:.6f}, \n"
    f"int(pdf_upper_cond)={int_upper_cond:.6f}"
)

# %%
# simulator for comparison
def simulate_single_bound_rt(V_A, theta_A, sigma_A=1.0, dt=1e-4, max_t=6.0):
    """
    Simple single-bound simulation:
    dA = V_A*dt + sigma_A*dW, hit when A >= theta_A.
    """
    A = 0.0
    t_now = 0.0
    dB = sigma_A * np.sqrt(dt)

    while t_now < max_t:
        A += V_A * dt + np.random.normal(0.0, dB)
        t_now += dt
        if A >= theta_A:
            return t_now

    return np.nan


def simulate_single_trial():
    return simulate_single_bound_rt(
        V_A=V_A,
        theta_A=theta_A,
        sigma_A=sigma_A,
        dt=sim_dt,
        max_t=T_dur,
    )


N_sim = int(50e3)
sim_dt = 1e-4
sim_results = [simulate_single_trial() for _ in tqdm(range(N_sim), desc="Simulating RTs")]
sim_rts = np.array([rt for rt in sim_results if np.isfinite(rt)], dtype=float)

print(
    f"Simulation: n_hit_by_Tdur={len(sim_rts)} / {N_sim}, "
    f"frac_hit_by_Tdur={len(sim_rts)/N_sim:.6f}, sim_dt={sim_dt}"
)

# %%
plt.figure(figsize=(7, 4))
plt.plot(t, pdf_upper_cond, label="pyddm theory")
if len(sim_rts) > 0:
    bins = np.arange(0, T_dur, 0.05)
    plt.hist(
        sim_rts,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        label=f"single-bound simulation",
    )
plt.xlabel("RT (s)")
plt.ylabel("Density")
plt.title("pyddm")
plt.legend()
plt.tight_layout()

# %%
