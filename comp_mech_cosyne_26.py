# %%
import numpy as np
import matplotlib.pyplot as plt


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0


def rgb01_to_hex(rgb):
    rgb = np.clip(np.round(rgb * 255), 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def blend_with_background(hex_color, background_hex, alpha):
    color = hex_to_rgb01(hex_color)
    background = hex_to_rgb01(background_hex)
    return rgb01_to_hex(alpha * color + (1 - alpha) * background)


def compute_mu_sigma(abl, ild, q_e, rate_lambda, t_0, chi):
    gain = 10 ** (rate_lambda * abl / 20)
    ild_term = rate_lambda * ild / chi
    mu = (2 * q_e / t_0) * gain * np.sinh(ild_term)
    sigma = np.sqrt((2 * q_e**2 / t_0) * gain * np.cosh(ild_term))
    return mu, sigma


def prob_hit_upper(ild, theta, rate_lambda, chi):
    return logistic(2 * theta * np.tanh(rate_lambda * ild / chi))


def simulate_ddm_trajectories(mu, sigma, theta, dt, max_time, n_traces, rng):
    n_steps = int(max_time / dt) + 1
    time = np.linspace(0, dt * (n_steps - 1), n_steps)
    trajectories = []
    hit_upper = 0

    for _ in range(n_traces):
        dv = np.zeros(n_steps)
        for idx in range(1, n_steps):
            dv[idx] = dv[idx - 1] + mu * dt + sigma * np.sqrt(dt) * rng.normal()
            if dv[idx] >= theta:
                dv[idx] = theta
                trajectories.append((time[: idx + 1], dv[: idx + 1]))
                hit_upper += 1
                break
            if dv[idx] <= -theta:
                dv[idx] = -theta
                trajectories.append((time[: idx + 1], dv[: idx + 1]))
                break
        else:
            trajectories.append((time, np.clip(dv, -theta, theta)))

    return trajectories, hit_upper / n_traces


def style_trajectory_axis(ax, theta, max_time):
    ax.set_xlim(0, max_time * 1e3)
    ax.set_ylim(-1.14 * theta, 1.14 * theta)
    ax.axis("off")

    ax.hlines(theta, 0, max_time * 1e3, color="#4f8f3b", lw=2.0, zorder=1)
    ax.hlines(-theta, 0, max_time * 1e3, color="#7b3fb2", lw=2.0, zorder=1)
    ax.vlines(0, -theta, theta, color="#2b2b2b", lw=2.1, zorder=3)


def style_summary_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# %%
# Editable parameters.
q_e = 1.0
rate_lambda = 0.13
T_0 = 0.2e-3
chi = 17.37
theta = 40.0

ild = 2.0
abl_low = 0.0
abl_high = 100.0

n_traces = 5
dt = 0.5e-3
max_time = 0.34
rng_seed = 7

save_svg = True
save_eps = True


# %%
colors = {
    "low": "#2f6db2",
    "high": "#c44e3b",
    "upper": "#4f8f3b",
    "lower": "#7b3fb2",
    "ink": "#1f1f1f",
    "grid": "#d9d2c7",
    "paper": "#ffffff",
}

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.facecolor": colors["paper"],
        "figure.facecolor": colors["paper"],
        "savefig.facecolor": colors["paper"],
    }
)

rng = np.random.default_rng(rng_seed)
low_trace_color = blend_with_background(colors["low"], colors["paper"], alpha=0.58)
high_trace_color = blend_with_background(colors["high"], colors["paper"], alpha=0.58)

mu_low, sigma_low = compute_mu_sigma(abl_low, ild, q_e, rate_lambda, T_0, chi)
mu_high, sigma_high = compute_mu_sigma(abl_high, ild, q_e, rate_lambda, T_0, chi)

traj_low, _ = simulate_ddm_trajectories(mu_low, sigma_low, theta, dt, max_time, n_traces, rng)
traj_high, _ = simulate_ddm_trajectories(
    mu_high, sigma_high, theta, dt, max_time, n_traces, rng
)
theory_p_upper = prob_hit_upper(ild, theta, rate_lambda, chi)


# %%
fig = plt.figure(figsize=(8.2, 4.8), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[6.4, 0.75], wspace=0.03)

ax_traj = fig.add_subplot(gs[0, 0])
ax_sum = fig.add_subplot(gs[0, 1])

style_trajectory_axis(ax_traj, theta, max_time)
style_summary_axis(ax_sum)

for time, dv in traj_low:
    ax_traj.plot(time * 1e3, dv, color=low_trace_color, lw=1.45, zorder=2)

for time, dv in traj_high:
    ax_traj.plot(time * 1e3, dv, color=high_trace_color, lw=1.45, zorder=2)

bar_positions = [0.34, 0.68]
bar_width = 0.18
bar_base = 0.26
bar_total_height = 0.34

for x_pos in bar_positions:
    ax_sum.bar(
        x_pos,
        (1 - theory_p_upper) * bar_total_height,
        bottom=bar_base,
        width=bar_width,
        color=colors["lower"],
        linewidth=0,
    )
    ax_sum.bar(
        x_pos,
        theory_p_upper * bar_total_height,
        bottom=bar_base + (1 - theory_p_upper) * bar_total_height,
        width=bar_width,
        color=colors["upper"],
        linewidth=0,
    )

ax_sum.text(
    0.51,
    bar_base + bar_total_height + 0.08,
    "P(RIGHT)",
    color=colors["upper"],
    ha="center",
    va="bottom",
    fontsize=10,
)

fig.text(
    0.012,
    0.53,
    "Decision Variable",
    rotation=90,
    ha="center",
    va="center",
    color=colors["ink"],
    fontsize=11,
)

fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.14)

if save_eps:
    fig.savefig("comp_mech_cosyne_26.eps", format="eps", bbox_inches="tight")
if save_svg:
    fig.savefig("comp_mech_cosyne_26.svg", format="svg", bbox_inches="tight")

plt.show()
