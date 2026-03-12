# %%
import numpy as np
import matplotlib.pyplot as plt


def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def d_prime(mu1, s1, mu2, s2):
    pooled_sd = np.sqrt((s1**2 + s2**2) / 2)
    return (mu2 - mu1) / pooled_sd


def normalize_to_dv(params):
    pooled_sd = np.sqrt((params["s1"]**2 + params["s2"]**2) / 2)
    midpoint = 0.5 * (params["mu1"] + params["mu2"])
    return {
        "mu1": (params["mu1"] - midpoint) / pooled_sd,
        "s1": params["s1"] / pooled_sd,
        "mu2": (params["mu2"] - midpoint) / pooled_sd,
        "s2": params["s2"] / pooled_sd,
        "dprime": d_prime(params["mu1"], params["s1"], params["mu2"], params["s2"]),
    }


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
# %%
set_1 = {
    "mu1": 20,
    "s1": 4,
    "mu2": 24,
    "s2": 4.8,
}

set_2 = {
    "mu1": 40,
    "s1": 8,
    "mu2": 48,
    "s2": 9.6,
}

# Equal d' is visible only after expressing both conditions on the same DV scale.
set_1_dv = normalize_to_dv(set_1)
set_2_dv = normalize_to_dv(set_2)

x = np.linspace(-4.4, 4.4, 1000)
fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=300)

blue_dark = "#255dab"
blue_light = "#5f88c4"
red_dark = "#c13a2d"
red_light = "#cc6c62"
purple = "#7b2cbf"

for params, offset, colors in (
    (set_1_dv, -3.1, (blue_dark, blue_light)),
    (set_2_dv, 3.1, (red_dark, red_light)),
):
    x_plot = x + offset
    y1 = gaussian_pdf(x, params["mu1"], params["s1"])
    y2 = gaussian_pdf(x, params["mu2"], params["s2"])
    ax.plot(x_plot, y1, color=colors[0], lw=3.0)
    ax.plot(x_plot, y2, color=colors[1], lw=3.0)

    dprime_y = 0.0
    ax.hlines(
        dprime_y,
        offset + params["mu1"],
        offset + params["mu2"],
        color=colors[0],
        lw=3.0,
        zorder=5,
        clip_on=False,
    )
    ax.scatter(
        [offset + params["mu1"], offset + params["mu2"]],
        [dprime_y, dprime_y],
        s=26,
        color=colors[0],
        zorder=6,
        clip_on=False,
    )

shared_dv_height = max(
    gaussian_pdf(0, set_1_dv["mu1"], set_1_dv["s1"]),
    gaussian_pdf(0, set_1_dv["mu2"], set_1_dv["s2"]),
)
ax.vlines(0, 0, shared_dv_height, color=purple, lw=2.4, zorder=7)

style_axis(ax)
ax.set_xlim(-8.2, 8.2)
ax.set_xticks([])
ax.set_ylim(0, None)
fig.tight_layout()
fig.savefig("DV_rep_cosyne_26.eps", format="eps", bbox_inches="tight")
fig.savefig("DV_rep_cosyne_26.svg", format="svg", bbox_inches="tight")
plt.show()
