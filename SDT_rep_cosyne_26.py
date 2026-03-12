# %%
import numpy as np
import matplotlib.pyplot as plt


def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def d_prime(mu_noise, sigma_noise, mu_signal, sigma_signal):
    pooled_sd = np.sqrt((sigma_noise**2 + sigma_signal**2) / 2)
    return (mu_signal - mu_noise) / pooled_sd


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["bottom"].set_color("#1f1f1f")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)


# %%
# Editable parameters.
mu_noise = -1.1
sigma_noise = 1.0
mu_signal = 1.1
sigma_signal = 1.0

x_padding = 3.2
save_svg = True
save_eps = True
save_pdf = True


# %%
colors = {
    "noise": "#d85a5a",
    "signal": "#2f2f36",
    "overlap": "#9e9e9e",
    "ink": "#1f1f1f",
}

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "hatch.linewidth": 0.8,
    }
)

x_min = min(mu_noise - x_padding * sigma_noise, mu_signal - x_padding * sigma_signal)
x_max = max(mu_noise + x_padding * sigma_noise, mu_signal + x_padding * sigma_signal)
x = np.linspace(x_min, x_max, 1400)

y_noise = gaussian_pdf(x, mu_noise, sigma_noise)
y_signal = gaussian_pdf(x, mu_signal, sigma_signal)
y_overlap = np.minimum(y_noise, y_signal)
criterion = 0.5 * (mu_noise + mu_signal)

fig, ax = plt.subplots(figsize=(6.2, 3.5), dpi=300)

ax.plot(x, y_noise, color=colors["noise"], lw=3.0)
ax.plot(x, y_signal, color=colors["signal"], lw=3.0)

ax.fill_between(
    x,
    0,
    y_overlap,
    facecolor=(1, 1, 1, 0),
    hatch="..",
    edgecolor=colors["overlap"],
    linewidth=0.0,
    zorder=0,
)

ax.vlines(
    criterion,
    -0.02,
    max(y_noise.max(), y_signal.max()) * 1.12,
    color=colors["ink"],
    lw=1.8,
    linestyles="--",
)

style_axis(ax)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.03, max(y_noise.max(), y_signal.max()) * 1.18)
ax.set_xlabel("Decision variable")

fig.tight_layout()

if save_eps:
    fig.savefig("SDT_rep_cosyne_26.eps", format="eps", bbox_inches="tight")
if save_pdf:
    fig.savefig("SDT_rep_cosyne_26.pdf", format="pdf", bbox_inches="tight")
if save_svg:
    fig.savefig("SDT_rep_cosyne_26.svg", format="svg", bbox_inches="tight")

plt.show()
