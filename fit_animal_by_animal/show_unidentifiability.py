# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def gama(lam, theta ,ild):
    return theta * np.tanh(lam * ild / 17.37)

# Publication grade styling
TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 24
LEGEND_FONTSIZE = 16
SUPTITLE_FONTSIZE = 24
font_family = "Helvetica"

# Color definitions
COLOR_CURVE_1 = 'black'
COLOR_CURVE_2 = '#9467bd'
COLOR_CURVE_3 = '#8c564b'

# ILD parameters
BIG_ILD = 16
SMALL_ILD = 4

mpl.rcParams.update({
    "savefig.pad_inches": 0.6,
    "font.family": "sans-serif",
    "font.sans-serif": [
        font_family,
        "Helvetica Neue",
        "TeX Gyre Heros",
        "Arial",
        "sans-serif",
    ],
    "axes.labelpad": 12,
})

# Parameters for the first curve (defines the target central slope)
lam1 = 0.13
theta1 = 30.0

# Compute target central slope: m = dγ/dILD|0 = θ·λ / 17.37
slope_m = theta1 * lam1 / 17.37

# Choose a second λ and set θ2 so the linear parts coincide with the same slope
lam2 = 2.13
theta2 = slope_m * 17.37 / lam2  # = (theta1*lam1)/lam2

# A third λ chosen to turn near the plot edges (≈ ±BIG_ILD).
# It keeps the same central slope but bends only close to the limits.
# lam3 = 13.37 / BIG_ILD
lam3 = 0.5
print(f'lam3 = {lam3}')
theta3 = slope_m * 17.37 / lam3

# Ranges
ild_vals_small = np.arange(-SMALL_ILD, SMALL_ILD, 0.01)
ild_vals_big = np.arange(-BIG_ILD, BIG_ILD, 0.01)

# Curves
gama_ild_1 = gama(lam1, theta1, ild_vals_small)
gama_ild_2 = gama(lam2, theta2, ild_vals_big)
gama_ild_3 = gama(lam3, theta3, ild_vals_big)

# Straight line with the same central slope
line_small = slope_m * ild_vals_small
line_big = slope_m * ild_vals_big

# %%
# Plot
fig, ax = plt.subplots(figsize=(4.5, 4))
# Using black and grey colors to avoid the restricted palette
ax.plot(ild_vals_small, gama_ild_1, label=f"tanh: λ={lam1:g}, θ={theta1:g}", lw=2, color=COLOR_CURVE_1)
ax.plot(ild_vals_big, gama_ild_2, label=f"tanh: λ={lam2:g}, θ={theta2:.3g}", lw=2, color=COLOR_CURVE_2)
ax.plot(ild_vals_big, gama_ild_3, label=f"tanh: λ={lam3:.3g}, θ={theta3:.3g}", lw=2, color=COLOR_CURVE_3)

# Plot tanh curves with different styles: solid from -SMALL_ILD to SMALL_ILD, dotted outside
# Using the same parameters as the main curves for consistency
ild_solid = np.arange(-SMALL_ILD, SMALL_ILD, 0.01)
ild_dotted_left = np.arange(-BIG_ILD, -SMALL_ILD, 0.01)
ild_dotted_right = np.arange(SMALL_ILD, BIG_ILD, 0.01)

gamma_solid = gama(lam1, theta1, ild_solid)
gamma_dotted_left = gama(lam1, theta1, ild_dotted_left)
gamma_dotted_right = gama(lam1, theta1, ild_dotted_right)

ax.plot(ild_solid, gamma_solid, "-", color="k", alpha=0.8, lw=3)
ax.plot(ild_dotted_left, gamma_dotted_left, ":", color="k", alpha=1, lw=2)
ax.plot(ild_dotted_right, gamma_dotted_right, ":", color="k", alpha=1, lw=2)

# Add vertical lines at -SMALL_ILD and SMALL_ILD
ax.axvline(x=-SMALL_ILD, color='gray', linestyle='--', alpha=0.7)
ax.axvline(x=SMALL_ILD, color='gray', linestyle='--', alpha=0.7)

# Add light green background between -SMALL_ILD and SMALL_ILD
ax.axvspan(-SMALL_ILD, SMALL_ILD, color='lightgreen', alpha=0.3)

# Scatter plot at specific ILD values on the sigmoid curves
ild_scatter = np.array([-BIG_ILD, -8, -SMALL_ILD, -2, -1, 1, 2, SMALL_ILD, 8, BIG_ILD])
gamma_scatter_1 = gama(lam1, theta1, ild_scatter)
gamma_scatter_2 = gama(lam2, theta2, ild_scatter)
gamma_scatter_3 = gama(lam3, theta3, ild_scatter)
ax.scatter(ild_scatter, gamma_scatter_2, color=COLOR_CURVE_2, s=50, zorder=5, marker='o')
ax.scatter(ild_scatter, gamma_scatter_3, color=COLOR_CURVE_3, s=50, zorder=5, marker='o')

# Styling adjustments for publication grade plot
ax.set_xlabel("ILD (dB)", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("γ(ILD)", fontsize=LABEL_FONTSIZE)
ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

# Set specific x-axis ticks
ax.set_xticks([-BIG_ILD, -SMALL_ILD, SMALL_ILD, BIG_ILD])

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Remove y-axis ticks and labels
ax.set_yticks([])
ax.set_ylabel('')

# Add lambda values as colored text BELOW the plot on the right, matching curve colors
text_x, text_y = 0.98, 0.6  # below the axes; right-aligned
line_h = 0.12  # vertical spacing between lines (in axes fraction)
lambda_lines = [
    # (f"λ={lam1:.2f}", COLOR_CURVE_1),
    (f"λ={lam3:.2f}", COLOR_CURVE_3),
    (f"λ={lam2:.2f}", COLOR_CURVE_2),
]
for i, (txt, col) in enumerate(lambda_lines):
    ax.text(
        text_x, text_y - i * line_h, txt,
        transform=ax.transAxes, ha='right', va='top', fontsize=14, color=col,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1.5)
    )

# Remove title and legend as requested
# ax.set_title("Same central slope: γ'0=θλ/17.37")
# ax.legend(frameon=False)
ax.grid(True, alpha=0.2)

out_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "unidentifiability_match_line.pdf")
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {out_path}")
# %%
# print(f'lam = {lam1}, theta1 = {theta1}')
# big_ild_test = np.arange(-100, 100, 0.1)
# gama_big_ild_test = gama(lam1, theta1, big_ild_test)
# plt.plot(big_ild_test, gama_big_ild_test)
# plt.title(f'γ for lam={lam1}, theta={theta1}')