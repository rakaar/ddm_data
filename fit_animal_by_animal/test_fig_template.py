# %%
import numpy as np
import matplotlib.pyplot as plt
import figure_template as ft      # your flexible builder

# ── 0.  Build the blank 5 × 6 canvas ───────────────────────────────
builder = ft.FigureBuilder(
    sup_title="Figure 2",
    n_rows=5,
    n_cols=6,
)

# ── 1.  Regular plots we already had (y=x and sin x) ───────────────
#  y = x  at (row 1, col 2)
ax1 = builder.fig.add_subplot(builder.gs[1, 2])
x = np.linspace(0, 1, 100)
ax1.plot(x, x)
ax1.set_title("y = x")

#  y = sin x  at (row 3, col 5)
ax2 = builder.fig.add_subplot(builder.gs[3, 5])
x2 = np.linspace(0, 2*np.pi, 200)
ax2.plot(x2, np.sin(x2))
ax2.set_title("y = sin x")

# ── 2.  NESTED grid (2 × 1) inside (row 1, col 0) ─────────────────
#  Step‑1:  split that outer cell into a sub‑GridSpec
outer_ss   = builder.gs[1, 0]                # SubplotSpec for that cell
inner_gs   = outer_ss.subgridspec(2, 1,      # 2 rows, 1 column
                                 hspace=0.25)

#  Step‑2:  draw e^x in the *top* sub‑axis
ax_top  = builder.fig.add_subplot(inner_gs[0, 0])
x3      = np.linspace(0, 3, 200)
ax_top.plot(x3, np.exp(x3))
ax_top.set_title("y = e^x")

#  Step‑3:  draw e^{-x} in the *bottom* sub‑axis
ax_bot  = builder.fig.add_subplot(inner_gs[1, 0])
ax_bot.plot(x3, np.exp(-x3))
ax_bot.set_title("y = e$^{-x}$")
# After you’ve created ax_top and ax_bot …
ft.shift_axes([ax_top, ax_bot], dy=-0.1)   # 3 % of the figure width to the LEFT


# ── 3.  Finish & display ───────────────────────────────────────────
builder.finish()
plt.show()

