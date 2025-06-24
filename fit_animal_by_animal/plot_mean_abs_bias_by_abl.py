import numpy as np
import matplotlib.pyplot as plt
from plot_slope_ratios_histograms import ABLS, biases, animals, COLORS

# Compute mean of abs(bias) for each ABL
mean_abs_bias = []
for abl in ABLS:
    abs_bias_vals = [np.abs(biases[abl].get(animal, np.nan)) for animal in animals]
    # Remove NaNs
    abs_bias_vals = [v for v in abs_bias_vals if not np.isnan(v)]
    mean_abs_bias.append(np.mean(abs_bias_vals))

fig, ax = plt.subplots(figsize=(4, 3))
ax.bar([str(abl) for abl in ABLS], mean_abs_bias, color=COLORS)
ax.set_xlabel('ABL', fontsize=13)
ax.set_ylabel('Mean |Bias| (|x0|)', fontsize=13)
ax.set_title('Mean Absolute Bias by ABL', fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
