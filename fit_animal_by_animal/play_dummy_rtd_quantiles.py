# %%
"""
Demo: two inverse‑Gaussian (Wald) RT distributions that overlap on the
leading edge and tail, yet diverge around the mode.  Because the CDF is the
integral of the PDF, small local differences in density accumulate: the 90 %
quantile ends up quite different even though the density curves look
superficially similar.

Run this script in a Python environment with NumPy, SciPy and Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgauss

# -----------------------------------------------------------------------------
# 1.  Define two inverse‑Gaussian distributions
# -----------------------------------------------------------------------------
# Parameters were chosen by trial‑and‑error so that the blue and red curves:
#   • start nearly together around x ≈ 0.05 s,
#   • share a similar heavy tail for x ≳ 1.5 s,
#   • have visibly different peaks.
# You can tweak these numbers to get an even closer match to your empirical
# curves, but they already illustrate the point.

mu_1, scale_1 = 0.60, 0.15     # “Model” – narrower, higher peak
mu_2, scale_2 = 1.00, 0.25     # “Empirical” – broader, lower peak

rtd_model = invgauss(mu_1, scale=scale_1)
rtd_empir = invgauss(mu_2, scale=scale_2)

x = np.linspace(0.01, 2.0, 2000)  # reaction time axis in seconds

# -----------------------------------------------------------------------------
# 2.  Plot PDFs (density functions)
# -----------------------------------------------------------------------------
pdf_model = rtd_model.pdf(x)
pdf_empir = rtd_empir.pdf(x)

plt.figure(figsize=(6, 3))
plt.plot(x, pdf_model, label=f"Model μ={mu_1}, λ={scale_1}", color="blue")
plt.plot(x, pdf_empir, label=f"Empirical μ={mu_2}, λ={scale_2}", color="red")
plt.xlabel("Reaction Time (s)")
plt.ylabel("Density")
plt.title("RT Distributions (PDF)")
plt.legend(loc="upper right")
plt.tight_layout()

# -----------------------------------------------------------------------------
# 3.  Plot CDFs (integrated densities)
# -----------------------------------------------------------------------------
cdf_model = rtd_model.cdf(x)
cdf_empir = rtd_empir.cdf(x)

plt.figure(figsize=(6, 3))
plt.plot(x, cdf_model, label="Model CDF", color="blue")
plt.plot(x, cdf_empir, label="Empirical CDF", color="red")
plt.xlabel("Reaction Time (s)")
plt.ylabel("Cumulative probability")
plt.title("RT Distributions (CDF)")
plt.legend(loc="lower right")
plt.tight_layout()

# -----------------------------------------------------------------------------
# 4.  Quantile‑probability plot (focus on ≤90 %)
# -----------------------------------------------------------------------------
probs = np.linspace(0.1, 0.9, 9)           # deciles from 10 % … 90 %
q_model = rtd_model.ppf(probs)
q_empir = rtd_empir.ppf(probs)

plt.figure(figsize=(6, 3))
plt.plot(probs, q_model, "bo-", label="Model Q(p)")
plt.plot(probs, q_empir, "rs-", label="Empirical Q(p)")

# Highlight the 90 % quantile gap
plt.axvline(0.9, linestyle="--", color="gray", lw=0.8)
plt.text(0.905, (q_model[-1] + q_empir[-1]) / 2,
         f"Δ @90% = {q_model[-1] - q_empir[-1]:.3f} s",
         va="center", ha="left")

plt.xlabel("Quantile p")
plt.ylabel("Reaction Time (s)")
plt.title("Quantile–Probability Plot (up to 90 %)")
plt.legend()
plt.tight_layout()

plt.show()