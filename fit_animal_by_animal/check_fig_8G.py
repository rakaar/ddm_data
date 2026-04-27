# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# User-provided parameters

lam = 2.13
ell = 0.9

# ILDs shown in panel G
# here using your convention: ILD = R - L
# ild_range = np.array([30, 20, 10, 0, -10, -20, -30], dtype=float)
ild_range = np.arange(30, -31, -2, dtype=float)
# paper intensity grid
levels = np.arange(0, 71, 10, dtype=float)
paper_ilds = np.arange(30, -31, -10, dtype=float)
fit_through_origin = False

def firing_rate(R, L):
    num = 10 ** ((R * lam) / 20.0)
    den = (10 ** ((R * lam * ell) / 20.0)) + (10 ** ((L * lam * ell) / 20.0))
    return num / den

ABLs = [20, 40, 60]
# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

for ABL in ABLs:
    rate_rl = []
    rate_r0 = []
    gain = []
    for ild in ild_range:
        R = ABL + ild/2
        L = ABL - ild/2

        firing_rl = firing_rate(R, L)
        firing_r0 = firing_rate(R, 0)

        rate_rl.append(firing_rl)
        rate_r0.append(firing_r0)
        gain.append(firing_rl / firing_r0)

    axes[0].plot(ild_range, rate_rl, marker='o', label=f'ABL {ABL}')
    axes[1].plot(ild_range, rate_r0, marker='o', label=f'ABL {ABL}')
    axes[2].plot(ild_range, gain, marker='o', label=f'ABL {ABL}')

axes[0].set_title('Rate_r(R, L)')
axes[1].set_title('Rate_r(R, 0)')
axes[2].set_title('Rate_r(R, L) / Rate_r(R, 0)')
axes[0].set_ylabel('Firing rate')
axes[2].set_ylabel('Gain')

for ax in axes:
    ax.set_xlabel('ILD')
    ax.invert_xaxis()
    ax.legend()

fig.suptitle('Firing rate and gain vs ILD across ABLs')
fig.tight_layout(rect=[0, 0, 1, 0.93])
# %%
fig, axes = plt.subplots(1, len(paper_ilds), figsize=(4 * len(paper_ilds), 4), sharex=True, sharey=True)

paper_gains = []

for ax, ild in zip(axes, paper_ilds):
    contra_only = []
    binaural = []

    for R in levels:
        L = R - ild
        if L < levels.min() or L > levels.max():
            continue

        C = firing_rate(R, 0)
        B = firing_rate(R, L)

        contra_only.append(C)
        binaural.append(B)

    contra_only = np.array(contra_only, dtype=float)
    binaural = np.array(binaural, dtype=float)

    if fit_through_origin:
        slope = np.sum(contra_only * binaural) / np.sum(contra_only ** 2)
        intercept = 0.0
    else:
        slope, intercept = np.polyfit(contra_only, binaural, 1)

    paper_gains.append(slope)

    ax.scatter(contra_only, binaural, s=35)

    line_x = np.linspace(0, max(contra_only.max(), binaural.max()), 200)
    ax.plot(line_x, line_x, '--', color='gray', alpha=0.7)
    ax.plot(line_x, slope * line_x + intercept, color='red')

    ax.set_title(f'ILD {ild:.0f}, g={slope:.3f}')
    ax.set_xlabel('Rate_r(R, 0)')

axes[0].set_ylabel('Rate_r(R, L)')
fig.suptitle('Paper-style matched-tone scatter fits by ILD')
fig.tight_layout(rect=[0, 0, 1, 0.92])

# %%
plt.figure(figsize=(6, 4))
plt.plot(paper_ilds, paper_gains, marker='o')
plt.title('Gain vs ILD from matched-tone slope fits')
plt.xlabel('ILD')
plt.ylabel('Gain (slope)')
plt.gca().invert_xaxis()
plt.tight_layout()
