# Group plots for gamma and omega vs ILD for all animals (not to PDF)
# Run this after running the main script so gamma_all_animals and omega_all_animals are available (or import them from a pickle if needed)

import matplotlib.pyplot as plt
import numpy as np

# If running standalone, load these from the main script's namespace or pickle
# from diagnostics_multiple_gamma_omega_fit_at_once import gamma_all_animals, omega_all_animals, ABLs_to_fit
# Or, if running in the same session, just use the variables

all_ILDs = [1,2,4,8,16]

# --- Plot gamma vs ILD for each ABL ---
plt.figure(figsize=(6,4))
for ABL in ABLs_to_fit:
    n_animals = len(next(iter(gamma_all_animals[ABL].values())))
    for i in range(n_animals):
        y = [gamma_all_animals[ABL][ILD][i] if i < len(gamma_all_animals[ABL][ILD]) else np.nan for ILD in all_ILDs]
        plt.plot(all_ILDs, y, color='skyblue', alpha=0.5, linewidth=1)
    mean = [np.nanmean(gamma_all_animals[ABL][ILD]) for ILD in all_ILDs]
    plt.plot(all_ILDs, mean, color='navy', marker='o', linewidth=2, label=f'ABL={ABL}')
plt.xlabel('ILD (dB)')
plt.ylabel('gamma')
plt.title('gamma vs ILD (all animals)')
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot omega vs ILD for each ABL ---
plt.figure(figsize=(6,4))
for ABL in ABLs_to_fit:
    n_animals = len(next(iter(omega_all_animals[ABL].values())))
    for i in range(n_animals):
        y = [omega_all_animals[ABL][ILD][i] if i < len(omega_all_animals[ABL][ILD]) else np.nan for ILD in all_ILDs]
        plt.plot(all_ILDs, y, color='lightcoral', alpha=0.5, linewidth=1)
    mean = [np.nanmean(omega_all_animals[ABL][ILD]) for ILD in all_ILDs]
    plt.plot(all_ILDs, mean, color='darkred', marker='o', linewidth=2, label=f'ABL={ABL}')
plt.xlabel('ILD (dB)')
plt.ylabel('omega')
plt.title('omega vs ILD (all animals)')
plt.legend()
plt.tight_layout()
plt.show()
