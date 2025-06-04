import matplotlib.pyplot as plt
from fit_mutiple_gama_omega_at_once import ABLs_to_fit, ILDs_to_fit

# Assumes gamma_ABL_XX_ILD_YY and omega_ABL_XX_ILD_YY are defined in the imported script
globals_ = globals()

def get_gamma(ABL, ILD):
    return (
        globals_.get('gamma_ABL_20_ILD_1') if ABL == 20 and abs(ILD) == 1 else
        globals_.get('gamma_ABL_20_ILD_4') if ABL == 20 and abs(ILD) == 4 else
        globals_.get('gamma_ABL_20_ILD_16') if ABL == 20 and abs(ILD) == 16 else
        globals_.get('gamma_ABL_60_ILD_1') if ABL == 60 and abs(ILD) == 1 else
        globals_.get('gamma_ABL_60_ILD_4') if ABL == 60 and abs(ILD) == 4 else
        globals_.get('gamma_ABL_60_ILD_16') if ABL == 60 and abs(ILD) == 16 else
        None
    )

def get_omega(ABL, ILD):
    return (
        globals_.get('omega_ABL_20_ILD_1') if ABL == 20 and abs(ILD) == 1 else
        globals_.get('omega_ABL_20_ILD_4') if ABL == 20 and abs(ILD) == 4 else
        globals_.get('omega_ABL_20_ILD_16') if ABL == 20 and abs(ILD) == 16 else
        globals_.get('omega_ABL_60_ILD_1') if ABL == 60 and abs(ILD) == 1 else
        globals_.get('omega_ABL_60_ILD_4') if ABL == 60 and abs(ILD) == 4 else
        globals_.get('omega_ABL_60_ILD_16') if ABL == 60 and abs(ILD) == 16 else
        None
    )

plt.figure(figsize=(8, 4))
for ABL in ABLs_to_fit:
    gammas = []
    ILDs_plot = []
    for ILD in ILDs_to_fit:
        gamma = get_gamma(ABL, ILD)
        if gamma is not None:
            gammas.append(gamma)
            ILDs_plot.append(ILD)
    plt.plot(ILDs_plot, gammas, marker='o', label=f'ABL={ABL}')
plt.xlabel('ILD (dB)')
plt.ylabel('gamma')
plt.title('gamma vs ILD for each ABL')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
for ABL in ABLs_to_fit:
    omegas = []
    ILDs_plot = []
    for ILD in ILDs_to_fit:
        omega = get_omega(ABL, ILD)
        if omega is not None:
            omegas.append(omega)
            ILDs_plot.append(ILD)
    plt.plot(ILDs_plot, omegas, marker='o', label=f'ABL={ABL}')
plt.xlabel('ILD (dB)')
plt.ylabel('omega')
plt.title('omega vs ILD for each ABL')
plt.legend()
plt.tight_layout()
plt.show()
