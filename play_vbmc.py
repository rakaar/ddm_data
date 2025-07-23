# %%
# ============================================================
# VBMC on a univariate Gaussian: infer μ (mean) and σ (std dev)
# ============================================================
import numpy as np
import scipy.stats as scs
from pyvbmc import VBMC
import matplotlib.pyplot as plt

# 1.  Synthetic data  ─────────────────────────────────────────
N          = 300                 # sample size
mu_true    = 1.5
sigma_true = 0.7

rng   = np.random.default_rng(0)
data  = rng.normal(mu_true, sigma_true, N)
data = []
# 2.  Model & log‑likelihood  ────────────────────────────────
#    Parameter vector θ = [ μ, log σ ]  (log σ keeps σ > 0 automatically)
def unpack(theta):
    """Convert vector to (mu, sigma). Works with broadcasting."""
    mu        = theta[..., 0]
    log_sigma = theta[..., 1]
    sigma     = np.exp(log_sigma)
    return mu, sigma

def log_likelihood(theta):
    mu, sigma = unpack(np.atleast_2d(theta))
    loglike =  np.sum(scs.norm.logpdf(data, loc=mu[:, None], scale=sigma[:, None]), axis=1)
    print(f'loglike = {loglike}')
    return loglike

# 3.  Prior & log‑joint ──────────────────────────────────────
prior_mu_loc   = 0.0   # fairly wide, uninformative
prior_mu_scale = 5.0
prior_logsig_loc   = -1.0   # centre near σ ≈ e^(−1) ≈ 0.37
prior_logsig_scale = 1.5

def log_prior(theta):
    mu, log_sigma = np.atleast_2d(theta)[:,0], np.atleast_2d(theta)[:,1]
    lp_mu    = scs.norm.logpdf(mu,       loc=prior_mu_loc,     scale=prior_mu_scale)
    lp_lsig  = scs.norm.logpdf(log_sigma,loc=prior_logsig_loc, scale=prior_logsig_scale)
    return lp_mu + lp_lsig

def log_joint(theta):
    return log_likelihood(theta) + log_prior(theta)

# 4.  Bounds, plausible box, start point ─────────────────────
D  = 2
LB = np.array([ -np.inf,   -5.0  ])[None, :]   # μ unbounded, log σ > −5  (≈ σ > 0.007)
UB = np.array([  np.inf,    3.0  ])[None, :]   #               log σ < 3  (≈ σ < 20)

PLB = np.array([prior_mu_loc - prior_mu_scale,
                prior_logsig_loc - prior_logsig_scale])[None, :]
PUB = np.array([prior_mu_loc + prior_mu_scale,
                prior_logsig_loc + prior_logsig_scale])[None, :]

x0 = np.zeros((1,D))   # [0, 0]  (μ≈0, log σ≈0 ⇒ σ≈1)

# 5.  Run VBMC ────────────────────────────────────────────────
vbmc    = VBMC(log_joint, x0, LB, UB, PLB, PUB)
vp, res = vbmc.optimize()

print("\n===== VBMC summary =====")
post_mean, post_cov = vp.moments(cov_flag=True)
print(f"Posterior mean   (μ, σ) ≈ ({post_mean[0]:5.3f}, {np.exp(post_mean[1]):5.3f})")
print(f"True parameters  (μ, σ) = ({mu_true:5.3f}, {sigma_true:5.3f})")
print(f"ELBO  = {res['elbo']:.2f}  ± {res['elbo_sd']:.2f}")
# %%
# 6.  Quick “corner” plot ────────────────────────────────────
samples = vp.sample(100_000)[0]                # draws in (μ, log σ)
mu_samp, sigma_samp = samples[:,0], np.exp(samples[:,1])

plt.figure(figsize=(5,5))
plt.scatter(mu_samp, sigma_samp, s=2, alpha=0.1)
plt.scatter(mu_true, sigma_true, marker='x', s=80, lw=2, label='truth')
plt.xlabel(r'$\mu$'); plt.ylabel(r'$\sigma$')
plt.title('Posterior samples')
plt.legend(); plt.tight_layout()
plt.show()

plt.figure(figsize=(9,3))
plt.subplot(1,2,1)
plt.hist(mu_samp, bins=60, density=True, alpha=0.5)
plt.axvline(np.mean(mu_samp), color='b', lw=2, label='mean')
plt.axvline(mu_true, lw=2, label='truth', color='g'); plt.legend()
plt.xlabel(r'$\mu$'); plt.title('Marginal posterior of μ')

plt.subplot(1,2,2)
plt.hist(sigma_samp, bins=60, density=True, alpha=0.5)
plt.axvline(np.mean(sigma_samp), color='b', lw=2, label='mean')
plt.axvline(sigma_true, lw=2, label='truth', color='g'); plt.legend()
plt.xlabel(r'$\sigma$'); plt.title('Marginal posterior of σ')
plt.tight_layout(); plt.show()
