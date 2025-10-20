# %%
import numpy as np
from scipy import stats
from scipy.integrate import quad

# 1. 
# q: rate norm distribution: gaussian with mean 0.9, std = 0.05
# p: uniform from 0 to 1 ( we will test by changing this distribution say 0 to 2)
# we want KL(q||p)

def compute_kl_divergence_gaussian_uniform(
    q_mean: float,
    q_std: float,
    p_lower: float,
    p_upper: float,
    integration_bounds: tuple = None
) -> float:
    """
    Compute KL(q||p) where:
    - q is a Gaussian distribution with given mean and std
    - p is a uniform distribution from p_lower to p_upper
    
    KL(q||p) = ∫ q(x) log(q(x)/p(x)) dx
    
    Parameters
    ----------
    q_mean : float
        Mean of the Gaussian distribution q
    q_std : float
        Standard deviation of the Gaussian distribution q
    p_lower : float
        Lower bound of the uniform distribution p
    p_upper : float
        Upper bound of the uniform distribution p
    integration_bounds : tuple, optional
        (lower, upper) bounds for numerical integration.
        If None, uses q_mean ± 6*q_std
    
    Returns
    -------
    float
        KL divergence KL(q||p)
    """
    # Define distributions
    q = stats.norm(loc=q_mean, scale=q_std)
    p = stats.uniform(loc=p_lower, scale=p_upper - p_lower)
    
    # Define the integrand: q(x) * log(q(x) / p(x))
    def integrand(x):
        q_x = q.pdf(x)
        p_x = p.pdf(x)
        
        # Avoid log(0) or division by zero
        if q_x < 1e-100:  # q(x) ≈ 0, contribution is negligible
            return 0.0
        if p_x < 1e-100:  # p(x) = 0 but q(x) > 0, KL divergence is infinite
            return np.inf
        
        return q_x * np.log(q_x / p_x)
    
    # Set integration bounds (cover most of the Gaussian mass)
    if integration_bounds is None:
        lower_bound = q_mean - 6 * q_std
        upper_bound = q_mean + 6 * q_std
    else:
        lower_bound, upper_bound = integration_bounds
    
    # Numerical integration
    result, error = quad(integrand, lower_bound, upper_bound, limit=100)
    
    return result


# %%
# Test case 1: q = N(0.9, 0.05), p = U(0, 1.5)
print("=" * 60)
print("Test 1: q = N(0.9, 0.05), p = Uniform(0, 1.5)")
print("=" * 60)

kl_div_1 = compute_kl_divergence_gaussian_uniform(
    q_mean=0.9,
    q_std=0.05,
    p_lower=0.0,
    p_upper=1.5
)
print(f"KL(q||p) = {kl_div_1:.6f}")

# %%
# Test case 2: q = N(0.9, 0.05), p = U(0, 3)
print("\n" + "=" * 60)
print("Test 2: q = N(0.9, 0.05), p = Uniform(0, 3)")
print("=" * 60)

kl_div_2 = compute_kl_divergence_gaussian_uniform(
    q_mean=0.9,
    q_std=0.05,
    p_lower=0.0,
    p_upper=3.0
)
print(f"KL(q||p) = {kl_div_2:.6f}")

# %%
# Comparison
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
print(f"KL divergence decreased by: {kl_div_1 - kl_div_2:.6f}")
print(f"Ratio KL(q||U(0,3)) / KL(q||U(0,1.5)) = {kl_div_2/kl_div_1:.4f}")
print("\nNote: Wider uniform prior → lower KL divergence")
print("      (less information needed to go from q to p)")