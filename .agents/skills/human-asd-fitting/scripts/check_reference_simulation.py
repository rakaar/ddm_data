# %% Independent Euler simulation versus the reference joint RT/choice likelihood.
"""Illustrative baseline check, with no participant data and no fitting.

Generate a DDM using Brownian increments, then add independent Gaussian delays.
Retain DDM trials first and mix lapses among retained trials, matching the
reference likelihood's interpretation of p. Finite time-step error is reported
separately from Monte Carlo uncertainty. This is not parameter recovery.
"""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm
import jax
import jax.numpy as jnp

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL/'assets/reference_code/ASD_Human_Data'))
from gaussian_delay_likelihood_wide import gaussian_delay_log_integral

# %% The agent can change these illustrative settings to validate an agreed model.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--trials', type=int, default=6000)
parser.add_argument('--dt', type=float, default=.0002)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
assert args.trials >= 1000 and 0 < args.dt <= .001
WINDOW = (0., 1.8)
CASES = [dict(gamma=-1.2, omega=2., w=.45, mu=.25, sigma=.06, p=.06, q=.55),
         dict(gamma=0., omega=2., w=.5, mu=.25, sigma=.06, p=.06, q=.55),
         dict(gamma=1.2, omega=2., w=.55, mu=.25, sigma=.06, p=.06, q=.55)]
rng = np.random.default_rng(args.seed)
probes = np.linspace(*WINDOW, 361)
results = []
integral = jax.jit(gaussian_delay_log_integral,
                   static_argnames=('cumulative', 'K', 'order'))


def simulate_retained_ddm(params, n):
    """Independent discretized SDE; does not sample the theoretical CDF."""
    retained_rt, retained_side = [], []
    while sum(len(v) for v in retained_rt) < n:
        batch = max(n, 2000)
        delay = truncnorm.rvs(-params['mu']/params['sigma'], np.inf,
            loc=params['mu'], scale=params['sigma'], size=batch, random_state=rng)
        evidence = np.full(batch, 2*params['w']-1.)
        decision = np.full(batch, np.nan)
        side = np.zeros(batch, dtype=int)
        active = np.flatnonzero(delay < WINDOW[1])
        drift_step = params['gamma']*params['omega']*args.dt
        noise_step = np.sqrt(params['omega']*args.dt)
        for step in range(1, int(np.ceil(WINDOW[1]/args.dt))+1):
            if not len(active):break
            now = step*args.dt
            evidence[active] += drift_step+noise_step*rng.standard_normal(len(active))
            hit = np.abs(evidence[active]) >= 1.
            ids = active[hit]
            decision[ids] = now
            side[ids] = np.where(evidence[ids] > 0, 1, -1)
            # Crossing is observed on a time grid; do not hide this discretization.
            active = active[~hit]
            active = active[delay[active]+now < WINDOW[1]]
        rt = decision+delay
        keep = np.isfinite(rt) & (rt >= WINDOW[0]) & (rt <= WINDOW[1])
        retained_rt.append(rt[keep])
        retained_side.append(side[keep])
    return np.concatenate(retained_rt)[:n], np.concatenate(retained_side)[:n]


# %% Check conditional theory, independent samples, and finite-window joint mass.
for case in CASES:
    simulated_rt, simulated_side = simulate_retained_ddm(case, args.trials)
    lapse = rng.random(args.trials) < case['p']
    simulated_rt[lapse] = rng.uniform(*WINDOW, size=lapse.sum())
    simulated_side[lapse] = np.where(rng.random(lapse.sum()) < case['q'], 1, -1)
    cdf = np.column_stack([np.exp(np.asarray(integral(jnp.asarray(probes),
        case['gamma'], case['omega'], case['w'], case['mu'], case['sigma'],
        side, cumulative=True, K=40, order=24))) for side in [-1, 1]])
    density = np.column_stack([np.exp(np.asarray(integral(jnp.asarray(probes),
        case['gamma'], case['omega'], case['w'], case['mu'], case['sigma'],
        side, cumulative=False, K=40, order=24))) for side in [-1, 1]])
    retained_mass = float(cdf[-1].sum())
    q_sides = np.array([1-case['q'], case['q']])
    joint_cdf = ((1-case['p'])*cdf/retained_mass
        + case['p']*q_sides[None, :]*probes[:, None]/WINDOW[1])
    joint_density = ((1-case['p'])*density/retained_mass
        + case['p']*q_sides[None, :]/WINDOW[1])
    empirical = np.column_stack([np.mean((simulated_rt[:, None] <= probes[None, :])
        & (simulated_side[:, None] == side), axis=0) for side in [-1, 1]])
    cdf_gap = float(np.abs(empirical.sum(1)-joint_cdf.sum(1)).max())
    joint_cdf_gap = float(np.abs(empirical-joint_cdf).max())
    choice_error = float(abs(np.mean(simulated_side == 1)-joint_cdf[-1, 1]))
    choice_se = float(np.sqrt(joint_cdf[-1, 1]*(1-joint_cdf[-1, 1])/args.trials))
    # sqrt(Omega*dt) captures the order of discrete boundary overshoot. This
    # allowance is a coarse simulation diagnostic, not a likelihood-accuracy gate.
    monte_carlo_allowance = float(np.sqrt(np.log(2000)/(2*args.trials)))
    time_step_allowance = float(1.5*np.sqrt(case['omega']*args.dt))
    allowance = monte_carlo_allowance+time_step_allowance
    mass = float(np.trapz(joint_density.sum(1), probes))
    report = dict(parameters=case, trials=args.trials, retained_ddm_mass=retained_mass,
        theory_p_right=float(joint_cdf[-1, 1]), empirical_p_right=float(np.mean(simulated_side == 1)),
        choice_error=choice_error, choice_monte_carlo_se=choice_se,
        marginal_cdf_gap=cdf_gap, joint_cdf_gap=joint_cdf_gap,
        diagnostic_allowance=allowance, monte_carlo_allowance=monte_carlo_allowance,
        time_step_allowance=time_step_allowance, conditional_density_mass=mass)
    assert abs(mass-1) < .001, report
    assert abs(joint_cdf[-1].sum()-1) < 1e-12, report
    assert max(cdf_gap, joint_cdf_gap, choice_error) < allowance, report
    results.append(report)

report = dict(status='passed', independent_simulator='Euler-Maruyama Brownian increments',
    seed=args.seed, dt_s=args.dt, rt_window_s=WINDOW, cases=results,
    limitations=['Illustrative reference-model conditions only; not a fit or parameter recovery.',
                 'Discrete crossings have finite-step bias; refine dt for tighter comparisons.',
                 'The conservative CDF allowance combines Monte Carlo error and a discretization heuristic.',
                 'New likelihood families, priors or windows require corresponding validation.'])
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
