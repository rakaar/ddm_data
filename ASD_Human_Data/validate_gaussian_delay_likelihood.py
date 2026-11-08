# %%
"""Three synthetic checks of random Gaussian sensory delay; no human fitting.

Run from repository root with .venv/bin/python. Sources, simulated samples,
numerical checks, timing and an overlay figure are saved in a separate folder.
"""
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ddm-gaussian-delay-matplotlib')
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time as clock

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import truncnorm
from numba import njit, prange, set_num_threads
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from gaussian_delay_likelihood import (
    gaussian_delay_grid, gaussian_delay_logpdf_quadrature,
    gaussian_delay_log_likelihood, ddm_grid_density,
)
from human_plain_ddm_diagnostics_utils import scipy_ddm, ddm_log_density

# %% Editable settings: deliberately broad delay distributions, varied drift/bias.
ROOT = Path(__file__).resolve().parent
CASES = [
    dict(name='Zero drift', gamma=0., omega=2.5, w=.50, delay_mu=.28, delay_sigma=.12),
    dict(name='Positive drift', gamma=1.8, omega=3.5, w=.38, delay_mu=.24, delay_sigma=.18),
    dict(name='Negative drift', gamma=-3., omega=1.4, w=.62, delay_mu=.40, delay_sigma=.20),
]
FIELDS = ('gamma', 'omega', 'w', 'delay_mu', 'delay_sigma')
HORIZON = 3.6
WINDOW = (0., 1.8)
GRID_DT = .0005
BIN_DT = .02
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--trials', type=int, default=200_000, help='per parameter set and simulation timestep')
parser.add_argument('--seed', type=int, default=20261022)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--output-dir', type=Path, default=ROOT/'gaussian_delay_validation')
parser.add_argument('--skip-simulation', action='store_true', help='reuse this folder\'s saved simulations')
args = parser.parse_args()
out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
set_num_threads(args.threads)


# %% Independent diffusion paths. No likelihood/CDF is used to generate exits.
@njit(parallel=True, cache=True)
def simulate_paths(n, gamma, omega, w, mu, sigma, ds, seed):
    """Dimensionless dX=gamma*ds+sqrt(ds)*N(0,1), absorbing bounds +/-1.

    Brownian-bridge crossing probabilities catch paths that cross a boundary
    and return between sampled endpoints. The probability of crossing BOTH
    boundaries in a single interval is negligible at ds <= .001 (width=2).
    Crossing time within the detected interval uses its midpoint; repeating
    with ds/2 checks this finite-time discretization. No paths are time-censored.
    """
    rt = np.empty(n)
    choice = np.empty(n, dtype=np.int8)
    delays = np.empty(n)
    decision = np.empty(n)
    root = np.sqrt(ds)
    for i in prange(n):
        np.random.seed((seed+i) % 2147483647)
        delay = mu+sigma*np.random.normal()
        while delay < 0:
            delay = mu+sigma*np.random.normal()
        x = 2*w-1
        step = 0
        side = 0
        while side == 0:
            xn = x+gamma*ds+root*np.random.normal()
            if xn >= 1:
                side = 1
            elif xn <= -1:
                side = -1
            else:
                upper = np.exp(-2*(1-x)*(1-xn)/ds)
                lower = np.exp(-2*(1+x)*(1+xn)/ds)
                u = np.random.random()
                if u < upper:
                    side = 1
                elif u < upper+lower:
                    side = -1
            x = xn
            step += 1
        decision[i] = (step-.5)*ds/omega
        delays[i] = delay
        rt[i] = delay+decision[i]
        choice[i] = side
    return rt, choice, delays, decision


def reference_convolution(t, p, side, cumulative=False):
    """Independent adaptive SciPy integral using the NumPy first-passage code."""
    if t <= 0:
        return 0.
    mu, sd = p['delay_mu'], p['delay_sigma']
    normalizer = sd*np.sqrt(2*np.pi)*ndtr(mu/sd)
    def integrand(delay):
        base = scipy_ddm(t-delay, p['gamma'], p['omega'], p['w'], 0.,
                         cumulative=cumulative, side=side, K=80)
        return float(base)*np.exp(-.5*((delay-mu)/sd)**2)/normalizer
    # Split near Gaussian peak and DDM image/eigen switch for adaptive integration.
    points = sorted(set(x for x in [mu, mu-3*sd, mu+3*sd, t-.6/p['omega']]
                        if 0 < x < t))
    value, error = quad(integrand, 0., t, points=points, epsabs=2e-11,
                        epsrel=2e-10, limit=200)
    return value


def timed_jit(fun, *inputs, repeats=12):
    start = clock.perf_counter()
    result = fun(*inputs)
    jax.block_until_ready(result)
    compile_s = clock.perf_counter()-start
    timings = []
    for _ in range(repeats):
        start = clock.perf_counter()
        result = fun(*inputs)
        jax.block_until_ready(result)
        timings.append(clock.perf_counter()-start)
    return result, dict(first_call_including_compile_s=compile_s,
                        median_ms=1000*float(np.median(timings)),
                        p10_ms=1000*float(np.percentile(timings, 10)),
                        p90_ms=1000*float(np.percentile(timings, 90)))


# %% Simulate both timesteps before comparing to any model curves.
records = []
samples = []
start_all = clock.perf_counter()
for i, p in enumerate(CASES):
    runs = []
    for resolution, ds in [('coarse', .001), ('fine', .0005)]:
        path = out/f'case_{i+1}_{resolution}_simulation.npz'
        start = clock.perf_counter()
        seed = args.seed+1_000_000*i+(100_000_000 if resolution == 'fine' else 0)
        if not args.skip_simulation:
            arrays = simulate_paths(args.trials, *[p[k] for k in FIELDS], ds, seed)
            np.savez_compressed(path, rt=arrays[0], choice=arrays[1], delay=arrays[2],
                                decision=arrays[3], seed=seed, dimensionless_dt=ds,
                                parameters=np.array([p[k] for k in FIELDS]),
                                parameter_order=np.array(FIELDS))
        saved = dict(np.load(path))
        assert len(saved['rt']) == args.trials
        assert float(saved['dimensionless_dt']) == ds and int(saved['seed']) == seed
        np.testing.assert_array_equal(saved['parameters'], [p[k] for k in FIELDS])
        assert tuple(saved['parameter_order']) == FIELDS
        assert np.all(saved['delay'] >= 0.)
        np.testing.assert_allclose(saved['rt'], saved['delay']+saved['decision'])
        elapsed = clock.perf_counter()-start
        print(f"Case {i+1} {resolution}: {args.trials:,} independent paths in {elapsed:.2f} s", flush=True)
        runs.append(saved)
    samples.append(runs)

# %% Three numerical routes: fast FFT, positive log-quadrature, adaptive SciPy.
params = {name:jnp.asarray([p[name] for p in CASES]) for name in FIELDS}
grid = jnp.linspace(0., HORIZON, round(HORIZON/GRID_DT)+1)
grid_fn = jax.jit(gaussian_delay_grid, static_argnames=('K',))
f40, c40 = grid_fn(params, grid, K=40)
f80, c80 = grid_fn(params, grid, K=80)
f12, c12 = grid_fn(params, grid, K=12)
f8, c8 = grid_fn(params, grid, K=8)
f40, c40, f80, c80, f12, f8 = map(np.asarray, [f40, c40, f80, c80, f12, f8])
np.testing.assert_allclose(f40, f80, rtol=2e-11, atol=2e-12)
np.testing.assert_allclose(c40, c80, rtol=2e-11, atol=2e-12)
np.testing.assert_allclose(f12, f40, rtol=2e-11, atol=2e-12)
np.testing.assert_allclose(f8, f40, rtol=2e-11, atol=2e-12)
# Kernel-only stress check: optimizations must preserve the established Wiener
# primitive over much wider parameters than the three simulated examples.
rng = np.random.default_rng(args.seed)
stress_gamma = np.r_[[-20.,0.,20.],rng.uniform(-20,20,61)]
stress_omega = np.exp(rng.uniform(np.log(.01),np.log(15.),64))
stress_w = np.r_[[.001,.5,.999],rng.uniform(.01,.99,61)]
stress_t = np.geomspace(1e-5,100.,300)
optimized_kernel = np.asarray(ddm_grid_density(jnp.asarray(stress_t),jnp.asarray(stress_gamma),
    jnp.asarray(stress_omega),jnp.asarray(stress_w),K=8))
legacy_kernel = np.exp(np.asarray(ddm_log_density(stress_t[None,None,:],
    stress_gamma[:,None,None],stress_omega[:,None,None],stress_w[:,None,None],0.,80,
    side=jnp.array([-1,1])[None,:,None])))
np.testing.assert_allclose(optimized_kernel,legacy_kernel,atol=2e-10,rtol=2e-10)
kernel_relative = float(np.max(np.abs(optimized_kernel-legacy_kernel)[legacy_kernel>1e-200]
                              /legacy_kernel[legacy_kernel>1e-200]))
fine_grid = jnp.linspace(0., HORIZON, round(HORIZON/(GRID_DT/2))+1)
f_fine, c_fine = map(np.asarray, grid_fn(params, fine_grid, K=40))
coarse_grid = jnp.linspace(0., HORIZON, round(HORIZON/(2*GRID_DT))+1)
f_coarse, c_coarse = map(np.asarray, grid_fn(params, coarse_grid, K=40))
mass_grid = jnp.linspace(0., 8., round(8./GRID_DT)+1)
_, mass_cdf = grid_fn(params, mass_grid, K=8)
mass_cdf = np.asarray(mass_cdf)
grid = np.asarray(grid)
reference_times = np.unique(np.r_[np.linspace(.02, 1.8, 47), [2., 2.5, 3., 3.6]])
reference_rows, curve_rows, histogram_rows = [], [], []
bins = np.arange(0., HORIZON+BIN_DT/2, BIN_DT)
centers = (bins[1:]+bins[:-1])/2
fig, axes = plt.subplots(3, 4, figsize=(19, 11), constrained_layout=True)
colors = ['#1f77b4', '#ff7f0e']

for i, p in enumerate(CASES):
    reference = np.array([[reference_convolution(t, p, side) for t in reference_times]
                          for side in (-1, 1)])
    reference_cdf = np.array([[reference_convolution(t, p, side, True) for t in reference_times]
                              for side in (-1, 1)])
    fft_at = np.array([np.interp(reference_times, grid, row) for row in f40[i]])
    fft_cdf_at = np.array([np.interp(reference_times, grid, row) for row in c40[i]])
    q96 = np.array([jnp.exp(gaussian_delay_logpdf_quadrature(
        reference_times, *[p[k] for k in FIELDS[:3]], p['delay_mu'], p['delay_sigma'],
        side, order=96)) for side in (-1, 1)])
    meaningful = reference > 1e-5
    numerical = dict(
        fft_max_absolute_density_error=float(np.max(np.abs(fft_at-reference))),
        fft_max_relative_density_error_where_pdf_gt_1e_minus5=float(np.max(np.abs(fft_at-reference)[meaningful]/reference[meaningful])),
        fft_max_absolute_cdf_error=float(np.max(np.abs(fft_cdf_at-reference_cdf))),
        quadrature96_max_absolute_density_error=float(np.max(np.abs(q96-reference))),
        K40_vs_K80_max_absolute_density_difference=float(np.max(np.abs(f40[i]-f80[i]))),
        K12_vs_K40_max_absolute_density_difference=float(np.max(np.abs(f12[i]-f40[i]))),
        K8_vs_K40_max_absolute_density_difference=float(np.max(np.abs(f8[i]-f40[i]))),
        grid_1ms_vs_halfms_max_density_difference=float(np.max(np.abs(f_coarse[i]-f40[i,:,::2]))),
        grid_halfms_vs_quarterms_max_density_difference=float(np.max(np.abs(f40[i]-f_fine[i,:,::2]))),
    )
    assert numerical['fft_max_absolute_density_error'] < 1e-4
    assert numerical['fft_max_absolute_cdf_error'] < 1e-5
    assert numerical['quadrature96_max_absolute_density_error'] < 1e-7
    moments = truncnorm.stats(-p['delay_mu']/p['delay_sigma'], np.inf,
                             loc=p['delay_mu'], scale=p['delay_sigma'], moments='mv')
    actual_mean, actual_sd = float(moments[0]), float(np.sqrt(moments[1]))
    fixed_total = scipy_ddm(grid, p['gamma'], p['omega'], p['w'], actual_mean)
    fixed_cdf = scipy_ddm(grid, p['gamma'], p['omega'], p['w'], actual_mean, cumulative=True)
    cdf_total = c40[i].sum(axis=0)
    Z = float(np.interp(WINDOW[1], grid, cdf_total))
    right_mass = float(np.interp(WINDOW[1], grid, c40[i,1]))
    eventual_right = (p['w'] if p['gamma']==0 else
                      np.expm1(-4*p['gamma']*p['w'])/np.expm1(-4*p['gamma']))
    eventual = np.array([1-eventual_right, eventual_right])
    full_mass_error = float(np.max(np.abs(mass_cdf[i,:,-1]-eventual)))
    assert full_mass_error < 1e-6
    # Numerically integrate conditional densities; do not normalize each bound alone.
    window_ids = grid <= WINDOW[1]
    conditional_mass = np.trapz(f40[i][:,window_ids].sum(axis=0)/Z, grid[window_ids])
    assert abs(conditional_mass-1) < 1e-10
    simulation_checks = []
    for resolution, saved in zip(('coarse', 'fine'), samples[i]):
        n = len(saved['rt'])
        empirical = np.stack([np.searchsorted(np.sort(saved['rt'][saved['choice']==side]), grid,
                                              side='right')/n for side in (-1,1)])
        total_gap = float(np.max(np.abs(empirical.sum(axis=0)-cdf_total)))
        choice_gap = float(np.max(np.abs(empirical-c40[i])))
        # Conservative 99.9% DKW bound for three CDFs (union bound), plus 0.001
        # for finite simulation steps. Sub-CDFs treat the other choice as +inf.
        sampling_bound = float(np.sqrt(np.log(6/.001)/(2*n)))
        assert max(total_gap, choice_gap) < sampling_bound+.001
        delay_mean_se = actual_sd/np.sqrt(n)
        assert abs(saved['delay'].mean()-actual_mean) < 5*delay_mean_se
        keep = saved['rt'] <= WINDOW[1]
        pright = float(np.mean(saved['choice'][keep] == 1))
        ptheory = right_mass/Z
        se = float(np.sqrt(ptheory*(1-ptheory)/keep.sum()))
        assert abs(pright-ptheory) < 5*se+.001
        simulation_checks.append(dict(resolution=resolution, trials=n,
            total_cdf_max_gap=total_gap, choice_subcdf_max_gap=choice_gap,
            dkw_99p9_percent_union_bound=sampling_bound, retained_trials=int(keep.sum()),
            retained_fraction=float(keep.mean()), expected_retained_fraction=Z,
            empirical_right_probability_in_window=pright, theory_right_probability_in_window=ptheory,
            binomial_standard_error=se))
    record = dict(case=i+1, **p, actual_delay_mean_s=actual_mean, actual_delay_sd_s=actual_sd,
                  negative_mass_removed=float(ndtr(-p['delay_mu']/p['delay_sigma'])),
                  retention_probability=Z, conditional_mass=float(conditional_mass),
                  full_mass_by_display_horizon=float(cdf_total[-1]),
                  full_mass_by_8s=float(mass_cdf[i,:,-1].sum()),
                  max_error_against_eventual_bound_masses=full_mass_error,
                  random_vs_same_mean_fixed_delay_cdf_gap=float(np.max(np.abs(cdf_total-fixed_cdf))),
                  numerical=numerical, simulation=simulation_checks)
    records.append(record)
    print(json.dumps(record), flush=True)
    for s, side in enumerate((-1, 1)):
        for j, t in enumerate(reference_times):
            reference_rows.append(dict(case=i+1, choice=side, rt=t, scipy_pdf=reference[s,j],
                fft_pdf=fft_at[s,j], quadrature96_pdf=q96[s,j], scipy_cdf=reference_cdf[s,j],
                fft_cdf=fft_cdf_at[s,j]))
    for j, t in enumerate(grid):
        curve_rows.append(dict(case=i+1, rt=t, left_pdf=f40[i,0,j], right_pdf=f40[i,1,j],
            left_cdf=c40[i,0,j], right_cdf=c40[i,1,j], fixed_mean_total_pdf=fixed_total[j]))
    fine = samples[i][1]
    n = len(fine['rt'])
    delay_x = np.linspace(0, 1.1, 551)
    axes[i,0].hist(fine['delay'], bins=np.arange(0, 1.12, .02), density=False,
                    weights=np.full(n, 1/(n*.02)), histtype='step', color='#888888', label='Simulation')
    axes[i,0].plot(delay_x, truncnorm.pdf(delay_x, -p['delay_mu']/p['delay_sigma'], np.inf,
                         loc=p['delay_mu'], scale=p['delay_sigma']), color='#7b3294', label='Truncated Gaussian')
    axes[i,0].axvline(actual_mean, color='black', ls=':', label='Actual delay mean')
    axes[i,0].set_xlim(0, 1.1)
    axes[i,0].set_title(f"{p['name']} | Γ={p['gamma']:g}, Ω={p['omega']:g}, w={p['w']:g}\n"
                       f"Gaussian μ={p['delay_mu']*1000:.0f}, σ={p['delay_sigma']*1000:.0f} ms")
    counts = np.histogram(fine['rt'], bins=bins)[0]
    axes[i,1].stairs(counts/(n*BIN_DT), bins, color='#999999', label='Simulated RTs')
    axes[i,1].plot(grid, f40[i].sum(axis=0), color='#222222', label='Random-delay likelihood')
    axes[i,1].plot(grid, fixed_total, color='#009E73', ls='--', label='Fixed delay, same mean')
    axes[i,1].set_title('RT distribution | both choices')
    for s, side in enumerate((-1,1)):
        bound_counts = np.histogram(fine['rt'][fine['choice']==side], bins=bins)[0]
        empirical_cdf = np.searchsorted(np.sort(fine['rt'][fine['choice']==side]), grid, side='right')/n
        label = 'Left' if side == -1 else 'Right'
        axes[i,2].stairs(bound_counts/(n*BIN_DT), bins, color=colors[s], alpha=.45)
        axes[i,2].plot(grid, f40[i,s], color=colors[s], label=label)
        axes[i,3].plot(grid, empirical_cdf, color=colors[s], alpha=.45, lw=3)
        axes[i,3].plot(grid, c40[i,s], color=colors[s], ls='--', label=label)
        for j, center in enumerate(centers):
            expected_prob = np.interp(bins[j+1],grid,c40[i,s])-np.interp(bins[j],grid,c40[i,s])
            histogram_rows.append(dict(case=i+1, choice=side, bin_left=bins[j], bin_right=bins[j+1],
                count=int(bound_counts[j]), denominator=n, joint_density=bound_counts[j]/(n*BIN_DT),
                model_bin_probability=expected_prob, expected_count=n*expected_prob))
    axes[i,2].set_title('Joint RT + choice densities')
    axes[i,3].set_title('Choice CDFs | simulation / likelihood')
    sim = simulation_checks[1]
    axes[i,3].text(.98,.23,f"P(right | RT ≤ 1.8 s)\nSimulation {sim['empirical_right_probability_in_window']:.4f}\n"
                   f"Likelihood {sim['theory_right_probability_in_window']:.4f}", transform=axes[i,3].transAxes,
                   ha='right', va='bottom', fontsize=9,
                   bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
    for j in range(1,4):
        axes[i,j].set_xlim(0, 2.4)
    for j in range(4):
        axes[i,j].grid(alpha=.15)
        axes[i,j].set_xlabel('Delay (s)' if j==0 else 'RT (s)')
        axes[i,j].set_ylabel('Probability' if j==3 else 'Density (s⁻¹)')
        axes[i,j].spines[['top','right']].set_visible(False)
        if i==0: axes[i,j].legend(fontsize=8, frameon=False)
fig.suptitle(f"Trial-varying Gaussian delay ≥ 0 | independent simulation versus likelihood\n"
             f"{args.trials:,} fine-step paths per case | no upper delay cutoff | raw densities, no display-window renormalization",
             fontsize=15)
fig.savefig(out/'gaussian_delay_simulation_validation.png', dpi=170)
fig.savefig(out/'gaussian_delay_simulation_validation.pdf')
plt.close(fig)
pd.DataFrame(reference_rows).to_csv(out/'independent_quadrature.csv', index=False)
pd.DataFrame(curve_rows).to_csv(out/'likelihood_curves.csv', index=False)
pd.DataFrame(histogram_rows).to_csv(out/'simulation_histograms.csv', index=False)

# %% Differentiability and fitting-sized timing use only synthetic trials.
# Twenty conditions / 6,600 trials is a representative group-sized objective.
pattern = np.arange(20)%3
bench_params = {name:jnp.asarray([CASES[i][name] for i in pattern]) for name in FIELDS}
bench_params['w'] = jnp.asarray(.5)  # the shared bias used by the group fitter
rt, choices, ids = [], [], []
for i, case_id in enumerate(pattern):
    s = samples[case_id][1]
    keep = np.flatnonzero((s['rt'] >= WINDOW[0]) & (s['rt'] <= WINDOW[1]))[:330]
    rt.extend(s['rt'][keep]); choices.extend(s['choice'][keep]); ids.extend([i]*len(keep))
data = dict(rt=jnp.asarray(rt), choice=jnp.asarray(choices), condition=jnp.asarray(ids),
            window_low=WINDOW[0], window_high=WINDOW[1])
bench_grid = jnp.linspace(0, WINDOW[1], round(WINDOW[1]/GRID_DT)+1)
benchmarks = {}
for K in (40, 12, 8):
    objective = jax.jit(jax.value_and_grad(lambda p: gaussian_delay_log_likelihood(p, data, bench_grid, K)))
    (value, gradient), timing = timed_jit(objective, bench_params)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in gradient.values())
    benchmarks[f'K{K}_joint_value_and_gradient'] = dict(timing, log_likelihood=float(value),
        trials=len(rt), conditions=20, physical_parameters=81, finite_gradients=True)
    print(f"Timing K={K}: {timing}", flush=True)

# The numeric-series optimization is checked on the actual objective gradient,
# not only the plotted curves.
objective40=jax.jit(jax.value_and_grad(lambda p:gaussian_delay_log_likelihood(p,data,bench_grid,40)))
objective8=jax.jit(jax.value_and_grad(lambda p:gaussian_delay_log_likelihood(p,data,bench_grid,8)))
value40,grad40=objective40(bench_params)
value8,grad8=objective8(bench_params)
np.testing.assert_allclose(value40,value8,rtol=1e-12,atol=1e-8)
for name in FIELDS:
    np.testing.assert_allclose(grad40[name],grad8[name],rtol=1e-9,atol=1e-8)

# Compare the interpolated FFT likelihood to direct positive quadrature on all
# 6,600 synthetic observations (20 genuinely distinct condition indices).
ids_array=data['condition']
def direct_full(p):
    probability=jnp.exp(gaussian_delay_logpdf_quadrature(data['rt'],p['gamma'][ids_array],
        p['omega'][ids_array],p['w'],p['delay_mu'][ids_array],p['delay_sigma'][ids_array],
        data['choice'],K=8))
    _,cdf=gaussian_delay_grid(p,bench_grid,K=8)
    return jnp.sum(jnp.log(probability)-jnp.log(cdf[:,:,-1].sum(axis=1)[ids_array]))
direct_value,direct_timing=timed_jit(jax.jit(direct_full),bench_params,repeats=3)
fft_value,fft_value_timing=timed_jit(jax.jit(lambda p:gaussian_delay_log_likelihood(
    p,data,bench_grid,8)),bench_params)
per_trial_error=abs(float(value8-direct_value))/len(rt)
assert per_trial_error < 1e-4
benchmarks['direct_quadrature96_value_only']=dict(direct_timing,log_likelihood=float(direct_value))
benchmarks['fft_K8_value_only']=dict(fft_value_timing,log_likelihood=float(fft_value))
benchmarks['fft_vs_direct_mean_log_likelihood_error_per_trial']=per_trial_error

# All five derivatives checked at each of the three parameter sets, against
# finite differences of independent adaptive integration (density and mass).
gradient_checks = []
for i, p in enumerate(CASES):
    vec = jnp.array([p[n] for n in FIELDS])
    probe_t, probe_side = .65, 1 if p['gamma'] >= 0 else -1
    def single_objective(x):
        q = {name:x[j:j+1] for j,name in enumerate(FIELDS)}
        d = dict(rt=jnp.array([probe_t]), choice=jnp.array([probe_side]),
                 condition=jnp.array([0]), window_low=0., window_high=1.8)
        return gaussian_delay_log_likelihood(q, d, bench_grid, K=12)
    gradient = np.asarray(jax.jit(jax.grad(single_objective))(vec))
    finite_difference = []
    for j, name in enumerate(FIELDS):
        h = 1e-5 if name in ('delay_mu','delay_sigma','w') else 1e-4
        vals=[]
        for sign in (-1,1):
            q = dict(p); q[name] += sign*h
            numerator = reference_convolution(probe_t,q,probe_side)
            mass = sum(reference_convolution(1.8,q,side,True) for side in (-1,1))
            vals.append(np.log(numerator/mass))
        finite_difference.append((vals[1]-vals[0])/(2*h))
    error = np.max(np.abs(gradient-finite_difference))
    assert error < 2e-4
    gradient_checks.append(dict(case=i+1, parameter_order=FIELDS, autodiff=gradient.tolist(),
        scipy_finite_difference=finite_difference, max_absolute_error=float(error)))

# Choice-label invariance for the marginal objective; tiny-density fallback.
small_data={k:(v[:60] if k in ('rt','choice','condition') else v) for k,v in data.items()}
original = gaussian_delay_log_likelihood(bench_params, small_data, bench_grid, K=12, marginal=True)
flipped = gaussian_delay_log_likelihood(bench_params, dict(small_data,choice=-small_data['choice']),
                                      bench_grid, K=12, marginal=True)
np.testing.assert_allclose(original,flipped,atol=0,rtol=0)
tail_params={name:jnp.array([value]) for name,value in
             dict(gamma=1.,omega=2.,w=.5,delay_mu=.35,delay_sigma=.03).items()}
tail_data=dict(rt=jnp.array([.04,.08,.12]),choice=jnp.array([-1,1,1]),condition=jnp.zeros(3,dtype=int),
               window_low=0.,window_high=1.8)
tail_value, tail_gradient = jax.jit(jax.value_and_grad(gaussian_delay_log_likelihood))(
    tail_params,tail_data,bench_grid)
assert np.isfinite(tail_value) and all(np.isfinite(np.asarray(v)).all() for v in tail_gradient.values())
tail_logpdfs = np.asarray(gaussian_delay_logpdf_quadrature(
    tail_data['rt'],1.,2.,.5,.35,.03,tail_data['choice'],order=192))
_,tail_cdf = gaussian_delay_grid(tail_params,bench_grid)
np.testing.assert_allclose(float(tail_value),tail_logpdfs.sum()-3*np.log(np.asarray(tail_cdf)[0,:,-1].sum()),
                           rtol=1e-9,atol=1e-7)

# %% Audit record and concise results, including computational limitations.
source_paths=[Path(__file__), ROOT/'gaussian_delay_likelihood.py', ROOT/'human_plain_ddm_diagnostics_utils.py']
report=dict(status='passed', data='Synthetic only; no human fits or trial selections changed',
    delay_distribution='Normal(mu,sigma) conditional on delay>=0; no upper cutoff',
    stochastic_delay='one independent draw per trial, independent of the diffusion path',
    params_are_underlying_gaussian=True, choices=[-1,1], dimensionless_bounds=[-1,1],
    physical_diffusion='dX = gamma*omega*dt + sqrt(omega)*dW; start=2*w-1',
    seed=args.seed, trials_per_case_per_timestep=args.trials, simulation_dimensionless_steps=[.001,.0005],
    grid_dt_s=GRID_DT, grid_method='zero-padded FFT, trapezoidal integration, linear density interpolation',
    kernel_stress_check=dict(parameter_sets=64,time_points=300,K_fast=8,K_reference=80,
        max_absolute_error=float(np.max(np.abs(optimized_kernel-legacy_kernel))),
        max_relative_error_where_pdf_gt_1e_minus200=kernel_relative),
    normalization='Integrate over delays first, then divide both choice densities by the same retained either-bound mass',
    cases=records, gradient_checks=gradient_checks, benchmarks=benchmarks,
    marginal_choice_invariance=True, tiny_density_fallback_value=float(tail_value),
    tiny_density_fallback_finite_gradients=True,
    timing_environment=dict(platform=platform.platform(), processor=platform.processor(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), jax_version=jax.__version__, x64=bool(jax.config.x64_enabled)),
    source_sha256={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    simulation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('case_*_simulation.npz'))},
    limitations=['Three synthetic regimes establish these checks, not accuracy throughout every possible prior.',
                 'Recheck grid and quadrature resolution before fitting much narrower delays, extreme bias or very fast decisions.',
                 'Benchmarks are warmed likelihood-plus-gradient evaluations, not full SVI updates or parameter recovery.',
                 'Simulation uses Brownian bridge crossings and interval midpoint exit times; both timesteps are audited.'],
    elapsed_s=clock.perf_counter()-start_all)
(out/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
pd.DataFrame([{k:v for k,v in r.items() if k not in ('numerical','simulation')} for r in records]).to_csv(out/'parameter_cases.csv',index=False)
print(f"PASS: saved three-case validation to {out}",flush=True)
