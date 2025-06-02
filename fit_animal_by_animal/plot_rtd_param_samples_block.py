# === RTD plot: 5 param samples per panel (NO EMPIRICAL) ===
print('\nPlotting 5 sampled-theory RTDs per (ABL, |ILD|) panel...')

import random

abs_ILD_arr = [abs(ild) for ild in ILD_arr]
abs_ILD_arr = sorted(list(set(abs_ILD_arr)))
max_xlim_RT = 0.6
fig, axes = plt.subplots(len(ABL_arr), len(abs_ILD_arr), figsize=(10,6), sharex=True, sharey=True)
for ax_row in axes:
    for ax in ax_row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# For each (ABL, |ILD|), plot 5 sampled-theory RTDs (averaged across animals)
for i, abl in enumerate(ABL_arr):
    for j, abs_ild in enumerate(abs_ILD_arr):
        all_sampled_rtds = [[] for _ in range(5)]  # 5 lists, one per sample
        t_pts = None
        for ild in [abs_ild, -abs_ild]:
            stim_key = (abl, ild)
            for batch_animal_pair, animal_data in rtd_data.items():
                if stim_key in animal_data:
                    # --- get the posterior samples for this animal ---
                    batch_name, animal_id = batch_animal_pair
                    # Load posterior samples
                    try:
                        filename = os.path.join(psycho_fits_repo_path, f'psycho_fit_{batch_name}_{animal_id}.pkl')
                        with open(filename, 'rb') as f:
                            vp = pickle.load(f)
                        vp = vp.vp
                        samples = vp.sample(int(1e6))[0]  # shape (N, 6)
                    except Exception as e:
                        print(f"Error loading samples for {batch_name}, {animal_id}: {e}")
                        continue
                    # --- get abort params ---
                    results_pkl = f'results_{batch_name}_animal_{animal_id}.pkl'
                    with open(results_pkl, 'rb') as f:
                        fit_results_data = pickle.load(f)
                    abort_keyname = "vbmc_aborts_results"
                    abort_params = {}
                    if abort_keyname in fit_results_data:
                        abort_samples = fit_results_data[abort_keyname]
                        abort_params['V_A'] = np.mean(abort_samples['V_A_samples'])
                        abort_params['theta_A'] = np.mean(abort_samples['theta_A_samples'])
                        abort_params['t_A_aff'] = np.mean(abort_samples['t_A_aff_samp'])
                    else:
                        continue
                    # --- get P_A, C_A, t_stim_samples ---
                    P_A_mean, C_A_mean, t_stim_samples = get_P_A_C_A(batch_name, int(animal_id), abort_params)
                    # --- sample 5 indices ---
                    n_samples = samples.shape[0]
                    idxs = random.sample(range(n_samples), 5)
                    for k, idx in enumerate(idxs):
                        param_set = {
                            'rate_lambda': samples[idx,0],
                            'T_0': samples[idx,1],
                            'theta_E': samples[idx,2],
                            'w': samples[idx,3],
                            't_E_aff': samples[idx,4],
                            'del_go': samples[idx,5],
                        }
                        # Use the same theoretical RTD machinery, but with this param set
                        try:
                            t_pts_0_1, rtd = get_theoretical_RTD_from_params(
                                P_A_mean, C_A_mean, t_stim_samples, abort_params, param_set, 0, False, abl, ild
                            )
                            if t_pts is None:
                                t_pts = t_pts_0_1
                            all_sampled_rtds[k].append(rtd)
                        except Exception as e:
                            print(f"Error computing RTD for {batch_name},{animal_id}, sample {idx}: {e}")
                            continue
        ax = axes[i, j]
        # Plot 5 curves (mean across animals for each param sample)
        colors = ['r', 'g', 'b', 'm', 'orange']
        for k in range(5):
            if all_sampled_rtds[k] and t_pts is not None:
                avg_rtd = np.nanmean(all_sampled_rtds[k], axis=0)
                ax.plot(t_pts, avg_rtd, color=colors[k % len(colors)], alpha=0.7, lw=2, label=f'Sample {k+1}')
        if i == len(ABL_arr) - 1:
            ax.set_xlabel('RT (s)', fontsize=12)
            ax.set_xticks([-0.1, max_xlim_RT])
            ax.set_xticklabels(['-0.1', max_xlim_RT], fontsize=12)
        ax.set_xlim(-0.1, max_xlim_RT)
        ax.set_yticks([0, 12])
        ax.set_ylim(0, 14)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        if j == 0:
            ax.set_ylabel(f'ABL={abl}', fontsize=12, rotation=0, ha='right', va='center')
        if i == 0:
            ax.set_title(f'|ILD|={abs_ild}', fontsize=12)
for ax_row in axes:
    for ax in ax_row:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(f'rtd_average_by_abs_ILD_FOLDED_param_samples_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()
