# Both ASLs pooled: Gaussian-delay + choice + uniform-lapse fit

The PNG/PDF overlay Control in blue and ASD in red in each of five RTD panels, with both groups in the right psychometric panel. Data and existing predictions pool ASL 20 and 40 cd/m². This is a derived visualization, without a new fit or parameter averaging.

Selection: Condition == RT, Valid == 1, 0 <= RT_s <= 1.8. Control: 6,574 trials / 11 participants. ASD: 4,770 trials / 8 participants. Both choices and all four RTs below 200 ms remain included. Trial pooling gives each retained trial equal weight, rather than each participant equal weight.

Each RTD is a retained-count-weighted mixture of four separately normalized ASL-by-sign predictions. Each psychometric point combines two ASLs at the same signed difference. Source curves were evaluated at arithmetic physical posterior means of the widened-prior Gaussian-delay + choice + uniform-lapse fits. Gaussian zero-truncation, 0–1.8 s retention normalization, and lapse contributions are already included in those source curves. Fitted parameters remain ASL-specific.

Reproduce from the repository root:

```bash
.venv/bin/python ASD_Human_Data/plot_human_gaussian_delay_pooled_asl.py
.venv/bin/python ASD_Human_Data/verify_human_gaussian_delay_pooled_asl.py
```

Both scripts accept `--fit-dir` and `--output-dir`; defaults point to the widened-prior fit and its `pooled_asl` subdirectory. The plot reads the source fit's saved tables, checks them against its publication hashes, and never changes its model artifacts.

Tables contain pooled probabilities, 20 ms histograms, 0.5 ms model curves, all 80 component weights, and 11,344 source-row identities. `plot_manifest.json` records input and output hashes. `independent_validation.json` reconstructs the outputs from raw trials and saved signed-condition predictions. `preservation_baseline.json` records 297 prior fit and published files plus the existing result-page prefix.

The source fits used variational approximations for pooled trials. Their terminal convergence checks passed, but local parameter stability at the restored best checkpoints was not separately confirmed; refer to the original wider-bounds result entry for details. This pooling neither changes convergence evidence nor tests equality of the fitted parameters across ASLs.
