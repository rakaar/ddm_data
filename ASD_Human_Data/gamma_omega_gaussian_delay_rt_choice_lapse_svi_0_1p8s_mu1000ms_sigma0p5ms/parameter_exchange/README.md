# Human RT parameter exchanges

Four model-only comparisons at arithmetic physical posterior means of the latest widened-bound Gaussian-delay + choice + uniform-lapse fit:

1. Control baseline; ASD with Control Omega.
2. Control with ASD Omega; ASD baseline.
3. Control baseline; ASD with Control Gaussian delay mu and sigma.
4. Control with ASD Gaussian delay mu and sigma; ASD baseline.

Match all transfers by ASL and signed Delta_dB. Starting bias, Gamma and both lapse probabilities remain from the receiving group. All curves are recomputed and renormalized over 0–1.8 s using the existing Gaussian-delay prediction utilities. Lapse probability is the receiving group's retained-trial lapse fraction, and lapse RT remains uniform over 0–1.8 s. Condition pooling uses the receiving group's retained-trial counts (Control 6574, ASD 4770). Both ASLs are pooled; RTDs and chronometrics additionally pool stimulus signs. No fit, empirical plots, posterior resampling or causal interpretation is involved.

Reproduce from the repository root:

```bash
.venv/bin/python ASD_Human_Data/plot_human_gaussian_delay_parameter_exchange.py
.venv/bin/python ASD_Human_Data/verify_human_gaussian_delay_parameter_exchange.py
```

The plotter accepts `--fit-dir`, `--output-dir`, `--scenario {all,1,2,3,4}` and `--threads`. All six distinct models are evaluated to retain shared axes even when rendering a single scenario. Baselines are compared with saved predictions. All four hybrids are checked at K=40/80 and at 0.25 ms versus 0.5 ms grids. Validation thresholds: integrated mass and choice probability errors <1e-5; density grid difference <1e-3; CDF, choice, retained mass and mean RT grid differences <1e-5; density- versus CDF-based mean RT agreement <1e-6 s.

`model_parameters.csv` records the source group for every block/condition. `condition_predictions.csv` and `condition_model_curves.csv` retain signed-condition predictions and retention probabilities. `mixture_weights.csv` records all recipient weights; pooled RTD, psychometric and chronometric files contain plotted values. Python condition indices are zero-based. Numerical checks, independent reconstruction and source hashes accompany the figures. These are deterministic sensitivity predictions using variational posterior means; no uncertainty intervals or causal group effects are inferred.
