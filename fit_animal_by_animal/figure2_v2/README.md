# Figure 2 v2 IPL SVI Condition-Delay Reproduction

This folder contains the compact files needed to redraw the current Fig. 2-style
IPL/vanilla condition-delay SVI diagnostic.

## What This Figure Shows

- Model family: direct IPL/vanilla SVI with condition-wise `t_E_aff`.
- Fit root used to build the panel data:
  `fit_animal_by_animal/numpyro_svi_vanilla_condition_delay_patience12_min50k_restore_best_outputs/`
- Psychometric and psychometric-slope model points omit SD model entries for
  `|ILD| > 8`.
- RT quantiles use q10/q30/q50/q70/q90 and the SD flat-delay rule beyond
  `|ILD| = 8`.
- Gamma/Omega condition points come from the 92-parameter Gamma/Omega/delay SVI
  condition posterior means; dashed model curves come from direct IPL SVI
  posterior-mean parameters.

## Files

- `plot_ipl_svi_fig2_v2.py`: redraws the five-panel figure from the panel pickles
  in this folder.
- `build_ipl_svi_fig2_v2_data.py`: rebuilds the compact panel pickles from the
  larger IPL SVI Fig. 2-like diagnostic payload.
- `ipl_svi_*_fig2_v2_data.pkl`: compact panel-data pickles used by the plot
  script.
- `ipl_svi_fig2_v2_bundle.pkl`: bundled copy of the compact panel data and
  validation checks.
- `figure2_v2_ipl_svi_condition_delay.png` / `.pdf`: current rendered figure.

## Reproduce The Figure From This Folder

From the repository root:

```bash
.venv/bin/python -u fit_animal_by_animal/figure2_v2/plot_ipl_svi_fig2_v2.py
```

This is the normal reproducible path for the backed-up Drive folder because it
uses only the compact pickles stored here.

## Rebuild The Panel Pickles

Only needed if the upstream IPL SVI diagnostic payload is available locally and
the compact panel pickles should be regenerated:

```bash
.venv/bin/python -u fit_animal_by_animal/figure2_v2/build_ipl_svi_fig2_v2_data.py
.venv/bin/python -u fit_animal_by_animal/figure2_v2/plot_ipl_svi_fig2_v2.py
```

The builder validates:

- 30 animals.
- 864 condition rows.
- zero SD psychometric/slope model entries at `|ILD| > 8`.
- 30 SD-flat model entries at `|ILD| = 16` for each ABL in the continuous
  quantile curves.
- paper quantiles exactly `[0.1, 0.3, 0.5, 0.7, 0.9]`.
