# %% Editable settings: equal ABL weights within each joint posterior draw
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

ROOT = Path(__file__).resolve().parents[1]
FIT_ROOT = ROOT/'asd_analysis/shank3_cohort1_wt_hom_gamma_by_ild'
DEST = FIT_ROOT/'human_style_comparison'
GROUPS = ['wt', 'hom']
COLORS = {'wt':'#4D4D4D', 'hom':'#C24A7A'}
ABLS = [20, 40, 60]
STEM = 'rat_WT_HOM_average_omega'
FIGURE_SIZE = (4.5, 5)
DPI = 200


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# %% Preserve every previous comparison export and the source posterior files
DEST.mkdir(exist_ok=True)
previous = {str(p):sha256(p) for p in DEST.iterdir() if p.is_file()
            and not p.name.startswith(STEM) and p.name != Path(__file__).name}
sources = {g:FIT_ROOT/g/'gamma_by_ild/posterior_samples.npz' for g in GROUPS}
source_hashes = {str(p):sha256(p) for p in sources.values()}
averages, rows, abl_rows = {}, [], []
for group in GROUPS:
    with np.load(sources[group]) as samples:
        omega = samples['omega'].copy()
    assert omega.shape == (10000, 3) and np.isfinite(omega).all()
    averages[group] = omega.mean(axis=1)
    # A second expression checks the intended equal weighting and draw alignment.
    np.testing.assert_allclose(averages[group], (omega[:,0]+omega[:,1]+omega[:,2])/3,
                               rtol=0, atol=1e-14)
    lower, upper = np.quantile(averages[group], [.025,.975])
    mean = averages[group].mean()
    np.testing.assert_allclose(mean, omega.mean(axis=0).mean(), atol=1e-14)
    rows.append(dict(group=group, mean=mean, lower95=lower, upper95=upper,
                     posterior_draws=len(omega), ABL_weight=1/3, unit='s^-1', color=COLORS[group]))
    for abl, values in zip(ABLS, omega.T):
        abl_rows.append(dict(group=group, ABL=abl, posterior_mean=values.mean(), weight=1/3))
summary = pd.DataFrame(rows)
summary.to_csv(DEST/f'{STEM}_summary.csv', index=False, float_format='%.17g')
pd.DataFrame(abl_rows).to_csv(DEST/f'{STEM}_ABL_means.csv', index=False, float_format='%.17g')
pd.DataFrame({'draw':np.arange(10000), **averages}).to_csv(
    DEST/f'{STEM}_posterior_draws.csv', index=False, float_format='%.17g')

# %% Compact square axes; posterior means and conditional variational intervals
plt.rcParams.update({'pdf.fonttype':42, 'font.size':11,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, ax = plt.subplots(figsize=FIGURE_SIZE)
fig.subplots_adjust(left=.20, right=.95, bottom=.25, top=.84)
artists = []
for position, row in enumerate(summary.itertuples()):
    artist = ax.errorbar(position, row.mean, yerr=[[row.mean-row.lower95],[row.upper95-row.mean]],
        fmt='o', linestyle='none', color=row.color, markersize=7, capsize=5,
        elinewidth=1.4, capthick=1.4)
    assert to_hex(artist.lines[0].get_color()).lower() == COLORS[row.group].lower()
    np.testing.assert_allclose(artist.lines[2][0].get_segments()[0][:,1],
                               [row.lower95,row.upper95], atol=1e-14)
    artists.append(artist)
ax.set(xticks=[0,1], xticklabels=['WT','HOM'], xlim=(-.6,1.6),
       ylabel='Mean Omega across ABLs (s⁻¹)', title='SHANK3 cohort 1')
ax.set_box_aspect(1)
ax.grid(False, which='both', axis='both')
fig.text(.5,.155,'Equal average of ABL 20, 40 and 60 dB', ha='center', fontsize=10)
fig.text(.5,.105,'Mean ± central 95% credible interval', ha='center', fontsize=10)
fig.text(.5,.055,'Conditional variational posterior;\nproactive / lapse parameters fixed',
         ha='center', fontsize=9)
fig.canvas.draw()
position = ax.get_position()
np.testing.assert_allclose(position.width*FIGURE_SIZE[0], position.height*FIGURE_SIZE[1], atol=1e-12)
assert not any(x.get_visible() for x in ax.get_xgridlines()+ax.get_ygridlines())
assert all(a.lines[0].get_linestyle() == 'None' for a in artists)
assert ax.get_ylim()[0] < summary.lower95.min() < summary.upper95.max() < ax.get_ylim()[1]
for ext in ['pdf', 'png']:
    fig.savefig(DEST/f'{STEM}.{ext}', dpi=DPI, facecolor='white')
plt.close(fig)

# %% Verify export, preservation and numerical round trip; record audit
saved = pd.read_csv(DEST/f'{STEM}_summary.csv', float_precision='round_trip')
np.testing.assert_array_equal(saved[['mean','lower95','upper95']], summary[['mean','lower95','upper95']])
info = subprocess.check_output(['pdfinfo',str(DEST/f'{STEM}.pdf')], text=True)
assert int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:'))) == 1
for name, expected in {**previous, **source_hashes}.items():
    assert sha256(Path(name)) == expected, name
shutil.copy2(Path(__file__), DEST/Path(__file__).name)
audit = dict(status='passed', fit_root=str(FIT_ROOT), source_hashes=source_hashes,
    previous_exports_unchanged=True, posterior_draws_per_group=10000,
    weighting='Equal ABL weights within each joint posterior draw; average before taking quantiles',
    interval='Central 95% conditional variational credible interval',
    colors=COLORS, no_grid=True, no_connecting_lines=True, single_page_PDF=True,
    refitted=False, visual_inspection='pending',
    artifacts={p.name:sha256(p) for p in DEST.glob(f'{STEM}*')
               if p.is_file() and p.name != f'{STEM}_validation.json'},
    plotting_source=str(Path(__file__).resolve()), plotting_source_sha256=sha256(Path(__file__)))
(DEST/f'{STEM}_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
print(summary.to_string(index=False))
print('PDF:', DEST/f'{STEM}.pdf')
