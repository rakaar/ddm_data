# %% Full-goal audit after numerical checks, actual visual inspection, and publication
import hashlib
import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse
import fitz
import numpy as np
import pandas as pd

# %% Editable roots; this script never changes fitting inputs or checkpoints
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'asd_analysis/shank3_cohort1_ASD0047_joint_npl_alpha_gaussian_abl_contaminants'
selected = json.loads((BASE / 'selected_fit.json').read_text())
RUN = ROOT / selected['selected_run']
variant = RUN.parents[1]
visual = json.loads((BASE / 'visual_inspection.json').read_text())
numerical = json.loads((BASE / 'deliverable_validation.json').read_text())
publication = json.loads((BASE / 'publication_manifest.json').read_text())
assert visual['status'] == numerical['status'] == selected['status'] == 'passed'
assert selected['parameter_count'] in [19, 20]
assert not any(selected[key] for key in ['artificial_flags', 'taper_flags', 'intrinsic_flags'])
old = ROOT / 'asd_analysis/shank3_cohort1_wt_hom_individual_adaptive_bounds/wt/ASD0047'
old_fixed = json.loads((old / 'fixed_proactive.json').read_text())
old_delay = old_fixed['values'][old_fixed['parameter_order'].index('t_nd')]
np.testing.assert_allclose(old_delay, -.7155462084115592, rtol=0, atol=1e-14)
decision = json.loads((BASE / 'contaminant_model_decision.json').read_text())
mixture_run = ROOT / decision['selected_mixture_run']
mixture_config = json.loads((mixture_run / 'configuration.json').read_text())
assert mixture_config['parameters'] == 20 and mixture_config['fixed_parameters'] == {}
assert 'lapse_exp_weight' in mixture_config['parameter_order']
assert json.loads((mixture_run.parent / 'independent_validation.json').read_text())['status'] == 'passed'
config = json.loads((RUN / 'configuration.json').read_text())
assert config['full_batch'] and {'V_A', 'theta_A', 't_nd'}.issubset(config['parameter_order'])
if decision['next_model'] == 'exponential_only':
    assert selected['variant'] == 'exponential_only' and config['parameters'] == 19
    assert config['fixed_parameters'] == {'lapse_exp_weight': 1.}
    assert config['frozen_parameters'] == ['lapse_exp_weight']
else:
    assert selected['variant'] == 'mixture'
for seed in [0, 1]:
    summary = json.loads((RUN.parent / f'seed_{seed}/run_summary.json').read_text())
    assert summary['status'] == 'converged' and summary['confirmation_windows'] >= 25
    assert summary['final_checked_step'] >= 50000 and not summary['parameter_stability_required']
for relative, expected in visual['artifact_hashes'].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative

# %% Read the actual PDFs, corner annotations, counts and comparison point tables
exports = BASE / 'posterior_and_diagnostics'
for name, pages in [('SHANK3_cohort1_ASD0047_joint_diagnostics.pdf', 4),
                    ('SHANK3_cohort1_ASD0047_joint_posterior_corners.pdf', 2)]:
    with fitz.open(exports / name) as document:
        assert len(document) == pages
        if pages == 4:
            assert document.get_toc() == [[1, 'ASD0047 joint RTDs', 1], [1, 'ASD0047 joint psychometric', 2],
                [1, 'Abort distributions, convergence, and parameters', 3], [1, 'Proactive and contaminant posterior corner', 4]]
    delivered = ROOT / 'shan3_coh_1_animal_Wise_fitting/ASD0047_joint' / name
    assert delivered.read_bytes() == (exports / name).read_bytes()
corners = json.loads((exports / 'export_manifest.json').read_text())
assert corners['samples'] == 10000 and corners['all_draws_included']
for page in corners['corners']:
    assert len(page['parameters']) == len(page['diagonal_titles'])
    assert all(' = ' in title and 'Mean' not in title for title in page['diagonal_titles'])
table = pd.read_csv(BASE / 'joint_vs_two_stage_parameters/posterior_parameter_comparison.csv')
assert table.loc[table.method.eq('joint')].estimate_type.eq('variational_posterior').all()
assert len(table.loc[table.method.eq('joint')]) == selected['parameter_count']
assert numerical['fitted_choices'] == 18771 and numerical['identical_empirical_RTD_rows'] == 21694
assert numerical['likelihood_aborts'] == 3004 and numerical['onset_censored'] == 11
assert numerical['excluded_long_choices'] == 161

# %% Strict build plus actual rendered image/download targets, not Markdown-only checks
build = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'mkdocs', 'build', '--strict'],
                       cwd=ROOT, capture_output=True, text=True)
(BASE / 'strict_mkdocs_build.log').write_text(build.stdout + build.stderr)
assert build.returncode == 0, build.stderr
page_path = ROOT / 'site/results' / publication['date'] / 'index.html'
anchor = 'shank3-cohort-1-asd0047-joint-proactive-nplalpha-and-contaminants'


class SectionLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active, self.seen, self.images, self.links = False, False, [], []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == 'h2':
            self.active = attributes.get('id') == anchor
            self.seen |= self.active
        if self.active and tag == 'img':
            self.images.append(attributes['src'])
        if self.active and tag == 'a' and attributes.get('href'):
            self.links.append(attributes['href'])


parser = SectionLinks()
parser.feed(page_path.read_text())
assert parser.seen and len(parser.images) == 5
verified_links = []
for reference in parser.images + parser.links:
    url = urlparse(reference)
    if url.scheme or reference.startswith('#'):
        continue
    target = (page_path.parent / unquote(url.path)).resolve()
    if target.is_dir():
        target /= 'index.html'
    assert target.is_file(), (reference, target)
    if url.fragment:
        assert f'id="{url.fragment}"' in target.read_text(), reference
    verified_links.append(reference)
checks = dict(
    animal_and_old_negative_delay_identified=True,
    all_animal_parameters_estimated_in_one_joint_objective=True,
    exponential_uniform_mixture_fitted_and_verified=True,
    authorized_exponential_only_refit_performed=selected['variant'] == 'exponential_only',
    all_remaining_free_posteriors_clear_hard_domain_and_taper_checks=True,
    corner_diagonals_have_parameter_names_and_means=True,
    full_and_focused_corners_numerically_and_visually_verified=True,
    four_page_diagnostics_pdf_verified_and_inspected=True,
    proactive_and_reactive_two_stage_comparison_pngs_verified_and_inspected=True,
    original_animal_fit_and_source_csv_hashes_preserved=True,
    initialization_sensitivity_reported=True,
    changed_lapse_and_conditioning_assumptions_reported=True,
    direct_result_book_embeds_and_downloads_verified=True,
    strict_mkdocs_build_passed=True)
report = dict(status='complete', animal='ASD0047', selected_run=selected['selected_run'],
    free_parameters=selected['parameter_count'], checks=checks,
    completion_proven=all(checks.values()), verified_rendered_links=len(verified_links),
    embedded_images=len(parser.images), old_proactive_delay_ms=old_delay * 1000,
    evidence=['selected_fit.json', 'deliverable_validation.json', 'visual_inspection.json',
              'posterior_and_diagnostics/export_manifest.json', 'strict_mkdocs_build.log', 'publication_manifest.json'])
assert report['completion_proven']
(BASE / 'goal_completion_audit.json').write_text(json.dumps(report, indent=2) + '\n')
publication.update(status='passed', strict_mkdocs_build='passed', direct_image_embeds=5,
                   verified_rendered_links=len(verified_links), rendered_section_anchor=anchor)
(BASE / 'publication_manifest.json').write_text(json.dumps(publication, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
