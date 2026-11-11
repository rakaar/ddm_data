---
name: human-asd-fitting
description: Guide a human collaborator through modeling decisions for human Control/ASD brightness-task fits, then implement, validate and report the agreed model using bundled DDM reference code. Use for planning, fitting or revising these human analyses.
---

# Human ASD fitting: discuss the model before fitting

Help someone who is not very mathematical make informed modeling decisions.
The human answers questions in conversation. The agent edits code, manages its
Python environment, implements the choices and generates reports. Never ask the
human to edit YAML, dependency lists or source code to express their choices.

Use the bundled code as a reference implementation, not as a reason to silently
choose its historical filters or parameter sharing. The reference has 16
parameters per group and pools ASLs; that is one supported starting example,
not a requirement for every new analysis. Read [the reference guide](references/model-and-code.md)
when inspecting data, proposing a model or selecting code to adapt.

## Start with the conversation

Read the user's request, existing decisions and available data first. Ask only
about consequential choices that remain unresolved. Use short batches of two
or three questions, ordinary language, and a recommended option with its
tradeoff. Avoid giving the human a long technical questionnaire. Continue
read-only inspection while waiting for answers.

For example: “Should both brightness levels use the same decision-time scale,
or should each level have its own? Separate values give more flexibility but
need more information from the data.” Then explain that this choice controls
how Omega is shared. Do not require the human to know parameter names first.

Explain a materially more specific interpretation before using it: “You wrote
X; I am using Y because Z.” This especially applies to pooling, RT cutoffs,
retention normalization, condition definitions and uncertainty.

### Decisions the human should understand

Cover these topics through the conversation; do not ask again about choices
already made in this session.

1. **Which observations?** Inspect available conditions and RT distributions.
   Explain RT versus short-duration SD and sequential CB. Show counts removed
   by proposed cutoffs, separately for Control and ASD; retain source identities.
   Discuss pooling across participants/sessions versus participant-wise fits,
   and keeping ASLs separate versus pooling them. Confirm whether both choices,
   including incorrect choices, are included. An unusually early RT should be
   investigated rather than automatically removed to accommodate a delay.
2. **What should the model explain?** Ask about joint RT + choice versus RT-only.
   RT-only sums the two bound-specific densities; it does not fit a histogram
   by MSE. Its psychometric is a prediction, and drift sign/bias may be weakly
   identified. Explain fixed versus variable nondecision delay and whether
   lapses are included. State the lapse RT distribution, choice rule and the
   meaning of its probability. Discuss reactive-only versus any added process
   when relevant; never add a proactive process silently.
3. **What varies across stimuli?** Describe Gamma as signed evidence strength,
   Omega as decision-time scale, delay as time outside evidence accumulation,
   and w as starting preference. Gamma also affects RT shape. Ask what varies
   with signed brightness difference, ASL or participant, and what is shared.
   If offering a TIED stimulus rule, distinguish it from free condition-wise
   parameters; an auditory stimulus mapping is not automatically a visual law.
   Display a sharing table and the resulting number of free parameters per
   group. Include delay-distribution and lapse parameters in that count.
4. **What values are plausible?** Propose actual priors, hard limits, units and
   starting values for the chosen model. Explain a uniform or trapezoidal prior
   using “allowed range” and “more plausible/flat range.” Show enough detail
   for the human to understand a restrictive assumption without asking them to
   derive a prior. Distinguish parameter bounds, sampled-delay support and RT
   selection. A fixed causal delay cannot exceed an observed non-lapse RT;
   a Gaussian delay mean is not a fixed delay. Do not silently remove trials,
   truncate the target prior or change sharing to make initialization work.

Summarize the agreed selection, objective, sharing/count, delay/lapse model and
priors in a short readable **fit specification**. Include unresolved issues and
the planned outputs. Ask for confirmation before the first expensive fit if
the decisions are not already authorized. One confirmation of a concrete
specification is sufficient; do not add repeated permission checkpoints. Save
the specification as an agent-written Markdown record and the executed settings
as a machine-readable run record. The human does not edit either file.

## Technical defaults the agent handles

Explain simply: “I will estimate a joint distribution of plausible parameter
values, allowing correlations, and compare its predictions with the data.”
Unless the user asks otherwise or the chosen model requires a justified change:

- Use NumPyro full-batch SVI with `AutoMultivariateNormal`: a full-rank Gaussian
  variational approximation in unconstrained coordinates. Do not ask the human
  to choose a covariance matrix or optimizer.
- Float64; one ELBO particle; seed 0; guide scale 0.1; ClippedAdam learning rate
  0.0002 and gradient clip 1. Use an isolated Python environment and manage the
  libraries yourself. Select CPU/GPU from actual availability and record versions.
- Check nonoverlapping 1,000-update loss windows. Each strictly lower mean resets
  patience. Default minimum 50,000, maximum 1,000,000 updates and 25 subsequent
  windows without improvement; restore the selected best checkpoint. The cap
  can stop an unconfirmed run. Track parameter stability as a diagnostic, not
  an extra stopping gate unless explicitly agreed. Confirmation is not proof
  of a global optimum.
- Save 10,000 aligned constrained posterior draws, means, SDs, 95% equal-tailed
  intervals, covariance/correlation matrices, losses, best/latest checkpoints,
  stopping evidence and provenance. Start in a fresh output directory.

Tell the human about material consequences such as uncertainty being approximate
or a fit failing validation. Do not turn routine implementation details into
questions. Preserve explicit settings and existing session authorization.

## Implement and check before the long fit

Reuse the bundled likelihood primitives where applicable. Adapt the driver,
condition mapping and reports to the agreed model; do not leave hard-coded
reference counts, windows or sharing in a changed analysis. Ensure the equations,
code and sharing table describe the same model. Record source versions and row
identities. Do not overwrite previous fit outputs or upload data as part of fitting.

Before optimization:

- Write the exact trial-level likelihood, including retained-window normalization
  over **both** bounds. Normalize after integrating over random delay. Distinguish
  a lapse fraction among retained trials from a preselection mixture fraction.
- Check density/CDF agreement, total conditional mass, finite gradients and
  numerical convergence across series truncation, integration grids/quadrature.
  Use K=40/80 comparisons and grid refinement where applicable; a historical
  validated K=8 shortcut is not a universal guarantee for new parameter regimes.
- Generate independent synthetic trials at known parameters and apply the same
  delay, lapses and selection as the likelihood. Compare choice fractions, RTDs
  and CDFs; state Monte Carlo and time-discretization error. Do not sample a
  numerical CDF and call agreement with that same CDF independent validation.
  The bundled [simulation check](scripts/check_reference_simulation.py) is a
  baseline example; adapt it when the model or selection changes.
- Run a small synthetic parameter-recovery fit using the agreed inference path.
  Assess identification and predictions, not exact recovery of every parameter.
  A simulator/theory comparison is not a recovery fit. Report which checks were
  actually run; do not claim a preflight checks everything.

Resolve concrete discrepancies before expensive fitting. If an independent
sampler or recovery check is unavailable, disclose that gap and resolve it with
the human before treating the workflow as validated.

## Report and discuss the result

Create these outputs from the same selected trials and actual condition weights:

1. **One readable report per group:** data RTD histograms plus model curves by
   ASL and absolute difference, data psychometric dots plus model probabilities,
   and chronometrics. Keep actual numerical stimulus positions. For independent
   condition parameters, connect measured model points only as guides; smooth
   intermediate predictions require an explicitly chosen stimulus rule.
2. **Posterior diagnostics:** parameter table with priors and intervals, useful
   corner plots, correlations, boundary accumulation and loss/stopping record.
   Distinguish the restored checkpoint from the final optimization step. Flag
   instability or a cap stop without confirmation directly.
3. **Control–ASD parameter comparison:** matched parameters and 95% variational
   credible intervals, Control blue and ASD red. Shared scalars get separate
   panels. Discuss selection/prior/model differences if comparisons are not
   matched. With group-pooled fits, these intervals do not represent variation
   across participants or establish a population-level group difference.

Use arithmetic posterior means of physical parameters for the requested single
model curves, and state that convention. Predictions averaged across joint
posterior draws are different and can show predictive uncertainty. A posterior
distribution over a delay parameter is distinct from trial-to-trial delay variability.

Inspect exported images for readability. Explain fit quality in ordinary language,
with condition-wise CDF discrepancies and psychometric errors as numerical support.
An RT-only psychometric must be labeled “choices not fitted.”

If posterior mass crowds a bound, inspect whether it is an artificial limit,
weak identification, an implementation error or model mismatch. Recommend a
specific change when justified and discuss it before changing a modeling choice
that is not already authorized. Widening a bound and refitting is not automatic;
physical probabilities and causality remain constraints. Poor fit without
boundary crowding still needs explanation. Publishing, messaging collaborators
and external backups require their own user request.
