# Human ASD fitting: a conversation with your agent

This folder contains a skill and reference Python code for fitting human
Control/ASD visual-task data. The agent explains the modeling choices in plain
language and handles the code and libraries.

## Start here

Give your agent this folder and say:

> Read SKILL.md and guide me through the important modeling decisions in plain
> language. Handle the code and libraries yourself. Summarize our agreed model
> before starting a long fit.

You discuss which trials to include, the RT range, which parameters vary across
stimuli, which are shared, delay/lapse assumptions and prior bounds. Full-rank
SVI and routine numerical settings have agent-managed defaults. No configuration
or dependency file needs to be edited by you.

## Contents

- [SKILL.md](SKILL.md): instructions for the agent's conversation, validation,
  fitting, posterior reports and Control–ASD comparisons.
- [Reference guide](references/model-and-code.md): data conventions, likelihood,
  parameter sharing and reference-code map.
- [Reference code](assets/reference_code/): the fitting driver, likelihoods,
  reporting code and their local imports, preserving the original project layout.
- [Independent simulation check](scripts/check_reference_simulation.py): synthetic
  RT/choice checks against the likelihood, without reading participant data.
- [Validation record](validation/package_checks.json) and
  [simulation results](validation/simulation_check.json).

The reference model has 16 parameters per group, with ASLs pooled and a shared
Gaussian delay distribution. It is an example to discuss and adapt, rather than
a compulsory model. Participant data are supplied separately. The reference
guide explains the data files and older audit dependencies.

Validation checked the skill format, source hashes, Python syntax, bundled
driver imports and independent simulations for three illustrative conditions
(6,000 trials each). The simulations have finite time-step error; this check is
not a parameter-recovery fit or validation of every possible adapted model.
