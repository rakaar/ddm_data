# lapse in PSIAM-TIED model.

**`simple_lapse_test.py`**
- Testing likelihood with simulated data - lapse model

**`test_lapses.py`**
- VBMC on simulated data - lapses, pro + TIED

**`test_lapse_sim_data_fix_rt_lapse.py`**
- VBMC on simulated data with fixed lapse RT window (0.9s) - only fits lapse_prob
- Similar to test_lapses.py but with T_lapse_max fixed to 0.9 and not fitted as a parameter

**`lapses_fit_single_animal.py`** (located in `../fit_animal_by_animal/`)
- VBMC fit on exp data single animal: lapses + vanilla model



