# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data (relative to this script's location)
df = pd.read_csv('../outExp.csv')

# Filter to Comparable batch
all_df = df[df['batch_name'] == 'Comparable']

# Pick a random animal
animals = all_df['animal'].unique()
if len(animals) == 0:
    raise ValueError('No animals found in the data!')
animal = np.random.choice(animals)
animal_df = all_df[all_df['animal'] == animal].copy()

# Normalize trial number within each session
def norm_trial_fn(x):
    return (x - x.min()) / (x.max() - x.min() if x.max() > x.min() else 1)
animal_df['norm_trial'] = animal_df.groupby('session')['trial'].transform(norm_trial_fn)

# Pick a random session for this animal
sessions = animal_df['session'].unique()
if len(sessions) == 0:
    raise ValueError(f'No sessions found for animal {animal}!')
session = np.random.choice(sessions)
session_df = animal_df[animal_df['session'] == session]

# Plot trial vs. norm_trial
plt.figure()
plt.scatter(session_df['trial'], session_df['norm_trial'])
plt.xlabel('Trial number')
plt.ylabel('Normalized trial (0-1)')
plt.title(f'Animal {animal}, Session {session}')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Pick a random animal (reuse animal_df from earlier)
animal = np.random.choice(animals)
animal_df = all_df[all_df['animal'] == animal].copy()

# Sort sessions
sessions = sorted(animal_df['session'].unique())
print(f"Animal: {animal}")
for i, sess in enumerate(sessions):
    trials = animal_df[animal_df['session'] == sess]['trial']
    print(f"Session {sess}: min trial = {trials.min()}, max trial = {trials.max()}")
    if i > 0:
        prev_max = animal_df[animal_df['session'] == sessions[i-1]]['trial'].max()
        curr_min = trials.min()
        print(f"  --> Gap from previous session: {curr_min - prev_max}")
