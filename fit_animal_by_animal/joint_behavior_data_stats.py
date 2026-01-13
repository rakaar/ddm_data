# %%
import os
import pandas as pd

# %%
# Load data from batch CSVs
DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
batch_dir = os.path.join(os.path.dirname(__file__), 'batch_csvs')
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)

# %%
# Get unique batch-animal pairs
batch_animal_pairs = list(map(tuple, merged_data[['batch_name', 'animal']].drop_duplicates().values))

# n = number of rats
n_rats = len(batch_animal_pairs)

# Sessions per animal
sessions_per_animal = []
print(f"{'Batch':<12} {'Animal':<8} {'Sessions':<8}")
print("-" * 30)
for batch, animal in batch_animal_pairs:
    animal_data = merged_data[(merged_data['batch_name'] == batch) & (merged_data['animal'] == animal)]
    n_sessions = animal_data['session'].nunique()
    sessions_per_animal.append(n_sessions)
    print(f"{batch:<12} {animal:<8} {n_sessions:<8}")
print("-" * 30)
avg_sessions_per_animal = sum(sessions_per_animal) / len(sessions_per_animal)
print(f"Average sessions per animal: {avg_sessions_per_animal:.1f}\n")

# Trials per session (for each animal, count trials in each of their sessions)
trials_per_session = []
for batch, animal in batch_animal_pairs:
    animal_data = merged_data[(merged_data['batch_name'] == batch) & (merged_data['animal'] == animal)]
    for session in animal_data['session'].unique():
        trials_per_session.append(len(animal_data[animal_data['session'] == session]))

total_sessions = len(trials_per_session)
total_trials_in_sessions = sum(trials_per_session)
avg_trials_per_session = total_trials_in_sessions / total_sessions

print(f"Total sessions across all animals: {total_sessions}")
print(f"Total trials: {total_trials_in_sessions}")
print(f"Average trials per session: {avg_trials_per_session:.1f}\n")

# Total trials
total_trials = len(merged_data)

# %%
print(f"(n = {n_rats} rats, {avg_sessions_per_animal:.1f} sessions per animal on average, {avg_trials_per_session:.1f} trials per session on average, total of {total_trials} trials)")

# Check: does product of averages equal total trials?
estimated_total = n_rats * avg_sessions_per_animal * avg_trials_per_session
deviation = estimated_total - total_trials
deviation_pct = 100 * deviation / total_trials

print(f"\nVerification:")
print(f"  n_rats × avg_sessions × avg_trials_per_session = {n_rats} × {avg_sessions_per_animal:.1f} × {avg_trials_per_session:.1f} = {estimated_total:.0f}")
print(f"  Actual total trials: {total_trials}")
print(f"  Deviation: {deviation:.0f} ({deviation_pct:.2f}%)")
