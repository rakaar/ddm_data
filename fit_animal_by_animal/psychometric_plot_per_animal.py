# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
exp_df = pd.read_csv('../outExp.csv')
if 'timed_fix' in exp_df.columns:
    exp_df.loc[:, 'RTwrtStim'] = exp_df['timed_fix'] - exp_df['intended_fix']
    exp_df = exp_df.rename(columns={'timed_fix': 'TotalFixTime'})

# remove rows where abort happened, and RT is nan
exp_df = exp_df[~((exp_df['RTwrtStim'].isna()) & (exp_df['abort_event'] == 3))].copy()

# in some cases, response_poke is nan, but success and ILD are present
# reconstruct response_poke from success and ILD 
mask_nan = exp_df['response_poke'].isna()
mask_success_1 = (exp_df['success'] == 1)
mask_success_neg1 = (exp_df['success'] == -1)
mask_ild_pos = (exp_df['ILD'] > 0)
mask_ild_neg = (exp_df['ILD'] < 0)

# For success == 1
exp_df.loc[mask_nan & mask_success_1 & mask_ild_pos, 'response_poke'] = 3
exp_df.loc[mask_nan & mask_success_1 & mask_ild_neg, 'response_poke'] = 2

# For success == -1
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_pos, 'response_poke'] = 2
exp_df.loc[mask_nan & mask_success_neg1 & mask_ild_neg, 'response_poke'] = 3

# Filter for LED2 batch with non-LED trials and relevant session types
exp_df_batch = exp_df[
    (exp_df['batch_name'] == 'LED2') &
    (exp_df['LED_trial'].isin([np.nan, 0])) &
    (exp_df['session_type'].isin([1,2]))
].copy()

# Define choice: 1 if response_poke == 3, -1 if response_poke == 2
exp_df_batch['choice'] = exp_df_batch['response_poke'].apply(lambda x: 1 if x == 3 else (-1 if x == 2 else np.nan))

# Get only valid trials (success is 1 or -1)
df_valid = exp_df_batch[exp_df_batch['success'].isin([1,-1])].copy()

# %%
# Create psychometric plot per animal

# Get unique animals and ABLs
unique_animals = df_valid['animal'].unique()
unique_abls = df_valid['ABL'].unique()

# Create a figure with subplots for each animal
fig, axes = plt.subplots(len(unique_animals), 1, figsize=(10, 5*len(unique_animals)), sharex=True)
if len(unique_animals) == 1:
    axes = [axes]  # Make sure axes is always a list

# Color palette for different ABLs
colors = sns.color_palette("husl", len(unique_abls))

# Create a mapping of ABL to color
abl_color_map = dict(zip(unique_abls, colors))

for i, animal in enumerate(unique_animals):
    ax = axes[i]
    animal_data = df_valid[df_valid['animal'] == animal]
    
    # Set the title for the subplot
    ax.set_title(f'Animal {animal}')
    
    # Loop through each ABL
    for abl in unique_abls:
        abl_data = animal_data[animal_data['ABL'] == abl]
        
        if len(abl_data) == 0:
            continue
        
        # Group by ILD and calculate probability of choice == 1
        grouped = abl_data.groupby('ILD')['choice'].apply(lambda x: np.mean(x == 1)).reset_index()
        grouped.columns = ['ILD', 'prob_choice_1']
        
        # Sort by ILD for plotting
        grouped = grouped.sort_values('ILD')
        
        # Plot the psychometric curve
        ax.plot(grouped['ILD'], grouped['prob_choice_1'], 'o-', label=f'ABL = {abl}', color=abl_color_map[abl])
        
        # Fit a logistic regression model
        if len(grouped) >= 3:  # Need at least 3 points for a meaningful fit
            try:
                # Prepare data for logistic regression
                X = abl_data['ILD'].values.reshape(-1, 1)
                y = (abl_data['choice'] == 1).astype(int).values
                
                # Fit logistic regression
                logit_model = stats.LogisticRegression()
                logit_model.fit(X, y)
                
                # Generate predictions for a smooth curve
                x_range = np.linspace(min(abl_data['ILD']), max(abl_data['ILD']), 100)
                y_pred = logit_model.predict_proba(x_range.reshape(-1, 1))[:, 1]
                
                # Plot the fitted curve
                ax.plot(x_range, y_pred, '--', color=abl_color_map[abl], alpha=0.7)
            except Exception as e:
                print(f"Could not fit logistic regression for animal {animal}, ABL {abl}: {e}")
    
    # Set labels and legend
    ax.set_ylabel('Probability of Choice == 1')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title='ABL')

# Set common x-label
plt.xlabel('ILD (Interaural Level Difference)')
plt.tight_layout()
plt.savefig('psychometric_per_animal.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Alternative approach: Use seaborn for a more polished look

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Create a FacetGrid for each animal
g = sns.FacetGrid(df_valid, col="animal", height=5, aspect=1.2, sharex=True, sharey=True, col_wrap=2)

# Map the plot function to the grid
g.map_dataframe(lambda data, color: sns.lineplot(x="ILD", y="choice", 
                                    data=data, 
                                    estimator=lambda x: np.mean(x == 1),
                                    ci=95, err_style="band",
                                    hue="ABL", palette="viridis"))

# Customize the plot
g.set_axis_labels("ILD (Interaural Level Difference)", "Probability of Choice == 1")
g.set_titles("Animal {col_name}")
g.add_legend(title="ABL")
g.tight_layout()
plt.savefig('psychometric_per_animal_seaborn.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
df_valid.columns
df_valid['ILD'].unique()

# %%
# animal 41, ABL 20, ILD -1, ILD 2
exp_df_ABL_20_ILD_minus_1 = exp_df[
    (exp_df['ABL'] == 20) &
    (exp_df['ILD'] == -1)
]

print(exp_df_ABL_20_ILD_minus_1['waveformL'].unique())
print(exp_df_ABL_20_ILD_minus_1['waveformR'].unique())

# %%
df_valid_animal_41 = df_valid[df_valid['animal'] == 41]
df_valid_animal_41_ABL_20_ILD_minus_1 = df_valid_animal_41[
    (df_valid_animal_41['ABL'] == 20) &
    (df_valid_animal_41['ILD'] == -1)
]
print(f'animal 41, ABL 20, ILD -1')
print(df_valid_animal_41_ABL_20_ILD_minus_1['waveformL'].unique())
print(df_valid_animal_41_ABL_20_ILD_minus_1['waveformR'].unique())

# %%
# animal 39, ABL 20 ILD -1
df_valid_animal_39 = df_valid[df_valid['animal'] == 39]
df_valid_animal_39_ABL_20_ILD_minus_1 = df_valid_animal_39[
    (df_valid_animal_39['ABL'] == 20) &
    (df_valid_animal_39['ILD'] == -1)
]
print(f'animal 39, ABL 20, ILD -1')
print(df_valid_animal_39_ABL_20_ILD_minus_1['waveformL'].unique())
print(df_valid_animal_39_ABL_20_ILD_minus_1['waveformR'].unique())

# %%
# read from batch_csvs
batch_LED_34 = pd.read_csv('/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs/batch_LED34_valid_and_aborts.csv')
batch_LED_34['batch_name'].unique()

# %%
# RANDOM ANIMAL FROM LED34
animal = np.random.choice(batch_LED_34['animal'].unique())
batch_LED_34_ABL_20_ILD_minus_1 = batch_LED_34[
    (batch_LED_34['animal'] == animal) &
    (batch_LED_34['ABL'] == 20) &
    (batch_LED_34['ILD'] == -1)
]
print(batch_LED_34_ABL_20_ILD_minus_1['waveformL'].unique())
print(batch_LED_34_ABL_20_ILD_minus_1['waveformR'].unique())

# %%