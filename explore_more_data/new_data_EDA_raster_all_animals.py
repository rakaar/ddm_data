import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# Ensure the pdfs directory exists (one level up from current directory)
pdf_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'pdfs'))
os.makedirs(pdf_dir, exist_ok=True)
pdf_path = os.path.join(pdf_dir, 'all_trials_raster_all_animals.pdf')

with PdfPages(pdf_path) as pdf:
    for animal in df_valid_and_aborts['animal'].unique():
        animal_df = df_valid_and_aborts[df_valid_and_aborts['animal'] == animal]
        sessions = sorted(animal_df['session'].unique())
        n_sessions = len(sessions)
        plt.figure(figsize=(12, max(2, n_sessions*0.18)))  # even closer rows
        for i, session in enumerate(sessions):
            session_df = animal_df[animal_df['session'] == session]
            trials = session_df['trial'].values
            total_fix = session_df['TotalFixTime'].values
            abort_event = session_df['abort_event'].values
            rtwrtstim = session_df['RTwrtStim'].values
            # Normalize trial numbers within this session
            if len(trials) == 0:
                continue
            min_trial = trials.min()
            max_trial = trials.max()
            if max_trial == min_trial:
                norm_trials = [0.5]*len(trials)
            else:
                norm_trials = (trials - min_trial) / (max_trial - min_trial)
            for x, tf, ab, rt in zip(norm_trials, total_fix, abort_event, rtwrtstim):
                if ab == 3 and tf < 0.3:
                    color = 'red'
                    size = 8  # 2x bigger than green
                elif rt > 1:
                    color = 'blue'
                    size = 8  # 2x bigger than green
                else:
                    color = 'green'
                    size = 4
                plt.scatter(x, i, color=color, s=size)
        plt.yticks(range(n_sessions), [f"Session {s}" for s in sessions])
        plt.gca().invert_yaxis()
        plt.xlabel('Normalized Trial Number')
        plt.ylabel('Session')
        plt.title(f'All Trials Raster: {animal}\nRed: abort_event==3 & Fix<0.3, Blue: RTwrtStim>1, Green: other')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"Saved raster plots for all animals to {pdf_path}")
