import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define file paths
conditions = [
    "LoganFeetBall", "LoganFeetBench",
    "LoganHandBall", "LoganHandBench"
]
# CHANGE FOR EACH PARTICIPANT - make sure all 'name' specific variables are modified
data_files = {cond: pd.read_csv(f'data/Logan/{cond}.csv') for cond in conditions}

# Load MVC data
# Load MVC data (selecting correct columns)
shoulderMVC = pd.read_csv('data/Logan/ShoulderMVC.txt', header=None, usecols=[1], names=['Shoulder'])
tricepMVC = pd.read_csv('data/Logan/TricepMVC.txt', header=None, usecols=[0], names=['Tricep'])

# EMG & Force offset correction. If offset < 0.1, ignore it
def calculate_offsets(df):
    if 'Tricep' in df and 'Shoulder' in df:
        tricep_mean = df['Tricep'].iloc[:5000].mean()
        shoulder_mean = df['Shoulder'].iloc[:5000].mean()

        if abs(tricep_mean) >= 0.1:
            df['Tricep'] -= tricep_mean
        if abs(shoulder_mean) >= 0.1:
            df['Shoulder'] -= shoulder_mean
    return df

# Full wave rectify signal while ensuring only existing columns are modified
def full_wave_rectify(df):
    if 'Tricep' in df:
        df['Tricep'] = df['Tricep'].abs()
    if 'Shoulder' in df:
        df['Shoulder'] = df['Shoulder'].abs()
    return df

# Root Mean Square (RMS) Function
def calculate_rms(signal):
    return np.sqrt(np.mean(signal ** 2)) if len(signal) > 0 else np.nan

# RMS computation using sliding window
#1ms = 2Hz; 75ms = 150Hz; 150ms = 300Hz
def compute_rms(df, window_size=300, step_size=150):
    rms_values = {col: [] for col in df.columns}  # Create dictionary for different muscle groups
    time_values = []

    for start in range(0, len(df) - window_size, step_size):
        time_values.append(start / 2000)  # Assume 2000 Hz sampling rate

        for col in df.columns:  # Process only available columns
            window = df[col].iloc[start: start + window_size]
            rms_values[col].append(calculate_rms(window))

    rms_df = pd.DataFrame({'Time (s)': time_values})
    for col in rms_values:
        rms_df[f'{col}_RMS'] = rms_values[col]

    return rms_df

# Apply RMS processing to MVC data (ensuring correct column selection)
shoulderMVC = compute_rms(full_wave_rectify(shoulderMVC))
tricepMVC = compute_rms(full_wave_rectify(tricepMVC))

# Normalize EMG (Dividing by MVC max)
def normalize_emg(rms_df):
    Tricep_MVC_max = tricepMVC['Tricep_RMS'].max()
    Shoulder_MVC_max = shoulderMVC['Shoulder_RMS'].max()

    if Tricep_MVC_max > 0 and Shoulder_MVC_max > 0:
        rms_df['Normalized_Tricep'] = rms_df['Tricep_RMS'] / Tricep_MVC_max * 100 # Convert to percentage
        rms_df['Normalized_Shoulder'] = rms_df['Shoulder_RMS'] / Shoulder_MVC_max * 100
    return rms_df


# Process all conditions
processed_data = {}
for cond, df in data_files.items():
    df = calculate_offsets(df)  # Apply offset correction
    df = full_wave_rectify(df)
    rms_df = compute_rms(df)  # Compute RMS
    rms_df = normalize_emg(rms_df)  # Normalize
    processed_data[cond] = rms_df

# Print mean activation percentage for each condition
print("Mean Activation Percentage (Normalized to MVC)")
for cond, df in processed_data.items():
    mean_tricep = df['Normalized_Tricep'].mean()
    mean_shoulder = df['Normalized_Shoulder'].mean()
    print(f"  {cond}: Tricep {mean_tricep:.2f}%, Shoulder {mean_shoulder:.2f}%")


# Perform paired t-tests between Swiss ball and non-Swiss ball trials,
# looking for differences between muscle amplitude. Differences across the different exercises was
# not examined-only between the stable and unstable conditions.
# Function to perform paired t-tests and additional stats
def perform_stat_tests(condition1, condition2):
    min_len = min(len(processed_data[condition1]), len(processed_data[condition2]))

    if min_len == 0:
        print(f"Skipping {condition1} & {condition2} due to insufficient data.")
        return

    tricep1 = processed_data[condition1]['Normalized_Tricep'][:min_len]
    tricep2 = processed_data[condition2]['Normalized_Tricep'][:min_len]
    shoulder1 = processed_data[condition1]['Normalized_Shoulder'][:min_len]
    shoulder2 = processed_data[condition2]['Normalized_Shoulder'][:min_len]

    # Paired t-test
    tricep_ttest = stats.ttest_rel(tricep1, tricep2, nan_policy='omit')
    shoulder_ttest = stats.ttest_rel(shoulder1, shoulder2, nan_policy='omit')

    # Effect size (Cohen's d)
    tricep_effect_size = (tricep1.mean() - tricep2.mean()) / np.std(np.concatenate((tricep1, tricep2)))
    shoulder_effect_size = (shoulder1.mean() - shoulder2.mean()) / np.std(np.concatenate((shoulder1, shoulder2)))

    # Pearson correlation (measuring similarity of activation patterns)
    tricep_corr, _ = stats.pearsonr(tricep1, tricep2)
    shoulder_corr, _ = stats.pearsonr(shoulder1, shoulder2)

    print(f"Statistical analysis between {condition1} & {condition2}:")
    print(f"  Paired t-test (Tricep EMG): t = {tricep_ttest.statistic:.3f}, p = {tricep_ttest.pvalue:.5f}")
    print(f"  Paired t-test (Shoulder EMG): t = {shoulder_ttest.statistic:.3f}, p = {shoulder_ttest.pvalue:.5f}")
    print(f"  Cohen's d (Tricep): {tricep_effect_size:.3f}, Shoulder: {shoulder_effect_size:.3f}")
    print(f"  Pearson correlation (Tricep): {tricep_corr:.3f}, Shoulder: {shoulder_corr:.3f}")
    print("-")


# Run statistical tests on the condition pairs
condition_pairs = [
    ("LoganFeetBall", "LoganFeetBench"),
    ("LoganHandBall", "LoganHandBench"),
    ("LoganFeetBall", "LoganHandBench"),
    ("LoganHandBall", "LoganFeetBench")
]

for pair in condition_pairs:
    perform_stat_tests(pair[0], pair[1])

# Plot Normalized EMG for both Tricep and Shoulder
for cond, df in processed_data.items():
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time (s)'], df['Normalized_Tricep'], label=f'{cond} Tricep', color='blue')
    plt.plot(df['Time (s)'], df['Normalized_Shoulder'], label=f'{cond} Shoulder', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized EMG (%)')
    plt.title(f'Normalized EMG for {cond}')
    plt.legend()
    plt.show()