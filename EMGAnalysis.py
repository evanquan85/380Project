import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('data/Aodhan/AodhanFeetBall.csv')
shoulderMVC = pd.read_csv('data/Aodhan/ShoulderMVC.txt')
tricepMVC = pd.read_csv('data/Aodhan/TricepMVC.txt')
# Sampling frequency in Hz
Fs = 2000
# EMG & Force offset. If offset < 0.1, ignore it
def calculate_offsets(df):
    emg_mean = df['Tricep'].iloc[:5000].mean()
    emg_mean = df['Shoulder'].iloc[:5000].mean()
    if abs(emg_mean) >= 0.1:
        df['Tricep'] = df['Tricep'] - emg_mean
        df['Shoulder'] = df['Shoulder'] - emg_mean
    return df
df = calculate_offsets(df)
#Full wave rectify signal
df['Tricep'] = df['Tricep'].abs()
df['Shoulder'] = df['Shoulder'].abs()
#Root mean square technique
#1ms = 2Hz; 75ms = 150Hz; 150ms = 300Hz
window_size = 300
step_size = 150

# Root Mean Square (RMS) Function
def calculate_rms(signal):
    return np.sqrt(np.mean(signal**2))

# Create an empty array to store RMS values
shoulder_rms_values = []
tricep_rms_values = []
time_values = []

# Compute RMS using a sliding window with overlap
for start in range(0, len(df) - window_size, step_size):
    # Extract window data
    tricep_window = df['Tricep'].iloc[start: start + window_size]
    shoulder_window = df['Shoulder'].iloc[start: start + window_size]

    # Compute RMS for each window
    tricep_rms_values.append(calculate_rms(tricep_window))
    shoulder_rms_values.append(calculate_rms(shoulder_window))

    # Store time for the center of the window
    time_values.append(df['Time (s)'].iloc[start + window_size // 2])

    # Create RMS DataFrame
rms_df = pd.DataFrame({
    'Time (s)': time_values,
    'Tricep_RMS': tricep_rms_values,
    'Shoulder_RMS': shoulder_rms_values
})

# Normalize EMG (Dividing by MVC max)
Tricep_MVC_max = tricepMVC.iloc[:, 0].max()
Shoulder_MVC_max = shoulderMVC.iloc[:, 0].max()

rms_df['Normalized_Tricep'] = rms_df['Tricep_RMS'] / Tricep_MVC_max
rms_df['Normalized_Shoulder'] = rms_df['Shoulder_RMS'] / Shoulder_MVC_max

# Perform paired t-tests between Swiss ball and non-Swiss ball trials,
# looking for differences between muscle amplitude. Differences across the different exercises was
# not examined-only between the stable and unstable conditions.


# Plot Normalized EMG for both Tricep and Shoulder
plt.figure(figsize=(10, 5))
plt.plot(rms_df['Time (s)'], rms_df['Normalized_Shoulder'], label='Normalized Shoulder EMG', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Normalized EMG (%)')
plt.legend()
plt.title('Normalized Shoulder EMG')
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(rms_df['Time (s)'], rms_df['Normalized_Tricep'], label='Normalized Tricep EMG', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Normalized EMG (%)')
plt.legend()
plt.title('Normalized Tricep EMG')
plt.show()

# Offset Shoulder EMG for visualization
shoulder_offset = 1  # Arbitrary offset so signals don't overlap

# Plot Normalized EMG for both Tricep and Shoulder with offset
plt.figure(figsize=(10, 5))
plt.plot(rms_df['Time (s)'], rms_df['Normalized_Tricep'], label='Tricep EMG', color='blue')
plt.plot(rms_df['Time (s)'], rms_df['Normalized_Shoulder'] + shoulder_offset, label='Shoulder EMG (Offset)', color='red')

# Formatting the plot
plt.xlabel('Time (s)')
plt.ylabel('Arbitrary EMG Activation')
plt.title('Tricep and Shoulder Activation Over Time')
plt.legend()
plt.show()