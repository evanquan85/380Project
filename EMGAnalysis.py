import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/AodhanFeetBall.csv')
shoulderMVC = pd.read_csv('data/ShoulderMVC.txt')
tricepMVC = pd.read_csv('data/TricepMVC.txt')
# EMG & Force offset. If offset < 0.1, ignore it
def calculate_offsets(df):
    emg_mean = df['Tricep'].iloc[:5000].mean()
    if abs(emg_mean) >= 0.1:
        df['Tricep'] = df['Tricep'] - emg_mean
    return df
df = calculate_offsets(df)
#Full wave rectify signal
df['Tricep'] = df['Tricep'].abs()
#Smooth and root-mean-square EMG
window_size = 50
df['Tricep'] = df['Tricep'].rolling(window=window_size, center=True).mean()
def calculate_rms(signal):
    return np.sqrt(np.mean(signal**2))
df['Tricep'] = df['Tricep'].rolling(window=1000).apply(calculate_rms, raw=True)

#Normalizing EMG data means dividing each data point in EMG signal by the MVC max voltage
#Normalized EMG = EMG/MVC max. Values range between 0 and 1 (convert to percentage)
Tricep_MVC_max = tricepMVC.iloc[:,0].max()
df['Normalized_Tricep'] = df['Tricep']/Tricep_MVC_max


# Convert to NumPy array (optional)
normalized_tricep_array = df['Normalized_Tricep'].to_numpy()

# Plot Normalized Tricep EMG
plt.plot(df['Time (s)'], df['Normalized_Tricep'], label='Normalized Tricep EMG')
plt.xlabel('Time (s)')
plt.ylabel('Normalized EMG (%)')
plt.legend()
plt.show()
