import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/FeetBall.txt')
df.drop(df.columns[2], axis=1, inplace=True)
df.columns = ['Tricep EMG', 'Ant Delt EMG']

plt.plot(df['Tricep EMG'], label='Tricep EMG')
plt.show()
plt.plot(df['Ant Delt EMG'], label='Ant Delt EMG')
plt.show()