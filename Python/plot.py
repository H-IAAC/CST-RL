import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_lighter(color, saturation_factor=0.2, value_factor=1.2):
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
    return mcolors.hsv_to_rgb([h, s * saturation_factor, min(v * value_factor, 1.0)])

# Function to compute the running average
def running_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Read CSV files into pandas DataFrames
csv_file1 = 'CSTRL\graphs\QLearning(alpha=[LinearDecreaseRate(a=0.1, epsToZero=20000)],epsilon=[LinearDecreaseRate(a=0.9, epsToZero=20000)],gamma=[0.9],feature_extractor=[FroggerFE]).csv'
csv_file2 = 'CSTRL\graphs\LFAQLearning(alpha=[LinearDecreaseRate(a=0.001, epsToZero=4000)],epsilon=[LinearDecreaseRate(a=0.9, epsToZero=4000)],gamma=[0.9],feature_extractor=[LCFE(power=1)]).csv'
csv_file3 = 'CSTRL\graphs\TensorforceLearner(configPath=dqn.json).csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
df3 = pd.read_csv(csv_file3)

# Plot raw data from all 3 CSV files in the same graph
fig, axes = plt.subplots()
fig.set_figwidth(7)
fig.set_figheight(4)
axes.plot(df1["Episode"], df1["Value"], label='Episode Tracker raw', color=make_lighter('orange'))
axes.plot(df2["Episode"], df2["Value"], label='No Convolution raw', color=make_lighter('red'))
axes.plot(df3["Episode"], df3["Value"], label='DQN raw', color=make_lighter('blue'))
axes.set_ylabel('Episodic return')
axes.set_xlabel('Step')

# Plot running averages of all 3 CSV files
window_size = 200  # Adjust as needed
mark_every = 600

m1 = running_average(df1["Value"], window_size)
m2 = running_average(df2["Value"], window_size)
m3 = running_average(df3["Value"], window_size)
max_episode = df1.size / 2 - 1
m2 = pd.concat([m2, m2.get([m2.size-1])])
e2 = pd.concat([df2["Episode"], pd.Series([max_episode])])
m3 = pd.concat([m3, m3.get([m3.size-1])])
e3 = pd.concat([df3["Episode"], pd.Series([max_episode])])

a1 = axes.plot(df1["Episode"], m1, label='Tabular Q-Learning', color='orange', marker='v', markevery=mark_every)[0]
a2 = axes.plot(e2, m2, label='LFA Q-Learning', color='red', marker='o', markevery=mark_every)[0]
a3 = axes.plot(e3, m3, label='DQN', color='blue', marker='^', markevery=mark_every)[0]

plt.tight_layout()
plt.legend(handles=[a1, a2, a3])
plt.grid(True)
plt.savefig("CSTRL\graphs\GodotTestbed.png")
