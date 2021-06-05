import matplotlib.pyplot as plt

N = [2, 3, 4, 5, 6]
error = [13.76, 11.68, 20.62, 12.48, 13.31]
time = [50.4, 50.25, 234.44, 189.33, 286.36]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of Splines')
ax1.set_ylabel('Intrusion Error', color=color)
ax1.plot(N, error, 'o-',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Operation Time [s]', color=color)  # we already handled the x-label with ax1
ax2.plot(N, time,'o-', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()