import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

sum_force_safe = np.load('sum_force_safe_failure.npy')
sum_force_PD = np.load('sum_force_PD_failure.npy')
sum_force_safe_appended = np.ones(np.shape(sum_force_PD)[0]) * sum_force_safe[-1]
sum_force_safe_appended[0:np.shape(sum_force_safe)[0]] = sum_force_safe
plt.figure()
plt.plot(sum_force_safe_appended, 'c', label='safe')
plt.plot(sum_force_PD, 'r--', label='nominal')
plt.legend()
# plt.title('Aggregate Force Applied')
plt.xlabel('simulation count')
plt.ylabel('aggregate normal force [N]')
plt.show()


# xee_safe = np.load('xee_safe.npy')
# xee_PD = np.load('xee_PD.npy')
#
# fig, axs = plt.subplots(2)
# fig.suptitle('End-Effector Trajectories')
# axs[0].plot(xee_PD[:, 0], 'r--', label='nominal')
# axs[0].plot(xee_safe[:, 0], 'c', label='safe')
# xs = np.linspace(0, np.shape(xee_PD)[0], np.shape(xee_PD)[0])
# ys = np.ones(np.shape(xee_PD)[0]) * -0.25
# axs[0].plot(xs, ys, 'gold', label='target')
# axs[0].set(ylabel='x position')
# axs[0].legend(loc='upper right')
# #
# axs[1].plot(xee_PD[:, 1], 'r--', label='nominal')
# axs[1].plot(xee_safe[:, 1], 'c', label='safe')
# xs = np.linspace(0, np.shape(xee_PD)[0], np.shape(xee_PD)[0])
# ys = np.ones(np.shape(xee_PD)[0]) * -0.15
# axs[1].plot(xs, ys, 'gold', label='target')
# axs[1].set(ylabel='y position')
# axs[1].legend(loc='upper right')
# #
# plt.show()