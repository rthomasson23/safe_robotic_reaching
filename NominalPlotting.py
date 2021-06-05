"""
File: Nominal Plotting
    This file is used to aid NominalTrajectory.py in plot generation.
"""
import matplotlib.pyplot as plt
import numpy as np

"""
Function kinPlots:
    This function plots the kinematic trajectory passed to it at 5
    time intervals, 3 of which are inputs. This works for a 3 link
    RRR manipulator.
"""
def kinPlots(q1,q2,q3,L,t,T_end,ax,lgnd):
    # xy position of distal end of each link over time
    x1 = L[0]*np.cos(q1)
    y1 = L[0]*np.sin(q1)
    x2 = L[1]*np.cos(q2) + x1
    y2 = L[1]*np.sin(q2) + y1
    x3 = L[2]*np.cos(q3) + x2
    y3 = L[2]*np.sin(q3) + y2

    # intermediate time indices
    t1 = int(np.floor((t[0]/T_end) * x1.size))
    t2 = int(np.floor((t[1]/T_end) * x1.size))
    t3 = int(np.floor((t[2]/T_end) * x1.size))

    # plot
    ax.plot([0, x1[-1], x2[-1], x3[-1]], [0, y1[-1], y2[-1], y3[-1]],'o-')
    ax.plot([0, x1[t3], x2[t3], x3[t3]], [0, y1[t3], y2[t3], y3[t3]],'o-')
    ax.plot([0, x1[t2], x2[t2], x3[t2]], [0, y1[t2], y2[t2], y3[t2]],'o-')
    ax.plot([0, x1[t1], x2[t1], x3[t1]], [0, y1[t1], y2[t1], y3[t1]],'o-')
    ax.plot([0, x1[0], x2[0], x3[0]], [0, y1[0], y2[0], y3[0]],'o-')
    plt.legend(['Obstacle','Obstacle',lgnd[4],lgnd[3],lgnd[2],lgnd[1],lgnd[0]])
    plt.show()
    return

"""
Function: splineData:
    This function plots the results of the spline trajectory optimizer
    when ran over a range of splines. The data was manually collecting
    and is therefore hard coded.
"""
def splineData():
    # data
    N = [2, 3, 4, 5, 6]
    error = [13.76, 11.68, 20.62, 12.48, 13.31]
    time = [50.4, 50.25, 234.44, 189.33, 286.36]

    # Plot based on code for plotting two y-axes found on https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
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
    return