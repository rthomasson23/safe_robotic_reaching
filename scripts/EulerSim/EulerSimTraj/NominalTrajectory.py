"""
File: Nominal Trajectory:
    This file determines and assesses the nominal trajectory for an RRR planar manipulator navigating a cluttered environment to a goal.
    If 'PD' is False, this file determines the nominal trajectory using a spline optimization process.
    If 'PD' is True, this file loads existing trajectory data from a PD controller. In this case it must be run with the debugger for cwd to work properly
    This file outputs 2 plots and one animation, in addition to printing relevant values, in the assessment of the nominal trajectory.

Created: 5/23/21 by Nathaniel Agharese
"""

import numpy as np
import scipy.optimize as scp
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import os
from NominalPlotting import kinPlots
from NominalPlotting import splineData

# True: loads and analyzes the PD data, must use the debugger for cwd to work properly
# False: runs the spline optimizer
PD = False

# Spline constants
q0 = np.array([0.99*np.pi/2, -0.99*np.pi, 0.99*np.pi])          # [rad] initial robot pose
dq0 = np.array([0, 0, 0])                                       # [rad/s] initial joint velocities
qT = np.array([np.pi/2, 0, 0])                                  # [rad] end pose
dqT = np.array([0, 0, 0])                                       # [rad/s] end joint velocities
uMax = 2                                                        # [Nm] limit on absolute value of torques
T_Max = 10                                                      # [s] maximum time for operation
dt = 0.001                                                      # [s] time step
numKnots = 1                                                    # [] number of knots connecting two cubic splines for a joint

# Environmental constants
L = [.1, .1, 0.09328183]                                        # [m] link length
w = 0.035                                                       # [m] link width
rk = w / 2                                                      # [m] radius of circles on arm
obstacles = np.array([[-0.1, 0.2],[0.1, 0.2]])                  # [x,y] Two obstacles
rj = np.array([[0.06],[0.06]])                                  # [m] obstacle radius
# Impassable obstacles
rj = np.array([[0.09],[0.09]])
numLinks = 3

# Optimization globals
z0 = np.ones(numKnots*numLinks + 1)                             # [rad,..,rad,s] design parameters for spline generation
for j in range(numKnots):
    z0[numLinks * j: numLinks * (j + 1)] = (j + 1) * (qT - q0) / (numKnots + 1) + q0
errInGain = 10
rho = 1                                                         # initial penalty gain
gamma = 2                                                       # penalty multiplier

"""
Function Generate Trajectory:
    This function generates the kinematic trajectory for a
    general vector of spline parameters z. Using the knot
    position and runtime values contained in z, this function
    calculates the corresponding cubic spline coefficients by
    solving the linear system of equations.  It then
    discretizes the kinematic trajectory, enforces the
    intitial and terminal constraints defined above, and
    returns a matrix of joint trajectories over time.
"""
def GenerateTraj(z):
    T = np.absolute(z[-1])
    # length of the time vector, must be a multiple of dt
    N = int(np.floor(T/dt))
    # number of cubic splines for a joint
    n = numKnots + 1
    Q = np.zeros((N,3*numLinks))
    for i in range(numLinks):
        # Solve for the spline parameters using PVA, initial, and terminal constraints
        # PVA: Position, Velocity, Acceleration
        A = np.zeros((4 * (numKnots + 1), 4 * (numKnots + 1)))
        B = np.zeros(4 * (numKnots + 1))
        # The terminal and initial constraints are constant and unique from intermediate
        # knot constraints
        A[0:2,2:4] = np.eye(2)[:,:]
        A_T = np.array([[3 * T**2, 2 * T, 1, 0],[T**3, T**2, T, 1]])
        A[-2:,-4:] = A_T[:,:]
        B[0] = dq0[i]
        B[1] = q0[i]
        B[-2] = dqT[i]
        B[-1] = qT[i]
        # Fill in the values for the intermediate knot constraints
        for j in range(numKnots):
            A[2 + 4*j, 4*j: 4*j + 8] = [0, 0, 0, 0, ((j+1)*T/n)**3, ((j+1)*T/n)**2, (j+1)*T/n, 1]
            A[3 + 4*j, 4*j: 4*j + 8] = [((j+1)*T/n)**3, ((j+1)*T/n)**2, (j+1)*T/n, 1, -((j+1)*T/n)**3, -((j+1)*T/n)**2, -(j+1)*T/n, -1]
            A[4 + 4*j, 4*j: 4*j + 8] = [3*((j+1)*T/n)**2, 2*(j+1)*T/n, 1, 0, -3*((j+1)*T/n)**2, -2*(j+1)*T/n, -1, 0]
            A[5 + 4*j, 4*j: 4*j + 8] = [6*(j+1)*T/2, 2, 0, 0, -6*(j+1)*T/2, -2, 0, 0]
            B[2 + 4*j] = z[i + numLinks*j]
            B[3 + 4*j] = 0
            B[4 + 4*j] = 0
            B[5 + 4*j] = 0
        x = np.linalg.solve(A,B)
        # Discretize and create the trajectory for this joint
        Qi = np.zeros((N,numLinks))
        for k in range(N):
            t = k*dt
            for j in range(n):
                if t <= (j+1)*T/n:
                    # Calculate the position, velocity, and acceleration for this joint at time t
                    Qi[k,0] = x[4*j]*t**3 + x[4*j + 1]*t**2 + x[4*j + 2]*t + x[4*j + 3]
                    Qi[k,1] = 3*x[4*j]*t**2 + 2*x[4*j + 1]*t + x[4*j + 2]
                    Qi[k,2] = 6*x[4*j]*t + 2*x[4*j + 1]
                    # Move on to the next set of cubic coefficients when done with this spline
                    break
        Q[:,i*numLinks] = Qi[:,0]
        Q[:,i*numLinks + 1] = Qi[:,1]
        Q[:,i*numLinks + 2] = Qi[:,2]
    return Q

"""
Function Intrusion:
    This function calculates the the intrusion cost for a set
    of joint angles Q. The intrusion cost is the largest,
    weighted percentage of intrusion that occurs along the
    manipulator. Percentage of intrusion is distance of
    intrusion over distance of largest possible intrusion.
"""
def Intrusion(Q):
    s = np.shape(Q)
    # number of obstacles
    nj = obstacles.shape[0]
    # number of circles overlayed on each link
    ni = [int(np.ceil(i / rk)) for i in L]
    xLast = 0
    yLast = 0
    cost = 0
    # Iterate through each link
    for k in range(numLinks):
        # Make the circles on the arm
        x = np.zeros((s[0],ni[k]))
        y = np.zeros((s[0],ni[k]))
        # Iterate through each circle on the link
        for i in range(ni[k]):
            # Calculate the xy position of this circle over time
            x[:,i] = xLast + rk*np.cos(Q[:,k])
            y[:,i] = yLast + rk*np.sin(Q[:,k])
            xLast = x[:,i]
            yLast = y[:,i]
            # Iterate through each obstacle
            for j in range(nj):
                obstacle = np.repeat(obstacles[j,:,np.newaxis],s[0],axis=1).T
                xy = np.array([x[:,i], y[:,i]]).T
                # The distance of intrusion via the euclidean norm
                d = np.linalg.norm(obstacle - xy, axis=1)
                e = (rk + rj[j]) - d
                # Don't reward being far from the obstacle
                e = np.maximum(int(0),e)
                # Penalize the percentage of total possible obstacle intrusion: overlapping circles
                e = e / (rk + rj[j])
                # Weight intrusion of proximal links more
                e = (ni[k]*numLinks - (i+1)*(k + 1) + 1) * e
                cost = np.maximum(cost,np.amax(e))
    return cost

"""
Function TrajError:
    This function calculates the objective function (a sum of
    intrusion error, time error, and penalty function value)
    for a given set of spline parameters.

    The minimum value of the penalty function is the maximum 
    value of the rest of the objective function so that all
    evaluations within the constraints are less than or equal
    to all evaluations that violate the constriants.
"""
def TrajError(z):
    # Nelder-Mead constraint assessment

    # Calculate the largest possible value for the objective function
    ni = [int(np.ceil(i / rk)) for i in L]
    errorMax = errInGain * np.sum(ni)
    global rho
    # upper limit on the penalty gain to avoid overflow issues.
    rhoCap = 10**10
    c1, c2, c3 = 0, 0, 0
    # time constraints
    T = np.absolute(z[-1])
    if T < dt or T > T_Max:
        c1 = rho * errorMax
    # kinematic constraints
    Q = GenerateTraj(z)
    qAbs = np.absolute(Q[:,[0,3,6]])
    if np.maximum(np.amax(qAbs - np.pi),0) > 0:
        c2 = rho*errorMax
    # torque constraints
    uAbs = np.absolute(Dynamics(z))
    if np.maximum(np.amax(uAbs - uMax),0) > 0:
        c3 = rho*errorMax

    # Evaluate the objective function

    # intrusion error
    s = np.shape(Q)
    q = np.zeros((s[0],3))
    q[:,0] = Q[:,0]
    q[:,1] = Q[:,3] + Q[:,0]
    q[:,2] = Q[:,6] + Q[:,3] + Q[:,0]
    errorIn = errInGain * Intrusion(q)
    # time error: percentage of total allowable time
    errorT =  (T / T_Max)
    # penalty function
    p = np.amax([c1, c2, c3])
    rho = np.where(p > 0, np.minimum(rho*gamma,rhoCap), rho)

    error = errorIn + p + errorT
    return error

"""
Function Dynamics:
    This function applies the dynamics of the manipulator
    to calculate the torque trajectories for each link. It
    takes in the spline parameters instead of the kinematics
    so that it can be applied to optimization methods that
    explicity handle constraints (SLSQP, etc.). It returns
    a 3*T x 1 vector, where the torques of each joint are
    stackec at each moment in time, and T is the numer of
    time steps.
"""
def Dynamics(z):
    Q = GenerateTraj(z)
    # Robot Constants
    L1, L2 = L[0], L[1]
    l1, l2, l3 = 0.0497183, 0.0497183, 0.0439482
    m1, m2, m3 = 0.643789, 0.643789, 0.58676
    Izz1, Izz2, Izz3 = 0.000724384, 0.000724384, 0.000503074
    n = numLinks
    s  = np.shape(Q)
    u = np.zeros((s[0]*n,1))
    # Run the dynamics to check the torque constraints
    for t in range(s[0]):
        q = np.zeros((n))
        dq = np.zeros((n))
        ddq = np.zeros((n))
        for i in range(n):
            q[i] = Q[t,i*n]
            dq[i] = Q[t,i*n + 1]
            ddq[i] = Q[t,i*n + 2]
        q2, q3 = q[1], q[2]
        dq1, dq2, dq3 = dq[0], dq[1], dq[2]
        # Define terms used for the Coriolis and Centrifugal forces
        dqSquared = np.array([dq1**2, dq2**2, dq3**2])
        dqdq = np.array([dq1*dq2, dq1*dq3, dq2*dq3])
        # mass matrix
        M = np.matrix([[Izz1 + Izz2 + Izz3 + L1**2*m2 + L1**2*m3 + L2**2*m3 + l1**2*m1 + l2**2*m2 + l3**2*m3 + 2*L1*l3*m3*np.cos(q2 + q3) + 2*L1*L2*m3*np.cos(q2) + 2*L1*l2*m2*np.cos(q2) + 2*L2*l3*m3*np.cos(q3), m3*L2**2 + 2*m3*np.cos(q3)*L2*l3 + L1*m3*np.cos(q2)*L2 + m2*l2**2 + L1*m2*np.cos(q2)*l2 + m3*l3**2 + L1*m3*np.cos(q2 + q3)*l3 + Izz2 + Izz3, Izz3 + l3**2*m3 + L1*l3*m3*np.cos(q2 + q3) + L2*l3*m3*np.cos(q3)],
            [m3*L2**2 + 2*m3*np.cos(q3)*L2*l3 + L1*m3*np.cos(q2)*L2 + m2*l2**2 + L1*m2*np.cos(q2)*l2 + m3*l3**2 + L1*m3*np.cos(q2 + q3)*l3 + Izz2 + Izz3, m3*L2**2 + 2*m3*np.cos(q3)*L2*l3 + m2*l2**2 + m3*l3**2 + Izz2 + Izz3, m3*l3**2 + L2*m3*np.cos(q3)*l3 + Izz3],
            [Izz3 + l3**2*m3 + L1*l3*m3*np.cos(q2 + q3) + L2*l3*m3*np.cos(q3), m3*l3**2 + L2*m3*np.cos(q3)*l3 + Izz3, m3*l3**2 + Izz3]])
        # centrifugal coefficients
        C = np.matrix([[0, -L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), -l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3))],
            [-L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), 0, -L2*l3*m3*np.sin(q3)],
            [-l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -L2*l3*m3*np.sin(q3), 0]])
        # coriolis coefficients
        B = np.matrix([[-2*L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3))],
            [-2*L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*L2*l3*m3*np.sin(q3)],
            [-2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*L2*l3*m3*np.sin(q3)]])
        # sum of centrifugal and coriolis forces
        v = np.matmul(C,dqSquared) + np.matmul(B,dqdq)

        u[n*t:n*t + n,0] = np.matmul(M,ddq) + v
    return u[:,0]

if(not PD):
    # Run the optimizer and print the corresponding relevant values

    print("N = ", numKnots)
    startT = time.time()
    res = scp.minimize(TrajError, z0, method='Nelder-Mead', jac=None,options = {'disp':True})
    endT = time.time()
    z = res.x
    print('Spline parameters are: ',z)
    print('Time of optimization: ',endT - startT)

    # Convert uOpt from 3*T x 1 to T x 3
    u = Dynamics(z)
    T = int(np.size(u)/numLinks)
    uOpt = np.zeros((T,numLinks))
    for t in range(T):
        uOpt[t,:] = np.transpose(u[numLinks*t:numLinks*t + numLinks])
    print('Maximum torque is', np.amax(np.absolute(u)))

    # Generate the kinematics
    Q = GenerateTraj(z)
    s  = np.shape(Q)
    q = np.zeros((s[0],3))
    q[:,0] = Q[:,0]
    q[:,1] = Q[:,3] + Q[:,0]
    q[:,2] = Q[:,6] + Q[:,3] + Q[:,0]

    # Setup for plotting kinematic trajectory
    t = [0.1, 0.2, 0.3]
    T_end = z[-1]
    lgnd = ['time = 0s','time = 0.1s','time = 0.2s','time = 0.3s','time = 0.4s']

    # Name for animation
    filename = 'Nelder_Mead.mp4'
else:
    # Only works using the debugger!
    cwd = os.getcwd()
    qPD = np.load(cwd + '/q_PD.npy')
    q1 = qPD[:,0]
    q2 = qPD[:,1]
    q3 = qPD[:,2]
    q = np.array([q1,q2 + q1,q3 + q2 + q1])
    # errorIn = errInGain * Intrusion(q.T)
    q = q.T
    T_end = 9.346
    dt = T_end/q1.size

    # Setup for plotting kinematic trajectory
    t = [0.5, 1.5, 4]
    lgnd = ['time = 0s','time = 0.5s','time = 1.5s','time = 4s','time = 9.3s']

    # Name for animation
    filename = 'PD.mp4'

# Calculate the optimal intrusion cost
errorIn = errInGain * Intrusion(q)
print("Intrusion error is:",errorIn)

# Plot the manipulator at discrete time steps

# Setup plot environment
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.3, 0.3), ylim=(-0.1, 0.3))
ax.set_aspect('equal', 'box')
ax.grid()
# Add the obstacles
for j in range(obstacles.shape[0]):
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    a = (rj[j][0]) * np.cos( theta ) + obstacles[j,0]
    b = (rj[j][0]) * np.sin( theta ) + obstacles[j,1]
    ax.plot(a,b)

# Create the plot of kinematic trajectories
kinPlots(q[:,0],q[:,1],q[:,2],L,t,T_end,ax,lgnd)

# Animate the motion of the robot based on the example on https://matplotlib.org/3.2.2/gallery/animation/double_pendulum_sgskip.html

# xy position of distal end of each link over time
x1 = L[0]*np.cos(q[:,0])
y1 = L[0]*np.sin(q[:,0])
x2 = L[1]*np.cos(q[:,1]) + x1
y2 = L[1]*np.sin(q[:,1]) + y1
x3 = L[2]*np.cos(q[:,2]) + x2
y3 = L[2]*np.sin(q[:,2]) + y2

# Re-setup the plot for animation:
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.3, 0.3), ylim=(-0.1, 0.3))
ax.set_aspect('equal', 'box')
ax.grid()
# Add the obstacles
for j in range(obstacles.shape[0]):
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    a = (rj[j][0]) * np.cos( theta ) + obstacles[j,0]
    b = (rj[j][0]) * np.sin( theta ) + obstacles[j,1]
    ax.plot(a,b)

# Line setup
line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.35, 0.1, '', transform=ax.transAxes, fontsize='xx-large', fontweight='bold')

"""
Function init:
    This function initializes the animation
"""
def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

"""
Function animate:
    This function is what updates the plot durinig
    each animation loop.
"""
def animate(i):
    thisx = [0, x1[i], x2[i], x3[i]]
    thisy = [0, y1[i], y2[i], y3[i]]
    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y1)),
                              interval=1, blit=True, init_func=init)
plt.show()
ani.save(filename, fps=1000)

# Plot the data of optimization results vs number of splines
splineData()