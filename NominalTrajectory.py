# This file calculates the nominal trajectory for an RRR planar manipulator navigating a cluttered environment to a goal
# Created: 5/23/21 by Nathaniel Agharese

# TO-DO:
# Generalize trajectory creation for N splines

import numpy as np
import scipy.optimize as scp
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import os

# Spline constants
q0 = np.array([0.99*np.pi/2, -0.99*np.pi, 0.99*np.pi])
q0 = np.array([0, np.pi/6, np.pi/6])
dq0 = np.array([0, 0, 0])
qT = np.array([np.pi/2, 0, 0])
dqT = np.array([0, 0, 0])
uMax = 5
T_Max = 5 # Maximum time for operation [s]
dt = 0.001
numKnots = 2

# Environmental constants
L = [.1, .1, 0.09328183]
rj = np.array([[0.06],[0.06]]) #obstacle radius
obstacles = np.array([[-0.1, 0.2],[0.1, 0.2]]) #Two obstacles
numLinks = 3

z0 = np.ones(numKnots*numLinks + 1)
for j in range(numKnots):
    z0[numLinks * j: numLinks * (j + 1)] = (j + 1) * (qT - q0) / (numKnots + 1) + q0

def GenerateTraj(z):
    T = np.absolute(z[-1])
    N = int(np.floor(T/dt))
    n = numKnots + 1
    Q = np.zeros((N,3*numLinks))
    for i in range(numLinks):
        # Solve for the spline parameters using PVA, initial, and terminal constraints
        A = np.zeros((4 * (numKnots + 1), 4 * (numKnots + 1)))
        B = np.zeros(4 * (numKnots + 1))
        A[0:2,2:4] = np.eye(2)[:,:]
        A_T = np.array([[3 * T**2, 2 * T, 1, 0],[T**3, T**2, T, 1]])
        A[-2:,-4:] = A_T[:,:]
        B[0] = dq0[i]
        B[1] = q0[i]
        B[-2] = dqT[i]
        B[-1] = qT[i]
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
                    Qi[k,0] = x[4*j]*t**3 + x[4*j + 1]*t**2 + x[4*j + 2]*t + x[4*j + 3]
                    Qi[k,1] = 3*x[4*j]*t**2 + 2*x[4*j + 1]*t + x[4*j + 2]
                    Qi[k,2] = 6*x[4*j]*t + 2*x[4*j + 1]
                    break
        Q[:,i*numLinks] = Qi[:,0]
        Q[:,i*numLinks + 1] = Qi[:,1]
        Q[:,i*numLinks + 2] = Qi[:,2]
    return Q

# Average percentage geometrical error over time, sum over links and obstacles, average over links
def Intrusion(Q):
    s = np.shape(Q)
    nj = obstacles.shape[0]
    w = 0.035 #link width
    rk = w / 2 #radius of circles on arm
    ni = [int(np.ceil(i / rk)) for i in L] # number of circles overlayed on each link
    xLast = 0
    yLast = 0
    cost = 0
    for k in range(numLinks):
        # Make the circles on the arm
        x = np.zeros((s[0],ni[k]))
        y = np.zeros((s[0],ni[k]))
        error_ij = np.zeros((ni[k],nj))
        # Iterate through each circle on the link
        for i in range(ni[k]):
            x[:,i] = xLast + rk*np.cos(Q[:,k])
            y[:,i] = yLast + rk*np.sin(Q[:,k])
            xLast = x[:,i]
            yLast = y[:,i]
            # Iterate through each obstacle
            for j in range(nj):
                obstacle = np.repeat(obstacles[j,:,np.newaxis],s[0],axis=1).T
                xy = np.array([x[:,i], y[:,i]]).T
                d = np.linalg.norm(obstacle - xy, axis=1)
                e = (rk + rj[j]) - d
                # Don't reward being far from the obstacle
                e = np.maximum(int(0),e)
                # Penalize the percentage of total possible obstacle intrusion: overlapping circles
                e = e / (rk + rj[j])
                # Average of intrusion over total time of intrusion
                T_in = np.sum(np.where(e==0,0,1))
                error_ij[i,j] = np.sum(e) / np.maximum(T_in,1)
        # Sum error along link length and across obstacles
        cost += np.sum(error_ij)
    # Average over links
    # cost = cost / numLinks
    return cost

def TrajError(z):
    Q = GenerateTraj(z)
    T = np.absolute(z[-1])
    # Call a helper function to calculate the actual error
    s = np.shape(Q)
    q = np.zeros((s[0],3))
    q[:,0] = Q[:,0]
    q[:,1] = Q[:,3] + Q[:,0]
    q[:,2] = Q[:,6] + Q[:,3] + Q[:,0]
    errorIn = 10 * Intrusion(q)
    errorT = 1 * (T / T_Max)
    error = errorIn + errorT
    return error

def Dynamics(z):
    Q = GenerateTraj(z)
    # Robot Constants
    L1, L2 = L[0], L[1]
    l1, l2, l3 = 0.0497183, 0.0497183, 0.0439482
    m1, m2, m3 = 0.643789, 0.643789, 0.58676
    Izz1, Izz2, Izz3 = 0.000724384, 0.000724384, 0.000503074
    n = numLinks # number of links
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
        dqSquared = np.array([dq1**2, dq2**2, dq3**2])
        dqdq = np.array([dq1*dq2, dq1*dq3, dq2*dq3])
        M = np.matrix([[Izz1 + Izz2 + Izz3 + L1**2*m2 + L1**2*m3 + L2**2*m3 + l1**2*m1 + l2**2*m2 + l3**2*m3 + 2*L1*l3*m3*np.cos(q2 + q3) + 2*L1*L2*m3*np.cos(q2) + 2*L1*l2*m2*np.cos(q2) + 2*L2*l3*m3*np.cos(q3), m3*L2**2 + 2*m3*np.cos(q3)*L2*l3 + L1*m3*np.cos(q2)*L2 + m2*l2**2 + L1*m2*np.cos(q2)*l2 + m3*l3**2 + L1*m3*np.cos(q2 + q3)*l3 + Izz2 + Izz3, Izz3 + l3**2*m3 + L1*l3*m3*np.cos(q2 + q3) + L2*l3*m3*np.cos(q3)],
            [m3*L2**2 + 2*m3*np.cos(q3)*L2*l3 + L1*m3*np.cos(q2)*L2 + m2*l2**2 + L1*m2*np.cos(q2)*l2 + m3*l3**2 + L1*m3*np.cos(q2 + q3)*l3 + Izz2 + Izz3, m3*L2**2 + 2*m3*np.cos(q3)*L2*l3 + m2*l2**2 + m3*l3**2 + Izz2 + Izz3, m3*l3**2 + L2*m3*np.cos(q3)*l3 + Izz3],
            [Izz3 + l3**2*m3 + L1*l3*m3*np.cos(q2 + q3) + L2*l3*m3*np.cos(q3), m3*l3**2 + L2*m3*np.cos(q3)*l3 + Izz3, m3*l3**2 + Izz3]])
        C = np.matrix([[0, -L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), -l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3))],
            [-L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), 0, -L2*l3*m3*np.sin(q3)],
            [-l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -L2*l3*m3*np.sin(q3), 0]])
        B = np.matrix([[-2*L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3))],
            [-2*L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*L2*l3*m3*np.sin(q3)],
            [-2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*L2*l3*m3*np.sin(q3)]])
        v = np.matmul(C,dqSquared) + np.matmul(B,dqdq)
        u[n*t:n*t + n,0] = np.matmul(M,ddq) + v
    return u[:,0]

ineqDynamics = {'type': 'ineq', 'fun': lambda z: np.array([np.amin(uMax - np.absolute(Dynamics(z)))])}
ineqTraj = {'type': 'ineq', 'fun': lambda z: np.array([np.amin(np.pi - np.absolute(GenerateTraj(z)[:,[0,3,6]]))])}
lb = -np.pi * np.ones(z0.size)
lb[-1] = 1
ub = np.pi * np.ones(z0.size)
ub[-1] = T_Max
bnds = scp.Bounds(lb,ub)

startT = time.time()
res = scp.minimize(TrajError, z0, method='SLSQP', jac=None, bounds = bnds,
    constraints=[ineqTraj,ineqDynamics], options = {'disp':True})
endT = time.time()
z = res.x
print('Spline parameters are: ',z)
print('Time of optimization: ',endT - startT)

# Convert uOpt from 3*T x 1 to T x 3 matrix
u = Dynamics(z)
T = int(np.size(u)/numLinks)
uOpt = np.zeros((T,numLinks))
for t in range(T):
    uOpt[t,:] = np.transpose(u[numLinks*t:numLinks*t + numLinks])
np.save('uOpt',uOpt)
np.save('zOpt',z)

# Plot the joint angles
Q = GenerateTraj(z)
np.save('Q_Opt',Q)
# cwd = os.getcwd()
# Q = np.load(cwd + '/Q_Opt.npy')
q1 = Q[:,0]
q2 = Q[:,3]
q3 = Q[:,6]
# plt.plot(q1)
# plt.plot(q2)
# plt.plot(q3)
# plt.show()

# # Animate PD controller output
# qPD = np.load(cwd + '/q_PD.npy')
# q1 = qPD[:,0]
# q2 = qPD[:,1]
# q3 = qPD[:,2]

# Animate the solution

#Links over time
x1 = L[0]*np.cos(q1)
y1 = L[0]*np.sin(q1)
x2 = L[1]*np.cos(q2 + q1) + x1
y2 = L[1]*np.sin(q2 + q1) + y1
x3 = L[2]*np.cos(q3 + q2 + q1) + x2
y3 = L[2]*np.sin(q3 + q2 + q1) + y2

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.4, 0.4), ylim=(-0.4, 0.4))
ax.grid()

#Obstacles
for j in range(obstacles.shape[0]):
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    a = rj[j][0] * np.cos( theta ) + obstacles[j,0]
    b = rj[j][0] * np.sin( theta ) + obstacles[j,1]
    ax.plot(a,b)

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i], x3[i]]
    thisy = [0, y1[i], y2[i], y3[i]]
    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(q1)),
                              interval=1, blit=True, init_func=init)

# ani.save('SLSQP_N2.mp4', fps=15)
plt.show()