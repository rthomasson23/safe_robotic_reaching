# This file calculates the nominal trajectory for an RRR planar manipulator navigating a cluttered environment to a goal
# Created: 5/23/21 by Nathaniel Agharese

import numpy as np
import scipy.optimize as scp
import matplotlib.pyplot as plt

# Define necessary constants
q0 = np.array([np.pi/9, -2*np.pi/9, np.pi/9])
dq0 = np.array([0, 0, 0])
qT = np.array([np.pi/2, -np.pi/3, 2*np.pi/3])
dqT = np.array([0, 0, 0])
uMax = 0.5
T_Max = 5 # Maximum time for operation [s]
numKnots = 1
numJoints = 3
z0 = np.ones((numKnots*numJoints + 1,1))
z0 = z0[:,0]
# z0[-1] = 2

def GenerateTraj(z,q0,dq0,qT,dqT):
    T = z[-1]
    dt = 0.001
    N = int(np.floor(T/dt))
    numJoints = np.size(z) - 1
    Q = np.zeros((N,3*numJoints))
    for i in range(numJoints):
        a1 = z[i]
        # Solve for the spline parameters using PVA, initial, and terminal constraints
        a = np.array([[0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [(T/2)**2, (T/2), 1, -(T/2)**3, -(T/2)**2, -(T/2), -1],
            [(T/2), 1/2, 0, -(3/2)*(T/2)**2, -(T/2), -1/2, 0],
            [1/2, 0, 0, -(3/2)*(T/2), -1/2, 0, 0],
            [0, 0, 0, T**3, T**2, T, 1],
            [0, 0, 0, 3*T**2, 2*T, 1, 0]])
        b = np.array([q0[i], dq0[i], -a1*(T/2)**3, -(3/2)*a1*(T/2)**2, -(3/2)*a1*(T/2), qT[i], dqT[i]])
        x = np.linalg.solve(a,b)
        # Discretize and create the trajectory for this joint
        Qi = np.zeros((N,3))
        for k in range(N):
            t = k*dt
            if t <= T/2:
                Qi[k,0] = a1*t**3 + x[0]*t**2 + x[1]*t + x[2]
                Qi[k,1] = 3*a1*t**2 + 2*x[0]*t + x[1]
                Qi[k,2] = 6*a1*t + 2*x[0]
            else:
                Qi[k,0] = x[3]*t**3 + x[4]*t**2 + x[5]*t + x[6]
                Qi[k,1] = 3*x[3]*t**2 + 2*x[4]*t + x[5]
                Qi[k,2] = 6*x[3]*t + 2*x[4]
        Q[:,i*numJoints] = Qi[:,0]
        Q[:,i*numJoints + 1] = Qi[:,1]
        Q[:,i*numJoints + 2] = Qi[:,2]
    # plt.plot(Q[:,0])
    # plt.show()
    return Q

def Intrusion(Q):
    s = np.shape(Q)
    numJoints = 3
    obstacles = np.array([[0, 0.1]]) #Start with one obstacle
    nj = 1
    L = [.1, .1, 0.09328183]
    rj = np.array([[0.5]]) #obstacle radius
    w = 0.035 #link width
    rk = w / 2 #radius of circles on arm
    ni = [int(np.ceil(i / rk)) for i in L] # number of circles overlayed on each link
    xLast = 0
    yLast = 0
    cost = 0
    for k in range(numJoints):
        # Make the circles on the arm
        x = np.zeros((s[0],ni[k]))
        y = np.zeros((s[0],ni[k]))
        dij = np.zeros((s[0],ni[k],nj))
        for i in range(ni[k]):
            x[:,i] = xLast + rk*np.cos(Q[:,k])
            y[:,i] = yLast + rk*np.sin(Q[:,k])
            xLast = x[:,i]
            yLast = y[:,i]
            for j in range(nj):
                for t in range(s[0]):
                    dij[t,i,j] = (rk + rj[j]) - np.linalg.norm(obstacles[j,:] - [x[t,i], y[t,i]])
                    dij[t,i,j] = np.maximum(0,dij[t,i,j])
        cost += np.sum(dij)

    return cost

def TrajError(z,q0,dq0,qT,dqT):
    Q = GenerateTraj(z,q0,dq0,qT,dqT)
    # # minimize the path length for testing this opmitiztion method: UPDATE TO ACTUAL INTRUSION ERROR LATER
    # error = np.linalg.norm(Q[:,0],1) + np.linalg.norm(Q[:,3],1) + np.linalg.norm(Q[:,6],1)

    # Call a helper function to calculate the actual error
    s = np.shape(Q)
    q = np.zeros((s[0],3))
    q[:,0] = Q[:,0]
    q[:,1] = Q[:,3]
    q[:,2] = Q[:,6]
    error = Intrusion(q)
    return error

def Dynamics(z,q0,dq0,qT,dqT):
    Q = GenerateTraj(z,q0,dq0,qT,dqT)
    # Robot Constants
    L1, L2 = .1, .1
    l1, l2, l3 = 0.0497183, 0.0497183, 0.0439482
    m1, m2, m3 = 0.643789, 0.643789, 0.58676
    Izz1, Izz2, Izz3 = 0.000724384, 0.000724384, 0.000503074
    n = 3 # number of links
    s  = np.shape(Q)
    u = np.zeros((s[0]*3,1))
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
        u[3*t:3*t + 3,0] = np.matmul(M,ddq) + v
    return u[:,0]

ineqDynamics = {'type': 'ineq', 'fun': lambda z: np.array([np.amin(uMax - np.absolute(Dynamics(z,q0,dq0,qT,dqT)))])}
ineqSpline = {'type': 'ineq', 'fun': lambda z: np.absolute(z) - 0.01}
ineqTime = {'type': 'ineq', 'fun': lambda z: T_Max - z[-1]}

res = scp.minimize(TrajError, z0, args=(q0,dq0,qT,dqT), method='SLSQP', jac=None, constraints=[ineqDynamics, ineqSpline, ineqTime])#, options = {'maxiter':1})
z = res.x
print('The optimization is a success: ',res.success)
print('Spline parameters are: ',z)

u = Dynamics(z,q0,dq0,qT,dqT)
# Convert uOpt from 3*T x 1 to T x 3 matrix
T = int(np.size(u)/3)
uOpt = np.zeros((T,3))
for t in range(T):
    uOpt[t,:] = np.transpose(u[3*t:3*t + 3])
np.save('uOpt',uOpt)
np.save('zOpt',z)
Q = GenerateTraj(z,q0,dq0,qT,dqT)
plt.plot(Q[:,0])
plt.plot(Q[:,3])
plt.plot(Q[:,6])
plt.show()

# Loop within the opitmizer
    # Generate the trajectory from the current set of spline parameters
    # Discretize the trajectory
    # Compute the required torque from the dynamics constraint
    # Verify the torque constraint
        # If valid, calculate the cost