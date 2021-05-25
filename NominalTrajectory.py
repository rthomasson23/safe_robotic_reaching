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
uMax = 20
T_Max = 15 # Maximum time for operation [s]
numKnots = 1
numJoints = 3
z0 = np.ones((numKnots*numJoints + 1,1))
z0 = z0[:,0]

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

def TrajError(z,q0,dq0,qT,dqT):
    Q = GenerateTraj(z,q0,dq0,qT,dqT)
    # minimize the path length for testing this opmitiztion method: UPDATE TO ACTUAL INTRUSION ERROR LATER
    error = np.linalg.norm(Q[:,0],1) + np.linalg.norm(Q[:,3],1) + np.linalg.norm(Q[:,6],1)
    return error

def Dynamics(z,q0,dq0,qT,dqT):
    Q = GenerateTraj(z,q0,dq0,qT,dqT)
    # Robot Constants
    L1, L2, L3 = 1, 1, 1
    l1, l2, l3 = 1/2, 1/2, 1/2
    m1, m2, m3 = 1, 1, 1
    Izz1, Izz2, Izz3 = (1/3)*m1*L1**2, (1/3)*m2*L2**2, (1/3)*m3*L3**2
    n = 3 # number of links
    s  = np.shape(Q)
    u = np.zeros((s[0]*3,1))
    # Run the dynamics to check the torque constraints
    for t in range(s[0]):
        q = Q[t,0:n]
        q2, q3 = q[1], q[2]
        dq = Q[t,n:2*n]
        dq1, dq2, dq3 = dq[0], dq[1], dq[2]
        dqSquared = np.array([dq1**2, dq2**2, dq3**2])
        dqdq = np.array([dq1*dq2, dq1*dq3, dq2*dq3])
        ddq = Q[t,2*n:3*n]
        M = np.array([[Izz1 + Izz2 + Izz3 + L1**2*m2 + L1**2*m3 + L2**2*m3 + l1**2*m1 + l2**2*m2 + l3**2*m3 + 2*L1*l3*m3*np.cos(q2 + q3) + 2*L1*L2*m3*np.cos(q2) + 2*L1*l2*m2*np.cos(q2) + 2*L2*l3*m3*np.cos(q3), m3*L2**2 + 2*m3*np.cos(q3)*L2*l3 + L1*m3*np.cos(q2)*L2 + m2*l2**2 + L1*m2*np.cos(q2)*l2 + m3*l3**2 + L1*m3*np.cos(q2 + q3)*l3 + Izz2 + Izz3, Izz3 + l3**2*m3 + L1*l3*m3*np.cos(q2 + q3) + L2*l3*m3*np.cos(q3)],
            [m3*L2**2 + 2*m3*np.cos(q3)*L2*l3 + L1*m3*np.cos(q2)*L2 + m2*l2**2 + L1*m2*np.cos(q2)*l2 + m3*l3**2 + L1*m3*np.cos(q2 + q3)*l3 + Izz2 + Izz3, m3*L2**2 + 2*m3*np.cos(q3)*L2*l3 + m2*l2**2 + m3*l3**2 + Izz2 + Izz3, m3*l3**2 + L2*m3*np.cos(q3)*l3 + Izz3],
            [Izz3 + l3**2*m3 + L1*l3*m3*np.cos(q2 + q3) + L2*l3*m3*np.cos(q3), m3*l3**2 + L2*m3*np.cos(q3)*l3 + Izz3, m3*l3**2 + Izz3]])
        C = np.array([[0, -L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), -l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3))],
            [-L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), 0, -L2*l3*m3*np.sin(q3)],
            [-l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -L2*l3*m3*np.sin(q3), 0]])
        B = np.array([[-2*L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3))],
            [-2*L1*(l2*m2*np.sin(q2) + l3*m3*np.sin(q2 + q3) + L2*m3*np.sin(q2)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*L2*l3*m3*np.sin(q3)],
            [-2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*l3*m3*(L1*np.sin(q2 + q3) + L2*np.sin(q3)), -2*L2*l3*m3*np.sin(q3)]])
        v = np.dot(C,dqSquared) + np.dot(B,dqdq)
        u[3*t:3*t + 3,0] = np.dot(M,ddq) + v
    return u[:,0]

ineqDynamics = {'type': 'ineq', 'fun': lambda z: np.array([np.amin(uMax - np.absolute(Dynamics(z,q0,dq0,qT,dqT)))])}
ineqSpline = {'type': 'ineq', 'fun': lambda z: z}
ineqTime = {'type': 'ineq', 'fun': lambda z: T_Max - z[-1]}

res = scp.minimize(TrajError, z0, args=(q0,dq0,qT,dqT), method='SLSQP', jac=None, constraints=[ineqDynamics, ineqSpline, ineqTime])
z = res.x
print('The optimization is a success: ',res.success)
print('Spline parameters are: ',z)

uOpt = Dynamics(z,q0,dq0,qT,dqT)
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