import numpy as np
import scipy.optimize as scp
import matplotlib.pyplot as plt

# Define necessary constants
q0 = np.array([np.pi/9, -2*np.pi/9, np.pi/9])
dq0 = np.array([0, 0, 0])

uOpt = np.load('uOpt.npy')
z = np.load('zOpt.npy')

T = z[-1]
dt = 0.001
tLength = int(np.floor(T/dt))

time = np.linspace(0,T,tLength)
q = np.zeros((tLength,3))
q[0,:] = q0[:]
dq = np.zeros((tLength,3))

ddq = np.zeros((tLength,3))
L1, L2 = .1, .1
l1, l2, l3 = 0.0497183, 0.0497183, 0.0439482
m1, m2, m3 = 0.643789, 0.643789, 0.58676
Izz1, Izz2, Izz3 = 0.000724384, 0.000724384, 0.000503074

for t in range(tLength):
    if t > 0:
        dq[t,:] = ddq[t-1,:] * dt + dq[t-1,:]
        q[t,:] = dq[t,:] * dt + q[t-1,:]
    
    # Run the dynamics to check the torque constraints
    q2, q3 = q[t,1], q[t,2]
    dq1, dq2, dq3 = dq[t,0], dq[t,1], dq[t,2]
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
    ddq[t,:] = np.transpose(np.matmul(np.linalg.inv(M),np.transpose(uOpt[t,:] - v)))

plt.plot(q[:,0])
plt.plot(q[:,1])
plt.plot(q[:,2])
plt.show()