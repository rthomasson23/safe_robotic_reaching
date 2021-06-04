import numpy as np


class NominalController(object):

  def __init__(self, pb, bodyUniqueID, jointIDs, ID_EE):
    self._pb = pb
    self.bodyID = bodyUniqueID
    self.jointIDs = jointIDs
    self.ID_EE = ID_EE


  def computePD_op(self, xdes, kp, kv_j, kv_op, maxTorque):
      n = len(self.jointIDs)
      q = np.zeros(n)
      qdot = np.zeros(n)
      for i in range(n):
          jointState = self._pb.getJointState(self.bodyID, self.jointIDs[i])
          q[i] = jointState[0]
          qdot[i] = jointState[1]

      # get current position of EE
      ee_state = self._pb.getLinkState(self.bodyID, self.ID_EE, computeLinkVelocity=True)
      x = np.asarray(ee_state[0])
      xdot = np.asarray(ee_state[6])

      # get Jacobian
      local_position = [0, 0, 0]
      des_qddot = np.zeros(n) # p sure this is just a dummy value but pybullet needs it
      Jv, Jw = self._pb.calculateJacobian(self.bodyID, self.ID_EE, local_position, list(q), list(qdot), list(des_qddot))
      Jv = np.asarray(Jv)

      xError = xdes - x
      xError[2] = 0 # we're neglecting any z-error since we consider it planar

      force = (kp * xError - kv_op * xdot)
      tau = np.dot(Jv.T, force) - kv_j * qdot  # cartesian space damping
      tau = np.clip(tau, -maxTorque, maxTorque)
      return tau
