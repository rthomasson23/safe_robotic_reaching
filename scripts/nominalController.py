import numpy as np


class NominalController(object):

  def __init__(self, pb, bodyUniqueID, jointIDs):
    self._pb = pb
    self.bodyID = bodyUniqueID
    self.jointIDs = jointIDs

  def computePD(self, qdes, kps, kds, maxForce):
      n = len(self.jointIDs)
      q = np.zeros(n)
      qdot = np.zeros(n)
      for i in range(n):
          jointState = self._pb.getJointState(self.bodyID, self.jointIDs[i])
          q[i] = jointState[0]
          qdot[i] = jointState[1]
      qError = qdes - q
      Kp = kps
      Kd = kds
      # force = -Kp*qError - Kd*qdot
      force = Kp*qError - Kd*qdot
      force = np.clip(force, -maxForce, maxForce)
      # print(qError)
      print(force)
      return force
