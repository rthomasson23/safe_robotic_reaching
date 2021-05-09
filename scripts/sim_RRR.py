import pybullet as p
import numpy as np
import scipy.signal
import copy
import math
import pybullet_data
import os
import time
from nominalController import NominalController

from utils.utils import sleeper

class RRRManipulator:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(),
                initBasePos=[0., 0., 0.],
                initPos=[0.3, .0, .0],
                initOrn=p.getQuaternionFromEuler([0., 0., 0.]),
                timeStep=0.001):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.sleeper = sleeper(p, self.timeStep)
    self.initBasePos = np.array(initBasePos)
    self.initPos = self.initBasePos + np.array(initPos)
    self.initOrn = initOrn

    # Load URDF model
    self.robotUid = p.loadURDF("../RRR_manipulator_description/RRR.urdf")

    # set the position of the base to be on the table
    p.resetBasePositionAndOrientation(self.robotUid,
                                      self.initBasePos,
                                      p.getQuaternionFromEuler([0, 0, 0]))

    # get important joint information
    self.getJointsInfo()

    # setup parameters for control
    self.setupControls()

    # reset home position for arm
    self.resetPose()



  def getJointsInfo(self):
    """ Finds the number of joints and the index of specific joints for controls """

    self.numArmJoints = 0
    # identify gripper joint and ee link
    self.ID_0 = -1
    self.ID_1 = -1
    self.ID_2 = -1
    self.ID_EE = -1
    for i in range(p.getNumJoints(self.robotUid)):
      jointType = p.getJointInfo(self.robotUid, i)[2]
      # get the gripper joint
      if (jointType == p.JOINT_FIXED):
        self.ID_EE = i
      if ("joint0" in str(p.getJointInfo(self.robotUid, i)[1])):
        self.ID_0 = i
      if ("joint1" in str(p.getJointInfo(self.robotUid, i)[1])):
        self.ID_1 = i
      if ("joint2" in str(p.getJointInfo(self.robotUid, i)[1])):
        self.ID_2 = i


      # accumulate number of joints
      if (jointType != p.JOINT_FIXED):
        self.numArmJoints += 1


  def resetPose(self):
    """ Sets home joint position """
    jointPoses = p.calculateInverseKinematics(self.robotUid,
                                              self.ID_EE,
                                              self.initPos,
                                              maxNumIterations=100,
                                              residualThreshold=.01)
    self.setArmJointDes(jointPoses[:self.numArmJoints])
    self.sleeper.sleepSim(1, self.step)


  def setupControls(self):
    """ Sets up tentacleBot controller """
    # create nominal controller instance
    jointIDs = [self.ID_0, self.ID_1, self.ID_2]
    self.armNomControl = NominalController(p, self.robotUid, jointIDs)

    # set default controller parameters
    self.desiredJointAngles = np.array([0, np.pi/6, np.pi/5, 0])
    self.maxTorque = 100.0
    self.kp = 1
    self.kv = 1

    # disable position control to use explicit Torque control
    for i in range(self.numArmJoints):
      p.setJointMotorControl2(self.robotUid, i+1, p.POSITION_CONTROL, targetPosition=0.00, force=0.00)


  def setArmJointDes(self, jointDes):
    """ Set the controlled target position for each joint """
    if (len(jointDes) != self.numArmJoints):
      print("Wrong jointDes vector size")
      raise Exception
    self.jointDes = jointDes


  def setArmPoseDes(self, pos, orn = p.getQuaternionFromEuler([math.pi, 0., 0.])):
    """ Set the arm target pose """
    if (len(pos) != 3) or (len(orn) != 4):
      print("Position not a 3D vector or Orientation not a Quaternion.")
    jointDes = p.calculateInverseKinematics(self.robotUid, self.ID_EE, targetPosition = np.array(pos))
    # jointDes = np.array([np.pi/4, np.pi/4, np.pi/4])
    self.setArmJointDes(jointDes[:self.numArmJoints])


  def getLocalEE(self):
    eePos = p.getLinkState(self.robotUid,
                          self.ID_EE)[0]
    return np.array(eePos) - self.initPos

  def getWorldEE(self):
    eePos = p.getLinkState(self.robotUid,
                          self.ID_EE)[0]
    return np.array(eePos)

  def getContact(self):
    contacts = p.getContactPoints(bodyA=self.robotUid,
                                linkIndexA=self.distalID)
    # TODO transform to local coordinates
    res = len(contacts) > 0
    if res:
      return res, contacts[0][5]
    else:
      # just return a dummy location
      return res, [0,0,0]


  def step(self):
    t = time.time()
    #print(t-self.prevTime)
    self.prevTime = t

    # arm control
    desq = np.array([np.sin(t)*np.pi/2, np.cos(t)*np.pi/2, -np.sin(0.3*t)*np.pi/2])
    # desq = self.desiredJointAngles
    force2apply = self.armNomControl.computePD(desq, 0.3, .1, self.maxTorque)
    p.setJointMotorControl2(self.robotUid, self.ID_0, p.TORQUE_CONTROL, force=force2apply[0])
    p.setJointMotorControl2(self.robotUid, self.ID_1, p.TORQUE_CONTROL, force=force2apply[1])
    p.setJointMotorControl2(self.robotUid, self.ID_2, p.TORQUE_CONTROL, force=force2apply[2])



if __name__ == "__main__":
  p.connect(p.GUI)
  urdfRoot=pybullet_data.getDataPath()
  p.loadURDF(os.path.join(urdfRoot, "plane.urdf"), 0, 0, 0)
  p.setGravity(0, 0, -9.81)
  h = RRRManipulator(initBasePos=np.array([-.1, 0, 0.025]),
              initPos=np.array([0., 0., 0.]))
  sim_count = 0
  init_pos = h.getWorldEE()
  while (1):
    sim_count += 1
    time.sleep(h.timeStep)
    if sim_count > 3000:
      t = time.time()
      # pos = [0.2*np.cos(t) + 0.2,  .25*np.cos(2*t), 0.] + init_pos
      # orn = p.getQuaternionFromEuler([0, 0., 0.])
      # h.setArmPoseDes(pos, orn)
      # print(h.getContact())
    h.step()
    p.stepSimulation()
