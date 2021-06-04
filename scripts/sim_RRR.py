import pybullet as p
import numpy as np
import scipy.signal
import copy
import math
import pybullet_data
import pandas as pd
import os
import time
from nominalController import NominalController
from safeController import SafeController
import matplotlib.pyplot as plt
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
    self.prevTime = time.time()

    # Load URDF model
    self.robotUid = p.loadURDF("../RRR_manipulator_description/RRR.urdf")

    # visualize contact?
    self.vis_bool = False

    # load obstacles
    # self.obs_ID = p.loadURDF("../obstacle_description/obstacle_fixed.urdf", -0.24, 0.02, 0.02) # case 1
    # self.obs_ID = p.loadURDF("../obstacle_description/obstacle_fixed.urdf", -.15, 0.15, 0.02)  # case 2
    self.obs_ID = p.loadURDF("../obstacle_description/obstacle_fixed.urdf", 0, -0.22, 0.02)  # case 3
    # self.obs_ID = p.loadURDF("../obstacle_description/obstacle_fixed.urdf", -0.05, -0.15, 0.02)  # case 4
    p.changeDynamics(self.obs_ID, -1, restitution=0.1)  # makes the simulation more stable

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

      # set link to be frictionless
      p.changeDynamics(self.robotUid, i, lateralFriction=0.0)  # makes the simulation more stable
      # accumulate number of joints
      if (jointType != p.JOINT_FIXED):
        self.numArmJoints += 1

  def resetPose(self):
    """ Sets home joint position """
    # case 1
    # p.resetJointState(self.robotUid, self.ID_0, np.pi / 9)
    # p.resetJointState(self.robotUid, self.ID_1, -np.pi / 9)
    # p.resetJointState(self.robotUid, self.ID_2, np.pi / 9)

    # case 2
    # p.resetJointState(self.robotUid, self.ID_0, -np.pi/8)
    # p.resetJointState(self.robotUid, self.ID_1, 0)
    # p.resetJointState(self.robotUid, self.ID_2, 0)

    # case 3
    p.resetJointState(self.robotUid, self.ID_0, -8*np.pi/9)
    p.resetJointState(self.robotUid, self.ID_1, 0)
    p.resetJointState(self.robotUid, self.ID_2, 0)

    # case 4
    # p.resetJointState(self.robotUid, self.ID_0, -np.pi)
    # p.resetJointState(self.robotUid, self.ID_1, 0)
    # p.resetJointState(self.robotUid, self.ID_2, np.pi)


  def setupControls(self):
    """ Sets up tentacleBot controller """
    self.maxTorque = 10
    self.kp = 30
    self.kv_j = .15
    self.kv_op = 5

    # create nominal controller instance
    jointIDs = [self.ID_0, self.ID_1, self.ID_2]
    self.armNomControl = NominalController(p, self.robotUid, jointIDs, self.ID_EE)

    # create safe controller instance
    self.armSafeControl = SafeController(p, self.robotUid, jointIDs, self.ID_0, self.ID_1, self.ID_2, self.maxTorque)

    # set default controller parameters
    self.xdes = np.array([-0.25, -0.15, 0])
    p.addUserDebugLine(list(self.xdes), [self.xdes[0], self.xdes[1], 1]) # show goal location

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

  def visContact(self):
    contacts_link0 = p.getContactPoints(bodyA=self.robotUid, linkIndexA=self.ID_0)
    contacts_link1 = p.getContactPoints(bodyA=self.robotUid, linkIndexA=self.ID_1)
    contacts_link2 = p.getContactPoints(bodyA=self.robotUid, linkIndexA=self.ID_2)

    if len(contacts_link0) > 0:
      contact = contacts_link0[0]
      pcontact = np.asarray(contact[5])
      n = -1*np.asarray(contact[7])
      mag = contact[9]
      scaling_factor = 5
      p.addUserDebugLine(list(pcontact), list(pcontact + mag*scaling_factor*n), lifeTime=0.5, lineColorRGB=[1,0,0],
                         lineWidth=5)
    if len(contacts_link1) > 0:
      contact = contacts_link1[0]
      pcontact = np.asarray(contact[5])
      n = -1 * np.asarray(contact[7])
      mag = contact[9]
      scaling_factor = 5
      self.sum_force.append(self.sum_force[-1] + mag)
      p.addUserDebugLine(list(pcontact), list(pcontact + mag * scaling_factor * n), lifeTime=0.5,
                         lineColorRGB=[0, 1, 0],
                         lineWidth=5)
    if len(contacts_link2) > 0:
      contact = contacts_link2[0]
      pcontact = np.asarray(contact[5])
      n = -1 * np.asarray(contact[7])
      mag = contact[9]
      scaling_factor = 1
      p.addUserDebugLine(list(pcontact), list(pcontact + mag * scaling_factor * n), lifeTime=0.5,
                         lineColorRGB=[0, 0, 1],
                         lineWidth=5)


  def step(self):
    t = time.time()
    # print(t-self.prevTime)
    self.prevTime = t

    # visualize contact points
    if self.vis_bool:
      self.visContact()

    # apply control from nominal PD controller
    # tau_nom = self.armNomControl.computePD_op(self.xdes, self.kp, self.kv_j, self.kv_op, self.maxTorque)
    # self.tau_nom = np.vstack([self.tau_nom, tau_nom])
    # p.setJointMotorControl2(self.robotUid, self.ID_0, p.TORQUE_CONTROL, force=tau_nom[0])
    # p.setJointMotorControl2(self.robotUid, self.ID_1, p.TORQUE_CONTROL, force=tau_nom[1])
    # p.setJointMotorControl2(self.robotUid, self.ID_2, p.TORQUE_CONTROL, force=tau_nom[2])


    # apply control from safe controller
    tau_nom = self.armNomControl.computePD_op(self.xdes, self.kp, self.kv_j, self.kv_op, self.maxTorque)
    tau_safe = self.armSafeControl.computeSafeControl(self.maxTorque, tau_nom)
    if tau_safe is None:
      tau_safe = tau_nom
    p.setJointMotorControl2(self.robotUid, self.ID_0, p.TORQUE_CONTROL, force=tau_safe[0])
    p.setJointMotorControl2(self.robotUid, self.ID_1, p.TORQUE_CONTROL, force=tau_safe[1])
    p.setJointMotorControl2(self.robotUid, self.ID_2, p.TORQUE_CONTROL, force=tau_safe[2])

    jointIDs = [self.ID_0, self.ID_1, self.ID_2]
    n = len(jointIDs)
    q = np.zeros(n)
    qdot = np.zeros(n)
    for i in range(n):
      jointState = p.getJointState(self.robotUid, jointIDs[i])
      q[i] = jointState[0]
      qdot[i] = jointState[1]


if __name__ == "__main__":
  p.connect(p.GUI)
  urdfRoot=pybullet_data.getDataPath()
  p.loadURDF(os.path.join(urdfRoot, "plane.urdf"), 0, 0, 0)

  p.setGravity(0, 0, -9.81)
  h = RRRManipulator(initBasePos=np.array([0, 0, 0.015]),
              initPos=np.array([0., 0., 0.]))

  sim_count = 0
  init_pos = h.getWorldEE()
  while (1):
    sim_count += 1
    time.sleep(h.timeStep)
    if sim_count > 3000:
      t = time.time()
    h.step()
    p.stepSimulation()



