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

    #
    self.tau_nom = np.zeros(3)
    self.tau_safe = np.zeros(3)
    self.tau_change = []
    self.t = [0]
    self.xee = np.zeros(3)
    self.sum_force = [0]
    self.vel_contact = np.zeros(3)

    # Load URDF model
    self.robotUid = p.loadURDF("../RRR_manipulator_description/RRR.urdf")

    # load some obstacles
    self.obs_ID = p.loadURDF("../obstacle_description/obstacle_fixed.urdf", -0.24, 0.02, 0.02) # good for fixed case
    # self.obs_ID = p.loadURDF("../obstacle_description/obstacle_1Dsliding.urdf", -200, 0.02, 0.02)  #
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
    p.resetJointState(self.robotUid, self.ID_0, np.pi / 9)
    p.resetJointState(self.robotUid, self.ID_1, -np.pi / 9)
    p.resetJointState(self.robotUid, self.ID_2, np.pi / 9)

    # p.resetJointState(self.robotUid, self.ID_0, 0)
    # p.resetJointState(self.robotUid, self.ID_1, 0)
    # p.resetJointState(self.robotUid, self.ID_2, 0)
    #
    # p.resetJointState(self.robotUid, self.ID_0, 0.99 * np.pi / 2)
    # p.resetJointState(self.robotUid, self.ID_1, -0.99 * np.pi)
    # p.resetJointState(self.robotUid, self.ID_2, 0.99 * np.pi)


  def setupControls(self):
    """ Sets up tentacleBot controller """
    # create nominal controller instance
    jointIDs = [self.ID_0, self.ID_1, self.ID_2]
    self.armNomControl = NominalController(p, self.robotUid, jointIDs, self.ID_EE)

    # create safe controller instance
    self.armSafeControl = SafeController(p, self.robotUid, jointIDs, self.ID_0, self.ID_1, self.ID_2)

    # set default controller parameters
    self.xdes = np.array([-0.25, -0.15, 0])
    # self.xdes = np.array([-0.25, -0.05, 0])
    # self.xdes = np.array([0, 1, 0])
    p.addUserDebugLine(list(self.xdes), [self.xdes[0], self.xdes[1], 1]) # show goal location
    self.maxTorque = 100.0
    self.kp = 30
    self.kv_j = .15
    self.kv_op = 5

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
    # self.visContact()

    # # # apply control from nominal PD controller
    # tau_nom = self.armNomControl.computePD_joints(np.array([np.pi/2, 0, 0]), .1, 0.1, self.maxTorque)
    # tau_nom = self.armNomControl.computePD_op(self.xdes, self.kp, self.kv_j, self.kv_op, self.maxTorque)
    # self.tau_nom = np.vstack([self.tau_nom, tau_nom])
    # p.setJointMotorControl2(self.robotUid, self.ID_0, p.TORQUE_CONTROL, force=tau_nom[0])
    # p.setJointMotorControl2(self.robotUid, self.ID_1, p.TORQUE_CONTROL, force=tau_nom[1])
    # p.setJointMotorControl2(self.robotUid, self.ID_2, p.TORQUE_CONTROL, force=tau_nom[2])


    # apply control from safe controller
    tau_nom = self.armNomControl.computePD_op(self.xdes, self.kp, self.kv_j, self.kv_op, self.maxTorque)
    self.tau_nom = np.vstack([self.tau_nom, tau_nom])
    tau_safe = self.armSafeControl.computeSafeControl(self.maxTorque, tau_nom)
    self.tau_safe = np.vstack([self.tau_safe, tau_safe])
    self.tau_change.append(np.linalg.norm(tau_safe - tau_nom))
    self.t.append(t)
    # tau_safe = self.armSafeControl.computeNCControl(tau_nom, self.obs_ID)
    if tau_safe is None:
      tau_safe = tau_nom
    p.setJointMotorControl2(self.robotUid, self.ID_0, p.TORQUE_CONTROL, force=tau_safe[0])
    p.setJointMotorControl2(self.robotUid, self.ID_1, p.TORQUE_CONTROL, force=tau_safe[1])
    p.setJointMotorControl2(self.robotUid, self.ID_2, p.TORQUE_CONTROL, force=tau_safe[2])

    ee_state = p.getLinkState(self.robotUid, self.ID_EE, computeLinkVelocity=False)
    xee = np.asarray(ee_state[0])
    self.xee = np.vstack([self.xee, xee])

    jointIDs = [self.ID_0, self.ID_1, self.ID_2]
    n = len(jointIDs)
    q = np.zeros(n)
    qdot = np.zeros(n)
    for i in range(n):
      jointState = p.getJointState(self.robotUid, jointIDs[i])
      q[i] = jointState[0]
      qdot[i] = jointState[1]
    # print(p.calculateMassMatrix(self.robotUid, list(q)))
    # print(p.calculateInverseDynamics(self.robotUid, list(q), [1,1,1], [0, 0, 0])) # this is coriolis, centrifugal + gravity
    # print(p.calculateJacobian(self.robotUid, self.ID_2, [0, 0, 0], list(q), list(qdot), [0,0,0])[0])


    # obstacleState = self._pb.getLinkState(obstacleID, 0)


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

  # np.save('sum_force_safe_failure', h.sum_force)
  # plt.rcParams.update({'font.size': 22})
  # fig, axs = plt.subplots(3)
  # fig.suptitle('Nominal and Safe Control Inputs')
  # axs[0].plot(h.tau_nom[:, 0], 'r--', label='nominal')
  # axs[0].plot(h.tau_safe[:, 0], 'c', label='safe')
  # axs[0].set( ylabel='tau proximal')
  # axs[0].legend(loc='upper right')
  #
  # axs[1].plot(h.tau_nom[:, 1], 'r--', label='nominal')
  # axs[1].plot(h.tau_safe[:, 1], 'c', label='safe')
  # axs[1].set( ylabel='tau medial')
  #
  # axs[2].plot(h.tau_nom[:, 2], 'r--', label='nominal')
  # axs[2].plot(h.tau_safe[:, 2], 'c', label='safe')
  # axs[2].set(xlabel='simulation count', ylabel='tau distal')
  # plt.show()


