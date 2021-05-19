import numpy as np
from scipy.spatial.transform import Rotation as R
from CBF_RRR_lineBarrier import *
# from CBF_RRR_NCBarrier import *
import cvxpy as cp
# import quadprog as qp



class SafeController(object):

    def __init__(self, pb, bodyUniqueID, jointIDs, ID_0, ID_1, ID_2):
        self._pb = pb
        self.bodyID = bodyUniqueID
        self.jointIDs = jointIDs
        self.ID_0 = ID_0
        self.ID_1 = ID_1
        self.ID_2 = ID_2
        # we need q, lx, ly, pcx, pcy, nx, ny


    def computeNCControl(self, tau_nom, obstacleID):
        n = len(self.jointIDs)
        q = np.zeros(n)
        qdot = np.zeros(n)
        for i in range(n):
            jointState = self._pb.getJointState(self.bodyID, self.jointIDs[i])
            q[i] = jointState[0]
            qdot[i] = jointState[1]

        q = np.hstack([q, qdot])  # our state includes positions and velocities

        obstacleState = self._pb.getLinkState(obstacleID, 0)
        if obstacleState is None:
            print('obstacle state is None')
            return tau_nom
        obs_x, obs_y, _ = obstacleState[0]
        # self._pb.addUserDebugLine([obs_x, obs_y, 0], [obs_x, obs_y, 1], lifeTime=0.2)

        LgB_array = None
        LfBpB_array = None

        D = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
        for d in D:
            hp_d = hp(d, q, L, obs_x, obs_y)
            # print(hp_d)
            hm_d = hm(d, q, L, obs_x, obs_y)
            # print(hm_d)
            hd_d = hd(d, q, L, obs_x, obs_y)
            # print("d")
            # print(d)
            # print(hd_d)
            # print(q)
            # print(obs_x)
            # print(obs_y)

            LfB_p = LfB_proximal(q, L, m, Ixx, Iyy, Izz, z, d, obs_x, obs_y, r)
            LfB_m = LfB_medial(q, L, m, Ixx, Iyy, Izz, z, d, obs_x, obs_y, r)
            LfB_d = LfB_distal(q, L, m, Ixx, Iyy, Izz, z, d, obs_x, obs_y, r)

            LgB_p = LgB_proximal(q, L, m, Ixx, Iyy, Izz, z, d, obs_x, obs_y, r)
            LgB_m = LgB_medial(q, L, m, Ixx, Iyy, Izz, z, d, obs_x, obs_y, r)
            LgB_d = LgB_distal(q, L, m, Ixx, Iyy, Izz, z, d, obs_x, obs_y, r)

            B_p = B_proximal(q, L, m, Ixx, Iyy, Izz, z, d, obs_x, obs_y, r)
            B_m = B_medial(q, L, m, Ixx, Iyy, Izz, z, d, obs_x, obs_y, r)
            B_d = B_distal(q, L, m, Ixx, Iyy, Izz, z, d, obs_x, obs_y, r)

            safety_proximal = LfB_p + np.dot(LgB_p, tau_nom) + B_p
            safety_medial = LfB_m + np.dot(LgB_m, tau_nom) + B_m
            safety_distal = LfB_d + np.dot(LgB_d, tau_nom) + B_d
            if (safety_proximal < 0) or (safety_medial < 0) or (safety_distal < 0):

        # # some visualization stuffs
        # pdx, pdy = pd(d, q, L)
        # self._pb.addUserDebugLine([pdx, pdy, 0], [pdx, pdy, 1], lifeTime=0.2)
        # pmx, pmy = pm(d, q, L)
        # self._pb.addUserDebugLine([pmx, pmy, 0], [pmx, pmy, 1], lifeTime=0.2)
        # ppx, ppy = pp(d, q, L)
        # self._pb.addUserDebugLine([ppx, ppy, 0], [ppx, ppy, 1], lifeTime=0.2)

                if LgB_array is None:
                    LgB_array = np.vstack([np.asarray(LgB_p), np.asarray(LgB_m), np.asarray(LgB_d)])
                    LfBpB_array = np.hstack([np.asarray(LfB_p + B_p), np.asarray(LfB_m + B_m), np.asarray(
                        LfB_d + B_d)])
                else:
                    LgB_curr = np.vstack([np.asarray(LgB_p), np.asarray(LgB_m), np.asarray(LgB_d)])
                    LfBpB_curr = np.hstack([np.asarray(LfB_p + B_p), np.asarray(LfB_m + B_m), np.asarray(
                            LfB_d + B_d)])
                    np.hstack([LgB_array, LgB_curr])
                    np.hstack([LfBpB_array, LfBpB_curr])

        if LgB_array is not None:
            H = 2 * np.eye(3)
            f = -2 * np.asarray(tau_nom)
            tau = cp.Variable(n)
            prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(tau, H) + f.T @ tau),
                              [-LgB_array @ tau <= LfBpB_array])
            prob.solve()
            tau_safe = tau.value
        else:
            tau_safe = tau_nom
        print(np.linalg.norm(tau_safe - tau_nom))
        return tau_safe

    def computeSafeControl(self, maxForce, tau_nom):
        LgB_array = None
        LfBpB_array = None

        n = len(self.jointIDs)
        q = np.zeros(n)
        qdot = np.zeros(n)
        for i in range(n):
            jointState = self._pb.getJointState(self.bodyID, self.jointIDs[i])
            q[i] = jointState[0]
            qdot[i] = jointState[1]

        q = np.hstack([q, qdot]) # our state includes positions and velocities

        contacts_link0 = self._pb.getContactPoints(bodyA=self.bodyID, linkIndexA=self.ID_0)
        contacts_link1 = self._pb.getContactPoints(bodyA=self.bodyID, linkIndexA=self.ID_1)
        contacts_link2 = self._pb.getContactPoints(bodyA=self.bodyID, linkIndexA=self.ID_2)

        inContact = False

        if len(contacts_link0) > 0:
            inContact = True

            contact = contacts_link0[0]
            nx, ny, _ = np.asarray(contact[7])
            pcontact = np.asarray(contact[5])
            pcx = pcontact[0]
            pcy = pcontact[1]
            lx, ly, _ = self.transform_world2link(pcontact, self.ID_0)
            lx = np.array([lx, 0, 0])
            ly = np.array([ly, 0, 0])

            LgB = LgB_proximal(q, L, l, m, Ixx, Iyy, Izz, z, lx, ly, nx, ny, pcx, pcy)
            LfB = LfB_proximal(q, L, l, m, Ixx, Iyy, Izz, z, lx, ly, nx, ny, pcx, pcy)
            B = B_proximal(q, L, l, m, Ixx, Iyy, Izz, z, lx, ly, nx, ny, pcx, pcy)
            if LgB_array is None:
                LgB_array = np.asarray(LgB)
                LfBpB_array = np.asarray(LfB + B)
            else:
                np.hstack([LgB_array, LgB])
                np.hstack([LfBpB_array, LfB + B])

        if len(contacts_link1) > 0:
            inContact = True

            contact = contacts_link1[0]
            nx, ny, _ = np.asarray(contact[7])
            pcontact = np.asarray(contact[5])
            pcx = pcontact[0]
            pcy = pcontact[1]
            lx, ly, _ = self.transform_world2link(pcontact, self.ID_1)
            lx = np.array([0, lx, 0])
            ly = np.array([0, ly, 0])

            LgB = LgB_medial(q, L, l, m, Ixx, Iyy, Izz, z, lx, ly, nx, ny, pcx, pcy)
            LfB = LfB_medial(q, L, l, m, Ixx, Iyy, Izz, z, lx, ly, nx, ny, pcx, pcy)
            B = B_medial(q, L, l, m, Ixx, Iyy, Izz, z, lx, ly, nx, ny, pcx, pcy)
            if LgB_array is None:
                LgB_array = np.asarray(LgB)
                LfBpB_array = np.asarray(LfB + B)
            else:
                np.hstack([LgB_array, LgB])
                np.hstack([LfBpB_array, LfB + B])

        if len(contacts_link2) > 0:
            inContact = True

            contact = contacts_link2[0]
            nx, ny, _ = np.asarray(contact[7])
            pcontact = np.asarray(contact[5])
            pcx = pcontact[0]
            pcy = pcontact[1]

            lx, ly, _ = self.transform_world2link(pcontact, self.ID_2)
            lx = np.array([0, 0, lx])
            ly = np.array([0, 0, ly])

            LgB = LgB_distal(q, L, l, m, Ixx, Iyy, Izz, z, lx, ly, nx, ny, pcx, pcy)
            LfB = LfB_distal(q, L, l, m, Ixx, Iyy, Izz, z, lx, ly, nx, ny, pcx, pcy)
            B = B_distal(q, L, l, m, Ixx, Iyy, Izz, z, lx, ly, nx, ny, pcx, pcy)
            if LgB_array is None:
                LgB_array = np.asarray(LgB)
                LfBpB_array = np.asarray(LfB + B)
            else:
                np.hstack([LgB_array, LgB])
                np.hstack([LfBpB_array, LfB + B])

        # solve a QP to find the safe controller
        if inContact:
            H = 2*np.eye(3)
            f = -2 * np.asarray(tau_nom)
            tau = cp.Variable(n)
            prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(tau, H) + f.T @ tau), [-LgB_array @ tau <= LfBpB_array])
            prob.solve()
            tau_safe = tau.value
        else:
            tau_safe = tau_nom

        return tau_safe

    def transform_world2link(self, pcontact, link_ID):
        pcontact = np.array([pcontact[0], pcontact[1], 0])
        link_state = self._pb.getLinkState(self.bodyID, link_ID)
        Pworld2link = np.asarray(link_state[4])
        quat = np.asarray(link_state[5])
        Rworld2link = (R.from_quat(quat)).as_matrix()
        # Tworld2link = np.zeros([4, 4])
        # Tworld2link[0:3, 0:3] = Rworld2link
        # Tworld2link[:, 3] = np.hstack([Pworld2link, 1])
        # plink = np.dot(Tworld2link, np.hstack([pcontact, 1]))[0:3]
        # plink = [np.sign(plink[0]) * 0.01750000, plink[1], 0]

        xlink = Rworld2link[:,0]
        ylink = Rworld2link[:,1]
        lx = np.dot(xlink, (pcontact - Pworld2link))
        ly = np.dot(ylink, (pcontact - Pworld2link))
        plink = [lx, ly, 0]
        print(plink)
        # plink[1] -= l[link_ID - 1]
        # self._pb.addUserDebugLine(list(Rworld2link[:,1]), [0,0,0], lifeTime=0.5, lineColorRGB=[0.5, 0.5, 0])
        # self._pb.addUserDebugLine(list(pcontact), list(pcontact + [0,0,1]), lifeTime=0.5, lineColorRGB=[0.5, 0.5, 0])
        # self._pb.addUserDebugLine(list(pcontact+[0,0,0]), list(pcontact+[1,0,0]), parentObjectUniqueId=self.bodyID,
        #                           parentLinkIndex=link_ID, lifeTime=0.5, lineColorRGB=[0.5, 0.5, 0.5])
        # self._pb.addUserDebugLine(list(plink_vis1), list(plink_vis2), parentObjectUniqueId=self.bodyID,
                                  # parentLinkIndex=link_ID, lifeTime=0.5, lineColorRGB=[0.5, 0.5, 0.5])

        return plink





