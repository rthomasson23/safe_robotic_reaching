import numpy as np
from scipy.spatial.transform import Rotation as R
from CBF_RRR_lineBarrier import *
# from CBF_RRR_NCBarrier import *
import cvxpy as cp
# import quadprog as qp

class SafeController(object):

    def __init__(self, pb, bodyUniqueID, jointIDs, ID_0, ID_1, ID_2, maxTorque):
        self._pb = pb
        self.bodyID = bodyUniqueID
        self.jointIDs = jointIDs
        self.ID_0 = ID_0
        self.ID_1 = ID_1
        self.ID_2 = ID_2

        self.maxTorque = maxTorque
        # we need q, lx, ly, pcx, pcy, nx, ny

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
            nx, ny, _ = (np.asarray(contact[7]))/np.linalg.norm(np.asarray(contact[7]))
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
            nx, ny, _ = (np.asarray(contact[7])) / np.linalg.norm(np.asarray(contact[7]))
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
            nx, ny, _ = (np.asarray(contact[7])) / np.linalg.norm(np.asarray(contact[7]))
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

        force = np.clip(tau_safe, -self.maxTorque, self.maxTorque)
        return tau_safe

    def transform_world2link(self, pcontact, link_ID):
        pcontact = np.array([pcontact[0], pcontact[1], 0])
        link_state = self._pb.getLinkState(self.bodyID, link_ID)
        Pworld2link = np.asarray(link_state[4])
        quat = np.asarray(link_state[5])
        Rworld2link = (R.from_quat(quat)).as_matrix()

        xlink = Rworld2link[:,0]
        ylink = Rworld2link[:,1]
        lx = np.dot(xlink, (pcontact - Pworld2link))
        ly = np.dot(ylink, (pcontact - Pworld2link))
        plink = [lx, ly, 0]

        return plink





