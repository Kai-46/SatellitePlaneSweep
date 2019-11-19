import numpy as np
from pyquaternion import Quaternion


class CameraModel(object):
    def __init__(self, params):
        # w, h, fx, fy, cx, cy, s, qvec, tvec
        assert (len(params) == 14)
        self.w, self.h = params[0:2]
        fx, fy, cx, cy, s = params[2:7]
        qvec = params[7:11]
        tvec = params[11:14]

        self.K = np.array([[fx, s, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]])
        self.K_inv = np.linalg.inv(self.K)
        self.R = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
        self.t = np.array(tvec).reshape((3, 1))

    # plane is a 4 by 1 vector
    def unproject(self, x, y, plane):
        point = np.array([x, y, 1.0]).reshape((3, 1))
        point = np.dot(self.K_inv, point)

        plane_normal = np.dot(self.R, plane[0:3, 0])
        plane_constant = np.dot(plane_normal.T, self.t) + plane[3, 0]

        depth = plane_constant / np.dot(plane_normal.T, point)
        point *= depth
        point = np.dot(self.R.T, point - self.t)

        return point

    def project(self, x, y, z):
        point = np.array([x, y, z]).reshape((3, 1))
        tmp = np.dot(self.K, np.dot(self.R, point) + self.t)

        return tmp[0, 0] / tmp[2, 0], tmp[1, 0] / tmp[2, 0]
