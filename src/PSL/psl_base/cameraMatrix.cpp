// This file is part of PlaneSweepLib (PSL)

// Copyright 2016 Christian Haene (ETH Zuerich)

// PSL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// PSL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with PSL.  If not, see <http://www.gnu.org/licenses/>.


#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "cameraMatrix.h"

namespace PSL {

CameraMatrix::CameraMatrix(const Eigen::Matrix<double, 3, 3> &K,
                           const Eigen::Matrix<double, 3, 3> &R,
                           const Eigen::Vector3d &T) {
  setKRT(K, R, T);
}

void CameraMatrix::setKRT(const Eigen::Matrix<double, 3, 3> &K, const Eigen::Matrix<double, 3, 3> &R,
                          const Eigen::Vector3d &T) {
  _K = K;
  _R = R;
  _T = T;

  _K /= _K(2, 2);
}

CameraMatrix::CameraMatrix(const CameraMatrix &otherCameraMatrix) {
  _K = otherCameraMatrix._K;
  _R = otherCameraMatrix._R;
  _T = otherCameraMatrix._T;
}

CameraMatrix &CameraMatrix::operator=(const CameraMatrix &otherCameraMatrix) {
  _K = otherCameraMatrix._K;
  _R = otherCameraMatrix._R;
  _T = otherCameraMatrix._T;

  return *this;
}

const Eigen::Matrix<double, 3, 3> &CameraMatrix::getK() const {
  return _K;
}

const Eigen::Matrix<double, 3, 3> &CameraMatrix::getR() const {
  return _R;
}

const Eigen::Vector3d &CameraMatrix::getT() const {
  return _T;
}

Eigen::Vector3d CameraMatrix::getC() const {
  Eigen::Matrix<double, 3, 1> C = -_R.transpose() * _T;
  return C;
}

Eigen::Matrix<double, 3, 4> CameraMatrix::getP() const {
  Eigen::Matrix<double, 3, 4> P;

  P.leftCols(3) = _R;
  P.rightCols(1) = _T;
  P = _K * P;

  // for numerical stability
  P /= P.cwiseAbs().maxCoeff();

  return P;
}

void CameraMatrix::getFloatP(float mat[12]) const {
  Eigen::Matrix<double, 3, 4> P = getP();

  // important: memory layout of eigen matrix might not be continous
  for (int i=0; i<3; ++i) {
    for (int j=0; j<4; ++j) {
      mat[i * 4 + j] = float(P(i, j));
    }
  }
}

// fx, fy, cx, cy, s, quaternion vector, translation vector
std::string CameraMatrix::toString() const {
  std::ostringstream oss;
  oss << std::setprecision(17);
  oss << _K(0, 0) << ", " << _K(1, 1) << ", " << _K(0, 2) << ", " << _K(1, 2) << ", " << _K(0, 1) << ", ";

  Eigen::Quaternion<double> qvec(_R);
  oss << qvec.w() << ", " << qvec.x() << ", " << qvec.y() << ", " << qvec.z() << ", ";

  oss << _T(0) << ", " << _T(1) << ", " << _T(2);

  return oss.str();
}

Eigen::Vector3d CameraMatrix::unprojectPoint(double x, double y, double depth) const {
  Eigen::Vector3d point;

  assert (abs(_K(2, 2) - 1.0) < 1e-4);

  point(0) = (x - _K(0, 2)) / _K(0, 0) * depth;
  point(1) = (y - _K(1, 2)) / _K(1, 1) * depth;
  point(2) = depth;

  point = _R.transpose() * (point - _T);

  // point is in scene coordinate frame
  return point;
}

void CameraMatrix::print() const {
  Eigen::IOFormat cleanFmt(4, 0, ", ", "\n", "[", "]");
  std::cout << "K: " << _K.format(cleanFmt) << std::endl;
  std::cout << "R: " << _R.format(cleanFmt) << std::endl;
  std::cout << "T: " << _T.format(cleanFmt) << std::endl;
}

Eigen::Vector3d CameraMatrix::unprojectPoint(double x, double y, const Eigen::Vector4d &plane) const {
  // first convert the plane from scene coordinate frame to camera coordinate frame
  // take the intersection of the camera ray and plane
  // convert the intersection back to scene coordinate frame

  // plane is n^T x - d=0
  // light ray in camera coordinate frame
  Eigen::Vector3d point;
  point << x, y, 1.0;
  point = _K.inverse() * point;

  // plane equation in camera coordinate frame
  Eigen::Vector3d plane_normal = _R * plane.head(3);
  double plane_constant = plane_normal.transpose() * _T + plane(3);

  double depth = plane_constant / (plane_normal.transpose() * point);
  point *= depth;
  point = _R.transpose() * (point - _T);

  // point is in scene coordinate frame
  return point;
}

double CameraMatrix::getPointDepth(double x, double y, const Eigen::Vector4d &plane) const {
  // first convert the plane from scene coordinate frame to camera coordinate frame
  // take the intersection of the camera ray and plane
  // convert the intersection back to scene coordinate frame

  // plane is n^T x - d=0
  // light ray in camera coordinate frame
  Eigen::Vector3d point;
  point(0) = (x - _K(0, 2)) / _K(0, 0);
  point(1) = (y - _K(1, 2)) / _K(1, 1);
  point(2) = 1.0;

  // plane equation in camera coordinate frame
  Eigen::Vector3d plane_normal = _R * plane.head(3);
  double plane_constant = plane_normal.transpose() * _T + plane(3);

  double depth = plane_constant / (plane_normal.transpose() * point);

  return depth;
}

bool checkPlane(const CameraMatrix &camera, const Eigen::Vector4d &plane) {
  // assume plane is parametrized as n^Tx - d=0
  // plane equation in camera coordinate frame
  Eigen::Vector3d plane_normal = camera.getR() * plane.head(3);
  double plane_constant = plane_normal.transpose() * camera.getT() + plane(3);

  // make sure normal is pointing away from camera center
  if (plane_normal(2) < 0.0) {
    plane_normal *= -1.0;
    plane_constant *= -1.0;
  }

  if (plane_constant > 0) {
    return true;
  } else {
    return false;
  }
}

void planeHomography(const CameraMatrix &refCam, const CameraMatrix &srcCam,
                     const Eigen::Vector4d &plane, float *H) {
  // assume plane is parametrized as n^Tx - d=0, with n pointing away from the origin
  // suppose plane equation is in the scene coordinate frame
  Eigen::Vector3d n = plane.head(3);
  double d = plane(3);

  Eigen::Matrix<double, 3, 4> refP = refCam.getP();
  Eigen::Matrix<double, 3, 4> srcP = srcCam.getP();

  // compute homography directly from two projection matrices
  // first rewrite x=P[X; 1] as x=SX
  Eigen::Matrix<double, 3, 3> refS = (refP.leftCols(3) + refP.rightCols(1) * n.transpose() / d);
  Eigen::Matrix<double, 3, 3> srcS = (srcP.leftCols(3) + srcP.rightCols(1) * n.transpose() / d);

  // compose homography
  Eigen::Matrix<double, 3, 3> hMat = srcS * refS.inverse();

  // for numeric stability
  hMat /= hMat.maxCoeff();

  // consumer gpu only supports single precision
  H[0] = (float) hMat(0, 0);
  H[1] = (float) hMat(0, 1);
  H[2] = (float) hMat(0, 2);
  H[3] = (float) hMat(1, 0);
  H[4] = (float) hMat(1, 1);
  H[5] = (float) hMat(1, 2);
  H[6] = (float) hMat(2, 0);
  H[7] = (float) hMat(2, 1);
  H[8] = (float) hMat(2, 2);
}

}  // namespace PSL
