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

#ifndef CAMERAMATRIX_H
#define CAMERAMATRIX_H

#include <Eigen/Dense>

namespace PSL {

class CameraMatrix {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CameraMatrix() {}
  CameraMatrix(const Eigen::Matrix<double, 3, 3> &K, const Eigen::Matrix<double, 3, 3> &R, const Eigen::Vector3d &T);

  void setKRT(const Eigen::Matrix<double, 3, 3> &K, const Eigen::Matrix<double, 3, 3> &R, const Eigen::Vector3d &T);

  // copy consturctor and assignment
  CameraMatrix(const CameraMatrix &otherCameraMatrix);
  CameraMatrix &operator=(const CameraMatrix &otherCameraMatrix);

  const Eigen::Matrix<double, 3, 3> &getK() const;
  const Eigen::Matrix<double, 3, 3> &getR() const;
  const Eigen::Vector3d &getT() const;
  Eigen::Vector3d getC() const;
  Eigen::Matrix<double, 3, 4> getP() const;
  void getFloatP(float mat[12]) const;

  void print() const;

  // fx, fy, cx, cy, s, quaternion vector, translation vector
  std::string toString() const;

  // conventional way of unprojection
  Eigen::Vector3d unprojectPoint(double x, double y, double depth) const;

  // plane is in scene coordinate frame
  double getPointDepth(double x, double y, const Eigen::Vector4d &plane) const;
  Eigen::Vector3d unprojectPoint(double x, double y, const Eigen::Vector4d &plane) const;

 private:
  Eigen::Matrix<double, 3, 3> _K;
  Eigen::Matrix<double, 3, 3> _R;
  Eigen::Vector3d _T;
};

// check whether a plane is in front of a camera
bool checkPlane(const CameraMatrix &camera, const Eigen::Vector4d &plane);

void planeHomography(const CameraMatrix &refCam, const CameraMatrix &srcCam,
                     const Eigen::Vector4d &plane, float *H);

} // namespace PSL


#endif // CAMERAMATRIX_H
