// ===============================================================================================================
// Copyright (c) 2019, Cornell University. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright otice, this list of conditions and
//       the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
//       the following disclaimer in the documentation and/or other materials provided with the distribution.
//
//     * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
//       promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.
//
// Author: Kai Zhang (kz298@cornell.edu)
//
// The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
// Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
// The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
// ===============================================================================================================

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
