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

#ifndef CUDAPLANESWEEP_H
#define CUDAPLANESWEEP_H

#include "psl_base/cameraMatrix.h"
#include "psl_base/grid.h"
#include "psl_cudaBase/deviceBuffer.cuh"
#include "psl_cudaBase/deviceImage.cuh"

#include <map>
#include <vector>

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

namespace PSL {

struct CudaPlaneSweepImage {
  PSL_CUDA::DeviceImage devImg;
  CameraMatrix cam;
};

class CudaPlaneSweep {
 public:
  CudaPlaneSweep();
  ~CudaPlaneSweep();

  // hyperparameters
  void setMatchWindowSize(int w, int h);
  void setOutputFolder(const std::string &outputFolder);

  // output
  void enableCostVolumeFilter(bool enable = true, int guidedFilterRadius = -1, float guidedFilterEps = -1.0f);
  void enableOutputCostVolume(bool enable = true);
  void enableOutputBest(bool enable = true);
  void enableOutputXYZMap(bool enable=true);
  void enableOutputPointCloud(bool enable = true);
  void enableFilterCostAbove(bool enable = true, float filterThreshold = 1.0f);
  void enableDebug(bool enable = true);

  // images
  int addImage(const cv::Mat &image, const CameraMatrix &cam);
  void deleteImage(int id);
  cv::Mat downloadImage(int id);

  void process(int refImgId, const std::vector<Eigen::Vector4d> &planes, const std::string &matchingCostName);

 private:
  void prepareFolders();

 private:
  std::map<int, CudaPlaneSweepImage> _images;
  int _nextId;

  int _matchWindowRadiusX;
  int _matchWindowRadiusY;

  // enabled features
  bool _costVolumeFilterEnabled;
  int _guidedFilterRadius;
  float _guidedFilterEps;

  bool _outputCostVolumeEnabled;
  bool _debugEnabled;
  bool _outputBestEnabled;
  bool _filterEnabled;
  float _filterThreshold;
  bool _outputPointCloudEnabled;
  bool _outputXYZMapEnabled;
  // for output
  std::string _outputFolder;
};

}    // namespace PSL


#endif // CUDAPLANESWEEP_H
