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
