// ===============================================================================================================
// Copyright (c) 2019, Cornell University. All rights reserved.
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General 
// Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied 
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with this program. If not, see 
// <https://www.gnu.org/licenses/>.
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



#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>

#include "psl_base/exception.h"
#include "psl_cudaBase/cudaBackbone.cuh"
#include "psl_stereo/cudaPlaneSweep.h"
#include "psl_base/ioTools.h"
#include "psl_base/gridVisualize.h"
#include "psl_base/ply.h"

namespace PSL {

CudaPlaneSweep::CudaPlaneSweep() {
  _nextId = 0;

  _matchWindowRadiusX = -1;        // set invalid default values
  _matchWindowRadiusY = -1;

  _costVolumeFilterEnabled = false;
  _guidedFilterRadius = -1;
  _guidedFilterEps = -1.0f;

  _outputCostVolumeEnabled = false;
  _outputBestEnabled = false;
  _debugEnabled = false;

  _filterEnabled = false;
  _filterThreshold = 1.0f;
  _outputPointCloudEnabled = false;

  _outputXYZMapEnabled = false;
}

CudaPlaneSweep::~CudaPlaneSweep() {
  // delete all the images from the gpu
  for (auto it = _images.begin(); it != _images.end(); it++) {
    it->second.devImg.deallocate();
  }
}

void CudaPlaneSweep::setMatchWindowSize(int w, int h) {
  if (w % 2 != 1 || h % 2 != 1) {
    PSL_THROW_EXCEPTION("window size must be odd")
  }
  _matchWindowRadiusX = w / 2;
  _matchWindowRadiusY = h / 2;
}

void CudaPlaneSweep::setOutputFolder(const std::string &outputFolder) {
  _outputFolder = outputFolder;
  if (_outputFolder.back() != '/') {
    _outputFolder += '/';
  }
}

void CudaPlaneSweep::enableCostVolumeFilter(bool enable, int guidedFilterRadius, float guidedFilterEps) {
  _costVolumeFilterEnabled = enable;
  _guidedFilterRadius = guidedFilterRadius;
  _guidedFilterEps = guidedFilterEps;
}

void CudaPlaneSweep::enableOutputCostVolume(bool enable) {
  _outputCostVolumeEnabled = enable;
}

void CudaPlaneSweep::enableOutputBest(bool enable) {
  _outputBestEnabled = enable;
}

void CudaPlaneSweep::enableFilterCostAbove(bool enable, float filterThreshold) {
  _filterEnabled = enable;
  _filterThreshold = filterThreshold;
}

void CudaPlaneSweep::enableOutputXYZMap(bool enable) {
  _outputXYZMapEnabled = enable;
}

void CudaPlaneSweep::enableOutputPointCloud(bool enable) {
  _outputPointCloudEnabled = enable;
}

void CudaPlaneSweep::enableDebug(bool enable) {
  _debugEnabled = enable;
}

int CudaPlaneSweep::addImage(const cv::Mat &image, const CameraMatrix &cam) {
  int id = _nextId;
  _images.emplace(id, CudaPlaneSweepImage{});
  CudaPlaneSweepImage &cPSI = _images[id];
  cPSI.cam = cam;        // copy of cam object

  if (image.channels() == 3) {
    // convert to grayscale
    cv::Mat corrImg;
    cvtColor(image, corrImg, CV_BGR2GRAY);
    cPSI.devImg.allocatePitchedAndUpload(corrImg);
  } else if (image.channels() == 1) {
    cPSI.devImg.allocatePitchedAndUpload(image);
  } else {
    PSL_THROW_EXCEPTION("Image format not supported for current color mode.")
  }

  _nextId += 1;
  return id;
}

void CudaPlaneSweep::deleteImage(int id) {
  if (_images.count(id) == 1) {
    CudaPlaneSweepImage &img = _images[id];
    // delete image on the device
    img.devImg.deallocate();
    // delete image form the map
    _images.erase(id);
  } else {
    std::stringstream strstream;
    strstream << "Image with ID " << id << " does not exist.";
    PSL_THROW_EXCEPTION(strstream.str().c_str());
  }
}

cv::Mat CudaPlaneSweep::downloadImage(int id) {
  if (_images.count(id) != 1) {
    std::stringstream errorMsg;
    errorMsg << "Cannot download image with ID " << id << ". ID invalid.";
    PSL_THROW_EXCEPTION(errorMsg.str().c_str());
  }

  return _images[id].devImg.download();
}

void CudaPlaneSweep::prepareFolders() {
  makeOutputFolder(_outputFolder);
  makeOutputFolder(_outputFolder + "costVolume");
  // debug
  if (_debugEnabled) {
    makeOutputFolder(_outputFolder + "debug");
    makeOutputFolder(_outputFolder + "debug/costVolume");
    makeOutputFolder(_outputFolder + "debug/compositeImage");
  }
  // best
  if (_outputBestEnabled) {
    makeOutputFolder(_outputFolder + "best");
  }
}

void CudaPlaneSweep::process(int refImgId,
                             const std::vector<Eigen::Vector4d> &planes,
                             const std::string &matchingCostName) {
  // check if refImgId has been added
  if (_images.count(refImgId) != 1) {
    std::stringstream strstream;
    strstream << "Image with ID " << refImgId << " does not exist.";
    PSL_THROW_EXCEPTION(strstream.str().c_str());
  }

  PSL_CUDA::MATCHING_COST_TYPE costType;
  if (matchingCostName == "ncc") {
    costType = PSL_CUDA::NCC_MATCHING_COST;
  } else if (matchingCostName == "census") {
    costType = PSL_CUDA::CENSUS_MATCHING_COST;
  } else {
    PSL_THROW_EXCEPTION("unsupported matching cost");
  }

  prepareFolders();

  const CudaPlaneSweepImage &refImg = _images[refImgId];
  const int numPlanes = planes.size();

  // create guided filter
  // guided filter is a linear filter
  cv::Ptr<cv::ximgproc::GuidedFilter> guidedFilterPtr;
  if (_costVolumeFilterEnabled) {
    if (_guidedFilterRadius <= 0 || _guidedFilterEps <= 0) {
      PSL_THROW_EXCEPTION("invalid setting of guided filter parameters");
    }

    guidedFilterPtr = cv::ximgproc::createGuidedFilter(refImg.devImg.download(), _guidedFilterRadius, _guidedFilterEps);
  }

  // check if planes are all in front of the reference camera
  for (int i = 0; i < numPlanes; ++i) {
    if (!checkPlane(refImg.cam, planes[i])) {
      std::cout << "plane idx: " << i << ", " << planes[i](3) << std::endl;
      PSL_THROW_EXCEPTION("Not all planes are in front of the reference camera");
    }
  }

  // for computing on GPU
  PSL_CUDA::DeviceBuffer<float> costAccumBuffer;    // accumulate cost from multiple views
  PSL_CUDA::DeviceBuffer<float> viewCntAccumBuffer;    // count how many effective views

  costAccumBuffer.allocatePitched(refImg.devImg.getWidth(), refImg.devImg.getHeight());
  viewCntAccumBuffer.allocatePitched(refImg.devImg.getWidth(), refImg.devImg.getHeight());

  // for outputing best
  PSL_CUDA::DeviceBuffer<float> bestPlaneBuffer;
  PSL_CUDA::DeviceBuffer<float> bestPlaneCostBuffer;

  const float absurdPlane = -1.0f;
  if (_outputBestEnabled) {
    bestPlaneBuffer.allocatePitched(refImg.devImg.getWidth(), refImg.devImg.getHeight());
    bestPlaneBuffer.clear(absurdPlane);
  }

  // const float absurdCost = 1e10f;
  const float absurdCost = 1.0f;
  if (_outputBestEnabled) {
    bestPlaneCostBuffer.allocatePitched(refImg.devImg.getWidth(), refImg.devImg.getHeight());
    bestPlaneCostBuffer.clear(absurdCost);
  }

  // for debugging
  PSL_CUDA::DeviceBuffer<float> intensityAccumBuffer;     // accumulate intensity from source views
  if (_debugEnabled) {
    intensityAccumBuffer.allocatePitched(refImg.devImg.getWidth(), refImg.devImg.getHeight());
  }

  // for storing cost slice on cpu
  Grid<float> tmpGrid;
  if (_outputCostVolumeEnabled || _debugEnabled || _costVolumeFilterEnabled) {
    // allocate space
    tmpGrid.resize(refImg.devImg.getWidth(), refImg.devImg.getHeight(), 1);
  }

  // start processing
  PSL_CUDA::initTexture();
  for (int i = 0; i < numPlanes; ++i) {
    // accumulate cost from multiple views
    std::cout << "sweeping plane " << i + 1 << "/" << numPlanes << "..." << std::endl;
    costAccumBuffer.clear(0.0f);
    viewCntAccumBuffer.clear(0.0f);

    if (_debugEnabled) {
      PSL_CUDA::copyImageToBuffer(refImg.devImg, intensityAccumBuffer);
    }

    for (auto it = _images.begin(); it != _images.end(); it++) {
      if (it->first == refImgId)
        continue;

      float hMat[9];
      planeHomography(refImg.cam, it->second.cam, planes[i], hMat);

      PSL_CUDA::costAndIntensityAccum(costType,
                                      refImg.devImg,
                                      it->second.devImg,
                                      hMat,
                                      _matchWindowRadiusX,
                                      _matchWindowRadiusY,
                                      costAccumBuffer,
                                      viewCntAccumBuffer,
                                      intensityAccumBuffer);

//			break;	// debug
    }

    // guided-filter costAccumBuffer
    if (_costVolumeFilterEnabled) {
      costAccumBuffer.download(tmpGrid.getDataPtr());

      // in-place filtering
      // tmpGrid is row-major; opencv also expects row-major
      cv::Mat tmpMat(tmpGrid.getHeight(), tmpGrid.getWidth(), CV_32F, tmpGrid.getDataPtr());
      guidedFilterPtr->filter(tmpMat, tmpMat);
      costAccumBuffer.upload(tmpGrid.getDataPtr());   // re-upload to GPU
    }

    PSL_CUDA::updateCostAndIntensityAndBestPlane(costAccumBuffer, viewCntAccumBuffer, absurdCost,
                                                 float(i), intensityAccumBuffer, bestPlaneCostBuffer, bestPlaneBuffer);

    if (_debugEnabled) {
      // save composite images
      intensityAccumBuffer.download(tmpGrid.getDataPtr());
      std::string
          outputImg = _outputFolder + "debug/compositeImage/composite_image_" + toFixedWidthString(i, 5) + ".jpg";
      tmpGrid(0, 0) = 0.0f;  // ensure [0.0, 1.0] is used
      tmpGrid(0, 1) = 1.0f;
      saveGridZSliceAsImage<float>(tmpGrid, 0, outputImg, 0.0, 1.0);
    }

    if (_outputCostVolumeEnabled) {
      costAccumBuffer.download(tmpGrid.getDataPtr());
      std::string outputFile = _outputFolder + "costVolume/cost_slice_" + toFixedWidthString(i, 5) + ".bin";
      tmpGrid.saveAsDataFile(outputFile);

      if (_debugEnabled) {
        std::string outputImg = _outputFolder + "debug/costVolume/cost_slice_" + toFixedWidthString(i, 5) + ".jpg";
        tmpGrid(0, 0) = 0.0f;
        tmpGrid(0, 1) = 1.0f;
        saveGridZSliceAsImage<float>(tmpGrid, 0, outputImg, 0.0, 1.0);
      }
    } else if (_debugEnabled) {
      costAccumBuffer.download(tmpGrid.getDataPtr());
      std::string outputImg = _outputFolder + "debug/costVolume/cost_slice_" + toFixedWidthString(i, 5) + ".jpg";
      tmpGrid(0, 0) = 0.0f;
      tmpGrid(0, 1) = 1.0f;
      saveGridZSliceAsImage<float>(tmpGrid, 0, outputImg, 0.0, 1.0);
    }
  }

  tmpGrid.freeMem();

  // output ref camera and ref image
  std::ofstream ofs;

  ofs.open(_outputFolder + "refCam.txt", std::ios::out);
  if (!ofs.is_open()) {
    PSL_THROW_EXCEPTION("failed to open file")
  }
  ofs << refImg.cam.toString() << "\n";
  ofs.close();

  // write the set of planes
  ofs.open(_outputFolder + "sweepPlanes.txt", std::ios::out);
  if (!ofs.is_open()) {
    PSL_THROW_EXCEPTION("failed to open file")
  }
  ofs << std::setprecision(17);
  for (int i = 0; i < numPlanes; ++i) {
    const Eigen::Vector4d &plane = planes[i];
    ofs << plane(0) << ", " << plane(1) << ", " << plane(2) << ", " << plane(3) << "\n";
  }
  ofs.close();

  // copy data from GPU
  if (_outputBestEnabled) {
    // filter the best plane
    PSL_CUDA::filterBestPlane(bestPlaneCostBuffer, _filterThreshold, bestPlaneBuffer, absurdPlane);

    Grid<float> bestPlanes(bestPlaneBuffer.getWidth(), bestPlaneBuffer.getHeight(), 1, absurdPlane);
    Grid<float> bestCosts(bestPlaneCostBuffer.getWidth(), bestPlaneCostBuffer.getHeight(), 1, absurdCost);

    bestPlaneBuffer.download(bestPlanes.getDataPtr());
    bestPlaneCostBuffer.download(bestCosts.getDataPtr());

    // filter value
    if (_filterEnabled) {
      for (int i = 0; i < bestPlanes.getWidth(); ++i) {
        for (int j = 0; j < bestPlanes.getHeight(); ++j) {
          if (bestCosts(i, j) > _filterThreshold) {
            bestPlanes(i, j) = absurdPlane;
          }
        }
      }
    }

    // guided filter the label map
    // cv::Mat tmpMat(bestPlanes.getHeight(), bestPlanes.getWidth(), CV_32F, bestPlanes.getDataPtr());
    // guidedFilterPtr->filter(tmpMat, tmpMat);

    // save as file and image
    std::string outputFile = _outputFolder + "best/best_plane_index.bin";
    bestPlanes.saveAsDataFile(outputFile);

    std::string outputImg = _outputFolder + "best/best_plane_index.png";
    saveGridZSliceAsImage<float>(bestPlanes, 0, outputImg);

    outputFile = _outputFolder + "best/best_plane_cost.bin";
    bestCosts.saveAsDataFile(outputFile);

    outputImg = _outputFolder + "best/best_plane_cost.png";
    bestCosts(0, 0) = 0.0f;
    bestCosts(0, 1) = 1.0f;
    saveGridZSliceAsImage<float>(bestCosts, 0, outputImg, 0.0, 1.0);

    if (_outputXYZMapEnabled || _outputPointCloudEnabled) {
      // compute XYZ map on GPU
      PSL_CUDA::DeviceBuffer<float> xMapBuffer;
      PSL_CUDA::DeviceBuffer<float> yMapBuffer;
      PSL_CUDA::DeviceBuffer<float> zMapBuffer;
      xMapBuffer.allocatePitched(refImg.devImg.getWidth(), refImg.devImg.getHeight());
      yMapBuffer.allocatePitched(refImg.devImg.getWidth(), refImg.devImg.getHeight());
      zMapBuffer.allocatePitched(refImg.devImg.getWidth(), refImg.devImg.getHeight());

      float P[12] = {0.0f};
      refImg.cam.getFloatP(P);

      int num_planes = planes.size();
      float *tmp_planes = new float[num_planes * 4];
      for (int i=0; i < num_planes; ++i) {
        const Eigen::Vector4d &plane = planes[i];
        for (int j=0; j < 4; ++j) {
          tmp_planes[i * 4 + j] = float(plane(j));
        }
      }

      float absurdXYZ = 1e10f;
      PSL_CUDA::unprojectLabelMap(bestPlaneBuffer, tmp_planes, num_planes, P, absurdXYZ, xMapBuffer, yMapBuffer, zMapBuffer);
      // free memory
      delete [] tmp_planes;

      // download to cpu
      Grid<float> xyzMap(bestPlaneBuffer.getWidth(), bestPlaneBuffer.getHeight(), 3, absurdXYZ);
      xMapBuffer.download(xyzMap.getDataPtr(0));
      yMapBuffer.download(xyzMap.getDataPtr(1));
      zMapBuffer.download(xyzMap.getDataPtr(2));

      if (_outputXYZMapEnabled) {
        xyzMap.saveAsDataFile(_outputFolder + "best/best_xyz_map.bin");
        // saveGridZSliceAsImage<float>(xyzMap, 0, _outputFolder + "best/best_x.png");
        // saveGridZSliceAsImage<float>(xyzMap, 1, _outputFolder + "best/best_y.png");
        // saveGridZSliceAsImage<float>(xyzMap, 2, _outputFolder + "best/best_z.png");
      }

      if (_outputPointCloudEnabled) {
        cv::Mat refImgGray = downloadImage(refImgId);

        std::vector<PlyPoint> plyPoints;
        std::string plyFilename = _outputFolder + "best/best_point_cloud.ply";
        std::cout << "writing to " << plyFilename << "..." << std::endl;
        for (int i = 0; i < xyzMap.getWidth(); ++i) {
          for (int j = 0; j < xyzMap.getHeight(); ++j) {
            float x = xyzMap(i, j, 0);
            float y = xyzMap(i, j, 1);
            float z = xyzMap(i, j, 2);
            if (z < absurdXYZ) {
              unsigned char grayVal = refImgGray.at<unsigned char>(j, i);
              plyPoints.emplace_back(x, y, z,
                                     grayVal, grayVal, grayVal);
            }
          }
        }

        WriteBinaryPlyPoints(plyFilename, plyPoints);
      }

      // free memory
      xyzMap.freeMem();
      xMapBuffer.deallocate();
      yMapBuffer.deallocate();
      zMapBuffer.deallocate();
    }

//    if (_outputPointCloudEnabled) {
//      cv::Mat refImgGray = downloadImage(refImgId);
//      // cv::imwrite(_outputFolder + "refImg.png", refImgGray);
//
//      std::vector<PlyPoint> plyPoints;
//      std::string plyFilename = _outputFolder + "best/best_point_cloud.ply";
//      std::cout << "writing to " << plyFilename << "..." << std::endl;
//      for (int i = 0; i < bestPlanes.getWidth(); ++i) {
//        for (int j = 0; j < bestPlanes.getHeight(); ++j) {
//          int plane_idx = bestPlanes(i, j);
//          if (plane_idx != absurdPlane) {
//            Eigen::Vector3d point = refImg.cam.unprojectPoint(i, j, planes[plane_idx]);
//            unsigned char grayVal = refImgGray.at<unsigned char>(j, i);
//            plyPoints.emplace_back((float) point(0), (float) point(1), (float) point(2),
//                                   grayVal, grayVal, grayVal);
//          }
//        }
//      }
//
//      WriteBinaryPlyPoints(plyFilename, plyPoints);
//    }
  }

  // deallocate GPU buffer
  costAccumBuffer.deallocate();
  viewCntAccumBuffer.deallocate();
  if (_outputBestEnabled) {
    bestPlaneBuffer.deallocate();
    bestPlaneCostBuffer.deallocate();
  }

  if (_debugEnabled) {
    intensityAccumBuffer.deallocate();
  }
}

} // namespace PSL
