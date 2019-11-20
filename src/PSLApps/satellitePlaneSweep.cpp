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


#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <chrono> 

#include "psl_base/exception.h"
#include "psl_base/cameraMatrix.h"
#include "psl_base/gridVisualize.h"
#include "psl_base/grid.h"
#include "psl_stereo/cudaPlaneSweep.h"
#include "psl_base/ioTools.h"

int main(int argc, char *argv[]) {
  std::string imageListFile;
  std::string imageFolder;
  std::string outputFolder;
  int refImgId;
  int windowRadius;

  bool filterCostVolume;
  int guidedFilterRadius;
  double guidedFilterEps;

  bool saveBest;
  bool saveCostVolume;
  bool saveXYZMap;
  bool savePointCloud;

  // plane equation is in the scene coordinate frame
  // n^T - d = 0, with n pointing away from the origin
  double nX;
  double nY;
  double nZ;
  double firstD;
  double lastD;
  int numPlanes;
  bool filter;
  double filterThres; // pixels whose cost above which would be replaced with empty pixels
  bool debug;
  std::string matchingCost;

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
      ("help", "Produce help message")
      ("imageList",
       boost::program_options::value<std::string>(&imageListFile),
       "image list specifying image id, image name, camera parameters")
      ("imageFolder", boost::program_options::value<std::string>(&imageFolder), "image folder containing the images")
      ("refImgId", boost::program_options::value<int>(&refImgId), "image index for the reference view")
      ("outputFolder", boost::program_options::value<std::string>(&outputFolder), "output folder")
      ("matchingCost", boost::program_options::value<std::string>(&matchingCost)->default_value("ncc"), "ncc or census")
      ("windowRadius",
       boost::program_options::value<int>(&windowRadius),
       "window radius for computing similarity metric")
      ("nX", boost::program_options::value<double>(&nX), "normal of sweeping planes")
      ("nY", boost::program_options::value<double>(&nY), "normal of sweeping planes")
      ("nZ", boost::program_options::value<double>(&nZ), "normal of sweeping planes")
      ("firstD", boost::program_options::value<double>(&firstD), "constant of first sweeping plane")
      ("lastD", boost::program_options::value<double>(&lastD), "constant of last sweeping plane")
      ("numPlanes", boost::program_options::value<int>(&numPlanes), "number of sweeping planes")
      ("filterCostVolume",
       boost::program_options::value<bool>(&filterCostVolume),
       "0 or 1; whether to filter the cost volume with a guided-filter")
      ("guidedFilterRadius",
       boost::program_options::value<int>(&guidedFilterRadius),
       "radius parameter for guided-filter")
      ("guidedFilterEps", boost::program_options::value<double>(&guidedFilterEps), "eps parameter for guided filter")
      ("saveBest",
       boost::program_options::value<bool>(&saveBest)->default_value(true),
       "0 or 1; whether to save the best label and cost")
      ("saveCostVolume",
       boost::program_options::value<bool>(&saveCostVolume)->default_value(false),
       "0 or 1; whether to save the entire cost volume")
      ("saveXYZMap",
       boost::program_options::value<bool>(&saveXYZMap)->default_value(false),
       "0 or 1; whether to save XYZ map")
      ("savePointCloud",
       boost::program_options::value<bool>(&savePointCloud)->default_value(false),
       "0 or 1; whether to save the point cloud")
      ("filter",
       boost::program_options::value<bool>(&filter)->default_value(false),
       "0 or 1; whether to filter the best label")
      ("filterThres",
       boost::program_options::value<double>(&filterThres),
       "between 0.0 and 1.0; cells with cost above this threshold will be filtered")
      ("debug", boost::program_options::value<bool>(&debug)->default_value(false), "0 or 1");

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  // add '/' to imageFolder
  if (imageFolder.back() != '/') {
    imageFolder += '/';
  }

  // add '/' to outputFolder
  if (outputFolder.back() != '/') {
    outputFolder += '/';
  }

  PSL::makeOutputFolder(outputFolder);

  // try to read image list
  std::ifstream imageListReader;
  imageListReader.open(imageListFile);
  if (!imageListReader.is_open()) {
    PSL_THROW_EXCEPTION("Failed to open file")
  }

  std::string line;
  std::vector<int> imageIds;
  std::vector<std::string> imageNames;
  std::vector<PSL::CameraMatrix> cameraMatrices;

  while (std::getline(imageListReader, line)) {
    std::istringstream iss(line);
    int imageId;
    std::string imageName;
    iss >> imageId >> imageName;

    // read fx, fy, cx, cy, s
    Eigen::Matrix<double, 3, 3> K = Eigen::Matrix<double, 3, 3>::Identity();
    iss >> K(0, 0);
    iss >> K(1, 1);
    iss >> K(0, 2);
    iss >> K(1, 2);
    iss >> K(0, 1);

    // read quaternion
    double qvec[4];
    iss >> qvec[0] >> qvec[1] >> qvec[2] >> qvec[3];
    Eigen::Matrix<double, 3, 3> R = Eigen::Quaternion<double>(qvec[0], qvec[1], qvec[2], qvec[3]).toRotationMatrix();

    // read translation
    Eigen::Vector3d T;
    iss >> T(0) >> T(1) >> T(2);

    // add
    imageIds.push_back(imageId);
    imageNames.push_back(imageName);
    cameraMatrices.emplace_back(K, R, T);
  }

  PSL::CudaPlaneSweep cPS;
  cPS.setMatchWindowSize(2 * windowRadius + 1, 2 * windowRadius + 1);
  cPS.setOutputFolder(outputFolder);

  cPS.enableCostVolumeFilter(filterCostVolume, guidedFilterRadius, (float) guidedFilterEps);
  cPS.enableOutputCostVolume(saveCostVolume);
  cPS.enableOutputBest(saveBest);
  cPS.enableDebug(debug);
  cPS.enableFilterCostAbove(filter, (float) filterThres);
  cPS.enableOutputXYZMap(saveXYZMap);
  cPS.enableOutputPointCloud(savePointCloud);

  // now we upload the images
  int refId = -1;     // internal id used by the plane sweep stereo
  for (int i = 0; i < imageNames.size(); ++i) {
    // load the image from disk
    cv::Mat image = cv::imread(imageFolder + imageNames[i]);
    if (image.empty()) {
      PSL_THROW_EXCEPTION("Failed to load image")
    }

    int id = cPS.addImage(image, cameraMatrices[i]);

    if (imageIds[i] == refImgId) {
      refId = id;
    }
  }

  // generate the planes
  std::vector<Eigen::Vector4d> planes;
  double interval = (lastD - firstD) / (numPlanes - 1);
  for (int i = 0; i < numPlanes; ++i) {
    Eigen::Vector4d currPlane;
    currPlane << nX, nY, nZ, (firstD + i * interval);
    planes.push_back(currPlane);
  }

  auto start = std::chrono::high_resolution_clock::now();

  cPS.process(refId, planes, matchingCost);

  auto stop = std::chrono::high_resolution_clock::now(); 
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
  std::cout << "\nelapsed time (mins): " << duration.count() / 1e6 / 60.0 << std::endl; 

  return 0;
}
