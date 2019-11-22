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


#include "grid.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <limits>

namespace PSL {

#define CLAMP_MIN_DEFAULT -1e30
#define CLMAP_MAX_DEFAULT 1e30

template<typename T>
void saveGridZSliceAsImage(const Grid<T> &grid, const int z, const std::string &fileName,
                           double clamp_min = CLAMP_MIN_DEFAULT, double clamp_max = CLMAP_MAX_DEFAULT) {
  // copy data to double cv::Mat
  int height = grid.getHeight();
  int width = grid.getWidth();
  cv::Mat_<double> sliceMat(height, width);

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      double val = (double) grid(j, i, z);
      if (val < clamp_min) {
        val = clamp_min;
      }

      if (val > clamp_max) {
        val = clamp_max;
      }
      sliceMat.at<double>(i, j) = val;
    }
  }

  double min, max;
  cv::minMaxLoc(sliceMat, &min, &max);
  cv::imwrite(fileName, (sliceMat - min) / (max - min) * 255);
}

template void saveGridZSliceAsImage<int>(const Grid<int> &grid,
                                         const int z,
                                         const std::string &fileName,
                                         double clamp_min,
                                         double clamp_max);
template void saveGridZSliceAsImage<double>(const Grid<double> &grid,
                                            const int z,
                                            const std::string &fileName,
                                            double clamp_min,
                                            double clamp_max);
template void saveGridZSliceAsImage<float>(const Grid<float> &grid,
                                           const int z,
                                           const std::string &fileName,
                                           double clamp_min,
                                           double clamp_max);

} // namespace PSL
