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
//	std::cout << "min: " << min << ", max: " << max << std::endl;
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
