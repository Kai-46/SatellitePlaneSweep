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
