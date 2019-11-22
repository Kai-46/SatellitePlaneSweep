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


#include "psl_cudaBase/deviceBuffer.cuh"
#include "psl_cudaBase/deviceImage.cuh"

namespace PSL_CUDA {
enum MATCHING_COST_TYPE {
  NCC_MATCHING_COST,
  CENSUS_MATCHING_COST
};

// function declarations
void initTexture();
void costAndIntensityAccum(MATCHING_COST_TYPE costType,
                           const DeviceImage &refImg,
                           const DeviceImage &srcImg,
                           float *hMat,
                           int radius_x,
                           int radius_y,
                           DeviceBuffer<float> &costAccumBuf,
                           DeviceBuffer<float> &viewCntAccumBuf,
                           DeviceBuffer<float> &intensityAccumBuf);
void updateCostAndIntensityAndBestPlane(DeviceBuffer<float> &costAccumBuf,
                                        DeviceBuffer<float> &viewCntAccumBuf,
                                        float absurdVal,
                                        float currPlaneIndex,
                                        DeviceBuffer<float> &intensityAccumBuf,
                                        DeviceBuffer<float> &bestPlaneCosts,
                                        DeviceBuffer<float> &bestPlanes);
void filterBestPlane(const DeviceBuffer<float> &bestPlaneCosts, float filterThres,
                     DeviceBuffer<float> &bestPlanes, float absurdPlane);

// implement a = a / (b + c)
//    void perPixelDivision(DeviceBuffer<float>& numeratorBuf, DeviceBuffer<float>& denominatorBuf, float denominatorBias=0.0f);
//	void scaleCost(DeviceBuffer<float>& costAccumBuf, float scale);


// unproject label map to xyz map
// need 3 by 4 projection matrix
void unprojectLabelMap(const DeviceBuffer<float> &bestPlanesBuf, float *planes, int num_planes, float *P, float absurdXYZ,
                       DeviceBuffer<float> &xMapBuf, DeviceBuffer<float> &yMapBuf, DeviceBuffer<float> &zMapBuf);


}    // namespace PSL_CUDA
