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


#include <assert.h>
// #include <iostream>
#include <stdio.h>


#include "psl_cudaBase/cudaCommon.cuh"
#include "psl_cudaBase/cudaBackbone.cuh"

namespace PSL_CUDA {

// texture memory are read-only
// reference image
texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> refGrayTexture;
// source image
texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> srcGrayTexture;

void initTexture() {
  refGrayTexture.addressMode[0] = cudaAddressModeBorder;
  refGrayTexture.addressMode[1] = cudaAddressModeBorder;
  refGrayTexture.filterMode = cudaFilterModePoint;
  refGrayTexture.normalized = false;

  srcGrayTexture.addressMode[0] = cudaAddressModeBorder;
  srcGrayTexture.addressMode[1] = cudaAddressModeBorder;
  srcGrayTexture.filterMode = cudaFilterModeLinear;
  srcGrayTexture.normalized = false;
}
// chop a big image into smaller tiles
// expand each tile by window radius at margin to get a block of pixels
// dedicate one thread to each pixel inside the block
// ref block and src block are cached with shared memory


// _global__ cannot use pass by reference.
__global__ void nccCostAndIntensityAccumKernel(DeviceBuffer<float> costAccumBuf,
                                               DeviceBuffer<float> intensityAccumBuf,
                                               DeviceBuffer<float> viewCntAccumBuf,
                                               int radius_x,
                                               int radius_y,
                                               float h11,
                                               float h12,
                                               float h13,
                                               float h21,
                                               float h22,
                                               float h23,
                                               float h31,
                                               float h32,
                                               float h33) {
  // block size is slightly bigger than tile size as a result of expansion at tile margin
  // ideally, block size should be a multiple of 32

  const int tile_w = blockDim.x - 2 * radius_x;
  const int tile_h = blockDim.y - 2 * radius_y;
  const int block_size = blockDim.x * blockDim.y;

  // dynamic linear shared memory; size specified during kernel launch
  // note that only one shared memory can be used
  extern __shared__ float ref_src_block[];

  // index in the linear shared memory
  const int index = threadIdx.y * blockDim.x + threadIdx.x;

  // index in the 2D reference image
  const int ref_x = blockIdx.x * tile_w - radius_x + threadIdx.x;
  const int ref_y = blockIdx.y * tile_h - radius_y + threadIdx.y;

  ref_src_block[index] = tex2D(refGrayTexture, ref_x, ref_y);

  // index in the 2D source image; apply homography
  float src_x = h11 * ref_x + h12 * ref_y + h13;
  float src_y = h21 * ref_x + h22 * ref_y + h23;
  const float z = h31 * ref_x + h32 * ref_y + h33;
  src_x /= z;
  src_y /= z;

  // the half pixel is explained here
  // https://stackoverflow.com/questions/38246917/where-does-normalized-texture-memory-start
  ref_src_block[block_size + index] = tex2D(srcGrayTexture, src_x + 0.5f, src_y + 0.5f);

  __syncthreads();

  // only threads that are both inside the orginal image and also inside the tile will write results
  const int width = costAccumBuf.getWidth();
  const int height = costAccumBuf.getHeight();
  if ((ref_x >= 0) && (ref_x < width) &&
      (ref_y >= 0) && (ref_y < height) &&
      (threadIdx.x >= radius_x) && (threadIdx.x < (blockDim.x - radius_x)) &&
      (threadIdx.y >= radius_y) && (threadIdx.y < (blockDim.y - radius_y))) {
    // ncc between two random variables X, Y
    // numerator = E[XY] - E[X]E[Y]
    // denominator = sqrt(E[X^2] - E[X]^2) sqrt(E[Y^2] - E[Y]^2)
    // ncc = numerator/denominator \in [-1, 1]
    // cost = (1 - ncc) / 2 \in [0, 1]
    const float elem_cnt = (2 * radius_x + 1) * (2 * radius_y + 1);
    float ref_sum = 0.0f;
    float src_sum = 0.0f;
    float sqr_ref_sum = 0.0f;
    float sqr_src_sum = 0.0f;
    float prod_sum = 0.0f;

    for (int dy = -radius_y; dy <= radius_y; ++dy) {
      for (int dx = -radius_x; dx <= radius_x; ++dx) {
        const int idx = index + dy * blockDim.x + dx;
        const float ref_pixel = ref_src_block[idx];
        const float src_pixel = ref_src_block[block_size + idx];

        ref_sum += ref_pixel;
        src_sum += src_pixel;
        sqr_ref_sum += ref_pixel * ref_pixel;
        sqr_src_sum += src_pixel * src_pixel;
        prod_sum += ref_pixel * src_pixel;
      }
    }

    // average
    ref_sum /= elem_cnt;
    src_sum /= elem_cnt;
    sqr_ref_sum /= elem_cnt;
    sqr_src_sum /= elem_cnt;
    prod_sum /= elem_cnt;

    // check whether all src pixels lie out of the boundary
    // since we are using cudaAddressModeBorder for texture memory, texels outside boundary will be zero
    // for numeric stability, the variance can't be two small
    const float ref_var = sqr_ref_sum - ref_sum * ref_sum;
    const float src_var = sqr_src_sum - src_sum * src_sum;
    const float var_threshold = 1e-5f;
    if (ref_var > var_threshold && src_var > var_threshold) {
      float ncc = (prod_sum - ref_sum * src_sum) /
          (sqrt((sqr_ref_sum - ref_sum * ref_sum) * (sqr_src_sum - src_sum * src_sum)));
//				assert (ncc > -1.001f && ncc < 1.001f);	// double-check if ncc computation is numerically stable
      if (ncc < -1.0f) {
        ncc = -1.0f;
      }
      if (ncc > 1.0f) {
        ncc = 1.0f;
      }
      costAccumBuf(ref_x, ref_y) += (1.0f - ncc) / 2.0f;    // accumulate cost
      if (intensityAccumBuf.isAllocated()) {  // intensityAccumBuf is only used for debugging purpose
        intensityAccumBuf(ref_x, ref_y) += ref_src_block[block_size + index];
      }
      viewCntAccumBuf(ref_x, ref_y) += 1.0f;        // an effective view
    }
  }
}

// _global__ cannot use pass by reference.
__global__ void censusCostAndIntensityAccumKernel(DeviceBuffer<float> costAccumBuf,
                                                  DeviceBuffer<float> intensityAccumBuf,
                                                  DeviceBuffer<float> viewCntAccumBuf,
                                                  int radius_x,
                                                  int radius_y,
                                                  float h11,
                                                  float h12,
                                                  float h13,
                                                  float h21,
                                                  float h22,
                                                  float h23,
                                                  float h31,
                                                  float h32,
                                                  float h33) {
  // block size is slightly bigger than tile size as a result of expansion at tile margin
  // ideally, block size should be a multiple of 32

  const int tile_w = blockDim.x - 2 * radius_x;
  const int tile_h = blockDim.y - 2 * radius_y;
  const int block_size = blockDim.x * blockDim.y;

  // dynamic linear shared memory; size specified during kernel launch
  // note that only one shared memory can be used
  extern __shared__ float ref_src_block[];

  // index in the linear shared memory
  const int index = threadIdx.y * blockDim.x + threadIdx.x;

  // index in the 2D reference image
  int ref_x = blockIdx.x * tile_w - radius_x + threadIdx.x;
  int ref_y = blockIdx.y * tile_h - radius_y + threadIdx.y;

  ref_src_block[index] = tex2D(refGrayTexture, ref_x, ref_y);

  // index in the 2D source image; apply homography
  float src_x = h11 * ref_x + h12 * ref_y + h13;
  float src_y = h21 * ref_x + h22 * ref_y + h23;
  const float z = h31 * ref_x + h32 * ref_y + h33;
  src_x /= z;
  src_y /= z;

  // the half pixel is explained here
  // https://stackoverflow.com/questions/38246917/where-does-normalized-texture-memory-start
  ref_src_block[block_size + index] = tex2D(srcGrayTexture, src_x + 0.5f, src_y + 0.5f);

  __syncthreads();

  // only threads that are both inside the orginal image and also inside the tile will write results
  const int width = costAccumBuf.getWidth();
  const int height = costAccumBuf.getHeight();
  if ((ref_x >= 0) && (ref_x < width) &&
      (ref_y >= 0) && (ref_y < height) &&
      (threadIdx.x >= radius_x) && (threadIdx.x < (blockDim.x - radius_x)) &&
      (threadIdx.y >= radius_y) && (threadIdx.y < (blockDim.y - radius_y))) {

    const float elem_cnt = (2 * radius_x + 1) * (2 * radius_y + 1);
    float cnt = 0.0f;    // count how many pixels that disobeys the relative order
    const float ref_pixel = ref_src_block[index];
    const float src_pixel = ref_src_block[block_size + index];
    for (int dy = -radius_y; dy <= radius_y; ++dy) {
      for (int dx = -radius_x; dx <= radius_x; ++dx) {
        if (dx == 0 && dy == 0) {   // center pixel
          continue;
        }

        const int idx = index + dy * blockDim.x + dx;
        if ((ref_src_block[idx] - ref_pixel) * (ref_src_block[block_size + idx] - src_pixel) <= 0.0f) {
          cnt += 1.0f;    // unconsensus count
        }
      }
    }
    // normalize cost to [0, 1]
    costAccumBuf(ref_x, ref_y) += (cnt / (elem_cnt - 1.0f));    // accumulate cost
    if (intensityAccumBuf.isAllocated()) {  // intensityAccumBuf is only used for debugging purpose
      intensityAccumBuf(ref_x, ref_y) += ref_src_block[block_size + index];
    }
    viewCntAccumBuf(ref_x, ref_y) += 1.0f;        // an effective view
  }
}

void costAndIntensityAccum(MATCHING_COST_TYPE costType,
                           const DeviceImage &refImg,
                           const DeviceImage &srcImg,
                           float *hMat,
                           int radius_x,
                           int radius_y,
                           DeviceBuffer<float> &costAccumBuf,
                           DeviceBuffer<float> &viewCntAccumBuf,
                           DeviceBuffer<float> &intensityAccumBuf) {
  // bind textures
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
  size_t offset = 0;
  PSL_CUDA_CHECKED_CALL(cudaBindTexture2D(&offset,
                                          refGrayTexture,
                                          refImg.getAddr(),
                                          channelDesc,
                                          refImg.getWidth(),
                                          refImg.getHeight(),
                                          refImg.getPitch());)
  PSL_CUDA_CHECKED_CALL(cudaBindTexture2D(&offset,
                                          srcGrayTexture,
                                          srcImg.getAddr(),
                                          channelDesc,
                                          srcImg.getWidth(),
                                          srcImg.getHeight(),
                                          srcImg.getPitch());)

  // compute kernel config
  // block size is slightly bigger than tile size as a result of expansion at tile margin
  // ideally, block size should be a multiple of 32
  // cuda device has a maximum limit on number of threads per block
  // The maximum number of threads in the block is limited to 1024
  // also since the size of shared memory is limited, we might try to avoid using a big block size unless we have to

  if (radius_x >= 16 or radius_y >= 16) {
    PSL_THROW_EXCEPTION("window size is too big; maximum supported window radius is 15")
  }

  const int block_w = 32;
  const int block_h = 32;
  const int tile_w = block_w - 2 * radius_x;
  const int tile_h = block_h - 2 * radius_y;

  dim3 blockDim(block_w, block_h);
  // compute optimal tiling based on block size
  dim3 gridDim(getNumTiles(refImg.getWidth(), tile_w), getNumTiles(refImg.getHeight(), tile_h));

  // launch kernel
  if (costType == NCC_MATCHING_COST) {
    nccCostAndIntensityAccumKernel << < gridDim, blockDim, 2 * block_w * block_h * sizeof(float) >>
        > (costAccumBuf, intensityAccumBuf, viewCntAccumBuf, radius_x, radius_y,
            hMat[0], hMat[1], hMat[2], hMat[3], hMat[4], hMat[5], hMat[6], hMat[7], hMat[8]);
  } else if (costType == CENSUS_MATCHING_COST) {
    censusCostAndIntensityAccumKernel << < gridDim, blockDim, 2 * block_w * block_h * sizeof(float) >>
        > (costAccumBuf, intensityAccumBuf, viewCntAccumBuf, radius_x, radius_y,
            hMat[0], hMat[1], hMat[2], hMat[3], hMat[4], hMat[5], hMat[6], hMat[7], hMat[8]);
  }

  PSL_CUDA_CHECK_ERROR

  // unbind textures
  PSL_CUDA_CHECKED_CALL(cudaUnbindTexture(refGrayTexture);)
  PSL_CUDA_CHECKED_CALL(cudaUnbindTexture(srcGrayTexture);)
}

__global__ void updateCostAndIntensityAndBestPlaneKernel(DeviceBuffer<float> costAccumBuf,
                                                         DeviceBuffer<float> viewCntAccumBuf,
                                                         float absurdVal,
                                                         float currPlaneIndex,
                                                         DeviceBuffer<float> intensityAccumBuf,
                                                         DeviceBuffer<float> bestPlaneCosts,
                                                         DeviceBuffer<float> bestPlanes) {
  // get position of outupt
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < costAccumBuf.getWidth() && y < costAccumBuf.getHeight()) {  // if you are confused by this index, please check the kernel config
    float view_cnt = viewCntAccumBuf(x, y);
    // update cost
    if (view_cnt > 0.0f) {
      float cost = costAccumBuf(x, y) / view_cnt;
      costAccumBuf(x, y) = cost;

      // update best plane
      if (bestPlaneCosts.isAllocated() && bestPlanes.isAllocated()) {
        if (cost < bestPlaneCosts(x, y)) {
          bestPlaneCosts(x, y) = cost;
          bestPlanes(x, y) = currPlaneIndex;
        }
      }
    } else {
      costAccumBuf(x, y) = absurdVal;
    }

    // update intensity
    if (intensityAccumBuf.isAllocated()) {
      intensityAccumBuf(x, y) /= (view_cnt + 1.0f);
    }
  }
}

void updateCostAndIntensityAndBestPlane(DeviceBuffer<float> &costAccumBuf,
                                        DeviceBuffer<float> &viewCntAccumBuf,
                                        float absurdVal,
                                        float currPlaneIndex,
                                        DeviceBuffer<float> &intensityAccumBuf,
                                        DeviceBuffer<float> &bestPlaneCosts,
                                        DeviceBuffer<float> &bestPlanes) {
  dim3 gridDim(getNumTiles(costAccumBuf.getWidth(), TILE_WIDTH), getNumTiles(costAccumBuf.getHeight(), TILE_HEIGHT));
  dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);

  updateCostAndIntensityAndBestPlaneKernel << < gridDim, blockDim >>
      > (costAccumBuf, viewCntAccumBuf, absurdVal, currPlaneIndex, intensityAccumBuf, bestPlaneCosts, bestPlanes);
  PSL_CUDA_CHECK_ERROR
}

//	__global__ void scaleCostKernel(DeviceBuffer<float> costAccumBuf, float scale) {
//		// index in the 2D reference image
//		int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
//		int y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
//
//		if (x < costAccumBuf.getWidth() && y < costAccumBuf.getHeight()) {
//			costAccumBuf(x, y) *= scale;
//		}
//	}
//
//	void scaleCost(DeviceBuffer<float>& costAccumBuf, float scale) {
//		dim3 gridDim(getNumTiles(costAccumBuf.getWidth(), TILE_WIDTH), getNumTiles(costAccumBuf.getHeight(), TILE_HEIGHT));
//		dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
//
//		scaleCostKernel<<<gridDim, blockDim>>>(costAccumBuf, scale);
//		PSL_CUDA_CHECK_ERROR
//	}


//	__global__ void perPixelDivisionKernel(DeviceBuffer<float> numeratorBuf, DeviceBuffer<float> denominatorBuf, float denominatorBias) {
//		// index in the 2D reference image
//		int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
//		int y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
//		if (x < numeratorBuf.getWidth() && y < numeratorBuf.getHeight()) {
//		    numeratorBuf(x, y) /= (denominatorBuf(x, y) + denominatorBias);
//		}
//	}
//
//	// implement a / (b + c)
//	void perPixelDivision(DeviceBuffer<float>& numeratorBuf, DeviceBuffer<float>& denominatorBuf, float denominatorBias) {
//		dim3 gridDim(getNumTiles(numeratorBuf.getWidth(), TILE_WIDTH), getNumTiles(numeratorBuf.getHeight(), TILE_HEIGHT));
//		dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
//
//		perPixelDivisionKernel<<<gridDim, blockDim>>>(numeratorBuf, denominatorBuf, denominatorBias);
//		PSL_CUDA_CHECK_ERROR
//	}


__global__ void filterBestPlaneKernel(DeviceBuffer<float> bestPlaneCostsBuf, float filterThres,
                                      DeviceBuffer<float> bestPlanesBuf, float absurdPlane) {
  // get position of outupt
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < bestPlaneCostsBuf.getWidth() && y < bestPlaneCostsBuf.getHeight()) {  // if you are confused by this index, please check the kernel config
    if (bestPlaneCostsBuf(x, y) > filterThres) {
      bestPlanesBuf(x, y) = absurdPlane;
    }
  }
}

void filterBestPlane(const DeviceBuffer<float> &bestPlaneCostsBuf, float filterThres,
                     DeviceBuffer<float> &bestPlanesBuf, float absurdPlane) {
  dim3 gridDim(getNumTiles(bestPlaneCostsBuf.getWidth(), TILE_WIDTH), getNumTiles(bestPlaneCostsBuf.getHeight(), TILE_HEIGHT));
  dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);

  filterBestPlaneKernel << < gridDim, blockDim >>
      > (bestPlaneCostsBuf, filterThres, bestPlanesBuf, absurdPlane);
  PSL_CUDA_CHECK_ERROR
}



__global__ void unprojectLabelMapKernel(DeviceBuffer<float> bestPlanesBuf, float *planes, int num_planes, 
                                        DeviceBuffer<float> xMapBuf, DeviceBuffer<float> yMapBuf, DeviceBuffer<float> zMapBuf,
                                        float absurdXYZ,
                                        float P11, float P12, float P13, float P14,
                                        float P21, float P22, float P23, float P24,
                                        float P31, float P32, float P33, float P34) {
  // get position of outupt
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < bestPlanesBuf.getWidth() && y < bestPlanesBuf.getHeight()) {
    // first interpolate plane equation
    float plane_idx = bestPlanesBuf(x, y);
    int plane_idx_below = __float2int_rd(plane_idx);
    int plane_idx_up = __float2int_ru(plane_idx);

    if (plane_idx_below >= 0 && plane_idx_up < num_planes) { // a valid value
      // interpolate d value of the constant
      // inverse proportional to the distance
      float d_val = (plane_idx_up - plane_idx) * planes[plane_idx_below * 4 + 3] +
                      (plane_idx - plane_idx_below) * planes[plane_idx_up * 4 + 3];
      if (plane_idx_below == plane_idx_up) {  // degenerate case
        d_val = planes[plane_idx_below * 4 + 3];
      }

      float tmp = planes[plane_idx_below * 4 + 2]; // nz
      // Z = a' * [X; Y; 1]
      float a[3] = {-planes[plane_idx_below * 4 + 0] / tmp,
                    -planes[plane_idx_below * 4 + 1] / tmp,
                    d_val / tmp};
      // problematic code, not a problem with precision
      // [u; v; 1] = A * [X; Y; 1]
      float A11 = P11 + P13 * a[0];
      float A12 = P12 + P13 * a[1];
      float A13 = P14 + P13 * a[2];

      float A21 = P21 + P23 * a[0];
      float A22 = P22 + P23 * a[1];
      float A23 = P24 + P23 * a[2];

      float A31 = P31 + P33 * a[0];
      float A32 = P32 + P33 * a[1];
      float A33 = P34 + P33 * a[2];

      // solve for X, Y
      // B is inverse of A
      // tmp = A11*A22*A33 - A11*A23*A32 - A12*A21*A33 + A12*A23*A31 + A13*A21*A32 - A13*A22*A31;
      // // float B11 = (A22 * A33 - A23 * A32) / tmp;
      // // float B12 = -(A21 * A33 - A23 * A31) / tmp;
      // // float B13 = (A21 * A32 - A22 * A31) / tmp;

      // // float B21 = -(A12 * A33 - A13 * A32) / tmp;
      // // float B22 = (A11 * A33 - A13 * A31) / tmp;
      // // float B23 = -(A11 * A32 - A12 * A31) / tmp;

      // // float B31 = (A12 * A23 - A13 * A22) / tmp;
      // // float B32 = -(A11 * A23 - A13 * A21) / tmp;
      // // float B33 = (A11 * A22 - A12 * A21) / tmp;

      // float B11 = (A22 * A33 - A23 * A32) / tmp;
      // float B12 = -(A12 * A33 - A13 * A32) / tmp;
      // float B13 = (A12 * A23 - A13 * A22) / tmp;

      // float B21 = -(A21 * A33 - A23 * A31) / tmp;
      // float B22 = (A11 * A33 - A13 * A31) / tmp;
      // float B23 = -(A11 * A23 - A13 * A21) / tmp;

      // float B31 = (A21 * A32 - A22 * A31) / tmp;
      // float B32 = -(A11 * A32 - A12 * A31) / tmp;
      // float B33 = (A11 * A22 - A12 * A21) / tmp;

      // tmp = B31 * x + B32 * y + B33;
      // float X = (B11 * x + B12 * y + B13) / tmp;
      // float Y = (B21 * x + B22 * y + B23) / tmp;

      // a more numeric stable way
      // B[X;Y] = m;
      float B11 = A11 - x * A31;
      float B12 = A12 - x * A32;
      float m1 = -(A13 - x * A33);

      float B21 = A21 - y * A31;
      float B22 = A22 - y * A32;
      float m2 = -(A23 - y * A33);

      // solve equation
      tmp = B11 * B22 - B12 * B21;
      float X = (B22 * m1 - B12 * m2) / tmp;
      float Y = (-B21 * m1 + B11 * m2) / tmp;
      float Z = a[0] * X + a[1] * Y + a[2];
      
      // if (x==y && x==2079) {
      //   printf("tmp: %.7e\n", tmp);
      //   printf("P:\n");
      //   printf("%.7e, %.7e, %.7e, %.7e\n", P11, P12, P13, P14);
      //   printf("%.7e, %.7e, %.7e, %.7e\n", P21, P22, P23, P24);
      //   printf("%.7e, %.7e, %.7e, %.7e\n", P31, P32, P33, P34);
      //   printf("nx, ny, nz, d: %.7e, %.7e, %.7e, %.7e\n", planes[idx], planes[idx+1], planes[idx+2], planes[idx+3]);
      //   printf("xy: %i, %i, XYZ: %.7e, %.7e, %.7e\n", x, y, X, Y, Z);
      //   // printf("A: %.7e, %.7e, %.7e, %.7e, %.7e, %.7e, %.7e, %.7e, %.7e, B: %.7e, %.7e, %.7e, %.7e, %.7e, %.7e, %.7e, %.7e, %.7e, plane_idx: %i, plane: %.7e, %.7e, %.7e, %.7e, xy: %i, %i, XYZ: %.7e, %.7e, %.7e\n", 
      //   //   A11, A12, A13, A21, A22, A23, A31, A32, A33,
      //   //   B11, B12, B13, B21, B22, B23, B31, B32, B33,
      //   //   plane_idx, planes[idx + 0], planes[idx + 1], planes[idx + 2], planes[idx + 3],
      //   //   x, y, X, Y, Z);

      //   tmp = A31 * X + A32 * Y + A33;
      //   float new_uu = (A11 * X + A12 * Y + A13) / tmp;
      //   float new_vv = (A21 * X + A22 * Y + A23) / tmp;
      //   printf("tmp: %.7e, new new uv: %.7e, %.7e\n", tmp, new_uu, new_vv);

      //   // check forward computation
      //   tmp = P31 * X + P32 * Y + P33 * Z + P34;
      //   float new_u = (P11 * X + P12 * Y + P13 * Z + P14) / tmp;
      //   float new_v = (P21 * X + P22 * Y + P23 * Z + P24) / tmp;
      //   printf("new uv: %.7e, %.7e\n", new_u, new_v);
      // }

      xMapBuf(x, y) = X;
      yMapBuf(x, y) = Y;
      zMapBuf(x, y) = Z;
    } else {
      xMapBuf(x, y) = absurdXYZ;
      yMapBuf(x, y) = absurdXYZ;
      zMapBuf(x, y) = absurdXYZ;
    }
  }
}

// __global__ void cudaPrint(float *planes, int num_planes) {
//   for (int i=0; i < num_planes; ++i) {
//     printf("plane %i: ", i);
//     for (int j=0; j < 4; ++j) {
//       printf("%.7e, ", planes[i * 4 + j]);
//     }
//     printf("\n");
//   }
// }


// __global__ void cudaTestUnproject(float *planes, int plane_idx, int x, int y, 
//                                   float P11, float P12, float P13, float P14,
//                                   float P21, float P22, float P23, float P24,
//                                   float P31, float P32, float P33, float P34) {

//     int idx = plane_idx * 4;
//     float tmp = planes[idx + 2]; // nz
//     // Z = a' * [X; Y; 1]
//     float a[3] = {-planes[idx + 0] / tmp,
//                   -planes[idx + 1] / tmp,
//                   planes[idx + 3] / tmp};
//     // [u; v; 1] = A * [X; Y; 1]
//     float A11 = P11 + P13 * a[0];
//     float A12 = P12 + P13 * a[1];
//     float A13 = P14 + P13 * a[2];

//     float A21 = P21 + P23 * a[0];
//     float A22 = P22 + P23 * a[1];
//     float A23 = P24 + P23 * a[2];

//     float A31 = P31 + P33 * a[0];
//     float A32 = P32 + P33 * a[1];
//     float A33 = P34 + P33 * a[2];


//     // solve for X, Y
//     tmp = A11 * A22 - A12 * A21 + A21 * A32 * x - A22 * A31 * x - A11 * A32 * y + A12 * A31 * y;
//     float X = (A12 * A23 - A13 * A22 + A22 * A33 * x - A23 * A32 * x - A12 * A33 * y + A13 * A32 * y) / tmp;
//     float Y = -(A11 * A23 - A13 * A21 + A21 * A33 * x - A23 * A31 * x - A11 * A33 * y + A13 * A31 * y) / tmp;
//     float Z = a[0] * X + a[1] * Y + a[2];


//     printf("P:\n");
//     printf("%.7e, %.7e, %.7e, %.7e\n", P11, P12, P13, P14);
//     printf("%.7e, %.7e, %.7e, %.7e\n", P21, P22, P23, P24);
//     printf("%.7e, %.7e, %.7e, %.7e\n", P31, P32, P33, P34);
//     printf("nx, ny, nz, d: %.7e, %.7e, %.7e, %.7e\n", planes[idx], planes[idx+1], planes[idx+2], planes[idx+3]);
//     printf("xy: %i, %i, XYZ: %.7e, %.7e, %.7e\n", x, y, X, Y, Z);
// }

void unprojectLabelMap(const DeviceBuffer<float> &bestPlanesBuf, float *planes, int num_planes, float *P, float absurdXYZ,
                       DeviceBuffer<float> &xMapBuf, DeviceBuffer<float> &yMapBuf, DeviceBuffer<float> &zMapBuf) {
  dim3 gridDim(getNumTiles(bestPlanesBuf.getWidth(), TILE_WIDTH), getNumTiles(bestPlanesBuf.getHeight(), TILE_HEIGHT));
  dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);

  // for (int i=0; i < num_planes; ++i) {
  //   std::cout << "plane " << i << ": ";
  //   for (int j=0; j < 4; ++j) {
  //     std::cout << planes[i * 4 + j] << ",";
  //   }
  //   std::cout << std::endl;
  // }

  float *planes_device;
  cudaMalloc((void **)&planes_device, num_planes * 4 * sizeof(float));
  cudaMemcpy(planes_device, planes, num_planes * 4 * sizeof(float), cudaMemcpyHostToDevice);

  // debug
  // cudaPrint<< <1, 1>> >(planes_device, num_planes);

  // std::cout << "P: ";
  // for (int i=0; i < 12; ++i) {
  //   std::cout << P[i] << ", ";
  // }
  // std::cout << std::endl;

  // cudaTestUnproject<< <1, 1>> >(planes_device, 0, 0, 0, 
  //   P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11]);

  unprojectLabelMapKernel << < gridDim, blockDim >>
      > (bestPlanesBuf, planes_device, num_planes,
          xMapBuf, yMapBuf, zMapBuf, absurdXYZ,
          P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11]);
  PSL_CUDA_CHECK_ERROR

  cudaFree(planes_device);
  PSL_CUDA_CHECK_ERROR
}

}    // namespace PSL_CUDA

