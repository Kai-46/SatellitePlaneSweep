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

#ifndef DEVICEBUFFER_H
#define DEVICEBUFFER_H

#include <string>
#include <cuda_runtime.h>

#include "psl_cudaBase/cudaCommon.cuh"
#include "psl_cudaBase/deviceImage.cuh"

namespace PSL_CUDA {

// 2D grid data on GPU memory, with each grid cell is a T
template<typename T>
class DeviceBuffer {
 public:
  __host__ __device__ DeviceBuffer() {
    _addr = (T *) 0;
    _width = 0;
    _height = 0;
    _pitch = 0;
  }

  __host__ __device__ ~DeviceBuffer() {}

  // host and device functions
  inline __host__ __device__ T *getAddr() const {
    return _addr;
  }

  inline __host__ __device__ int getWidth() const {
    return _width;
  }

  inline __host__ __device__ int getHeight() const {
    return _height;
  }

  inline __host__ __device__ int getPitch() const {
    return (int) _pitch;
  }

  inline __host__ __device__ bool isAllocated() const {
    return _addr != 0;
  }

  // host only functions
  void allocatePitched(int width, int height) {
    PSL_CUDA_CHECKED_CALL(cudaMallocPitch(&_addr, &_pitch, width * sizeof(T), height);)
    _width = width;
    _height = height;
  }

  void deallocate() {
    if (this->isAllocated()) {
      PSL_CUDA_CHECKED_CALL(cudaFree((void *) _addr);)
    }
    _addr = (T *) 0;
    _width = 0;
    _height = 0;
    _pitch = 0;
  }

  void clear(T value);

  // transfer between host and device
  void upload(T *dataPtr) {
    // always assume host memory is not padded along the column direction
    PSL_CUDA_CHECKED_CALL(cudaMemcpy2D(_addr,
                                       _pitch,
                                       dataPtr,
                                       _width * sizeof(T),
                                       _width * sizeof(T),
                                       _height,
                                       cudaMemcpyHostToDevice);)
  }

  void download(T *dstPtr) {
    // always assume host memory is not padded along the column direction
    PSL_CUDA_CHECKED_CALL(cudaMemcpy2D(dstPtr,
                                       sizeof(T) * _width,
                                       _addr,
                                       _pitch,
                                       sizeof(T) * _width,
                                       _height,
                                       cudaMemcpyDeviceToHost);)
  }

  // device only functions
  inline __device__ T &operator()(unsigned int x, unsigned int y) {
    return *((T *) ((char *) _addr + y * _pitch) + x);
  }

 private:
  T *_addr;
  int _width;
  int _height;
  size_t _pitch;
};

void copyImageToBuffer(const DeviceImage &srcImg, DeviceBuffer<float> &destBuf);

}    // namespace PSL_CUDA

#endif // CUDADEVICEBUFFER_H
