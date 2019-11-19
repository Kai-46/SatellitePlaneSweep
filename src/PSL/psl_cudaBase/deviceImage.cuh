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

#ifndef DEVICEIMAGE_H
#define DEVICEIMAGE_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "psl_cudaBase/cudaCommon.cuh"

namespace PSL_CUDA {

// 3D grid data on GPU memory, with each grid cell is a byte
// used for storing gray-scale or BGR images
// memory layout is (channel, column, row), with channel changing fastest, row slowest
class DeviceImage {
 public:
  // cuda would implicitly instantiate an object on device when it's passed to device code
  __host__ __device__ DeviceImage() {
    _addr = 0;
    _width = 0;
    _height = 0;
    _numChannels = 0;
    _pitch = 0;
  }

  __host__ __device__ ~DeviceImage() {}

  // host and device functions
  inline __host__ __device__ unsigned char *getAddr() const {
    return _addr;
  }

  inline __host__ __device__ int getWidth() const {
    return _width;
  }

  inline __host__ __device__ int getHeight() const {
    return _height;
  }

  inline __host__ __device__ int getNumChannels() const {
    return _numChannels;
  }

  inline __host__ __device__ int getPitch() const {
    return (int) _pitch;
  }

  inline __host__ __device__ bool isAllocated() const {
    return _addr != 0;
  }

  // host only functions
  void allocatePitched(int width, int height, int numChannels) {
    PSL_CUDA_CHECKED_CALL(cudaMallocPitch(&_addr, &_pitch, width * numChannels, height);)
    _width = width;
    _height = height;
    _numChannels = numChannels;
  }

  void deallocate() {
    if (this->isAllocated()) {
      PSL_CUDA_CHECKED_CALL(cudaFree((void *) _addr);)
    }
    _addr = 0;
    _width = 0;
    _height = 0;
    _numChannels = 0;
    _pitch = 0;
  }

  void allocatePitchedAndUpload(const cv::Mat &img) {
    if (img.type() != CV_8UC3 && img.type() != CV_8UC1) {
      PSL_THROW_EXCEPTION ("Only BGR and Grayscale supported!")
    }

    allocatePitched(img.cols, img.rows, img.channels());
    // opencv mat are stored in the way (BGR) (BGR) (BGR) ..., row-major
    PSL_CUDA_CHECKED_CALL(cudaMemcpy2D(_addr,
                                       _pitch,
                                       img.data,
                                       img.step,
                                       _width * _numChannels,
                                       _height,
                                       cudaMemcpyHostToDevice);)
  }

  cv::Mat download() const {
    if (_numChannels == 1) {
      cv::Mat img(_height, _width, CV_8UC1);
      PSL_CUDA_CHECKED_CALL(cudaMemcpy2D(img.data,
                                         img.step,
                                         _addr,
                                         _pitch,
                                         _width * _numChannels,
                                         _height,
                                         cudaMemcpyDeviceToHost);)
      return img;
    } else {
      cv::Mat img(_height, _width, CV_8UC3);
      PSL_CUDA_CHECKED_CALL(cudaMemcpy2D(img.data,
                                         img.step,
                                         _addr,
                                         _pitch,
                                         _width * _numChannels,
                                         _height,
                                         cudaMemcpyDeviceToHost);)
      return img;
    }
  }

  // device only functions
  inline __device__ unsigned char &operator()(unsigned int x, unsigned int y) {
    return *(_addr + y * _pitch + x);
  }

  inline __device__ unsigned char &operator()(unsigned int x, unsigned int y, unsigned int c) {
    return *(_addr + y * _pitch + x * _numChannels + c);
  }

 private:
  // accessible on both host and device
  unsigned char *_addr;
  int _width;
  int _height;
  int _numChannels;
  size_t _pitch;
};

} // namespace PSL_CUDA


#endif // CUDADEVICEIMAGE_H
