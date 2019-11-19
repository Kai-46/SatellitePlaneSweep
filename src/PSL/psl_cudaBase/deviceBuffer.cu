#include "psl_cudaBase/deviceBuffer.cuh"

namespace PSL_CUDA {

// kernel code
// __global__ cannot use pass by reference
template<typename T>
__global__ void deviceBufferClearKernel(DeviceBuffer<T> devBuf, T value) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < devBuf.getWidth() && y < devBuf.getHeight()) {
    devBuf(x, y) = value;
  }
}

template<typename T>
void DeviceBuffer<T>::clear(T value) {
  dim3 gridDim(getNumTiles(_width, TILE_WIDTH), getNumTiles(_height, TILE_HEIGHT));
  dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
  deviceBufferClearKernel<T> << < gridDim, blockDim >> > (*this, value);
  PSL_CUDA_CHECK_ERROR
}

// explicit template instantiation
template
class DeviceBuffer<unsigned char>;
template
class DeviceBuffer<int>;
template
class DeviceBuffer<float>;

// kernel code
// __global__ cannot use pass by reference
__global__ void copyImageToBufferKernel(DeviceImage srcImg, DeviceBuffer<float> destBuf) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < srcImg.getWidth() && y < srcImg.getHeight()) {
    destBuf(x, y) = (float) srcImg(x, y) / 255.0;
  }
}

void copyImageToBuffer(const DeviceImage &srcImg, DeviceBuffer<float> &destBuf) {
  assert (srcImg.getNumChannels() == 1 && srcImg.getWidth() == destBuf.getWidth()
              && srcImg.getHeight() == destBuf.getHeight());

  dim3 gridDim(getNumTiles(srcImg.getWidth(), TILE_WIDTH), getNumTiles(srcImg.getHeight(), TILE_HEIGHT));
  dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
  copyImageToBufferKernel << < gridDim, blockDim >> > (srcImg, destBuf);
  PSL_CUDA_CHECK_ERROR
}

}    // namespace PSL_CUDA
