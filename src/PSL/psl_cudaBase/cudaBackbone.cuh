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
