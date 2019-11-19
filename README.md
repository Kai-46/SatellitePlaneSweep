Prerequisites
--------------------------------

The code is written in C++ and CUDA. To use it you need a CUDA compatible Nvidia GPU.
The following libraries are required:

GCC Toolchain on Linux </br>
CMake </br>
Nvidia CUDA </br>
Boost (system filesystem program_options) </br>
OpenCV </br>
Eigen3 </br>

There is a helper script 'install_opencv.sh' to assist installing opencv on your computer.

Instructions Linux
-----------------------------

mkdir build </br>
cd build </br>
cmake .. </br>
make </br>

General Usage
-----------------------------
```{r, engine='bash', count_lines}
SimplePlaneSweepStereo/build/bin/simplePinholePlaneSweep \
    --imageFolder {} \
    --imageList {} \
    --refImgId {} \
    --outputFolder {} \
    --matchingCost census  \
    --windowRadius {} \
    --nX {} --nY {} --nZ {} \
    --firstD {} --lastD {} --numPlanes {} \
    --filterCostVolume 1 --guidedFilterRadius 9 --guidedFilterEps 0.04 \
    --saveCostVolume {} \
    --debug 0 \
    --saveBest 0 --filter 0 --filterThres {} \
    --saveXYZMap {} \
    --savePointCloud {}
```

Brief explanation of the options
* imageFolder: directory that contains your reference and source images
* imageList: a .txt file with each line being {img_id} {img_name} {fx fy cx cy s qw qx qy qz tx ty tz}; the number of lines should be equal to the number of reference and source images; {img_id} is an integer identifier, while {img_name} is the file name for the image inside {imageFolder}; {fx fy cx cy s qw qx qy qz tx ty tz} are the camera intrinsics and extrinsics
* refImgId: specify which image in the {imageList} you would like to be the reference image; the other images inside {imageList} will automatically become the source images
* outputFolder: where to save the program's output
* nX, nY, nZ: they jointly define the normal direction of the sweeping planes in scene coordinate frame
* firstD, lastD: constants of the first and last sweep plane, respectively
* numPlanes: number of sweeping planes to be used
* filterCostVolume: whether to filter each slice of the cost volume with a guided-filter
* guidedFilterRadius, guidedFilterEps: parameters of the guided-filter
* saveCostVolume: whether to save the cost volume
* debug: if enabled, the program will output visualizations of the cost volumes and the warped images
* saveBest, filter, filterThres: whether to save the best plane hypothesis; before saving, an optional step filtering out the best plane hypothesis that have a cost above {filterThres} can be performed.
* saveXYZMap: whether to save the XYZ map, for which each pixel is the 3D point (X,Y,Z) in scene coordinate frame
* savePointCloud: whether to save the point cloud

Interfacing with Python
-----------------------------
scripts/binary_grid_io.py can be used to read the output files of the C++ program as numpy array.
