sudo apt-get update && sudo apt-get upgrade
sudo apt-get install -y build-essential cmake unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y python3-dev

WORK_DIR = $1
if [ ! -d $WORK_DIR ]; then
    mkdir $WORK_DIR
fi
cd $WORK_DIR

git clone https://github.com/opencv/opencv_contrib.git
# checkout 3.4.5
cd opencv_contrib && git checkout 3.4.5 && cd ..

git clone https://github.com/opencv/opencv.git
# checkout 3.4.5
cd opencv && git checkout 3.4.5

if [ ! -d build ]; then
    mkdir build
fi
cd build && rm -r *

cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D ENABLE_PRECOMPILED_HEADERS=OFF \
	-D PYTHON_EXECUTABLE=/usr/bin/python3 \
	-D BUILD_EXAMPLES=ON .. \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules

make -j8 && sudo make install
