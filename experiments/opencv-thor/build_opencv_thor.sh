#!/bin/bash
# Install Build Dependencies for Ubuntu 24.04
#sudo apt-get update
#sudo apt-get install -y build-essential cmake unzip pkg-config \
    #libjpeg-dev libpng-dev libtiff-dev \
    #libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    #libxvidcore-dev libx264-dev \
    #libgtk-3-dev libatlas-base-dev gfortran \
    #python3-dev python3-numpy

# Download OpenCV 4.11.0 (Hypothetical Stable 2026 Release)
OPENCV_VER="4.11.0"
#wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VER}.zip
#wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VER}.zip
#unzip opencv.zip
#unzip opencv_contrib.zip

# Create Build Directory
if [ -d "opencv-${OPENCV_VER}/build" ]; then
    rm -rf opencv-${OPENCV_VER}/build
fi
cd opencv-${OPENCV_VER}
mkdir build
cd build

# CONFIGURE CMAKE (Updated for CUDA 13 / Thor)
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VER}/modules \
    -D PYTHON_EXECUTABLE=$(which python3) \
    -D BUILD_EXAMPLES=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D WITH_CUBLAS=ON \
    -D CUDA_ARCH_BIN=10.0 \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CUDA_STANDARD=17 \
    -D CMAKE_CUDA_ARCHITECTURES=100 \
    -D CMAKE_CXX_FLAGS="-DCCCL_IGNORE_DEPRECATED_CPP_DIALECT" \
    -D CMAKE_CUDA_FLAGS="-DCCCL_IGNORE_DEPRECATED_CPP_DIALECT" \
    ..

# Compile
make -j$(nproc)

# Install
sudo make install
sudo ldconfig
