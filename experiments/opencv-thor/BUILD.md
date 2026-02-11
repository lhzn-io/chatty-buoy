# Building OpenCV 4.11.0 on Jetson Thor (JetPack 7.0 / CUDA 13.0)

This guide documents the steps and patches required to build OpenCV 4.11.0 with CUDA 13.0 support on NVIDIA Jetson Thor (Blackwell architecture).

## Prerequisites

- **Hardware**: Jetson AGX Thor (Blackwell `sm_100`)
- **JetPack**: 7.0+
- **CUDA**: 13.0
- **Compiler**: GCC 13+ (Supporting C++17/20)

## 1. Install Dependencies

Ensure the basic build tools and CUDA toolkit are installed:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    python3-dev python3-numpy \
    nvidia-cuda-toolkit
```

## 2. Prepare Source Code

Clone or download OpenCV 4.11.0 and OpenCV Contrib:

```bash
# Example setup structure
mkdir -p opencv-thor
cd opencv-thor
# (Download/Extract opencv-4.11.0 and opencv_contrib-4.11.0 here)
```

## 3. Apply Required Patches (CUDA 13.0 Compatibility)

Several patches are needed to handle C++17 requirements, deprecated CUDA APIs, and library conflicts in CUDA 13.0.

### A. Force C++17 Standard
**File**: `opencv-4.11.0/cmake/OpenCVDetectCUDA.cmake`
**Issue**: CUDA 13.0 requires C++17, but OpenCV defaults to C++14 for modern CUDA versions.
**Fix**:
Change the default standard to C++17:
```cmake
if(CUDA_VERSION VERSION_LESS "11.0")
  list(APPEND CUDA_NVCC_FLAGS "--std=c++11")
else()
  list(APPEND CUDA_NVCC_FLAGS "--std=c++17") # Change to c++17
endif()
```

### B. Fix `NppStreamContext` (Missing API)
**File**: `opencv-4.11.0/modules/core/include/opencv2/core/private.cuda.hpp`
**Issue**: `nppGetStreamContext` was removed in CUDA 13.0.
**Fix**: Manually initialize the context using CUDA Runtime APIs.
```cpp
// Add includes
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Inside NppStreamHandler constructor:
inline NppStreamHandler::NppStreamHandler(cudaStream_t stream) {
    // OLD: m_context = nppGetStreamContext();
    // NEW:
    cudaStreamGetFlags(stream, &m_context.nCudaStreamFlags);
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    m_context.nCudaDeviceId = device_id;
    m_context.nMultiProcessorCount = props.multiProcessorCount;
    m_context.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    m_context.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
    m_context.nSharedMemPerBlock = props.sharedMemPerBlock;
    m_context.hCudaStream = stream;
}
```

### C. Handle Deprecated `cudaDeviceProp` Fields
**File**: `opencv-4.11.0/modules/core/src/cuda_info.cpp`
**Issue**: Fields like `memPitch` are deprecated/removed.
**Fix**: Guard access with version checks.
```cpp
#if CUDART_VERSION < 12000
    // access prop.memPitch, etc.
#endif
```

### D. disable WaveletMatrix Optimization in `cudafilters`
**File**: `opencv_contrib-4.11.0/modules/cudafilters/src/cuda/wavelet_matrix_feature_support_checks.h`
**Issue**: This feature pulls in `libcu++` headers that cause `std::array` conflicts in the Thor environment.
**Fix**: Explicitly disable the feature.
```cpp
// Add at the end of the file, before the last #endif
#undef __OPENCV_USE_WAVELET_MATRIX_FOR_MEDIAN_FILTER_CUDA__
```

### E. Fix FP16 Casting in `cuda4dnn`
**Files**:
1. `opencv-4.11.0/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp`
2. `opencv-4.11.0/modules/dnn/src/cuda4dnn/primitives/region.hpp`
**Issue**: Comparison of `__half` types related to weights fails compilation.
**Fix**: Begin explicit casts to float.
```cpp
// normalize_bbox.hpp
if ((float)weight != 1.0f) { ... }

// region.hpp
if ((float)nms_iou_threshold > 0.0f) { ... }
```

### F. Fix `thrust::not1` Deprecation in `videostab`
**File**: `opencv_contrib-4.11.0/modules/videostab/src/cuda/global_motion.cu`
**Issue**: `thrust::not1` is removed in C++17.
**Fix**: Use `thrust::logical_not`.
```cpp
// Replace: thrust::not1(thrust::identity<uchar>())
// With:    thrust::logical_not<uchar>()
```

## 4. Build and Install

Use the following CMake configuration (adjusted for `sm_100` architecture):

```bash
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.11.0/modules \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D WITH_CUBLAS=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CUDA_STANDARD=17 \
    -D CUDA_ARCH_BIN=10.0 \
    -D CMAKE_CUDA_ARCHITECTURES=100 \
    -D CMAKE_CXX_FLAGS="-DCCCL_IGNORE_DEPRECATED_CPP_DIALECT" \
    -D CMAKE_CUDA_FLAGS="-DCCCL_IGNORE_DEPRECATED_CPP_DIALECT" \
    ..

make -j$(nproc)
sudo make install
sudo ldconfig
```

## 5. Verification

Verify the installation in Python:

```bash
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}'); print(f'CUDA Devices: {cv2.cuda.getCudaEnabledDeviceCount()}');"
```
