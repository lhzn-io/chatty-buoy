# Use the official DeepStream 9.0 Triton image (Multi-arch for Thor)
FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

# 1. Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git build-essential libglib2.0-dev libgstrtspserver-1.0-dev \
    libjansson-dev libyaml-cpp-dev libopencv-dev wget xz-utils \
    python3-pip python3-setuptools python3-tk kmod && \
    rm -rf /var/lib/apt/lists/*

# Install static ffmpeg to avoid shared library conflicts with DeepStream's libav
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz && \
    tar xvf ffmpeg-release-arm64-static.tar.xz && \
    cp /ffmpeg-*-static/ffmpeg /usr/local/bin/ && \
    rm -rf ffmpeg-*

# 2. Install Ultralytics and Flask for web streaming
RUN pip3 install --no-cache-dir ultralytics onnx onnxscript jetson-stats onnxslim onnxruntime flask

# FIX: Reinstall PyTorch explicitly to ensure the CUDA-enabled version is active.
# This often overwrites a CPU-only version installed by a dependency.
RUN pip3 install --no-cache-dir torch torchvision --upgrade
# Note: Since you are using the NVIDIA DeepStream base image, PyTorch should default
# to the correct CUDA wheel that matches the image's libraries.

# 3. Setup DeepStream-Yolo for YOLOv11 support
WORKDIR /opt/nvidia/deepstream/deepstream-8.0/sources
RUN git clone https://github.com/marcoslucianops/DeepStream-Yolo.git

# 4. Compile the Custom BBox Parser for Blackwell (CUDA 13.0)
WORKDIR /opt/nvidia/deepstream/deepstream-8.0/sources/DeepStream-Yolo
RUN export CUDA_VER=13.0 && \
    make -C nvdsinfer_custom_impl_Yolo clean && \
    make -C nvdsinfer_custom_impl_Yolo

WORKDIR /app
