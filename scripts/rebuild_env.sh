#!/bin/bash
set -e

ENV_NAME="chatty-buoy"
PYTHON_VERSION="3.11"

# Add a check for a -y flag to bypass the confirmation prompt
if [[ ! " $@ " =~ " -y " ]]; then
    echo "Caution: This will remove and recreate the '$ENV_NAME' environment."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🗑️ Removing existing environment..."
micromamba remove -n $ENV_NAME --all -y || true

echo "📦 Creating environment from environment.yml..."
micromamba create -f environment.yml -y

echo "Activating environment..."
eval "$(micromamba shell hook --shell=bash)"
micromamba activate $ENV_NAME

echo "Installing PyTorch from NVIDIA..."
wget https://developer.download.nvidia.com/compute/redist/jp/v6.0/pytorch/torch-2.3.0-cp311-cp311-linux_aarch64.whl
pip install torch-2.3.0-cp311-cp311-linux_aarch64.whl

echo "🔗 Performing Environment Linking (Fixing SBSA/Thor Library Paths)..."
# Link the hidden pip libraries into the main environment library path
ENV_PREFIX=$(micromamba env list | grep $ENV_NAME | awk '{print $NF}')

if [ -z "$ENV_PREFIX" ]; then
    echo "❌ Error: Could not determine environment prefix."
    exit 1
fi

echo "   Prefix: $ENV_PREFIX"

# libcudss (CUDA Sparse Solvers)
if [ -f "$ENV_PREFIX/lib/python$PYTHON_VERSION/site-packages/nvidia/cu13/lib/libcudss.so.0" ]; then
    ln -sf "$ENV_PREFIX/lib/python$PYTHON_VERSION/site-packages/nvidia/cu13/lib/libcudss.so.0" "$ENV_PREFIX/lib/libcudss.so.0"
    echo "   [✓] libcudss linked"
else
    echo "   [!] libcudss not found in expected pip location"
fi

# libnvpl (NVIDIA Performance Libraries)
NVPL_LIB_DIR="$ENV_PREFIX/lib/python$PYTHON_VERSION/site-packages/nvpl/lib"
if [ -d "$NVPL_LIB_DIR" ]; then
    for lib in "$NVPL_LIB_DIR"/*.so.0; do
        ln -sf "$lib" "$ENV_PREFIX/lib/$(basename $lib)"
    done
    echo "   [✓] NVPL libraries linked"
else
    echo "   [!] NVPL lib directory not found"
fi

echo "✅ Setup Complete. Run 'micromamba activate $ENV_NAME' to start."