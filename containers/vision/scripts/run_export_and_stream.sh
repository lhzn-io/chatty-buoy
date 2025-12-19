#!/bin/bash
# run_export_and_stream.sh

set -e # Exit immediately if any command fails

MODEL_NAME="yolo11n.pt"
MODEL_PATH="/app/${MODEL_NAME}"
CONFIG_FILE="/app/config/deepstream_app_config.txt"

# The Ultralytics command generates 'yolo11n.engine' in the current working directory (/app)
ENGINE_FILE="/app/yolo11n.engine"

echo "--- 1. Checking for existing TensorRT Engine ---"
if [ ! -f "$ENGINE_FILE" ]; then
    echo "Engine ($ENGINE_FILE) not found."

    # Check if the PyTorch weights exist (mounted from host)
    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: Model weights ($MODEL_NAME) not found in /app/"
        echo "Please place your .pt file in the project folder and try again."
        exit 1
    fi

    echo "Starting direct TensorRT engine build via Ultralytics..."

    # 1. Use the official Ultralytics command to create the ONNX file
    yolo export model=${MODEL_PATH} format=onnx imgsz=640 opset=17

    # 2. Use the native trtexec tool to convert ONNX to TensorRT engine
    echo "Starting trtexec conversion..."
    /usr/src/tensorrt/bin/trtexec \
        --onnx=/app/yolo11n.onnx \
        --saveEngine=/app/yolo11n.engine \
        --memPoolSize=workspace:4096

    # Check if the engine was created successfully
    if [ ! -f "$ENGINE_FILE" ]; then
        echo "ERROR: Ultralytics failed to create the TensorRT engine. Check yolo export logs."
        exit 1
    fi
    echo "TensorRT engine generation complete."

else
    echo "TensorRT Engine ($ENGINE_FILE) found. Skipping conversion."
fi

echo "--- 2. Finalizing setup and running DeepStream pipeline ---"

# Install your local custom code (must be done after mount)
# pip install -e /lhzn-io/kanoa-mlops

# Run the DeepStream pipeline using the generated/existing engine
# DeepStream's NvInfer plugin will now try to parse the official Ultralytics engine output.
/opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-app -c ${CONFIG_FILE}

# Fallback block remains the same for debugging if deepstream-app fails
if [ $? -ne 0 ]; then
    echo "DeepStream application failed! Container will remain running for debugging."
    tail -f /dev/null
fi
