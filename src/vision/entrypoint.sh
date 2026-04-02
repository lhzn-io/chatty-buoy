#!/bin/bash
set -e

echo "--- Watchstander DeepStream Service Starting ---"

# Check if TensorRT engine exists, export if not
if [ ! -f "/app/models/yolo26n.engine" ]; then
    echo "Model (/app/models/yolo26n.engine) not found. Starting export process..."
    
    # 1. Export to ONNX (using native ultralytics CLI)
    if [ ! -f "/app/models/yolo26n.onnx" ]; then
        echo "Exporting PT -> ONNX using native CLI..."
        cd /app/models || exit 1
        yolo export model=yolo26n.pt format=onnx dynamic=False simplify=True device=cpu
        cd /app || exit 1
    fi
    
    # 2. Build Engine using trtexec (Native TensorRT tool)
    echo "Building ONNX -> Engine using trtexec (this may take a while)..."
    # --fp16 for performance on Jetson
    # --saveEngine=yolo26n.engine
    /usr/bin/trtexec --onnx=/app/models/yolo26n.onnx --saveEngine=/app/models/yolo26n.engine --fp16
else
    echo "Model found (/app/models/yolo26n.engine). Skipping export."
fi

echo "Starting GStreamer Pipeline..."
python3 watchstander_ds.py
