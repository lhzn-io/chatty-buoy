# 🚀 Project Archive: DeepStream YOLOv11 Engine Deployment on NVIDIA Thor

## 1. Project Description
This project aims to deploy an **Ultralytics YOLOv11n** object detection model onto an **NVIDIA Thor/Blackwell** platform running the **DeepStream 8.0 SDK** within a Docker container. The primary objective is to create a highly optimized TensorRT engine file (`yolo11n.engine`) for high-performance, real-time video stream analytics.

## 2. Key Findings & Resolved Issues

The deployment process involved overcoming several complex, version-specific incompatibilities across the PyTorch, Docker, and NVIDIA software stack.

| # | Issue/Error Received | Root Cause | Resolution | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `UnpicklingError: unsupported pickle protocol: 5` | PyTorch 2.0+ security measure prevents loading older weights by default. | **Dockerfile Patch:** Added `weights_only=False` to `torch.load` call. | ✅ Fixed |
| 2 | `AttributeError: 'float' object has no attribute 'node'` | PyTorch's new FX/Dynamo ONNX exporter was incompatible with YOLO's `.item()` operation. | Bypassed the unstable exporter by switching the flow to use the **Ultralytics ONNX exporter** instead of the community DeepStream one. | ✅ Fixed |
| 3 | `ValueError: Invalid CUDA 'device=0' requested.` | The `pip install ultralytics` command downgraded the base image's CUDA-enabled PyTorch to a generic **CPU-only** wheel (`torch-2.9.1+cpu`). | **Manual Fix:** Uninstalled CPU-only PyTorch and reinstalled the correct **CUDA 13.0 wheel** using `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`. | ✅ Fixed |
| 4 | `ModuleNotFoundError: No module named 'tensorrt'` | Ultralytics exporter failed to install necessary proprietary Python bindings (`nvidia-tensorrt`) due to missing system tools (`cmake`) and proprietary package index requirements. | **Abandoned Ultralytics Direct Export:** Switched to the robust, native **`trtexec`** command-line tool. | ✅ Fixed |
| 5 | `Error[10]: Could not find any implementation for node {ForeignNode[...]}` | The ONNX file exported by PyTorch was too complex, containing redundant or unsupported `Cast` and `Reshape` nodes, causing the TensorRT compiler to crash. | **Path Change:** Switched the `yolo export` command to a simplified, **FP32, Opset 17** format to generate a cleaner ONNX graph for `trtexec` to consume. | 🚧 Current |
| 6 | `[E] Static model does not take explicit shapes...` | Conflicting flags were used during the `trtexec` compilation (telling it the shape was dynamic while the ONNX file was static). | **`trtexec` Syntax Simplification:** Removed the unnecessary `--min/opt/maxShapes` flags. | ✅ Fixed |

## 3. Current Status and Path Forward

We are currently executing the final, clean, and native TensorRT conversion command.

### **Current Step: TRT Engine Building**
* **Input:** Cleaned-up `yolo11n.onnx` (exported with simplified settings).
* **Tool:** Native system executable `trtexec` (TensorRT Command-Line Tool).
* **Status:** The build process is currently running, which is a highly CPU-intensive operation that takes several minutes.

### **Final Production Pipeline (Post-Engine Build)**

The final, reliable conversion flow, which will be automated in `run_export_and_stream.sh`, is:

1.  **Check for Engine:** If `/app/yolo11n.engine` exists, skip to step 4.
2.  **Generate Clean ONNX:**
    ```bash
    yolo export model=/app/yolo11n.pt format=onnx imgsz=640 opset=17
    ```
3.  **Compile TRT Engine:**
    ```bash
    /usr/src/tensorrt/bin/trtexec \
        --onnx=/app/yolo11n.onnx \
        --saveEngine=/app/yolo11n.engine \
        --memPoolSize=workspace:4096MiB \
        --fp16
    ```
4.  **Launch DeepStream:**
    ```bash
    /opt/nvidia/deepstream/deepstream-8.0/bin/deepstream-app -c /app/deepstream_app_config.txt
    ```

Once the current `trtexec` command completes, the final engine will be ready for DeepStream inference.