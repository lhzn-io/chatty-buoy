# Current Status: Async Cascade, Planning Live & Sentinel Vision Active

**Date**: 2026-05-02
**Phase**: Phase 4 In Progress (Vision Integration)

## Architecture State
We have successfully implemented the **"Async Cascade" Architecture (v2.0)** alongside **Sentinel Vision**:
*   **L0 Gatekeeper**: Filters noise and routes "Planning" requests.
*   **L1 Front-End**: `Gemma-4-E4B-Multimodal` handles fluid chat (<100ms) and invokes tools.
*   **L2 Dispatcher**: Handled by L1 function calling via vLLM.
*   **L3 Cortex**: Offline (disabled to free VRAM). Deep reasoning is now offloaded to `Cosmos-Reason2-2B` for Vision.
*   **Sentinel Vision**: `watchstander.py` continuously monitors the camera feed using OpenCV motion detection and invokes `Cosmos-Reason2-2B` (via vLLM) to publish VLM scene summaries to Redis out-of-band.

## Recent Achievements
*   ✅ **Planning Mode**: "Plan a mission..." requests bypass the chat loop and trigger deep reasoning.
*   ✅ **Cosmos-Reason2-2B Onboarding**: Added setup scripts to fetch the FP8 static KV8 model payload from NGC.
*   ✅ **Watchstander Sentinel Mode**: Upgraded the standard YOLO spotter loop to hold a rolling buffer of frames. Large motion gradients trigger the `Cosmos-Reason-2B` endpoint asynchronously. 
*   ✅ **Tool Registry Expansion**: Added `get_scene_summary` and `analyze_video_vqa` tools to enable agentic video comprehension.

## Next Steps
1.  **Hardware Test**: Validate the complete ensemble running correctly on Jetson Thor under heavy camera load.
2.  **Optimize Motion Thresholds**: Fine-tune OpenCV `absdiff` logic.
3.  **Phase 4 Completion**: Merge vision event logic natively into the chat loop dispatcher.

## Troubleshooting Tools
*   **Live VLM WebUI**: If `cosmos-vision` or the camera pipeline fails during testing, the `NVIDIA-AI-IOT/live-vlm-webui` package is available for manual WebRTC camera feeds and prompt testing.
    *   Installation: `pipx install live-vlm-webui`
    *   Command: `live-vlm-webui --api-base http://localhost:8010/v1 --model nvidia/cosmos-reason2-2b-fp8`

See [Roadmap](planning/roadmap.md) for details.
