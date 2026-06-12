# NVFP4 Migration Plan (Jetson Thor / Blackwell)

**Target Engines:** vLLM / TensorRT-LLM natively
**Target Models:** Nemotron-3 (Nano/Super), Gemma-4 (IT/A4B)
**Hardware:** Jetson Thor (Blackwell Architecture)

This document outlines the phased migration of the `cortex-service` and `front-end-service` to NVIDIA's native **NVFP4** 4-bit floating-point format to maximize token throughput and minimize memory bandwidth bottlenecks on the Jetson Thor's unified memory architecture.

## Phase 1: Environment & Dependency Upgrades
Before touching the models, the host environment must support Blackwell's native 4-bit Tensor Cores.
1. **Host OS / JetPack:** Ensure Jetson Thor is running the latest JetPack/L4T release that officially exposes Blackwell CUDA capabilities (`sm_100`/`sm_120` depending on the exact Thor die).
2. **Docker Stack:** Ensure the `nvidia` default runtime passes the correct CUDA environment to the containers.
3. **vLLM / TensorRT-LLM Base:** In `docker-compose.yaml` (and specifically `containers/cortex/Dockerfile`), we must migrate to a base image that supports NVFP4 (e.g., the bleeding-edge NVIDIA Triton Inference Server image or the latest `vllm/vllm-openai` image compiled for Thor architecture).

## Phase 2: Procuring Pre-Quantized NVFP4 Checkpoints
To bypass the extremely time-consuming ModelOpt (AMMO) calibration process, we will utilize pre-quantized NVFP4 checkpoints provided by the community and official NVIDIA channels on Hugging Face.

**Verified NVFP4 Targets:**
* **Nemotron-3 Series:**
  * `nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4` (Lightweight/Fast)
  * `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` (Balanced Cortex)
  * `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (Heavy Reasoning)
* **Gemma-4 Series:**
  * `nvidia/Gemma-4-31B-IT-NVFP4`
  * `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`

> *Note: We are explicitly eschewing Llama-3 variants in favor of natively scaled Gemma-4 and Nemotron-3 block-quantized weights.*

## Phase 3: TensorRT Engine Compilation
NVFP4 requires ahead-of-time compilation tailored strictly to our Jetson Thor constraints.
1. **TRT-LLM Build:** Use the `trtllm-build` CLI inside the `cortex-service` build pipeline to convert the downloaded NVFP4 checkpoint into a TensorRT Engine.
2. **Compilation Parameters:** Explicitly set `--use_nvfp4` and optimize for batch size 1 (since this is an on-device orchestrator agent).
   ```bash
   trtllm-build --checkpoint_dir ./models/downloaded/gemma-4-nvfp4-ckpt \
                --output_dir ./models/cortex/engine \
                --use_nvfp4 \
                --max_batch_size 1 \
                --max_input_len 4096
   ```

## Phase 4: `cortex-service` Integration (`docker-compose.yaml`)
Since `start_cortex.sh` is now obsolete, all orchestration happens via Docker Compose.
1. **vLLM Backend Swap:** `cortex-service` must be configured to use the TensorRT-LLM backend rather than the default PyTorch/XFormers backend.
2. **Compose Configuration Update:** In `docker-compose.yaml`, update the `cortex-service` arguments:
   ```yaml
   services:
     cortex-service:
       image: vllm/vllm-openai:latest-trtllm
       command: >
         --model /app/models/cortex/engine
         --tensor-parallel-size 1
         --gpu-memory-utilization 0.9
   ```

## Phase 5: `front-end-service` Adjustments
The `front-end-service` generally communicates with `cortex-service` via OpenAI-compatible REST APIs. Network/timeout handling must be adjusted for the massive leap in speed.
1. **TTFT Expectations:** NVFP4 drastically reduces memory bandwidth, making Time-To-First-Token (TTFT) extremely fast. We will tighten timeout parameters in `front-end-service` (e.g., dropping max waits from 2000ms to 500ms).
2. **Streaming Generation:** Ensure `front-end-service` processes SSE (Server-Sent Events) synchronously fast enough. NVFP4 generation on a Thor will yield hundreds of tokens per second. If the frontend async loop isn't highly optimized, the LLM will outpace the frontend's ability to render/route to the TTS engine.