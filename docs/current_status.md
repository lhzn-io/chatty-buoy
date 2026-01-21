# Current System Status: PersonaPlex + FunctionGemma

**Last Updated**: 2026-01-21

## 1. Architecture Overview

The system currently runs two concurrent Large Language Models (LLMs) on a single Jetson Thor device to provide an audio-enabled, tool-aware maritime assistant ("Skipper").

1.  **Conversational Engine (PersonaPlex)**:
    *   **Role**: Handles personality, general knowledge, and conversational flow.
    *   **Model**: `mistralai/Mistral-7B-Instruct-v0.3` (masquerading as `nvidia/personaplex-7b-v1` for client compatibility).
    *   **Port**: 8000
    *   **Memory**: Constrained to ~0.25 utilization to fit alongside Gemma.
    *   **Status**: ✅ ACTIVE

2.  **Tooling Engine (FunctionGemma)**:
    *   **Role**: Dedicated function calling and structured reasoning.
    *   **Model**: `google/functiongemma-270m-it` (Lightweight variant).
    *   **Port**: 8001
    *   **Memory**: Low footprint (0.05 utilization).
    *   **Status**: ✅ ACTIVE (Container running) but ⚠️ API Issues (400 Bad Request).

3.  **Client Application**:
    *   **Script**: `examples/personaplex_system_chat_audio.py`
    *   **Orchestrator**: `chatty_buoy.crew.crew_member.CrewMember`
    *   **Environment**: `micromamba run -n chatty-buoy`

## 2. Hardware Context

*   **Device**: Jetson Thor
*   **Memory**: Unified Memory (~122GB Total)
*   **Constraints**: Running two LLMs simultaneously led to OOM errors with the 12B model. Switched to 270M for FunctionGemma to alleviate pressure.

## 3. Current Configuration Details

### Docker Services

**PersonaPlex (Port 8000)**
*   File: `docker/vllm/docker-compose.personaplex.yml`
*   Command: `vllm serve --model mistralai/Mistral-7B-Instruct-v0.3 ... --gpu-memory-utilization 0.25`

**FunctionGemma (Port 8001)**
*   File: `docker/vllm/docker-compose.gemma-function.yml`
*   Command: `vllm serve --model google/functiongemma-270m-it ... --gpu-memory-utilization 0.05`

### Python Environment
*   **Env Name**: `chatty-buoy`
*   **Key Libs**: `sounddevice`, `httpx`, `vllm` (client-side?)

## 4. Known Issues & Blockers

### 🔴 Blocker: Native Audio Integration Missing
The client script (`examples/personaplex_system_chat_audio.py`) works for **Text** but is silent for **Voice** because we lack a valid TTS/Audio backend.
*   **User Requirement**: "Pure audio out of Personaplex" (no local synth like espeak).
*   **Current State**:
    *   Audio Hardware verified: Jabra Speak 710 @ 16kHz (Input/Output loopback verified).
    *   Text flow is working. "Skipper" generates text responses.
    *   **The Problem**: We are currently just printing the text. We need to implement the actual audio streaming API from the `personaplex` container (likely the Moshi integration in `../personaplex`) instead of using a local TTS.

### ✅ Resolved: Tool Dispatch Failure
The 400 Bad Request error from FunctionGemma is fixed.
*   **Fix**: Switched from native `tool_choice="auto"` (which vLLM/FunctionGemma struggled with) to **Manual Few-Shot Prompting**.
*   **Mechanism**: `crew_member.py` now sends a system prompt telling the model to output a specific JSON structure.
*   **Status**: Working.

### ✅ Resolved: Hardware Specifications Hallucination
The model was hallucinating specs (e.g., claiming to be on a generic Linux box).
*   **Fix**: Implemented real hardware introspection in `_tool_get_system_info` (reading `/proc/device-tree/model`, `nvidia-smi`, `psutil`).
*   **Status**: Working.

## 5. Deployment Instructions

1.  **Start Services**:
    ```bash
    # Start PersonaPlex
    docker compose -f docker/vllm/docker-compose.personaplex.yml up -d
    
    # Start FunctionGemma
    docker compose -f docker/vllm/docker-compose.gemma-function.yml up -d
    ```

2.  **Verify Services**:
    ```bash
    curl http://localhost:8000/health
    curl http://localhost:8001/health
    ```

3.  **Run Client**:
    ```bash
    micromamba run -n chatty-buoy python examples/personaplex_system_chat_audio.py
    ```

## 6. Document Consolidation (Archived)
The following files have been superseded by the current implementation and this status doc:
*   `docs/deployment-configurations.md`
*   `docs/setup-streaming-object-detection.md`
*   Various untracked `.md` files in `docs/` relating to initial setup ideas.
