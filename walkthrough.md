# Walkthrough: State-Aware Watchstander Prompt Optimization

This update introduces state-awareness and targeted querying to the vision analysis pipeline. By maintaining the "last known state" and allowing specific user questions to reach the vision model, we reduce repetitive descriptions and improve the relevance of the vessel's situation reports.

## Changes Overview

### 1. Centralized Prompt Management
- **File**: [src/orchestrator/prompts.py](src/orchestrator/prompts.py)
- **Action**: Refactored hardcoded prompts from the Watchstander service into a centralized template file.
- **Improved Prompt**: Introduced `WATCHSTANDER_USER_PROMPT_TEMPLATE` which uses `{previous_state}` and `{user_query}` placeholders. It explicitly instructs the model to focus on **changes** and **anomalies**.

### 2. Intelligent Tooling Interface
- **File**: [src/orchestrator/agent_orchestrator.py](src/orchestrator/agent_orchestrator.py)
- **Action**: Enhanced the `check_camera_feed` tool.
- **Logic**: 
    - It now automatically retrieves the last analysis result from Redis to provide "context" to the next vision request.
    - It passes an optional `specific_query` parameter downstream.

### 3. State-Aware Vision Processing
- **File**: [src/watchstander/watchstander.py](src/watchstander/watchstander.py)
- **Action**: Updated the video processing loop and Redis control listener.
- **Logic**:
    - The `analyze_scene` command now accepts a JSON payload with `previous_state` and `user_query`.
    - These values are injected into the centralized template before being sent to the Cosmos VLM.

### 4. Schema Update
- **File**: [src/orchestrator/tool_schema.py](src/orchestrator/tool_schema.py)
- **Action**: Added `specific_query` to the `CAMERA_TOOL` definition.
- **Impact**: Allows the L1 agent (Quint) to ask detailed questions like "Is there any white water?" or "Is that boat moving?".

---

## Testing Methodology

To verify the changes, follow these three steps:

### Test 1: Syntax & Integration
Ensure no regressions were introduced in the communication bridge.
```bash
python3 -m py_compile src/orchestrator/prompts.py src/watchstander/watchstander.py src/orchestrator/agent_orchestrator.py
```

### Test 2: Prompt Injection (Manual)
Trigger a manual analysis via Redis with a mock "previous state" and "user query" to verify the logs show the correctly formatted prompt.
```bash
# In a terminal with access to the redis container:
redis-cli publish vision_control '{"command": "analyze_scene", "previous_state": "The water is calm with no traffic.", "user_query": "Are there any waves?"}'
```
*Verify: Check the Watchstander logs to see if "Are there any waves?" was prioritized.*

### Test 3: End-to-End (Conversational)
Use the local audio or text CLI to ask a visual question.
1. Start the stack.
2. Run `python3 scripts/text_cli.py`.
3. Input: "Hey Quint, check the camera, are there any white caps on the water?"
4. Observe the tool call logs to ensure `specific_query` is populated.

---

## Benefits
- **Reduced Latency**: By focusing on changes, the VLM produces shorter, more concise responses.
- **Better UX**: The bridge between "user intent" and "vision analysis" is now direct.
- **Consistency**: All maritime-vessel logic (qualitative positions instead of bearings) is enforced across both service layers.
