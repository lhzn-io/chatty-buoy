# How to Add a New Tool (Async Cascade)

This guide walks through adding a new capability (tool) to the agent, enabling L2 (Dispatcher) to invoke it and L3 (Cortex) to reason about the results.

## 1. Define the Tool Schema
Open `src/orchestrator/tool_schema.py` and add the JSON definition for your tool.

```python
# src/orchestrator/tool_schema.py

MY_NEW_TOOL = {
    "name": "check_weather",
    "description": "Get current weather conditions for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City or specific coordinates."}
        },
        "required": ["location"]
    }
}

# Add to the exported list
AVAILABLE_TOOLS = [AIS_TOOL, SYSTEM_TOOL, NAV_TOOL, MY_NEW_TOOL]
```

## 2. Implement the Logic
Update `src/orchestrator/agent_orchestrator.py` to handle the execution of this tool.

```python
# src/orchestrator/agent_orchestrator.py

    async def _execute_tool(self, tool_call):
        name = tool_call.get("name")
        args = tool_call.get("parameters", {})
        
        # ... existing tools ...
        
        if name == "check_weather":
            location = args.get("location", "local")
            # Call external API or sensor logic here
            return f"Weather at {location}: Sunny, 22C, Wind 5kt NE"
        
        return "Unknown Tool"
```

## 3. Verify Tool Invocation
To confirm the Front-End (L1) acknowledges it and the Back-End (L2/L3) executes it, use the verification script.

### Update the Verification Script
Modify `tests/test_async_cascade.py` to simulate a user query that triggers your new tool.

```python
# tests/test_async_cascade.py

# ... inside verify function ...
orch.text_queue.put("What's the weather like?")

# Update mock for L2 to return your tool call
if url == L2_URI:
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": '```json\n{"name": "check_weather", "parameters": {"location": "current"}}\n```'
            }
        }]
    }
```

### Run the Test
```bash
python3 tests/test_async_cascade.py
```
If successful, you will see:
-   **L1 (Gemma)**: "Checking weather..." (or similar conversational filler).
-   **L2 (Dispatcher)**: Calls `check_weather`.
-   **L3 (Cortex)**: Receives "Sunny, 22C..." and generates an update.

## Architecture Note: Complex Planning
For complex queries requiring multi-step plans (e.g., "Plan a route avoiding storms and refueling halfway"), the current `FunctionGemma` (L2) might struggle. 

**Recommended Approach:**
1.  **L1 Hand-off**: If L1 detects a high-level goal, it can trigger a special tool `consult_cortex(goal="...")`.
2.  **L3 Planning**: L3 (Nemotron) receives this, generates a multi-step plan (Step 1: Check Weather, Step 2: Check Fuel, Step 3: Calculate Route), and sends it back to L2 for execution.
3.  **Cyclic Execution**: L2 executes each step and reports back to L3.

*Note: Nemotron needs the system prompt to include the list of `AVAILABLE_TOOLS` to generate valid plans.*
