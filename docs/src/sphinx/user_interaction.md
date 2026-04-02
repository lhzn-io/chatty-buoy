# Quint (Async Cascade) User Guide

## 1. Interaction Modes

### Mode A: Fast Chat (The "Face")
**Goal:** Immediate, conversational responses.
**How to Invoke:** Just speak naturally.
**Architecture Path:** L0 -> L1 (Gemma) -> L2 (Dispatcher) -> L3 (Cortex).
**Example:**
> **You:** "Hey Quint, is there any traffic ahead?"
> **Quint:** "Checking AIS targets now..." (L1)
> **(Pause):** (L2 tools execute, L3 analyzes)
> **Quint:** "Captain, update: I see two tankers on collision course..." (L3)

### Mode B: Planning Mode (The "Brain")
**Goal:** Complex mission planning, strategies, or multi-step analysis.
**How to Invoke:** Use hotwords **"Plan"**, **"Mission"**, **"Strategy"**.
**Architecture Path:** L0 -> L3 (Cortex Direct).
**Example:**
> **You:** "Plan a mission to scan sector 7 and report fuel status."
> **Quint:** "Acknowledged. Initiating strategic plan. Step 1: Check fuel levels. Step 2: Set waypoint to Sector 7. Step 3: Begin AIS scan..."

## 2. Available Tools

The agent decides which tool to use based on your request.

| Tool Name | Trigger Examples | Description |
| :--- | :--- | :--- |
| `get_system_status` | "Status report", "System check", "How are the engines?" | Returns CPU, RAM, GPU, and Power stats. |
| `get_ais_targets` | "Is there traffic?", "Scan for ships", "AIS check" | Returns list of nearby vessels (simulated). |
| `set_waypoint` | "Set course for...", "New waypoint" | Sets navigation target. |

## 3. System Stats
Quint has access to the machine's telemetry (Jetson Thor).
**Try asking:**
- "System status report."
- "What is your GPU usage?"
- "Check power levels."
**Quint will respond with:**
- "CPU: 12% | RAM: 4.2/32GB | GPU: 45% | Power: 22W"

## 4. How to Run
```bash
# Start the full stack (requires Docker + Jetson/GPU)
docker-compose up

# OR run the orchestrator locally (if services are running)
python3 src/orchestrator/agent_orchestrator.py
```
