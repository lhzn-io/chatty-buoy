# Project Roadmap

**Version:** 2.0 (Async Cascade)
**Date:** 2026-02-10

This roadmap tracks the evolution of **Chatty Buoy**, from its initial foundation to a fully autonomous maritime agent.

## Phase 1: Foundation (Completed)
**Goal**: Establish a rock-solid, low-latency conversational loop on Jetson Thor.
- [x] **Infrastructure**: Hybrid Docker Stack (Riva, vLLM, Postgres).
- [x] **Audio Loop**: ALSA Input -> Riva ASR -> Orchestrator -> TTS.
- [x] **Basic "Reflex" Agent**: Simple synchronous responses.

## Phase 2: The Async Cascade (Completed)
**Goal**: Decouple conversation from heavy reasoning to ensure <100ms fluidity.
- [x] **L2 Dispatcher**: Deploy `FunctionGemma-270m` for tool calling.
- [x] **L1 Front-End**:  Deploy `Gemma-3-4B-IT` for chat/personality.
- [x] **L3 Cortex**: Integrate `Nemotron-3-30B` for deep reasoning.
- [x] **Orchestrator V2**: Implement Async Fork logic (L1 talks while L2 works).

## Phase 2.5: Planning Capabilities (Completed)
**Goal**: Enable complex, multi-step mission planning.
- [x] **Hotword Router**: L0 Gatekeeper detects "Plan", "Mission", "Strategy".
- [x] **Meta-Planner**: Direct L0 -> L3 handoff for strategic plan generation.

## Phase 3: Situational Awareness (In Progress)
**Goal**: Give the agent eyes without blinding the brain.
### Phase 3.1: "The Watchstander" (Fast Stream / DLA) - **IN PROGRESS**
**Objective**: Real-time safety monitoring using low-power hardware (DLA).
- [ ] **The Engine**: YOLOv11-S compiled for Jetson DLA (Deep Learning Accelerator).
- [ ] **Event Stream**: Publishes structured contacts to Redis `vision_events`.
- [ ] **Reflex Loop**: "Collision Guard" logic for immediate safety alerts.
- [ ] **Integration**: `get_visual_contacts()` tool for Cortex awareness.

### Phase 3.2: "The Analyst" (VLM) - **DEFERRED**
**Reason**: VRAM constraints on AGX Thor (serving 70B+ of LLM weights).
- [ ] **Visual Reasoning**: VILA / Llava integration.
- [ ] **Scene Analysis**: "What is that structure on the horizon?"

## Phase 4: Full Autonomy (Future)
**Goal**: Proactive monitoring and safety.
- [ ] **Multi-Modal RAG**: Querying manuals based on visual context.
- [ ] **Hardware Deployment**: Final optimization for Jetson Thor edge deployment.
- [ ] **Performance Overhaul**: Execute the [NVFP4 Migration Plan](nvfp4_upgrade_plan.md) to maximize hardware throughput.
