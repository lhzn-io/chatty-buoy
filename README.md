# chatty-buoy

**An edge-deployed AI crewmate for marine safety and situational awareness.**

**chatty-buoy** is a locally-hosted, multimodal AI assistant designed for NVIDIA Jetson hardware. It acts as an always-on watchstander that monitors NMEA data, transcribes VHF radio traffic, and analyzes visual feeds in real-time.

It keeps a vigilant eye on the horizon so you don't have to—though it may occasionally offer an unsolicited critique of your docking maneuvers.

**No cloud. No subscription. No latency. Just 100% local intelligence.**

## Architecture

For a detailed breakdown of the system capabilities, service orchestration, and Jetson Thor integration, please refer to the [Quintessential Architecture](docs/planning/quint_architecture.md).

## Setup & Usage

### 1. Requirements
*   NVIDIA Jetson AGX Thor (JetPack 7.0+)
*   Docker & NVIDIA Container Runtime
*   `micromamba` (Environment: `chatty-buoy`)
*   **Optional**: `ngc` CLI (Install manually if you need to download new Riva models).
    > [Install NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli)

### 2. Start the Stack

**Step A: Application Infrastructure (Docker)**
Starts Riva (ASR), Triton (Cortex), and Postgres (Memory).
```bash
docker compose up -d
```

**Step B: Voice Synthesis**
The voice service uses **Chatterbox-Turbo (350M)** by default for expressive, low-latency speech.
*   **Primary**: Chatterbox-Turbo (runs on GPU 0).
*   **Fallback**: Kokoro-FastAPI (82M) is available in `src/voice/Kokoro-FastAPI` for ultra-low resource environments.

**Step C: The Agent**
Interact with the system.
```bash
micromamba run -n chatty-buoy python3 src/agent_reflex.py
```

## Roadmap

We are currently in **Phase 3 (Situational Awareness)**. See [Roadmap](docs/planning/roadmap.md) for the journey ahead, including NMEA integration, Sonar analysis, and Vision capabilities.

## License

MIT
