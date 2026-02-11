# chatty-buoy

**Your virtual crew member who never shuts up (in a good way).**

Think of that local guy who's been on the water for 40 years — he seen the ebbs and flows, knows every boat, every captain, and won't shut up about it. Now imagine he's an AI running on your boat, listening to all 55 marine VHF channels at once, watching your cameras, reading your instruments, and actually helpful *and* entertaining.

**chatty-buoy is an edge AI for boats.** It's the crew member who:
- Listens to *everything* on VHF + AIS (all channels, all vessels, all the time)
- Watches your cameras, chartplotter, radar, and transducer — tells you what's coming
- Knows every ledge, drop-off, and contour line like the back of his hand
- Always knows what's going down on 16 before you do (and has opinions about it)
- Will help you dock but definitely has thoughts about your approach angle

No cloud. No subscription. No connectivity. Just a really chatty AI running locally on your vessel.

If things turn south, he's right there with you. You might think you're SOL, but you're not Solo!

## Architecture: The "Speed Demon" Hybrid Stack

The system runs on a **Hybrid Architecture** optimized for Jetson Thor (ARM64), balancing containerized stability with native performance:

*   **Audio (Hearing)**: `nvcr.io/nvidia/riva/riva-speech:2.24.0-l4t-aarch64` (Docker). High-performance streaming speech recognition.
*   **Cortex (Reasoning)**: `cortex-service` (Docker). Primary: **Olmo3-7B**. Option: **Nemotron-3-Nano** (NVFP4) for Thor.
*   **Speaking (TTS)**: **Native Local Execution** of **KokoroTTS**. Runs directly on Metal/CUDA for <200ms synthesis, bypassing container overhead.
*   **Memory (RAG)**: `pgvector` (Docker). Vector database for long-term knowledge retention.

See [Quintessential Architecture](docs/quint_architecture.md) for deep dives.

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

**Step B: Voice Synthesis (Local)**
Starts the optimal local TTS server.
```bash
# In a new terminal
micromamba activate chatty-buoy
bash scripts/start_tts.sh
```

**Step C: The Agent**
Interact with the system.
```bash
micromamba run -n chatty-buoy python3 src/agent_reflex.py
```

## Roadmap

We are currently in **Phase 1 (Foundation)**. See [Roadmap](docs/roadmap.md) for the journey ahead, including NMEA integration, Sonar analysis, and Vision capabilities.

## License

MIT
