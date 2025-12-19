# chatty-buoy

**Your virtual crew member who never shuts up (in a good way).**

Think of that local guy who's been on the water for 40 years — he seen the ebbs and flows, knows every boat, every captain, and won't shut up about it. Now imagine that guy, but he's an AI running on your boat, listening to all 55 marine VHF channels at once, watching your cameras, reading your instruments, and actually helpful instead of just entertaining.

**chatty-buoy is an edge AI for boats.** It's the crew member who:
- Listens to *everything* on VHF + AIS (all channels, all vessels, all the time)
- Watches your cameras, chartplotter, radar, and transducer — tells you what's coming
- Knows every ledge, drop-off, and contour line like the back of his hand
- Always knows what's going down on 16 before you do (and has opinions about it)
- Will help you dock but definitely has thoughts about your approach angle

No cloud. No subscription. No connectivity. Just a really chatty AI running locally on your vessel.

If things turn south, he's right there with you. You might think you're SOL, but you're not Solo!

## Overview

chatty-buoy combines edge compute (NVIDIA Jetson/Thor), software-defined radio (SDR), and computer vision to provide AI-powered maritime monitoring without cloud dependencies. Built for environments where privacy, latency, and offline operation matter.

**Target deployments:**

- Small craft with power constraints
- Research vessels and autonomous buoys
- Luxury yachts requiring offline AI autonomy
- Distributed coastal sensor networks

## Architecture

Built on NVIDIA Jetson Thor (Blackwell sm_110) using custom vLLM image:
`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor`

## Use Cases

### 1. Yacht/Research-Ship Integrated Deployment
**Target Users:** Private yachts, research vessels, oceanographic institutions

**What It Does:**
- Modular architecture with distributed sensor arrays
- Onboard crew interaction interfaces
- Dynamic power management with optional satellite uplink
- Designed for variable crew sizes and extended operational ranges

**Why:** Can be configured for both short-term research expeditions and long-term private monitoring with mobility and range.

---

### 2. Mobile Working Vessel
**Target Users:** Fishing charter operators, marine surveyors, research teams, long-range sailors

**What It Does:**
- Power-efficient operation using vessel's 36V DC system (trolling motor batteries)
- AGX Orin inference with prioritized VHF monitoring
- Real-time sensor fusion from AIS, sonar, and weather sensors
- Optional "moored inference" — continuous operation while anchored
- 10-15 hour runtime on battery power

**Why:** Ideal for fishing, surveying, research missions, and extended passages with minimal power draw. Integrates with existing electrical system, no separate power install required.

---

### 3. Lite Buoy + Shore Station (Distributed)
**Target Users:** Government monitoring networks, academic research, commercial fisheries

**What It Does:**
- Two-part deployment: lightweight AI-enabled floating buoy + centralized shore station
- Buoy performs real-time edge inference (video, VHF monitoring)
- Shore station handles heavy lifting (LLM analysis, video archiving)
- Communication via LoRa (10km range) or Iridium (global)

**Why:** Reduces floating power requirements, enables smaller/more mobile buoys, scalable architecture (1 shore station, N buoys).

---

### 4. Mega-Yacht (Thor Multi-Node Cluster)
**Target Users:** Ultra-high-net-worth individuals, private security, research vessels in high-risk zones

**What It Does:**
- **Offline AI autonomy**: Run 402B parameter models (Llama 4 Maverick) locally, zero cloud dependency
- **Sensor fusion**: NMEA 2000 + radar + AIS + video + VHF + hydrophone → unified threat awareness
- **Privacy-first**: No external API calls (location/conversation privacy)
- **Starlink-denied ops**: Full capability in contested waters (South China Sea, Russia EEZ, military exclusion zones)

**Hardware:**
- 2-4x NVIDIA Thor SoCs (245-491GB VRAM total)
- <10ms inter-GPU latency (NVLink/Infiniband)
- 6x 4K cameras (360° coverage)
- Dual SDR (VHF + SSB 2-30 MHz)
- Liquid cooling + gyro-stabilized mounting

**Real-World Scenario:**
> Yacht transiting contested waters. Starlink blocked. Multi-Thor cluster provides GPT5/Gemini-class reasoning for nav decisions, crew safety alerts (VHF monitoring), and vessel detection (radar + AIS + video fusion) — all offline.

**Power:** 400-800W (manageable on yacht generators 25-50 kW)

**Why:** Security, privacy, and offline autonomy in environments where cloud access is unavailable, untrusted, or denied.

---

## Setup

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- Python 3.11+ with conda

### Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate chatty-buoy

# Initialize infrastructure from kanoa-mlops templates
kanoa mlops init --dir .
```

⚠️ **Important**: The `docker/` directory is generated by `kanoa mlops init` and is not tracked in git. Run the init command to generate the latest infrastructure configs from kanoa-mlops templates.

## Usage

### Interactive Mode

```bash
# Start any vLLM or Ollama service interactively
kanoa mlops serve

# List available models
kanoa mlops list

# Check service status
kanoa mlops status
```

### Direct Commands

```bash
# Start specific service
kanoa mlops serve vllm molmo
kanoa mlops serve vllm gemma3 --model google/gemma-3-12b-it

# Stop services
kanoa mlops stop vllm molmo
```

## Models

Current models configured:

- **gemma3**: Google Gemma 3 (12B, 27B variants)
- **molmo**: AllenAI Molmo 7B (multimodal)
- **olmo3**: AllenAI OLMo 3 (7B, 32B variants)

Models are cached in `~/.cache/huggingface/hub/`.

## Docker Services

Services are defined in `docker/vllm/` and `docker/ollama/`:

- `docker-compose.gemma3.yml` - Gemma 3 inference
- `docker-compose.molmo.yml` - Molmo multimodal
- `docker-compose.olmo3.yml` - OLMo 3 inference
- `docker-compose.ollama.yml` - Ollama runtime

## Deployment Configurations

See [deployment-configurations.md](docs/deployment-configurations.md) for complete specifications, power budgets, and architectural details across all deployment modalities.

## License

MIT
