# Chatty Buoy Hardware Specifications

This document defines the hardware architectures for the Chatty Buoy project, detailing system configurations for real-time marine data processing, AI-driven situational awareness, and deployment scenarios ranging from autonomous buoys to crewed vessels.

---

## Table of Contents

1. [Hardware Configurations](#hardware-configurations) — Technical specifications
2. [Use Cases & Target Users](#use-cases--target-users) — Deployment scenarios
3. [Future Upgrades](#future-upgrades)

---

## Hardware Configurations

### Config 1: Full Deploy (Island/Platform)

**Core Compute:**
- **Model**: NVIDIA Thor SoC (ARM-based)
- **GPU**: Integrated NVIDIA GPU with CUDA support
- **Memory**: 16-24GB unified memory
- **Storage**: 512GB+ NVMe SSD
- **OS**: Ubuntu 22.04 LTS (ARM64)
- **Power Draw**: 60-80W peak

**Sensors:**
- **Cameras**: 1-4x IP cameras (1080p, RTSP, IP67+, 5-15W each)
- **VHF Radio**: Marine VHF with USB audio capture (5-10W)

**Networking:**
- Ethernet primary
- Optional 4G/5G LTE modem
- 8-port PoE+ managed switch (20-30W)

**Power System:**
- **Solar**: 400-800W (2-4x panels)
- **Battery**: 5-10 kWh LiFePO4 (24V nominal)
- **Charge Controller**: MPPT 40-60A
- **Inverter**: 1000-1500W pure sine wave

**Daily Power Budget**: ~3 kWh (125W average, 180W peak)

**Environmental:**
- NEMA 4X/IP66 enclosure
- Active cooling (12V fans, 10-20W)
- Operating temp: 0-45°C

---

### Config 2: Mobile Working Vessel

**Core Compute:**
- **Model**: NVIDIA Jetson AGX Orin 64GB
- **GPU**: 2048 CUDA cores (Ampere)
- **Power**: 25-40W typical, 60W peak
- **Storage**: 512GB NVMe SSD

**Marine VHF SDR:**
- **SDR**: RTL-SDR Blog V4 or HackRF One
- **Bandwidth**: 2.4-20 MHz (covers 156-162 MHz marine band)
- **Processing**: 55+ channel parallel demux + wake-word detection
- **Power**: 2-5W

**Sensors:**
- **Cameras**: 2x IP cameras (bow + stern)
- **GPS**: USB GPS dongle
- **AIS**: Optional receiver

**Power System (Marine Integration):**
- **Source**: 36V trolling motor system (3x 12V in series)
- **Converter**: 36V → 12V DC-DC buck (10A, 120W, isolated)
- **Total Load**: 70W continuous (AGX + SDR + cameras + GPS)
- **DC Current @ 12V**: 6A continuous
- **Runtime (engine off)**: 10-15 hours (don't fully discharge trolling batteries)

**Installation:**
- 10A fuse on 12V output
- 10 AWG marine tinned copper
- Pelican case (IP67) + 12V fan (5-10W)

**Environmental:**
- NEMA 4X or Pelican case
- Conformal coating (salt protection)
- Rubber mounting (shock/vibration)

---

### Config 3: Mega-Yacht (Thor Multi-Node)

**Core Compute:**
- **Model**: NVIDIA Thor SoC (or 2-4x Thor daisy-chained via NVLink/Infiniband)
- **GPU**: 120GB VRAM per unit (245-491GB total for 2-4x cluster)
- **Power**: 100-150W sustained per unit (200W peak)
- **Storage**: 2TB NVMe SSD per node
- **Networking**: NVLink/Infiniband for <10ms inter-GPU latency

**Sensors (Enhanced):**
- **Cameras**: 6x 4K IP cameras (360° + underwater)
- **SDR**: HackRF One dual-channel (VHF + SSB 2-30 MHz)
- **GPS/DGPS**: RTK (cm-level precision)
- **AIS**: Dual receivers
- **Weather Station**: Wind, pressure, humidity, SST
- **Hydrophone**: Underwater audio
- **NMEA 2000**: Full integration (GPS, depth, autopilot, radar, transducers)

**Power System (Yacht Electrical):**
- **Primary**: 25-50 kW diesel generator + 48V lithium house bank (200+ kWh)
- **Distribution**: 48V → 12V buck (1500W capacity, Victron Skylla-i)
- **Backup**: 500F supercapacitor UPS (genset fail-over)
- **Total Load**: 250W sustained (single Thor), 400-800W (2-4x cluster)

**Environmental:**
- Carbon fiber shroud (aesthetic + thermal)
- Liquid cooling (closed-loop chiller, 5kW capacity)
- Gyro-stabilized mounting (pitch/roll compensation)

**Networking:**
- **Primary**: Starlink Pro (40 Mbps, 50ms latency)
- **Backup**: 4G/5G hotspot
- **Onboard**: Gigabit WiFi 6E mesh

---

### Config 4: Distributed (Lite Buoy + Shore Station)

**Lite Buoy (Onboard):**
- **Compute**: Jetson AGX Orin 32GB (20W typical, 60W peak)
- **Storage**: 256GB NVMe SSD
- **Power**: 200W solar + 3-4 kWh LiFePO4 (isolated from vessel)
- **Sensors**: 2-4x cameras, VHF SDR, optional weather/hydrophone
- **Networking**: WiFi (local), LoRa (10km), or Iridium (global)
- **Enclosure**: NEMA 4X/IP67
- **Data Flow**: Real-time inference + event streaming to shore

**Shore Station (Onshore):**
- **Compute**: NVIDIA Thor SoC (or equivalent, 16-24GB VRAM)
- **Storage**: 2TB+ NVMe SSD
- **Power**: 1200W solar + 8-12 kWh LiFePO4 (backup generator optional)
- **Networking**: 4G/5G, Starlink, or Ethernet
- **Data Flow**: Heavy AI analysis + video archive

**Link Limitations:**
- LoRa: 10km max, 50 kbps
- Iridium: Global, 2 kbps burst

---

## Use Cases & Target Users

### 1. Full Deploy (Island/Platform)
**Target Users:**
- Research institutions (marine biology, oceanography)
- Government agencies (NOAA, Coast Guard monitoring stations)
- Remote island communities (weather + vessel monitoring)

**Use Cases:**
- Autonomous 24/7 monitoring of marine protected areas
- Remote weather stations with AI-driven storm prediction
- Airgapped deployments (no cloud connectivity)

**Why This Config:**
Long-term autonomous operation in fixed locations with variable power (solar/battery). Ideal for research, regulatory compliance, and offline-first deployments.

---

### 2. Mobile Working Vessel
**Target Users:**
- Fishing charter operators, marine surveyors, research teams, long-range sailors

**Use Cases:**
- Power-efficient operation using vessel's 36V DC system (trolling motor batteries)
- Real-time sensor fusion from AIS, sonar, and weather sensors
- Optional "moored inference" — continuous operation while anchored
- 10-15 hour runtime on battery power

**Why This Config:**
Minimal power draw integrates seamlessly with existing electrical systems. Ideal for fishing, surveying, research missions.

---

### 3. Mega-Yacht (Thor Multi-Node Cluster)
**Target Users:**
- Ultra-high-net-worth individuals, private security, research vessels in high-risk zones

**Use Cases:**
- Run 402B parameter models (Llama 4 Maverick) locally, zero cloud dependency
- NMEA 2000 + radar + AIS + video + VHF + hydrophone → unified threat awareness
- Privacy-first: no external API calls (location/conversation privacy)
- Full capability in contested waters (offline Starlink-independent operations)

**Why This Config:**
Security, privacy, and offline autonomy where cloud access is unavailable, untrusted, or denied. Manages 400-800W on standard yacht generators (25-50 kW).

---

### 4. Distributed (Lite Buoy + Shore Station)
**Target Users:**
- Government monitoring networks, academic research, commercial fisheries

**Use Cases:**
- Lightweight AI-enabled floating buoy + centralized shore station
- Real-time edge inference (video, VHF monitoring) on buoy
- Heavy lifting (LLM analysis, video archiving) on shore
- Communication via LoRa (10km range) or Iridium (global)

**Why This Config:**
Reduces floating power requirements, enables smaller/more mobile buoys, scalable architecture (1 shore station, N buoys).

---

**Status**: Living document  
**Last Updated**: 2025-12-19
