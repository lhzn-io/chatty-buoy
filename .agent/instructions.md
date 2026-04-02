# Chatty-Buoy Developer Instructions & Best Practices

This document serves as the "System Prompt" for any AI agent or developer working on the `chatty-buoy` repository. Adhere to these rules strictly.

## 1. Core Philosophy: "The Speed Demon"
*   **Latency is King**: Every millisecond counts. Prefer streaming architectures (gRPC, GStreamer) over request/response (HTTP REST) where possible.
*   **Hybrid Architecture**: We fundamentally run a **Hybrid Stack**.
    *   **Docker**: Managed Services (Riva, Triton, Postgres).
    *   **Native**: Latency-critical logic (TTS, Audio I/O) runs locally via `micromamba` to bypass container overhead on Jetson.
*   **Offline First**: Ensure all logic runs without internet. No dependence on OpenAI/Anthropic APIs for runtime.

## 2. Coding & Environment Standards (Strict)
*   **Micromamba**: **ALWAYS** use the correct environment.
    *   Command: `micromamba run -n chatty-buoy python script.py`
    *   Interactive: `micromamba activate chatty-buoy`
    *   **NEVER** run pip/python in base env.
*   **Type Hinting**: All new Python code must be fully type-hinted.
*   **Asynchronous I/O**: Use `asyncio` for network/DB calls.
*   **Safety**: Scripts must be idempotent and use `set -e` in bash.

## 3. Git & Commit Hygiene
*   **Atomic Commits**: Stage specific files (`git add file.py`), **NEVER** `git add .`.
*   **Message Format**: Conventional Commits (`feat:`, `fix:`, `docs:`). **NO EMOJIS**.
*   **No Secrets**: Never commit `.env` files.

## 4. Architecture Specifics
*   **Audio Pipeline**: GStreamer via `gi.repository`.
*   **Services**:
    *   **ASR**: gRPC (Riva)
    *   **Brain**: gRPC/HTTP (Triton)
    *   **TTS**: HTTP (Local FastAPI).
*   **Docker**:
    *   **Pin Versions**: Use specific tags (e.g., `riva-speech:2.24.0-l4t-aarch64`).
    *   **Permissions**: Set `chmod 644` on config files mounted to containers.

## 5. Documentation Habits
*   **Preserve Context**: Read existing docs before editing.
*   **Roadmap Alignment**: Ensure changes match `docs/roadmap.md`.
*   **User Guide**: Update `docs/src/sphinx/user_guide.rst` for runtime changes.

## 6. Hardware Awareness (Jetson AGX Thor)
*   **Compute**: Offload to GPU/DLA where possible.
*   **Memory**: Monitor Unified Memory usage.
*   **Platform**: Expect ARM64 (aarch64) architecture at all times.

## 7. Boundaries
*   ✅ **Always**: Pin images, use relative paths, document prerequisites.
*   ⚠️ **Ask First**: Changing default ports, major version upgrades.
*   🚫 **Never**: Hardcode absolute paths, run as root unnecessarily, commit secrets.
