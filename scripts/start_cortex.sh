#!/bin/bash
set -e

echo "Starting Cortex Service Pre-Flight..."

# 1. Download Reasoning Parser Plugin (if not present)
if [ ! -f /models/config/nano_v3_reasoning_parser.py ]; then
    echo "Downloading nano_v3_reasoning_parser.py..."
    mkdir -p /models/config
    wget -O /models/config/nano_v3_reasoning_parser.py https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/resolve/main/nano_v3_reasoning_parser.py
else
    echo "Parser plugin found."
fi

# 2. Launch vLLM
echo "Launching vLLM (Nemotron-3-Nano-30B-A3B-NVFP4)..."
exec vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin /models/config/nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.45 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
