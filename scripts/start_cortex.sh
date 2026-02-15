#!/bin/bash
set -e

echo "Starting Cortex Service Pre-Flight..."



# 2. Launch vLLM
echo "Launching vLLM (AllenAI Olmo-3-7B-Think)..."
exec vllm serve allenai/Olmo-3-7B-Think \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser pythonic \
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.15 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
