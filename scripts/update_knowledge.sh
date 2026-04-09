#!/bin/bash
echo "=== Chatty-Buoy Knowledge Ingestion ==="
echo "Hashing and processing PDFs in ./pdfs/..."
export PYTHONPATH=$(pwd)
micromamba run -n chatty-buoy python3 src/cortex/rag.py
echo "=== Done! ==="
