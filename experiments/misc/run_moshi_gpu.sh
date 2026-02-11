#!/bin/bash
set -e

# Path to the environment
ENV_NAME="chatty-buoy"
# Libraries are now symlinked in $CONDA_PREFIX/lib, no LD_LIBRARY_PATH Needed for SBSA/Thor support.

# Activate environment and run
micromamba run -n $ENV_NAME python -m moshi.server --port 8998 $@
