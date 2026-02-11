#!/bin/bash
echo "Dropping caches..."
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
echo "Caches dropped."
