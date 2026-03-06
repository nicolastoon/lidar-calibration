#!/bin/bash
set -e

echo "================================"
echo "  LiDAR-Camera Calibration Tool"
echo "================================"

# If no arguments passed, show help
if [ "$#" -eq 0 ]; then
    python3 /calibration/calibration.py --help
    exit 0
fi

python3 /calibration/calibration.py "$@"