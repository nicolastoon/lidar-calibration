#!/bin/bash
docker run --rm \
  -v /mnt/c/Users/nicol/documents/github/lidar-calibration/bags:/bags \
  lidar-camera-calibration \
    --bag /bags/calib_20260227_014256 \
    --checkerboard 8 6 \
    --square-size 0.025 \
    --output /bags/result