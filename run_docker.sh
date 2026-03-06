#!/bin/bash
docker run --rm \
  -v /mnt/c/Users/nicol/documents/github/lidar-calibration/bags:/bags \
  lidar-calibration \
    --bag /bags/calib_20260306_090152 \
    --output /bags/result

# docker run --rm -v C:/Users/nicol/documents/github/lidar-calibration/bags:/bags lidar-calibration --bag /bags/calib_20260306_090152 --output /bags/result