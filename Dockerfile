FROM ros:humble-ros-base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    libopencv-dev \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-sensor-msgs-py \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install \
    rosbags \
    open3d \
    numpy \
    opencv-python \
    matplotlib \
    scipy

WORKDIR /calibration

COPY calibration.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/calibration/entrypoint.sh"]