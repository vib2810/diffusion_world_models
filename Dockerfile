# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libopus-dev \
    libvpx-dev \
    pkg-config \
    libsrtp2-dev \
    libgl1-mesa-glx \
    libglib2.0-dev \
    xcb \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Python 3.6 or 3.7

# Install any needed packages specified in requirements.txt
RUN pip install torch==1.2.0 torchvision==0.4.0
RUN pip install gym==0.12.0 atari-py==0.1.4 scikit-image==0.15.0 matplotlib==3.0.2

# install h5py
RUN pip install h5py

# install tk
RUN apt-get update && apt-get install -y tk

# set workdir as home/ros_ws
WORKDIR /home/mj_ws

CMD [ "bash" ]