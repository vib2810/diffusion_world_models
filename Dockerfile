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
RUN pip install opencv-python

# Install any needed packages specified in requirements.txt
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install gym==0.12.0 atari-py==0.1.4 scikit-image==0.15.0 matplotlib==3.0.2

# install h5py
RUN pip install h5py
RUN pip install wandb

# install tk
RUN apt-get update && apt-get install -y tk

# set workdir as home/ros_ws
WORKDIR /home/mj_ws

CMD [ "bash" ]