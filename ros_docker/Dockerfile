# Use CUDA 12.4 runtime on Ubuntu 20.04 for RTX 5090 support
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"

# 1. Install System Dependencies & ROS Noetic
RUN apt-get update && apt-get install -y \
    curl git wget lsb-release build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - \
    && apt-get update && apt-get install -y \
    ros-noetic-ros-base ros-noetic-cv-bridge \
    ros-noetic-image-transport ros-noetic-compressed-image-transport \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /root/miniconda3 \
    && rm miniconda.sh

# 3. Create Conda Environment
# Note: We install dependencies here so they are cached in the image
RUN conda create -n policy_env python=3.9 -y
SHELL ["conda", "run", "-n", "policy_env", "/bin/bash", "-c"]

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install opencv-python rospkg pyyaml dill hydra-core omegaconf lightning diffusers

# 4. Final ROS Setup
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "conda activate policy_env" >> ~/.bashrc

# Set the working directory to where we will mount the code
WORKDIR /app/Policy-Lightning

# Entrypoint to ensure ROS and Conda are sourced
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "policy_env", "bash"]