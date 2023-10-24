# Base image
FROM nvidia/cuda:12.0.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Set work directory
WORKDIR /app

# Update and install basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    vim \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Download Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install Miniconda
RUN bash ~/miniconda.sh -b -p /miniconda

# Clean up Miniconda installation
RUN rm ~/miniconda.sh

# Create symlinks
RUN ln -s /miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN echo ". /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate base" >> ~/.bashrc

# Add Miniconda to PATH so conda is usable
ENV PATH /miniconda/bin:$PATH

# Copy environment.yml
COPY gym_mujoco.yml /app/gym_mujoco.yml

# Use environment.yml to create conda environment
RUN conda env create -f /app/gym_mujoco.yml

# Ensure that the conda environment is activated when running the container
ENV CONDA_DEFAULT_ENV gym_mujoco
ENV CONDA_PREFIX /miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH $CONDA_PREFIX/bin:$PATH

RUN mkdir $HOME/.mujoco
COPY mujoco210-linux-x86_64.tar.gz .
RUN tar -xvf mujoco210-linux-x86_64.tar.gz -C $HOME/.mujoco/

ENV LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/root/.mujoco/mujoco210/bin"
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libGLEW.so"
ENV PATH="$LD_LIBRARY_PATH:$PATH" 
RUN . ~/.bashrc

RUN echo "conda activate gym_mujoco" >> $HOME/.bashrc
RUN . ~/.bashrc


RUN apt-get update && apt-get install -y \
    patchelf \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libglew1.5 \
    libglew-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Ensure system is updated and install the libosmesa6-dev package
RUN apt-get update && apt-get -y upgrade && apt-get install -y libosmesa6-dev

WORKDIR $HOME/.mujoco
RUN git clone https://github.com/openai/mujoco-py
WORKDIR $HOME/.mujoco/mujoco-py
RUN pip install -r requirements.txt
RUN pip install -r requirements.dev.txt
RUN pip3 install -e . --no-cache

# Set work directory
WORKDIR /app

CMD tail -f /dev/null
