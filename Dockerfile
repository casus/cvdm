# Use an official lightweight image as a base
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Set up working directory
WORKDIR /app

# Copy current directory contents into the container
COPY . .

# Set environment name
ENV ENV_NAME=cvdm

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin:$PATH
ENV PYTHONPATH=$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/lib/python3.10/site-packages:$PYTHONPATH

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Set up micromamba environment, install dependencies and pip packages
RUN micromamba create -n $ENV_NAME -y && \
    micromamba install -n $ENV_NAME \
    tensorflow==2.15.0 \
    keras==2.15.* \
    matplotlib==3.8.0 \
    tqdm==4.65.0 \
    scikit-learn==1.4.2 \
    scikit-image==0.22.0 \
    einops==0.7.0 \
    neptune==1.10.2 -y && \
    /usr/local/bin/micromamba run -n $ENV_NAME pip3 install opencv-python==4.9.0.80 \
    tensorflow-addons==0.23.0 \
    cupy-cuda12x==13.3.0 \
    pytest && \
    /usr/local/bin/micromamba run -n $ENV_NAME python -m pip install .

# Default command when the container starts (optional)
CMD ["/bin/bash"]
