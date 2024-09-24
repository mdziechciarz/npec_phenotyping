# Use the appropriate base image
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20240418.v1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    mesa-utils \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda \
    && rm /miniconda.sh

# Set path to conda
ENV PATH /opt/conda/bin:$PATH

# Copy the environment file
COPY requirements.txt /tmp/requirements.txt

# Create the conda environment
RUN conda create --name project_environment python=3.9 \
    && conda run -n project_environment pip install -r /tmp/requirements.txt

# Activate the environment
RUN echo "source activate project_environment" > ~/.bashrc
ENV PATH /opt/conda/envs/project_environment/bin:$PATH

# Set the working directory
WORKDIR /workspace
