# Use official CUDA-enabled PyTorch base image (matching VideoPainter CUDA version)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    SAM2_BUILD_CUDA=0

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-venv \
        python3.10-dev \
        python3-distutils \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libavutil-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libffi-dev \
        sudo \
        unzip \
        wget \
        && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Some workflow runners call `python` explicitly; ensure it exists.
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create working directory
WORKDIR /workspace/sam2

# Upgrade pip and install base dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy cython

# Copy SAM2 setup files first (better layer caching)
COPY setup.py /workspace/sam2/setup.py
COPY README.md /workspace/sam2/README.md
COPY pyproject.toml /workspace/sam2/pyproject.toml

# Install SAM2 dependencies (editable install will happen after copying full source)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies for video processing
RUN pip install --no-cache-dir \
    opencv-python \
    opencv-python-headless \
    tqdm \
    Pillow \
    gcsfs \
    google-cloud-storage \
    flytekit>=1.13.0 \
    "transformers>=4.46.0" \
    "accelerate>=0.30.0" \
    qwen-vl-utils \
    safetensors \
    sentencepiece

# Copy HLX workflow wheel
COPY hlx_wf-2.4.2.dev9-py3-none-any.whl /workspace/sam2/

# Install the HLX workflow wheel (after flytekit, matching VideoPainter pattern)
RUN python3 -m pip install /workspace/sam2/hlx_wf-2.4.2.dev9-py3-none-any.whl

# Copy entire project (at the end for better layer caching)
COPY . /workspace/sam2

# Install SAM2 in editable mode (now that full source is present)
RUN pip install -e . --no-build-isolation

# Expose no specific port (this is a batch processing container, not a server)
# The workflow orchestrator will handle container lifecycle

# Default command (general-purpose like VideoPainter)
CMD ["/bin/bash"]
