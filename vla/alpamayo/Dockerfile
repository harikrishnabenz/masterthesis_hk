# Use official CUDA-enabled base image with development tools
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        && add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        curl \
        wget \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        sudo \
        unzip \
        && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12

# Ensure python command exists (some tools call `python` explicitly)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create working directory
WORKDIR /workspace/alpamayo

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv --version

# Copy only dependency manifests first (better layer caching)
COPY pyproject.toml uv.lock* /workspace/alpamayo/

# Copy source files needed for editable install
COPY src/ /workspace/alpamayo/src/

# Bootstrap pip for Python 3.12 and install setuptools
RUN python3 -m ensurepip --upgrade && \
    python3 -m pip install --upgrade pip setuptools wheel

# Create virtual environment and install dependencies
# Note: This will download ~22GB model weights on first inference run
RUN /root/.local/bin/uv venv ar1_venv && \
    . ar1_venv/bin/activate && \
    /root/.local/bin/uv sync --active && \
    python3 -c "import torch; print('torch', torch.__version__)" && \
    python3 -c "import transformers; print('transformers', transformers.__version__)" && \
    python3 -c "from transformers import AutoTokenizer; print('AutoTokenizer import ok')"


COPY hlx_wf-2.4.2.dev9-py3-none-any.whl /workspace/VideoPainter//

# Install the wheel
RUN python3 -m pip install /workspace/VideoPainter/hlx_wf-2.4.2.dev9-py3-none-any.whl


# Set the virtual environment in PATH
ENV PATH="/workspace/alpamayo/ar1_venv/bin:$PATH"

# Copy project files (at the end for better layer caching)
COPY . /workspace/alpamayo

# Verify installation
RUN python3 --version && \
    pip list | grep -E "torch|transformers|diffusers" || true

# Note: Alpamayo-R1-10B model (~22GB) will be downloaded from HuggingFace on first run
# Model cache location: /root/.cache/huggingface
# Requires GPU with ≥24 GB VRAM for inference

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["/bin/bash"]
