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

# Bootstrap pip for Python 3.12 and install setuptools
RUN python3 -m ensurepip --upgrade && \
    python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies directly into system Python (like VideoPainter)
# so flytekit and torch share the same interpreter.
RUN pip install --no-cache-dir \
        "accelerate>=1.12.0" \
        "av>=16.0.1" \
        "einops>=0.8.1" \
        "hydra-colorlog>=1.2.0" \
        "hydra-core>=1.3.2" \
        "pandas>=2.3.3" \
        "physical_ai_av>=0.1.0" \
        "pillow>=12.0.0" \
        "torch==2.8.0" \
        "torchvision>=0.23.0" \
        "transformers==4.57.1" \
        psutil && \
    pip install --no-cache-dir --no-build-isolation "flash-attn>=2.8.3" && \
    python3 -c "import torch; print('torch', torch.__version__)" && \
    python3 -c "import transformers; print('transformers', transformers.__version__)" && \
    python3 -c "from transformers import AutoTokenizer; print('AutoTokenizer import ok')"

# Copy and install the editable project (src/)
COPY src/ /workspace/alpamayo/src/
COPY pyproject.toml /workspace/alpamayo/
RUN pip install --no-cache-dir -e .

COPY hlx_wf-2.4.2.dev9-py3-none-any.whl /workspace/alpamayo/

# Install the hlx_wf wheel into system Python
RUN python3 -m pip install /workspace/alpamayo/hlx_wf-2.4.2.dev9-py3-none-any.whl

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
