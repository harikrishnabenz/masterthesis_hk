# Use official CUDA-enabled PyTorch base image with development tools
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

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
        sudo \
        unzip \
        wget \
        && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Some workflow runners call `python` explicitly; ensure it exists.
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create working directory
WORKDIR /workspace/VideoPainter


# Copy only dependency manifests first (better layer caching)
COPY requirements.txt /workspace/VideoPainter/requirements.txt


# Upgrade pip and install Python dependencies
# Install setuptools first to ensure pkg_resources is available for git-based packages
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir setuptools && \
    pip install --no-cache-dir numpy==1.26.0 cython && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt && \
    pip install --no-cache-dir psutil && \
    python3 -c "import transformers; print('transformers', transformers.__version__)" && \
    python3 -c "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; print('qwen2_5_vl import ok')" && \
    python3 -c "import accelerate; print('accelerate', accelerate.__version__)" && \
    python3 -c "import qwen_vl_utils; print('qwen_vl_utils', getattr(qwen_vl_utils, '__version__', 'unknown'))" && \
    python3 -m pip install --no-cache-dir "flytekit>=1.13.0"

# Note: Qwen2.5-VL-72B-Instruct model (~140GB) will be downloaded from HuggingFace on first run
# Model cache location: /root/.cache/huggingface

# Install local editable packages (need sources present)
COPY diffusers/ /workspace/VideoPainter/diffusers/
RUN cd diffusers && pip install -e .

# Optional: install app (for gradio demo)
COPY app/ /workspace/VideoPainter/app/
RUN cd app && pip install -e . --no-build-isolation || true

COPY hlx_wf-2.4.2.dev9-py3-none-any.whl /workspace/VideoPainter//

# Install the wheel
RUN python3 -m pip install /workspace/VideoPainter/hlx_wf-2.4.2.dev9-py3-none-any.whl

# Pre-download evaluation model weights so they're baked into the image
# and don't need to be fetched at runtime (saves ~2 min per pod start).
#   1. squeezenet1_1 (4.7 MB)  – LPIPS backbone
#   2. CLIP ViT-B/32 (338 MB)  – temporal consistency
RUN python3 -c "\
import torch, os; \
torch.hub.load_state_dict_from_url( \
    'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth', \
    map_location='cpu'); \
print('squeezenet cached at', os.path.expanduser('~/.cache/torch/hub/checkpoints/squeezenet1_1-b8a52dc0.pth'))"

# Copy project files (at the end)
COPY . /workspace/VideoPainter

# Expose gradio port
EXPOSE 7860

# Default command
CMD ["/bin/bash"]