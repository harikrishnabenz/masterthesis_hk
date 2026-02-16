# Docker Setup for Alpamayo

This document provides instructions for running Alpamayo in a Docker container with GPU support.

## Prerequisites

- Docker (>= 20.10)
- NVIDIA Docker runtime ([nvidia-docker](https://github.com/NVIDIA/nvidia-docker))
- NVIDIA GPU with ≥24 GB VRAM
- Docker Compose (optional, for easier management)

## Verify NVIDIA Docker Support

Before building, verify that your system supports NVIDIA Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Building the Docker Image

### Option 1: Using Docker

```bash
docker build -t alpamayo:latest .
```

### Option 2: Using Docker Compose

```bash
docker-compose build
```

## Running the Container

### Option 1: Using Docker

```bash
docker run --gpus all -it \
  -v $(pwd):/workspace/alpamayo \
  -v $(pwd)/cache:/workspace/cache \
  -p 8888:8888 \
  --shm-size=16g \
  alpamayo:latest
```

### Option 2: Using Docker Compose

```bash
docker-compose up -d
docker-compose exec alpamayo bash
```

## HuggingFace Authentication

Once inside the container, you need to authenticate with HuggingFace:

```bash
# Install huggingface_hub (if not already installed)
pip install huggingface_hub

# Login with your token
huggingface-cli login
```

Get your access token at: https://huggingface.co/settings/tokens

Make sure you have requested access to:
- 🤗 [Physical AI AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)

## Running Inference

Inside the container:

```bash
# Activate the virtual environment (should be automatic)
source ar1_venv/bin/activate

# Run the test inference script
python src/alpamayo_r1/test_inference.py
```

## Running Jupyter Notebook

To run the interactive notebook:

```bash
# Inside the container
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then access it from your browser at: `http://localhost:8888`

## Stopping the Container

### If using Docker Compose:

```bash
docker-compose down
```

### If using Docker directly:

```bash
docker stop <container_id>
```

## Troubleshooting

### GPU Not Detected

If the GPU is not detected inside the container:
1. Verify NVIDIA Docker runtime is installed: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
2. Check Docker daemon configuration at `/etc/docker/daemon.json`
3. Restart Docker daemon: `sudo systemctl restart docker`

### Out of Memory Errors

If you encounter CUDA OOM errors:
1. Ensure your GPU has at least 24 GB VRAM
2. Close other GPU-intensive applications
3. Reduce `num_traj_samples` in the inference script

### Slow Model Download

The model weights are 22 GB and will be downloaded on first run. This can take several minutes depending on your network speed.

## Cache Management

Model weights and datasets are cached in the `./cache` directory (mounted volume). This prevents re-downloading on container restarts.

To clear the cache:

```bash
rm -rf ./cache/huggingface
```
