# Master Pipeline Orchestrator Dockerfile
# This is a lightweight container for orchestrating the three-stage pipeline via HLX workflows
# It does NOT contain the heavy ML dependencies (those are in individual stage containers)

FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install additional Python dependencies for workflow orchestration
RUN pip install --no-cache-dir \
    google-cloud-storage \
    gcsfs \
    pyyaml \
    requests

# Copy the entire project structure (for workflow imports)
COPY . /workspace/

# Install HLX workflow package from the copied files
RUN pip install --no-cache-dir /workspace/segmentation/sam2/hlx_wf-2.4.2.dev9-py3-none-any.whl

# Create a convenience script for running the master workflow
RUN echo '#!/bin/bash\n\
set -euo pipefail\n\
echo "Master Pipeline Orchestrator"\n\
echo "Available commands:"\n\
echo "  - Run master workflow: hlx wf run --team-space research --domain prod workflow.master_pipeline_wf [args]"\n\
echo "  - Run individual workflows: cd segmentation/sam2 && hlx wf run ..."\n\
echo ""\n\
exec "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
