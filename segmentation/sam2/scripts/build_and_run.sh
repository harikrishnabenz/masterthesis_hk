#!/bin/bash
# filepath: build_and_run.sh

set -euo pipefail

# ----------------------------------------------------------------------------------
# OUTPUT FOLDER (override: SAM2_OUTPUT_BASE="gs://..." bash scripts/build_and_run.sh)
# ----------------------------------------------------------------------------------
SAM2_OUTPUT_BASE="${SAM2_OUTPUT_BASE:-gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/output/sam2}"
export SAM2_OUTPUT_BASE

# Generate timestamp for unique image tagging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build the Docker image
docker compose build

# Tag the image for Google Artifact Registry
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2:${TIMESTAMP}"
docker tag sam2/frontend "${REMOTE_IMAGE}"

# Push the image to the registry
docker push "${REMOTE_IMAGE}"

echo "Image built with timestamp: ${TIMESTAMP}"
echo "Output folder: ${SAM2_OUTPUT_BASE}"

# Ensure workflow.py uses this exact image when hlx packages the workflow
export SAM2_CONTAINER_IMAGE="${REMOTE_IMAGE}"

# Run the HLX workflow for SAM2 segmentation
# Processes videos with SAM2, generates masks, and uploads to GCS in two formats:
#   1. Raw outputs (masks + visualizations + segmented videos)
#   2. VideoPainter preprocessed format (for video inpainting)

hlx wf run \
  --team-space research \
  --domain prod \
  workflow.sam2_segmentation_wf \
  --run_id "10_150f_caption_fps8" \
  --max_frames "150"

echo "Timing report will be uploaded as: ${SAM2_OUTPUT_BASE}/12/12.txt"