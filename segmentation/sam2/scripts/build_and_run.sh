#!/bin/bash
# filepath: build_and_run.sh

set -euo pipefail



# Build the Docker image
docker compose build

# Tag the image for Google Artifact Registry
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2"
docker tag sam2/frontend "${REMOTE_IMAGE}"

# Push the image to the registry
docker push "${REMOTE_IMAGE}"

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

echo "Timing report will be uploaded as: gs://.../sam2_final_output/12/12.txt"