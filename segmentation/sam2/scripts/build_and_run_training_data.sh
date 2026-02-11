#!/bin/bash

set -euo pipefail

# FluxFill training data generation (mp4 -> first frame -> SAM2 mask -> Qwen caption)
# Hard-coded config: edit these constants when you want a different slice.
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2"
TEAM_SPACE="research"
DOMAIN="prod"

NUM_VIDEOS="10000"
START_INDEX="0"
OUTPUT_RUN_ID="trainingdata_chunk199"

DOWNLOAD_BATCH_SIZE="100"

# Frame numbers (1-based) to extract per video. Comma-separated.
FRAME_NUMBERS="1,100,200,300,400,500"

# Chunk range: process chunk_0000 to chunk_0200 (inclusive)
# Each chunk folder contains 100+ mp4 files
CHUNK_START="0"
CHUNK_END="199"

QWEN_DEVICE="cuda:0"
SAM2_DEVICE="cuda:0"

# 0 means: keep walking the GCS prefix until enough .mp4s are found (or the prefix is exhausted).
MAX_WALK_FILES="0"

# Ensure we run from the sam2 repo root (docker-compose.yaml lives there)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Building docker image (sam2/frontend)..."
docker compose build

echo "Tagging and pushing image: ${REMOTE_IMAGE}"
docker tag sam2/frontend "${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

# Ensure both workflows use the just-pushed image
export SAM2_CONTAINER_IMAGE="${REMOTE_IMAGE}"
export FLUXFILL_DATA_CONTAINER_IMAGE="${REMOTE_IMAGE}"

echo "Running HLX workflow: data_generation.fluxfill_data_generation_wf"
echo "  NUM_VIDEOS=${NUM_VIDEOS} START_INDEX=${START_INDEX} OUTPUT_RUN_ID=${OUTPUT_RUN_ID}"
echo "  DOWNLOAD_BATCH_SIZE=${DOWNLOAD_BATCH_SIZE}"
echo "  FRAME_NUMBERS=${FRAME_NUMBERS}"
echo "  CHUNK_START=${CHUNK_START} CHUNK_END=${CHUNK_END}"

hlx wf run \
	--team-space "${TEAM_SPACE}" \
	--domain "${DOMAIN}" \
	data_generation.fluxfill_data_generation_wf \
	--num_videos "${NUM_VIDEOS}" \
	--start_index "${START_INDEX}" \
	--output_run_id "${OUTPUT_RUN_ID}" \
	--qwen_device "${QWEN_DEVICE}" \
	--sam2_device "${SAM2_DEVICE}" \
	--max_walk_files "${MAX_WALK_FILES}" \
	--download_batch_size "${DOWNLOAD_BATCH_SIZE}" \
	--frame_numbers "${FRAME_NUMBERS}" \
	--chunk_start "${CHUNK_START}" \
	--chunk_end "${CHUNK_END}"

echo "If successful, data is under:"
echo "  gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data/${OUTPUT_RUN_ID}/"

