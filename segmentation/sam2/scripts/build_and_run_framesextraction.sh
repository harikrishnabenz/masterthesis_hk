#!/bin/bash
# Extract frames from driving videos (chunks 0000–0025) and upload to GCS.
# Uses the SAM2 Docker image (has ffmpeg + Python).

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# HLX execution names must match ^[a-z][a-z\-0-9]*$ — no underscores
EXEC_TIMESTAMP=$(echo "${TIMESTAMP}" | tr '_' '-')
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2:${TIMESTAMP}"
TEAM_SPACE="research"
DOMAIN="prod"

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_FOLDER="${OUTPUT_FOLDER:-trainingdata_chunk00_25}"
CHUNK_START="${CHUNK_START:-0}"
CHUNK_END="${CHUNK_END:-25}"
FRAME_NUMBERS="${FRAME_NUMBERS:-1,100,200,300,400,500}"
NUM_WORKERS="${NUM_WORKERS:-28}"

# Ensure we run from the sam2 repo root (docker-compose.yaml lives there)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Building docker image (sam2/frontend)..."
docker compose build

echo "Tagging and pushing image: ${REMOTE_IMAGE}"
docker tag sam2/frontend "${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

export SAM2_CONTAINER_IMAGE="${REMOTE_IMAGE}"

echo "================================================================================"
echo " FRAME EXTRACTION"
echo "================================================================================"
echo "  Output folder:  ${OUTPUT_FOLDER}"
echo "  Chunks:         ${CHUNK_START} – ${CHUNK_END}"
echo "  Frame numbers:  ${FRAME_NUMBERS}"
echo "  Workers:        ${NUM_WORKERS}"
echo "  Docker image:   ${REMOTE_IMAGE}"
echo "================================================================================"

hlx wf run \
    --team-space "${TEAM_SPACE}" \
    --domain "${DOMAIN}" \
    --execution-name "frames-${EXEC_TIMESTAMP}" \
    workflow_framesextraction.extract_frames_wf \
    --output_folder_name "${OUTPUT_FOLDER}" \
    --chunk_start "${CHUNK_START}" \
    --chunk_end "${CHUNK_END}" \
    --frame_numbers "${FRAME_NUMBERS}" \
    --num_workers "${NUM_WORKERS}"

echo ""
echo "Workflow submitted. Output will be at:"
echo "  gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data/${OUTPUT_FOLDER}/images/"
