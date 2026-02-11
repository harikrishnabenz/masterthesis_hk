#!/bin/bash

set -euo pipefail

# FluxFill dataset filtering/sorting (reads an existing run folder and creates a new one with suffix)
# Hard-coded config: edit these constants when you want a different selection.

REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2"
TEAM_SPACE="research"
DOMAIN="prod"

# Input dataset (output of data_generation.py)
INPUT_DIR="gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data/test_data_10000_7b"

# Filtering instruction
COUNT="single"      # single|double|unknown|any
COLOR="white"      # white|yellow|mixed|unknown|any
PATTERN="solid"    # solid|dashed|mixed|unknown|any

# Output naming
# Automatically name output by the filter instruction.
SUFFIX="${COUNT}_${COLOR}_${PATTERN}"
# If OUTPUT_DIR is empty, workflow writes to <INPUT_DIR>__<SUFFIX>
OUTPUT_DIR=""

# CSV ordering + optional cap
SORT="prompt"    # prompt|image
LIMIT="0"        # 0 means no limit

# Ensure we run from the sam2 repo root (docker-compose.yaml lives there)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Building docker image (sam2/frontend)..."
docker compose build

echo "Tagging and pushing image: ${REMOTE_IMAGE}"
docker tag sam2/frontend "${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

# Ensure workflow uses the just-pushed image
export SAM2_CONTAINER_IMAGE="${REMOTE_IMAGE}"
export FLUXFILL_DATA_CONTAINER_IMAGE="${REMOTE_IMAGE}"

echo "Running HLX workflow: filter_fluxfill_dataset.filter_fluxfill_dataset_wf"
echo "  INPUT_DIR=${INPUT_DIR} SUFFIX=${SUFFIX} OUTPUT_DIR=${OUTPUT_DIR}"
echo "  COUNT=${COUNT} COLOR=${COLOR} PATTERN=${PATTERN} SORT=${SORT} LIMIT=${LIMIT}"

hlx wf run \
	--team-space "${TEAM_SPACE}" \
	--domain "${DOMAIN}" \
	filter_fluxfill_dataset.filter_fluxfill_dataset_wf \
	--input_dir "${INPUT_DIR}" \
	--suffix "${SUFFIX}" \
	--output_dir "${OUTPUT_DIR}" \
	--count "${COUNT}" \
	--color "${COLOR}" \
	--pattern "${PATTERN}" \
	--sort "${SORT}" \
	--limit "${LIMIT}"

echo "If successful, filtered data is under:"
if [[ -n "${OUTPUT_DIR}" ]]; then
	echo "  ${OUTPUT_DIR}/"
else
	echo "  ${INPUT_DIR}__${SUFFIX}/"
fi

