#!/bin/bash

set -euo pipefail

# FluxFill dataset filtering/sorting – runs 5 filter combinations in sequence.
# Builds the Docker image once, then submits one HLX workflow per combination.

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2:${TIMESTAMP}"
TEAM_SPACE="research"
DOMAIN="prod"

# Input datasets (output of data_generation.py) – comma-separated for merging
DATA_BASE="gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data"
INPUT_DIR="${DATA_BASE}/trainingdata_chunk00_25,${DATA_BASE}/trainingdata_chunk26_49"

# CSV ordering + optional cap
SORT="prompt"    # prompt|image
LIMIT="0"        # 0 means no limit

# Keep only samples with clear lane markings (no 'unknown' attributes)
REQUIRE_CLEAR_ROAD="1"   # 1|0

# ── 5 filter combinations ────────────────────────────────────────────────────
#  FORMAT: "COUNT COLOR PATTERN"
COMBOS=(
	"single white  solid"
	"double white  solid"
	"single yellow solid"
	"double yellow solid"
	"single white  dashed"
)

# Ensure we run from the sam2 repo root (docker-compose.yaml lives there)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# ── Build & push image (once) ────────────────────────────────────────────────
echo "Building docker image (sam2/frontend)..."
docker compose build

echo "Tagging and pushing image: ${REMOTE_IMAGE}"
docker tag sam2/frontend "${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

echo "Image built with timestamp: ${TIMESTAMP}"

export SAM2_CONTAINER_IMAGE="${REMOTE_IMAGE}"
export FLUXFILL_DATA_CONTAINER_IMAGE="${REMOTE_IMAGE}"

# ── Submit one workflow per combination ───────────────────────────────────────
for i in "${!COMBOS[@]}"; do
	read -r COUNT COLOR PATTERN <<< "${COMBOS[$i]}"
	SUFFIX="${COUNT}_${COLOR}_${PATTERN}"
	OUTPUT_DIR="${DATA_BASE}/chunk00_49_${SUFFIX}"

	echo ""
	echo "================================================================================"
	echo " [$(( i + 1 ))/${#COMBOS[@]}]  ${COUNT} / ${COLOR} / ${PATTERN}"
	echo "   Input  : ${INPUT_DIR}"
	echo "   Output : ${OUTPUT_DIR}"
	echo "================================================================================"

	hlx wf run \
		--team-space "${TEAM_SPACE}" \
		--domain "${DOMAIN}" \
		--execution-name "filter-fluxfill-${SUFFIX//[_]/-}-${TIMESTAMP}" \
		filter_fluxfill_dataset.filter_fluxfill_dataset_wf \
		--input_dir "${INPUT_DIR}" \
		--suffix "${SUFFIX}" \
		--output_dir "${OUTPUT_DIR}" \
		--count "${COUNT}" \
		--color "${COLOR}" \
		--pattern "${PATTERN}" \
		--sort "${SORT}" \
		--limit "${LIMIT}" \
		--require_clear_road "${REQUIRE_CLEAR_ROAD}"

	echo "  → Submitted: ${OUTPUT_DIR}/"
done

echo ""
echo "All ${#COMBOS[@]} filter workflows submitted."
echo "Output folders:"
for i in "${!COMBOS[@]}"; do
	read -r COUNT COLOR PATTERN <<< "${COMBOS[$i]}"
	echo "  ${DATA_BASE}/chunk00_49_${COUNT}_${COLOR}_${PATTERN}/"
done

