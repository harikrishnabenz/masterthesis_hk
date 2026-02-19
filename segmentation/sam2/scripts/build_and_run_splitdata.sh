#!/bin/bash
# ==============================================================================
# DATA SPLITTING — Build and Run Script
# ==============================================================================
#
# Reuses the SAM2 container to run the data-splitting workflow.
# The workflow copies preprocessed VP data to a new GCS folder whose name
# encodes the video count:
#
#   preprocessed_data_vp/<SOURCE_FOLDER>
#       → preprocessed_data_vp/<RUN_ID>_<N>videos
#
# Usage:
#   bash segmentation/sam2/scripts/build_and_run_splitdata.sh
#
# Overrides:
#   SOURCE_FOLDER=006_20260219_135038  (default)
#   NUM_SPLITS=3                       (default: 1 = no split, just copy)
#   RUN_ID=006                         (default: extracted from SOURCE_FOLDER)
#
# Examples:
#   # Copy all to a single folder named with video count
#   bash segmentation/sam2/scripts/build_and_run_splitdata.sh
#
#   # Split into 3 parts (e.g. 10 videos → 4+3+3)
#   NUM_SPLITS=3 bash segmentation/sam2/scripts/build_and_run_splitdata.sh
# ==============================================================================

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SAM2_DIR}/../.." && pwd)"

# ── Configuration ─────────────────────────────────────────────────────────────
GCS_BUCKET="mbadas-sandbox-research-9bb9c7f"















SOURCE_FOLDER="${SOURCE_FOLDER:-006_20260219_135038}"

# Number of parts to split the dataset into.
# Examples:  NUM_SPLITS=1 → copy all (no split)
#            NUM_SPLITS=3 → 3 roughly-equal parts
#            NUM_SPLITS=5 → 5 roughly-equal parts
NUM_SPLITS="${NUM_SPLITS:-5}"
























RUN_ID="${RUN_ID:-${SOURCE_FOLDER%%_*}}"              # first token before '_'
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_RUN_ID="${RUN_ID}_${RUN_TIMESTAMP}"

PREPROCESSED_BASE="gs://${GCS_BUCKET}/workspace/user/hbaskar/outputs/preprocessed_data_vp"

# ── Docker image registry (same as SAM2) ─────────────────────────────────────
REGISTRY="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research"
SAM2_LOCAL_IMAGE="sam2/frontend"
# Image name: harimt_sam2_datasplit_<run_id>_<datetime>:<run_id>_<datetime>
SAM2_REMOTE_IMAGE="${REGISTRY}/harimt_sam2_datasplit_${RUN_ID}_${RUN_TIMESTAMP}"
SAM2_TAGGED="${SAM2_REMOTE_IMAGE}:${RUN_ID}_${RUN_TIMESTAMP}"

# ── Print summary ─────────────────────────────────────────────────────────────
echo "================================================================================"
echo " DATA SPLITTING — BUILD AND RUN"
echo "================================================================================"
echo ""
echo "  SOURCE_FOLDER:      ${SOURCE_FOLDER}"
echo "  NUM_SPLITS:         ${NUM_SPLITS}"
echo "  RUN_ID:             ${RUN_ID}"
echo "  MASTER_RUN_ID:      ${MASTER_RUN_ID}"
echo "  PREPROCESSED_BASE:  ${PREPROCESSED_BASE}"
echo "  Source path:        ${PREPROCESSED_BASE}/${SOURCE_FOLDER}/"
echo "  Docker image:       ${SAM2_TAGGED}"
echo ""
echo "================================================================================"

# ── Build & push the SAM2 image ──────────────────────────────────────────────
echo ""
echo "▸ Building SAM2 image (reused for data splitting) …"
pushd "${SAM2_DIR}" > /dev/null
docker compose build
docker tag "${SAM2_LOCAL_IMAGE}" "${SAM2_TAGGED}"
docker tag "${SAM2_LOCAL_IMAGE}" "${SAM2_REMOTE_IMAGE}:latest"
docker push "${SAM2_TAGGED}"
docker push "${SAM2_REMOTE_IMAGE}:latest"
popd > /dev/null
echo "  ✓ Image pushed: ${SAM2_TAGGED}"

# ── Export env vars for workflow serialisation ────────────────────────────────
export SAM2_CONTAINER_IMAGE="${SAM2_TAGGED}"
export PREPROCESSED_BASE

# ── Submit the workflow ──────────────────────────────────────────────────────
echo ""
echo "▸ Submitting data-splitting workflow …"
cd "${SAM2_DIR}"

hlx wf run \
  --team-space research \
  --domain prod \
  --execution-name "datasplit-${MASTER_RUN_ID//_/-}" \
  workflow_datasplitting.datasplitting_wf \
  --run_id "${MASTER_RUN_ID}" \
  --source_folder "${SOURCE_FOLDER}" \
  --num_splits "${NUM_SPLITS}" \
  --preprocessed_base "${PREPROCESSED_BASE}"

echo ""
echo "================================================================================"
echo " WORKFLOW SUBMITTED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "  MASTER_RUN_ID:   ${MASTER_RUN_ID}"
echo "  NUM_SPLITS:      ${NUM_SPLITS}"
echo "  Source:           ${PREPROCESSED_BASE}/${SOURCE_FOLDER}/"
if [[ "${NUM_SPLITS}" -eq 1 ]]; then
  echo "  Destination:      ${PREPROCESSED_BASE}/${MASTER_RUN_ID}_<N>videos/"
else
  echo "  Destinations:     ${PREPROCESSED_BASE}/${MASTER_RUN_ID}_<M>videos_part1of${NUM_SPLITS}/"
  echo "                    …"
  echo "                    ${PREPROCESSED_BASE}/${MASTER_RUN_ID}_<M>videos_part${NUM_SPLITS}of${NUM_SPLITS}/"
fi
echo "    (N = total videos, M = videos per part)"
echo ""
echo "================================================================================"
