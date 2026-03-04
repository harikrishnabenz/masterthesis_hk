#!/bin/bash
# ==================================================================================
# MASTER PIPELINE — Build and Run Script (v2 — Playground Bucket)
# ==================================================================================
#
# Orchestrates a flexible multi-stage pipeline:
#
#   Stage 1  SAM2 Segmentation
#            Input : raw driving videos (GCS / chunks:// URI)
#            Output: masks + VP-preprocessed data
#
#   Stage 2  VideoPainter Editing
#            Input : SAM2 preprocessed data
#            Output: inpainted / edited videos
#
#   Stage 3  Alpamayo VLA Inference
#            Input : VP edited videos
#            Output: trajectory predictions + reasoning
#
# A single RUN_ID + RUN_TIMESTAMP is shared across all stages.
#
# Usage — select stages with STAGES=<digits>:
#   STAGES=1   bash scripts/build_and_run_2.sh   # SAM2 only (needs chunk config)
#   STAGES=2   SAM2_DATA_RUN_ID=003_…  bash scripts/build_and_run_2.sh  # VP only
#   STAGES=3   VP_DATA_RUN_ID=003_…  bash scripts/build_and_run_2.sh  # Alp only
#   STAGES=12  bash scripts/build_and_run_2.sh   # SAM2 → VP (needs chunk config)
#   STAGES=23  SAM2_DATA_RUN_ID=003_…  bash scripts/build_and_run_2.sh  # VP → Alp
#   STAGES=123 bash scripts/build_and_run_2.sh   # Full pipeline (needs chunk config)
#
# Other options:
#   RUN_ID=002 bash scripts/build_and_run_2.sh   # custom run id
# ==================================================================================

set -euo pipefail

# ── Resolve repo root (this script lives in scripts/) ────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ==============================================================================
# SHARED CONFIGURATION
# ==============================================================================
GCS_BUCKET="mbadas-sandbox-playground-44e6bd8/genai_videopainter"
ALL_CKPT_BASE="gs://${GCS_BUCKET}/all_checkpoints"

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${RUN_ID:-chong}"
MASTER_RUN_ID="${RUN_ID}_${RUN_TIMESTAMP}"

STAGES="${STAGES:-123b}"         # default: full pipeline with SAM3

# ── Derive SAM model from STAGES suffix (a=sam2, b=sam3) ─────────────────────
STAGES_NUM="${STAGES//[ab]/}"    # strip suffix → pure digits (e.g. 123b → 123)
STAGES_SUFFIX="${STAGES//[0-9]/}" # strip digits → suffix only (e.g. 123b → b)

if [[ "${STAGES_SUFFIX}" == "a" ]]; then
    SAM_MODEL="sam2"
elif [[ "${STAGES_SUFFIX}" == "b" ]]; then
    SAM_MODEL="sam3"
elif [[ -z "${STAGES_SUFFIX}" ]]; then
    # No suffix — stages without SAM (2, 23, 3)
    SAM_MODEL="none"
else
    echo "ERROR: STAGES='${STAGES}' has invalid suffix '${STAGES_SUFFIX}'. Use 'a' (SAM2) or 'b' (SAM3)."
    exit 1
fi

# SAM3 text prompt for concept-based segmentation (only used when SAM_MODEL=sam3)
SAM3_TEXT_PROMPT="${SAM3_TEXT_PROMPT:-road surface}"


# ===============================================================================
# MODEL CHECKPOINT GCS FOLDER PATHS (Mounted folders in containers)
# These must match the FuseBucket prefixes used in workflow_master.py
# ===============================================================================

# GCS folder for SAM2 model checkpoints
SAM2_MODEL_GCS_CKPT_FOLDER="${ALL_CKPT_BASE}/sam2-hiera-large/sam2-hiera-large"

# GCS folder for SAM3 model checkpoints (shares the sam2-hiera-large parent)
SAM3_MODEL_GCS_CKPT_FOLDER="${ALL_CKPT_BASE}/sam2-hiera-large/sam3"

# GCS folder for VideoPainter model checkpoints
VIDEOPAINTER_MODEL_GCS_CKPT_FOLDER="${ALL_CKPT_BASE}/VideoPainter/VideoPainter"

# GCS folder for CogVideoX-5b-I2V checkpoint (used by VideoPainter)
COGVIDEOX_MODEL_GCS_CKPT_FOLDER="${ALL_CKPT_BASE}/CogVideoX-5b-I2V/CogVideoX-5b-I2V"

# GCS folder for VideoPainter VLM (Qwen) checkpoint
VIDEOPAINTER_VLM_GCS_CKPT_FOLDER="${ALL_CKPT_BASE}/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct"

# GCS folder for CLIP checkpoint
CLIP_MODEL_GCS_CKPT_FOLDER="${ALL_CKPT_BASE}/clip-vit-large-patch14/clip-vit-large-patch14"

# GCS folder for FluxFill inpainting checkpoint
FLUXFILL_MODEL_GCS_CKPT_FOLDER="${ALL_CKPT_BASE}/flux_inp/flux_inp"

# GCS folder for VideoPainter LoRA checkpoints (per-prompt)
VIDEOPAINTER_LORA_GCS_CKPT_FOLDER="gs://${GCS_BUCKET}/training/trained_checkpoint"

# GCS folder for Alpamayo model checkpoints
ALPAMAYO_MODEL_GCS_CKPT_FOLDER="${ALL_CKPT_BASE}/alpamayo-r1-10b/alpamayo-r1-10b"


# ==============================================================================
# INPUT CONFIGURATION — set these before running
# ==============================================================================

# ── Stage selection ─
# Suffix a = SAM2, b = SAM3 (for stages that include SAM):
#   1a, 1b, 12a, 12b, 123a, 123b
# Stages without SAM need no suffix: 2, 23, 3

# ── Stage 1 (SAM2/SAM3) inputs ───────────────────────────────────────────────
SAM2_CHUNK_START="${SAM2_CHUNK_START:-1}"
SAM2_CHUNK_END="${SAM2_CHUNK_END:-1}"
SAM2_FILES_PER_CHUNK="${SAM2_FILES_PER_CHUNK:-1}"

# ── Stage 2 (VP) input — required when running VP without SAM2 (STAGES=2,23) ─
SAM2_DATA_RUN_ID="${SAM2_DATA_RUN_ID:-}"

# ── Stage 3 (Alpamayo) input — required when running Alp without VP (STAGES=3)
VP_DATA_RUN_ID="${VP_DATA_RUN_ID:-}"


# Parse which stages are active (use STAGES_NUM, the numeric part)
RUN_SAM2=0; RUN_SAM3=0; RUN_SAM=0; RUN_VP=0; RUN_ALP=0
if [[ "${STAGES_NUM}" == *1* ]]; then
    RUN_SAM=1
    if [[ "${SAM_MODEL}" == "sam3" ]]; then
        RUN_SAM3=1
    elif [[ "${SAM_MODEL}" == "sam2" ]]; then
        RUN_SAM2=1
    fi
fi
[[ "${STAGES_NUM}" == *2* ]] && RUN_VP=1
[[ "${STAGES_NUM}" == *3* ]] && RUN_ALP=1

# Validate STAGES format: digits + optional a/b suffix
if [[ ! "${STAGES}" =~ ^[123]+[ab]?$ ]]; then
    echo "ERROR: STAGES='${STAGES}' is invalid."
    echo "       Use: 1a, 1b, 12a, 12b, 123a, 123b, 2, 23, 3"
    exit 1
fi

# Stages that include SAM (1) must have a/b suffix
if [[ ${RUN_SAM} -eq 1 && -z "${STAGES_SUFFIX}" ]]; then
    echo "ERROR: STAGES='${STAGES}' includes SAM (1) but has no suffix."
    echo "       Use 'a' for SAM2 or 'b' for SAM3 (e.g. 1a, 12b, 123a)."
    exit 1
fi

# ── Input folder dependency checks ───────────────────────────────────────────
if [[ ${RUN_VP} -eq 1 && ${RUN_SAM} -eq 0 && -z "${SAM2_DATA_RUN_ID}" ]]; then
    echo "ERROR: STAGES=${STAGES} includes VP (2) without SAM (1)."
    echo "       You must set SAM2_DATA_RUN_ID (the SAM output folder name)."
    echo "       Example: SAM2_DATA_RUN_ID=003_20260217_162441 STAGES=${STAGES} bash scripts/build_and_run_2.sh"
    exit 1
fi

if [[ ${RUN_ALP} -eq 1 && ${RUN_VP} -eq 0 && -z "${VP_DATA_RUN_ID}" ]]; then
    echo "ERROR: STAGES=${STAGES} includes Alpamayo (3) without VP (2)."
    echo "       You must set VP_DATA_RUN_ID (the VP output folder name)."
    echo "       Example: VP_DATA_RUN_ID=003_20260217_162441 STAGES=${STAGES} bash scripts/build_and_run_2.sh"
    exit 1
fi

# ==============================================================================
# STAGE 1: SAM SEGMENTATION CONFIGURATION (SAM2 or SAM3)
# ==============================================================================
SAM2_INPUT_PARENT="${SAM2_INPUT_PARENT:-gs://${GCS_BUCKET}/input}"
SAM2_CAMERA_SUBFOLDER="${SAM2_CAMERA_SUBFOLDER:-camera_front_tele_30fov}"
SAM2_INPUT_BASE="${SAM2_INPUT_BASE:-${SAM2_INPUT_PARENT}/${SAM2_CAMERA_SUBFOLDER}}"
SAM2_MAX_FRAMES="${SAM2_MAX_FRAMES:-100}"

SAM2_VIDEO_URIS="${SAM2_VIDEO_URIS:-chunks://${SAM2_INPUT_BASE#gs://}?start=${SAM2_CHUNK_START}&end=${SAM2_CHUNK_END}&per_chunk=${SAM2_FILES_PER_CHUNK}}"

SAM2_OUTPUT_BASE="${SAM2_OUTPUT_BASE:-gs://${GCS_BUCKET}/output/sam2}"
SAM2_PREPROCESSED_OUTPUT_BASE="${SAM2_PREPROCESSED_OUTPUT_BASE:-gs://${GCS_BUCKET}/output/preprocessed_data_vp}"

# SAM3-specific output base (raw sam3 outputs separate from sam2)
SAM3_OUTPUT_BASE="${SAM3_OUTPUT_BASE:-gs://${GCS_BUCKET}/output/sam3}"

# Calculate expected video count for display
SAM2_TOTAL_CHUNKS=$(( SAM2_CHUNK_END - SAM2_CHUNK_START + 1 ))
SAM2_EXPECTED_VIDEOS=$(( SAM2_TOTAL_CHUNKS * SAM2_FILES_PER_CHUNK ))

# ==============================================================================
# STAGE 2: VIDEOPAINTER CONFIGURATION
# ==============================================================================
VP_OUTPUT_BASE="${VP_OUTPUT_BASE:-gs://${GCS_BUCKET}/output/vp}"


# ── Trained FluxFill LoRA checkpoints (per-prompt) ────────────────────────────
# Base path under the GCS bucket where all checkpoint folders live.
LORA_CKPT_BASE="genai_videopainter/training/trained_checkpoint"

# Per-prompt checkpoint folders.  Keys match the PROMPTS array (1-5).
# Override individual paths:
#   LORA_CKPT_1="genai_videopainter/training/..." bash scripts/build_and_run_2.sh
declare -A LORA_CHECKPOINTS
LORA_CHECKPOINTS[1]="${LORA_CKPT_1:-${LORA_CKPT_BASE}/fluxfill_single_white_solid_20260222_201003}"    # Prompt 1 — single solid white
LORA_CHECKPOINTS[2]="${LORA_CKPT_2:-${LORA_CKPT_BASE}/fluxfill_double_white_solid_20260222_201003}"    # Prompt 2 — double solid white
LORA_CHECKPOINTS[3]="${LORA_CKPT_3:-${LORA_CKPT_BASE}/fluxfill_single_yellow_solid_20260222_201003}"   # Prompt 3 — single solid yellow
LORA_CHECKPOINTS[4]="${LORA_CKPT_4:-${LORA_CKPT_BASE}/fluxfill_double_yellow_solid_20260222_201003}"   # Prompt 4 — double solid yellow
LORA_CHECKPOINTS[5]="${LORA_CKPT_5:-${LORA_CKPT_BASE}/fluxfill_single_white_dashed_20260222_201003}"   # Prompt 5 — single dashed white

# Legacy single-path fallback (used when lora_scale == 0, i.e. no LoRA).
# When lora_scale > 0 the per-prompt LORA_CHECKPOINTS are used instead.
TRAINED_FLUXFILL_GCS_PATH="${TRAINED_FLUXFILL_GCS_PATH:-${LORA_CHECKPOINTS[1]}}"

# VP run suffix (used in Docker image naming) — set after NUM_PROMPTS is computed below
CHECKPOINT_TIMESTAMP=$(basename "${LORA_CHECKPOINTS[1]}" | grep -oE '[0-9]{8}_[0-9]{6}' | head -1 || true)

# LLM model path inside VP container
VP_LLM_MODEL="${VP_LLM_MODEL:-/workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct}"


# ----------------------------------------------------------------------------------
# PROMPT SELECTION
# ----------------------------------------------------------------------------------
# Pass PROMPT_IDS to choose which editing prompts to run.  Examples:
#   PROMPT_IDS=1        → prompt 1 only
#   PROMPT_IDS=123      → prompts 1, 2, 3
#   PROMPT_IDS=15       → prompts 1 and 5
#   PROMPT_IDS=12345    → all five (default)
#
# OR pass a custom prompt directly (when PROMPT_IDS is not set):
#   CUSTOM_PROMPT='Your custom editing instruction here' bash scripts/build_and_run_2.sh
PROMPT_IDS="${PROMPT_IDS:-4}"
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"


declare -A PROMPTS
PROMPTS[1]='Single solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'
PROMPTS[2]='Double solid white continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'
PROMPTS[3]='Single solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'
PROMPTS[4]='Double solid yellow continuous line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'
PROMPTS[5]='Single dashed white intermitted line, aligned exactly to the original lane positions and perspective; keep road texture, lighting, and shadows unchanged'

# Build VIDEO_EDITING_INSTRUCTIONS from selected prompt IDs or CUSTOM_PROMPT
VIDEO_EDITING_INSTRUCTIONS=""
if [[ -n "${PROMPT_IDS}" ]]; then
  # Use predefined prompts by ID
  for (( i=0; i<${#PROMPT_IDS}; i++ )); do
    pid="${PROMPT_IDS:$i:1}"
    if [[ -z "${PROMPTS[$pid]+x}" ]]; then
      echo "ERROR: Invalid prompt ID '${pid}'. Valid IDs are 1-5."
      exit 1
    fi
    if [[ -n "${VIDEO_EDITING_INSTRUCTIONS}" ]]; then
      VIDEO_EDITING_INSTRUCTIONS+=$'\n'
    fi
    VIDEO_EDITING_INSTRUCTIONS+="${PROMPTS[$pid]}"
  done
  NUM_PROMPTS=${#PROMPT_IDS}
elif [[ -n "${CUSTOM_PROMPT}" ]]; then
  # Use custom prompt directly
  echo "Using CUSTOM_PROMPT: ${CUSTOM_PROMPT:0:80}…"
  VIDEO_EDITING_INSTRUCTIONS="${CUSTOM_PROMPT}"
  PROMPT_IDS="c"    # tag for run naming (lowercase for RFC 1123 compliance)
  NUM_PROMPTS=1
else
  if [[ ${RUN_VP} -eq 1 ]]; then
    echo "ERROR: No prompt specified. Set PROMPT_IDS (e.g. 1, 123) or CUSTOM_PROMPT."
    echo "  Examples:"
    echo "    PROMPT_IDS=1 bash scripts/build_and_run_2.sh"
    echo "    CUSTOM_PROMPT='Replace lane markings with red dashed lines' bash scripts/build_and_run_2.sh"
    exit 1
  fi
  NUM_PROMPTS=0
fi

# Determine LoRA mode (lora_scale > 0 ⇒ per-prompt checkpoint runs)
USE_LORA=0
if command -v bc &>/dev/null; then
    [[ $(echo "${VP_IMG_INPAINTING_LORA_SCALE:-0} > 0" | bc -l 2>/dev/null) -eq 1 ]] && USE_LORA=1
else
    # Fallback: treat anything other than "0" / "0.0" / "0.00" as lora-enabled
    [[ ! "${VP_IMG_INPAINTING_LORA_SCALE:-0}" =~ ^0(\.0+)?$ ]] && USE_LORA=1
fi

if [[ ${USE_LORA} -eq 1 ]]; then
    VP_RUN_SUFFIX="${VP_RUN_SUFFIX:-lora_${NUM_PROMPTS}prompt_${CHECKPOINT_TIMESTAMP}}"
else
    VP_RUN_SUFFIX="${VP_RUN_SUFFIX:-withoutlora_${NUM_PROMPTS}prompt_${CHECKPOINT_TIMESTAMP}}"
fi

# VP inference parameters
VP_NUM_INFERENCE_STEPS="${VP_NUM_INFERENCE_STEPS:-50}"
VP_GUIDANCE_SCALE="${VP_GUIDANCE_SCALE:-6.0}"
VP_STRENGTH="${VP_STRENGTH:-1.0}"
VP_CAPTION_REFINE_ITERS="${VP_CAPTION_REFINE_ITERS:-10}"
VP_CAPTION_REFINE_TEMPERATURE="${VP_CAPTION_REFINE_TEMPERATURE:-0.1}"
# Reduced dilate size to minimize border artifacts during camera movement
VP_DILATE_SIZE="${VP_DILATE_SIZE:-8}"
# Reduced feather size for more precise masking
VP_MASK_FEATHER="${VP_MASK_FEATHER:-4}"
# Enable border-aware masking to prevent black borders during camera movement
VP_BORDER_AWARE_MASKING="${VP_BORDER_AWARE_MASKING:-true}"
# Method for border handling: inpaint (best), blur (fast), interpolate (experimental)
VP_BORDER_METHOD="${VP_BORDER_METHOD:-inpaint}"
VP_KEEP_MASKED_PIXELS="${VP_KEEP_MASKED_PIXELS:-False}"
VP_IMG_INPAINTING_LORA_SCALE="${VP_IMG_INPAINTING_LORA_SCALE:-0.0}"
VP_SEED="${VP_SEED:-42}"

# ==============================================================================
# STAGE 3: ALPAMAYO CONFIGURATION
# ==============================================================================
ALPAMAYO_OUTPUT_BASE="${ALPAMAYO_OUTPUT_BASE:-output/alpamayo}"
ALPAMAYO_MODEL_ID="${ALPAMAYO_MODEL_ID:-/workspace/alpamayo/checkpoints/alpamayo-r1-10b}"
ALPAMAYO_NUM_TRAJ_SAMPLES="${ALPAMAYO_NUM_TRAJ_SAMPLES:-1}"
ALPAMAYO_VIDEO_NAME="${ALPAMAYO_VIDEO_NAME:-auto}"

# If true, replace all non-target camera frames with black (zero) frames so
# the model prediction depends ONLY on the generated front camera video.
# Override: ALPAMAYO_BLACK_NON_TARGET_CAMERAS=true STAGES=3 bash scripts/build_and_run_2.sh
ALPAMAYO_BLACK_NON_TARGET_CAMERAS="${ALPAMAYO_BLACK_NON_TARGET_CAMERAS:-true}"

# HuggingFace token (needed by Alpamayo for ego-motion data)
HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "${HF_TOKEN}" ]]; then
    echo "WARNING: HF_TOKEN not set. Alpamayo may need it for ego-motion data."
    echo "         Set it with: HF_TOKEN=hf_xxx bash scripts/build_and_run_2.sh"
fi

# ==============================================================================
# DOCKER IMAGE REGISTRY
# ==============================================================================
REGISTRY="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research"

SAM2_LOCAL_IMAGE="sam2/frontend"
SAM2_REMOTE_IMAGE="${REGISTRY}/harimt_sam2_${MASTER_RUN_ID}"

SAM3_LOCAL_IMAGE="sam3/frontend"
SAM3_REMOTE_IMAGE="${REGISTRY}/harimt_sam3_${MASTER_RUN_ID}"

VP_LOCAL_IMAGE="videopainter:latest"
VP_REMOTE_IMAGE="${REGISTRY}/harimt_vp${VP_RUN_SUFFIX}_${MASTER_RUN_ID}"

ALP_LOCAL_IMAGE="alpamayo:latest"
ALP_REMOTE_IMAGE="${REGISTRY}/alpamayo_vla_${MASTER_RUN_ID}"

MASTER_LOCAL_IMAGE="master-pipeline:latest"
MASTER_REMOTE_IMAGE="${REGISTRY}/master_pipeline_${MASTER_RUN_ID}"

# Tagged with the shared run id
SAM2_TAGGED="${SAM2_REMOTE_IMAGE}:${MASTER_RUN_ID}"
SAM3_TAGGED="${SAM3_REMOTE_IMAGE}:${MASTER_RUN_ID}"
VP_TAGGED="${VP_REMOTE_IMAGE}:${MASTER_RUN_ID}"
ALP_TAGGED="${ALP_REMOTE_IMAGE}:${MASTER_RUN_ID}"
MASTER_TAGGED="${MASTER_REMOTE_IMAGE}:${MASTER_RUN_ID}"

# ==============================================================================
# PRINT CONFIGURATION SUMMARY
# ==============================================================================
# Build a human-readable label for the selected stages
STAGES_LABEL=""
[[ ${RUN_SAM} -eq 1 ]] && STAGES_LABEL="${SAM_MODEL^^}"  # uppercase SAM2 or SAM3
[[ ${RUN_VP} -eq 1 ]]   && STAGES_LABEL="${STAGES_LABEL:+${STAGES_LABEL} → }VideoPainter"
[[ ${RUN_ALP} -eq 1 ]]  && STAGES_LABEL="${STAGES_LABEL:+${STAGES_LABEL} → }Alpamayo"

echo "================================================================================"
echo " MASTER PIPELINE — BUILD AND RUN (v2 — Playground Bucket)"
echo "================================================================================"
echo ""
echo "  GCS_BUCKET:          ${GCS_BUCKET}"
echo "  ALL_CKPT_BASE:       ${ALL_CKPT_BASE}"
echo "  RUN_ID:              ${RUN_ID}"
echo "  MASTER_RUN_ID:       ${MASTER_RUN_ID}"
echo "  TIMESTAMP:           ${RUN_TIMESTAMP}"
echo "  STAGES:              ${STAGES}  (${STAGES_LABEL})"
echo "  SAM_MODEL:           ${SAM_MODEL}"
if [[ ${RUN_VP} -eq 1 && ${RUN_SAM} -eq 0 ]]; then
    echo "  SAM2_DATA_RUN_ID:   ${SAM2_DATA_RUN_ID}  (reusing SAM2 output)"
fi
if [[ ${RUN_ALP} -eq 1 && ${RUN_VP} -eq 0 ]]; then
    echo "  VP_DATA_RUN_ID:     ${VP_DATA_RUN_ID}  (reusing VP output)"
fi
echo ""

if [[ ${RUN_SAM} -eq 1 ]]; then
    echo " ── Stage 1: ${SAM_MODEL^^} Segmentation ───────────────────────────────────────────"
    echo "  Model:              ${SAM_MODEL}"
    echo "  Input:              chunks ${SAM2_CHUNK_START}–${SAM2_CHUNK_END} (${SAM2_TOTAL_CHUNKS} chunks × ${SAM2_FILES_PER_CHUNK} files ≈ ${SAM2_EXPECTED_VIDEOS} videos)"
    echo "  Input base:         ${SAM2_INPUT_BASE}"
    echo "  Max frames:         ${SAM2_MAX_FRAMES}"
    if [[ "${SAM_MODEL}" == "sam3" ]]; then
        echo "  Text prompt:        ${SAM3_TEXT_PROMPT}"
        echo "  SAM3 output:        ${SAM3_OUTPUT_BASE}/${MASTER_RUN_ID}/"
        echo "  SAM3 ckpt:          ${SAM3_MODEL_GCS_CKPT_FOLDER}"
        echo "  Docker image:       ${SAM3_TAGGED}"
    else
        echo "  SAM2 output:        ${SAM2_OUTPUT_BASE}/${MASTER_RUN_ID}/"
        echo "  SAM2 ckpt:          ${SAM2_MODEL_GCS_CKPT_FOLDER}"
        echo "  Docker image:       ${SAM2_TAGGED}"
    fi
    echo "  Preprocessed (→VP): ${SAM2_PREPROCESSED_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    echo ""
fi

if [[ ${RUN_VP} -eq 1 ]]; then
    echo " ── Stage 2: VideoPainter Editing ────────────────────────────────────────────"
    if [[ ${RUN_SAM} -eq 1 ]]; then
        echo "  Input (from ${SAM_MODEL^^}):  ${SAM2_PREPROCESSED_OUTPUT_BASE}/${MASTER_RUN_ID}/  (auto from Stage 1)"
    else
        echo "  Input (SAM data):   ${SAM2_PREPROCESSED_OUTPUT_BASE}/${SAM2_DATA_RUN_ID}/"
    fi
    echo "  VP output:          ${VP_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    echo "  VP ckpt:            ${VIDEOPAINTER_MODEL_GCS_CKPT_FOLDER}"
    echo "  VLM ckpt:           ${VIDEOPAINTER_VLM_GCS_CKPT_FOLDER}"
    echo "  CogVideoX ckpt:     ${COGVIDEOX_MODEL_GCS_CKPT_FOLDER}"
    echo "  CLIP ckpt:          ${CLIP_MODEL_GCS_CKPT_FOLDER}"
    echo "  FluxFill ckpt:      ${FLUXFILL_MODEL_GCS_CKPT_FOLDER}"
    echo "  LoRA mode:          $(if [[ ${USE_LORA} -eq 1 ]]; then echo "ON (scale=${VP_IMG_INPAINTING_LORA_SCALE})"; else echo "OFF"; fi)"
    if [[ ${USE_LORA} -eq 1 ]]; then
        echo "  LoRA checkpoints:"
        for (( _i=0; _i<${#PROMPT_IDS}; _i++ )); do
            _pid="${PROMPT_IDS:$_i:1}"
            echo "    Prompt ${_pid}:       ${LORA_CHECKPOINTS[$_pid]}"
        done
    else
        echo "  FluxFill LoRA ckpt: ${TRAINED_FLUXFILL_GCS_PATH}"
    fi
    echo "  Inference steps:    ${VP_NUM_INFERENCE_STEPS}"
    echo "  Guidance scale:     ${VP_GUIDANCE_SCALE}"
    echo "  Strength:           ${VP_STRENGTH}"
    echo "  Refine iters:       ${VP_CAPTION_REFINE_ITERS}"
    echo "  Docker image:       ${VP_TAGGED}"
    echo ""
fi

if [[ ${RUN_ALP} -eq 1 ]]; then
    echo " ── Stage 3: Alpamayo VLA Inference ──────────────────────────────────────────"
    if [[ ${RUN_VP} -eq 1 ]]; then
        echo "  Input (from VP):    ${VP_OUTPUT_BASE}/${MASTER_RUN_ID}/  (auto from Stage 2)"
    else
        echo "  Input (VP data):    ${VP_OUTPUT_BASE}/${VP_DATA_RUN_ID}/"
    fi
    echo "  Output:             gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    echo "  Model:              ${ALPAMAYO_MODEL_ID}"
    echo "  Alpamayo ckpt:      ${ALPAMAYO_MODEL_GCS_CKPT_FOLDER}"
    echo "  Traj samples:       ${ALPAMAYO_NUM_TRAJ_SAMPLES}"
    echo "  Video name filter:  ${ALPAMAYO_VIDEO_NAME}"
    echo "  Black non-target:   ${ALPAMAYO_BLACK_NON_TARGET_CAMERAS}"
    echo "  Docker image:       ${ALP_TAGGED}"
    echo ""
fi

echo "================================================================================"

# ==============================================================================
# BUILD & PUSH DOCKER IMAGES
# ==============================================================================
echo ""
echo "Building and pushing Docker images …"
echo ""

# ── Stage 1: SAM Segmentation (SAM2 or SAM3) ───────────────────────────────
if [[ ${RUN_SAM} -eq 1 ]]; then
    if [[ "${SAM_MODEL}" == "sam3" ]]; then
        echo "▸ Building SAM3 image …"
        pushd segmentation/sam3 > /dev/null
        docker compose build
        docker tag "${SAM3_LOCAL_IMAGE}" "${SAM3_TAGGED}"
        docker tag "${SAM3_LOCAL_IMAGE}" "${SAM3_REMOTE_IMAGE}:latest"
        docker push "${SAM3_TAGGED}"
        docker push "${SAM3_REMOTE_IMAGE}:latest"
        popd > /dev/null
        echo "  ✓ SAM3 image pushed: ${SAM3_TAGGED}"
    else
        echo "▸ Building SAM2 image …"
        pushd segmentation/sam2 > /dev/null
        docker compose build
        docker tag "${SAM2_LOCAL_IMAGE}" "${SAM2_TAGGED}"
        docker tag "${SAM2_LOCAL_IMAGE}" "${SAM2_REMOTE_IMAGE}:latest"
        docker push "${SAM2_TAGGED}"
        docker push "${SAM2_REMOTE_IMAGE}:latest"
        popd > /dev/null
        echo "  ✓ SAM2 image pushed: ${SAM2_TAGGED}"
    fi
else
    echo "  ⏭ SAM build skipped (not in STAGES=${STAGES})"
fi

# ── Stage 2: VideoPainter ─────────────────────────────────────────────────
if [[ ${RUN_VP} -eq 1 ]]; then
    echo "▸ Building VideoPainter image …"
    pushd generation/VideoPainter > /dev/null
    docker compose build
    docker tag "${VP_LOCAL_IMAGE}" "${VP_TAGGED}"
    docker tag "${VP_LOCAL_IMAGE}" "${VP_REMOTE_IMAGE}:latest"
    docker push "${VP_TAGGED}"
    docker push "${VP_REMOTE_IMAGE}:latest"
    popd > /dev/null
    echo "  ✓ VP image pushed: ${VP_TAGGED}"
else
    echo "  ⏭ VP build skipped (not in STAGES=${STAGES})"
fi

# ── Stage 3: Alpamayo ─────────────────────────────────────────────────────
if [[ ${RUN_ALP} -eq 1 ]]; then
    echo "▸ Building Alpamayo image …"
    pushd vla/alpamayo > /dev/null
    docker compose build
    docker tag "${ALP_LOCAL_IMAGE}" "${ALP_TAGGED}"
    docker tag "${ALP_LOCAL_IMAGE}" "${ALP_REMOTE_IMAGE}:latest"
    docker push "${ALP_TAGGED}"
    docker push "${ALP_REMOTE_IMAGE}:latest"
    popd > /dev/null
    echo "  ✓ Alpamayo image pushed: ${ALP_TAGGED}"
else
    echo "  ⏭ Alpamayo build skipped (not in STAGES=${STAGES})"
fi

# ── Master orchestrator (only needed for multi-stage pipelines) ───────────
if [[ ${#STAGES_NUM} -gt 1 ]]; then
    echo "▸ Building Master orchestrator image …"
    docker compose build
    docker tag "${MASTER_LOCAL_IMAGE}" "${MASTER_TAGGED}"
    docker tag "${MASTER_LOCAL_IMAGE}" "${MASTER_REMOTE_IMAGE}:latest"
    docker push "${MASTER_TAGGED}"
    docker push "${MASTER_REMOTE_IMAGE}:latest"
    echo "  ✓ Master image pushed: ${MASTER_TAGGED}"
else
    echo "  ⏭ Master orchestrator build skipped (single-stage STAGES_NUM=${STAGES_NUM})"
fi

# ==============================================================================
# EXPORT ENV VARS FOR WORKFLOW SERIALISATION
# ==============================================================================
export SAM2_CONTAINER_IMAGE="${SAM2_TAGGED}"
export SAM3_CONTAINER_IMAGE="${SAM3_TAGGED}"
export VP_CONTAINER_IMAGE="${VP_TAGGED}"
export ALPAMAYO_CONTAINER_IMAGE="${ALP_TAGGED}"
export SAM2_OUTPUT_BASE
export SAM3_OUTPUT_BASE
export SAM2_PREPROCESSED_OUTPUT_BASE
export VP_OUTPUT_BASE
export TRAINED_FLUXFILL_GCS_PATH
export ALPAMAYO_OUTPUT_BASE
export HF_TOKEN
export STAGES
export SAM_MODEL
export SAM3_TEXT_PROMPT

# ==============================================================================
# SUBMIT MASTER WORKFLOW
# ==============================================================================
echo ""
echo "================================================================================"
echo " LAUNCHING MASTER WORKFLOW"
echo "================================================================================"
echo ""

# ── Helper: build VP_COMMON_ARGS for a given prompt + lora scale ────────────
# Usage: build_vp_args <instructions_text> <lora_scale>
build_vp_args() {
    local _instr="$1"
    local _lora="$2"
    VP_COMMON_ARGS=(
        --vp_video_editing_instructions "${_instr}"
        --vp_llm_model "${VP_LLM_MODEL}"
        --vp_num_inference_steps "${VP_NUM_INFERENCE_STEPS}"
        --vp_guidance_scale "${VP_GUIDANCE_SCALE}"
        --vp_strength "${VP_STRENGTH}"
        --vp_caption_refine_iters "${VP_CAPTION_REFINE_ITERS}"
        --vp_caption_refine_temperature "${VP_CAPTION_REFINE_TEMPERATURE}"
        --vp_dilate_size "${VP_DILATE_SIZE}"
        --vp_mask_feather "${VP_MASK_FEATHER}"
        --vp_img_inpainting_lora_scale "${_lora}"
        --vp_seed "${VP_SEED}"
    )
    if [[ "${VP_KEEP_MASKED_PIXELS}" =~ ^[Tt]rue$ ]]; then
        VP_COMMON_ARGS+=(--vp_keep_masked_pixels)
    fi
}

# ── Helper: common Alpamayo arguments ───────────────────────────────────────
ALP_COMMON_ARGS=(
    --alp_model_id "${ALPAMAYO_MODEL_ID}"
    --alp_num_traj_samples "${ALPAMAYO_NUM_TRAJ_SAMPLES}"
    --alp_video_name "${ALPAMAYO_VIDEO_NAME}"
)
# Only add the boolean flag when true (omitting it = False for HLX bool params)
if [[ "${ALPAMAYO_BLACK_NON_TARGET_CAMERAS}" =~ ^[Tt]rue$ ]]; then
    ALP_COMMON_ARGS+=(--alp_black_non_target_cameras)
fi

# ══════════════════════════════════════════════════════════════════════════════
# submit_workflow — submit a single workflow run
#
# Globals read: STAGES, SAM2_VIDEO_URIS, SAM2_MAX_FRAMES, SAM2_DATA_RUN_ID,
#               VP_DATA_RUN_ID, VP_OUTPUT_BASE, VP_COMMON_ARGS, ALP_COMMON_ARGS
# ══════════════════════════════════════════════════════════════════════════════
submit_workflow() {
    local _run_id="$1"         # unique run id for this submission
    local _prompt_tag="$2"     # short tag for execution-name (e.g. "p3")

    # Build SAM3-specific args if needed
    local SAM3_ARGS=()
    if [[ "${SAM_MODEL}" == "sam3" ]]; then
        SAM3_ARGS=(--sam3_text_prompt "${SAM3_TEXT_PROMPT}")
    fi

    case "${STAGES}" in
        1a)
            echo "Submitting SAM2-only pipeline …"
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "sam2-${_run_id//_/-}" \
              workflow_master.sam2_only_wf \
              --run_id "${_run_id}" \
              --sam_video_uris "${SAM2_VIDEO_URIS}" \
              --sam_max_frames "${SAM2_MAX_FRAMES}"
            ;;
        1b)
            echo "Submitting SAM3-only pipeline …"
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "sam3-${_run_id//_/-}" \
              workflow_master.sam3_only_wf \
              --run_id "${_run_id}" \
              --sam_video_uris "${SAM2_VIDEO_URIS}" \
              --sam_max_frames "${SAM2_MAX_FRAMES}" \
              "${SAM3_ARGS[@]}"
            ;;
        2)
            echo "  Using SAM data from run: ${SAM2_DATA_RUN_ID}"
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "vp-${_prompt_tag}-${_run_id//_/-}" \
              workflow_master.vp_only_wf \
              --run_id "${_run_id}" \
              --sam_data_run_id "${SAM2_DATA_RUN_ID}" \
              "${VP_COMMON_ARGS[@]}"
            ;;
        3)
            echo "  Using VP data run: ${VP_DATA_RUN_ID}"
            local _vp_gcs="${VP_OUTPUT_BASE}/${VP_DATA_RUN_ID}/"
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "alp-${_run_id//_/-}" \
              workflow_master.alpamayo_only_wf \
              --run_id "${_run_id}" \
              --vp_output_gcs_path "${_vp_gcs}" \
              "${ALP_COMMON_ARGS[@]}"
            ;;
        12a)
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "sam2-vp-${_prompt_tag}-${_run_id//_/-}" \
              workflow_master.sam2_vp_wf \
              --run_id "${_run_id}" \
              --sam_video_uris "${SAM2_VIDEO_URIS}" \
              --sam_max_frames "${SAM2_MAX_FRAMES}" \
              "${VP_COMMON_ARGS[@]}"
            ;;
        12b)
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "sam3-vp-${_prompt_tag}-${_run_id//_/-}" \
              workflow_master.sam3_vp_wf \
              --run_id "${_run_id}" \
              --sam_video_uris "${SAM2_VIDEO_URIS}" \
              --sam_max_frames "${SAM2_MAX_FRAMES}" \
              "${SAM3_ARGS[@]}" \
              "${VP_COMMON_ARGS[@]}"
            ;;
        23)
            echo "  Using SAM data from run: ${SAM2_DATA_RUN_ID}"
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "vp-alp-${_prompt_tag}-${_run_id//_/-}" \
              workflow_master.vp_alpamayo_wf \
              --run_id "${_run_id}" \
              --sam_data_run_id "${SAM2_DATA_RUN_ID}" \
              "${VP_COMMON_ARGS[@]}" \
              "${ALP_COMMON_ARGS[@]}"
            ;;
        123a)
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "sam2-full-${_prompt_tag}-${_run_id//_/-}" \
              workflow_master.sam2_pipeline_wf \
              --run_id "${_run_id}" \
              --sam_video_uris "${SAM2_VIDEO_URIS}" \
              --sam_max_frames "${SAM2_MAX_FRAMES}" \
              "${VP_COMMON_ARGS[@]}" \
              "${ALP_COMMON_ARGS[@]}"
            ;;
        123b)
            hlx wf run \
              --team-space research \
              --domain prod \
              --execution-name "sam3-full-${_prompt_tag}-${_run_id//_/-}" \
              workflow_master.sam3_pipeline_wf \
              --run_id "${_run_id}" \
              --sam_video_uris "${SAM2_VIDEO_URIS}" \
              --sam_max_frames "${SAM2_MAX_FRAMES}" \
              "${SAM3_ARGS[@]}" \
              "${VP_COMMON_ARGS[@]}" \
              "${ALP_COMMON_ARGS[@]}"
            ;;
        *)
            echo "ERROR: STAGES='${STAGES}' is not a supported combination."
            echo "       Supported: 1a, 1b, 2, 3, 12a, 12b, 23, 123a, 123b"
            exit 1
            ;;
    esac
}

# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH: LoRA mode (per-prompt separate runs) vs standard mode (single run)
#
# LoRA mode (lora_scale > 0):
#   SAM runs ONCE (shared), then VP (+Alp) fans out per prompt/checkpoint.
#   STAGES=123a → 1× sam2_only_wf  +  N× vp_alpamayo_wf
#   STAGES=123b → 1× sam3_only_wf  +  N× vp_alpamayo_wf
#   STAGES=12a  → 1× sam2_only_wf  +  N× vp_only_wf
#   STAGES=12b  → 1× sam3_only_wf  +  N× vp_only_wf
#   STAGES=23   → N× vp_alpamayo_wf  (SAM already done)
#   STAGES=2    → N× vp_only_wf      (SAM already done)
#   STAGES=3    → N× alpamayo_only_wf (VP already done per prompt)
#
# Standard mode (lora_scale == 0):
#   All prompts bundled into a single workflow submission.
# ══════════════════════════════════════════════════════════════════════════════
SUBMITTED_RUN_IDS=()
SAM2_LORA_RUN_ID=""           # set when SAM2 is submitted in LoRA mode

if [[ ${USE_LORA} -eq 1 && ${RUN_VP} -eq 1 ]]; then
    # ── LoRA mode: SAM once, then VP (+Alp) per prompt ──────────────────────
    echo "LoRA mode ON (scale=${VP_IMG_INPAINTING_LORA_SCALE})"
    echo ""

    # ── Step 1: Submit SAM once (if included in STAGES) ─────────────────────
    if [[ ${RUN_SAM2} -eq 1 || ${RUN_SAM3} -eq 1 ]]; then
        echo "▶ Submitting ${SAM_MODEL^^} once (shared across all ${NUM_PROMPTS} LoRA prompt runs) …"
        SAM2_LORA_RUN_ID="${MASTER_RUN_ID}"
        _saved_stages="${STAGES}"
        # Submit SAM-only: 1a or 1b
        STAGES="1${STAGES_SUFFIX}"
        submit_workflow "${SAM2_LORA_RUN_ID}" "${SAM_MODEL}"
        STAGES="${_saved_stages}"
        SUBMITTED_RUN_IDS+=("${SAM2_LORA_RUN_ID} (${SAM_MODEL^^})")
        # Point subsequent VP runs at this SAM output
        SAM2_DATA_RUN_ID="${SAM2_LORA_RUN_ID}"
        echo ""
        echo "  ⚠  VP/Alp runs below depend on ${SAM_MODEL^^} completing first."
        echo "     SAM2 data run ID: ${SAM2_DATA_RUN_ID}"
        echo ""
    fi

    # ── Step 2: Per-prompt VP (+Alp) runs ────────────────────────────────────
    # Remove "1" from STAGES_NUM → the per-prompt stages (e.g. 123→23, 12→2, 23→23)
    LORA_PER_PROMPT_STAGES="${STAGES_NUM//1/}"

    if [[ -n "${LORA_PER_PROMPT_STAGES}" ]]; then
        _saved_stages="${STAGES}"
        STAGES="${LORA_PER_PROMPT_STAGES}"

        echo "▶ Submitting ${NUM_PROMPTS} per-prompt VP/Alp run(s) (STAGES=${STAGES}) …"
        echo ""

        for (( _i=0; _i<${#PROMPT_IDS}; _i++ )); do
            _pid="${PROMPT_IDS:$_i:1}"
            _ckpt="${LORA_CHECKPOINTS[$_pid]}"
            _prompt_text="${PROMPTS[$_pid]}"
            _per_prompt_run_id="${RUN_ID}_p${_pid}_lora_${RUN_TIMESTAMP}"

            echo "────────────────────────────────────────────────────────────────────────────────"
            echo "  Prompt ${_pid}: ${_prompt_text:0:80}…"
            echo "  Checkpoint: ${_ckpt}"
            echo "  Run ID:     ${_per_prompt_run_id}"
            echo "────────────────────────────────────────────────────────────────────────────────"

            # Re-export checkpoint path so workflow_master.py picks it up
            export TRAINED_FLUXFILL_GCS_PATH="${_ckpt}"

            # Build VP args with this single prompt + its LoRA scale
            build_vp_args "${_prompt_text}" "${VP_IMG_INPAINTING_LORA_SCALE}"

            submit_workflow "${_per_prompt_run_id}" "p${_pid}"
            SUBMITTED_RUN_IDS+=("${_per_prompt_run_id}")
            echo ""
        done

        STAGES="${_saved_stages}"
    fi

else
    # ── Standard mode (no LoRA, or LoRA without VP stage) ────────────────────
    build_vp_args "${VIDEO_EDITING_INSTRUCTIONS}" "${VP_IMG_INPAINTING_LORA_SCALE}"
    submit_workflow "${MASTER_RUN_ID}" "p${PROMPT_IDS}"
    SUBMITTED_RUN_IDS+=("${MASTER_RUN_ID}")
fi

echo ""
echo "================================================================================"
echo " WORKFLOW(S) SUBMITTED SUCCESSFULLY"
echo "================================================================================"
echo ""
if [[ ${USE_LORA} -eq 1 && ${RUN_VP} -eq 1 ]]; then
    echo "  Mode:                 LoRA (scale=${VP_IMG_INPAINTING_LORA_SCALE})"
    echo "  Submissions:          ${#SUBMITTED_RUN_IDS[@]} total"
    echo ""
    for _rid in "${SUBMITTED_RUN_IDS[@]}"; do
        echo "    • ${_rid}"
    done
    echo ""
else
    echo "  Shared RUN_ID:        ${MASTER_RUN_ID}"
fi
echo "  Stages:               ${STAGES}  (${STAGES_LABEL})"
echo ""
if [[ ${RUN_SAM} -eq 1 ]]; then
    echo "  Stage 1 — ${SAM_MODEL^^} (runs once):"
    if [[ -n "${SAM2_LORA_RUN_ID}" ]]; then
        echo "    Run ID:             ${SAM2_LORA_RUN_ID}"
    fi
    echo "    Raw output:         ${SAM2_OUTPUT_BASE}/${SAM2_LORA_RUN_ID:-${MASTER_RUN_ID}}/"
    if [[ "${SAM_MODEL}" == "sam3" ]]; then
        echo "    SAM3 raw output:    ${SAM3_OUTPUT_BASE}/${SAM2_LORA_RUN_ID:-${MASTER_RUN_ID}}/"
        echo "    Text prompt:        ${SAM3_TEXT_PROMPT}"
    fi
    echo "    Preprocessed (→VP): ${SAM2_PREPROCESSED_OUTPUT_BASE}/${SAM2_LORA_RUN_ID:-${MASTER_RUN_ID}}/"
    echo ""
fi
if [[ ${RUN_VP} -eq 1 ]]; then
    echo "  Stage 2 — VideoPainter:"
    if [[ ${USE_LORA} -eq 1 ]]; then
        for _rid in "${SUBMITTED_RUN_IDS[@]}"; do
            # Skip the SAM-only entry
            [[ "${_rid}" == *"(SAM"*")"* ]] && continue
            echo "    Edited videos:      ${VP_OUTPUT_BASE}/${_rid}/"
        done
    else
        echo "    Edited videos:      ${VP_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    fi
    echo ""
fi
if [[ ${RUN_ALP} -eq 1 ]]; then
    echo "  Stage 3 — Alpamayo:"
    if [[ ${USE_LORA} -eq 1 ]]; then
        for _rid in "${SUBMITTED_RUN_IDS[@]}"; do
            [[ "${_rid}" == *"(SAM"*")"* ]] && continue
            echo "    Predictions:        gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${_rid}/"
        done
    else
        echo "    Predictions:        gs://${GCS_BUCKET}/${ALPAMAYO_OUTPUT_BASE}/${MASTER_RUN_ID}/"
    fi
    echo ""
fi
echo "================================================================================"