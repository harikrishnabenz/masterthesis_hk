# Technical Documentation: Novel Ideas & Contributions

## Master Thesis — Video Inpainting for Lane Marking Generation in Autonomous Driving

**Author:** Hariharan Baskar  
**Date:** February 19, 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Upstream Repositories vs. Local Modifications — Structural Comparison](#2-upstream-repositories-vs-local-modifications)
   - 2.1 [TencentARC/VideoPainter vs generation/VideoPainter](#21-tencentarcvideopainter-vs-generationvideopainter)
   - 2.2 [facebookresearch/sam2 vs segmentation/sam2](#22-facebookresearchsam2-vs-segmentationsam2)
   - 2.3 [NVlabs/alpamayo vs vla/alpamayo](#23-nvlabsalpamayo-vs-vlaalpamayo)
3. [Novel Contribution #1: Three-Stage Master Pipeline Orchestrator](#3-novel-contribution-1-three-stage-master-pipeline-orchestrator)
4. [Novel Contribution #2: SAM2-Based Road Segmentation Pipeline for AD Videos](#4-novel-contribution-2-sam2-based-road-segmentation-pipeline)
5. [Novel Contribution #3: FluxFill Integration into VideoPainter's Edit Pipeline](#5-novel-contribution-3-fluxfill-integration-into-videopainters-edit-pipeline)
6. [Novel Contribution #4: VLM-Guided Iterative Caption Refinement for Inpainting QA](#6-novel-contribution-4-vlm-guided-iterative-caption-refinement)
7. [Novel Contribution #5: Automated FluxFill Training Data Generation from Physical AI](#7-novel-contribution-5-automated-fluxfill-training-data-generation)
8. [Novel Contribution #6: Semantic Lane-Attribute Dataset Filtering & Curation](#8-novel-contribution-6-semantic-lane-attribute-dataset-filtering)
9. [Novel Contribution #7: FluxFill LoRA Fine-Tuning Pipeline for Lane Markings](#9-novel-contribution-7-fluxfill-lora-fine-tuning-pipeline)
10. [Novel Contribution #8: Alpamayo VLA Trajectory Evaluation of Inpainted Videos](#10-novel-contribution-8-alpamayo-vla-trajectory-evaluation)
11. [Novel Contribution #9: Trajectory Visualization & Comparison Video Rendering](#11-novel-contribution-9-trajectory-visualization--comparison-rendering)
12. [Novel Contribution #10: Cloud-Native Containerized ML Infrastructure](#12-novel-contribution-10-cloud-native-containerized-ml-infrastructure)
13. [End-to-End Data Flow](#13-end-to-end-data-flow)
14. [Technical Specifications](#14-technical-specifications)

---

## 1. Executive Summary

This thesis introduces a **novel end-to-end pipeline** that chains three state-of-the-art vision foundation models—**SAM2** (Meta), **VideoPainter** (TencentARC/SIGGRAPH 2025), and **Alpamayo-R1-10B** (NVIDIA)—into a unified system for **generating synthetic lane markings in autonomous driving videos and quantitatively evaluating their impact on trajectory prediction**.

None of the three upstream repositories were designed to interoperate. Each was released as a standalone research artifact. This thesis contributes **10 novel technical ideas** that transform them into a cohesive, production-grade pipeline:

| # | Novel Idea | Files Added/Modified | Lines of Novel Code |
|---|-----------|---------------------|-------------------|
| 1 | Three-stage master pipeline orchestrator | `workflow_master.py`, `scripts/build_and_run.sh`, root `Dockerfile` | ~1,400 |
| 2 | SAM2 road segmentation for AD videos | `process_videos_sam2.py`, `process_vide_sam2_hlxwf.py`, `workflow_sam2.py` | ~1,630 |
| 3 | FluxFill first-frame integration in VideoPainter | Modified `infer/edit_bench.py` | ~500 |
| 4 | VLM iterative caption refinement (Qwen2.5-VL) | Modified `infer/edit_bench.py` | ~300 |
| 5 | Automated FluxFill training data generation | `data_generation.py` | ~1,112 |
| 6 | Semantic lane-attribute dataset filtering | `filter_fluxfill_dataset.py` | ~725 |
| 7 | FluxFill LoRA fine-tuning pipeline | `training_workflow.py`, `train/train_fluxfill_inpaint_lora.py` | ~600 |
| 8 | Alpamayo VLA trajectory evaluation | `run_inference.py`, `workflow_alpamayo.py` | ~1,040 |
| 9 | Trajectory visualization & comparison rendering | `visualize_video.py` | ~530 |
| 10 | Cloud-native containerized ML infrastructure | 4 Dockerfiles, 4 docker-compose.yaml, build scripts | ~800 |
| | **Total novel code** | | **~8,600+** |

---

## 2. Upstream Repositories vs. Local Modifications

### 2.1 TencentARC/VideoPainter vs generation/VideoPainter

**Upstream** ([github.com/TencentARC/VideoPainter](https://github.com/TencentARC/VideoPainter)): SIGGRAPH 2025 paper — dual-stream video inpainting with CogVideoX-5B backbone and a lightweight context encoder. The repository provides:

| Directory | Upstream Purpose |
|-----------|-----------------|
| `app/` | Gradio demo application |
| `diffusers/` | Modified HuggingFace Diffusers (CogVideoX support) |
| `infer/` | Inference scripts (`edit_bench.py`, shell runners) |
| `train/` | Training scripts (CogVideoX fine-tuning) |
| `evaluate/` | LPIPS, temporal consistency, CLIP-based evaluation |
| `data_utils/` | VPData download and preprocessing |
| `env.sh`, `requirements.txt` | Environment setup |

**Local additions & modifications** (entirely novel files marked with ★):

| File/Directory | Status | Purpose |
|---------------|--------|---------|
| ★ `Dockerfile` | **New** (95 lines) | Full CUDA 12.1 container with Qwen VLM, Diffusers, workflow SDK, pre-cached eval models |
| ★ `docker-compose.yaml` | **New** | Container orchestration for local and cloud builds |
| ★ `workflow_vp.py` | **New** (1,620 lines) | Complete cloud workflow: GCS FUSE-mounted storage, multi-video discovery, VLM symlink management, output upload, prompt parsing, data staging |
| ★ `training_workflow.py` | **New** (300 lines) | FluxFill LoRA training cloud workflow: data download from GCS, training subprocess, checkpoint upload |
| ★ `workflow_sdk.whl` | **New** | Custom workflow SDK for cloud orchestration |
| ★ `data_1/` | **New** | Local test data (meta.csv, masks/, raw_videos/) |
| ★ `data_unprocessed/` | **New** | Intermediate processing data (frames/, segmentation_mask/) |
| ★ `ckpt/` | **New** | Checkpoint directories (CLIP ViT-L/14, CogVideoX-5b-I2V, SDXL inpaint) |
| `infer/edit_bench.py` | **Modified** | FluxFill pipeline integration, VLM refinement loop, multi-GPU orchestration, mask dilation/feathering |
| `train/train_fluxfill_inpaint_lora.py` | **Modified** | LoRA fine-tuning for FluxFill on lane-marking CSVs |
| `diffusers/` | **Modified** | Bug fixes (missing `outputs.py`), custom patches for CogVideoX |

**Key architectural difference:** Upstream VideoPainter runs only CogVideoX for video inpainting. The local version introduces a **dual-model pipeline** (FluxFill → CogVideoX) where FluxFill generates a high-quality semantically-correct first frame, and CogVideoX propagates the edit temporally across all subsequent frames.

---

### 2.2 facebookresearch/sam2 vs segmentation/sam2

**Upstream** ([github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)): Meta's foundation model for promptable visual segmentation in images and videos. The repository provides:

| Directory | Upstream Purpose |
|-----------|-----------------|
| `sam2/` | Core model code (SAM2Base, image/video predictors, memory bank) |
| `checkpoints/` | Download scripts for SAM2/SAM2.1 checkpoints |
| `demo/` | Frontend + backend web demo |
| `notebooks/` | Jupyter notebooks for image/video prediction |
| `training/` | Fine-tuning code |
| `tools/` | Benchmarking utilities |
| `sav_dataset/` | SA-V dataset utilities |

**Local additions (all novel):**

| File | Status | Purpose | Lines |
|------|--------|---------|-------|
| ★ `process_videos_sam2.py` | **New** | Complete road segmentation pipeline with 9-point grid initialization, morphological filtering, multi-format output, VP-compatible preprocessing, GCS upload | 1,118 |
| ★ `process_vide_sam2_hlxwf.py` | **New** | Cloud workflow wrapper: CLI argument parsing, global configuration injection | 67 |
| ★ `workflow_sam2.py` | **New** | Cloud workflow: FUSE-mounted checkpoint access, chunks:// URI resolver, GCS video download, output upload | 444 |
| ★ `data_generation.py` | **New** | FluxFill training data generation: SAM2 masks + Qwen2.5-VL captions → CSV dataset | 1,112 |
| ★ `filter_fluxfill_dataset.py` | **New** | Lane-attribute filtering: regex-based prompt parsing, multi-field filtering, GCS I/O | 725 |
| ★ `Dockerfile` | **New** | CUDA 12.1 container with SAM2, Qwen VLM, gcsfs, workflow SDK | 92 |
| ★ `docker-compose.yaml` | **New** | Container orchestration |
| ★ `workflow_sdk.whl` | **New** | Cloud workflow SDK |
| ★ `OPTIMIZATION_SUMMARY.md` | **New** | Performance optimization notes |
| ★ `README_PIPELINE.md` | **New** | Pipeline-specific documentation |
| ★ `scripts/` | **New** | Build scripts for segmentation, data generation, filtering workflows |

**Key architectural difference:** Upstream SAM2 is a general-purpose segmentation model with interactive point/box prompts. The local version adds an **automated road segmentation pipeline** that uses a fixed 9-point grid optimized for front-facing vehicle cameras, produces VideoPainter-compatible output formats, and includes complete GCS integration for cloud-scale processing.

---

### 2.3 NVlabs/alpamayo vs vla/alpamayo

**Upstream** ([github.com/NVlabs/alpamayo](https://github.com/NVlabs/alpamayo)): NVIDIA's Alpamayo-R1-10B Vision-Language-Action model for autonomous driving. The repository provides:

| Directory | Upstream Purpose |
|-----------|-----------------|
| `src/alpamayo_r1/` | Model code: VLA architecture, diffusion trajectory head, geometry, action space |
| `notebooks/` | Interactive inference notebook |
| `pyproject.toml`, `uv.lock` | Dependency management (Python 3.12 + uv) |

Upstream's `test_inference.py` loads a single clip from NVIDIA's PhysicalAI-AV dataset and runs forward inference producing trajectory predictions. It has **no** support for processing edited videos, no batch processing, no visualization, and no evaluation metrics.

**Local additions (all novel):**

| File | Status | Purpose | Lines |
|------|--------|---------|-------|
| ★ `run_inference.py` | **New** | VP→Alpamayo bridge: filename parsing, frame replacement, dual-inference (original vs. generated), minADE computation, NPZ/JSON output, overlay+comparison video rendering | ~470 |
| ★ `workflow_alpamayo.py` | **New** | Cloud workflow: FUSE-mounted checkpoints + video data, batch video discovery, model loading once, per-video inference loop, GCS upload, aggregate reporting | 573 |
| ★ `visualize_video.py` | **New** | Trajectory rendering: pinhole projection, BEV inset, progressive reveal, camera-yaw-aware ego→cam transform, side-by-side comparison videos | ~530 |
| ★ `download_checkpoints.py` | **New** | Checkpoint download utility |
| ★ `Dockerfile` | **New** | Python 3.12, CUDA 12.1, flash-attn, physical_ai_av patch, workflow SDK | ~100 |
| ★ `docker-compose.yaml` | **New** | Container orchestration |
| ★ `README_DOCKER.md` | **New** | Docker-specific documentation |
| ★ `workflow_sdk.whl` | **New** | Cloud workflow SDK |
| ★ `checkpoints/` | **New** | Model checkpoint directory |
| ★ `npz/` | **New** | Visualization data archives |
| ★ `scripts/` | **New** | Build and run scripts |

**Key architectural difference:** Upstream Alpamayo runs inference on raw PhysicalAI-AV dataset clips. The local version creates a **VideoPainter-to-Alpamayo data bridge** that:
1. Parses clip_id and camera_name from VP output filenames
2. Loads the original clip's multi-camera frames + ego-motion from PhysicalAI-AV
3. Replaces exactly one camera's frames with VideoPainter-edited frames
4. Runs inference **twice** (original and generated) for A/B comparison
5. Computes minADE against ground-truth trajectories
6. Renders overlay and side-by-side comparison videos

---

## 3. Novel Contribution #1: Three-Stage Master Pipeline Orchestrator

### Problem
SAM2, VideoPainter, and Alpamayo each run in isolated environments with incompatible Python versions (3.10 vs 3.12), different GPU memory profiles, and no shared data interfaces. Running the full pipeline manually requires sequential Docker builds, GCS path management, and careful data handoffs.

### Solution
A **master workflow orchestrator** (`workflow_master.py`, 712 lines) that defines a DAG of three `@task` nodes, each running in its own container image, connected by data-dependency edges that enforce strict sequential execution:

```
SAM2 ──(run_id)──▶ VideoPainter ──(gcs_path)──▶ Alpamayo
```

### Technical Details

**Workflow Graph Architecture:**
- Each stage is a `@task` with its own `container_image`, GPU node allocation (A100 80GB), and cloud storage FUSE mounts
- The master orchestrator container is **lightweight** (Python 3.10-slim, ~200 MB) — it only serializes the workflow graph; heavy ML deps live in stage containers
- Data flows through GCS paths: Stage 1 writes to `gs://…/preprocessed_data_vp/<run_id>/`, Stage 2 reads from there and writes to `gs://…/vp/<run_id>/`, Stage 3 reads from there
- A single `run_id` propagates through all stages for end-to-end traceability

**Partial Workflow Support:** The orchestrator defines **7 workflow variants** for flexible execution:
| Workflow | Stages | Use Case |
|----------|--------|----------|
| `master_pipeline_wf` | 1→2→3 | Full pipeline |
| `sam2_only_wf` | 1 | Segmentation only |
| `sam2_vp_wf` | 1→2 | Segmentation + editing |
| `vp_only_wf` | 2 | Editing only (reuse SAM2 data) |
| `vp_alpamayo_wf` | 2→3 | Editing + evaluation |
| `alpamayo_only_wf` | 3 | Evaluation only (reuse VP data) |

**Build Script Intelligence** (`scripts/build_and_run.sh`, ~400 lines):
- `STAGES` variable selects which stages to run (e.g., `STAGES=12` runs SAM2→VP)
- Validates input dependencies (e.g., VP without SAM2 requires `SAM2_DATA_RUN_ID`)
- Builds only the Docker images needed for the selected stages
- Generates unique `MASTER_RUN_ID = RUN_ID_TIMESTAMP` for each execution
- Supports 5 pre-defined lane marking editing prompts, selectable via `PROMPT_IDS`
- Exports all configuration as environment variables for workflow serialization

**Why This Is Novel:** No prior work has chained a segmentation model, a video inpainting model, and a VLA trajectory model into a single orchestrated pipeline. The master orchestrator solves the container isolation problem while maintaining data lineage.

---

## 4. Novel Contribution #2: SAM2-Based Road Segmentation Pipeline

### Problem
SAM2 provides general-purpose segmentation with interactive prompts. Autonomous driving requires **automated road region detection** across thousands of videos without manual annotation.

### Solution
A complete road segmentation pipeline (`process_videos_sam2.py`, 1,118 lines) that uses SAM2's video predictor with a **fixed 9-point grid** optimized for front-facing vehicle camera geometry.

### Technical Details

**Automatic Road Initialization (9-Point Grid):**
```
Points: (0.5, 0.85), (0.4, 0.75), (0.6, 0.75),
        (0.3, 0.70), (0.7, 0.70), (0.5, 0.65),
        (0.35, 0.80), (0.65, 0.80), (0.5, 0.75)
```
These normalized coordinates target the lower 65–85% of the frame where the road surface typically appears in dashcam footage. Each point is assigned label=1 (foreground) to seed the SAM2 prompt.

**Processing Pipeline per Video:**
1. **Frame Extraction:** FFmpeg-based extraction at 8 FPS (configurable), up to 150 frames per video
2. **FPS Detection:** Multi-method detection using `r_frame_rate`, `avg_frame_rate`, container metadata, and frame-count/duration fallback
3. **SAM2 Inference:** Video predictor in VOS mode for temporal consistency across frames
4. **Morphological Filtering:** Erosion + dilation to clean mask boundaries, fill small holes
5. **Multi-Format Output:**
   - Binary PNG masks per frame (0=keep, 255=inpaint)
   - Compressed NPZ archive (`all_masks.npz`) for efficient downstream loading
   - Visualization overlay videos (segmented regions highlighted)
   - Metadata CSV with video statistics
6. **VideoPainter-Compatible Preprocessing:** Creates the exact folder structure VP's `edit_bench.py` expects:
   ```
   preprocessed_data_vp/<run_id>/<video_id>/
     ├── meta.csv
     ├── masks/<video_id>/all_masks.npz
     └── raw_videos/<video_id>/<video_id>.0.mp4
   ```
7. **GCS Upload:** Parallel upload of results and preprocessed data to separate GCS prefixes

**Chunks:// URI Protocol:** A custom URI scheme for batch video selection:
```
chunks://bucket/prefix/camera_folder?start=0&end=10&per_chunk=5
```
Resolves to individual `gs://` paths by listing chunk directories (`chunk_0000/`, `chunk_0001/`, …) and selecting up to `per_chunk` MP4 files from each.

**Why This Is Novel:** Prior road segmentation pipelines use specialized models (lane detectors, semantic segmentation). This is the first to repurpose SAM2 as an automated road segmentor specifically for generating video inpainting masks, producing output directly compatible with VideoPainter's expected input format.

---

## 5. Novel Contribution #3: FluxFill Integration into VideoPainter's Edit Pipeline

### Problem
VideoPainter's original `edit_bench.py` generates all frames using only CogVideoX-5B. The first frame often lacks semantic precision for fine-grained lane marking attributes (single vs. double, white vs. yellow, solid vs. dashed) because CogVideoX optimizes for temporal consistency rather than per-frame detail.

### Solution
A **dual-model pipeline** where FluxFill generates a semantically-precise first frame, and CogVideoX propagates the edit temporally:

```
Input Video + Mask
       │
       ▼
   FluxFill → High-quality first frame (image inpainting)
       │
       ▼
   CogVideoX → Temporally consistent video (video diffusion)
       │
       ▼
   Edited Video
```

### Technical Details

**Modified File:** `infer/edit_bench.py`

**FluxFillPipeline Integration:**
- Loads `FluxFillPipeline` from a local checkpoint or GCS-mounted path
- Supports optional LoRA weights (`--img_inpainting_lora_path`, `--img_inpainting_lora_scale`)
- First frame is generated at the video's native resolution
- Mask is applied with configurable dilation and Gaussian feathering

**Mask Processing Pipeline:**
- `_dilate_and_feather_mask_images()`: Expands inpaint regions using morphological dilation (configurable `dilate_size`, default 24px), then applies Gaussian blur for smooth boundaries (configurable `mask_feather`, default 8px)
- `--keep_masked_pixels` flag: When enabled, preserves the original background pixels outside the mask in the final composite, preventing FluxFill from altering non-road regions

**Multi-GPU Orchestration:**
- Qwen VLM on `cuda:1` (when available, controlled by `VP_QWEN_DEVICE`)
- FluxFill + CogVideoX on `cuda:0` (controlled by `VP_FLUX_DEVICE`, `VP_COG_DEVICE`)
- Explicit model unloading between stages (`_unload_qwen_model`) to reclaim VRAM
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for dynamic memory management

**Why This Is Novel:** No prior work has combined a still-image inpainting model (FluxFill) with a video diffusion model (CogVideoX) in a sequential first-frame → temporal-propagation pipeline for video editing.

---

## 6. Novel Contribution #4: VLM-Guided Iterative Caption Refinement

### Problem
Text-guided inpainting is highly sensitive to prompt quality. Generic prompts like "white lane marking" produce inconsistent results. The model needs specific visual texture details, but manually crafting prompts per video is infeasible at scale.

### Solution
An **automated QA loop** using Qwen2.5-VL (7B) that evaluates FluxFill's output and iteratively refines the caption until the generated image matches the editing instruction.

### Technical Details

**Function:** `_qwen_refine_first_frame_caption()` in `infer/edit_bench.py`

**Refinement Loop (up to N iterations, default 10):**
1. FluxFill generates a first frame using the current caption
2. Qwen2.5-VL receives: `[original_frame, generated_frame, editing_instruction]`
3. VLM evaluates semantic correctness:
   - Lane marking count (single vs. double)
   - Color accuracy (white vs. yellow)
   - Pattern type (solid vs. dashed)
   - Perspective alignment with original lanes
   - Road texture preservation
4. VLM outputs: `PASS` or `FAIL` with structured reasoning
5. On `FAIL`: VLM generates a revised caption with more specific visual details
6. Loop continues until `PASS` or max iterations reached

**Configurable Parameters:**
- `--caption_refine_iters` (default: 10): Maximum refinement iterations
- `--caption_refine_temperature` (default: 0.1): VLM sampling temperature (low = deterministic)
- `VP_UNLOAD_QWEN_AFTER_USE=1`: Unload VLM after refinement to free VRAM for CogVideoX

**Why This Is Novel:** This is the first use of a VLM as an automated quality assessor in a video inpainting loop. The feedback mechanism creates a closed-loop system where the editing model's output quality is verified against the semantic intent before temporal propagation.

---

## 7. Novel Contribution #5: Automated FluxFill Training Data Generation

### Problem
Training a specialized FluxFill LoRA for lane markings requires thousands of (image, mask, caption) triplets. Manually annotating road images with masks and lane-attribute captions is prohibitively expensive.

### Solution
A fully automated pipeline (`data_generation.py`, 1,112 lines) that generates training datasets from raw autonomous driving videos using SAM2 + Qwen2.5-VL.

### Technical Details

**Pipeline Stages:**
1. **GCS Video Discovery:** Lists MP4 files in chunk-structured GCS folders, selectable by slice
2. **Frame Extraction:** FFmpeg-based extraction of specific frame numbers per video
3. **SAM2 Mask Generation:** Image predictor mode with road-optimized point grid → binary masks
4. **VLM Caption Generation (Qwen2.5-VL-7B):**
   - Receives the frame image as input
   - Extracts structured lane attributes:
     - **Count:** single / double / unknown
     - **Color:** white / yellow / mixed / unknown
     - **Pattern:** solid / dashed / mixed / unknown
   - Generates normalized prompt: `"road with {count} {color} {pattern} lane markings"`
5. **Dataset Assembly:**
   ```
   dataset_folder/
     ├── train.csv       (image, mask, prompt, prompt_2)
     ├── images/*.png    (original frames)
     └── masks/*.png     (binary road masks)
   ```
6. **GCS Upload:** Complete dataset uploaded to GCS for training

**Scale:** Successfully generated 10,000+ training samples from NVIDIA PhysicalAI-AV data.

**Why This Is Novel:** This is the first automated pipeline that combines a segmentation model (SAM2) with a vision-language model (Qwen2.5-VL) to generate structured training data for inpainting models, with semantically-parsed lane marking attributes.

---

## 8. Novel Contribution #6: Semantic Lane-Attribute Dataset Filtering

### Problem
Raw generated datasets contain mixed lane attributes. Training a LoRA for "single white solid" lane markings requires filtering out all other combinations.

### Solution
A multi-attribute filtering system (`filter_fluxfill_dataset.py`, 725 lines) with regex-based prompt parsing and configurable quality controls.

### Technical Details

**Attribute Parsing (Regex):**
```python
r"road with (?P<count>...) (?P<color>...) (?P<pattern>...) lane markings"
```
Extracts `count`, `color`, `pattern` from normalized captions produced by `data_generation.py`.

**Filtering Modes:**
| Parameter | Values | Effect |
|-----------|--------|--------|
| `count` | single / double / any | Filter by lane count |
| `color` | white / yellow / mixed / any | Filter by lane color |
| `pattern` | solid / dashed / mixed / any | Filter by lane pattern |
| `REQUIRE_CLEAR_ROAD` | 0 / 1 | Reject prompts with "unknown" attributes |

**Copy Modes:** `copy` / `hardlink` / `symlink` (GCS always uses copy)

**GCS Support:** Both input and output can be GCS paths. Local staging is used for filtering, then batch upload.

**Why This Is Novel:** First semantic curation tool for lane marking datasets that filters by structured visual attributes extracted from VLM captions.

---

## 9. Novel Contribution #7: FluxFill LoRA Fine-Tuning Pipeline

### Problem
The base FluxFill model generates generic inpainting. Lane markings require specific visual patterns (dashed spacing, line thickness, color fidelity) that the base model doesn't produce reliably.

### Solution
A complete LoRA fine-tuning pipeline (`training_workflow.py` + `train/train_fluxfill_inpaint_lora.py`) that trains FluxFill on curated lane-marking datasets.

### Technical Details

**Training Configuration:**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `rank` | 16 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling factor |
| `learning_rate` | 1e-5 | AdamW optimizer LR |
| `lr_scheduler` | cosine | Learning rate schedule |
| `lr_warmup_steps` | 50 | Warmup steps |
| `num_train_epochs` | 5 | Training epochs |
| `train_batch_size` | 1 | Per-GPU batch size |
| `gradient_accumulation_steps` | 1 | Effective batch multiplier |
| `mixed_precision` | bf16 | Mixed precision training |
| `checkpointing_steps` | 100 | Save frequency |

**Cloud Workflow Integration:**
- Cloud storage FUSE mounts for FluxFill base checkpoint (`ckpt/flux_inp`)
- Training data downloaded from GCS to local scratch
- Output checkpoints uploaded to GCS with timestamp-based run IDs
- A100 80GB GPU allocation

**Why This Is Novel:** First LoRA fine-tuning pipeline specifically designed for FluxFill inpainting on structured lane marking datasets, integrated into a cloud workflow system.

---

## 10. Novel Contribution #8: Alpamayo VLA Trajectory Evaluation

### Problem
Video inpainting quality is traditionally measured by pixel-level metrics (LPIPS, SSIM, FID). These metrics don't capture whether the edited lane markings would actually affect autonomous driving behavior.

### Solution
Use NVIDIA's Alpamayo-R1-10B VLA model to **predict driving trajectories on both original and inpainted videos**, then measure the trajectory divergence via minADE.

### Technical Details

**VideoPainter → Alpamayo Data Bridge** (`run_inference.py`):

1. **Filename Parsing:** Extracts `clip_id` and `camera_name` from VP output filenames:
   ```
   <clip_id>.<camera_name>_vp_edit_sample0_generated.mp4
   ```

2. **Original Data Loading:** Uses `load_physical_aiavdataset()` to retrieve:
   - Multi-camera frames (7 views: front_wide, front_tele, cross_left/right, rear_left/right, rear_tele)
   - Ego-motion history (16 timesteps at CONTENT_TIME_STEP)
   - Ground-truth future trajectory (64 waypoints at 10 Hz, 6.4s horizon)

3. **Temporal Alignment:**
   - VP video encoded at 8 fps → `CONTENT_TIME_STEP = 0.125s`
   - `t0_us = (total_frames − 1) × 0.125s × 1e6` for dataset alignment
   - Last 4 consecutive frames per camera used for inference

4. **Dual Inference:**
   - **Original inference:** Run Alpamayo on unmodified dataset frames → `original_min_ade`
   - **Frame replacement:** Substitute one camera's last-4-frames with VP output (with bilinear resize if needed)
   - **Generated inference:** Run Alpamayo on modified frames → `min_ade_meters`
   - Both share identical ego-motion history and all other camera views

5. **minADE Computation:**
   $$\text{minADE} = \min_{s \in S} \frac{1}{T} \sum_{t=1}^{T} \| \hat{p}_{xy}^{(s)}(t) - p_{xy}^{gt}(t) \|_2$$
   where $S$ = trajectory samples, $T$ = 64 waypoints, XY-plane only.

6. **Chain-of-Thought Extraction:** Both original and generated inference produce reasoning traces from the VLA model's language head.

**Output Artifacts per Video:**
| File | Format | Content |
|------|--------|---------|
| `*_inference.json` | JSON | Predictions, minADE (both original & generated), reasoning traces, temporal config, metrics |
| `*_vis_data.npz` | NPZ | Trajectory tensors, image frames, camera indices, full original camera frames |
| `*_overlay.mp4` | H.264 | Trajectory overlay on generated video |
| `*_comparison.mp4` | H.264 | Side-by-side original vs. generated with trajectories |

**Why This Is Novel:** This is the **first use of a VLA model as an evaluation metric for video inpainting**. By comparing trajectory predictions on original vs. edited videos, we obtain a functionally meaningful measure of inpainting quality that goes beyond pixel-level metrics.

---

## 11. Novel Contribution #9: Trajectory Visualization & Comparison Rendering

### Problem
Raw trajectory predictions (numpy arrays of 3D waypoints) are not interpretable without visualization. Comparing original vs. edited video trajectories requires careful frame-by-frame alignment.

### Solution
A comprehensive visualization system (`visualize_video.py`, ~530 lines) that renders:
1. **Trajectory overlay videos** with projected 3D waypoints on camera frames
2. **Side-by-side comparison videos** showing original and generated frames with their respective trajectories

### Technical Details

**3D → 2D Projection (Pinhole Camera Model):**
```
Ego frame:   x=forward, y=left, z=up
Camera frame: x=right, y=down, z=forward (depth)

Transform: x_cam = −y_ego, y_cam = −z_ego, z_cam = x_ego
```

**Camera-Yaw-Aware Transform:** For non-front cameras, the ego-to-camera transform includes a yaw rotation:
```python
CAMERA_YAW_RAD = {
    "camera_front_wide_120fov": 0.0,
    "camera_cross_left_120fov": π/2,
    "camera_cross_right_120fov": −π/2,
    "camera_rear_tele_30fov": π,
    ...
}
```

**Focal Length:** $f_x = \frac{W}{2 \cdot \tan(\text{FOV}/2)}$, with FOV auto-detected from camera name (120°, 70°, 30°).

**Visualization Features:**
- Progressive reveal mode: linearly maps frame index to waypoint count
- BEV inset (250×250 px) at bottom-left: ego history (gray), GT (red), predictions (cyan)
- 5 cycling colors for multiple trajectory samples
- Semi-transparent legend box
- HUD overlay with minADE, clip_id, camera info
- Points behind camera (z_cam ≤ 0.5m) filtered out

**Comparison Video Layout:**
```
┌─────────────────────────────────────┐
│ Original (left) │ Generated (right) │
│  + GT trajectory│  + pred trajectory│
│  + orig pred    │  + VP frames      │
│  BEV inset      │  BEV inset        │
│  Original minADE│  Generated minADE │
└─────────────────────────────────────┘
```

**Why This Is Novel:** First trajectory visualization system that renders projected 3D driving trajectories on both original and inpainted video frames for visual comparison of VLA model behavior changes.

---

## 12. Novel Contribution #10: Cloud-Native Containerized ML Infrastructure

### Problem
Three models with incompatible dependencies (Python 3.10 vs 3.12, different PyTorch versions, flash-attn only for Alpamayo) cannot share a single environment. Cloud-scale processing requires reproducible, isolated environments.

### Solution
Four purpose-built Docker containers orchestrated through cloud workflows with GCS FUSE-mounted data volumes.

### Container Architecture

| Container | Base Image | Python | Key Dependencies | Size |
|-----------|-----------|--------|-----------------|------|
| SAM2 | `nvidia/cuda:12.1.1-cudnn8-runtime` | 3.10 | SAM2, Qwen2.5-VL, gcsfs, Workflow SDK | ~8 GB |
| VideoPainter | `nvidia/cuda:12.1.1-cudnn8-devel` | 3.10 | CogVideoX, FluxFill, Diffusers, Qwen, Workflow SDK | ~15 GB |
| Alpamayo | `nvidia/cuda:12.1.1-cudnn8-devel` | 3.12 | Alpamayo-R1, flash-attn, physical_ai_av, Workflow SDK | ~12 GB |
| Master | `python:3.10-slim` | 3.10 | Workflow SDK, gcsfs (lightweight orchestrator) | ~500 MB |

**Alpamayo Dockerfile Patch:**
```dockerfile
# physical_ai_av returns read-only Arrow-backed arrays; scipy needs writable buffers
RUN sed -i 's/\.to_numpy())/\.to_numpy().copy())/' \
    /usr/local/lib/python3.12/dist-packages/physical_ai_av/egomotion.py
```

**GCS Data Architecture:**
```
gs://<project-bucket>/<workspace>/
  ├── input/data/camera_front_tele_30fov/chunk_NNNN/  (source videos)
  ├── outputs/
  │   ├── sam2/<run_id>/           (segmentation masks + visualizations)
  │   ├── preprocessed_data_vp/<run_id>/  (VP-compatible data)
  │   ├── vp/<run_id>/            (edited videos)
  │   └── alpamayo/<run_id>/      (trajectory predictions + reports)
  └── checkpoints/
      ├── sam2/                    (SAM2 model weights)
      ├── videopainter/            (CogVideoX, FluxFill, CLIP, VLM weights)
      └── alpamayo/                (Alpamayo-R1-10B weights)
```

**Cloud Storage FUSE Mounts (zero-copy model access):**
- Checkpoints mounted read-only via FUSE → no download latency
- Symlinks redirect container-expected paths to mount points
- Metadata prefetching warms the FUSE cache before inference

---

## 13. End-to-End Data Flow

```
Raw AD Videos (PhysicalAI-AV, GCS)
        │
        ▼
┌─── STAGE 1: SAM2 Segmentation ────────────────────────────┐
│ • Download videos from GCS (gcsfs)                         │
│ • Extract frames at 8 FPS (ffmpeg)                         │
│ • Run SAM2 with 9-point road grid (sam2.1_hiera_large)     │
│ • Generate binary masks + NPZ archives                     │
│ • Create VP-compatible folder structure                     │
│ Output: gs://…/preprocessed_data_vp/<run_id>/              │
└────────────────────────────────────────────────────────────┘
        │ (data dependency: run_id)
        ▼
┌─── STAGE 2: VideoPainter Editing ──────────────────────────┐
│ • Mount SAM2 preprocessed data via cloud storage FUSE mount  │
│ • Stage data to local scratch (masks + raw_videos)          │
│ • For each editing instruction:                             │
│   ├─ VLM refines caption (Qwen2.5-VL, up to 10 iters)     │
│   ├─ FluxFill generates first frame (+ optional LoRA)      │
│   ├─ CogVideoX-5B propagates edit to all frames            │
│   └─ Output: edited video per instruction per input video  │
│ Output: gs://…/vp/<run_id>/                                │
└────────────────────────────────────────────────────────────┘
        │ (data dependency: GCS path)
        ▼
┌─── STAGE 3: Alpamayo VLA Inference ────────────────────────┐
│ • Discover VP output videos via cloud storage FUSE mount    │
│ • Load Alpamayo-R1-10B model once (bfloat16, flash-attn)   │
│ • For each edited video:                                    │
│   ├─ Parse clip_id + camera from filename                   │
│   ├─ Load original multi-camera data from PhysicalAI-AV    │
│   ├─ Run inference on ORIGINAL frames → orig_minADE         │
│   ├─ Replace target camera with VP frames                   │
│   ├─ Run inference on GENERATED frames → gen_minADE         │
│   ├─ Save JSON results + NPZ tensors                        │
│   └─ Render overlay + comparison videos                     │
│ Output: gs://…/alpamayo/<run_id>/                          │
└────────────────────────────────────────────────────────────┘
```

---

## 14. Technical Specifications

### Compute Requirements

| Stage | GPU | VRAM | Duration (per video) | Container |
|-------|-----|------|---------------------|-----------|
| SAM2 | A100 80GB | ~8 GB | ~2 min | `sam2_container` |
| VideoPainter | A100 80GB | ~40 GB (peak) | ~10 min | `vp_container` |
| Alpamayo | A100 80GB | ~24 GB | ~1 min | `alpamayo_vla` |

### Model Specifications

| Model | Parameters | Precision | Source |
|-------|-----------|-----------|--------|
| SAM2.1 Hiera-Large | 305M | float32 | Meta (Apache 2.0) |
| CogVideoX-5B-I2V | 5B | bfloat16 | THUDM |
| FluxFill | ~12B | bfloat16 | Black Forest Labs |
| Qwen2.5-VL-7B | 7B | bfloat16 | Alibaba (Apache 2.0) |
| Alpamayo-R1-10B | 10B | bfloat16 | NVIDIA (non-commercial weights) |

### Key Hyperparameters (Inference Defaults)

| Parameter | Value | Stage |
|-----------|-------|-------|
| `num_inference_steps` | 50–70 | VideoPainter |
| `guidance_scale` | 6.0 | VideoPainter |
| `strength` | 1.0 | VideoPainter |
| `dilate_size` | 24 px | VideoPainter |
| `mask_feather` | 8 px | VideoPainter |
| `caption_refine_iters` | 10 | VideoPainter (VLM QA) |
| `num_traj_samples` | 1 | Alpamayo |
| `top_p` | 0.98 | Alpamayo |
| `temperature` | 0.6 | Alpamayo |
| `max_generation_length` | 256 tokens | Alpamayo |
| `trajectory_horizon` | 6.4s (64 waypoints @ 10 Hz) | Alpamayo |

### Output Data Formats

| Artifact | Format | Location |
|----------|--------|----------|
| Segmentation masks | Binary PNG + NPZ | `gs://…/sam2/<run_id>/` |
| VP preprocessed data | MP4 + NPZ + CSV | `gs://…/preprocessed_data_vp/<run_id>/` |
| Edited videos | H.264 MP4 | `gs://…/vp/<run_id>/` |
| Trajectory predictions | JSON | `gs://…/alpamayo/<run_id>/` |
| Visualization tensors | NPZ | `gs://…/alpamayo/<run_id>/` |
| Overlay/comparison videos | H.264 MP4 | `gs://…/alpamayo/<run_id>/` |
| Aggregate reports | TXT | `gs://…/alpamayo/<run_id>/` |
