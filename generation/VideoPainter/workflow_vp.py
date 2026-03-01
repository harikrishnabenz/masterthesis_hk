"""VideoPainter edit workflow for HLX.

Runs the VideoPainter edit/inpaint pipeline (infer/edit_bench.py) inside the
container.

This workflow is designed to consume preprocessed per-video folders produced by
generation/VideoPainter/data_preprocessing.py and stored in GCS under
VP_DATA_PREFIX.

All configuration constants live in config_vp.py.
All helper functions / dataclasses live in helpers_vp.py.
"""

import gc
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from hlx.wf import DedicatedNode, Node, fuse_prefetch_metadata, task, workflow
from hlx.wf.mounts import FuseBucket

from config_vp import (
	BASE_WORKDIR,
	COMPUTE_NODE,
	CONTAINER_IMAGE,
	DEFAULT_BRANCH_PATH,
	DEFAULT_CKPT_DIR,
	DEFAULT_DATA_DIR,
	DEFAULT_DATA_RUN_ID,
	DEFAULT_IMG_INPAINT_PATH,
	DEFAULT_MODEL_PATH,
	GCS_OUTPUT_BASE,
	MOUNTED_CKPT_PATH,
	SCRATCH_BASE,
	TRAINED_FLUXFILL_DEST_PATH,
	TRAINED_FLUXFILL_FUSE_MOUNT_NAME,
	TRAINED_FLUXFILL_FUSE_MOUNT_ROOT,
	TRAINED_FLUXFILL_GCS_PREFIX,
	USE_QWEN2_5_VL_7B,
	VLM_7B_DEST_PATH,
	VLM_7B_GCS_PREFIX,
	VP_BUCKET,
	VP_BUCKET_PREFIX,
	VP_DATA_FUSE_MOUNT_NAME,
	VP_DATA_FUSE_MOUNT_ROOT,
	VP_DATA_PREFIX,
	VP_FLUX_DEVICE_DEFAULT,
	VP_FUSE_MOUNT_NAME,
	VP_FUSE_MOUNT_ROOT,
	VP_VLM_7B_FUSE_MOUNT_NAME,
	VP_VLM_7B_FUSE_MOUNT_ROOT,
)

from helpers_vp import (
	VPVideoMetrics,
	_evaluate_video,
	_get_rss_mb,
	_get_torch_cuda_metrics,
	_lane_spec_to_instruction,
	_list_preprocessed_video_ids,
	_parse_instruction_list,
	_reset_torch_cuda_peaks,
	_resolve_preprocessed_video_paths,
	_run_edit_bench,
	_sanitize_folder_component,
	_select_video_file_under_root,
	_stage_preprocessed_inputs,
	_upload_outputs,
	_upload_run_source_files,
	_video_base_name_from_meta,
	_write_videopainter_run_report,
	ensure_symlink,
	ensure_writable_dir,
	upload_file_to_gcs,
)

logger = logging.getLogger(__name__)


@task(
	compute=DedicatedNode(
		node=COMPUTE_NODE,
		ephemeral_storage="max",
		max_duration="3d",
	),
	container_image=CONTAINER_IMAGE,
	environment={
		"PYTHONUNBUFFERED": "1",
		"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
		"VP_COG_DEVICE": "cuda:0",
		"VP_FLUX_DEVICE": VP_FLUX_DEVICE_DEFAULT,
		# Let infer/edit_bench.py auto-pick a different GPU for Qwen when available.
		"VP_QWEN_DEVICE": "auto",
		"VP_UNLOAD_QWEN_AFTER_USE": "1",
	},
	mounts=[
		FuseBucket(
			bucket=VP_BUCKET,
			name=VP_FUSE_MOUNT_NAME,
			prefix=VP_BUCKET_PREFIX,
		),
		FuseBucket(
			bucket=VP_BUCKET,
			name=VP_DATA_FUSE_MOUNT_NAME,
			prefix=VP_DATA_PREFIX,
		),
		FuseBucket(
			bucket=VP_BUCKET,
			name=TRAINED_FLUXFILL_FUSE_MOUNT_NAME,
			prefix=TRAINED_FLUXFILL_GCS_PREFIX,
		),
		*([
			FuseBucket(
				bucket=VP_BUCKET,
				name=VP_VLM_7B_FUSE_MOUNT_NAME,
				prefix=VLM_7B_GCS_PREFIX,
			),
		] if USE_QWEN2_5_VL_7B else []),
	],
)
def run_videopainter_edit_many(
	*,
	data_run_id: str = DEFAULT_DATA_RUN_ID,
	output_run_id: Optional[str] = None,
	data_video_ids: str = "auto",
	data_video_id: Optional[str] = None,
	inpainting_sample_id: int = 0,
	prompt: str = "",
	model_path: str = DEFAULT_MODEL_PATH,
	inpainting_branch: str = DEFAULT_BRANCH_PATH,
	img_inpainting_model: str = DEFAULT_IMG_INPAINT_PATH,
	img_inpainting_lora_path: str = "",
	img_inpainting_lora_scale: float = 1.0,
	output_name_suffix: str = "vp_edit.mp4",
	num_inference_steps: int = 50,
	guidance_scale: float = 6.0,
	num_videos_per_prompt: int = 1,
	dtype: str = "bfloat16",
	inpainting_frames: int = 49,
	down_sample_fps: int = 8,
	overlap_frames: int = 0,
	prev_clip_weight: float = 0.0,
	strength: float = 1.0,
	# Lane-line spec options (optional):
	# - Provide a single spec via lane_count/lane_color/lane_pattern, OR
	# - Provide many specs via lane_specs (same delimiter rules as video_editing_instructions).
	lane_count: str = "",
	lane_color: str = "",
	lane_pattern: str = "",
	lane_specs: str = "",
	video_editing_instruction: str = "auto",
	video_editing_instructions: str = "",
	# Default to disabling the LLM-based prompt editing/captioning so the workflow
	# does not require `transformers` / `qwen-vl-utils` inside the container.
	# To enable, pass a non-disabled string and ensure dependencies are installed.
	llm_model: str = "disabled",
	dilate_size: int = 0,
	mask_feather: int = 0,
	caption_refine_iters: int = 0,
	caption_refine_temperature: float = 0.2,
	keep_masked_pixels: bool = False,
	seed: int = 42,
) -> str:
	"""Process multiple videos ('auto' for all, or comma-separated video IDs)."""

	from hlx.wf.mounts import MOUNTPOINT
	logger.info("MOUNTPOINT=%s", MOUNTPOINT)
	logger.info("VP_FUSE_MOUNT_ROOT=%s", VP_FUSE_MOUNT_ROOT)
	logger.info("MOUNTED_CKPT_PATH=%s", MOUNTED_CKPT_PATH)
	logger.info("VP_DATA_FUSE_MOUNT_ROOT=%s", VP_DATA_FUSE_MOUNT_ROOT)
	logger.info("DEFAULT_CKPT_DIR=%s", DEFAULT_CKPT_DIR)
	logger.info("data_run_id=%s", data_run_id)
	effective_output_run_id = (output_run_id or data_run_id).strip()
	if not effective_output_run_id:
		raise ValueError("output_run_id must be non-empty when provided")
	logger.info("output_run_id=%s", effective_output_run_id)
	logger.info("USE_QWEN2_5_VL_7B=%s", USE_QWEN2_5_VL_7B)

	# Prefetch checkpoints from GCS.
	fuse_prefetch_metadata(MOUNTED_CKPT_PATH)

	# If we mount a dedicated VLM folder (7B), keep ckpt as a writable dir and
	# symlink required subfolders so we can add ckpt/vlm/Qwen2.5-VL-7B-Instruct.
	if not USE_QWEN2_5_VL_7B:
		ensure_symlink(MOUNTED_CKPT_PATH, DEFAULT_CKPT_DIR)
	else:
		ensure_writable_dir(DEFAULT_CKPT_DIR)
		for rel in ("CogVideoX-5b-I2V", "VideoPainter", "flux_inp"):
			src = os.path.join(MOUNTED_CKPT_PATH, rel)
			dest = os.path.join(DEFAULT_CKPT_DIR, rel)
			if os.path.exists(src):
				ensure_symlink(src, dest)
		ensure_writable_dir(os.path.join(DEFAULT_CKPT_DIR, "vlm"))

		try:
			fuse_prefetch_metadata(VP_VLM_7B_FUSE_MOUNT_ROOT)
		except Exception:
			logger.info("VLM 7B prefetch skipped/failed; continuing.")

		# Create symlink for VLM 7B mount
		ensure_symlink(VP_VLM_7B_FUSE_MOUNT_ROOT, VLM_7B_DEST_PATH)

	# Mount and symlink the trained FluxFill checkpoint
	try:
		fuse_prefetch_metadata(TRAINED_FLUXFILL_FUSE_MOUNT_ROOT)
	except Exception:
		logger.info("Trained FluxFill prefetch skipped/failed; continuing.")
	ensure_symlink(TRAINED_FLUXFILL_FUSE_MOUNT_ROOT, TRAINED_FLUXFILL_DEST_PATH)
	logger.info("Trained FluxFill checkpoint mounted at: %s", TRAINED_FLUXFILL_DEST_PATH)

	# Keep /workspace/VideoPainter/data present and pointing at the mounted dataset.
	ensure_symlink(VP_DATA_FUSE_MOUNT_ROOT, DEFAULT_DATA_DIR)

	try:
		fuse_prefetch_metadata(VP_DATA_FUSE_MOUNT_ROOT)
	except Exception:
		logger.info("Data prefetch skipped/failed; continuing.")

	# Resolve which video IDs to run
	video_ids: list[str]
	if data_video_id and data_video_id.strip().lower() != "auto":
		video_ids = [data_video_id.strip()]
	elif not data_video_ids or data_video_ids.strip().lower() == "auto":
		video_ids = _list_preprocessed_video_ids(data_run_id=data_run_id)
	else:
		normalized = data_video_ids.replace(",", " ")
		video_ids = [v.strip() for v in normalized.split() if v.strip()]

	if not video_ids:
		raise RuntimeError(f"No videos found at gs://{VP_BUCKET}/{VP_DATA_PREFIX}/{data_run_id}")

	# Validate model inputs exist before running the heavy command.
	paths_to_check = {
		"model_path": model_path,
		"inpainting_branch": inpainting_branch,
		"img_inpainting_model": img_inpainting_model,
	}
	missing = {k: v for k, v in paths_to_check.items() if v and not os.path.exists(v)}
	if missing:
		for k, v in missing.items():
			logger.error("Missing path (%s): %s", k, v)
		raise RuntimeError(f"Required paths missing: {', '.join(missing.keys())}")

	# Stage outputs to scratch then upload once to GCS.
	staged_output_dir = os.path.join(SCRATCH_BASE, effective_output_run_id)
	gcs_save_path = os.path.join(GCS_OUTPUT_BASE, effective_output_run_id)
	Path(staged_output_dir).mkdir(parents=True, exist_ok=True)

	logger.info("GCS output folder for this run: %s", gcs_save_path)
	logger.info("Running %d videos: %s", len(video_ids), ", ".join(video_ids[:50]))

	# Build instruction(s) from explicit lane specs if provided.
	instruction_list: list[str] = []
	if lane_specs and str(lane_specs).strip():
		instruction_list = _parse_instruction_list(lane_specs)
	elif (lane_count or lane_color or lane_pattern) and _lane_spec_to_instruction(lane_count, lane_color, lane_pattern):
		instruction_list = [_lane_spec_to_instruction(lane_count, lane_color, lane_pattern)]
	else:
		instruction_list = _parse_instruction_list(video_editing_instructions)
		if not instruction_list:
			instruction_list = [video_editing_instruction]

	logger.info("Running %d instruction(s)", len(instruction_list))

	# -----------------------------------------------------------------------
	# PRE-LOAD ALL HEAVY MODELS ONCE (Qwen, FluxFill, CogVideoX)
	# -----------------------------------------------------------------------
	cog_device = (os.environ.get("VP_COG_DEVICE") or "cuda:0").strip()
	flux_device = (os.environ.get("VP_FLUX_DEVICE") or VP_FLUX_DEVICE_DEFAULT).strip()
	qwen_device = (os.environ.get("VP_QWEN_DEVICE") or "auto").strip()
	dtype_torch = torch.bfloat16 if dtype == "bfloat16" else torch.float16

	preloaded_models: dict | None = None
	try:
		sys.path.insert(0, os.path.join(BASE_WORKDIR, "infer"))
		from edit_bench import preload_models, unload_all_models  # type: ignore[import-untyped]

		logger.info("Pre-loading all models (Qwen=%s, FluxFill=%s, CogVideoX=%s)...",
					llm_model, img_inpainting_model, model_path)
		load_start = time.perf_counter()
		preloaded_models = preload_models(
			model_path=model_path,
			inpainting_branch=inpainting_branch,
			img_inpainting_model=img_inpainting_model if img_inpainting_model else None,
			img_inpainting_lora_path=img_inpainting_lora_path if img_inpainting_lora_path else None,
			img_inpainting_lora_scale=img_inpainting_lora_scale,
			llm_model=llm_model if llm_model else None,
			dtype=dtype_torch,
			cog_device=cog_device,
			flux_device=flux_device,
			qwen_device=qwen_device if qwen_device else None,
		)
		load_s = time.perf_counter() - load_start
		logger.info("All models pre-loaded in %.1fs — will reuse across %d video(s) × %d instruction(s)",
					load_s, len(video_ids), len(instruction_list))
	except Exception as e:
		logger.warning("Model preloading failed (%s); falling back to per-video subprocess mode.", e)
		preloaded_models = None

	# Collect evaluation jobs to run *after* all generation is complete.
	deferred_evals: list[dict] = []

	# -----------------------------------------------------------------------
	# Two-phase execution (single-GPU optimisation):
	#   Phase 1 ("first_frame"): Qwen + FluxFill for ALL videos × instructions
	#   Model swap:              Unload Qwen + FluxFill → build CogVideoX
	#   Phase 2 ("video"):       CogVideoX for ALL videos × instructions
	# -----------------------------------------------------------------------
	logger.info(
		"Two-phase mode: Phase 1 (Qwen+FluxFill) for %d video(s) × %d instruction(s), "
		"then Phase 2 (CogVideoX).",
		len(video_ids), len(instruction_list),
	)

	# Pre-compute stable instruction directory names (shared by both phases).
	used_instr_dirs: set[str] = set()
	instr_dirs: list[str] = []
	for instr_idx, instruction in enumerate(instruction_list, start=1):
		instr_dir = _sanitize_folder_component(instruction, max_len=120)
		if instr_dir in used_instr_dirs:
			instr_dir = f"{instr_dir}_{instr_idx:02d}"
		used_instr_dirs.add(instr_dir)
		instr_dirs.append(instr_dir)

	# Helper: common arguments for _run_edit_bench.
	def _make_edit_bench_kwargs(
		*, output_path: str, inpainting_mask_meta: str, video_root: str,
		instruction: str, phase: str,
	) -> dict:
		return dict(
			output_path=output_path,
			inpainting_mask_meta=inpainting_mask_meta,
			image_or_video_path=video_root,
			prompt=prompt,
			model_path=model_path,
			inpainting_branch=inpainting_branch,
			img_inpainting_model=img_inpainting_model,
			img_inpainting_lora_path=img_inpainting_lora_path,
			img_inpainting_lora_scale=img_inpainting_lora_scale,
			num_inference_steps=num_inference_steps,
			guidance_scale=guidance_scale,
			num_videos_per_prompt=num_videos_per_prompt,
			dtype=dtype,
			inpainting_sample_id=inpainting_sample_id,
			inpainting_frames=inpainting_frames,
			down_sample_fps=down_sample_fps,
			overlap_frames=overlap_frames,
			prev_clip_weight=prev_clip_weight,
			strength=strength,
			video_editing_instruction=instruction,
			llm_model=llm_model,
			dilate_size=dilate_size,
			mask_feather=mask_feather,
			caption_refine_iters=caption_refine_iters,
			caption_refine_temperature=caption_refine_temperature,
			keep_masked_pixels=keep_masked_pixels,
			seed=seed,
			preloaded_models=preloaded_models,
			run_phase=phase,
		)

	per_instr_metrics: dict[str, list[VPVideoMetrics]] = {d: [] for d in instr_dirs}

	# Track Phase 1 timings so they can be combined with Phase 2 in final metrics.
	phase1_timings: dict[tuple[str, str], float] = {}  # (instr_dir, vid) -> seconds

	# -----------------------------------------------------------------------
	# PHASE 1: Qwen + FluxFill (first-frame generation for ALL videos)
	# -----------------------------------------------------------------------
	logger.info("=== Phase 1: Qwen + FluxFill (first-frame generation) ===")

	for instr_idx, (instruction, instr_dir) in enumerate(
		zip(instruction_list, instr_dirs), start=1
	):
		logger.info("Instruction %d/%d -> %s", instr_idx, len(instruction_list), instr_dir)

		for vid in video_ids:
			logger.info("[%s][%s] Starting (phase=first_frame)", instr_dir, vid)
			inpainting_mask_meta, video_root = _stage_preprocessed_inputs(
				data_run_id=data_run_id,
				data_video_id=vid,
				inpainting_sample_id=inpainting_sample_id,
			)

			output_name = f"{vid}_{output_name_suffix}" if output_name_suffix else f"{vid}_vp_edit.mp4"
			output_path = os.path.join(staged_output_dir, instr_dir, vid, output_name)
			Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

			_reset_torch_cuda_peaks()
			gen_start = time.perf_counter()
			_run_edit_bench(**_make_edit_bench_kwargs(
				output_path=output_path,
				inpainting_mask_meta=inpainting_mask_meta,
				video_root=video_root,
				instruction=instruction,
				phase="first_frame",
			))
			gen_s = time.perf_counter() - gen_start
			phase1_timings[(instr_dir, vid)] = gen_s
			logger.info("[%s][%s] Phase 1 done (%.1fs)", instr_dir, vid, gen_s)

	# -----------------------------------------------------------------------
	# MODEL SWAP: Unload Qwen + FluxFill → Load CogVideoX
	# -----------------------------------------------------------------------
	logger.info("=== Model swap: Unloading Qwen + FluxFill, building CogVideoX ===")
	from edit_bench import (  # type: ignore[import-untyped]
		_unload_qwen_model,
		_build_cog_pipeline,
	)
	_unload_qwen_model()
	if preloaded_models.get("flux_pipe") is not None:
		del preloaded_models["flux_pipe"]
	preloaded_models["flux_pipe"] = None
	gc.collect()
	torch.cuda.empty_cache()

	_cog_params = preloaded_models.get("_cog_build_params", {})
	logger.info("Building CogVideoX pipeline: %s", _cog_params.get("model_path"))
	cog_pipe = _build_cog_pipeline(**_cog_params)
	preloaded_models["cog_pipe"] = cog_pipe
	preloaded_models["_cog_deferred"] = False
	logger.info("CogVideoX loaded on GPU — starting Phase 2")

	# -----------------------------------------------------------------------
	# PHASE 2: CogVideoX video generation (ALL videos)
	# -----------------------------------------------------------------------
	logger.info("=== Phase 2: CogVideoX (video generation) ===")
	for instr_idx, (instruction, instr_dir) in enumerate(
		zip(instruction_list, instr_dirs), start=1
	):
		logger.info("Instruction %d/%d -> %s (phase=video)", instr_idx, len(instruction_list), instr_dir)

		for vid in video_ids:
			logger.info("[%s][%s] Starting (phase=video)", instr_dir, vid)
			inpainting_mask_meta, video_root = _stage_preprocessed_inputs(
				data_run_id=data_run_id,
				data_video_id=vid,
				inpainting_sample_id=inpainting_sample_id,
			)

			output_name = f"{vid}_{output_name_suffix}" if output_name_suffix else f"{vid}_vp_edit.mp4"
			output_path = os.path.join(staged_output_dir, instr_dir, vid, output_name)

			_reset_torch_cuda_peaks()
			gen_start = time.perf_counter()
			_run_edit_bench(**_make_edit_bench_kwargs(
				output_path=output_path,
				inpainting_mask_meta=inpainting_mask_meta,
				video_root=video_root,
				instruction=instruction,
				phase="video",
			))
			gen_s = time.perf_counter() - gen_start
			device, gpu_name, gpu_cc, peak_alloc_mb, peak_reserved_mb = _get_torch_cuda_metrics()
			rss_mb = _get_rss_mb()

			remote_output = os.path.join(gcs_save_path, instr_dir, vid, output_name)
			logger.info("[%s][%s] Uploading to %s", instr_dir, vid, remote_output)
			upload_start = time.perf_counter()
			uploaded = _upload_outputs(output_path=output_path, remote_output_path=remote_output)
			upload_s = time.perf_counter() - upload_start
			logger.info("[%s][%s] Uploaded: %s", instr_dir, vid, ", ".join(uploaded))

			deferred_evals.append({
				"instr_dir": instr_dir,
				"vid": vid,
				"output_path": output_path,
				"inpainting_mask_meta": inpainting_mask_meta,
				"instruction": instruction,
			})

			p1_s = phase1_timings.get((instr_dir, vid), 0.0)
			total_gen_s = p1_s + gen_s
			logger.info("[%s][%s] Phase 2 done (%.1fs); total generation: %.1fs (P1=%.1fs + P2=%.1fs)",
						instr_dir, vid, gen_s, total_gen_s, p1_s, gen_s)

			per_instr_metrics[instr_dir].append(
				VPVideoMetrics(
					video_id=vid,
					output_name=os.path.join(instr_dir, vid, output_name),
					generation_s=float(total_gen_s),
					phase1_s=float(p1_s),
					phase2_s=float(gen_s),
					upload_s=float(upload_s),
					device=device,
					gpu_name=gpu_name,
					gpu_compute_capability=gpu_cc,
					peak_gpu_mem_allocated_mb=float(peak_alloc_mb),
					peak_gpu_mem_reserved_mb=float(peak_reserved_mb),
					rss_mb=float(rss_mb),
				)
			)

	# Write per-instruction reports.
	for instr_dir in instr_dirs:
		per_video_metrics = per_instr_metrics[instr_dir]
		if not per_video_metrics:
			continue
		try:
			report_local = os.path.join(staged_output_dir, instr_dir, f"{effective_output_run_id}.txt")
			_write_videopainter_run_report(
				run_id=effective_output_run_id,
				report_path=report_local,
				input_data_run_id=data_run_id,
				per_video=per_video_metrics,
			)
			report_remote = os.path.join(gcs_save_path, instr_dir, f"{effective_output_run_id}.txt")
			upload_file_to_gcs(report_local, report_remote)
			logger.info("Uploaded VideoPainter run report: %s", report_remote)
		except Exception as e:
			logger.info("Failed to write/upload VideoPainter run report (non-fatal): %s", e)

	# ---------------------------------------------------------------------------
	# DEFERRED EVALUATION PASS
	# ---------------------------------------------------------------------------
	EVAL_TIMEOUT_S = int(os.environ.get("VP_EVAL_TIMEOUT_S", "1800"))

	if deferred_evals:
		logger.info("Freeing GPU memory before evaluation pass (%d videos)...", len(deferred_evals))
		# Unload all preloaded generation models before evaluation.
		if preloaded_models is not None:
			try:
				from edit_bench import unload_all_models  # type: ignore[import-untyped]
				unload_all_models(preloaded_models)
				preloaded_models = None
				logger.info("Preloaded models unloaded successfully.")
			except Exception as e:
				logger.warning("Failed to unload preloaded models: %s", e)
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		# Pre-check: try importing MetricsCalculator once before looping.
		_metrics_available = False
		try:
			logger.info("[EVAL] Pre-checking MetricsCalculator import...")
			sys.path.insert(0, os.path.join(BASE_WORKDIR, "evaluate"))

			class _EvalImportTimeout(Exception):
				pass

			def _import_alarm_handler(signum, frame):
				raise _EvalImportTimeout("MetricsCalculator import timed out after 120s")

			old_handler = signal.signal(signal.SIGALRM, _import_alarm_handler)
			signal.alarm(120)  # 2-minute timeout for import
			try:
				from metrics import MetricsCalculator  # noqa: F811
				_metrics_available = True
				logger.info("[EVAL] MetricsCalculator imported successfully")
			except _EvalImportTimeout:
				logger.warning("[EVAL] MetricsCalculator import timed out — skipping all evaluations")
			except ImportError as imp_err:
				logger.warning("[EVAL] MetricsCalculator import failed (%s) — skipping all evaluations", imp_err)
			finally:
				signal.alarm(0)
				signal.signal(signal.SIGALRM, old_handler)
		except Exception as e:
			logger.warning("[EVAL] Pre-check failed (%s) — skipping all evaluations", e)

		if _metrics_available:
			# Create MetricsCalculator ONCE and reuse across all videos.
			_eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			logger.info("[EVAL] Creating shared MetricsCalculator on %s (loaded once for %d videos)", _eval_device, len(deferred_evals))
			_shared_metrics_calculator = MetricsCalculator(_eval_device)
			logger.info("[EVAL] Shared MetricsCalculator ready (CLIP, LPIPS, etc. loaded)")

			for eval_info in deferred_evals:
				e_instr_dir = eval_info["instr_dir"]
				e_vid = eval_info["vid"]
				logger.info("[EVAL][%s][%s] Running deferred evaluation (timeout=%ds)", e_instr_dir, e_vid, EVAL_TIMEOUT_S)
				eval_start = time.perf_counter()
				try:
					_, raw_root = _resolve_preprocessed_video_paths(
						data_run_id=data_run_id,
						data_video_id=e_vid,
					)
					video_base_name = _video_base_name_from_meta(
						meta_csv_path=eval_info["inpainting_mask_meta"],
						inpainting_sample_id=inpainting_sample_id,
					)

					expected_filename = f"{video_base_name}.0.mp4"
					original_video_path = _select_video_file_under_root(
						raw_root=raw_root,
						video_base_name=video_base_name,
						data_video_id=e_vid,
						expected_filename=expected_filename,
					)

					mounted_video_dir = os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id, e_vid)
					mounted_masks_preferred = os.path.join(mounted_video_dir, "masks", e_vid, "all_masks.npz")
					mounted_masks_legacy = os.path.join(mounted_video_dir, "mask_root", e_vid, "all_masks.npz")
					mask_path = mounted_masks_preferred if os.path.exists(mounted_masks_preferred) else mounted_masks_legacy

					eval_output_dir = os.path.join(staged_output_dir, e_instr_dir, e_vid)

					# Run evaluation with a timeout using SIGALRM
					class _EvalTimeout(Exception):
						pass

					def _eval_alarm_handler(signum, frame):
						raise _EvalTimeout(f"Evaluation timed out after {EVAL_TIMEOUT_S}s")

					old_handler = signal.signal(signal.SIGALRM, _eval_alarm_handler)
					signal.alarm(EVAL_TIMEOUT_S)
					try:
						eval_result = _evaluate_video(
							original_path=original_video_path,
							generated_path=eval_info["output_path"],
							mask_path=mask_path,
							caption=eval_info["instruction"],
							output_dir=eval_output_dir,
							video_id=e_vid,
							metrics_calculator=_shared_metrics_calculator,
						)
					except _EvalTimeout:
						logger.warning("[EVAL][%s][%s] TIMED OUT after %ds — skipping", e_instr_dir, e_vid, EVAL_TIMEOUT_S)
						eval_result = {'error': f'Timed out after {EVAL_TIMEOUT_S}s'}
					finally:
						signal.alarm(0)
						signal.signal(signal.SIGALRM, old_handler)

					if "error" not in eval_result:
						eval_remote_path = os.path.join(gcs_save_path, e_instr_dir, e_vid, f"eval_{e_vid}.txt")
						upload_file_to_gcs(eval_result["output_path"], eval_remote_path)
						logger.info("[EVAL][%s][%s] Uploaded: %s", e_instr_dir, e_vid, eval_remote_path)
					else:
						logger.warning("[EVAL][%s][%s] Failed: %s", e_instr_dir, e_vid, eval_result["error"])

				except Exception as e:
					logger.warning("[EVAL][%s][%s] Failed: %s", e_instr_dir, e_vid, e)

				eval_s = time.perf_counter() - eval_start
				logger.info("[EVAL][%s][%s] Evaluation time: %.2fs", e_instr_dir, e_vid, eval_s)
			# Clean up the shared calculator after all evaluations are done.
			logger.info("[EVAL] All %d evaluations processed — releasing shared MetricsCalculator", len(deferred_evals))
			del _shared_metrics_calculator
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		else:
			logger.info("[EVAL] Skipped evaluation for all %d videos (MetricsCalculator unavailable)", len(deferred_evals))

	logger.info("All videos completed. Outputs saved under: %s", gcs_save_path)

	# Upload workflow + inference entrypoints used for this run (best-effort)
	try:
		_upload_run_source_files(gcs_save_path=gcs_save_path)
	except Exception as e:
		logger.info("Failed to upload source files (non-fatal): %s", e)
	return gcs_save_path


@workflow
def videopainter_many_wf(
	data_run_id: str = DEFAULT_DATA_RUN_ID,
	output_run_id: Optional[str] = None,
	data_video_ids: str = "auto",
	inpainting_sample_id: int = 0,
	prompt: str = "",
	model_path: str = DEFAULT_MODEL_PATH,
	inpainting_branch: str = DEFAULT_BRANCH_PATH,
	img_inpainting_model: str = DEFAULT_IMG_INPAINT_PATH,
	img_inpainting_lora_path: str = "",
	img_inpainting_lora_scale: float = 1.0,
	output_name_suffix: str = "vp_edit.mp4",
	num_inference_steps: int = 50,
	guidance_scale: float = 6.0,
	num_videos_per_prompt: int = 1,
	dtype: str = "bfloat16",
	inpainting_frames: int = 49,
	down_sample_fps: int = 8,
	overlap_frames: int = 0,
	prev_clip_weight: float = 0.0,
	strength: float = 1.0,
	video_editing_instruction: str = "auto",
	video_editing_instructions: str = "",
	llm_model: str = "disabled",
	dilate_size: int = 0,
	mask_feather: int = 0,
	caption_refine_iters: int = 0,
	caption_refine_temperature: float = 0.2,
	keep_masked_pixels: bool = False,
	seed: int = 42,
) -> str:
	"""Process multiple videos (comma-separated IDs or 'auto' for all)."""
	return run_videopainter_edit_many(
		data_run_id=data_run_id,
		output_run_id=output_run_id,
		data_video_ids=data_video_ids,
		inpainting_sample_id=inpainting_sample_id,
		prompt=prompt,
		model_path=model_path,
		inpainting_branch=inpainting_branch,
		img_inpainting_model=img_inpainting_model,
		img_inpainting_lora_path=img_inpainting_lora_path,
		img_inpainting_lora_scale=img_inpainting_lora_scale,
		output_name_suffix=output_name_suffix,
		num_inference_steps=num_inference_steps,
		guidance_scale=guidance_scale,
		num_videos_per_prompt=num_videos_per_prompt,
		dtype=dtype,
		inpainting_frames=inpainting_frames,
		down_sample_fps=down_sample_fps,
		overlap_frames=overlap_frames,
		prev_clip_weight=prev_clip_weight,
		strength=strength,
		video_editing_instruction=video_editing_instruction,
		video_editing_instructions=video_editing_instructions,
		llm_model=llm_model,
		dilate_size=dilate_size,
		mask_feather=mask_feather,
		caption_refine_iters=caption_refine_iters,
		caption_refine_temperature=caption_refine_temperature,
		keep_masked_pixels=keep_masked_pixels,
		seed=seed,
	)
