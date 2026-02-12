"""VideoPainter FluxFill LoRA training workflow for HLX.

Trains `generation/VideoPainter/train/train_fluxfill_inpaint_lora.py` on a FluxFill
CSV dataset (train.csv + images/ + masks/) stored in GCS.

This workflow:
  - Mounts the VideoPainter bucket prefix (ckpt + training data) via FuseBucket
  - Symlinks mounted ckpt into /workspace/VideoPainter/ckpt
  - Runs single-GPU training
  - Uploads the resulting output_dir (checkpoints + LoRA weights) to GCS

Input dataset example:
  gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data/single_white_solid_clearroad_10000

Checkpoint output base example:
  gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/trained_checkpoint
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import gcsfs

from hlx.wf import DedicatedNode, Node, fuse_prefetch_metadata, task, workflow
from hlx.wf.mounts import MOUNTPOINT, FuseBucket

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# CONTAINER IMAGE
# --------------------------------------------------------------------------------------
# Use the same VideoPainter container image by default; runner scripts can override via env.
REMOTE_IMAGE = "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp"
CONTAINER_IMAGE_DEFAULT = f"{REMOTE_IMAGE}:latest"
CONTAINER_IMAGE = os.environ.get("VP_CONTAINER_IMAGE", CONTAINER_IMAGE_DEFAULT)


# --------------------------------------------------------------------------------------
# GCS CONFIG
# --------------------------------------------------------------------------------------
VP_BUCKET = "mbadas-sandbox-research-9bb9c7f"
VP_BUCKET_PREFIX = "workspace/user/hbaskar/Video_inpainting/videopainter"

# Mount only the FluxFill checkpoint folder (not the entire videopainter prefix)
FLUXFILL_CKPT_PREFIX = f"{VP_BUCKET_PREFIX}/ckpt/flux_inp"

DEFAULT_INPUT_DATA_DIR = (
	f"gs://{VP_BUCKET}/{VP_BUCKET_PREFIX}/training/data/single_white_solid_clearroad_10000"
)
DEFAULT_OUTPUT_CHECKPOINT_DIR = (
	f"gs://{VP_BUCKET}/{VP_BUCKET_PREFIX}/training/trained_checkpoint"
)


# --------------------------------------------------------------------------------------
# PATHS (inside container)
# --------------------------------------------------------------------------------------
BASE_WORKDIR = "/workspace/VideoPainter"
DEFAULT_CKPT_DIR = os.path.join(BASE_WORKDIR, "ckpt")

FLUXFILL_FUSE_MOUNT_NAME = "fluxfill-ckpt"
FLUXFILL_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, FLUXFILL_FUSE_MOUNT_NAME)

# FluxFill base model path (inside container) after symlinking.
DEFAULT_PRETRAINED_MODEL_PATH = os.path.join(DEFAULT_CKPT_DIR, "flux_inp")


def _gcs_strip_scheme(uri: str) -> str:
	return uri[len("gs://") :] if (uri or "").startswith("gs://") else (uri or "")


def _split_gcs_uri(uri: str) -> Tuple[str, str]:
	u = (uri or "").strip()
	if u.startswith("gs://"):
		u = u[len("gs://") :]
	if "/" not in u:
		raise ValueError(f"Invalid GCS path (expected bucket/prefix): {uri}")
	bucket, key = u.split("/", 1)
	return bucket, key.rstrip("/")


def _ensure_symlink(src: str, dest: str) -> None:
	if os.path.islink(dest):
		return
	if os.path.exists(dest):
		# Replace existing directory/file with symlink.
		if os.path.isdir(dest) and not os.path.islink(dest):
			shutil.rmtree(dest)
		else:
			os.remove(dest)
	os.makedirs(os.path.dirname(dest), exist_ok=True)
	os.symlink(src, dest)


def _gcs_to_local_path(gcs_dir: str, local_tmp: str) -> str:
	"""Download GCS directory to local temporary path (training data is read-only)."""
	bucket, key = _split_gcs_uri(gcs_dir)
	if bucket != VP_BUCKET:
		raise ValueError(f"Expected bucket '{VP_BUCKET}', got '{bucket}'")

	fs = gcsfs.GCSFileSystem(token="google_default")
	local_data_dir = os.path.join(local_tmp, "training_data")
	Path(local_data_dir).mkdir(parents=True, exist_ok=True)
	
	remote_path = f"{bucket}/{key.strip('/')}"
	logger.info("Downloading training data from gs://%s to %s", remote_path, local_data_dir)
	
	# Download images/, masks/, and train.csv
	for subdir in ["images", "masks"]:
		remote_subdir = f"{remote_path}/{subdir}"
		local_subdir = os.path.join(local_data_dir, subdir)
		Path(local_subdir).mkdir(parents=True, exist_ok=True)
		try:
			files = fs.find(remote_subdir)
			for i, f in enumerate(files):
				if i % 100 == 0:
					logger.info("Downloading %s: %d files", subdir, i)
				local_file = os.path.join(local_subdir, os.path.basename(f))
				fs.get(f, local_file)
		except Exception as e:
			logger.warning("Failed to download %s: %s", subdir, e)
	
	# Download train.csv
	train_csv_remote = f"{remote_path}/train.csv"
	train_csv_local = os.path.join(local_data_dir, "train.csv")
	fs.get(train_csv_remote, train_csv_local)
	
	return local_data_dir


def _upload_directory_to_gcs(*, local_dir: str, gcs_prefix: str) -> None:
	fs = gcsfs.GCSFileSystem(token="google_default")
	base = Path(local_dir)
	out_bucket, out_key = _split_gcs_uri(gcs_prefix)
	if out_bucket != VP_BUCKET:
		raise ValueError(f"Expected output bucket '{VP_BUCKET}', got '{out_bucket}'")
	out_root = f"{out_bucket}/{out_key.strip('/')}"

	for path in base.rglob("*"):
		if path.is_dir():
			continue
		rel = path.relative_to(base).as_posix()
		remote = f"{out_root.rstrip('/')}/{rel}"
		remote_parent = os.path.dirname(remote)
		if remote_parent:
			fs.makedirs(remote_parent, exist_ok=True)
		fs.put(str(path), remote)


def _safe_run_id(text: str) -> str:
	s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (text or "").strip()).strip("_")
	return s or datetime.utcnow().strftime("fluxfill_train_%Y%m%d_%H%M%S")


@task(
	compute=DedicatedNode(
		node=Node.A100_80GB_1GPU,
		ephemeral_storage="max",
		max_duration="3d",
	),
	container_image=CONTAINER_IMAGE,
	environment={
		"PYTHONUNBUFFERED": "1",
		"PYTHONPATH": BASE_WORKDIR,
		"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
		"CUDA_HOME": "/usr/local/cuda-12.1",
		"DS_SKIP_CUDA_CHECK": "1",
		"TOKENIZERS_PARALLELISM": "false",
	},
	mounts=[
		FuseBucket(
			bucket=VP_BUCKET,
			name=FLUXFILL_FUSE_MOUNT_NAME,
			prefix=FLUXFILL_CKPT_PREFIX,
		),
	],
)
def train_fluxfill_lora_task(
	*,
	input_data_dir: str = DEFAULT_INPUT_DATA_DIR,
	output_checkpoint_dir: str = DEFAULT_OUTPUT_CHECKPOINT_DIR,
	output_run_id: Optional[str] = None,
	pretrained_model_name_or_path: str = DEFAULT_PRETRAINED_MODEL_PATH,
	# Training hyperparameters (improved defaults for stability)
	seed: Optional[int] = 42,
	mixed_precision: str = "bf16",
	gradient_checkpointing: bool = False,
	train_batch_size: int = 1,
	gradient_accumulation_steps: int = 1,
	learning_rate: float = 1e-5,
	lr_scheduler: str = "cosine",
	lr_warmup_steps: int = 50,
	max_train_steps: Optional[int] = None,
	num_train_epochs: int = 5,
	rank: int = 16,
	lora_alpha: int = 16,
	checkpointing_steps: int = 100,
	resume_from_checkpoint: str = "",
	invert_mask: bool = False,
) -> str:
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)

	run_id = _safe_run_id(output_run_id or "")
	logger.info("run_id=%s", run_id)
	logger.info("CONTAINER_IMAGE=%s", CONTAINER_IMAGE)
	logger.info("FLUXFILL_FUSE_MOUNT_ROOT=%s", FLUXFILL_FUSE_MOUNT_ROOT)

	# Prefetch checkpoint metadata best-effort.
	try:
		fuse_prefetch_metadata(FLUXFILL_FUSE_MOUNT_ROOT)
	except Exception as e:
		logger.warning("Prefetch failed (non-fatal) for %s: %s", FLUXFILL_FUSE_MOUNT_ROOT, e)

	if not os.path.exists(FLUXFILL_FUSE_MOUNT_ROOT):
		raise FileNotFoundError(f"Missing mounted FluxFill checkpoint: {FLUXFILL_FUSE_MOUNT_ROOT}")

	# Ensure /workspace/VideoPainter/ckpt/flux_inp points at the mounted flux_inp folder.
	_ensure_symlink(FLUXFILL_FUSE_MOUNT_ROOT, DEFAULT_PRETRAINED_MODEL_PATH)

	if not os.path.exists(pretrained_model_name_or_path):
		raise FileNotFoundError(f"Missing pretrained_model_name_or_path: {pretrained_model_name_or_path}")

	# Download training data to local (images, masks, train.csv)
	tmp_root = os.path.join(tempfile.gettempdir(), "fluxfill_lora_training", run_id)
	local_data_dir = _gcs_to_local_path(input_data_dir, tmp_root)
	train_csv = os.path.join(local_data_dir, "train.csv")
	if not os.path.exists(train_csv):
		raise FileNotFoundError(f"Missing train.csv at: {train_csv} (input_data_dir={input_data_dir})")

	local_out = os.path.join(tmp_root, "output")
	Path(local_out).mkdir(parents=True, exist_ok=True)

	cmd = [
		"python",
		os.path.join("train", "train_fluxfill_inpaint_lora.py"),
		"--pretrained_model_name_or_path",
		pretrained_model_name_or_path,
		"--train_csv",
		train_csv,
		"--output_dir",
		local_out,
		"--mixed_precision",
		str(mixed_precision),
		"--train_batch_size",
		str(int(train_batch_size)),
		"--gradient_accumulation_steps",
		str(int(gradient_accumulation_steps)),
		"--learning_rate",
		str(float(learning_rate)),
		"--lr_scheduler",
		str(lr_scheduler),
		"--lr_warmup_steps",
		str(int(lr_warmup_steps)),
		"--num_train_epochs",
		str(int(num_train_epochs)),
		"--rank",
		str(int(rank)),
		"--lora_alpha",
		str(int(lora_alpha)),
		"--checkpointing_steps",
		str(int(checkpointing_steps)),
	]
	
	# Add max_train_steps only if specified (otherwise trains for full epochs)
	if max_train_steps is not None:
		cmd.extend(["--max_train_steps", str(int(max_train_steps))])

	if seed is not None:
		cmd.extend(["--seed", str(int(seed))])
	if gradient_checkpointing:
		cmd.append("--gradient_checkpointing")
	if invert_mask:
		cmd.append("--invert_mask")
	if (resume_from_checkpoint or "").strip():
		cmd.extend(["--resume_from_checkpoint", str(resume_from_checkpoint).strip()])

	logger.info("Training command: %s", " ".join(cmd))
	
	# Run training and capture output
	# Set TOKENIZERS_PARALLELISM to suppress warnings
	env = os.environ.copy()
	env["TOKENIZERS_PARALLELISM"] = "false"
	result = subprocess.run(
		cmd,
		cwd=BASE_WORKDIR,
		capture_output=True,
		text=True,
		env=env,
	)
	
	# Log stdout and stderr
	if result.stdout:
		logger.info("Training stdout:\\n%s", result.stdout)
	if result.stderr:
		logger.warning("Training stderr:\\n%s", result.stderr)
	
	if result.returncode != 0:
		raise RuntimeError(
			f"Training failed with exit code {result.returncode}.\\n"
			f"Stderr: {result.stderr}\\n"
			f"Stdout: {result.stdout}"
		)

	# Upload to output_checkpoint_dir/<run_id>/...
	dest = output_checkpoint_dir.rstrip("/") + "/" + run_id
	logger.info("Uploading checkpoints: %s -> %s", local_out, dest)
	_upload_directory_to_gcs(local_dir=local_out, gcs_prefix=dest)

	logger.info("Upload complete: %s", dest)
	return dest + "/"


@workflow
def fluxfill_training_wf(
	input_data_dir: str = DEFAULT_INPUT_DATA_DIR,
	output_checkpoint_dir: str = DEFAULT_OUTPUT_CHECKPOINT_DIR,
	output_run_id: Optional[str] = None,
	pretrained_model_name_or_path: str = DEFAULT_PRETRAINED_MODEL_PATH,
	seed: Optional[int] = 42,
	mixed_precision: str = "bf16",
	gradient_checkpointing: bool = False,
	train_batch_size: int = 1,
	gradient_accumulation_steps: int = 1,
	learning_rate: float = 1e-5,
	lr_scheduler: str = "cosine",
	lr_warmup_steps: int = 50,
	max_train_steps: Optional[int] = None,
	num_train_epochs: int = 5,
	rank: int = 16,
	lora_alpha: int = 16,
	checkpointing_steps: int = 100,
	resume_from_checkpoint: str = "",
	invert_mask: bool = False,
) -> str:
	return train_fluxfill_lora_task(
		input_data_dir=input_data_dir,
		output_checkpoint_dir=output_checkpoint_dir,
		output_run_id=output_run_id,
		pretrained_model_name_or_path=pretrained_model_name_or_path,
		seed=seed,
		mixed_precision=mixed_precision,
		gradient_checkpointing=gradient_checkpointing,
		train_batch_size=train_batch_size,
		gradient_accumulation_steps=gradient_accumulation_steps,
		learning_rate=learning_rate,
		lr_scheduler=lr_scheduler,
		lr_warmup_steps=lr_warmup_steps,
		max_train_steps=max_train_steps,
		num_train_epochs=num_train_epochs,
		rank=rank,
		lora_alpha=lora_alpha,
		checkpointing_steps=checkpointing_steps,
		resume_from_checkpoint=resume_from_checkpoint,
		invert_mask=invert_mask,
	)

