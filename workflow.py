"""Master workflow that chains SAM2 → VideoPainter → Alpamayo sequentially.

This workflow orchestrates the complete pipeline, ensuring each stage completes
before the next begins, with data flowing from one stage to the next via GCS paths.

Uses @dynamic so that workflow parameters are materialised as real Python values,
allowing conditionals, string operations, and dict-building that are forbidden
inside a plain @workflow (where parameters are Flytekit Promise objects).
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional

from hlx.wf import ContainerImage, SharedNode, dynamic, task, workflow


# ---------------------------------------------------------------------------
# Barrier task – accepts the previous stage's output to enforce ordering.
# Flytekit schedules tasks in parallel unless there is a data dependency.
# This tiny task creates that dependency chain: SAM2 → VP → Alpamayo.
# ---------------------------------------------------------------------------
@task(compute=SharedNode(), container_image=ContainerImage.PYTHON_3_10.value)
def _barrier(prev_result: str, pass_through: str) -> str:
    """Forces sequential execution: returns pass_through after prev_result completes."""
    return pass_through


@task(compute=SharedNode(), container_image=ContainerImage.PYTHON_3_10.value)
def _build_gcs_path(prev_result: str, base: str, suffix: str) -> str:
    """Build a GCS path after the previous stage completes (enforces ordering)."""
    return f"{base}/{suffix}"

# ---------------------------------------------------------------------------
# Make sub-workflow modules importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _sub in ("segmentation/sam2", "generation/VideoPainter", "vla/alpamayo"):
    _p = str(_HERE / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GCS path constants
# ---------------------------------------------------------------------------
_OUTPUT_BASE = (
    "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar"
    "/Video_inpainting/videopainter/training/output"
)


# ---------------------------------------------------------------------------
# @dynamic – can use plain Python on its arguments
# ---------------------------------------------------------------------------
@dynamic(
    compute=SharedNode(),
    container_image=ContainerImage.PYTHON_3_10.value,
)
def _master_pipeline_dynamic(
    # SAM2 parameters
    sam2_run_id: str = "experiment_01",
    sam2_max_frames: int = 150,
    sam2_video_uris: str = "default",
    # VideoPainter parameters
    vp_data_run_id: str = "",
    vp_instruction: str = "",
    vp_num_samples: int = 1,
    # Alpamayo parameters
    alpamayo_run_id: str = "",
    alpamayo_num_traj_samples: int = 1,
    alpamayo_model_id: str = "nvidia/Alpamayo-R1-10B",
) -> str:
    """Chain SAM2 → VideoPainter → Alpamayo sequentially.

    All three stages run as separate HLX tasks in their own GPU containers.
    Data flows between stages via GCS paths.
    """
    # -- resolve defaults that depend on other args ---------------------------
    effective_vp_data_run_id = vp_data_run_id if vp_data_run_id else sam2_run_id
    effective_alpamayo_run_id = alpamayo_run_id if alpamayo_run_id else sam2_run_id

    # -- parse comma-separated video URIs ------------------------------------
    video_uris: Optional[List[str]] = None
    if sam2_video_uris and sam2_video_uris != "default":
        video_uris = [u.strip() for u in sam2_video_uris.split(",") if u.strip()]

    logger.info("=" * 80)
    logger.info("MASTER PIPELINE – SEQUENTIAL EXECUTION")
    logger.info("=" * 80)
    logger.info("Stage 1: SAM2 Segmentation  (run_id=%s)", sam2_run_id)
    logger.info("Stage 2: VideoPainter Editing (data_run_id=%s)", effective_vp_data_run_id)
    logger.info("Stage 3: Alpamayo VLA Inference (run_id=%s)", effective_alpamayo_run_id)
    logger.info("=" * 80)

    # ── Helper: stub heavy packages so sub-workflow modules can be imported ──
    # The @dynamic container is lightweight (python:3.10-slim).  The sub-workflow
    # modules import cv2/numpy/torch/psutil at module level, but those are only
    # *used* inside @task bodies which execute in their own GPU containers.
    # Injecting thin stubs lets us import the @task definitions for DAG compilation.
    import types
    import importlib.util as _ilu

    _STUB_PACKAGES = ("cv2", "numpy", "psutil")
    _stubs_added: list[str] = []
    for _pkg in _STUB_PACKAGES:
        if _pkg not in sys.modules:
            _stub = types.ModuleType(_pkg)
            _stub.ndarray = type("ndarray", (), {})       # numpy.ndarray
            sys.modules[_pkg] = _stub
            _stubs_added.append(_pkg)

    # torch needs a deeper stub tree because flytekit.extras.pytorch accesses
    # torch.nn.Module, torch.optim.Optimizer, torch.utils.data, etc. at import
    # time when it discovers "torch" in sys.modules.
    if "torch" not in sys.modules:
        _torch = types.ModuleType("torch")
        _torch.Tensor = type("Tensor", (), {})
        _torch.device = type("device", (), {})

        _nn = types.ModuleType("torch.nn")
        _nn.Module = type("Module", (), {})
        _torch.nn = _nn
        sys.modules["torch.nn"] = _nn

        _optim = types.ModuleType("torch.optim")
        _optim.Optimizer = type("Optimizer", (), {})
        _torch.optim = _optim
        sys.modules["torch.optim"] = _optim

        _utils = types.ModuleType("torch.utils")
        _data = types.ModuleType("torch.utils.data")
        _data.Dataset = type("Dataset", (), {})
        _data.DataLoader = type("DataLoader", (), {})
        _utils.data = _data
        _torch.utils = _utils
        sys.modules["torch.utils"] = _utils
        sys.modules["torch.utils.data"] = _data

        sys.modules["torch"] = _torch
        _stubs_added.append("torch")

    if "numpy" in _stubs_added and "np" not in sys.modules:
        sys.modules["np"] = sys.modules["numpy"]
    if _stubs_added:
        logger.info("Injected import stubs for DAG compilation: %s", _stubs_added)

    def _load(name: str, path: str):
        spec = _ilu.spec_from_file_location(name, path)
        mod = _ilu.module_from_spec(spec)
        sys.modules[name] = mod          # register so flytekit's tracker can find it
        spec.loader.exec_module(mod)
        return mod

    sam2_mod = _load("sam2_wf", str(_HERE / "segmentation" / "sam2" / "workflow.py"))

    sam2_result = sam2_mod.run_sam2_segmentation(
        run_id=sam2_run_id,
        video_uris=video_uris,
        max_frames=sam2_max_frames,
    )

    # ── barrier: VP waits for SAM2 ──────────────────────────────────────────
    # _barrier consumes sam2_result (creating a data dependency) and passes
    # through the data_run_id that VP needs.  This guarantees SAM2 finishes
    # before VP starts.
    vp_data_run_id_after_sam2 = _barrier(
        prev_result=sam2_result,
        pass_through=effective_vp_data_run_id,
    )

    # ── STAGE 2 ── VideoPainter Editing ─────────────────────────────────────
    vp_mod = _load("vp_wf", str(_HERE / "generation" / "VideoPainter" / "workflow.py"))

    vp_result = vp_mod.run_videopainter_edit_many(
        data_run_id=vp_data_run_id_after_sam2,       # ← depends on SAM2 via barrier
        video_editing_instructions=vp_instruction,
        num_videos_per_prompt=vp_num_samples,
    )

    # ── barrier: Alpamayo waits for VP ──────────────────────────────────────
    # Build the GCS path *and* create a data dependency on vp_result
    vp_output_gcs = _build_gcs_path(
        prev_result=vp_result,
        base=f"{_OUTPUT_BASE}/vp",
        suffix=effective_vp_data_run_id,
    )
    alpamayo_run_id_after_vp = _barrier(
        prev_result=vp_result,
        pass_through=effective_alpamayo_run_id,
    )

    # ── STAGE 3 ── Alpamayo VLA Inference ───────────────────────────────────
    alpamayo_mod = _load("alpamayo_wf", str(_HERE / "vla" / "alpamayo" / "workflow.py"))

    alpamayo_result = alpamayo_mod.run_alpamayo_inference_task(
        video_data_gcs_path=vp_output_gcs,            # ← depends on VP via _build_gcs_path
        output_run_id=alpamayo_run_id_after_vp,        # ← depends on VP via _barrier
        model_id=alpamayo_model_id,
        num_traj_samples=alpamayo_num_traj_samples,
    )

    logger.info("=" * 80)
    logger.info("MASTER PIPELINE COMPLETE")
    logger.info("=" * 80)

    return "Pipeline complete"


# ---------------------------------------------------------------------------
# @workflow – the proper entry-point for `hlx wf run`
# A @dynamic cannot be the top-level entry; it must be called from a @workflow.
# ---------------------------------------------------------------------------
@workflow
def master_pipeline_wf(
    # SAM2 parameters
    sam2_run_id: str = "experiment_01",
    sam2_max_frames: int = 150,
    sam2_video_uris: str = "default",
    # VideoPainter parameters
    vp_data_run_id: str = "",
    vp_instruction: str = "",
    vp_num_samples: int = 1,
    # Alpamayo parameters
    alpamayo_run_id: str = "",
    alpamayo_num_traj_samples: int = 1,
    alpamayo_model_id: str = "nvidia/Alpamayo-R1-10B",
) -> str:
    """Master pipeline workflow: SAM2 → VideoPainter → Alpamayo."""
    return _master_pipeline_dynamic(
        sam2_run_id=sam2_run_id,
        sam2_max_frames=sam2_max_frames,
        sam2_video_uris=sam2_video_uris,
        vp_data_run_id=vp_data_run_id,
        vp_instruction=vp_instruction,
        vp_num_samples=vp_num_samples,
        alpamayo_run_id=alpamayo_run_id,
        alpamayo_num_traj_samples=alpamayo_num_traj_samples,
        alpamayo_model_id=alpamayo_model_id,
    )
