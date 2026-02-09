# VLM 72B Mount Timing Issue - Fix Explanation

## Problem

When `USE_QWEN2_5_VL_72B=True`, the workflow was failing with:
```
OSError: Incorrect path_or_model_id: '/workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-72B-Instruct'
```

This occurred even though the GCS path exists with all model files.

## Root Cause: Race Condition

The issue was a **race condition between FUSE mount initialization and symlink creation**:

1. **FuseBucket mount is declared** in the `@task` decorator
2. **Task function starts executing immediately** (mounts may still be initializing)
3. **Symlink creation happens immediately** before the FUSE mount is ready:
   ```python
   ensure_symlink(VP_VLM_72B_FUSE_MOUNT_ROOT, VLM_72B_DEST_PATH)
   ```
   This creates a symlink to `/mnt/vp-vlm-72b/` which might not exist yet
4. **Model loading fails later** because the symlink points to an unmounted path

## Solution

Added a **wait mechanism** to ensure the FUSE mount is ready before creating the symlink:

### New Function: `wait_for_mount()`

```python
def wait_for_mount(mount_path: str, timeout_seconds: int = 60, check_files: bool = False) -> bool:
    """Wait for a FUSE mount to become available and readable.
    
    Args:
        mount_path: Path to the mount point to check
        timeout_seconds: Maximum time to wait (default 60s)
        check_files: If True, verify mount contains readable files (not just exists)
    
    Returns:
        True if mount is ready, False if timeout
    """
```

**Features:**
- Polls the mount path every 1 second for up to 60 seconds
- When `check_files=True`, verifies the mount is responsive by listing directory contents
- Logs progress every 10 seconds to help diagnose timeout issues
- Catches `OSError` and `IOError` exceptions that occur during mount initialization

### Updated Symlink Creation

Both `run_videopainter_edit()` and `run_videopainter_edit_many()` now:

```python
# Wait for the FUSE mount to be ready before creating symlink
logger.info("Waiting for VLM 72B mount to be ready at %s...", VP_VLM_72B_FUSE_MOUNT_ROOT)
if wait_for_mount(VP_VLM_72B_FUSE_MOUNT_ROOT, timeout_seconds=60, check_files=True):
    ensure_symlink(VP_VLM_72B_FUSE_MOUNT_ROOT, VLM_72B_DEST_PATH)
else:
    logger.error("VLM 72B mount did not become ready! Symlink may not work.")
    logger.error("Expected mount at: %s", VP_VLM_72B_FUSE_MOUNT_ROOT)
    logger.error("This will likely cause the model loading to fail.")
```

## What Happens Now

1. FuseBucket mount is declared in the task decorator
2. Task function starts
3. **We WAIT for the mount to be ready** (up to 60 seconds)
4. Once `/mnt/vp-vlm-72b/` is accessible and contains files, we create the symlink
5. Model loading now succeeds because the symlink points to a mounted, accessible path

## Testing

The fix should resolve the error. The workflow will now:
- Wait for the mount to initialize
- Log when the mount is ready
- Create the symlink only after the mount is confirmed accessible
- Provide clear error messages if the mount fails to initialize

## Configuration Changes

No configuration changes needed. The fix is transparent and works with:
```bash
USE_QWEN2_5_VL_72B=True
LLM_MODEL_SIZE="72B"
```
