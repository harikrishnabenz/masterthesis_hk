# Data Generation Optimization Summary

## Date: 2026-02-12

## Problem Analysis

From the GKE logs, the data generation pipeline was experiencing severe inefficiency:

### Issues Identified:
1. **Model reloading**: Each worker process loaded Qwen2.5-VL-7B (~7GB model) independently
   - Loading time: ~2 minutes per worker
   - With 8 workers, this meant 8 separate model loads
   - Total overhead: 16+ minutes per chunk just for model loading

2. **Multiprocessing memory duplication**: Using `multiprocessing.Pool` meant each worker had separate memory space
   - Models couldn't be shared between workers
   - 8x memory consumption (8 × 7GB = ~56GB just for Qwen models)

3. **No upload visibility**: Files were uploaded but no progress logs
   - Couldn't track which files were being uploaded
   - Hard to diagnose stalls or failures

4. **No model pre-loading**: Models loaded on first use within each worker
   - Inconsistent startup time
   - Race conditions as multiple workers tried to load simultaneously

## Solutions Implemented

### 1. Multi-GPU Model Caching
**Changed**: Global model variables now support multiple GPUs
```python
# Before:
_qwen_model = None  # Single global model

# After:
_qwen_models = {}  # device -> (model, processor, model_path)
_sam2_predictors = {}  # device -> (predictor, key)
```

**Benefit**: Each GPU can have its own cached model instance

### 2. Threading Instead of Multiprocessing
**Changed**: `multiprocessing.Pool` → `ThreadPool`
```python
# Before:
with Pool(processes=num_workers) as pool:
    results = pool.map(_process_single_video, process_args)

# After:
with ThreadPool(processes=num_workers) as pool:
    results = pool.map(_process_single_video, process_args)
```

**Benefit**: 
- Threads share memory space → models loaded once per GPU, shared by all threads
- Reduces memory consumption by ~7x
- Eliminates repeated model loading

### 3. Model Pre-loading
**Added**: New function `_preload_models_on_gpus()` that loads models before processing
```python
_preload_models_on_gpus(
    sam2_checkpoint_path=sam2_checkpoint_path,
    sam2_config_name=sam2_config_name,
    qwen_model_path=QWEN_MODEL_FUSE_ROOT,
    num_gpus=8,
)
```

**Benefit**:
- Models loaded once at startup (parallel across GPUs)
- All workers immediately have access to loaded models
- Predictable startup time
- No race conditions

### 4. Upload Progress Logging
**Changed**: Added progress tracking for file uploads
```python
# Log progress every 10 files
if (idx + 1) % 10 == 0 or (idx + 1) == len(chunk_rows):
    logger.info("Upload progress: %d/%d pairs (%d files)", 
                idx + 1, len(chunk_rows), upload_count)
```

**Benefit**: 
- Clear visibility into upload progress
- Easier to diagnose issues
- User can see that work is happening

## Expected Performance Improvements

### Before Optimization:
- **Model loading per chunk**: 16+ minutes (8 workers × 2 min each)
- **Memory usage**: ~56GB (8 × 7GB Qwen models)
- **Processing efficiency**: Low (waiting for models to load)
- **Visibility**: Poor (no upload logs)

### After Optimization:
- **Model loading**: 2-3 minutes once at startup (parallel loading)
- **Memory usage**: ~7-14GB (1 model per GPU, shared by threads)
- **Processing efficiency**: High (models pre-loaded and cached)
- **Visibility**: Good (progress logs every 10 files)

### Estimated Speedup:
- **Startup**: 16 minutes → 3 minutes (5x faster)
- **Per-chunk processing**: No repeated model loads (13+ minutes saved per chunk)
- **Overall throughput**: **3-5x faster** for typical workloads

## How to Deploy

1. Rebuild the container (if needed):
   ```bash
   cd segmentation/sam2
   bash scripts/build_and_run.sh
   ```

2. The workflow will automatically use the optimized code

3. Monitor logs for:
   - "Pre-loading models on all GPUs..." - should see this once at startup
   - "Models pre-loaded on X GPUs" - confirms successful pre-load
   - "Upload progress: X/Y pairs" - tracks upload progress
   - Worker logs should NOT show repeated "Loading weights" messages

## Validation

### Signs of Success:
✅ Models load only once per GPU at startup
✅ "Loading weights" appears 8 times total (once per GPU)
✅ Upload progress logs appear regularly
✅ Processing starts immediately after pre-loading
✅ No "gke-gcsfuse-sidecar closing reader" spam

### Signs of Issues:
❌ "Loading weights" appears repeatedly during processing
❌ Long gaps between log messages
❌ Memory errors (OOM)
❌ No upload progress logs

## Technical Notes

### Why Threading Works Here:
1. **GIL is not an issue**: Model inference releases GIL (mostly CUDA operations)
2. **I/O bound operations**: File uploads, downloads benefit from threading
3. **Shared memory**: Critical for avoiding model duplication
4. **CUDA safety**: Each thread uses a different GPU (no conflicts)

### Limitations:
- Requires all GPUs to have sufficient VRAM (~14GB per GPU)
- Threading doesn't help pure CPU-bound Python code (but most work is GPU/I/O)
- If GIL becomes a bottleneck, consider async/await for I/O operations

## Future Improvements (Optional)

1. **Async uploads**: Use `asyncio` + `aiohttp` for concurrent uploads
2. **Batched inference**: Process multiple frames simultaneously on each GPU
3. **Progressive saving**: Upload files as they're generated (don't wait for chunk)
4. **Model quantization**: Use int8/fp16 models to reduce VRAM usage
5. **Dynamic worker allocation**: Scale workers based on available VRAM
