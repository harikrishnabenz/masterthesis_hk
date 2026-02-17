# SAM2 → VideoPainter Pipeline

This document explains how to run the SAM2 segmentation workflow and use its output as input to VideoPainter.

## Overview

1. **SAM2 Workflow**: Segments road regions in videos and outputs preprocessed data to GCS
2. **VideoPainter Workflow**: Uses the SAM2 output to perform video editing/inpainting

## Step 1: Run SAM2 Segmentation

```bash
cd segmentation/sam2

# Option 1: Use a custom run identifier
bash scripts/build_and_run.sh my_dataset_v1

# Option 2: Auto-generate identifier (run_YYYYMMDD_HHMMSS)
bash scripts/build_and_run.sh
```

### What happens:
- Processes videos and creates segmentation masks
- Uploads to GCS with your specified `run_id`:
  - Raw output: `gs://.../sam2_final_output/{run_id}/`
  - Preprocessed data: `gs://.../preprocessed_data_vp/{run_id}/`

### Check the logs:
Look for this message in the output:
```
==========================================
SAM2 Segmentation Workflow
RUN_ID: my_dataset_v1
==========================================

NOTE: To use this output in VideoPainter, run:
  cd ../../generation/VideoPainter
  DATA_RUN_ID=my_dataset_v1 bash scripts/build_and_run.sh
```

**Save the run_id** - you'll need it for the next step!

## Step 2: Run VideoPainter on SAM2 Output

```bash
cd generation/VideoPainter

# Use the run_id from SAM2
bash scripts/build_and_run.sh my_dataset_v1

# OR set via environment variable
DATA_RUN_ID=my_dataset_v1 bash scripts/build_and_run.sh
```

### What happens:
- VideoPainter mounts the SAM2 preprocessed data from: 
  `gs://.../preprocessed_data_vp/{run_id}/`
- Processes each video folder created by SAM2
- Outputs edited videos to GCS

## Run Identifier Examples

Choose meaningful identifiers for your data:

```bash
# Dataset versions
bash scripts/build_and_run.sh dataset_v1
bash scripts/build_and_run.sh dataset_v2

# Experiment names
bash scripts/build_and_run.sh experiment_highway_only
bash scripts/build_and_run.sh test_batch_001

# Date-based (auto-generated if not specified)
bash scripts/build_and_run.sh  # Creates: run_20260128_143052
```

## Output Structure

### SAM2 Output
```
gs://bucket/workspace/user/hbaskar/outputs/
├── sam2_final_output/{run_id}/
│   └── {video_id}/
│       ├── masks/
│       ├── visualizations/
│       └── {video_id}_segmented.mp4
│
└── preprocessed_data_vp/{run_id}/
    └── {video_id}/
        ├── meta.csv
        ├── raw_videos/
        │   └── {video_id[:-3]}/{video_id}.0.mp4
        └── mask_root/
            └── {video_id}/all_masks.npz
```

### VideoPainter Input
VideoPainter mounts from:
```
gs://bucket/.../preprocessed_data_vp/{run_id}/
```

## Troubleshooting

### VideoPainter: "ERROR: DATA_RUN_ID not specified"
You need to specify which SAM2 output to use:
```bash
bash scripts/build_and_run.sh my_dataset_v1
```

### "No video IDs found under mounted dataset"
- Check that SAM2 workflow completed successfully for that run_id
- Verify the identifier matches: `gsutil ls gs://bucket/.../preprocessed_data_vp/{run_id}/`

### "Missing meta.csv for video_id"
- The SAM2 preprocessing step may have failed for that video
- Check SAM2 logs for errors during preprocessing

### Using a different run_id
List available run outputs:
```bash
gsutil ls gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/preprocessed_data_vp/
```

Then use any run_id you see:
```bash
bash scripts/build_and_run.sh experiment_highway_only
```

## Advanced: Custom Video URIs

To process different videos, modify [workflow.py](workflow.py):

```python
DEFAULT_VIDEO_URIS = [
    "https://storage.googleapis.com/.../your_video.mp4",
    # Add more URIs...
]
```

Or pass via workflow parameter:
```bash
hlx wf run workflow_sam2.sam2_segmentation_wf \
    --run_id my_custom_run \
    --video_uris "['gs://bucket/video1.mp4', 'gs://bucket/video2.mp4']"
```

## Configuration Details

### SAM2 Workflow ([workflow_sam2.py](workflow_sam2.py))
- **run_id**: Required parameter (passed via command line)
- Output paths constructed dynamically: `{base_path}/{run_id}/`

### VideoPainter Workflow ([../../generation/VideoPainter/workflow_vp.py](../../generation/VideoPainter/workflow_vp.py))
- **DATA_RUN_ID**: Environment variable (set in build_and_run.sh)
- Used to construct the GCS mount prefix at workflow deployment time
- Default: "latest" (if not specified)

