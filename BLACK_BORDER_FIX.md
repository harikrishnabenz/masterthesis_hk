# Black Border Artifacts Fix - Camera Movement Issues

## Problem Description

When the camera moves up and down during video processing, black regions appear at the borders of the road. This is caused by:

1. **Aggressive mask dilation** - Large dilation size (24 pixels) expands masks beyond intended boundaries
2. **Pure black masking** - Masked regions are filled with solid black (0,0,0), creating harsh artifacts
3. **Camera motion exceeding frame boundaries** during video stabilization

## Solutions Implemented

### Solution 1: Reduced Mask Dilation and Feathering

**Changes made:**
- Reduced `VP_DILATE_SIZE` from 24 to 8 pixels
- Reduced `VP_MASK_FEATHER` from 8 to 4 pixels

**Files modified:**
- `scripts/build_and_run.sh`
- `workflow_master.py`

This minimizes the expansion of mask regions, keeping them closer to the original road boundaries.

### Solution 2: Content-Aware Border Handling

**Changes made:**
- Added `VP_BORDER_AWARE_MASKING=true` environment variable
- Added `VP_BORDER_METHOD=inpaint` environment variable  
- Modified `generation/VideoPainter/infer/edit_bench.py` to use OpenCV inpainting instead of black fill

**Available border methods:**
- `inpaint` - Content-aware inpainting using OpenCV TELEA algorithm (recommended)
- `blur` - Gaussian blur of surrounding regions (faster alternative)
- `interpolate` - Manual nearest-neighbor interpolation (experimental)

**Files modified:**
- `generation/VideoPainter/infer/edit_bench.py`
- `scripts/build_and_run.sh`

### Solution 3: Enhanced Border Utilities

**Created new file:**
- `generation/VideoPainter/infer/border_utils.py`

Contains advanced functions for:
- Content-aware masking with multiple algorithms
- Border padding for camera motion compensation
- Improved video processing with better edge handling

## Usage

### Basic Usage (Recommended Defaults)
```bash
bash scripts/build_and_run.sh
```

The script now uses the improved defaults automatically.

### Custom Border Handling
```bash
# Use blur instead of inpainting (faster)
VP_BORDER_METHOD=blur bash scripts/build_and_run.sh

# Disable border-aware masking (revert to original behavior)
VP_BORDER_AWARE_MASKING=false bash scripts/build_and_run.sh

# Custom dilation/feather values
VP_DILATE_SIZE=12 VP_MASK_FEATHER=6 bash scripts/build_and_run.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VP_DILATE_SIZE` | 8 | Mask dilation size in pixels (was 24) |
| `VP_MASK_FEATHER` | 4 | Mask feathering size in pixels (was 8) |
| `VP_BORDER_AWARE_MASKING` | true | Enable content-aware masking |
| `VP_BORDER_METHOD` | inpaint | Border handling method (inpaint/blur/interpolate) |

## Expected Results

- **Eliminated black border artifacts** during camera movement
- **Smoother transitions** at mask boundaries
- **Better content preservation** around road edges
- **Improved visual quality** of generated videos

## Troubleshooting

### If you still see black borders:
1. Try reducing dilation size further: `VP_DILATE_SIZE=4`
2. Switch to blur method: `VP_BORDER_METHOD=blur`  
3. Check video source quality and camera movement intensity

### Performance considerations:
- `inpaint` method: Best quality, moderate speed
- `blur` method: Good quality, fastest speed
- `interpolate` method: Experimental, slowest

### Fallback behavior:
If inpainting fails, the system automatically falls back to blur method to ensure processing continues.