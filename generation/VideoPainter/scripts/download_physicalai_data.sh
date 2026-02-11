#!/bin/bash
set -e

# Configuration
START_INDEX=0  # Starting file index (0 for first batch, 50 for second batch, etc.)
N_FILES=200      # Number of zip files to download (change this)
CAMERA="camera_front_tele_30fov"
DEST_GCS="gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data_physical_ai"
WORK_DIR="/tmp/physicalai_download"
HF_DATASET="nvidia/PhysicalAI-Autonomous-Vehicles"

echo "=== PhysicalAI Data Download Script ==="
echo "Camera: $CAMERA"
echo "Starting from index: $START_INDEX"
echo "Number of files: $N_FILES (each ~1GB)"
echo "Range: chunk_$(printf "%04d" $START_INDEX) to chunk_$(printf "%04d" $((START_INDEX + N_FILES - 1)))"
echo "Destination: $DEST_GCS/$CAMERA"
echo ""

# Install huggingface-hub if not present
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    pip install -q huggingface-hub
fi

# Create working directory
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Track statistics
DOWNLOADED=0
UPLOADED=0
START_TIME=$(date +%s)

# Download and upload files one by one
echo "Starting download and upload process..."
echo "========================================"
echo ""

for i in $(seq $START_INDEX $((START_INDEX + N_FILES - 1))); do
    CHUNK=$(printf "%04d" $i)
    FILE="camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_${CHUNK}.zip"
    ZIP_PATH="camera/$CAMERA/${CAMERA}.chunk_${CHUNK}.zip"
    EXTRACT_DIR="camera/$CAMERA/chunk_${CHUNK}"
    
    echo ">>> File $((i - START_INDEX + 1))/$N_FILES: chunk_${CHUNK} <<<"
    echo ""
    
    # Download
    echo "  [1/3] Downloading (~1GB)..."
    DOWNLOAD_START=$(date +%s)
    huggingface-cli download "$HF_DATASET" "$FILE" \
        --repo-type dataset \
        --local-dir . || {
            echo "  ⚠ Warning: chunk_${CHUNK} not found, stopping..."
            break
        }
    DOWNLOAD_END=$(date +%s)
    DOWNLOAD_TIME=$((DOWNLOAD_END - DOWNLOAD_START))
    echo "  ✓ Downloaded in ${DOWNLOAD_TIME}s"
    DOWNLOADED=$((DOWNLOADED + 1))
    
    # Unzip into subfolder
    echo "  [2/3] Unzipping to chunk_${CHUNK}/..."
    UNZIP_START=$(date +%s)
    if [ -f "$ZIP_PATH" ]; then
        mkdir -p "$EXTRACT_DIR"
        unzip -q -n "$ZIP_PATH" -d "$EXTRACT_DIR/"
        rm -f "$ZIP_PATH"  # Delete zip to save space
        UNZIP_END=$(date +%s)
        UNZIP_TIME=$((UNZIP_END - UNZIP_START))
        echo "  ✓ Unzipped in ${UNZIP_TIME}s"
    else
        echo "  ✗ Error: Zip file not found at $ZIP_PATH"
        continue
    fi
    
    # Upload to GCS immediately
    echo "  [3/3] Uploading to GCS..."
    UPLOAD_START=$(date +%s)
    gsutil -m cp -r "$EXTRACT_DIR" "$DEST_GCS/$CAMERA/" || {
        echo "  ✗ GCS upload failed!"
        exit 1
    }
    UPLOAD_END=$(date +%s)
    UPLOAD_TIME=$((UPLOAD_END - UPLOAD_START))
    echo "  ✓ Uploaded in ${UPLOAD_TIME}s"
    UPLOADED=$((UPLOADED + 1))
    
    # Cleanup local extracted files
    rm -rf "$EXTRACT_DIR"
    
    # Summary for this file
    TOTAL_TIME=$((DOWNLOAD_TIME + UNZIP_TIME + UPLOAD_TIME))
    echo ""
    echo "  ✓ Completed chunk_${CHUNK} in ${TOTAL_TIME}s total"
    echo "  Progress: ${UPLOADED}/${N_FILES} files uploaded"
    echo ""
    echo "========================================"
    echo ""
done

# Final cleanup
rm -rf "$WORK_DIR"

# Final summary
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_ELAPSED / 60))
SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "=== COMPLETE ==="
echo "Downloaded: $DOWNLOADED files"
echo "Uploaded: $UPLOADED files"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Destination: $DEST_GCS/$CAMERA"
echo ""
