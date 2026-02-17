"""
SAM 2.1 Video Processing Script for HLX Workflow
Wrapper script that accepts parameters and calls process_videos_sam2.py
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List

# Add sam2 to path
BASE_WORKDIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_WORKDIR))

# Import the main processing script
# Use custom namespace to prevent auto-execution
exec_globals = {"__name__": "__imported__", "__file__": str(BASE_WORKDIR / "process_videos_sam2.py")}
with open(BASE_WORKDIR / "process_videos_sam2.py", 'r') as f:
    exec(f.read(), exec_globals)

# Extract the function we need
process_all_videos = exec_globals["process_all_videos"]


def main():
    parser = argparse.ArgumentParser(description="SAM2 Processing for HLX Workflow")
    parser.add_argument("--video-uris", nargs="+", required=True, help="List of video URIs/paths to process")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--model-cfg", required=True, help="Model config file")
    parser.add_argument("--output-bucket", required=True, help="GCS output bucket")
    parser.add_argument("--preprocessed-bucket", required=True, help="GCS preprocessed bucket")
    parser.add_argument("--upload-gcp", action="store_true", help="Upload to GCP")
    parser.add_argument("--upload-local", action="store_true", help="Keep local copies")
    parser.add_argument("--max-frames", type=int, default=150, help="Max frames per video")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    
    args = parser.parse_args()
    
    # Override configuration in the imported function's global namespace
    process_all_videos.__globals__["SAM2_CHECKPOINT"] = args.checkpoint
    process_all_videos.__globals__["MODEL_CFG"] = args.model_cfg
    process_all_videos.__globals__["UPLOAD_TO_GCP"] = args.upload_gcp
    process_all_videos.__globals__["UPLOAD_TO_LOCAL"] = args.upload_local
    process_all_videos.__globals__["GCP_OUTPUT_BUCKET"] = args.output_bucket
    process_all_videos.__globals__["GCP_PREPROCESSED_BUCKET"] = args.preprocessed_bucket
    process_all_videos.__globals__["MAX_FRAMES"] = args.max_frames
    process_all_videos.__globals__["TIMESTAMP"] = args.run_id
    
    # Set up directories
    BASE_DATA_DIR = Path("/tmp/sam2_data")
    OUTPUT_DIR = BASE_DATA_DIR / f"output_{args.run_id}"
    FRAMES_DIR = BASE_DATA_DIR / f"frames_{args.run_id}"
    
    process_all_videos.__globals__["BASE_DATA_DIR"] = BASE_DATA_DIR
    process_all_videos.__globals__["OUTPUT_DIR"] = OUTPUT_DIR
    process_all_videos.__globals__["FRAMES_DIR"] = FRAMES_DIR
    
    print(f"Starting SAM2 processing for run_id: {args.run_id}")
    print(f"Processing {len(args.video_uris)} videos")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output bucket: {args.output_bucket}")
    print(f"Preprocessed bucket: {args.preprocessed_bucket}")
    print()
    
    # Run the processing
    process_all_videos(args.video_uris)


if __name__ == "__main__":
    main()
