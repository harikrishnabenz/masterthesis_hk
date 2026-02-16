#!/usr/bin/env python3
"""
Script to download Alpamayo-R1-10B model checkpoints from HuggingFace.
"""
from huggingface_hub import snapshot_download
import os

print("=" * 80)
print("Downloading Alpamayo-R1-10B model (~22GB)")
print("This may take several minutes depending on your network speed...")
print("=" * 80)

try:
    # Download the model
    snapshot_download(
        repo_id='nvidia/Alpamayo-R1-10B',
        cache_dir='./checkpoints',
        local_dir='./checkpoints/alpamayo-r1-10b',
        local_dir_use_symlinks=False
    )
    print("\n" + "=" * 80)
    print("Download complete!")
    print(f"Model saved to: ./checkpoints/alpamayo-r1-10b")
    print("=" * 80)
except Exception as e:
    print(f"\nError downloading model: {e}")
    print("\nMake sure you have:")
    print("1. Requested access to: https://huggingface.co/nvidia/Alpamayo-R1-10B")
    print("2. Authenticated with: huggingface-cli login")
    exit(1)
