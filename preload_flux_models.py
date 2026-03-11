#!/usr/bin/env python3
"""
Pre-download Flux.1 model files with better error handling and retry logic.
Run this before starting the server to avoid timeout issues during warmup.
"""

import os
import sys
import time
from pathlib import Path

# Set aggressive timeouts
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes per file
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Use faster transfer if available


def download_with_retry(download_func, max_retries=3, timeout_per_retry=300):
    """Download with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            start = time.time()
            result = download_func()
            elapsed = time.time() - start
            print(f"  ✓ Downloaded in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ✗ Failed after {elapsed:.2f}s: {e}")
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # 1s, 2s, 4s
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"  Giving up after {max_retries} attempts")
                raise


def main():
    model_id = "black-forest-labs/FLUX.1-schnell"

    print(f"Pre-downloading model files for {model_id}")
    print("=" * 60)
    print(f"Cache location: {os.getenv('HF_HOME', '~/.cache/huggingface')}")
    print("=" * 60)
    print()

    # Import here to see any import errors
    try:
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import CLIPTokenizer, T5TokenizerFast
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        print("Make sure you're in the correct Python environment")
        sys.exit(1)

    # 1. Download T5 Tokenizer (this is the one that was hanging)
    print("1. Downloading T5 Tokenizer...")
    print("-" * 60)
    try:
        _ = download_with_retry(
            lambda: T5TokenizerFast.from_pretrained(
                model_id,
                subfolder="tokenizer_2",
                torch_dtype=torch.bfloat16,
            )
        )
        print("✓ T5 Tokenizer loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to download T5 tokenizer: {e}")
        print("This is likely a network connectivity issue to HuggingFace Hub")
        sys.exit(1)

    # 2. Download CLIP Tokenizer
    print("2. Downloading CLIP Tokenizer...")
    print("-" * 60)
    try:
        _ = download_with_retry(
            lambda: CLIPTokenizer.from_pretrained(
                model_id,
                subfolder="tokenizer",
            )
        )
        print("✓ CLIP Tokenizer loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to download CLIP tokenizer: {e}")
        sys.exit(1)

    # 3. Download Scheduler config
    print("3. Downloading Scheduler config...")
    print("-" * 60)
    try:
        _ = download_with_retry(
            lambda: FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_id,
                subfolder="scheduler",
            )
        )
        print("✓ Scheduler config loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to download scheduler: {e}")
        sys.exit(1)

    # 4. Verify cache contents
    print("4. Verifying downloaded files...")
    print("-" * 60)
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        model_dirs = list(cache_dir.glob(f"models--{model_id.replace('/', '--')}*"))
        if model_dirs:
            for model_dir in model_dirs:
                size_mb = sum(
                    f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                ) / (1024 * 1024)
                print(f"  {model_dir.name}: {size_mb:.2f} MB")
        else:
            print("  Warning: No cached model directories found")
    print()

    print("=" * 60)
    print("✓ All model files downloaded successfully!")
    print("You can now start the server")
    print("=" * 60)


if __name__ == "__main__":
    main()
