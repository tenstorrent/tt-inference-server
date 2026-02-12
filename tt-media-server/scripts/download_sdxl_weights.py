# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#!/usr/bin/env python3
# filepath: /localdev/idjuric/tt-inference-server/tt-media-server/scripts/download_sdxl_weights.py

import os
import subprocess
import sys
from pathlib import Path

from config.constants import SupportedModels


def install_huggingface_hub():
    """Install huggingface_hub package if not already installed"""
    try:
        print("✓ huggingface_hub already installed")
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"]
        )
        print("✓ huggingface_hub installed successfully")


def download_sdxl_model(local_dir="./models/stable-diffusion-xl-base-1.0"):
    """Download SDXL model weights"""
    try:
        from huggingface_hub import snapshot_download

        # Create local directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        print(f"Downloading SDXL model to: {os.path.abspath(local_dir)}")
        print("This may take several minutes...")

        snapshot_download(
            repo_id=SupportedModels.STABLE_DIFFUSION_XL_BASE.value,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        print(f"✓ Model downloaded successfully to: {os.path.abspath(local_dir)}")

        # Verify download
        model_index_path = os.path.join(local_dir, "model_index.json")
        if os.path.exists(model_index_path):
            print("✓ Model download verified")
        else:
            print("⚠ Warning: model_index.json not found, download may be incomplete")

    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        sys.exit(1)


def download_sd3_5_large_model(local_dir="./models/stable-diffusion-3.5-large"):
    """Download Stable Diffusion 3.5 Large model weights"""
    try:
        from huggingface_hub import snapshot_download

        # Create local directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        print(f"Downloading SD3.5 Large model to: {os.path.abspath(local_dir)}")
        print("This may take several minutes...")

        snapshot_download(
            repo_id=SupportedModels.STABLE_DIFFUSION_3_5_LARGE.value,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        print(f"✓ Model downloaded successfully to: {os.path.abspath(local_dir)}")

        # Verify download
        model_index_path = os.path.join(local_dir, "model_index.json")
        if os.path.exists(model_index_path):
            print("✓ Model download verified")
        else:
            print("⚠ Warning: model_index.json not found, download may be incomplete")

    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        sys.exit(1)


def get_directory_size(path):
    """Calculate directory size in GB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024**3)  # Convert to GB
    except Exception:
        return 0


def main():
    """Main function"""
    print("SDXL Model Downloader")
    print("=" * 50)

    # Default download location
    default_dir = "./models/stable-diffusion-xl-base-1.0"

    # Allow custom directory via command line argument
    if len(sys.argv) > 1:
        local_dir = sys.argv[1]
    else:
        local_dir = default_dir

    # Check if model already exists
    if os.path.exists(local_dir) and os.path.exists(
        os.path.join(local_dir, "model_index.json")
    ):
        size_gb = get_directory_size(local_dir)
        print(f"Model already exists at: {os.path.abspath(local_dir)}")
        print(f"Directory size: {size_gb:.2f} GB")

        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Skipping download.")
            return

    # Install dependencies and download
    install_huggingface_hub()
    download_sdxl_model(local_dir)

    # Show final stats
    size_gb = get_directory_size(local_dir)
    print("\nDownload completed!")
    print(f"Location: {os.path.abspath(local_dir)}")
    print(f"Size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
