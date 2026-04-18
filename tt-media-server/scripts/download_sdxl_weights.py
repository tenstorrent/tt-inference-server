# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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
    """Download SDXL 1024 model weights"""
    _download_model(
        repo_id=SupportedModels.STABLE_DIFFUSION_XL_BASE.value,
        local_dir=local_dir,
        label="SDXL 1024",
    )


def download_sdxl_512_model(local_dir="./models/SDXL-512"):
    """Download SDXL 512 model weights (hotshotco/SDXL-512)"""
    _download_model(
        repo_id=SupportedModels.STABLE_DIFFUSION_XL_512.value,
        local_dir=local_dir,
        label="SDXL 512",
    )


def _download_model(repo_id: str, local_dir: str, label: str):
    """Download a HuggingFace model to a local directory."""
    try:
        from huggingface_hub import snapshot_download

        # Create local directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        print(f"Downloading {label} model ({repo_id}) to: {os.path.abspath(local_dir)}")
        print("This may take several minutes...")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        print(
            f"✓ {label} model downloaded successfully to: {os.path.abspath(local_dir)}"
        )

        # Verify download
        model_index_path = os.path.join(local_dir, "model_index.json")
        if os.path.exists(model_index_path):
            print(f"✓ {label} model download verified")
        else:
            print(
                f"⚠ Warning: model_index.json not found, {label} download may be incomplete"
            )

    except Exception as e:
        print(f"✗ Error downloading {label} model: {e}")
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


def _check_and_download(download_fn, local_dir, label):
    """Check if model exists, prompt for re-download, then download."""
    if os.path.exists(local_dir) and os.path.exists(
        os.path.join(local_dir, "model_index.json")
    ):
        size_gb = get_directory_size(local_dir)
        print(f"{label} model already exists at: {os.path.abspath(local_dir)}")
        print(f"Directory size: {size_gb:.2f} GB")

        response = input(f"Do you want to re-download {label}? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print(f"Skipping {label} download.")
            return

    download_fn(local_dir)

    size_gb = get_directory_size(local_dir)
    print(f"\n{label} download completed!")
    print(f"Location: {os.path.abspath(local_dir)}")
    print(f"Size: {size_gb:.2f} GB")


def main():
    """Main function.

    Usage:
        python download_sdxl_weights.py [--all | --512 | --1024] [directory]

    By default downloads the 1024 model. Use --512 for the 512 model,
    or --all to download both.
    """
    print("SDXL Model Downloader")
    print("=" * 50)

    install_huggingface_hub()

    args = sys.argv[1:]
    mode = "1024"
    custom_dir = None

    for arg in args:
        if arg == "--all":
            mode = "all"
        elif arg == "--512":
            mode = "512"
        elif arg == "--1024":
            mode = "1024"
        else:
            custom_dir = arg

    if mode in ("1024", "all"):
        local_dir = custom_dir or "./models/stable-diffusion-xl-base-1.0"
        _check_and_download(download_sdxl_model, local_dir, "SDXL 1024")

    if mode in ("512", "all"):
        local_dir = custom_dir or "./models/SDXL-512"
        _check_and_download(download_sdxl_512_model, local_dir, "SDXL 512")


if __name__ == "__main__":
    main()
