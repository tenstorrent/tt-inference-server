# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import urllib.parse
import hashlib
import requests
import torch
from tabulate import tabulate
import json
from pathlib import Path
from torch.hub import load_state_dict_from_url


def get_file(path):
    """Get a file from local filesystem, cache, or URL.

    This function handles both local files and URLs, retrieving from cache when available
    or downloading/fetching as needed. For URLs, it creates a unique cached filename using
    a hash of the URL to prevent collisions.

    Args:
        path: Path to a local file or URL to download

    Returns:
        Path to the file in the cache
    """
    # Check if path is a URL - handle URLs and files differently
    path_is_url = path.startswith(("http://", "https://"))

    if path_is_url:
        # Create a hash from the URL to ensure uniqueness and prevent collisions
        url_hash = hashlib.md5(path.encode()).hexdigest()[:10]

        # Get filename from URL, or create one if not available
        file_name = os.path.basename(urllib.parse.urlparse(path).path)
        if not file_name:
            file_name = f"downloaded_file_{url_hash}"
        else:
            file_name = f"{url_hash}_{file_name}"

        rel_path = Path("url_cache")
        cache_dir_fallback = Path.home() / ".cache/url_cache"
    else:
        rel_dir, file_name = os.path.split(path)
        rel_path = Path("models/tt-ci-models-private") / rel_dir
        cache_dir_fallback = Path.home() / ".cache/lfcache" / rel_dir

    # Determine the base cache directory based on environment variables
    if (
        "DOCKER_CACHE_ROOT" in os.environ
        and Path(os.environ["DOCKER_CACHE_ROOT"]).exists()
    ):
        cache_dir = Path(os.environ["DOCKER_CACHE_ROOT"]) / rel_path
    elif "LOCAL_LF_CACHE" in os.environ:
        cache_dir = Path(os.environ["LOCAL_LF_CACHE"]) / rel_path
    else:
        cache_dir = cache_dir_fallback

    file_path = cache_dir / file_name

    # Support case where shared cache is read only and file not found. Can read files from it, but
    # fall back to home dir cache for storing downloaded files. Common w/ CI cache shared w/ users.
    cache_dir_rdonly = not os.access(cache_dir, os.W_OK)
    if not file_path.exists() and cache_dir_rdonly and cache_dir != cache_dir_fallback:
        print(
            f"Warning: {cache_dir} is read-only, using {cache_dir_fallback} for {path}"
        )
        cache_dir = cache_dir_fallback
        file_path = cache_dir / file_name

    cache_dir.mkdir(parents=True, exist_ok=True)

    # If file is not found in cache, download URL from web, or get file from IRD_LF_CACHE web server.
    if not file_path.exists():
        if path_is_url:
            try:
                print(f"Downloading file from URL {path} to {file_path}")
                response = requests.get(path, stream=True, timeout=(15, 60))
                response.raise_for_status()  # Raise exception for HTTP errors

                with open(file_path, "wb") as f:
                    f.write(response.content)

            except Exception as e:
                raise RuntimeError(f"Failed to download {path}: {str(e)}")
        elif "DOCKER_CACHE_ROOT" in os.environ:
            raise FileNotFoundError(
                f"File {file_path} is not available, check file path. If path is correct, DOCKER_CACHE_ROOT syncs automatically with S3 bucket every hour so please wait for the next sync."
            )
        else:
            if "IRD_LF_CACHE" not in os.environ:
                raise ValueError(
                    "IRD_LF_CACHE environment variable is not set. Please set it to the address of the IRD LF cache."
                )
            print(f"Downloading file from path {path} to {cache_dir}/{file_name}")
            exit_code = os.system(
                f"wget -nH -np -R \"indexg.html*\" -P {cache_dir} {os.environ['IRD_LF_CACHE']}/{path} --connect-timeout=15 --read-timeout=60 --tries=3"
            )
            # Check for wget failure
            if exit_code != 0:
                raise RuntimeError(
                    f"wget failed with exit code {exit_code} when downloading {os.environ['IRD_LF_CACHE']}/{path}"
                )

            # Ensure file_path exists after wget command
            if not file_path.exists():
                raise RuntimeError(
                    f"Download appears to have failed: File {file_name} not found in {cache_dir} after wget command"
                )

    return file_path


def load_class_labels(file_path):
    """Load class labels from a JSON or TXT file."""
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            class_idx = json.load(f)
        return [class_idx[str(i)][1] for i in range(len(class_idx))]
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]


def print_compiled_model_results(compiled_model_out, use_1k_labels: bool = True):
    if use_1k_labels:
        imagenet_class_index_path = str(
            get_file(
                "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
            )
        )
    else:
        imagenet_class_index_path = str(
            get_file(
                "https://raw.githubusercontent.com/mosjel/ImageNet_21k_Original_OK/main/imagenet_21k_original_OK.txt"
            )
        )

    class_labels = load_class_labels(imagenet_class_index_path)

    # Get top-1 class index and probability
    compiled_model_top1_probabilities, compiled_model_top1_class_indices = torch.topk(
        compiled_model_out[0].softmax(dim=1) * 100, k=1
    )
    compiled_model_top1_class_idx = compiled_model_top1_class_indices[0, 0].item()
    compiled_model_top1_class_prob = compiled_model_top1_probabilities[0, 0].item()

    # Get class label
    compiled_model_top1_class_label = class_labels[compiled_model_top1_class_idx]

    # Prepare results
    table = [
        ["Metric", "Compiled Model"],
        ["Top 1 Predicted Class Label", compiled_model_top1_class_label],
        ["Top 1 Predicted Class Probability", compiled_model_top1_class_prob],
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)