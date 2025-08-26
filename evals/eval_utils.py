# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import zipfile
import urllib.request
from pathlib import Path


DEFAULT_DATASET_ROOT = Path(os.path.expanduser("~/.cache/tt-eval-datasets"))


def get_coco_dataset(
    dataset_root: Path = None,
) -> (Path, Path):
    """
    Downloads and extracts the COCO 2017 validation dataset from Hugging Face if not already present.

    Args:
        dataset_root (Path): The root directory to store datasets. Defaults to HF cache.

    Returns:
        tuple[Path, Path]: A tuple containing the path to the validation images
                           and the path to the validation annotations file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to download the COCO dataset from Hugging Face. "
            "Please install it with: pip install datasets"
        )

    # Use the Hugging Face cache by default
    cache_dir = dataset_root or None

    # Load the COCO 2017 validation split
    # This will download and cache the dataset if not already present
    print("Loading COCO 2017 validation dataset from Hugging Face...")
    coco_dataset = load_dataset("detection-datasets/coco", split="validation", cache_dir=cache_dir)
    print("Dataset loaded successfully.")

    # The 'datasets' library stores the data in a structured way. We need to find the paths.
    dataset_cache_dir = Path(coco_dataset.cache_files[0]["filename"]).parent

    # Search for the images directory and annotations file to be robust to cache structure changes
    images_dir = None
    for path in dataset_cache_dir.rglob("val2017"):
        if path.is_dir():
            images_dir = path
            break
    
    annotation_file = None
    for path in dataset_cache_dir.rglob("instances_val2017.json"):
        if path.is_file():
            annotation_file = path
            break

    if not images_dir or not annotation_file:
        raise FileNotFoundError(
            f"Could not find expected dataset structure in {dataset_cache_dir}. "
            "Please check the Hugging Face cache directory."
        )

    return images_dir, annotation_file


def score_task_single_key(results, task_name, kwargs):
    result_key = kwargs["result_keys"][0]

    res = results[task_name]
    assert (
        result_key in res
    ), f"task_name:= {task_name} result_key:= {result_key} not found in results"

    score = res[result_key]
    if kwargs["unit"] == "percent":
        score *= 100.0
    return score


def score_task_keys_mean(results, task_name, kwargs):
    result_keys = kwargs["result_keys"]

    res = results[task_name]
    values = []
    for key in result_keys:
        assert (
            key in res
        ), f"task_name:= {task_name} result_key:= {key} not found in results"
        values.append(res[key])

    score = sum(values) / len(values)

    if kwargs["unit"] == "percent":
        score *= 100.0
    return score


def score_multilevel_keys_mean(results, task_name, kwargs):
    result_keys = kwargs["result_keys"]
    values = []

    assert isinstance(result_keys[0], tuple), "result_keys should be a tuple of keys"

    for multilevel_keys in result_keys:
        # apply all keys in order
        sub_res = results.copy()
        for sub_key in multilevel_keys:
            assert (
                sub_key in sub_res
            ), f"task_name:= {task_name} result_key:= {sub_key} not found in results"
            sub_res = sub_res[sub_key]
        values.append(sub_res)

        score = sum(values) / len(values)

    if kwargs["unit"] == "percent":
        score *= 100.0
    return score


def score_object_detection_map(results, task_name, kwargs):
    """Score object detection mAP results."""
    res = results[task_name]
    
    metric_key = kwargs.get("metric_key", "mAP")
    assert metric_key in res, f"Metric key {metric_key} not found in results"
    
    score = res[metric_key]
    if kwargs.get("unit") == "percent":
        score *= 100.0
    
    return score
