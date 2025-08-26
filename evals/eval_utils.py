# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Union


DEFAULT_DATASET_ROOT = Path(os.path.expanduser("~/.cache/tt-eval-datasets"))


def get_coco_dataset(cache_dir: Union[str, Path, None] = None):
    """
    Loads the COCO 2017 validation dataset from Hugging Face using the 'datasets' library.

    This function will download and cache the dataset if it's not already present.
    The Hugging Face cache is used by default (~/.cache/huggingface/datasets).

    Args:
        cache_dir (str, Path, optional): A specific cache directory to use. Defaults to None.

    Returns:
        datasets.Dataset: The loaded validation split as a Hugging Face Dataset object.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to work with the COCO dataset. "
            "Please install it with: pip install datasets"
        )

    # Load the COCO 2017 validation split
    # This will download and cache the dataset if not already present
    print("Loading COCO 2017 validation dataset from Hugging Face...")
    coco_dataset = load_dataset("detection-datasets/coco", split="val", cache_dir=cache_dir)
    print("Dataset loaded successfully.")

    return coco_dataset


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
