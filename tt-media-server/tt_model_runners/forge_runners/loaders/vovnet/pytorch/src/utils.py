# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import zipfile
from six.moves import urllib
import torch
from loguru import logger
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import requests
import shutil
import time
from ....tools.utils import get_file


def preprocess_steps(model_type):
    model = model_type(True, True).eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    return model, img_tensor


def preprocess_timm_model(model_name):
    use_pretrained_weights = True
    if model_name == "ese_vovnet99b":
        use_pretrained_weights = False
    model = timm.create_model(model_name, pretrained=use_pretrained_weights)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    return model, img_tensor


def download_model(download_func, *args, num_retries=3, timeout=180, **kwargs):
    for _ in range(num_retries):
        try:
            return download_func(*args, **kwargs)
        except (
            requests.exceptions.HTTPError,
            urllib.error.HTTPError,
            requests.exceptions.ReadTimeout,
            urllib.error.URLError,
        ):
            logger.trace("HTTP error occurred. Retrying...")
            shutil.rmtree(os.path.expanduser("~") + "/.cache", ignore_errors=True)
            shutil.rmtree(
                os.path.expanduser("~") + "/.torch/models", ignore_errors=True
            )
            shutil.rmtree(
                os.path.expanduser("~") + "/.torchxrayvision/models_data",
                ignore_errors=True,
            )
            os.makedirs(os.path.expanduser("~") + "/.cache", exist_ok=True)
        time.sleep(timeout)

    logger.error("Failed to download the model after multiple retries.")
    assert False, "Failed to download the model after multiple retries."
